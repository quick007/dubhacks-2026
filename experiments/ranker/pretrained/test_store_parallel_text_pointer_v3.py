"""CPU pointer boundary/numerical tests; no native feature or model execution."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import store_parallel_text_pointer_v3 as pointer
from . import store_parallel_text_head_v1 as base
from . import store_parallel_text_data_v1 as data
from .store_parallel_text_run_v1 import load_packet


class PointerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.packet = load_packet()
        bundle = base.engineered_fixture(cls.packet)
        cls.feature_map = base.validate_bundle(bundle, cls.packet, fixture=True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def rows(self):
        return base.inference_rows(self.packet['cases']['eval'][0]['rows'])

    def test_only_public_fact_words_roles_counts_and_complete_source(self):
        rows = self.rows(); batch = pointer.inference_tensors(rows, self.feature_map, self.packet['vocabulary'])
        self.assertFalse(batch['sourceMask'][0].any())
        self.assertEqual(batch['factLengths'][0].tolist(), [0, 0, 0])
        for row, words, roles, positions, mask, lengths in zip(rows, batch['sourceIds'], batch['sourceRoles'], batch['sourcePositions'], batch['sourceMask'], batch['factLengths']):
            if row['input']['kind'] == 'section': continue
            facts = row['input']['source']['facts']
            groups = [data.tokens(facts['name']), data.tokens(facts['selectedColor']), ['$' + data.money(facts['itemPrice'])]]
            expected = [word for group in groups for word in group]
            self.assertEqual([self.packet['vocabulary'][i] for i in words[mask]], expected)
            self.assertEqual(roles[mask].tolist(), [role + 1 for role, group in enumerate(groups) for _ in group])
            self.assertEqual(positions[mask].tolist(), [pos for group in groups for pos in range(len(group))])
            self.assertEqual(lengths.tolist(), [len(group) / 4 for group in groups])
        for row in self.packet['labels']['train'] + self.packet['labels']['eval']:
            pointer.inference_tensors(base.inference_rows([row]), self.feature_map, self.packet['vocabulary'])

    def test_heading_mask_is_finite_normalized_and_eos_not_source(self):
        rows = self.rows(); batch = pointer.inference_tensors(rows, self.feature_map, self.packet['vocabulary'])
        head = pointer.PointerWordHead(self.packet['vocabulary'], width=16)
        out = head(batch)
        self.assertTrue(torch.isfinite(out['probabilities']).all())
        self.assertTrue(torch.allclose(out['probabilities'].sum(-1), torch.ones(20, 14), atol=1e-5))
        self.assertEqual(out['attention'][0].sum().item(), 0)
        self.assertFalse(torch.any(batch['sourceIds'][batch['sourceMask']] == 0))
        with torch.no_grad():
            head.copy_gate.weight.zero_(); head.copy_gate.bias.fill_(1000)
        saturated = head(batch)
        self.assertTrue(torch.isfinite(saturated['probabilities']).all())
        self.assertTrue(torch.allclose(saturated['probabilities'].sum(-1), torch.ones(20, 14), atol=1e-5))
        self.assertEqual(saturated['copyProbability'][0].sum().item(), 0)

    def test_runtime_teacher_blocked_no_targets_or_stale_fact_binding(self):
        rows = self.rows(); head = pointer.PointerWordHead(self.packet['vocabulary'], width=16)
        with patch.object(data, 'teacher', side_effect=AssertionError('teacher lookup')), patch.object(data, 'build', side_effect=AssertionError('teacher packet')):
            result = pointer.generate(head, rows, self.feature_map)
        self.assertEqual((result['fieldCount'], result['feedforwardCalls'], result['autoregressiveSteps']), (20, 1, 0))
        self.assertFalse(result['teacherStringsRead']); self.assertFalse(result['outputFormatsOrRepairsUsed'])
        with self.assertRaises(ValueError):
            pointer.inference_tensors([{**rows[1], 'targetText': 'hidden'}], self.feature_map, self.packet['vocabulary'])
        for key, value in (('selectedColor', 'Wrong'), ('itemPrice', 999)):
            changed = deepcopy(rows[1]); changed['input']['source']['facts'][key] = value
            with self.assertRaises(ValueError):
                pointer.inference_tensors([changed], self.feature_map, self.packet['vocabulary'])

    def test_pointer_and_vocab_receive_real_loss_gradients(self):
        rows = self.packet['labels']['train'][:6]
        batch = pointer.inference_tensors(base.inference_rows(rows), self.feature_map, self.packet['vocabulary'])
        head = pointer.PointerWordHead(self.packet['vocabulary'], width=16)
        targets = base.encode_labels(rows, self.packet['vocabulary'])
        out = head(batch)['probabilities'].clamp_min(1e-9).log()
        torch.nn.functional.nll_loss(out.flatten(0, 1), targets.flatten(), ignore_index=-100).backward()
        for parameter in (head.role_keys.weight, head.copy_gate.weight, head.words.weight, head.pointer_query.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_fixed_predeclared_arms_and_unchanged_split(self):
        self.assertEqual(len(pointer.ARMS), 2)
        self.assertEqual([arm['steps'] for arm in pointer.ARMS], [800, 800])
        train = {row['featureKeys'][0] for row in self.packet['labels']['train'] if row['input']['kind'] != 'section'}
        evaluation = {row['featureKeys'][0] for row in self.packet['labels']['eval'] if row['input']['kind'] != 'section'}
        self.assertEqual((len(train), len(evaluation), len(train & evaluation)), (34, 14, 0))

    def test_train_packet_does_not_decode_heldout_labels_or_teacher_constructor(self):
        original = Path.read_text
        def guarded(path, *args, **kwargs):
            if path.name == 'labels.eval.json':
                raise AssertionError('heldout target text decoded before fit')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', guarded), patch.object(data, 'teacher', side_effect=AssertionError('teacher constructor')), patch.object(data, 'build', side_effect=AssertionError('teacher build')):
            packet = pointer.load_research_packet()
        self.assertNotIn('eval', packet['labels'])
        self.assertFalse(packet['heldoutLabelsDecoded'])

    def test_both_key_types_preserve_source_conditioned_shape(self):
        batch = pointer.inference_tensors(self.rows(), self.feature_map, self.packet['vocabulary'])
        for lexical in (False, True):
            head = pointer.PointerWordHead(self.packet['vocabulary'], lexical_keys=lexical, width=16)
            self.assertEqual(head(batch)['probabilities'].shape, (20, 14, 107))
            self.assertEqual(head.word_keys is not None, lexical)

    def test_both_final_heads_frozen_before_any_heldout_target_decode(self):
        bundle = base.engineered_fixture(self.packet)
        original_validate = base.validate_bundle
        original_read = Path.read_text
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory); source = directory / 'source'; source.mkdir()
            output = directory / 'new-run'
            torch.save(bundle, source / 'features.pt')
            (source / 'results.json').write_text(json.dumps({'status': 'completed-candidate-not-promoted',
                'shutdownIntegrity': {'frozen927HashesUnchanged': True, 'copyCheckpointUnchanged': True},
                'featureBundle': {'file': 'features.pt', 'sha256': pointer.sha(source / 'features.pt')}}))
            packet = pointer.load_research_packet()
            def guarded(path, *args, **kwargs):
                if path.name == 'labels.eval.json':
                    self.assertTrue((output / 'ALL_HEADS_FROZEN.json').exists())
                    for arm in pointer.ARMS:
                        self.assertTrue((output / arm['name'] / 'HEAD_FROZEN.json').exists())
                return original_read(path, *args, **kwargs)
            def fake_fit(packet, feature_map, arm):
                self.assertNotIn('eval', packet['labels'])
                head = pointer.PointerWordHead(packet['vocabulary'], lexical_keys=arm['lexicalKeys'], width=16)
                return head, {'lossSnapshots': [{'step': 800, 'loss': 1.}], 'fitSeconds': 0.,
                              'train': {'strata': {'fixtureOnly': True}}}
            with patch.object(Path, 'read_text', guarded), patch.object(pointer, 'load_research_packet', return_value=packet), patch.object(base, 'validate_bundle', side_effect=lambda b, p: original_validate(b, p, fixture=True)), patch.object(pointer, 'fit_arm', side_effect=fake_fit), patch('builtins.print'):
                result = pointer.run_cpu(source, output)
            self.assertEqual(result['status'], 'completed-cpu-pointer-research-not-promoted')
            self.assertTrue(result['allHeadsFrozenBeforeAnyHeldoutTargetDecode'])


if __name__ == '__main__':
    unittest.main()
