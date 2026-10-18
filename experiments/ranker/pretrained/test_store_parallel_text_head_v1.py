"""CPU numerical/boundary tests; engineered fixture is never Qwen evidence."""
import copy
import json
from pathlib import Path
import unittest

import torch

from . import store_parallel_text_data_v1 as data
from . import store_parallel_text_head_v1 as head


class ParallelTextHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).resolve().parent / 'store_catalog_v1'
        cls.packet = data.build({'inputs': json.loads((base / 'model_inputs.json').read_text())})
        with head.bounded_cpu_threads():
            cls.bundle = head.engineered_fixture(cls.packet)

    def test_native_rejects_fixture_and_exact_feature_order_checksum(self):
        with self.assertRaises(ValueError):
            head.validate_bundle(self.bundle, self.packet)
        values = head.validate_bundle(self.bundle, self.packet, fixture=True)
        self.assertEqual(len(values), 48)
        changed = {**self.bundle, 'featureKeys': list(reversed(self.bundle['featureKeys']))}
        with self.assertRaises(ValueError):
            head.validate_bundle(changed, self.packet, fixture=True)
        changed = {**self.bundle, 'features': self.bundle['features'].clone()}
        changed['features'][0, 0] += .0001
        with self.assertRaises(ValueError):
            head.validate_bundle(changed, self.packet, fixture=True)

    def test_all_twenty_fields_one_forward_no_teacher_or_previous_word_inputs(self):
        feature_map = head.validate_bundle(self.bundle, self.packet, fixture=True)
        model = head.ParallelTextHead(self.packet['vocabulary'], width=16)
        rows = head.inference_rows(self.packet['cases']['eval'][0]['rows'])
        previous = data.teacher
        try:
            data.teacher = lambda *a, **k: (_ for _ in ()).throw(AssertionError('teacher reached'))
            with head.bounded_cpu_threads():
                result = head.generate(model, rows, feature_map)
        finally:
            data.teacher = previous
        self.assertEqual((result['fieldCount'], result['feedforwardCalls'], result['autoregressiveSteps']), (20, 1, 0))
        self.assertFalse(result['teacherStringsRead'])

    def test_private_target_and_mismatched_fact_binding_rejected(self):
        feature_map = head.validate_bundle(self.bundle, self.packet, fixture=True)
        row = head.inference_rows(self.packet['labels']['train'])[0]
        with self.assertRaises(ValueError):
            head.serving_tensors([{**row, 'targetText': 'Do not read this'}], feature_map)
        modified = copy.deepcopy(row)
        modified['input']['hiddenPersona'] = 'private'
        with self.assertRaises(ValueError):
            head.serving_tensors([modified], feature_map)
        modified = copy.deepcopy(row)
        modified['input']['source']['facts']['selectedColor'] = 'Wrong color'
        with self.assertRaises(ValueError):
            head.serving_tensors([modified], feature_map)
        missing = dict(feature_map)
        del missing[row['featureKeys'][0]]
        with self.assertRaises(ValueError):
            head.serving_tensors([row], missing)

    def test_eos_is_learned_and_no_length_or_pad_repair(self):
        vocab = [data.EOS, data.PAD, 'Sage', 'selected']
        result = head.decode_ids([2, 3, 0] + [1] * (data.SLOTS - 3), vocab)
        self.assertEqual(result['text'], 'Sage selected')
        self.assertEqual(result['finishReason'], 'eos')
        for values in ([2] * data.SLOTS, [0] + [2] * (data.SLOTS - 1), [1, 2, 0] + [1] * (data.SLOTS - 3)):
            self.assertIsNone(head.decode_ids(values, vocab)['text'])
        with self.assertRaises(ValueError):
            head.decode_ids([2] * (data.SLOTS - 1), vocab)

    def test_label_slot_training_masks_only_after_eos(self):
        rows = self.packet['labels']['train'][:2]
        encoded = head.encode_labels(rows, self.packet['vocabulary'])
        for row, target in zip(rows, encoded):
            length = len(data.tokens(row['targetText']))
            self.assertEqual(target[length], 0)
            self.assertTrue(torch.all(target[length + 1:] == -100))
        with self.assertRaises(ValueError):
            head.encode_labels([{'targetText': 'Nonexistentword'}], self.packet['vocabulary'])

    def test_case_receipt_preserves_six_cards_and_rejects_wrong_binding(self):
        feature_map = head.validate_bundle(self.bundle, self.packet, fixture=True)
        model = head.ParallelTextHead(self.packet['vocabulary'], width=16)
        case = self.packet['cases']['eval'][0]
        with head.bounded_cpu_threads():
            result = head.evaluate_cases(model, [case], feature_map)
        self.assertEqual(result['caseCount'], 1)
        self.assertEqual(result['receipts'][0]['result']['fieldCount'], 20)
        self.assertIn('React/images excluded', result['timingScope'])
        broken = copy.deepcopy(case)
        broken['rows'][1]['featureKeys'] = broken['rows'][4]['featureKeys']
        with self.assertRaises(ValueError):
            head.evaluate_cases(model, [broken], feature_map)

    def test_fit_decreases_engineered_fixture_loss_without_native_claim(self):
        model, report = head.fit(self.packet, self.bundle, steps=5, width=16, fixture=True)
        self.assertLess(report['lossSnapshots'][-1]['loss'], report['lossSnapshots'][0]['loss'])
        self.assertEqual(report['status'], 'engineered-fixture-only')
        self.assertEqual(report['sourceEvidence']['nativeForwards'], 0)
        self.assertEqual(report['backboneUpdates'], 0)
        self.assertEqual(report['evaluation']['eval']['fields'], 210)
        self.assertLess(sum(p.numel() for p in model.parameters()), 1000000)

    def test_model_construction_preserves_caller_rng_and_cpu_shape(self):
        torch.manual_seed(987)
        state = torch.random.get_rng_state().clone()
        model = head.ParallelTextHead(self.packet['vocabulary'], width=16)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        with self.assertRaises(ValueError):
            model(torch.ones(2, 12), torch.ones(2, 9))


if __name__ == '__main__':
    unittest.main()
