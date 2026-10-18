"""CPU-only guards for the new fit and real complete-view readiness boundary."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from .short_copy_fit import (CONFIG, DATA, FIRST_FIELDS, SECOND_FIELDS, arm_roster,
    assemble_section, dataset_packet, decode_complete_rows, encoded_payload,
    evaluation_modes, section_projection, serving_rows, shuffled_order, summary)


class ShortCopyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.packet = dataset_packet()

    def test_import_never_loads_native_runtime(self):
        code = ('import sys; import experiments.ranker.pretrained.short_copy_fit; '
            'assert not any(k == "mlx" or k.startswith("mlx.") for k in sys.modules)')
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)

    def test_all_frozen_teacher_inputs_match_serving_without_targets(self):
        self.assertEqual(len(self.packet['train']['fields']), 240)
        for split, group in self.packet.items():
            for case in group['cases']:
                fields = [f for f in group['fields'] if f['caseId'] == case['caseId']]
                common, rows, ordered, _ = serving_rows(case, fields)
                self.assertEqual(tuple(r['fields'][0] for r in rows), FIRST_FIELDS + SECOND_FIELDS)
                self.assertEqual(sum(r['maxTokens'] for r in rows[:10]), 208)
                self.assertTrue(all('targetValue' not in r and 'text' not in r for r in rows))
                self.assertEqual([r['messages'] for r in rows], [f['messages'] for f in ordered])
        case = self.packet['train']['cases'][0]
        fields = copy.deepcopy(self.packet['train']['fields'][:20])
        fields[0]['messages'][0]['content'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'serving renderer'):
            serving_rows(case, fields)

    def test_only_one_real_update_per_frozen_row_and_explicit_modes(self):
        order = shuffled_order()
        self.assertEqual(sorted(order), list(range(240)))
        self.assertEqual(order, shuffled_order())
        self.assertNotEqual(order, list(range(240)))
        self.assertGreater(len({i // 20 for i in order[:20]}), 6)
        for bad in (239, 241, True):
            with self.assertRaises(ValueError):
                shuffled_order(bad)
        for bad in ('new-C,new-C', 'prior-STB', 'template,new-C'):
            with self.assertRaises(ValueError):
                evaluation_modes(bad)
        self.assertEqual(CONFIG['totalFeatureArtifactBudgetBytes'], 5 * 2**30)

    def native_teacher_rows(self, case, split):
        fields = [f for f in self.packet[split]['fields'] if f['caseId'] == case['caseId']]
        _, rows, ordered, _ = serving_rows(case, fields)
        for row, field in zip(rows, ordered, strict=True):
            row.update(text=field['targetValue'], tokenIds=[5], rawSampledTokenIds=[5, 99],
                stopTokenId=99, arSteps=2, finishReason='stop')
        return rows

    def test_both_complete_sections_use_unchanged_strict_validator(self):
        case = self.packet['dev']['cases'][0]; rows = self.native_teacher_rows(case, 'dev')
        before = copy.deepcopy(rows)
        for section in (0, 1):
            result = assemble_section(case, rows, section)
            self.assertTrue(result['validation']['valid'], result['validation']['rejections'])
            self.assertGreater(len(encoded_payload(result)), 0)
        self.assertEqual(rows, before)
        plan, projected = section_projection(case, rows, 1)
        self.assertEqual(plan['sections'][0]['sectionId'], case['pendingPagePlan']['sections'][1]['sectionId'])
        by_global = {r['fields'][0]: r for r in rows}
        for row in projected:
            source = by_global[row['sourceGlobalField']]
            for key in ('text', 'tokenIds', 'rawSampledTokenIds', 'stopTokenId', 'arSteps', 'finishReason', 'maxTokens'):
                self.assertEqual(row[key], source[key])
        bad = copy.deepcopy(rows); bad[3]['text'] += '\nextra'
        self.assertFalse(assemble_section(case, bad, 0)['validation']['valid'])
        bad = copy.deepcopy(rows); bad[3]['finishReason'] = 'length'
        self.assertFalse(assemble_section(case, bad, 0)['validation']['valid'])
        with self.assertRaisesRegex(ValueError, 'ten unique'):
            section_projection(case, rows[:-1], 1)

    def test_failures_keep_fixed_quality_denominators_and_no_speedup_samples(self):
        arms = arm_roster(self.packet)
        self.assertEqual(len(arms), 24)
        self.assertNotIn('frozen', {a['mode'] for a in arms})
        arms[0]['status'] = 'failed'
        values = summary(arms)
        failed = values['dev/prior-STB/width-3']
        self.assertEqual(failed['fixedCases'], 3)
        self.assertEqual(failed['failed'], 1)
        self.assertEqual(failed['validThreeCardRateFixedDenominator'], 0)
        self.assertEqual(failed['validFirstSectionReadyMs'], [])
        self.assertFalse(failed['rankingQualityClaim'])

    def decoder_stubs(self, fail=False):
        package = types.ModuleType('mlx_lm'); package.__path__ = []
        native = types.ModuleType('mlx_lm.generate'); sample = types.ModuleType('mlx_lm.sample_utils')
        state = {'finished': 0, 'closed': False, 'calls': 0}
        class Generator:
            def __init__(self, *args, **kwargs):
                self.live = {}; self.uid = 0
            def insert(self, prompts, **kwargs):
                ids = []
                for prompt in prompts:
                    uid = self.uid; self.uid += 1; ids.append(uid); self.live[uid] = 0
                return ids
            def next_generated(self):
                state['calls'] += 1
                if fail and state['calls'] == 2:
                    raise RuntimeError('injected native failure')
                responses = []
                for uid in list(self.live):
                    self.live[uid] += 1; end = self.live[uid] == 2
                    responses.append(types.SimpleNamespace(uid=uid, token=99 if end else 5,
                        finish_reason='stop' if end else None))
                    if end:
                        del self.live[uid]; state['finished'] += 1
                return responses
            def close(self):
                state['closed'] = True
        native.BatchGenerator = Generator; native.StopSequences = lambda value: value; native.generation_stream = object()
        sample.make_sampler = lambda **kwargs: None
        modules = {'mlx_lm': package, 'mlx_lm.generate': native, 'mlx_lm.sample_utils': sample}
        bb = types.SimpleNamespace(model=object(), check_memory=lambda: None,
            mx=types.SimpleNamespace(synchronize=lambda *args: None),
            tokenizer=types.SimpleNamespace(eos_token_ids=[99], decode=lambda tokens: 'fresh value'))
        return modules, bb, state

    def test_real_complete_first_view_callback_precedes_full_completion(self):
        case = self.packet['dev']['cases'][0]; rows = self.native_teacher_rows(case, 'dev')
        modules, bb, observed = self.decoder_stubs(); callbacks = []
        with patch.dict(sys.modules, modules), patch('experiments.ranker.pretrained.parallel_decode.clone_prefix_state', return_value={}), \
                patch('experiments.ranker.pretrained.parallel_decode.exact_row_suffix', return_value=([1], [])):
            result, timing = decode_complete_rows(bb, {'tokens': [1]}, rows, 3,
                lambda first: callbacks.append((copy.deepcopy(first), observed['finished'])))
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(len(callbacks[0][0]), 10)
        self.assertLess(callbacks[0][1], 20)
        self.assertTrue(all(r['finishReason'] == 'stop' for r in callbacks[0][0]))
        self.assertEqual(len(result), 20)
        self.assertEqual(timing['maximumRowsEmittingTogether'], 3)
        self.assertTrue(observed['closed'])

    def test_decoder_failure_retains_raw_partial_evidence_and_closes_generator(self):
        case = self.packet['dev']['cases'][0]; rows = self.native_teacher_rows(case, 'dev')
        modules, bb, observed = self.decoder_stubs(fail=True)
        with patch.dict(sys.modules, modules), patch('experiments.ranker.pretrained.parallel_decode.clone_prefix_state', return_value={}), \
                patch('experiments.ranker.pretrained.parallel_decode.exact_row_suffix', return_value=([1], [])):
            with self.assertRaisesRegex(RuntimeError, 'injected native failure') as caught:
                decode_complete_rows(bb, {'tokens': [1]}, rows, 3, lambda _: None)
        self.assertEqual(len(caught.exception.partial_rows), 3)
        self.assertEqual(caught.exception.partial_rows[0]['rawSampledTokenIds'], [5])
        self.assertIsNone(caught.exception.partial_rows[0]['finishReason'])
        self.assertTrue(observed['closed'])


if __name__ == '__main__':
    unittest.main()
