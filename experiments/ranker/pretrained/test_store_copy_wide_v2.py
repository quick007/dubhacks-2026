"""CPU-only width10/20 contracts; native generation is fully stubbed."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from . import store_copy_data_v2 as data
from . import store_copy_wide_v2 as wide


class WideCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.cases, cls.fields = data.load_dataset()

    def bound(self):
        return copy.deepcopy(self.cases['eval'][0]['bound'])

    def teacher_rows(self):
        case = self.cases['eval'][0]
        common, rows = data.layout(case['bound'])
        fields = {r['rowId']: r for r in self.fields['eval'] if r['caseId'] == case['caseId']}
        for row in rows:
            row.update(text=fields[row['rowId']]['targetValue'], tokenIds=[5], rawSampledTokenIds=[5, 99],
                stopTokenId=99, arSteps=2, finishReason='stop')
        return rows

    def stubs(self, fail_call=None, mismatch_arm=None):
        package = types.ModuleType('mlx_lm'); package.__path__ = []
        native = types.ModuleType('mlx_lm.generate'); sample = types.ModuleType('mlx_lm.sample_utils')
        observed = {'calls': 0, 'finished': 0, 'closed': 0, 'constructors': [], 'insertions': []}
        class Generator:
            def __init__(self, *args, **kwargs):
                self.live = {}; self.uid = 0; self.arm = len(observed['constructors'])
                observed['constructors'].append(kwargs)
            def insert(self, prompts, **kwargs):
                ids = []
                for prompt in prompts:
                    uid = self.uid; self.uid += 1; ids.append(uid); self.live[uid] = 0
                observed['insertions'].append({'uids': ids, 'kwargs': kwargs})
                return ids
            def next_generated(self):
                observed['calls'] += 1
                if observed['calls'] == fail_call:
                    raise TimeoutError('injected decode failure')
                rows = []
                for uid in list(self.live):
                    self.live[uid] += 1
                    end = self.live[uid] == (2 if uid < 10 else 4)
                    token = 99 if end else 5
                    distribution = np.array([-.5, -1.5, -2.5], dtype=np.float32)
                    if self.arm == mismatch_arm:
                        distribution[0] = -.25
                    rows.append(types.SimpleNamespace(uid=uid, token=token,
                        finish_reason='stop' if end else None, logprobs=distribution))
                    if end:
                        del self.live[uid]; observed['finished'] += 1
                return rows
            def close(self):
                observed['closed'] += 1
        native.BatchGenerator = Generator; native.StopSequences = lambda value: value; native.generation_stream = object()
        sample.make_sampler = lambda **kwargs: None
        modules = {'mlx_lm': package, 'mlx_lm.generate': native, 'mlx_lm.sample_utils': sample}
        memory = {'mlxActiveBytes': 1, 'mlxPeakBytes': 2, 'processBudgetBytes': 7 * 2**30}
        bb = types.SimpleNamespace(model=object(), np=np, check_memory=lambda: None,
            runtime_memory=lambda: memory,
            mx=types.SimpleNamespace(synchronize=lambda *args: None, eval=lambda *args: None, float32=np.float32),
            tokenizer=types.SimpleNamespace(eos_token_ids=[99], decode=lambda ids: 'fresh sampled value'))
        return modules, bb, observed, memory

    def native_patches(self, modules):
        return patch.dict(sys.modules, modules)

    def test_import_never_loads_mlx(self):
        subprocess.run([sys.executable, '-c', 'import sys; import experiments.ranker.pretrained.store_copy_wide_v2; '
            'assert not any(k == "mlx" or k.startswith("mlx.") for k in sys.modules)'], check=True, capture_output=True)

    def test_only_explicit10_or20_and_all_fields_unchanged(self):
        bound = self.bound(); original = copy.deepcopy(bound)
        for width in (10, 20):
            common, rows = wide.serving_jobs(bound, width)
            self.assertEqual(tuple(r['fields'][0] for r in rows), tuple(data.FIELDS))
            self.assertEqual(sum(r['maxTokens'] for r in rows), 416)
            self.assertEqual(sum(r['maxTokens'] for r in rows[:10]), 208)
            self.assertTrue(all('targetValue' not in r for r in rows))
        self.assertEqual(bound, original)
        for width in (True, 3, 6, 10., 20., '20', None):
            with self.assertRaises(ValueError):
                wide.validate_width(width)

    def test_both_widths_use20_independent_forks_and_first10_callback(self):
        _, jobs = wide.serving_jobs(self.bound(), 10)
        for width in (10, 20):
            modules, bb, observed, _ = self.stubs(); callbacks = []
            with self.native_patches(modules), \
                    patch('experiments.ranker.pretrained.parallel_decode.clone_prefix_state', return_value={}) as clone, \
                    patch('experiments.ranker.pretrained.parallel_decode.exact_row_suffix', return_value=([1], [])):
                rows, timing = wide.decode_complete_rows(bb, {'tokens': [1]}, jobs, width,
                    lambda first: callbacks.append((copy.deepcopy(first), observed['finished'])))
            self.assertEqual(clone.call_count, 20)
            self.assertEqual(len(rows), 20)
            self.assertEqual(len(callbacks), 1)
            self.assertEqual(len(callbacks[0][0]), 10)
            self.assertEqual(callbacks[0][1], 10)
            self.assertTrue(all(r['finishReason'] == 'stop' for r in rows))
            self.assertTrue(all(r['rawSampledTokenIds'][-1] == r['stopTokenId'] == 99 for r in rows))
            self.assertEqual(timing['maximumRowsEmittingTogether'], width)
            self.assertEqual(observed['constructors'][0]['completion_batch_size'], width)
            self.assertEqual(observed['closed'], 1)
            self.assertLessEqual(max(a['activeBound'] for a in timing['admissions']), width)
            self.assertEqual(sum(a['admitted'] for a in timing['admissions']), 20)

    def test_decode_failures_close_generator_and_keep_raw_rows(self):
        _, jobs = wide.serving_jobs(self.bound(), 20)
        modules, bb, observed, _ = self.stubs(fail_call=2)
        with self.native_patches(modules), \
                patch('experiments.ranker.pretrained.parallel_decode.clone_prefix_state', return_value={}), \
                patch('experiments.ranker.pretrained.parallel_decode.exact_row_suffix', return_value=([1], [])):
            with self.assertRaises(TimeoutError) as caught:
                wide.decode_complete_rows(bb, {'tokens': [1]}, jobs, 20, lambda rows: None)
        self.assertEqual(len(caught.exception.partial_rows), 20)
        self.assertTrue(all(row['rawSampledTokenIds'] == [5] for row in caught.exception.partial_rows))
        self.assertEqual(observed['closed'], 1)

    def test_no_redistributing_field_caps_even_with_same416_total(self):
        _, jobs = wide.serving_jobs(self.bound(), 20)
        jobs[0]['maxTokens'] += 1; jobs[1]['maxTokens'] -= 1
        self.assertEqual(sum(row['maxTokens'] for row in jobs), 416)
        with self.assertRaisesRegex(ValueError, 'unchanged actual-store fields'):
            wide.decode_complete_rows(None, None, jobs, 20, lambda rows: None)

    def test_native_controls_compare_only_samewidth_and_preserve_caps(self):
        for width in (10, 20):
            modules, bb, observed, _ = self.stubs(); state = {'tokens': [1], 'cache': []}
            with tempfile.TemporaryDirectory() as temporary, self.native_patches(modules), \
                    patch('experiments.ranker.pretrained.branch_local_copy.prepare_root', return_value=(state, {})) as prepare, \
                    patch('experiments.ranker.pretrained.fused_prefix._complete_cache'), \
                    patch('experiments.ranker.pretrained.native_batch_diagnosis.state_hash', return_value='same'), \
                    patch('experiments.ranker.pretrained.parallel_decode.clone_prefix_state', return_value=state), \
                    patch('experiments.ranker.pretrained.parallel_decode.exact_row_suffix', return_value=([1], [])):
                result = wide.native_controls(bb, self.bound(), Path(temporary) / 'controls', width=width)
            self.assertEqual(result['status'], 'exact-controls-passed')
            self.assertEqual(result['width'], width)
            self.assertEqual(len(result['rows']), width)
            self.assertEqual(len(result['arms']), 3)
            self.assertEqual(prepare.call_count, width + 1)
            self.assertTrue(all(a['maximumRowsEmittingTogether'] == width for a in result['arms']))
            self.assertTrue(all(len(r['distributionSha256']) == len(r['rawSampledTokenIds']) for a in result['arms'] for r in a['rows']))
            self.assertEqual(sum(result['originalCaps']), 208 if width == 10 else 416)
            self.assertEqual(observed['closed'], 3)

    def test_samewidth_distribution_mismatch_fails_without_token_repair(self):
        modules, bb, observed, _ = self.stubs(mismatch_arm=1); state = {'tokens': [1], 'cache': []}
        with tempfile.TemporaryDirectory() as temporary, self.native_patches(modules), \
                patch('experiments.ranker.pretrained.branch_local_copy.prepare_root', return_value=(state, {})), \
                patch('experiments.ranker.pretrained.fused_prefix._complete_cache'), \
                patch('experiments.ranker.pretrained.native_batch_diagnosis.state_hash', return_value='same'), \
                patch('experiments.ranker.pretrained.parallel_decode.clone_prefix_state', return_value=state), \
                patch('experiments.ranker.pretrained.parallel_decode.exact_row_suffix', return_value=([1], [])):
            destination = Path(temporary) / 'controls'
            with self.assertRaisesRegex(AssertionError, 'distribution mismatch'):
                wide.native_controls(bb, self.bound(), destination, width=20)
            result = json.loads((destination / 'results.json').read_text())
            self.assertEqual(result['status'], 'failed-strict-control')
            self.assertEqual(len(result['arms']), 2)
            self.assertEqual(result['arms'][0]['rows'][0]['rawSampledTokenIds'], result['arms'][1]['rows'][0]['rawSampledTokenIds'])

    def test_guard_uses_caller_budget_and_never_increases_it(self):
        _, bb, _, memory = self.stubs()
        self.assertEqual(wide.DEFAULT_MEMORY_GIB, 7)
        memory['mlxPeakBytes'] = memory['processBudgetBytes'] + 1
        with self.assertRaises(MemoryError):
            wide.guard_memory(bb)
        self.assertEqual(memory['processBudgetBytes'], 7 * 2**30)
        # A separately approved caller may supply another inspected guard.
        memory['processBudgetBytes'] = 8 * 2**30
        self.assertEqual(wide.guard_memory(bb)['processBudgetBytes'], 8 * 2**30)

    def test_run_case_preserves_true_caller_start_and_serialized_endpoints(self):
        modules, bb, _, _ = self.stubs(); state = {'tokens': [1], 'cache': []}; rows = self.teacher_rows()
        for width in (10, 20):
            def decode(backbone, source, jobs, requested_width, first_ready):
                self.assertEqual(requested_width, width); self.assertEqual(len(jobs), 20)
                first_ready(rows[:10]); return rows, {'stubbed': True}
            started = time.perf_counter() - .1
            with patch('experiments.ranker.pretrained.branch_local_copy.prepare_root', return_value=(state, {})), \
                    patch('experiments.ranker.pretrained.fused_prefix._complete_cache'), \
                    patch('experiments.ranker.pretrained.native_batch_diagnosis.state_hash', return_value='same'), \
                    patch.object(wide, 'decode_complete_rows', side_effect=decode):
                result = wide.run_case(bb, self.bound(), width=width, request_started=started)
            self.assertTrue(result['requestTimingIncludesCallerRanking'])
            self.assertEqual(result['width'], width)
            self.assertTrue(result['firstThreeReceipt']['valid'])
            self.assertTrue(result['fullSixReceipt']['valid'])
            self.assertGreaterEqual(result['firstThreeReceipt']['validatedAndSerializedAtMs'], 100)
            self.assertGreaterEqual(result['fullSixReceipt']['validatedAndSerializedAtMs'], result['firstThreeReceipt']['validatedAndSerializedAtMs'])
            self.assertEqual(len(result['fullSix']['payload']['sections']), 2)
            self.assertEqual(len(wide.serialized_payload(result['fullSix'])), result['fullSixReceipt']['payloadBytes'])

    def test_incomplete_checkpoint_rejected_before_native_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary); (path / 'results.json').write_text(json.dumps({'status': 'started'}))
            with self.assertRaisesRegex(ValueError, 'Completed exact v2 fit'):
                wide.require_completed_checkpoint(path)

    def test_full_failure_preserves_valid_first_three_and_observed_rows(self):
        _, bb, _, _ = self.stubs(); state = {'tokens': [1], 'cache': []}; rows = self.teacher_rows()
        def decode(backbone, source, jobs, width, first_ready):
            first_ready(rows[:10])
            error = TimeoutError('full six failed after first3')
            error.partial_rows = rows[:15]
            raise error
        with patch('experiments.ranker.pretrained.branch_local_copy.prepare_root', return_value=(state, {})), \
                patch('experiments.ranker.pretrained.fused_prefix._complete_cache'), \
                patch('experiments.ranker.pretrained.native_batch_diagnosis.state_hash', return_value='same'), \
                patch.object(wide, 'decode_complete_rows', side_effect=decode):
            with self.assertRaises(TimeoutError) as caught:
                wide.run_case(bb, self.bound(), width=20)
        result = caught.exception.partial_result
        self.assertTrue(result['firstThreeReceipt']['valid'])
        self.assertEqual(result['rawRows'], rows[:15])
        self.assertNotIn('fullSixReceipt', result)


if __name__ == '__main__':
    unittest.main()
