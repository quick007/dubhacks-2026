"""CPU fakes only: no real native generation, model load or latency claim."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from . import store_generated_copy_cache_run_v1 as driver
from .store_catalog_v1 import load_snapshot
from .store_copy_data_v2 import target_values
from .store_demo_engine import demo_layout
from .store_generated_copy_cache_v1 import producer_spec, sha, utc_now, warm_plan


class CacheRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = load_snapshot()

    def dependencies(self, root, *, listening=None, bad_shutdown=False, fail_generation=False, bad_control=False):
        calls = []; owned = [False]; requests = []
        spec = producer_spec('a' * 64, 'test-revision', 'b' * 64)
        cases, manifest = driver.public_frontier()
        prepared = {'version': driver.VERSION, 'snapshot': self.snapshot, 'producerSpec': spec,
            'warmPlan': warm_plan(self.snapshot), 'cases': cases, 'guardProbes': driver.guard_probes(self.snapshot),
            'sourceSha256': {name: sha(driver.HERE / name) for name in driver.SOURCES}}
        @contextmanager
        def lock():
            self.assertFalse(owned[0]); owned[0] = True; calls.append('lock-enter')
            try:
                yield
            finally:
                calls.append('lock-exit'); owned[0] = False
        def rank(request, *, copy_adapter):
            self.assertTrue(owned[0]); requests.append(deepcopy(request))
            self.assertEqual(set(request), {'query', 'preferences', 'itemBudget', 'hardRequirements'})
            self.assertNotIn('caseId', json.dumps(request))
            return {'scores16': list(range(16, 0, -1)), 'fakeNative': True}
        class Engine:
            def __init__(inner):
                self.assertTrue(owned[0]); calls.append('engine-load')
                inner.copy_sha = spec['checkpointSha256']; inner.adapter = object()
                inner.startup = {'fake': True}; inner.session = types.SimpleNamespace(
                    snapshot=self.snapshot, products=self.snapshot['inputs']['products'], rank_request=rank)
                inner.bb = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
                    eos_token_ids=[99], decode=lambda ids: 'Native value.'))
            def close(inner):
                self.assertTrue(owned[0]); calls.append('engine-close')
                return {'frozen927HashesUnchanged': not bad_shutdown, 'copyCheckpointUnchanged': True}
        def controls(engine, bound, out):
            self.assertTrue(owned[0]); calls.append('controls')
            self.assertEqual(len(bound['cards']), 6)
            out.mkdir(parents=True); value = {'status': 'failed' if bad_control else 'exact-controls-passed', 'width': 10}
            (out / 'results.json').write_text(json.dumps(value))
            return value
        def generate(engine, cards, given_spec, control_sha, output):
            self.assertTrue(owned[0]); calls.append('generate')
            self.assertEqual(given_spec, spec); self.assertEqual(len(control_sha), 64)
            self.assertEqual(len(cards), 6)
            if fail_generation:
                raise RuntimeError('injected native failure')
            _, rows = demo_layout(cards); self.assertEqual(len(rows), 20)
            self.assertEqual(sum(r['maxTokens'] for r in rows), 416)
            values = target_values({'cards': cards}, 0)
            for row in rows:
                row.update(text=values[row['fields'][0]], tokenIds=[5], rawSampledTokenIds=[5, 99],
                    stopTokenId=99, arSteps=2, finishReason='stop')
            at = utc_now()
            result = {'status': 'completed', 'cards': cards, 'actualTokenizerDecodeAndEOSVerified': True,
                'rows': rows, 'startedAt': at, 'completedAt': at, 'coldGenerationMs': 8000,
                'provenance': {'nativeGeneration': True, 'fallback': False, 'checkpointSha256': spec['checkpointSha256'],
                    'modelRevision': spec['modelRevision'], 'tokenizerSha256': spec['tokenizerSha256'],
                    'nativeWidth': 10, 'sameWidthControlsSha256': control_sha}}
            output.mkdir(parents=True)
            (output / 'native.json').write_text(json.dumps(result))
            return result
        labels = json.loads((driver.FRONTIER / 'offline-relevance-labels.json').read_text())
        return {'prepare': lambda *_: deepcopy(prepared), 'lock': lock, 'factory': Engine,
            'controls': controls, 'generate': generate, 'reset': lambda _: calls.append('reset'),
            'service_listening': listening or (lambda: False), 'labels': lambda: labels}, calls, requests

    def test_import_and_cpu_preparation_no_mlx_or_native_ownership(self):
        subprocess.run([sys.executable, '-c', 'import sys; import experiments.ranker.pretrained.store_generated_copy_cache_run_v1; '
            'assert not any(k == "mlx" or k.startswith("mlx.") for k in sys.modules)'], check=True, capture_output=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls, _ = self.dependencies(root,
                listening=lambda: self.fail('CPU preparation touched service'))
            report = driver.run(root / 'run', _dependencies=deps)
            self.assertEqual(report['status'], 'cpu-preflight-only'); self.assertEqual(calls, [])
            self.assertTrue((root / 'run/preflight.json').is_file())

    def test_refuse_active_service_before_any_output_or_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls, _ = self.dependencies(root, listening=lambda: True)
            with self.assertRaisesRegex(RuntimeError, 'listening'):
                driver.run(root / 'run', native=True, _dependencies=deps)
            self.assertFalse((root / 'run').exists()); self.assertEqual(calls, [])

    def test_recheck_service_under_lock_and_preserve_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); states = iter((False, True))
            deps, calls, _ = self.dependencies(root, listening=lambda: next(states))
            report = driver.run(root / 'run', native=True, _dependencies=deps)
            self.assertEqual(report['status'], 'failed-cache-candidate')
            self.assertEqual(calls, ['lock-enter', 'lock-exit'])
            self.assertEqual(report['summary']['unattempted'], 32)
            with self.assertRaises(FileExistsError):
                driver.run(root / 'run', _dependencies=deps)

    def test_full_roster_single_owner_controls_then48colors_fresh32queries_and_cold_then_hit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls, requests = self.dependencies(root)
            report = driver.run(root / 'run', native=True, _dependencies=deps)
            self.assertEqual(report['status'], 'completed-cache-candidate', report.get('error'))
            self.assertEqual(calls.count('engine-load'), 1); self.assertEqual(calls.count('generate'), 9)
            self.assertLess(calls.index('controls'), calls.index('generate'))
            self.assertEqual(calls[-2:], ['engine-close', 'lock-exit'])
            self.assertEqual(report['warmSummary']['completed'], 8)
            self.assertEqual(report['warmSummary']['originalColdGenerationMs'], [8000] * 8)
            self.assertEqual(len(requests), 41)  #32 frontier+7guards+2missproof
            self.assertEqual(report['summary']['completed'], 32); self.assertEqual(report['summary']['unattempted'], 0)
            frontier = [r for r in report['records'] if r['phase'] == 'frontier']
            self.assertTrue(all(r['quality']['cardCount'] == 6 and r['quality']['requiredFields'] == 20 for r in frontier))
            self.assertTrue(all(r['cache']['copySource'] == 'generated-earlier' and not r['cache']['freshPerRequestGenerationClaim'] for r in frontier))
            self.assertTrue(all(r['firstThreeAvailableAtMs'] == r['fullSerializedReadyMs'] for r in frontier))
            self.assertEqual(report['firstThreeTimingScope'], driver.FIRST3_SCOPE)
            self.assertTrue(all(r['guardCheck']['passed'] for r in report['records'] if r['phase'] == 'guards'))
            cold = next(r for r in report['records'] if r['phase'] == 'cold-miss')
            warm = next(r for r in report['records'] if r['phase'] == 'warm-repeat')
            self.assertTrue(cold['cache']['nativeMissGeneration']); self.assertFalse(warm['cache']['nativeMissGeneration'])
            self.assertEqual(warm['cache']['copySource'], 'generated-earlier')
            self.assertTrue(report['shutdownIntegrity']['frozen927HashesUnchanged'])
            for row in frontier:
                path = root / 'run' / ('frontier-' + row['caseId'] + '.payload.json')
                self.assertEqual(sha(path), row['payloadSha256']); self.assertEqual(path.stat().st_size, row['payloadBytes'])

    def test_native_failure_preserves_warm_failure_and_fixed_denominators_and_closes_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls, _ = self.dependencies(root, fail_generation=True)
            report = driver.run(root / 'run', native=True, _dependencies=deps)
            self.assertEqual(report['status'], 'failed-cache-candidate')
            self.assertEqual(report['warmRecords'][0]['status'], 'failed')
            self.assertEqual(report['summary']['unattempted'], 32)
            self.assertEqual(report['warmSummary']['unattempted'], 7)
            self.assertEqual(calls[-2:], ['engine-close', 'lock-exit'])

    def test_failed_controls_prevent_every_native_warm_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls, _ = self.dependencies(root, bad_control=True)
            report = driver.run(root / 'run', native=True, _dependencies=deps)
            self.assertEqual(report['status'], 'failed-cache-candidate')
            self.assertNotIn('generate', calls); self.assertEqual(report['warmSummary']['unattempted'], 8)

    def test_shutdown927_failure_invalidates_finished_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls, _ = self.dependencies(root, bad_shutdown=True)
            report = driver.run(root / 'run', native=True, _dependencies=deps)
            self.assertEqual(report['status'], 'failed-cache-candidate')
            self.assertIn('shutdownError', report); self.assertEqual(report['summary']['completed'], 32)

    def test_quality_never_equates20_attempts_with_useful_title_body(self):
        result = {'payload': {'sections': [{'cards': [{}] * 6}], 'acceptedFields': 19, 'withheldFields': 1, 'requiredFields': 20},
            'validation': {'fields': [{'field': f'{kind}{i}', 'accepted': not (kind == 'C' and i == 4)}
                for i in range(1, 7) for kind in ('T', 'B', 'C')] + [{'field': 'S1', 'accepted': True}, {'field': 'S2', 'accepted': True}]}}
        value = driver.quality(result)
        self.assertTrue(value['firstThree']['coreTitleBodyPass']); self.assertFalse(value['fullSix']['coreTitleBodyPass'])
        self.assertFalse(value['fullSix']['allFieldsPass']); self.assertEqual(value['fullSix']['coreAccepted'], 11)

    def test_actual_producer_checks_decode_and_eos_and_preserves_failed_native_tokens(self):
        cards = warm_plan(self.snapshot)['groups'][0]['cards']
        spec = producer_spec('a' * 64, 'test-revision', 'b' * 64)
        for failure in (None, 'wrong-text', 'wrong-eos'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                _, rows = demo_layout(cards); values = target_values({'cards': cards}, 0)
                decoded = {}
                for index, row in enumerate(rows):
                    stop = 98 if failure == 'wrong-eos' else 99
                    text = values[row['fields'][0]]; decoded[index + 1] = text
                    row.update(text=text, tokenIds=[index + 1], rawSampledTokenIds=[index + 1, stop],
                        stopTokenId=stop, arSteps=2, finishReason='stop')
                tokenizer = types.SimpleNamespace(eos_token_ids=[99], decode=lambda ids:
                    'tampered' if failure == 'wrong-text' else decoded[ids[0]])
                engine = types.SimpleNamespace(bb=types.SimpleNamespace(tokenizer=tokenizer))
                out = Path(temporary) / 'native'
                with patch('experiments.ranker.pretrained.branch_local_copy.prepare_root', return_value=({'cache': [], 'tokens': [1]}, {})), \
                        patch('experiments.ranker.pretrained.fused_prefix._complete_cache'), \
                        patch('experiments.ranker.pretrained.native_batch_diagnosis.state_hash', return_value='unchanged'), \
                        patch('experiments.ranker.pretrained.store_demo_engine.decode_rows', return_value=(rows, {'fake': True})), \
                        patch('experiments.ranker.pretrained.store_copy_wide_v2.guard_memory', return_value={}):
                    if failure:
                        with self.assertRaisesRegex(AssertionError, 'tokenizer'):
                            driver.native_generate(engine, cards, spec, 'c' * 64, out)
                    else:
                        result = driver.native_generate(engine, cards, spec, 'c' * 64, out)
                        self.assertTrue(result['actualTokenizerDecodeAndEOSVerified'])
                saved = json.loads((out / 'native.json').read_text())
                self.assertEqual(saved['rows'], rows)
                self.assertEqual(saved['status'], 'failed' if failure else 'completed')
                self.assertGreaterEqual(saved['coldGenerationMs'], 0)
                self.assertTrue(saved['startedAt']); self.assertTrue(saved['completedAt'])

    def test_actual_native_callback_on_cold_miss_uses_owned_outer_deadline_without_nested_alarm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls, _ = self.dependencies(root)
            deps['generate'] = driver.native_generate
            def decode(bb, state, jobs, width, first, cancelled):
                rows = deepcopy(jobs)
                for row in rows:
                    row.update(text='Native value.', tokenIds=[5], rawSampledTokenIds=[5, 99],
                        stopTokenId=99, arSteps=2, finishReason='stop')
                return rows, {'fake': True}
            with patch('experiments.ranker.pretrained.branch_local_copy.prepare_root', return_value=({'cache': [], 'tokens': [1]}, {})), \
                    patch('experiments.ranker.pretrained.fused_prefix._complete_cache'), \
                    patch('experiments.ranker.pretrained.native_batch_diagnosis.state_hash', return_value='unchanged'), \
                    patch('experiments.ranker.pretrained.store_demo_engine.decode_rows', side_effect=decode), \
                    patch('experiments.ranker.pretrained.store_copy_wide_v2.guard_memory', return_value={}):
                report = driver.run(root / 'run', native=True, _dependencies=deps)
            self.assertEqual(report['status'], 'completed-cache-candidate', report.get('error'))
            cold = json.loads((root / 'run/fresh-miss-native/native.json').read_text())
            warm = json.loads((root / 'run/warm-native/catalog-colors-00/native.json').read_text())
            self.assertEqual(cold['deadlineOwner'], 'complete-request')
            self.assertEqual(warm['deadlineOwner'], 'native-generation')
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)
            self.assertFalse(driver._REQUEST_DEADLINE_OWNED.get())

    def test_unscored_relevance_is_unknown_and_never_reported_as_zero_matches(self):
        result = driver.summarize([])
        self.assertIsNone(result['top1PublicFunctionMatches'])
        self.assertEqual(result['top1PublicFunctionScored'], 0)
        self.assertEqual(result['top1PublicFunctionUnscored'], 32)

    def test_verified_warm_reuse_skips48colors_preserves_original_and_runs_real_new_miss_callback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); old_deps, _, _ = self.dependencies(root)
            original = driver.run(root / 'original', native=True, _dependencies=old_deps)
            self.assertEqual(original['status'], 'completed-cache-candidate')
            before = {str(path.relative_to(root / 'original')): sha(path)
                for path in (root / 'original').rglob('*') if path.is_file()}
            deps, calls, _ = self.dependencies(root)
            report = driver.run(root / 'repair', native=True, reuse_warm_cache=root / 'original', _dependencies=deps)
            self.assertEqual(report['status'], 'completed-cache-candidate', report.get('error'))
            self.assertEqual(calls.count('generate'), 1)
            self.assertEqual(report['warmSummary']['newNativeWarmingPages'], 0)
            self.assertEqual(report['warmSummary']['reusedProductColors'], 48)
            self.assertTrue(report['reusedOriginalFilesUnchanged'])
            after = {str(path.relative_to(root / 'original')): sha(path)
                for path in (root / 'original').rglob('*') if path.is_file()}
            self.assertEqual(before, after)
            self.assertFalse((root / 'repair/warm-cache').exists())
            # Altering raw native provenance must refuse reuse before any output.
            raw_path = root / 'original/warm-native/catalog-colors-00/native.json'
            raw = json.loads(raw_path.read_text()); raw['actualTokenizerDecodeAndEOSVerified'] = False
            raw_path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, 'not fully verified'):
                driver.run(root / 'bad-repair', _dependencies=deps, reuse_warm_cache=root / 'original')
            self.assertFalse((root / 'bad-repair').exists())


if __name__ == '__main__':
    unittest.main()
