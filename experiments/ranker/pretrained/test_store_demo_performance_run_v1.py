"""CPU-only failure and accounting tests; fake engines never load native MLX."""
from contextlib import contextmanager, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from . import store_demo_engine
from . import store_demo_performance_run_v1 as driver
from .store_copy_data_v2 import CAPS, FIELDS


class PerformanceDriverTests(unittest.TestCase):
    def dependencies(self, root, *, mutate=None, shutdown=None, listening=None):
        calls = []; owned = [False]; ticks = [0.]; sources = {}
        original_decode = store_demo_engine.decode_rows
        preflight = {'version': 'fake-cpu-preflight', 'nativeExecution': False,
            'cases':[{'caseId':str(i),'armOrder':['A','B'],'conventionalPromptTokens':1334,
                'conventionalPromptSha256':'unused'} for i in range(3)]}
        def profile(*args, **kwargs):
            raise AssertionError('Fake engine must never call a native decoder')
        def clock():
            ticks[0] += .001
            return ticks[0]
        def source_sha(path):
            key = Path(path).name
            return sources.get(key, 'frozen-' + key)
        @contextmanager
        def lock():
            self.assertFalse(owned[0]); owned[0] = True; calls.append('lock-enter')
            try:
                yield
            finally:
                calls.append('lock-exit'); owned[0] = False
        class Engine:
            def __init__(inner):
                self.assertTrue(owned[0]); calls.append('engine-load')
                inner.output = root / 'engine'; inner.startup = {'fake': True}
                inner.index = 0
            def personalize(inner, request, request_id, started, emit, cancelled):
                self.assertTrue(owned[0]); self.assertIsInstance(started, float)
                self.assertFalse(cancelled())
                mode = 'profile-v2' if store_demo_engine.decode_rows is profile else 'baseline'
                if mode == 'baseline':
                    self.assertIs(store_demo_engine.decode_rows, original_decode)
                calls.append(mode); inner.index += 1
                value = {'status': 'completed',
                    'cards': [{'facts': {'name': str(i)}, 'action': {'productId': i}} for i in range(6)],
                    'finalOrder': list(range(16)),
                    'rawRows': [{'fields': [field], 'maxTokens': CAPS[field[0]], 'tokenIds': [7],
                        'rawSampledTokenIds': [7, 99], 'stopTokenId': 99, 'arSteps': 2,
                        'finishReason': 'stop', 'text': 'fixture'} for field in FIELDS],
                    'events': [{'type': 'rank', 'timing': {'readyMs': 10.}},
                        {'type': 'firstThree', 'timing': {'readyMs': 100.}},
                        {'type': 'complete', 'timing': {'readyMs': 200.},
                            'acceptedFields': 20, 'requiredFields': 20}],
                    'preparation': {'fake': True}, 'decodeTiming': {'fake': True}, 'memory': {}}
                if mutate:
                    mutate(value, mode, inner.index)
                # Warmup timings are intentionally huge: exclusion must affect medians.
                if inner.index == 1:
                    for event in value['events']:
                        event['timing']['readyMs'] = 99999.
                directory = inner.output / 'requests' / request_id
                directory.mkdir(parents=True)
                (directory / 'receipt.json').write_text(json.dumps(value))
                for event in value['events']:
                    emit(event)
            def close(inner):
                self.assertTrue(owned[0]); calls.append('engine-close')
                if isinstance(shutdown, BaseException):
                    raise shutdown
                return shutdown if shutdown is not None else {
                    'frozen927HashesUnchanged': True, 'copyCheckpointUnchanged': True}
        return {'preflight': lambda manifest: preflight, 'source_sha': source_sha,
            'service_listening': listening or (lambda: False), 'lock': lock,
            'factory': Engine, 'profile': profile, 'clock': clock,
            'reset': lambda engine: calls.append('reset')}, calls, sources, preflight

    def test_cpu_preflight_does_not_touch_service_lock_engine_or_mutate_source_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deps, calls, _, original = self.dependencies(root,
                listening=lambda: self.fail('CPU-only preflight touched service'))
            report = driver.run(root / 'run', _dependencies=deps)
            self.assertEqual(report['status'], 'cpu-preflight-only')
            self.assertFalse(report['nativeExecution']); self.assertEqual(calls, [])
            self.assertNotIn('performanceVersion', original)
            value=report['preflight']
            self.assertEqual(value['version'], 'store-demo-performance.v1')
            self.assertEqual([c['armOrder'] for c in value['cases']], [list(pair) for pair in driver.ORDER])
            self.assertTrue(all('conventionalPromptTokens' not in c for c in value['cases']))
            self.assertEqual(original['cases'][0]['armOrder'], ['A','B'])
            self.assertEqual(value['cachePolicy'], driver.CACHE_POLICY)

    def test_service_refusal_happens_before_output_and_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls, _, _ = self.dependencies(root, listening=lambda: True)
            with self.assertRaisesRegex(RuntimeError, 'listening'):
                driver.run(root / 'run', native=True, _dependencies=deps)
            self.assertFalse((root / 'run').exists()); self.assertEqual(calls, [])

    def test_service_starting_during_preflight_is_refused_under_lock_before_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); states = iter((False, True))
            deps, calls, _, _ = self.dependencies(root, listening=lambda: next(states))
            report = driver.run(root / 'run', native=True, _dependencies=deps)
            self.assertEqual(report['status'], 'failed-profile-comparison')
            self.assertEqual(calls, ['lock-enter', 'lock-exit'])
            self.assertEqual(report['summary']['baseline']['unattempted'], 3)

    def test_single_engine_ab_ba_ab_full20_exact_parity_and_warmup_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls, _, _ = self.dependencies(root)
            original = store_demo_engine.decode_rows
            report = driver.run(root / 'run', native=True, _dependencies=deps)
            self.assertEqual(report['status'], 'completed-profile-comparison')
            self.assertEqual([r['mode'] for r in report['records']],
                ['baseline', 'baseline', 'profile-v2', 'profile-v2', 'baseline', 'baseline', 'profile-v2'])
            self.assertEqual(calls.count('engine-load'), 1); self.assertEqual(calls.count('reset'), 7)
            self.assertEqual(calls[-2:], ['engine-close', 'lock-exit'])
            self.assertIs(store_demo_engine.decode_rows, original)
            self.assertTrue(all(p['fullTokensBindingsAndEOSExact'] for p in report['parity']))
            for mode in ('baseline', 'profile-v2'):
                self.assertEqual(report['summary'][mode], {'fixedCases': 3, 'attempts': 3,
                    'completedAttempts': 3, 'failedAttempts': 0, 'unattempted': 0,
                    'rankMs': 10., 'firstThreeAttemptMs': 100., 'fullAttemptMs': 200.})
            self.assertTrue(all(r['cardCount'] == 6 and r['requiredFields'] == 20 for r in report['records']))
            self.assertTrue(report['runtimeSourceHashesUnchanged'])
            self.assertEqual(json.loads((root / 'run/results.json').read_text()), report)

    def test_parity_checks_every_token_eos_and_product_binding(self):
        mutations = {
            'tokens': lambda value: value['rawRows'][-1]['tokenIds'].append(8),
            'raw-eos': lambda value: value['rawRows'][-1]['rawSampledTokenIds'].__setitem__(-1, 98),
            'stop': lambda value: value['rawRows'][-1].__setitem__('stopTokenId', 98),
            'binding': lambda value: value['cards'][-1]['action'].__setitem__('productId', 99),
            'order': lambda value: value['finalOrder'].reverse(),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                def mutate(value, mode, index):
                    if mode == 'profile-v2':
                        mutation(value)
                deps, calls, _, _ = self.dependencies(root, mutate=mutate)
                report = driver.run(root / 'run', native=True, _dependencies=deps)
                self.assertEqual(report['status'], 'failed-profile-comparison')
                self.assertFalse(report['parity'][0]['fullTokensBindingsAndEOSExact'])
                self.assertEqual(calls[-2:], ['engine-close', 'lock-exit'])

    def test_full_workload_checks_actual_roster_caps_and_cards_not_just_claimed_count(self):
        mutations = {
            'missing-row': lambda v: v['rawRows'].pop(),
            'reordered-row': lambda v: v['rawRows'].reverse(),
            'redistributed-cap': lambda v: (v['rawRows'][0].__setitem__('maxTokens', 17),
                v['rawRows'][1].__setitem__('maxTokens', 23)),
            'missing-card': lambda v: v['cards'].pop(),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                deps, _, _, _ = self.dependencies(root, mutate=lambda v, m, i: mutation(v))
                report = driver.run(root / 'run', native=True, _dependencies=deps)
                self.assertEqual(report['status'], 'failed-profile-comparison')
                self.assertEqual(report['records'][0]['status'], 'failed-attempt')
                self.assertIn('416token cap', report['error']['message'])

    def test_exception_retains_failed_request_identity_and_fixed_denominators(self):
        def mutate(value, mode, index):
            if index == 3:
                raise RuntimeError('fake observer failure')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls, _, _ = self.dependencies(root, mutate=mutate)
            original = store_demo_engine.decode_rows
            report = driver.run(root / 'run', native=True, _dependencies=deps)
            row = report['records'][-1]
            self.assertEqual(row['status'], 'failed-attempt'); self.assertIn(row['requestId'], row['resultPath'])
            self.assertFalse(Path(row['resultPath']).exists()); self.assertGreater(row['failedAtMs'], 0)
            self.assertEqual(report['summary']['profile-v2']['failedAttempts'], 1)
            self.assertEqual(report['summary']['profile-v2']['unattempted'], 2)
            self.assertIsNone(report['summary']['profile-v2']['fullAttemptMs'])
            self.assertEqual(report['summary']['baseline']['completedAttempts'], 1)
            self.assertEqual(calls[-2:], ['engine-close', 'lock-exit'])
            self.assertIs(store_demo_engine.decode_rows, original)

    def test_invalid_native_timing_is_preserved_as_failure_without_nonfinite_json(self):
        for bad_time in (float('nan'), float('inf'), -1., 201., True):
            with self.subTest(bad_time=bad_time), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                def mutate(value, mode, index):
                    if index == 2:
                        value['events'][1]['timing']['readyMs'] = bad_time
                deps, _, _, _ = self.dependencies(root, mutate=mutate)
                report = driver.run(root / 'run', native=True, _dependencies=deps)
                self.assertEqual(report['status'], 'failed-profile-comparison')
                self.assertIn('finite, nonnegative and ordered', report['error']['message'])
                saved = json.loads((root / 'run/results.json').read_text())
                self.assertEqual(saved['records'][-1]['status'], 'failed-attempt')
                self.assertEqual(saved['summary']['baseline']['failedAttempts'], 1)
                self.assertIsNone(saved['summary']['baseline']['fullAttemptMs'])

    def test_deadline_covers_request_before_any_callback_and_restores_signal(self):
        def mutate(value, mode, index):
            signal.raise_signal(signal.SIGALRM)
            self.fail('Expired request returned without deadline')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls, _, _ = self.dependencies(root, mutate=mutate)
            previous = signal.getsignal(signal.SIGALRM)
            def unguarded(*args):
                self.fail('Request had no active deadline')
            signal.signal(signal.SIGALRM, unguarded)
            try:
                report = driver.run(root / 'run', native=True, _dependencies=deps)
                self.assertIs(signal.getsignal(signal.SIGALRM), unguarded)
                self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))
            finally:
                signal.signal(signal.SIGALRM, previous)
            self.assertEqual(report['error']['type'], 'TimeoutError')
            self.assertEqual(report['records'][0]['status'], 'failed-attempt')
            self.assertEqual(calls[-2:], ['engine-close', 'lock-exit'])

    def test_shutdown_exception_or_false_integrity_fails_and_releases_lock(self):
        for shutdown in (RuntimeError('close failed'),
                {'frozen927HashesUnchanged': False, 'copyCheckpointUnchanged': True}):
            with self.subTest(shutdown=shutdown), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); deps, calls, _, _ = self.dependencies(root, shutdown=shutdown)
                report = driver.run(root / 'run', native=True, _dependencies=deps)
                self.assertEqual(report['status'], 'failed-profile-comparison')
                self.assertIn('shutdownError', report)
                self.assertEqual(calls[-2:], ['engine-close', 'lock-exit'])

    def test_source_change_fails_final_provenance_without_overwriting_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls, sources, _ = self.dependencies(root)
            factory = deps['factory']
            def changed_factory():
                sources['store_demo_decode_profile_v2.py'] = 'modified'
                return factory()
            deps['factory'] = changed_factory
            report = driver.run(root / 'run', native=True, _dependencies=deps)
            self.assertEqual(report['status'], 'failed-profile-comparison')
            self.assertIn('sourceError', report)
            self.assertNotEqual(json.loads((root / 'run/preflight.json').read_text())['profileSourceSha256'], 'modified')
            self.assertEqual(calls[-2:], ['engine-close', 'lock-exit'])

    def test_factory_failure_releases_ownership_and_serializes_empty_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls, _, _ = self.dependencies(root)
            def fail():
                raise RuntimeError('native load failed')
            deps['factory'] = fail
            report = driver.run(root / 'run', native=True, _dependencies=deps)
            self.assertEqual(report['status'], 'failed-profile-comparison')
            self.assertEqual(calls, ['lock-enter', 'lock-exit'])
            self.assertEqual(report['records'], [])
            self.assertEqual(report['summary']['baseline']['unattempted'], 3)

    def test_existing_outputs_and_invalid_widths_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, _, _, _ = self.dependencies(root)
            with self.assertRaises(FileExistsError):
                driver.run(root, _dependencies=deps)
            for width in (6, True, 10., 11):
                with self.subTest(width=width), self.assertRaises(ValueError):
                    driver.run(root / 'new', width=width, _dependencies=deps)

    def test_failed_native_cli_returns_nonzero(self):
        for status, expected in (('failed-profile-comparison', 1),
                ('completed-profile-comparison', 0), ('cpu-preflight-only', 0)):
            with self.subTest(status=status), patch.object(driver, 'run', return_value={
                    'status': status, 'nativeExecution': status != 'cpu-preflight-only'}), \
                    patch('sys.argv', ['performance', '--out', '/unused']), redirect_stdout(StringIO()):
                self.assertEqual(driver.main(), expected)


if __name__ == '__main__':
    unittest.main()
