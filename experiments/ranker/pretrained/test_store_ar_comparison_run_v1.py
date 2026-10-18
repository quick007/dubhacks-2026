"""CPU-only fake-engine execution tests; no native worker/model/GPU imports."""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import inspect
from io import StringIO
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .store_ar_comparison_run_v1 import ARM_ORDER, fixed_cases, main, run_comparison, summarize
from .store_autoregressive_baseline import run_resident_case
from .store_catalog_v1 import load_snapshot
from .store_demo_engine import checked_request
from .store_rank_session import action_bindings


class DriverTests(unittest.TestCase):
    def dependencies(self, root, *, listening=False, control_failure=False):
        calls = []; owned = [False]; clock_value = [100.]
        snapshot = load_snapshot()
        def clock():
            clock_value[0] += .001
            return clock_value[0]
        @contextmanager
        def lock():
            self.assertFalse(owned[0]); owned[0] = True; calls.append('lock-enter')
            try:
                yield
            finally:
                calls.append('lock-exit'); owned[0] = False
        def preflight(manifest):
            return {'version': 'test', 'nativeExecution': False, 'cases': [
                {**case, 'armOrder': list(order)} for case, order in zip(fixed_cases(), ARM_ORDER, strict=True)]}
        class Engine:
            def __init__(inner):
                self.assertTrue(owned[0]); calls.append('engine-load')
                inner.output = root / 'hybrid'; inner.adapter = object(); inner.bb = object()
                inner.startup = {'sourceTensorsHashed': 927}
                inner.session = SimpleNamespace(snapshot=snapshot)
                inner.binding = SimpleNamespace(assert_current_sources=lambda receipt: True)
            def personalize(inner, value, request_id, started, emit, cancelled):
                self.assertTrue(owned[0]); self.assertIsInstance(started, float)
                calls.append('B-' + value['query'])
                request, _ = checked_request(value)
                cards = action_bindings(request, list(range(6)), snapshot)
                sections = []
                for start in (0, 3):
                    section = {'heading': 'Browse the collection', 'cards': []}
                    for card in cards[start:start + 3]:
                        section['cards'].append({'originalName': card['facts']['name'],
                            'title': 'generated title', 'bodyCopy': 'generated body', 'badge': None})
                    sections.append(section)
                events = [{'type': 'firstThree', 'sections': sections[:1], 'acceptedFields': 9, 'requiredFields': 10,
                    'timing': {'readyMs': 1000.}}, {'type': 'complete', 'sections': sections,
                    'acceptedFields': 18, 'requiredFields': 20, 'timing': {'readyMs': 1500.}}]
                directory = inner.output / 'requests' / request_id; directory.mkdir(parents=True)
                (directory / 'receipt.json').write_text(json.dumps({'status': 'completed', 'input': value,
                    'cards': cards, 'events': events, 'decodeTiming': {'actualSampledTokens': 173}}))
                for event in events:
                    emit(event)
            def close(inner):
                self.assertTrue(owned[0]); calls.append('engine-close')
                return {'frozen927HashesUnchanged': True, 'copyCheckpointUnchanged': True}
        def baseline(bb, value, snapshot, out, **kwargs):
            self.assertTrue(owned[0]); self.assertIsInstance(kwargs['request_started'], float)
            self.assertEqual(kwargs['mode'], 'frozen-base')
            calls.append('A-' + value['query'])
            return {'status': 'completed-attempt', 'summary': {'firstThreeAttemptMs': 2000., 'fullAttemptMs': 3000.,
                'usableFirstThreeMs': 2000., 'usableFullMs': 3000.}, 'generatedTokens': 400,
                'firstThree': {'validation': {'allPracticalFieldsAccepted': True}},
                'full': {'cardCount': 6, 'validation': {'requiredFields': 20, 'allPracticalFieldsAccepted': True}}}
        def controls(engine, out):
            self.assertTrue(owned[0]); calls.append('controls')
            if control_failure:
                raise ValueError('test native cache parity failure')
            return [{'status': 'exact-controls-passed'}]
        return {'service_listening': lambda: listening, 'preflight': preflight, 'native_lock': lock,
            'engine_factory': Engine, 'baseline': baseline,
            'reset_runtime': lambda engine: calls.append('reset'), 'cache_controls': controls, 'clock': clock}, calls

    def test_no_run_only_prepares_cpu_without_service_or_native_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root, listening=True)
            result = run_comparison(root / 'prepare', _dependencies=deps)
            self.assertEqual(result['status'], 'cpu-preflight-only')
            self.assertFalse(result['nativeExecution']); self.assertEqual(calls, [])

    def test_service_refused_before_creating_output_or_taking_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root, listening=True)
            with self.assertRaisesRegex(RuntimeError, 'listening'):
                run_comparison(root / 'run', run=True, _dependencies=deps)
            self.assertFalse((root / 'run').exists()); self.assertEqual(calls, [])

    def test_one_engine_fixed_ab_ba_ab_warmups_and_close_inside_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root)
            result = run_comparison(root / 'run', run=True, _dependencies=deps)
            self.assertEqual(result['status'], 'completed-comparison')
            self.assertEqual([r['arm'] for r in result['records']], ['A', 'B', 'B', 'A', 'A', 'B'])
            self.assertEqual([r['arm'] for r in result['warmups']], ['A', 'B'])
            self.assertEqual(calls.count('engine-load'), 1)
            self.assertEqual(calls[-2:], ['engine-close', 'lock-exit'])
            self.assertTrue(result['shutdownIntegrity']['frozen927HashesUnchanged'])
            self.assertTrue(all(r['cardCount'] == 6 and r['copyFields'] == 20 for r in result['records']))
            self.assertEqual(result['summary']['conventional-frozen-base']['attempts'], 3)
            self.assertEqual(result['summary']['hybrid-v3-rank-copy-v2']['attempts'], 3)

    def test_control_failure_preserves_report_and_closes_before_unlock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root, control_failure=True)
            result = run_comparison(root / 'run', run=True, _dependencies=deps)
            self.assertEqual(result['status'], 'failed-comparison')
            self.assertEqual(result['records'], [])
            self.assertEqual(calls[-2:], ['engine-close', 'lock-exit'])
            saved = json.loads((root / 'run/results.json').read_text())
            self.assertIn('parity failure', saved['error']['message'])

    def test_conventional_call_deadline_covers_prefill_before_first_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root)
            ordinary = deps['baseline']; attempts = [0]
            def blocked_prefill(*args, **kwargs):
                attempts[0] += 1
                if attempts[0] == 2:  # Keep the warmup; interrupt the measured A arm.
                    signal.raise_signal(signal.SIGALRM)
                    self.fail('Expired deadline returned to blocked prefill')
                return ordinary(*args, **kwargs)
            deps['baseline'] = blocked_prefill
            previous = signal.getsignal(signal.SIGALRM)
            def unguarded(_signum, _frame):
                raise AssertionError('Baseline prefill had no active deadline')
            signal.signal(signal.SIGALRM, unguarded)
            try:
                result = run_comparison(root / 'run', run=True, _dependencies=deps)
                self.assertIs(signal.getsignal(signal.SIGALRM), unguarded)
                self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)
            finally:
                signal.signal(signal.SIGALRM, previous)
            self.assertEqual(result['status'], 'failed-comparison')
            self.assertEqual(result['error']['type'], 'TimeoutError')
            self.assertIn('fixed 120-second deadline', result['error']['message'])
            self.assertEqual(len(result['records']), 1)
            self.assertEqual(result['records'][0]['status'], 'driver-failed-attempt')
            self.assertEqual(result['records'][0]['arm'], 'A')
            self.assertEqual(calls[-2:], ['engine-close', 'lock-exit'])
            saved = json.loads((root / 'run/results.json').read_text())
            self.assertEqual(saved['records'][0]['error']['type'], 'TimeoutError')

    def test_hybrid_call_has_same_deadline_before_ranking_or_root_returns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root)
            ordinary_factory = deps['engine_factory']; attempts = [0]
            def factory():
                engine = ordinary_factory(); ordinary = engine.personalize
                def blocked_root(*args, **kwargs):
                    attempts[0] += 1
                    if attempts[0] == 2:  # Keep B warmup; interrupt measured B.
                        signal.raise_signal(signal.SIGALRM)
                        self.fail('Expired deadline returned to blocked hybrid root')
                    return ordinary(*args, **kwargs)
                engine.personalize = blocked_root
                return engine
            deps['engine_factory'] = factory
            previous = signal.getsignal(signal.SIGALRM)
            def unguarded(_signum, _frame):
                raise AssertionError('Hybrid root had no active deadline')
            signal.signal(signal.SIGALRM, unguarded)
            try:
                result = run_comparison(root / 'run', run=True, _dependencies=deps)
                self.assertIs(signal.getsignal(signal.SIGALRM), unguarded)
                self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)
            finally:
                signal.signal(signal.SIGALRM, previous)
            self.assertEqual(result['status'], 'failed-comparison')
            self.assertEqual(result['error']['type'], 'TimeoutError')
            self.assertIn('fixed 120-second deadline', result['error']['message'])
            self.assertEqual([row['arm'] for row in result['records']], ['A', 'B'])
            self.assertEqual(result['records'][-1]['status'], 'driver-failed-attempt')
            self.assertEqual(calls[-2:], ['engine-close', 'lock-exit'])

    def test_existing_output_is_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root)
            with self.assertRaises(FileExistsError):
                run_comparison(root, run=True, _dependencies=deps)
            self.assertEqual(calls, [])

    def test_legacy_helper_has_optional_common_request_timer(self):
        parameter = inspect.signature(run_resident_case).parameters['request_started']
        self.assertIsNone(parameter.default)

    def test_failed_proxy_does_not_become_speedup_or_usable_latency(self):
        result = summarize([{'arm': 'B', 'status': 'completed', 'cardCount': 6, 'copyFields': 20,
            'firstThreeAttemptMs': 1000., 'fullAttemptMs': 1500., 'usableFirstThreeMs': None, 'usableFullMs': None}])
        hybrid = result['hybrid-v3-rank-copy-v2']
        self.assertEqual(hybrid['medianAttemptFullMs'], 1500.)
        self.assertIsNone(hybrid['medianUsableFullMs'])
        self.assertEqual(hybrid['fullSixUsableProxyPasses'], 0)

    def test_native_cli_failure_status_is_nonzero_but_success_and_cpu_preflight_are_not(self):
        module = 'experiments.ranker.pretrained.store_ar_comparison_run_v1'
        for status in ('failed-comparison', 'failed-shutdown', 'failed-shutdown-integrity'):
            output = StringIO()
            with patch(module + '.run_comparison', return_value={'status': status, 'nativeExecution': True}), \
                    patch('sys.argv', ['driver', '--run', '--out', 'unused']), redirect_stdout(output):
                with self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 1)
            self.assertEqual(json.loads(output.getvalue())['status'], status)
        for status, native in (('completed-comparison', True), ('cpu-preflight-only', False)):
            arguments = ['driver', '--out', 'unused'] + (['--run'] if native else [])
            with patch(module + '.run_comparison', return_value={'status': status, 'nativeExecution': native}), \
                    patch('sys.argv', arguments), redirect_stdout(StringIO()):
                self.assertIsNone(main())


if __name__ == '__main__':
    unittest.main()
