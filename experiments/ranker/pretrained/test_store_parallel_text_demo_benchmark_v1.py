"""CPU fake-engine benchmark tests; no native model/tokenizer execution."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import store_parallel_text_demo_benchmark_v1 as bench
from . import store_parallel_text_head_v1 as model
from .store_parallel_text_run_v1 import load_packet, frontier_cases
from .store_catalog_v1 import load_snapshot


class DemoBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.snapshot = load_snapshot(); cls.packet = load_packet()
        bundle = model.engineered_fixture(cls.packet)
        cls.feature_map = model.validate_bundle(bundle, cls.packet, fixture=True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def deps(self, *, listening=False, fail_at=None, integrity=True, contradictory_error=False):
        calls, owned = [], [False]
        runtime = {'manifest': {'headSha256': 'cpu-fixture'}, 'teacherDataRead': False}
        head = model.ParallelTextHead(self.packet['vocabulary'], width=16)
        @contextmanager
        def lock():
            owned[0] = True; calls.append('lock-enter')
            try: yield
            finally: owned[0] = False; calls.append('lock-exit')
        snapshot = self.snapshot
        class Engine:
            def __init__(inner):
                self.assertTrue(owned[0]); calls.append('engine-load')
                inner.adapter = object(); inner.startup = {'sourceTensorsHashed': 927}
                def rank_request(request, *, copy_adapter):
                    calls.append('rank')
                    self.assertNotIn('frontier-', str(request)); self.assertNotIn('expectedTop', str(request))
                    if calls.count('rank') == fail_at:
                        raise TimeoutError('injected complete-query timeout')
                    if contradictory_error and len({r['value'] for r in request['hardRequirements'] if r['attribute'] == 'color'}) > 1:
                        raise ValueError('No single declared variant satisfies every hard color requirement')
                    return {'scores16': list(range(16))}
                inner.session = SimpleNamespace(snapshot=snapshot, products=snapshot['inputs']['products'], rank_request=rank_request)
                inner.binding = SimpleNamespace(assert_current_sources=lambda receipt: True)
            def close(inner):
                self.assertTrue(owned[0]); calls.append('engine-close')
                return {'frozen927HashesUnchanged': integrity, 'copyCheckpointUnchanged': True}
        prepared = {'version': bench.VERSION, 'cases': frontier_cases(), 'runtime': runtime,
                    'scope': 'TRAIN-ALL CPU fixture', 'guardProbes': bench.GUARDS, 'sourceSha256': {'fixture': 'cpu'}}
        def load(head_run):
            self.assertTrue(owned[0]); calls.append('load-runtime')
            return head, self.feature_map, runtime
        def relevance(out, roster):
            self.assertTrue((Path(out) / 'PREDICTIONS_FROZEN.json').exists()); calls.append('read-offline-gold')
            return bench.relevance_after_freeze(out, roster)
        return {'service_listening': lambda: listening, 'preflight': lambda h, m: prepared,
            'native_lock': lock, 'engine_factory': Engine, 'load_runtime': load,
            'source_hashes': lambda: {'fixture': 'cpu'}, 'relevance': relevance}, calls

    def test_default_cpu_only_and_unused_output(self):
        deps, calls = self.deps(listening=True)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'cpu'
            result = bench.run_benchmark(out, '/unused/head', _dependencies=deps)
            self.assertEqual(result['status'], 'cpu-preflight-only'); self.assertEqual(calls, [])
            with self.assertRaises(FileExistsError):
                bench.run_benchmark(out, '/unused/head', _dependencies=deps)

    def test_listening_service_refused_without_loading_or_output(self):
        deps, calls = self.deps(listening=True)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'native'
            with self.assertRaisesRegex(RuntimeError, 'listening'):
                bench.run_benchmark(out, '/unused/head', run=True, _dependencies=deps)
            self.assertFalse(out.exists()); self.assertEqual(calls, [])

    def test_complete_32_fresh_queries_styles_and_variable_guards(self):
        deps, calls = self.deps(contradictory_error=True)
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            out = Path(directory) / 'native'
            result = bench.run_benchmark(out, '/unused/head', run=True, _dependencies=deps)
            self.assertEqual(result['status'], 'completed-train-all-fresh-benchmark-not-promoted', result.get('error'))
            self.assertEqual(result['summary']['plannedRequests'], 32)
            self.assertEqual(result['summary']['completedRequests'], 32)
            self.assertTrue(result['summary']['sixCardsEveryRequest']); self.assertTrue(result['summary']['twentyFieldsEveryRequest'])
            self.assertEqual(calls.count('rank'), 41)
            self.assertEqual(len(result['styleCounterfactuals']), 8)
            self.assertEqual(len(result['guardProbes']), 8)
            self.assertTrue(all(g['passed'] for g in result['guardProbes']))
            self.assertEqual(result['guardProbes'][5]['status'], 'expected-safe-rejection')
            rejected = json.loads((out / 'guard-05.json').read_text())
            self.assertEqual(rejected['expectedSafeError']['type'], 'ValueError')
            self.assertIsNone(rejected['ranked']); self.assertIsNone(rejected['rankThroughJsonMs'])
            guard = json.loads((out / 'guard-07.json').read_text())
            self.assertEqual(len(guard['cards']), 1); self.assertFalse(guard['copyGenerationPerformed'])
            self.assertTrue(result['offlineRelevance']['labelsReadAfterPredictionFreeze'])
            self.assertLess(calls.index('read-offline-gold'), calls.index('engine-close'))
            self.assertLess(calls.index('engine-close'), calls.index('lock-exit'))
            request = json.loads((out / 'request-000.json').read_text())
            self.assertNotIn('caseId', request); self.assertEqual(request['generated']['fieldCount'], 20)

    def test_failed_request_retains_fixed_denominator_and_identity(self):
        deps, calls = self.deps(fail_at=2)
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            out = Path(directory) / 'native'
            result = bench.run_benchmark(out, '/unused/head', run=True, _dependencies=deps)
            self.assertEqual(result['status'], 'failed-benchmark')
            self.assertEqual(result['summary']['failedRequests'], 1)
            self.assertEqual(result['summary']['unattemptedRequests'], 31)
            self.assertFalse(result['summary']['sixCardsEveryRequest'])
            failed = json.loads((out / 'request-000.json').read_text())
            self.assertEqual(failed['archiveCaseId'], frontier_cases()[0]['caseId'])
            self.assertNotIn('read-offline-gold', calls)
            self.assertLess(calls.index('engine-close'), calls.index('lock-exit'))

    def test_warmup_failure_no_fake_completed_requests(self):
        deps, calls = self.deps(fail_at=1)
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            out = Path(directory) / 'native'
            result = bench.run_benchmark(out, '/unused/head', run=True, _dependencies=deps)
            self.assertEqual(result['summary']['unattemptedRequests'], 32)
            self.assertEqual(result['summary']['completedRequests'], 0)
            self.assertTrue((out / 'warmup-failure.json').exists())
            self.assertNotIn('read-offline-gold', calls)

    def test_relevance_requires_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Freeze'):
                bench.relevance_after_freeze(directory, [])

    def test_guard_expected_rejection_is_narrow_and_records_actual_error(self):
        counter = [0]
        session = SimpleNamespace(products=self.snapshot['inputs']['products'], forward_calls=0)
        def reject(request, **kwargs):
            session.forward_calls += 1
            raise ValueError('No single declared variant satisfies every hard color requirement')
        session.rank_request = reject
        engine = SimpleNamespace(session=session, adapter=object())
        result = bench.guard_probe(engine, bench.GUARDS[5])
        self.assertTrue(result['passed']); self.assertEqual(result['status'], 'expected-safe-rejection')
        self.assertEqual(result['nativeQueryForwardCounterDelta'], 1)
        self.assertEqual(result['cards'], []); self.assertFalse(result['copyGenerationPerformed'])
        with self.assertRaises(ValueError):
            bench.guard_probe(engine, bench.GUARDS[0])
        session.rank_request = lambda *a, **k: (_ for _ in ()).throw(ValueError('feature checksum mismatch'))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            bench.guard_probe(engine, bench.GUARDS[5])

    def test_shutdown_integrity_failure_is_not_success(self):
        deps, calls = self.deps(integrity=False)
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            result = bench.run_benchmark(Path(directory) / 'native', '/unused/head', run=True, _dependencies=deps)
        self.assertEqual(result['status'], 'failed-shutdown-integrity')

    def test_failure_cli_exits_nonzero(self):
        with patch('sys.argv', ['bench', '--run', '--head-run', '/tmp/head', '--out', '/tmp/unused']), patch.object(bench, 'run_benchmark', return_value={'status': 'failed-benchmark', 'nativeExecution': True}), patch('builtins.print'):
            with self.assertRaises(SystemExit) as error:
                bench.main()
        self.assertEqual(error.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
