"""Fake-engine CPU tests of the isolated driver; no native execution."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import store_parallel_text_run_v1 as run
from . import store_parallel_text_head_v1 as head
from . import store_parallel_text_data_v1 as data
from .store_catalog_v1 import load_snapshot
from .store_rank_session import action_bindings


class ParallelTextRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = load_snapshot()
        cls.packet = run.load_packet()
        with head.bounded_cpu_threads():
            cls.bundle = head.engineered_fixture(cls.packet)

    def dependencies(self, *, listening=False, fail_query=False, fail_query_at=None, integrity=True):
        calls = []; owned = [False]
        @contextmanager
        def lock():
            self.assertFalse(owned[0]); owned[0] = True; calls.append('lock-enter')
            try:
                yield
            finally:
                owned[0] = False; calls.append('lock-exit')
        snapshot = self.snapshot
        class Engine:
            def __init__(inner):
                self.assertTrue(owned[0]); calls.append('engine-load')
                inner.startup = {'sourceTensorsHashed': 927}
                inner.adapter = object()
                def rank_request(value, *, copy_adapter):
                    self.assertIs(copy_adapter, inner.adapter)
                    self.assertEqual(set(value), {'query', 'preferences', 'itemBudget', 'hardRequirements'})
                    self.assertNotIn('frontier-', str(value))
                    calls.append('rank')
                    if fail_query or calls.count('rank') == fail_query_at:
                        raise TimeoutError('injected complete-query failure')
                    return {'scores16': list(range(16)), 'source': 'engineered CPU test only'}
                inner.session = SimpleNamespace(snapshot=snapshot, products=snapshot['inputs']['products'], rank_request=rank_request)
                inner.binding = SimpleNamespace(assert_current_sources=lambda receipt: calls.append('source-check'))
            def close(inner):
                self.assertTrue(owned[0]); calls.append('engine-close')
                return {'frozen927HashesUnchanged': integrity, 'copyCheckpointUnchanged': True}
        def preflight(manifest):
            return {'version': run.VERSION, 'nativeExecution': False, 'cases': run.frontier_cases(),
                'featureRows': [], 'sourceSha256': {'test': 'fixture'}, 'packetManifestSha256': 'fixture-only'}
        def extractor(*args):
            self.assertTrue(owned[0]); calls.append('extract')
            return self.bundle
        def fitter(packet, bundle, **config):
            self.assertEqual(config, {'steps': 160, 'learning_rate': .004, 'width': 128})
            calls.append('fit')
            return head.fit(packet, bundle, steps=3, width=16, fixture=True)
        return {'service_listening': lambda: listening, 'preflight': preflight,
                'native_lock': lock, 'engine_factory': Engine, 'packet': lambda: self.packet,
                'extractor': extractor, 'fitter': fitter,
                'feature_validator': lambda b, p: head.validate_bundle(b, p, fixture=True),
                'source_hashes': lambda: {'test': 'fixture'}}, calls

    def test_public_frontier_fixed_and_no_gold_file_read(self):
        original = Path.read_text
        def guarded(path, *args, **kwargs):
            if path.name == 'offline-relevance-labels.json':
                raise AssertionError('offline gold read')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', guarded):
            cases = run.frontier_cases()
        self.assertEqual(len(cases), 32)
        self.assertTrue(all(set(c['request']) == {'query'} for c in cases))

    def test_cpu_default_does_not_check_service_or_load_engine(self):
        deps, calls = self.dependencies(listening=True)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'cpu'
            result = run.run_candidate(out, _dependencies=deps)
            self.assertEqual(result['status'], 'cpu-preflight-only')
            self.assertFalse(result['nativeExecution'])
            self.assertEqual(calls, [])
            with self.assertRaises(FileExistsError):
                run.run_candidate(out, _dependencies=deps)

    def test_listening_service_refused_without_engine_or_output(self):
        deps, calls = self.dependencies(listening=True)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'candidate'
            with self.assertRaisesRegex(RuntimeError, 'listening'):
                run.run_candidate(out, run=True, _dependencies=deps)
            self.assertFalse(out.exists())
            self.assertEqual(calls, [])

    def test_runtime_rows_exact_source_binding_and_20_fields(self):
        request = {'query': 'browse', 'preferences': [], 'itemBudget': None, 'hardRequirements': []}
        cards = action_bindings(request, list(range(6)), self.snapshot)
        original = data.teacher
        try:
            data.teacher = lambda *a, **k: (_ for _ in ()).throw(AssertionError('teacher entered runtime'))
            rows = run.runtime_rows(cards, 'plain')
        finally:
            data.teacher = original
        self.assertEqual(len(rows), 20)
        self.assertEqual([r['field'] for r in rows[:4]], ['S1', 'T1', 'B1', 'C1'])
        feature_map = head.validate_bundle(self.bundle, self.packet, fixture=True)
        features, _ = head.serving_tensors(head.inference_rows(rows), feature_map)
        self.assertEqual(features.shape, (20, 4096))
        with self.assertRaises(ValueError):
            run.runtime_rows(cards[:2], 'plain')

    def test_endpoint_withholds_rejected_output_without_repair(self):
        request = {'query': 'browse', 'preferences': [], 'itemBudget': None, 'hardRequirements': []}
        cards = action_bindings(request, list(range(6)), self.snapshot)
        rows = run.runtime_rows(cards, 'plain')
        generated = {'fields': [{'text': 'Unsafe text kept raw', 'accepted': False, 'rejections': ['wrong-color']} for _ in rows]}
        payload = run.endpoint(cards, rows, generated, card_limit=6)
        self.assertEqual(payload['validation']['requiredFields'], 20)
        self.assertFalse(payload['validation']['titleBodyPerCardProxy'])
        for source, actual in zip(cards, [c for s in payload['sections'] for c in s['cards']]):
            self.assertIsNone(actual['title']); self.assertIsNone(actual['bodyCopy'])
            self.assertEqual(actual['itemPrice'], source['facts']['itemPrice'])
            self.assertEqual(actual['originalName'], source['facts']['name'])
            self.assertEqual(actual['variantId'], source['action']['variantId'])
        self.assertEqual(generated['fields'][0]['text'], 'Unsafe text kept raw')

    def test_complete_fake_cpu_driver_32_requests_integrity_reload_and_timings(self):
        deps, calls = self.dependencies()
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            out = Path(directory) / 'candidate'
            result = run.run_candidate(out, run=True, _dependencies=deps)
            self.assertEqual(result['status'], 'completed-candidate-not-promoted', result.get('error'))
            self.assertEqual(len(result['requests']), 32)
            self.assertEqual(len(result['styleCounterfactuals']), 8)
            self.assertTrue(result['reloadExactSlotParity'])
            self.assertTrue(result['summary']['sixCardsEveryRequest'])
            self.assertTrue(result['summary']['twentyFieldsEveryRequest'])
            first = json.loads((out / 'request-000.json').read_text())
            self.assertNotIn('caseId', first)
            self.assertEqual(first['input'], run.frontier_cases()[0]['request'])
            self.assertEqual(first['generated']['autoregressiveSteps'], 0)
            self.assertGreaterEqual(first['timing']['attemptFullSixMs'], first['timing']['attemptFirstThreeMs'])
            self.assertEqual(json.loads(first['serialized']['fullSixJson']), first['fullSix'])
            self.assertEqual(calls.count('rank'), 33)
            self.assertLess(calls.index('engine-close'), calls.index('lock-exit'))

    def test_failure_closes_engine_inside_lock_and_retains_receipt(self):
        deps, calls = self.dependencies(fail_query=True)
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            out = Path(directory) / 'candidate'
            result = run.run_candidate(out, run=True, _dependencies=deps)
            self.assertEqual(result['status'], 'failed-candidate')
            self.assertEqual(result['error']['type'], 'TimeoutError')
            self.assertEqual(result['summary']['plannedRequests'], 32)
            self.assertEqual(result['summary']['completedRequests'], 0)
            self.assertEqual(result['summary']['unattemptedRequests'], 32)
            self.assertFalse(result['summary']['sixCardsEveryRequest'])
            self.assertTrue((out / 'warmup-failure.json').exists())
            self.assertTrue((out / 'fit.json').exists())
            self.assertTrue((out / 'head.pt').exists())
            self.assertTrue((out / 'results.json').exists())
            self.assertIn('shutdownIntegrity', result)
            self.assertLess(calls.index('engine-close'), calls.index('lock-exit'))

    def test_timed_failure_fixed_denominator_and_failed_case_receipt(self):
        deps, calls = self.dependencies(fail_query_at=2)
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            out = Path(directory) / 'candidate'
            result = run.run_candidate(out, run=True, _dependencies=deps)
            self.assertEqual(result['status'], 'failed-candidate')
            self.assertEqual(result['summary']['plannedRequests'], 32)
            self.assertEqual(result['summary']['failedRequests'], 1)
            self.assertEqual(result['summary']['unattemptedRequests'], 31)
            self.assertEqual(result['requests'][0]['caseId'], run.frontier_cases()[0]['caseId'])
            self.assertEqual(result['requests'][0]['status'], 'failed-attempt')
            failed = json.loads((out / 'request-000.json').read_text())
            self.assertEqual(failed['archiveCaseId'], run.frontier_cases()[0]['caseId'])
            self.assertEqual(failed['error']['type'], 'TimeoutError')

    def test_failed_shutdown_cannot_be_completed_status(self):
        deps, _ = self.dependencies(integrity=False)
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            result = run.run_candidate(Path(directory) / 'candidate', run=True, _dependencies=deps)
        self.assertEqual(result['status'], 'failed-shutdown-integrity')

    def test_cli_failure_exits_nonzero(self):
        with patch('sys.argv', ['parallel', '--run', '--out', '/tmp/unused']), patch.object(run, 'run_candidate', return_value={'status': 'failed-candidate', 'nativeExecution': True}), patch('builtins.print'):
            with self.assertRaises(SystemExit) as error:
                run.main()
        self.assertEqual(error.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
