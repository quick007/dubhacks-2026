"""CPU-only full comparison driver tests with fake resident engine/cache."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .store_fast_decode_run_v1 import BASELINE, CANDIDATE, GIB, configure_experimental_budget, require_headroom, run, validate_budget
from .store_catalog_v1 import load_snapshot
from .store_copy_data_v2 import CAPS, FIELDS, KIND
from .store_demo_engine import checked_request
from .store_rank_session import action_bindings


class FastDriverTests(unittest.TestCase):
    def dependencies(self, root, listening=False):
        calls = []; owned = [False]; clock_value = [100.]; snapshot = load_snapshot()
        def clock():
            clock_value[0] += .001
            return clock_value[0]
        @contextmanager
        def lock():
            self.assertFalse(owned[0]); owned[0] = True; calls.append('lock-enter')
            try:
                yield
            finally:
                owned[0] = False; calls.append('lock-exit')
        class Cache:
            def __init__(inner, engine):
                inner.hits = 0; inner.misses = 0
            def prepare_root(inner, bb, common, **kwargs):
                self.assertTrue(owned[0]); hit = inner.misses > 0
                if hit:
                    inner.hits += 1
                else:
                    inner.misses += 1
                calls.append('cache-hit' if hit else 'cache-miss')
                return {}, {'rootPrepareMs': 1. if hit else 500., 'commonRootCacheHit': hit}
            def assert_immutable(inner):
                self.assertTrue(owned[0]); return True
            def receipt(inner):
                return {'hits': inner.hits, 'misses': inner.misses}
            def clear(inner):
                calls.append('cache-clear')
        class Engine:
            def __init__(inner):
                self.assertTrue(owned[0]); calls.append('load-one-engine')
                inner.output = root / 'engine'; inner.width = 10
                class Mx:
                    def set_memory_limit(self, value):
                        calls.append(('set-memory-limit', value)); return 7 * GIB
                inner.bb = SimpleNamespace(memory_budget=7 * GIB, mx=Mx(), check_memory=lambda: None)
                inner.bb.runtime_memory = lambda: {'mlxActiveBytes': 5 * GIB, 'mlxPeakBytes': 6 * GIB,
                    'processBudgetBytes': inner.bb.memory_budget}
                inner.startup = {'width': 10, 'sourceTensorsHashed': 927}
                inner.session = SimpleNamespace(snapshot=snapshot)
            def personalize(inner, value, request_id, started, emit, cancelled):
                self.assertTrue(owned[0]); calls.append(('request', value['query'], inner.width))
                if inner.width == 20:
                    from .branch_local_copy import prepare_root
                    _, preparation = prepare_root(inner.bb, {'role': 'user', 'content': 'same neutral root'})
                else:
                    preparation = {'rootPrepareMs': 500., 'commonRootCacheHit': False}
                request, _ = checked_request(value)
                indices = list(range(6))
                if 'light' in value['query']:
                    indices = [7, 1, 2, 3, 4, 5]
                elif 'notebook' in value['query']:
                    indices = [2, 0, 1, 3, 4, 5]
                cards = action_bindings(request, indices, snapshot)
                sections = []
                for start in (0, 3):
                    section = {'heading': 'Native generated heading', 'cards': []}
                    for card in cards[start:start + 3]:
                        section['cards'].append({'originalName': card['facts']['name'],
                            'title': 'Native generated title', 'bodyCopy': 'Native generated body', 'badge': 'Native badge'})
                    sections.append(section)
                ready = 3000. if inner.width == 20 else 5000.
                events = [{'type': 'rank', 'timing': {'readyMs': 200.}},
                    {'type': 'firstThree', 'timing': {'readyMs': ready - 500.}, 'sections': sections[:1],
                        'acceptedFields': 10, 'requiredFields': 10},
                    {'type': 'complete', 'timing': {'readyMs': ready}, 'sections': sections,
                        'acceptedFields': 20, 'requiredFields': 20}]
                rows = [{'fields': [field], 'maxTokens': CAPS[field[0]], 'kind': KIND[field[0]],
                    'tokenIds': [inner.width + i], 'rawSampledTokenIds': [inner.width + i, 99],
                    'stopTokenId': 99, 'arSteps': 2, 'finishReason': 'stop', 'text': field}
                    for i, field in enumerate(FIELDS)]
                directory = inner.output / 'requests' / request_id; directory.mkdir(parents=True)
                (directory / 'receipt.json').write_text(json.dumps({'status': 'completed', 'input': value,
                    'cards': cards, 'finalOrder': indices, 'rawRows': rows, 'events': events,
                    'preparation': preparation, 'decodeTiming': {'width': inner.width},
                    'memory': {'processBudgetBytes': inner.bb.memory_budget}}))
                for event in events:
                    emit(event)
            def close(inner):
                self.assertTrue(owned[0]); calls.append('engine-close')
                return {'frozen927HashesUnchanged': True, 'copyCheckpointUnchanged': True}
        return {'service_listening': lambda: listening, 'preflight': lambda manifest: {'version': 'fake', 'nativeExecution': False},
            'lock': lock, 'factory': Engine, 'cache_factory': Cache, 'clock': clock,
            'reset': lambda engine: calls.append('reset'),
            'memory_snapshot': lambda: {'totalBytes': 24 * GIB, 'availableBytes': 15 * GIB,
                'percentUsed': 37.5, 'swapUsedBytes': 0, 'swapFreeBytes': 8 * GIB},
            'controls': lambda engine, out: [{'width': w, 'status': 'exact-controls-passed'} for w in (10, 20)]}, calls

    def test_cpu_default_never_constructs_engine_or_takes_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root, listening=True)
            report = run(root / 'out', _dependencies=deps)
            self.assertEqual(report['status'], 'cpu-preflight-only')
            self.assertEqual(calls, [])

    def test_listening_service_refused_without_output_or_automatic_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root, listening=True)
            with self.assertRaisesRegex(RuntimeError, 'still listening'):
                run(root / 'out', native=True, _dependencies=deps)
            self.assertFalse((root / 'out').exists()); self.assertEqual(calls, [])

    def test_one_engine_all_six_cards_twenty_fields_crosswidth_difference_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root)
            report = run(root / 'out', native=True, _dependencies=deps)
            self.assertEqual(report['status'], 'completed-fast-decode-comparison')
            self.assertEqual(calls.count('load-one-engine'), 1)
            self.assertEqual([row['mode'] for row in report['records'] if not row['excludedWarmupOrControl']],
                [BASELINE, CANDIDATE, CANDIDATE, BASELINE, BASELINE, CANDIDATE])
            self.assertEqual(len(report['cacheParity']), 2)
            self.assertTrue(all(row['exact'] for row in report['cacheParity']))
            self.assertFalse(report['crossWidthTokenEqualityRequired'])
            self.assertTrue(all(s['cardCounts'] == [6, 6, 6] and s['fieldCounts'] == [20, 20, 20]
                for s in report['summary'].values()))
            self.assertEqual(report['cacheReceipt'], {'hits': 5, 'misses': 1})
            self.assertEqual(calls[-3:], ['cache-clear', 'engine-close', 'lock-exit'])

    def test_previous_results_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root)
            with self.assertRaises(FileExistsError):
                run(root, native=True, _dependencies=deps)
            self.assertEqual(calls, [])

    def test_budget_only7_or8_and_no_implicit_change(self):
        self.assertEqual(validate_budget(7), 7 * GIB)
        self.assertEqual(validate_budget(8), 8 * GIB)
        for value in (True, 8., 6, 9, '8', None):
            with self.assertRaises(ValueError):
                validate_budget(value)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root)
            result = run(root / 'default', native=True, _dependencies=deps)
            self.assertEqual(result['memoryGuardGiB'], 7)
            self.assertFalse(result['memoryGuardRaised'])
            self.assertFalse(any(isinstance(call, tuple) and call[0] == 'set-memory-limit' for call in calls))

    def test_explicit8_applied_after_default_initialization_and_recorded_for_both_arms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root)
            result = run(root / 'explicit8', native=True, memory_gib=8, _dependencies=deps)
            self.assertEqual(result['status'], 'completed-fast-decode-comparison')
            self.assertTrue(result['memoryGuardRaised']); self.assertFalse(result['memoryGuardRaisedAutomatically'])
            self.assertEqual(result['memoryConfiguration']['before']['processBudgetBytes'], 7 * GIB)
            self.assertEqual(result['memoryConfiguration']['after']['processBudgetBytes'], 8 * GIB)
            self.assertLess(calls.index('load-one-engine'), calls.index(('set-memory-limit', 8 * GIB)))
            self.assertTrue(all(row['memory']['processBudgetBytes'] == 8 * GIB for row in result['records']))
            self.assertEqual(result['startup']['experimentalMemoryConfiguration']['requestedMemoryGiB'], 8)

    def test_insufficient8_headroom_refused_before_model_load_and_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root)
            deps['memory_snapshot'] = lambda: {'totalBytes': 24 * GIB, 'availableBytes': 9 * GIB}
            result = run(root / 'too-tight', native=True, memory_gib=8, _dependencies=deps)
            self.assertEqual(result['status'], 'failed-fast-decode-comparison')
            self.assertEqual(result['error']['type'], 'MemoryError')
            self.assertEqual(result['systemMemoryBeforeLoad']['availableBytes'], 9 * GIB)
            self.assertNotIn('load-one-engine', calls)

    def test_afterload_headroom_refused_without_allocator_mutation(self):
        engine = SimpleNamespace(bb=SimpleNamespace(memory_budget=7 * GIB,
            runtime_memory=lambda: {'mlxActiveBytes': 5 * GIB, 'mlxPeakBytes': 6 * GIB, 'processBudgetBytes': 7 * GIB}), startup={})
        with self.assertRaises(MemoryError):
            configure_experimental_budget(engine, 8, {'totalBytes': 24 * GIB, 'availableBytes': 4 * GIB})
        self.assertEqual(engine.bb.memory_budget, 7 * GIB)
        self.assertNotIn('experimentalMemoryConfiguration', engine.startup)

    def test_explicit8_preflight_does_not_apply_allocator_or_load_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); deps, calls = self.dependencies(root)
            result = run(root / 'prepare8', memory_gib=8, _dependencies=deps)
            self.assertEqual(result['preflight']['configuredBudgetBytes'], 8 * GIB)
            self.assertTrue(result['preflight']['memoryIncreaseExplicit'])
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
