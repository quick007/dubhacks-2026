"""CPU-injected observer accounting/parity/lifetime tests; no native execution."""
from contextlib import contextmanager
from copy import deepcopy
import json
import importlib.util
import sys
import sysconfig
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import torch
from . import store_fast_path_profile_v1 as profile
from . import store_fast_path_profile_run_v1 as run
from . import store_candidate_engine_v1 as candidate
from . import store_parallel_text_head_v1 as text_head
from . import store_parallel_text_data_v1 as data
from .store_catalog_v1 import load_snapshot


class Clock:
    def __init__(self): self.value = 0.
    def __call__(self): return self.value
    def add(self, value): self.value += value


class FakeHead:
    def probabilities(self, role, values): return [0.]


class FakeEngine:
    def __init__(self, output):
        self.output = Path(output); self.fail = False; self.startup = {'fixtureOnly': True}
        self.snapshot = load_snapshot(); self.products = self.snapshot['inputs']['products']
        self.binding = SimpleNamespace(assert_current_sources=lambda receipt: True)
        self.bb = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [1, 2]))
        def encode(rows):
            self.bb.tokenizer.encode(rows[0]); return torch.ones((1, 4096)) / 64, {'fixtureOnly': True}
        self.bb.encode = encode
        self.session = SimpleNamespace(source_validator=self.binding.assert_current_sources,
            store=SimpleNamespace(load=lambda row: (torch.ones((1, 4096)) / 64, {})), head=FakeHead())
        self.session.rank_request = self.rank_request
        self.head = text_head.ParallelTextHead([data.EOS, data.PAD, 'selected', 'Your', 'collection'], width=8)
        self.feature_map = {}
        for product in self.products:
            for color in product['colorVariants']:
                record = data.public_record({'name': product['name'], 'category': product['category'],
                    'itemPrice': product['itemPrice'], 'selectedColor': color['colorName']})
                self.feature_map[record['featureKey']] = torch.ones(4096) / 64
    def _assert_candidate(self): self.binding.assert_current_sources(self.snapshot['receipt'])
    def rank_request(self, request, **kwargs):
        self.session.source_validator(self.snapshot['receipt']); self.bb.tokenizer.encode(request['query'])
        self.bb.encode([request['query']])
        for i in range(16): self.session.store.load(i)
        for role in ('catalog', 'query'): self.session.head.probabilities(role, None)
        self.session.source_validator(self.snapshot['receipt'])
        return {'scores16': list(range(16))}
    def personalize(self, value, request_id, started, emit, cancelled):
        request, policy = candidate.checked_request(value); self._assert_candidate()
        ranked = self.session.rank_request(request); order, guards = candidate.guarded_order(ranked['scores16'], request, policy, self.products)
        cards = candidate.action_bindings(request, order[:6], self.snapshot)
        directory = self.output / 'requests' / request_id; directory.mkdir(parents=True)
        metadata, _ = candidate.public_sections(cards, [], self.products)
        emit({'type': 'rank', 'requestId': request_id, 'sections': metadata, 'provenance': {'fixtureOnly': True}})
        if self.fail: raise RuntimeError('injected CPU fixture failure')
        rows = candidate.parallel_rows(cards)
        generated = text_head.generate(self.head, text_head.inference_rows(rows), self.feature_map)
        events = []
        for kind, limit in [('firstThree', 3), ('complete', 6)]:
            sections, validation = candidate.parallel_payload(cards, rows, generated, limit)
            event = {'type': kind, 'requestId': request_id, 'sections': sections, 'validation': validation,
                **{k: validation[k] for k in ('acceptedFields', 'withheldFields', 'requiredFields')},
                'copySource': {'kind': 'fresh-parallel-head'}, 'provenance': {'fixtureOnly': True}}
            candidate.canonical(event); emit(event); events.append(event)
        receipt = {'status': 'completed', 'ranking': ranked, 'finalOrder': order, 'cards': cards,
            'runtimeRows': rows, 'parallelGeneration': generated}
        path = directory / 'receipt.tmp'; path.write_text(candidate.json.dumps(receipt)); path.replace(directory / 'receipt.json')
    def close(self):
        return {k: True for k in ('frozen927HashesUnchanged', 'copyCheckpointUnchanged', 'v5ServingExportUnchanged', 'v5HeadCheckpointUnchanged')}


class FastProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_nested_clipped_accounting_does_not_add_inclusive_times(self):
        clock = Clock(); trace = profile.PhaseTrace(clock); trace.origin = 0.
        with trace.phase('request', start=0.):
            clock.add(1)
            with trace.phase('encode'):
                clock.add(2)
                with trace.phase('tokenizer'): clock.add(1)
                clock.add(1)
            clock.add(1)
        whole = trace.report(); cutoff = trace.report(3.5)
        self.assertEqual(whole['wallMs'], 6000)
        self.assertEqual(whole['phases']['request']['exclusiveMs'], 2000)
        self.assertEqual(whole['phases']['encode']['exclusiveMs'], 3000)
        self.assertAlmostEqual(cutoff['exclusiveAccountedMs'], cutoff['wallMs'])
        self.assertAlmostEqual(cutoff['unaccountedMs'], 0.)

    def test_fake_engine_same_output_and20_word_slots_with_scope_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = FakeEngine(temporary); original = (engine.bb.encode, candidate.canonical, text_head.generate, Path.write_text)
            a = profile.measure_request(engine, {'query': 'a notebook for quick notes'}, 'a')
            b = profile.measure_request(engine, {'query': 'a notebook for quick notes'}, 'b', instrumented=True)
            self.assertEqual(a['parity']['sha256'], b['parity']['sha256'])
            self.assertEqual(b['parity']['copyFieldCount'], 20); self.assertEqual(b['parity']['cardCount'], 6)
            phases = b['profile']['throughFullSixWire']['phases']
            self.assertEqual(phases['catalogFeatureLoad']['calls'], 16)
            self.assertEqual(phases['freshBackboneEncode']['calls'], 1)
            self.assertEqual(phases['tokenizerEncode']['calls'], 2)
            self.assertEqual(phases['parallelWordSlotDecode']['calls'], 20)
            self.assertEqual(phases['parallelHeadForward']['calls'], 1)
            self.assertAlmostEqual(b['profile']['throughFullSixWire']['unaccountedMs'], 0., places=6)
            self.assertEqual(original, (engine.bb.encode, candidate.canonical, text_head.generate, Path.write_text))

    def test_delegated_tokenizer_writes_restore_underlying_method_on_success_and_error(self):
        class UnderlyingTokenizer:
            def encode(self, *args, **kwargs): return [1, 2]
        class DelegatingTokenizer:
            # Reproduces installed MLX TokenizerWrapper's write delegation.
            def __init__(self, tokenizer): object.__setattr__(self, '_tokenizer', tokenizer)
            def __getattr__(self, name): return getattr(self._tokenizer, name)
            def __setattr__(self, name, value): setattr(self._tokenizer, name, value)
        with tempfile.TemporaryDirectory() as temporary:
            engine = FakeEngine(temporary); underlying = UnderlyingTokenizer()
            engine.bb.tokenizer = DelegatingTokenizer(underlying)
            original = underlying.encode
            a = profile.measure_request(engine, {'query': 'a bottle for drinking water'}, 'plain')
            b = profile.measure_request(engine, {'query': 'a bottle for drinking water'}, 'delegated', instrumented=True)
            self.assertEqual(a['parity']['sha256'], b['parity']['sha256'])
            self.assertEqual(b['profile']['throughFullSixWire']['phases']['tokenizerEncode']['calls'], 2)
            self.assertEqual(underlying.encode, original)
            self.assertNotIn('encode', vars(engine.bb.tokenizer)); self.assertNotIn('encode', vars(underlying))
            engine.fail = True
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                profile.measure_request(engine, {'query': 'a bottle for drinking water'}, 'failed-delegated', instrumented=True)
            self.assertEqual(underlying.encode, original)
            self.assertNotIn('encode', vars(engine.bb.tokenizer)); self.assertNotIn('encode', vars(underlying))

    def test_actual_installed_wrapper_and_local_hf_tokenizer_cpu_only(self):
        from transformers import AutoTokenizer
        # Read the pure-Python installed module directly: importing mlx_lm's
        # package initializer is unnecessary for this CPU delegation test.
        file = Path(sysconfig.get_paths()['purelib']) / 'mlx_lm/tokenizer_utils.py'
        spec = importlib.util.spec_from_file_location('tokenizer_utils_cpu_profile_fixture', file)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        packet = json.loads(Path('data/ranker/pretrained/actual-store-function-v3-attempt-1/packet.json').read_text())
        hf = AutoTokenizer.from_pretrained(packet['model']['modelPath'], local_files_only=True)
        wrapper = module.TokenizerWrapper(hf)
        query = 'a bottle for drinking water'; ids = wrapper.encode(query, add_special_tokens=False)
        with tempfile.TemporaryDirectory() as temporary:
            engine = FakeEngine(temporary); engine.bb.tokenizer = wrapper
            original = hf.encode
            a = profile.measure_request(engine, {'query': query}, 'local-hf-a')
            b = profile.measure_request(engine, {'query': query}, 'local-hf-b', instrumented=True)
            self.assertEqual(a['parity']['sha256'], b['parity']['sha256'])
            self.assertEqual(wrapper.encode(query, add_special_tokens=False), ids)
            self.assertEqual(hf.encode, original); self.assertNotIn('encode', vars(wrapper))
            engine.fail = True
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                profile.measure_request(engine, {'query': query}, 'local-hf-error', instrumented=True)
            self.assertEqual(hf.encode, original); self.assertNotIn('encode', vars(wrapper))
        self.assertFalse(any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules))

    def test_exception_restores_all_hooks_and_captures_partial_phase_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = FakeEngine(temporary); engine.fail = True
            original = (engine.bb.encode, candidate.canonical, text_head.generate, Path.write_text)
            with self.assertRaisesRegex(RuntimeError, 'fixture failure') as caught:
                profile.measure_request(engine, {'query': 'light'}, 'fail', instrumented=True)
            self.assertGreater(caught.exception.profile_failure_timing['failedAtMs'], 0)
            self.assertIn('freshBackboneEncode', caught.exception.profile_partial_trace['phases'])
            self.assertEqual(original, (engine.bb.encode, candidate.canonical, text_head.generate, Path.write_text))

    def test_known_browser_query_roster_only_counterbalanced_per_query(self):
        cases = run.public_cases(10)
        self.assertEqual(len(cases), 10); self.assertEqual(len({r['request']['query'] for r in cases}), 4)
        self.assertEqual(cases[0]['armOrder'], list(reversed(cases[4]['armOrder'])))
        self.assertEqual(set(cases[0]['request']), {'query'})
        with self.assertRaises(ValueError): run.public_cases(60)

    def dependencies(self, *, fail=False, listening=False):
        calls = []; held = [False]
        @contextmanager
        def lock():
            held[0] = True; calls.append('lock-enter')
            try: yield
            finally: held[0] = False; calls.append('lock-exit')
        class Engine:
            startup = {'fixtureOnly': True}
            def close(inner):
                self.assertTrue(held[0]); calls.append('close')
                return FakeEngine.close(inner)
        cases = [{'caseId': 'one', 'request': {'query': 'known public fixture'}, 'armOrder': ['B-profile', 'A-original'], 'group': 'public'}]
        def measure(engine, request, rid, instrumented):
            self.assertTrue(held[0]); calls.append('B' if instrumented else 'A')
            if fail and len(calls) > 4: raise RuntimeError('injected driver failure')
            value = {'status': 'completed', 'arm': 'B-profile' if instrumented else 'A-original',
                'parity': {'sha256': 'same'}, 'timing': {'fullSixWireReadyMs': 110 if instrumented else 100}}
            if instrumented: value['profile'] = {'throughFullSixWire': {'phases': {'encode': {'exclusiveMs': 70}}}}
            return value
        return {'prepare': lambda *args: {'cases': cases, 'sourceSha256': {}}, 'lock': lock,
            'factory': lambda: Engine(), 'measure': measure, 'assert_external_pins': lambda preflight: None,
            'service_listening': lambda: listening}, calls

    def test_driver_owns_lock_through_close_and_summary_is_not_ui_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            deps, calls = self.dependencies()
            result = run.run(Path(temporary) / 'out', None, None, None, None, execute=True, _dependencies=deps)
            self.assertEqual(result['status'], 'completed-profile-not-optimized')
            self.assertEqual(calls[-2:], ['close', 'lock-exit'])
            self.assertEqual(result['summary']['exactParityPairs'], 1)
            self.assertEqual(result['summary']['observerOverhead']['medianMs'], 10)
            self.assertNotIn('incrementalMinusGeneric', result['summary'])
            self.assertIn('never generic UI latency', result['summary']['comparisonScope'])

    def test_constructor_failure_releases_owner_lock_without_any_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            deps, calls = self.dependencies()
            def fail_factory():
                calls.append('factory-failed')
                raise RuntimeError('injected initialization failure; real ImageCandidate owns partial-native cleanup')
            deps['factory'] = fail_factory
            result = run.run(Path(temporary) / 'fail', None, None, None, None, execute=True, _dependencies=deps)
            self.assertEqual(result['status'], 'failed-profile')
            self.assertEqual(calls, ['lock-enter', 'factory-failed', 'lock-exit'])
            self.assertEqual(result['summary']['unattemptedArms'], 2)

    def test_cpu_default_does_not_load_and_live_service_refuses(self):
        with tempfile.TemporaryDirectory() as temporary:
            deps, calls = self.dependencies(listening=True)
            result = run.run(Path(temporary) / 'cpu', None, None, None, None, _dependencies=deps)
            self.assertFalse(result['nativeExecution']); self.assertEqual(calls, [])
            with self.assertRaisesRegex(RuntimeError, 'listening'):
                run.run(Path(temporary) / 'blocked', None, None, None, None, execute=True, _dependencies=deps)
            self.assertEqual(calls, [])


if __name__ == '__main__': unittest.main()
