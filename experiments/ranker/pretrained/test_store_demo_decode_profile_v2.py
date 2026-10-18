"""Meaningful CPU phase/accounting/parity tests, with injected fake native ops."""
from collections import deque
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import store_demo_engine
from .store_demo_decode_profile_v1 import decode_rows_profiled as outer_profile
from .store_demo_decode_profile_v2 import (PhaseCollector, assert_installed_sources,
    assert_reference_file, decode_rows_profiled, profiled_classes, token_parity)
from .test_store_demo_decode_profile_v1 import ProfileTests


class FakeClock:
    def __init__(self):
        self.value = 0.
    def __call__(self):
        return self.value
    def advance(self, value):
        self.value += value


class NativePhaseTests(unittest.TestCase):
    def fixture(self):
        clock = FakeClock(); calls = []
        class Cache:
            state = ['fake-cache-state']
        class Mx:
            def synchronize(self, stream):
                calls.append(('sync', stream)); clock.advance(.0001)
            def eval(self, *values):
                calls.append(('eval', values)); clock.advance(.0002)
            def array(self, value):
                return list(value)
        mx = Mx()
        class Prompt:
            def __init__(self, model, uids, caches, tokens=None, prefill_step_size=256,
                    samplers=None, fallback_sampler=None, logits_processors=None, stop_sequences=None, max_tokens=None):
                self.model = model; self.uids = list(uids); self.prompt_cache = [Cache()] if uids else []
                self.tokens = list(tokens or [[] for _ in uids]); self.prefill_step_size = prefill_step_size
                self.samplers = list(samplers or [None] * len(uids)); self.fallback_sampler = fallback_sampler
                self.logits_processors = list(logits_processors or [[] for _ in uids])
                self.stop_sequences = list(stop_sequences or [None] * len(uids))
                self.max_tokens = list(max_tokens or [20] * len(uids))
                calls.append(('merge', tuple(uids))); clock.advance(.001)
            @classmethod
            def empty(cls, model, sampler, prefill_step_size=256):
                return cls(model, [], [], fallback_sampler=sampler, prefill_step_size=prefill_step_size)
            def extend(self, batch):
                calls.append(('prompt-extend',)); clock.advance(.0015)
                self.uids.extend(batch.uids); self.prompt_cache = [Cache()] if self.uids else []
                self.tokens.extend(batch.tokens); self.samplers.extend(batch.samplers)
                self.logits_processors.extend(batch.logits_processors)
                self.stop_sequences.extend(batch.stop_sequences); self.max_tokens.extend(batch.max_tokens)
            def filter(self, keep):
                calls.append(('prompt-filter', tuple(keep))); clock.advance(.0008)
                for key in ('uids', 'tokens', 'samplers', 'logits_processors', 'stop_sequences', 'max_tokens'):
                    setattr(self, key, [getattr(self, key)[i] for i in keep])
                self.prompt_cache = [Cache()] if keep else []
            def split(self, indices):
                calls.append(('prompt-copy',)); clock.advance(.002)
                result = self.__class__.__new__(self.__class__)
                result.__dict__ = deepcopy(self.__dict__)
                self.filter([i for i in range(len(self.uids)) if i not in indices])
                result.filter(indices)
                return result
            def prompt(self, tokens):
                if not tokens:
                    return
                calls.append(('prefill-model', [len(row) for row in tokens])); clock.advance(.003)
        class Generation:
            def __init__(self, model, uids, inputs, prompt_cache, tokens, samplers,
                    fallback_sampler, logits_processors, stop_sequences, max_tokens):
                self.model = model; self.uids = list(uids); self.prompt_cache = prompt_cache
                self.tokens = tokens; self.samplers = samplers; self.fallback_sampler = fallback_sampler
                self.logits_processors = logits_processors; self.stop_sequences = stop_sequences; self.max_tokens = max_tokens
                self._next_tokens = inputs; self._next_logprobs = []
                if self.uids:
                    self._step()
            @classmethod
            def empty(cls, model, sampler):
                return cls(model, [], [], [], [], [], sampler, [], [], [])
            def _step(self):
                calls.append(('AR-model', len(self.uids))); clock.advance(.005)
                return [7] * len(self.uids), []
            def next(self):
                result = self._step()
                self.extract_cache(0); self.filter(list(range(len(self.uids))))
                calls.append(('stop-handling',)); clock.advance(.0006)
                return result
            def extend(self, batch):
                calls.append(('generation-extend',)); clock.advance(.001)
                self.uids.extend(batch.uids); self.prompt_cache = [Cache()] if self.uids else []
            def filter(self, keep):
                calls.append(('generation-filter',)); clock.advance(.001)
                self.uids = [self.uids[i] for i in keep]
                self.prompt_cache = [Cache()] if keep else []; self._next_tokens = [7] * len(keep)
            def extract_cache(self, index):
                calls.append(('extract-host-only',)); clock.advance(.0004)
                return [Cache()]
        class Batch:
            def __init__(self, model, sampler=None, prefill_step_size=256, **kwargs):
                self.model = model; self.sampler = sampler; self.prefill_step_size = prefill_step_size
                self._prompt_batch = Prompt.empty(model, sampler)
                self._generation_batch = Generation.empty(model, sampler)
                self._unprocessed_sequences = deque(); self._currently_processing = []
        native = SimpleNamespace(PromptProcessingBatch=Prompt, GenerationBatch=Generation, BatchGenerator=Batch)
        collector = PhaseCollector(lambda: mx.synchronize('test-stream'), clock=clock)
        classes = profiled_classes(native, collector, mx, 'test-stream')
        return collector, classes, clock, calls

    def test_nested_exclusive_times_do_not_double_count_decode(self):
        clock = FakeClock()
        collector = PhaseCollector(lambda: clock.advance(.0001), clock=clock)
        def transition():
            clock.advance(.002)
            collector.measure('decode', lambda: clock.advance(.005))
            clock.advance(.003)
        collector.measure('transition', transition)
        result = collector.report()
        self.assertAlmostEqual(result['phases']['decode']['exclusiveMs'], 5.1)
        self.assertAlmostEqual(result['phases']['transition']['exclusiveMs'], 5.1)
        self.assertAlmostEqual(result['accountedExclusivePhaseMs'] + result['preBoundarySyncMs'], clock() * 1000)
        self.assertFalse(result['inclusivePhaseTimesAreAdditive'])

    def test_actual_prefill_and_initial_then_next_ar_steps_are_distinct(self):
        collector, (_, Prompt, Generation), clock, calls = self.fixture()
        prompt = Prompt(model=object(), uids=[0, 1], caches=[[], []])
        prompt.prompt([[1, 2, 3], [4, 5]])
        generated = prompt.generate([[8], [9]])
        self.assertIsInstance(generated, Generation)
        generated.next()
        report = collector.report()['phases']
        self.assertEqual(report['suffixPrefill']['calls'], 1)
        self.assertEqual(report['autoregressiveStep']['calls'], 2)
        self.assertGreaterEqual(report['suffixPrefill']['exclusiveMs'], 3.)
        self.assertGreaterEqual(report['autoregressiveStep']['exclusiveMs'], 10.)
        self.assertTrue(any(event['phase'] == 'suffixPrefill' and event['usefulInputTokens'] == 5
            and event['paddedInputTokens'] == 6 for event in collector.events))
        self.assertEqual(sum(call[0] == 'AR-model' for call in calls), 2)
        self.assertEqual(report['finishedRowCacheExtractionHost']['calls'], 1)
        self.assertTrue(any(event['phase'] == 'finishedRowCacheExtractionHost' and not event['synchronized'] for event in collector.events))

    def test_cache_copy_split_filter_materialization_separate_from_model(self):
        collector, (_, Prompt, _), clock, calls = self.fixture()
        prompt = Prompt(model=object(), uids=[0, 1], caches=[[], []])
        split = prompt.split([0, 1])
        report = collector.report()['phases']
        self.assertEqual(split.uids, [0, 1]); self.assertEqual(prompt.uids, [])
        self.assertEqual(report['promptCacheSplitAndCopy']['calls'], 1)
        self.assertEqual(report['promptCacheFilter']['calls'], 2)
        self.assertTrue(any(e['phase'] == 'promptCacheSplitAndCopy' and e['allRowsTransition'] for e in collector.events))
        self.assertFalse(any(call[0] in ('AR-model', 'prefill-model') for call in calls))
        self.assertTrue(any(call[0] == 'eval' for call in calls))

    def test_admission_constructor_retains_native_pending_order_and_caps(self):
        collector, (Batch, Prompt, _), clock, calls = self.fixture()
        generator = Batch(object(), sampler=object(), completion_batch_size=10, prefill_batch_size=10)
        for uid in (8, 3):
            generator._unprocessed_sequences.append((uid, [[11, 12], [13]], 24 + uid,
                [], [2], object(), [], object()))
        batch = generator._make_batch(2)
        self.assertIsInstance(batch, Prompt)
        self.assertEqual(batch.uids, [8, 3]); self.assertEqual(batch.max_tokens, [32, 27])
        self.assertEqual(generator._currently_processing, [[[[11, 12], [13]], 0, 3], [[[11, 12], [13]], 0, 3]])
        self.assertFalse(any(call[0] == 'AR-model' for call in calls))

    def test_phase_failure_keeps_event_and_balances_stack(self):
        clock = FakeClock(); collector = PhaseCollector(lambda: clock.advance(.0001), clock=clock)
        def fail():
            clock.advance(.003); raise ValueError('injected failure')
        with self.assertRaises(ValueError):
            collector.measure('suffixPrefill', fail)
        self.assertEqual(collector.stack, [])
        self.assertTrue(collector.events[0]['failed'])
        self.assertGreaterEqual(collector.events[0]['exclusiveMs'], 3.)


class IntegrationContractTests(unittest.TestCase):
    def test_physical_source_and_installed_library_pins_cpu_only(self):
        self.assertEqual(len(assert_reference_file()), 64)
        self.assertEqual(assert_installed_sources()['mlxLmVersion'], '0.32.2')

    def test_profile_works_during_owner_context_redirect_and_preserves_tokens(self):
        fixture = ProfileTests()
        bb, state, jobs, deps, clock, calls = fixture.fixture()
        reference, _ = outer_profile(bb, state, jobs, 10, lambda rows: None, lambda: False,
            _dependencies=deps, _clock=clock)
        bb, state, jobs, deps, clock, calls = fixture.fixture()
        base_generator = deps['BatchGenerator']
        def factory(collector):
            class TimedFake(base_generator):
                def next_generated(self):
                    return collector.measure('autoregressiveStep', lambda: super(TimedFake, self).next_generated())
            return TimedFake
        deps['profiled_generator_factory'] = factory
        old_verifier = outer_profile.__globals__['assert_reference']
        with patch.object(store_demo_engine, 'decode_rows', decode_rows_profiled):
            rows, timing = decode_rows_profiled(bb, state, jobs, 10, lambda rows: None, lambda: False,
                _dependencies=deps, _clock=clock)
        self.assertTrue(token_parity(reference, rows)['exactTokensEOSAndFields'])
        self.assertIs(outer_profile.__globals__['assert_reference'], old_verifier)
        self.assertEqual(timing['profile']['version'], 'store-demo-native-decode-profile.v2')
        self.assertTrue(timing['profile']['synchronizedNativePhases'])
        self.assertGreater(timing['profile']['nativePhases']['phases']['autoregressiveStep']['calls'], 0)
        self.assertTrue(timing['profile']['measurementLimitations'])

    def test_token_or_eos_divergence_fails_and_retains_parity_report(self):
        rows = [{'fields': ['T1'], 'rawSampledTokenIds': [1, 99], 'stopTokenId': 99}]
        changed = deepcopy(rows); changed[0]['rawSampledTokenIds'][0] = 2
        with self.assertRaises(AssertionError) as caught:
            token_parity(rows, changed)
        self.assertFalse(caught.exception.parity_report['exactTokensEOSAndFields'])


if __name__ == '__main__':
    unittest.main()
