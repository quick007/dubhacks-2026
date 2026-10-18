from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import store_persona_adapter_runtime_v3 as candidate
from . import store_persona_adapter_runtime_v2 as previous


class ColorCacheTests(unittest.TestCase):
    def branch(self, capacity=32):
        branch = previous.LearnedColorBranch.__new__(previous.LearnedColorBranch)
        branch.formatter = NS(public_context_text=lambda value: json.dumps(value, sort_keys=True))
        branch.cache = {'features': {}}; branch.dynamic = None
        branch.catalog = {'products': [{'productId': 'a'}], 'vectors': torch.zeros(1, 4096)}
        branch.colors = torch.zeros(1, 3); branch.indices = torch.tensor([0])
        branch.head = lambda features, *args: (features[:, :1], features[:, :1] * 2)
        return candidate.MultiColorBranch.from_branch(branch, capacity)

    def test_interleaved_contexts_reuse_only_exact_full_text_and_preserve_scores(self):
        branch = self.branch(); calls = []
        def encode(text, **kwargs):
            calls.append(text); return torch.full((1, 4096), float(len(text))), {'backboneForwardCalls': 1}
        a = {'query': 'Browse the store', 'publicHistory': [], 'persona': {'background': ['A']}}
        b = {**a, 'persona': {'background': ['B']}}
        first, first_proof = branch.scores(a, encode)
        branch.scores(b, encode)
        again, proof = branch.scores(a, encode)
        self.assertEqual(len(calls), 2); self.assertTrue(torch.equal(first, again))
        self.assertEqual(proof['kind'], 'exact-full-text-multi-color-context')
        self.assertEqual(proof['nativeBackboneForwardCalls'], 0)
        changed = {**a, 'query': 'A genuinely different query'}
        branch.scores(changed, encode); self.assertEqual(len(calls), 3)
        self.assertEqual(first_proof['nativeBackboneForwardCalls'], 1)
        self.assertIsNone(branch.dynamic)

    def test_preencoding_never_runs_variant_head_and_records_tensor_pin(self):
        branch = self.branch(); calls = []; head_calls = []
        branch.head = lambda features, *args: (head_calls.append(1) or features[:, :1], features[:, :1])
        value = {'query': 'Browse the store', 'publicHistory': []}
        def encode(text, **kwargs):
            calls.append(text); return torch.arange(4096.).reshape(1, -1), {}
        row = branch.prime(value, encode)
        self.assertEqual(head_calls, []); self.assertEqual(len(calls), 1)
        text = branch.formatter.public_context_text(value)
        self.assertEqual(row['cacheKeySha256'], hashlib.sha256(text.encode()).hexdigest())
        _, proof = branch.scores(value, encode)
        self.assertEqual(proof['kind'], 'preencoded-public-color-context')
        self.assertEqual(proof['featureTensorSha256'], row['featureTensorSha256'])
        self.assertEqual(len(calls), 1); self.assertEqual(len(head_calls), 1)
        branch.encoded[text][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'tensor changed'): branch.scores(value, encode)

    def test_bounded_lru_and_failure_do_not_cache_failed_forward(self):
        branch = self.branch(capacity=1)
        def encode(text, **kwargs): return torch.zeros(1, 4096), {}
        for query in ['A', 'B']: branch.scores({'query': query, 'publicHistory': []}, encode)
        self.assertEqual(len(branch.encoded), 1); self.assertEqual(len(branch.feature_checksums), 1)
        def fail(*args, **kwargs): raise RuntimeError('native failed')
        with self.assertRaisesRegex(RuntimeError, 'native failed'):
            branch.scores({'query': 'C', 'publicHistory': []}, fail)
        self.assertEqual(len(branch.encoded), 1); self.assertIsNone(branch.dynamic)
        for capacity in (0, 129, True):
            with self.assertRaises(ValueError): self.branch(capacity)

    def test_public_ten_requests_match_ui_empty_search_without_mutating_seed(self):
        original = json.loads(candidate.PUBLIC_PEOPLE.read_text())
        rows = candidate.public_browse_requests()
        self.assertEqual(len(rows), 10)
        for (_, request), person in zip(rows, original['people'], strict=True):
            expected = deepcopy(person['request']); expected['query'] = 'Browse the store'
            self.assertEqual(request, expected)
        rows[0][1]['persona']['background'].append('test mutation')
        self.assertEqual(json.loads(candidate.PUBLIC_PEOPLE.read_text()), original)


class RuntimePolicyTests(unittest.TestCase):
    def runtime(self, fresh=True):
        runtime = candidate.AdapterPersonaRuntime.__new__(candidate.AdapterPersonaRuntime)
        runtime.force_fresh_query = fresh; runtime.preencode_public_profiles = True
        runtime.query_cache = ('same query', torch.ones(1, 4096)); runtime.encoding_calls = []
        return runtime

    def test_forced_query_recomputes_every_request_and_preserves_base_math_result(self):
        runtime = self.runtime(); counts = []
        def original(self, *args, **kwargs):
            self_test.assertIsNone(self.query_cache); counts.append(1)
            self.query_cache = ('same query', torch.ones(1, 4096))
            self.encoding_calls.append({'nativeBackboneForwardCalls': 1, 'elapsedMs': 7.})
            return ('request', 'policy', [3, 1], [], ['card'], {'queryFeature': {'cacheHit': False}})
        self_test = self
        with patch.object(previous.AdapterPersonaRuntime, 'rank', original):
            for _ in range(2):
                result = runtime.rank({}, None, None)
                self.assertEqual(result[:5], ('request', 'policy', [3, 1], [], ['card']))
                self.assertEqual(result[-1]['featureEncoding']['nativeBackboneForwardCalls'], 1)
                self.assertEqual(result[-1]['featureEncoding']['encodingMs'], 7.)
                self.assertFalse(result[-1]['queryFeature']['cacheHit'])
                self.assertIsNone(runtime.query_cache)
        self.assertEqual(len(counts), 2)

    def test_query_control_mode_and_cancel_cleanup(self):
        runtime = self.runtime(fresh=False)
        with patch.object(previous.AdapterPersonaRuntime, 'rank', return_value=(0, 0, [], [], [], {'queryFeature': {'cacheHit': True}})):
            runtime.rank({}, None, None)
        self.assertIsNotNone(runtime.query_cache)
        runtime.force_fresh_query = True
        def cancel(self, *args, **kwargs):
            self.query_cache = ('temporary', torch.zeros(1, 4096))
            raise RuntimeError('cancelled in native stage')
        with patch.object(previous.AdapterPersonaRuntime, 'rank', cancel):
            with self.assertRaisesRegex(RuntimeError, 'cancelled'): runtime.rank({}, None, None)
        self.assertIsNone(runtime.query_cache)

    def test_encode_retains_actual_native_token_and_forward_measurements(self):
        runtime = self.runtime()
        native = {'backboneForwardCalls': 1, 'batchRows': 1, 'truncatedRows': 0,
            'tokenizeMs': 2.5, 'forwardMs': 173.5, 'paddedTokensPerRow': 73}
        with patch.object(previous.AdapterPersonaRuntime, 'encode_base', return_value=(torch.ones(1, 4096), native)):
            runtime.encode_base('the actual full query')
        measured = runtime.encoding_calls[0]
        self.assertEqual(measured['nativeTokenizeMs'], 2.5)
        self.assertEqual(measured['nativeForwardMs'], 173.5)
        self.assertEqual(measured['paddedTokensPerRow'], 73)
        self.assertEqual(measured['nativeBackboneForwardCalls'], 1)
        self.assertEqual(measured['textSha256'], hashlib.sha256(b'the actual full query').hexdigest())

    def test_attach_preencodes_only_color_ten_and_leaves_projection_idle(self):
        runtime = self.runtime(); runtime.public_people = candidate.PUBLIC_PEOPLE
        runtime.public_catalog = []; runtime.catalog = {'products': []}
        runtime.color = NS(prime=lambda value, encode: {'nativeBackboneForwardCalls': 1, 'featureTensorSha256': 'pin'})
        idle = []; runtime.ownership = NS(assert_idle=lambda: idle.append(1)); runtime.assert_current = lambda: True
        runtime.encode_base = lambda *a, **k: (_ for _ in ()).throw(AssertionError('prime injection owns fake forward'))
        with patch.object(previous.AdapterPersonaRuntime, 'attach'), patch.object(candidate.contract, 'common_request'):
            runtime.attach(None, None)
        self.assertEqual(len(runtime.preencoding['rows']), 10)
        self.assertEqual(runtime.preencoding['nativeBackboneForwardCalls'], 10)
        self.assertEqual(runtime.preencoding['rankingWarmResponses'], 0)
        self.assertEqual(runtime.preencoding['copyWarmResponses'], 0)
        self.assertEqual(len(idle), 1)
        self.assertEqual(runtime.color.warm_cache_sha, runtime.preencoding['warmCacheSha256'])


class EngineTests(unittest.TestCase):
    def test_engine_configures_runtime_before_attachment_and_keeps_close_proofs(self):
        from .test_store_persona_adapter_v2 import EngineLifecycleTests
        from . import store_persona_adapter_engine_v3 as engine
        fixture = EngineLifecycleTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        options = []
        class Runtime(fixture.Runtime):
            def __init__(self, *args, **kwargs):
                options.append(kwargs); super().__init__(*args, **kwargs)
                self.preencoding = {'rows': [], 'encodingMs': 5., 'rankingWarmResponses': 0}
        klass = engine.engine_class(base_cls=fixture.Base, runtime_factory=Runtime,
            intent_factory=fixture.intent, cleanup=fixture.cleanup)
        value = klass(output=fixture.output, serving_manifest='serving.json', serving_sha='s', catalog_feature_root='catalog',
            color_head='color.pt', color_cache='cache.pt', query_intent_head='intent.pt', query_intent_sha='g',
            force_fresh_query=True, preencode_public_profiles=True, color_cache_entries=32)
        self.assertTrue(options[0]['force_fresh_query']); self.assertTrue(options[0]['preencode_public_profiles'])
        self.assertEqual(value.startup['persona']['measurementEngine'], engine.VERSION)
        self.assertEqual(value.startup['persona']['featurePreencoding']['encodingMs'], 5.)
        self.assertTrue(value.close()['frozen927HashesUnchanged'])

    def test_preencoding_failure_uses_existing_owned_native_cleanup(self):
        from .test_store_persona_adapter_v2 import EngineLifecycleTests
        from . import store_persona_adapter_engine_v3 as engine
        fixture = EngineLifecycleTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        class Runtime(fixture.Runtime):
            def attach(self, *args): raise RuntimeError('public encoding cancelled')
        klass = engine.engine_class(base_cls=fixture.Base, runtime_factory=Runtime,
            intent_factory=fixture.intent, cleanup=fixture.cleanup)
        with self.assertRaisesRegex(RuntimeError, 'public encoding cancelled'):
            klass(output=fixture.output, serving_manifest='serving.json', serving_sha='s', catalog_feature_root='catalog',
                color_head='color.pt', color_cache='cache.pt', query_intent_head='intent.pt', query_intent_sha='g')
        self.assertEqual(sum(isinstance(item, tuple) and item[0] == 'cleanup' for item in fixture.events), 1)


if __name__ == '__main__': unittest.main()
