from contextlib import nullcontext
from copy import deepcopy
import hashlib
import io
import json
import queue
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import store_persona_query_projection_runtime_v1 as candidate
from . import store_persona_query_projection_engine_v1 as engine
from .store_demo_engine import ranking_input
from .store_function_adaptation import request_text


def full(query='a notebook for café notes', budget=40, requirements=None):
    return request_text(ranking_input({'query': query, 'itemBudget': budget, 'hardRequirements': requirements or []}))


class Tokenizer:
    def encode(self, text, **kwargs): return list(text.encode())


def runtime():
    value = candidate.AdapterPersonaRuntime.__new__(candidate.AdapterPersonaRuntime)
    value.bb = NS(tokenizer=Tokenizer(), revision='test-only', mx=NS(eval=lambda *a: None, synchronize=lambda: None), check_memory=lambda: None)
    value.query_prefix = NS(binding={'assetKey': 'a' * 64}, preparation={'prefixTokens': 26})
    value.ownership = NS(assert_idle=lambda: True, select=lambda mode: nullcontext())
    value.query_encoding_mode = 'ordinary'; value.query_feature_cache_mode = 'off'; value.query_feature_lru = candidate.OrderedDict()
    value.route_projection_mode = 'ordinary'; value.projector = None; value.encoding_calls = []; value.query_cache = None
    return value


def fake_encoder(calls):
    def encode(self, text, **kwargs):
        calls.append(text); features = torch.ones(1, 4096) / 64
        self.encoding_calls.append({'textSha256': hashlib.sha256(text.encode()).hexdigest(), 'textBytes': len(text.encode()),
            'nativeBackboneForwardCalls': 1, 'batchRows': 1, 'truncatedRows': 0, 'adaptersDisabled': True,
            'nativeTokenizeMs': .2, 'nativeForwardMs': 99., 'paddedTokensPerRow': len(text.encode())})
        return features, {'backboneForwardCalls': 1, 'batchRows': 1, 'tokenizeMs': .2, 'forwardMs': 99., 'paddedTokensPerRow': len(text.encode()), 'truncatedRows': 0}
    return encode


class EncodingTests(unittest.TestCase):
    def test_compact_keeps_entire_literal_unicode_query_budget_and_typed_requirements(self):
        requirements = [{'attribute': 'color', 'value': 'Ivory'}]
        text = full('all café words "and $30" stay here', 30, requirements)
        compact = candidate.compact_query_text(text)
        payload = json.loads(compact.removeprefix('Shopping request: '))
        self.assertEqual(payload, {'query': 'all café words "and $30" stay here', 'itemBudget': 30, 'hardRequirements': requirements})
        for changed in [text.replace('Choose a product for my stated use.', 'other semantic meaning'), 'wrong']:
            with self.assertRaises(ValueError): candidate.compact_query_text(changed)

    def test_compact_uses_original_base_and_different_actual_text_tokens_with_honest_audit(self):
        value = runtime(); calls = []; value.set_query_encoding_mode('compact')
        with patch.object(candidate.previous.measurements.AdapterPersonaRuntime, 'encode_base', fake_encoder(calls)):
            features, native = value.encode_base(full())
        self.assertEqual(calls, [candidate.compact_query_text(full())]); self.assertEqual(native['backboneForwardCalls'], 1)
        trace = value.encoding_calls[-1]
        self.assertNotEqual(trace['originalFormatterTextSha256'], trace['wholeTextSha256'])
        self.assertLess(trace['wholeRequestTokens'], trace['originalFormatterTokens'])
        self.assertTrue(trace['allLiteralQueryAndBudgetHardRequirementsRetained'])
        self.assertEqual(trace['featureSha256'], candidate.previous.measurements.tensor_sha(features))
        with self.assertRaises(ValueError): value.set_query_encoding_mode('prefix')

    def test_exact_full_query_lru_preserves_budgets_guard_keys_and_actual_zero_calls(self):
        value = runtime(); calls = []; value.set_query_feature_cache_mode('lru32')
        with patch.object(candidate.previous.AdapterPersonaRuntime, 'encode_base', fake_encoder(calls)):
            first, _ = value.encode_base(full()); hit, native = value.encode_base(full())
            self.assertTrue(torch.equal(first, hit)); self.assertEqual(len(calls), 1)
            self.assertEqual(native['backboneForwardCalls'], 0); self.assertEqual(native['batchRows'], 0)
            self.assertEqual(value.encoding_calls[-1]['paddedTokensPerRow'], 0); self.assertEqual(value.encoding_calls[-1]['freshSuffixTokens'], 0)
            value.encode_base(full(budget=41)); value.encode_base(full(requirements=[{'attribute': 'color', 'value': 'Ivory'}]))
            self.assertEqual(len(calls), 3)
            value.set_query_feature_cache_mode('off'); value.encode_base(full()); self.assertEqual(len(calls), 4)
        self.assertEqual(len(value.query_feature_lru), 3)
        for features, checksum in value.query_feature_lru.values():
            self.assertEqual(tuple(features.shape), (1, 4096)); self.assertEqual(checksum, candidate.previous.measurements.tensor_sha(features))

    def test_cache_corruption_rejected_and_modes_use_different_keys(self):
        value = runtime(); value.set_query_feature_cache_mode('lru32'); calls = []
        with patch.object(candidate.previous.AdapterPersonaRuntime, 'encode_base', fake_encoder(calls)):
            value.encode_base(full())
        key = next(iter(value.query_feature_lru)); value.query_feature_lru[key][0][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'feature changed'): value.encode_base(full())
        with self.assertRaises(ValueError): value.set_query_feature_cache_mode('answers')

    def test_public_default_prewarm_deduplicates_exact_eight_budgets_not_people_or_scores(self):
        from .store_persona_http_semantic_probe_v1 import public_catalog
        from . import store_qwen_lexical_choice_inputs_v2 as formatter
        value = runtime(); value.public_catalog = formatter.load_public_catalog(); value.catalog = {'products': public_catalog()}
        calls = []
        with patch.object(candidate.previous.AdapterPersonaRuntime, 'encode_base', fake_encoder(calls)):
            receipt = value.prewarm_default_query_features()
        self.assertEqual(len(calls), 8); self.assertEqual(receipt['uniquePublicFullQueryKeys'], 8)
        self.assertEqual(len(receipt['rows']), 8); self.assertEqual(sum(len(r['publicSourcePersonIds']) for r in receipt['rows']), 10)
        self.assertEqual(receipt['nativeBackboneForwardCalls'], 8); self.assertEqual(len(value.query_feature_lru), 8)
        self.assertEqual(value.query_feature_cache_mode, 'off'); self.assertEqual(value.encoding_calls, [])
        self.assertFalse(receipt['scoresRankingsResponsesOrRawTextCached'])

    def test_hit_path_actual_query_feature_counts_corrected_after_inherited_rank(self):
        value = runtime(); value.projector = NS(receipt={}); value.query_feature_cache_mode = 'lru32'
        returned = (None, None, [], {}, [], {'queryEncoding': {'queryFeatureCacheHit': True},
            'queryFeature': {'cacheHit': False, 'nativeBackboneForwardCalls': 1}, 'rankDecision': {'selected16ProjectionApplied': False}})
        with patch.object(candidate.previous.AdapterPersonaRuntime, 'rank', return_value=deepcopy(returned)):
            result = value.rank({}, None, None)
        self.assertEqual(result[-1]['queryFeature']['nativeBackboneForwardCalls'], 0)
        self.assertTrue(result[-1]['queryFeature']['cacheHit']); self.assertFalse(result[-1]['queryFeature']['forceFreshQuery'])
        self.assertTrue(result[-1]['queryFeature']['originalOneEntryQueryCacheDisabled'])


class ProjectionTests(unittest.TestCase):
    def test_selected_cached_profile_uses_bound_scorer_once_and_unknown_miss_stays_ordinary(self):
        from .test_store_persona_adapter_v4 import item
        value = runtime(); value.route_projection_mode = 'selected'
        value.manifest = {'adapter': {'sha256': 'a' * 64}}; value.rank_adapter = object()
        row = {'tokenIdsSha256': 'b' * 64}; pure = item(); key = 'profile'
        value.active_profile = (key, row['tokenIdsSha256'])
        value.profile_features = candidate.OrderedDict({key: pure})
        value.profile_metadata = {key: {'promptTokenIdsSha256': row['tokenIdsSha256'],
            'pureFeatureSha256': candidate.v4.pure_feature_sha(pure), 'nativeCachedRouteScoresExact': True}}
        calls = []; value.projector = NS(score=lambda *args: calls.append(args) or np.arange(16, dtype=np.float32))
        scores, proof = value.lexical_scores(row)
        self.assertEqual(len(calls), 1); self.assertTrue(proof['selected16ProjectionApplied'])
        self.assertEqual(proof['nativeBodyForwards'], 0); self.assertEqual(scores.tolist(), list(range(16)))
        value.profile_features.clear()
        with patch.object(candidate.previous.AdapterPersonaRuntime, 'lexical_scores', return_value=(scores, {'cacheHit': False})) as ordinary:
            _, miss = value.lexical_scores(row); ordinary.assert_called_once()
        self.assertFalse(miss['selected16ProjectionApplied'])


class HandlerTests(unittest.TestCase):
    def test_three_controls_modes_get_and_narrow_portless_cors(self):
        worker = NS(health=lambda: {'ready': True}, active_cancel=None, jobs=queue.Queue())
        query = engine.Controller(('ordinary', 'compact'), 'ordinary'); route = engine.Controller(('ordinary', 'selected'), 'ordinary')
        cache = engine.Controller(('off', 'lru32'), 'off'); Handler = engine.handler_for(worker, query, route, cache)
        def control(path, value, origin=engine.PORTLESS_ORIGIN):
            handler = Handler.__new__(Handler); handler.path = path; raw = json.dumps(value).encode()
            handler.headers = {'Content-Length': str(len(raw)), 'Origin': origin}; handler.rfile = io.BytesIO(raw)
            handler.connection = NS(settimeout=lambda x: None); response = []; handler.respond = lambda status, body: response.append((status, body))
            handler.do_POST(); return response[0]
        self.assertEqual(control(engine.previous.CONTROL_PATH, {'mode': 'compact'})[0], 200)
        self.assertEqual(control(engine.ROUTE_CONTROL, {'mode': 'selected'})[0], 200)
        self.assertEqual(control(engine.CACHE_CONTROL, {'mode': 'lru32'})[0], 200)
        self.assertEqual(control(engine.previous.CONTROL_PATH, {'mode': 'prefix'})[0], 400)
        self.assertEqual(control(engine.CACHE_CONTROL, {'mode': 'lru32'}, 'https://unrelated.example')[0], 403)
        handler = Handler.__new__(Handler); handler.headers = {'Origin': engine.PORTLESS_ORIGIN}; headers = []
        handler.send_header = lambda key, value: headers.append((key, value)); handler.cors()
        self.assertIn(('Access-Control-Allow-Origin', engine.PORTLESS_ORIGIN), headers)
        self.assertNotIn(('Access-Control-Allow-Origin', '*'), headers)


if __name__ == '__main__': unittest.main()
