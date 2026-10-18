from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import store_persona_adapter_runtime_v4 as candidate
from . import store_persona_adapter_runtime_v3 as previous
from . import store_persona_adapter_runtime_v2 as original
from . import store_persona_adapter_contract_v2 as contract


def item():
    return {**{key: np.zeros((1, 16, width), dtype=np.float32) for key, width in [('x', 12288), ('h', 4096), ('y', 4096)]},
        'dtypes': {k: 'mlx.core.bfloat16' for k in ('x', 'h', 'y')}, 'routeTokenIds': list(range(16))}


def runtime():
    result = candidate.AdapterPersonaRuntime.__new__(candidate.AdapterPersonaRuntime)
    result.profile_features = candidate.OrderedDict(); result.profile_metadata = {}; result.active_profile = None
    result.bb = NS(mx=NS(eval=lambda *a: None, synchronize=lambda: None), check_memory=lambda: None)
    result.rank_adapter = object(); result.ownership = NS(select=lambda mode: nullcontext())
    result.manifest = {'adapter': {'sha256': 'a' * 64}}
    return result


class ProfilePolicyTests(unittest.TestCase):
    def test_query_budget_and_hard_fields_are_excluded_without_mutating_public_context(self):
        value = previous.public_browse_requests()[0][1]; saved = deepcopy(value)
        normalized = candidate.profile_request(value); key = candidate.profile_key(value)
        for change in [{'query': 'work bag'}, {'itemBudget': 30}, {'hardRequirements': [{'attribute': 'color', 'value': 'Ivory'}]},
                       {'query': 'Trail mug under $20', 'itemBudget': 20, 'hardRequirements': []}]:
            changed = {**value, **change}
            self.assertEqual(candidate.profile_request(changed), normalized)
            self.assertEqual(candidate.profile_key(changed), key)
        self.assertEqual(normalized['query'], 'Browse the store'); self.assertIsNone(normalized['itemBudget'])
        self.assertEqual(normalized['hardRequirements'], [])
        self.assertEqual(normalized['persona'], value['persona']); self.assertEqual(normalized['publicHistory'], value['publicHistory'])
        normalized['persona']['background'].append('mutation'); self.assertEqual(value, saved)
        for mutate in [lambda v: v['persona']['background'].append('new observed context'),
                       lambda v: v['publicHistory'][-1].update(eventType='view' if v['publicHistory'][-1]['eventType'] != 'view' else 'purchase')]:
            changed = deepcopy(value); mutate(changed)
            self.assertNotEqual(candidate.profile_key(changed), key)

    def test_neutral_public_row_uses_original_formatter_and_same_profile_key_across_queries(self):
        result = runtime(); seen = []
        def make_row(self, value):
            seen.append(deepcopy(value)); return {'tokenIdsSha256': 'b' * 64}
        source = previous.public_browse_requests()[0][1]
        with patch.object(original.AdapterPersonaRuntime, 'public_row', make_row):
            result.public_row(source); first = result.active_profile
            result.public_row({**source, 'query': 'new actual query', 'itemBudget': 5})
        self.assertEqual(first, result.active_profile); self.assertEqual(seen[0], seen[1])
        self.assertIsNone(seen[0]['itemBudget']); self.assertEqual(seen[0]['hardRequirements'], [])

    def test_color_encoder_conditions_on_neutral_profile_but_ownership_history_survives(self):
        branch = candidate.FactorizedColorBranch.__new__(candidate.FactorizedColorBranch)
        source = previous.public_browse_requests()[0][1]; seen = []
        def score(self, value, encode):
            seen.append(deepcopy(value)); return torch.zeros(1), {'cacheHit': True}
        with patch.object(previous.MultiColorBranch, 'scores', score):
            _, proof = branch.scores(source, None)
            branch.scores({**source, 'query': 'different', 'itemBudget': 1}, None)
        self.assertEqual(seen[0], seen[1]); self.assertEqual(seen[0]['publicHistory'], source['publicHistory'])
        self.assertFalse(proof['actualQueryOrGuardFieldsUsedForContext']); self.assertEqual(proof['conditioningPolicy'], candidate.POLICY)


class FeatureCacheTests(unittest.TestCase):
    def test_capture_parity_is_exact_and_sanitized_without_cached_scores(self):
        scores = torch.arange(16, dtype=torch.float32)
        proof = {'nativeRouteScores': scores.tolist(), 'sourceCacheUnchanged': True,
            'adapterUnchanged': True, 'nativeRouteScoresTensor': {'diagnostic': True}}
        checked = candidate.checked_capture_proof(proof, scores)
        self.assertTrue(checked['nativeCachedRouteScoresExact'])
        self.assertEqual(checked['nativeRouteScoresSha256'], checked['cachedRouteScoresSha256'])
        self.assertNotIn('nativeRouteScores', checked); self.assertNotIn('nativeRouteScoresTensor', checked)
        for changed in [{**proof, 'nativeRouteScores': (scores + .125).tolist()},
                        {**proof, 'sourceCacheUnchanged': False}, {**proof, 'adapterUnchanged': False}]:
            with self.assertRaisesRegex(ValueError, 'match EXACTLY'): candidate.checked_capture_proof(changed, scores)

    def test_pure_feature_hash_checks_all_tensors_and_rejects_score_or_target_fields(self):
        value = item(); checksum = candidate.pure_feature_sha(value)
        changed = deepcopy(value); changed['x'][0, 0, 0] = 1
        self.assertNotEqual(candidate.pure_feature_sha(changed), checksum)
        for key in ('scores', 'targetWeights', 'personId', 'profileId'):
            with self.assertRaises(ValueError): candidate.pure_feature_sha({**value, key: []})

    def test_unknown_profile_captured_once_then_fresh_projection_without_stored_logits(self):
        result = runtime(); captures = []; projections = []
        def capture(row, **kwargs):
            captures.append(row)
            return item(), torch.arange(16.), {'nativeBodyForwards': 3, 'cacheHit': False, 'targetFreeCapture': {'nativeCachedRouteScoresExact': True}}
        result._capture_profile = capture
        row = {'tokenIdsSha256': 'b' * 64}; result.active_profile = ('profile-a', row['tokenIdsSha256'])
        with patch.object(candidate.scoring, 'score', side_effect=lambda *a: projections.append(1) or np.arange(16, dtype=np.float32)):
            first, miss = result.lexical_scores(row)
            again, hit = result.lexical_scores(row)
        self.assertTrue(torch.equal(first, again)); self.assertEqual(len(captures), 1); self.assertEqual(len(projections), 1)
        self.assertFalse(miss['cacheHit']); self.assertTrue(hit['cacheHit']); self.assertEqual(hit['nativeBodyForwards'], 0)
        self.assertEqual(set(result.profile_features['profile-a']), {'x', 'h', 'y', 'dtypes', 'routeTokenIds'})
        self.assertFalse(hit['scoresOrRankingsOrResponseWordsCached'])
        result.profile_features['profile-a']['y'][0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, 'tensor or prompt binding changed'): result.lexical_scores(row)

    def test_failed_capture_and_cancel_never_create_cache_entries_and_lru_is_bounded(self):
        result = runtime(); row = {'tokenIdsSha256': 'b' * 64}; result.active_profile = ('a', row['tokenIdsSha256'])
        def fail(*a, **k): raise RuntimeError('native parity mismatch')
        result._capture_profile = fail
        with self.assertRaisesRegex(RuntimeError, 'parity mismatch'): result.lexical_scores(row)
        self.assertEqual(len(result.profile_features), 0)
        with self.assertRaisesRegex(Exception, 'Cancelled'): result.lexical_scores(row, cancelled=lambda: True)
        result._capture_profile = lambda *a, **k: (item(), torch.zeros(16), {'targetFreeCapture': {'nativeCachedRouteScoresExact': True}})
        for index in range(33):
            result.active_profile = (str(index), row['tokenIdsSha256']); result.lexical_scores(row)
        self.assertEqual(len(result.profile_features), 32); self.assertNotIn('0', result.profile_features)
        self.assertEqual(len(result.profile_metadata), 32)


class OriginalQueryGuardTests(unittest.TestCase):
    def test_actual_query_v5_and_guards_are_not_normalized(self):
        from .store_demo_engine import guarded_order
        from .store_function_adaptation import request_text
        from .store_demo_engine import ranking_input
        from .store_persona_http_semantic_probe_v1 import public_catalog
        from . import store_qwen_lexical_choice_inputs_v2 as formatter
        products = public_catalog(); public = formatter.load_public_catalog()
        source = previous.public_browse_requests()[0][1]
        a = {**source, 'query': 'a notebook for notes', 'itemBudget': 90}
        b = {**source, 'query': 'Trail mug under $20', 'itemBudget': 20}
        ar, ap, _ = contract.common_request(a, public, products); br, bp, _ = contract.common_request(b, public, products)
        self.assertNotEqual(request_text(ranking_input(ar)), request_text(ranking_input(br)))
        self.assertEqual(candidate.profile_key(a), candidate.profile_key(b))
        first, _ = guarded_order(list(range(16)), ar, ap, products)
        second, guards = guarded_order(list(range(16)), br, bp, products)
        self.assertGreater(len(first), 0); self.assertEqual(second, [])
        self.assertTrue(guards['exactNamePolicy']['ineligibleExactIdentityAbstention'])

    def test_full_inherited_rank_uses_changed_actual_query_features_for_v5_scores(self):
        from .store_persona_http_semantic_probe_v1 import public_catalog
        from . import store_qwen_lexical_choice_inputs_v2 as formatter
        result = runtime(); products = public_catalog(); public = formatter.load_public_catalog()
        copy_adapter = object(); result.ownership.copy = copy_adapter
        result.assert_current = lambda **kwargs: True
        result.public_catalog = public
        result.catalog = {'products': products, 'variants': [
            {'productIndex': index, **variant} for index, product in enumerate(products) for variant in product['variants']]}
        result.query_intent = NS(p_specific=lambda query: 1.)
        result.force_fresh_query = True; result.preencode_public_profiles = True; result.query_cache = None
        result.encoding_calls = []; result.integrity_checks = []
        texts = []
        def encode(text, **kwargs):
            texts.append(text); query = torch.zeros(1, 4096)
            query[0, 2 if 'writing notes' in text else 8] = 1
            result.encoding_calls.append({'nativeBackboneForwardCalls': 1, 'elapsedMs': 1.})
            return query, {'backboneForwardCalls': 1}
        result.encode_base = encode
        result.public_row = lambda value: {'profileKey': candidate.profile_key(value)}
        result.lexical_scores = lambda row, **kwargs: (torch.zeros(16), {'nativeBodyForwards': 0, 'profileKey': row['profileKey']})
        result.color = NS(scores=lambda *args: (torch.zeros(len(result.catalog['variants'])), {'nativeBackboneForwardCalls': 0}))
        head = NS(probabilities=lambda role, values: torch.eye(16) if role == 'catalog' else values[:, :16])
        session = NS(bb=result.bb, head=head, products=products, catalog_rows=list(range(16)),
            store=NS(load=lambda index: (torch.eye(16)[index:index + 1], {})))
        source = previous.public_browse_requests()[0][1]
        a = {**source, 'query': 'something for writing notes', 'itemBudget': 90}
        b = {**source, 'query': 'something for carrying books', 'itemBudget': 90}
        first = result.rank(a, session, copy_adapter); second = result.rank(b, session, copy_adapter)
        self.assertEqual(first[-1]['rankDecision']['profileKey'], second[-1]['rankDecision']['profileKey'])
        self.assertNotEqual(first[-1]['v5Scores16'], second[-1]['v5Scores16'])
        self.assertEqual(first[2][0], 2); self.assertEqual(second[2][0], 8)
        self.assertIn('writing notes', texts[0]); self.assertIn('carrying books', texts[1])
        self.assertEqual(len(texts), 2)

    def test_rank_calls_inherited_actual_pipeline_and_discloses_conditioning_change(self):
        result = runtime(); actual = {'query': 'work bag', 'itemBudget': 30, 'hardRequirements': []}; seen = []
        def rank(self, value, *a, **k):
            seen.append(deepcopy(value)); return (value, {}, [], {}, [], {'featureEncoding': {}})
        with patch.object(previous.AdapterPersonaRuntime, 'rank', rank):
            output = result.rank(actual, None, None)
        self.assertEqual(seen, [actual]); self.assertEqual(output[0], actual)
        self.assertTrue(output[-1]['contextConditioning']['actualRequestUsedForFreshQueryHeadAndHardGuards'])
        self.assertTrue(output[-1]['contextConditioning']['differentFromV2V3'])
        self.assertFalse(output[-1]['featureEncoding']['rankXHYCacheIncludesActualWholePromptAndQuery'])


class EngineTests(unittest.TestCase):
    def test_v4_engine_uses_fixed_profile_policy_and_preserves_old_owned_cleanup(self):
        from .test_store_persona_adapter_v2 import EngineLifecycleTests
        from . import store_persona_adapter_engine_v4 as engine
        fixture = EngineLifecycleTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        class Runtime(fixture.Runtime):
            def __init__(self, *a, **k):
                self_test.assertTrue(k['force_fresh_query']); self_test.assertTrue(k['preencode_public_profiles'])
                super().__init__(*a, **k); self.preencoding = {'version': candidate.VERSION}
        self_test = self
        value = engine.engine_class(base_cls=fixture.Base, runtime_factory=Runtime,
            intent_factory=fixture.intent, cleanup=fixture.cleanup)(output=fixture.output,
            serving_manifest='serving.json', serving_sha='s', catalog_feature_root='catalog', color_head='color.pt',
            color_cache='cache.pt', query_intent_head='intent.pt', query_intent_sha='g')
        self.assertEqual(value.startup['persona']['measurementEngine'], engine.VERSION)
        self.assertFalse(value.startup['persona']['wholePromptRankCacheIncludesQuery'])
        self.assertTrue(value.startup['persona']['contextConditioning']['differentFromV2V3'])
        self.assertTrue(value.close()['frozen927HashesUnchanged'])


if __name__ == '__main__': unittest.main()
