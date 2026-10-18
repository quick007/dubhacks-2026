"""Ordinary V4 full-query encoding + optional selected16 frozen LM projection.

The prepared prefix exists solely for matched memory/startup accounting; the
known-failed prefix encoder cannot be selected in this variant.
"""
from copy import deepcopy
from pathlib import Path
import time
import json
import hashlib
from collections import OrderedDict
import torch

from . import store_persona_query_prefix_runtime_v1 as previous
from . import store_persona_adapter_runtime_v4 as v4
from . import store_persona_adapter_contract_v2 as contract
from . import store_persona_route_projection_v2 as projection

VERSION = 'persona-query-selected-projection-runtime.v1'
PROJECTION_SOURCE_SHA = 'd1d45fffe658860ba63b0f492e2ce836856e62956bd662cc0d224706ae293316'
MODES = ('ordinary', 'selected')


def compact_query_text(text):
    instruction = previous.reuse.INSTRUCTION
    if not text.startswith(instruction): raise ValueError('Original whole public formatter text required')
    value = json.loads(text[len(instruction):])
    request = value.get('request', {})
    if (value.get('schemaVersion') != 'actual-store-function-adaptation.v1/query' or
            set(request) != {'query', 'preferences', 'itemBudget', 'hardRequirements'} or
            request['query'] != 'Choose a product for my stated use.' or
            type(request['preferences']) is not list or len(request['preferences']) != 1 or
            type(request['preferences'][0]) is not str):
        raise ValueError('Exact runtime original query/preferences contract required; no semantic fields dropped')
    return 'Shopping request: ' + json.dumps({'query': request['preferences'][0],
        'itemBudget': request['itemBudget'], 'hardRequirements': request['hardRequirements']},
        ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class AdapterPersonaRuntime(previous.AdapterPersonaRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if contract.sha(projection.__file__) != PROJECTION_SOURCE_SHA:
            raise ValueError('Exact reviewed selected16 projection helper required')
        self.projector = None; self.route_projection_mode = 'ordinary'
        self.query_feature_cache_mode = 'off'; self.query_feature_lru = OrderedDict()
        self.pins[str(Path(__file__))] = contract.sha(__file__)
        self.pins[str(Path(projection.__file__))] = PROJECTION_SOURCE_SHA
        self.pins[str(Path(__file__).with_name('store_persona_portless_http_v1.py'))] = contract.sha(Path(__file__).with_name('store_persona_portless_http_v1.py'))

    def prewarm_default_query_features(self, modes=('ordinary',)):
        from .store_demo_engine import ranking_input
        from .store_function_adaptation import request_text
        self.ownership.assert_idle(); groups = {}
        for person_id, value in previous.measurements.public_browse_requests():
            request, _, _ = contract.common_request(value, self.public_catalog, self.catalog['products'])
            text = request_text(ranking_input(request))
            groups.setdefault(text, []).append(person_id)
        if len(groups) != 8: raise ValueError('Exact reviewed ten public Browse profiles must form eight full query/budget/guard keys')
        old_mode, old_cache = self.query_encoding_mode, self.query_feature_cache_mode
        started = time.perf_counter(); rows = []
        try:
            self.query_feature_cache_mode = 'lru32'
            for mode in modes:
                self.set_query_encoding_mode(mode)
                for text, people in groups.items():
                    features, native = self.encode_base(text)
                    trace = self.encoding_calls[-1]
                    rows.append({key: deepcopy(trace[key]) for key in (
                        'queryEncodingMode', 'originalFormatterTextSha256', 'originalFormatterTokenSha256',
                        'originalFormatterTokens', 'wholeTextSha256', 'wholeTokenSha256', 'wholeRequestTokens',
                        'featureSha256', 'queryFeatureCacheKeySha256', 'queryFeatureCacheHit', 'elapsedMs')})
                    rows[-1].update(publicSourcePersonIds=people, dtype=str(features.dtype), shape=list(features.shape),
                        nativeBackboneForwardCalls=native['backboneForwardCalls'])
        finally:
            self.query_encoding_mode, self.query_feature_cache_mode = old_mode, old_cache
            self.encoding_calls.clear(); self.query_cache = None
        return {'version': VERSION, 'scope': 'exact public default Browse query/budget/guard feature preencoding only',
            'uniquePublicFullQueryKeys': 8, 'publicProfiles': 10, 'encodingModes': list(modes), 'rows': rows,
            'encodingMs': (time.perf_counter() - started) * 1000,
            'nativeBackboneForwardCalls': sum(row['nativeBackboneForwardCalls'] for row in rows),
            'queryFeatureCacheEntries': len(self.query_feature_lru), 'capacity': 32,
            'scoresRankingsResponsesOrRawTextCached': False, 'cachedValues': 'normalized query4096 features plus integrity hash',
            'actualQueriesBudgetsHardRequirementsPreserved': True, 'defaultQueryFeatureCacheMode': old_cache}

    def prepare_route_projection(self, *, projector_factory=None):
        if self.projector is not None: raise ValueError('Selected16 projection is prepared once')
        self.ownership.assert_idle()
        factory = projector_factory or projection.SelectedRouteProjection.prepare
        with self.ownership.select('rank'):
            self.projector = factory(self.bb, self.manifest['routeTokenIds'])
        self.ownership.assert_idle()
        return deepcopy(self.projector.receipt)

    def set_query_encoding_mode(self, mode):
        self.ownership.assert_idle()
        if mode not in ('ordinary', 'compact') or self.query_prefix is None:
            raise ValueError('Only ordinary/compact query modes; failed prefix branch disabled')
        self.query_encoding_mode = mode

    def set_query_feature_cache_mode(self, mode):
        self.ownership.assert_idle()
        if mode not in ('off', 'lru32'): raise ValueError('Only off/lru32 query feature policies allowed')
        self.query_feature_cache_mode = mode

    def encode_base(self, text, *, cap=512):
        if not text.startswith(previous.reuse.PREFIX):
            return super().encode_base(text, cap=cap)
        started = time.perf_counter(); mode = self.query_encoding_mode
        original_ids = self.bb.tokenizer.encode(text, add_special_tokens=False)
        encoded_text = compact_query_text(text) if mode == 'compact' else text
        encoded_ids = self.bb.tokenizer.encode(encoded_text, add_special_tokens=False)
        if not 0 < len(encoded_ids) <= cap: raise ValueError('Complete encoded actual query exceeds cap; no truncation')
        key = previous.reuse.hashed({'version': VERSION, 'modelRevision': self.bb.revision,
            'assetKey': self.query_prefix.binding['assetKey'], 'encodingMode': mode, 'originalFormatterText': text})
        hit = self.query_feature_cache_mode == 'lru32' and key in self.query_feature_lru
        if hit:
            features, expected_sha = self.query_feature_lru[key]
            if previous.measurements.tensor_sha(features) != expected_sha: raise ValueError('Cached exact query feature changed')
            features = features.clone(); self.query_feature_lru.move_to_end(key)
            native = {'backboneForwardCalls': 0, 'batchRows': 0, 'truncatedRows': 0, 'paddedTokensPerRow': 0,
                'tokenizeMs': (time.perf_counter() - started) * 1000, 'forwardMs': 0., 'queryFeatureCacheHit': True}
            self.encoding_calls.append({'textSha256': hashlib.sha256(encoded_text.encode()).hexdigest(),
                'textBytes': len(encoded_text.encode()), 'nativeBackboneForwardCalls': 0, 'batchRows': 0,
                'adaptersDisabled': True, 'truncatedRows': 0, 'nativeTokenizeMs': native['tokenizeMs'],
                'nativeForwardMs': 0., 'paddedTokensPerRow': 0})
        elif mode == 'ordinary':
            features, native = super().encode_base(text, cap=cap)
        elif mode == 'compact':
            # Call the unchanged base encoder with the explicitly NEW text policy,
            # bypassing only the prefix helper's ordinary/prefix mode dispatch.
            features, native = previous.measurements.AdapterPersonaRuntime.encode_base(self, encoded_text, cap=cap)
        else: raise ValueError('Failed prefix/unknown encoding policy forbidden')
        if tuple(features.shape) != (1, 4096) or not torch.isfinite(features).all():
            raise ValueError('Finite exact normalized query4096 required')
        if not hit and self.query_feature_cache_mode == 'lru32':
            self.query_feature_lru[key] = (features.detach().clone(), previous.measurements.tensor_sha(features))
            while len(self.query_feature_lru) > 32: self.query_feature_lru.popitem(last=False)
        trace = self.encoding_calls[-1]
        trace.update(queryEncodingMode=mode, originalFormatterTextSha256=hashlib.sha256(text.encode()).hexdigest(),
            originalFormatterTokenSha256=previous.reuse.hashed(original_ids), originalFormatterTokens=len(original_ids),
            actualEncodingText=encoded_text, wholeTextSha256=hashlib.sha256(encoded_text.encode()).hexdigest(),
            wholeTokenSha256=previous.reuse.hashed(encoded_ids), fullTokenIds=encoded_ids,
            wholeRequestTokens=len(encoded_ids), constantPrefixTokens=0, prefixTokenSha256=previous.reuse.hashed([]),
            reusedPrefixTokens=0, freshSuffixTokens=0 if hit else len(encoded_ids),
            featureSha256=previous.measurements.tensor_sha(features), featureVector4096=features[0].tolist(),
            prefixProof=deepcopy(native),
            elapsedMs=(time.perf_counter() - started) * 1000, queryFeatureCacheMode=self.query_feature_cache_mode,
            queryFeatureCacheHit=hit, queryFeatureCacheKeySha256=key, queryFeatureCacheEntries=len(self.query_feature_lru),
            noQueryFeatureOrScoreOrRankingOrAnswerCache=self.query_feature_cache_mode == 'off',
            noScoreOrRankingOrAnswerCache=True, compactConditioningDifferentFromOriginal=mode == 'compact',
            allLiteralQueryAndBudgetHardRequirementsRetained=True)
        return features, native

    def set_route_projection_mode(self, mode):
        self.ownership.assert_idle()
        if mode not in MODES or self.projector is None: raise ValueError('Prepared ordinary/selected projection mode required')
        self.route_projection_mode = mode

    def assert_current(self, *, full=False):
        result = super().assert_current(full=full)
        if getattr(self, 'projector', None) is not None: self.projector.assert_current(full=full)
        if full:
            for features, expected in getattr(self, 'query_feature_lru', {}).values():
                if previous.measurements.tensor_sha(features) != expected: raise ValueError('Resident query feature cache changed')
        return result

    def lexical_scores(self, row, *, cancelled=lambda: False):
        from .store_demo_engine import RequestCancelled
        if cancelled(): raise RequestCancelled('Cancelled before projection arm')
        key = self.active_profile[0] if self.active_profile is not None else None
        if self.route_projection_mode == 'ordinary' or key not in self.profile_features:
            scores, native = super().lexical_scores(row, cancelled=cancelled)
            native.update(routeProjectionMode=self.route_projection_mode, selected16ProjectionApplied=False,
                projectionPolicy='ordinary reference; unknown-profile misses always retain native/full exact capture')
            return scores, native
        if self.projector is None or self.active_profile[1] != row['tokenIdsSha256']:
            raise ValueError('Prepared selected projection with exact neutral profile binding required')
        item = self.profile_features[key]; metadata = self.profile_metadata[key]
        if metadata['promptTokenIdsSha256'] != row['tokenIdsSha256'] or v4.pure_feature_sha(item) != metadata['pureFeatureSha256']:
            raise ValueError('Pure profile cache changed before selected projection')
        started = time.perf_counter(); mx = self.bb.mx
        with self.ownership.select('rank'):
            values = self.projector.score(self.bb, self.rank_adapter, item)
            mx.eval(values); mx.synchronize(); self.bb.check_memory()
        scores = torch.tensor(values.tolist(), dtype=torch.float32)
        if tuple(scores.shape) != (16,) or not torch.isfinite(scores).all():
            raise ValueError('Fresh finite16 selected projection route scores required')
        self.profile_features.move_to_end(key)
        return scores, {'kind': 'factorized-profile-XHY-feature-reuse', 'cacheHit': True,
            'nativeBodyForwards': 0, 'nativeLanguageModelProjectionCalls': 1, 'gradientEvaluationCalls': 0,
            'catalogPrefixCacheUsed': False, 'nativeCachedRouteScoresExactAtCapture': metadata['nativeCachedRouteScoresExact'],
            'elapsedMs': (time.perf_counter() - started) * 1000,
            'adapterCheckpointSha256': self.manifest['adapter']['sha256'], 'promptTokenIdsSha256': row['tokenIdsSha256'],
            'profileKeySha256': key, 'pureFeatureSha256': metadata['pureFeatureSha256'], 'profileCacheEntries': len(self.profile_features),
            'conditioningPolicy': v4.POLICY, 'actualQueryOrGuardFieldsUsedForContext': False,
            'actualQueryUsedForFreshQueryHeadAndGuards': True, 'freshLearnedDecision': True,
            'cachedScoringMode': 'plain', 'dynamicMode': 'gradient-public-XHY-last16',
            'copyAdapterParticipated': False, 'generatedContinuationTokens': 0, 'optimizerUpdates': 0,
            'targetWeightsProvided': False, 'scoresOrRankingsOrResponseWordsCached': False,
            'routeProjectionMode': 'selected', 'selected16ProjectionApplied': True,
            'selectedRouteRows': 16, 'projectionSourceSha256': PROJECTION_SOURCE_SHA,
            'projectionPolicy': 'same final-MLP adapter/residual/norm; original frozen quantized LM rows only'}

    def rank(self, value, session, copy_adapter, *, cancelled=lambda: False):
        result = super().rank(value, session, copy_adapter, cancelled=cancelled)
        query = result[-1]['queryEncoding']
        qfeature = result[-1]['queryFeature']
        qfeature.update(cacheHit=query.get('queryFeatureCacheHit', False),
            nativeBackboneForwardCalls=0 if query.get('queryFeatureCacheHit') else 1,
            forceFreshQuery=self.query_feature_cache_mode == 'off',
            originalOneEntryQueryCacheDisabled=True, queryFeatureCacheMode=self.query_feature_cache_mode)
        query['featureAuditOnlyNeverInferenceReuse'] = self.query_feature_cache_mode == 'off'
        query['featureAuditEvidenceNotAnswerCache'] = True
        result[-1]['routeProjection'] = {'version': VERSION, 'mode': self.route_projection_mode,
            'selected16ProjectionApplied': result[-1]['rankDecision']['selected16ProjectionApplied'],
            'queryEncodingMode': self.query_encoding_mode, 'failedPrefixBranchDisabled': True,
            'preparationExcludedFromRequest': deepcopy(self.projector.receipt),
            'projectionSourceSha256': PROJECTION_SOURCE_SHA, 'nativeExactnessOrSpeedNotImplied': True}
        return result
