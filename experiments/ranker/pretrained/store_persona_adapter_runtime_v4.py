"""Serving-only factorized public-profile feature candidate; no training or promotion.

DIFFERENT conditioning from V2/V3: only persona/history condition lexical/color
features. Original actual query and guards still condition fresh query/v5 math.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import store_persona_adapter_runtime_v2 as original
from . import store_persona_adapter_runtime_v3 as previous
from . import store_persona_adapter_contract_v2 as contract
from . import store_persona_choice_capture_v1 as capture
from . import store_persona_choice_scoring_v1 as scoring

VERSION = 'persona-adapter-runtime.v4-factorized-profile'
POLICY = 'factorized-public-persona-history.v1'


def profile_request(value):
    """Remove only query/guard conditioning; retain exact observed public context."""
    return {'query': 'Browse the store', 'itemBudget': None, 'hardRequirements': [],
        'persona': deepcopy(value.get('persona', {'background': [], 'colorPreferences': []})),
        'publicHistory': deepcopy(value.get('publicHistory', []))}


def profile_key(value):
    return hashlib.sha256(json.dumps(profile_request(value), sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def pure_feature_sha(item):
    scoring.validate(item)
    digest = hashlib.sha256(json.dumps({'dtypes': item['dtypes'], 'routeTokenIds': item['routeTokenIds']},
        sort_keys=True, separators=(',', ':')).encode())
    for key in ('x', 'h', 'y'):
        array = np.ascontiguousarray(item[key])
        digest.update(json.dumps([key, str(array.dtype), list(array.shape)], separators=(',', ':')).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def checked_capture_proof(proof, reconstructed):
    native = torch.tensor(proof['nativeRouteScores'], dtype=torch.float32)
    if (tuple(reconstructed.shape) != (16,) or not torch.isfinite(reconstructed).all() or
            not torch.equal(native, reconstructed) or proof.get('sourceCacheUnchanged') is not True or
            proof.get('adapterUnchanged') is not True):
        raise ValueError('Neutral profile native route scores and cached/plain reconstruction must match EXACTLY')
    sanitized = {k: deepcopy(v) for k, v in proof.items() if k not in ('nativeRouteScores', 'nativeRouteScoresTensor')}
    sanitized.update(nativeCachedRouteScoresExact=True,
        nativeRouteScoresSha256=hashlib.sha256(native.numpy().tobytes()).hexdigest(),
        cachedRouteScoresSha256=hashlib.sha256(reconstructed.numpy().tobytes()).hexdigest())
    return sanitized


class FactorizedColorBranch(previous.MultiColorBranch):
    def prime(self, value, encode_base):
        return super().prime(profile_request(value), encode_base)

    def scores(self, value, encode_base):
        scores, proof = super().scores(profile_request(value), encode_base)
        proof.update(conditioningPolicy=POLICY, profileKeySha256=profile_key(value),
            actualQueryOrGuardFieldsUsedForContext=False, cacheProducerSourceSha256=contract.sha(__file__),
            cachePolicy='bounded exact neutral profile text; actual query/budget/hard requirements excluded')
        return scores, proof


class AdapterPersonaRuntime(previous.AdapterPersonaRuntime):
    def __init__(self, *args, force_fresh_query=True, preencode_public_profiles=True,
                 color_cache_entries=32, **kwargs):
        if force_fresh_query is not True or preencode_public_profiles is not True or color_cache_entries != 32:
            raise ValueError('V4 requires ten-profile preencoding, fresh B1 queries and bounded32-entry caches')
        super().__init__(*args, force_fresh_query=True, preencode_public_profiles=True,
            color_cache_entries=32, **kwargs)
        self.color = FactorizedColorBranch.from_branch(self.color, 32)
        self.profile_features = OrderedDict(); self.profile_metadata = {}; self.active_profile = None
        self.pins[str(Path(__file__))] = contract.sha(__file__)

    def assert_current(self, *, full=False):
        result = super().assert_current(full=full)
        if full and hasattr(self, 'profile_features'):
            for key, item in self.profile_features.items():
                if pure_feature_sha(item) != self.profile_metadata[key]['pureFeatureSha256']:
                    raise ValueError('Resident pure profile feature cache changed')
        return result

    def public_row(self, value):
        # The inherited formatter/tokenizer literal721 prefix is unchanged.
        normalized = profile_request(value)
        row = original.AdapterPersonaRuntime.public_row(self, normalized)
        self.active_profile = (profile_key(value), row['tokenIdsSha256'])
        return row

    def _capture_profile(self, row, *, cancelled):
        from .store_demo_engine import RequestCancelled
        from mlx_lm.models.cache import make_prompt_cache
        from .parallel_decode import clone_prefix_state
        from .native_batch_diagnosis import state_hash
        checked = contract.suffix_contract(row); mx = self.bb.mx
        if self.bb.tokenizer.encode(row['prompt'], add_special_tokens=False) != row['tokenIds']:
            raise ValueError('Complete neutral profile prompt/token identity differs')
        attempted = 0; completed = 0; catalog_prefills = 0
        def prefill(tokens, state):
            nonlocal attempted, completed
            for offset in range(0, len(tokens), 256):
                if cancelled(): raise RequestCancelled('Cancelled between profile prefix batches')
                attempted += 1
                hidden = self.bb.model.model(mx.array([tokens[offset:offset + 256]]), cache=state)
                mx.eval(hidden, [entry.state for entry in state]); mx.synchronize(); self.bb.check_memory()
                completed += 1
        catalog_hit = self.catalog_state is not None
        with self.ownership.select('base'):
            if self.catalog_state is None:
                state = make_prompt_cache(self.bb.model)
                before = completed; prefill(checked['catalogPrefixTokenIds'], state)
                catalog_prefills = completed - before
                self.catalog_state = {'model': self.bb.model, 'cache': state,
                    'tokens': checked['catalogPrefixTokenIds'], 'messages': [], 'consumed': False}
                self.catalog_state_hash = state_hash(state)
            if self.catalog_state['tokens'] != checked['catalogPrefixTokenIds'] or self.catalog_state.get('consumed'):
                raise ValueError('Exact pristine shared catalog prefix required')
            branch = clone_prefix_state(self.bb, self.catalog_state)
            prefill(checked['prefixTokenIds'][721:], branch['cache'])
            branch['tokens'] = list(checked['prefixTokenIds'])
        with self.ownership.select('rank'):
            if cancelled(): raise RequestCancelled('Cancelled before neutral profile final16 capture')
            attempted += 1
            item, proof = capture.capture(self.bb, self.rank_adapter, branch, row['tokenIds'], row['routeTokenIds'])
            scores = scoring.score(self.bb, self.rank_adapter, item)
            mx.eval(scores); mx.synchronize(); self.bb.check_memory(); completed += 1
        if state_hash(self.catalog_state['cache']) != self.catalog_state_hash:
            raise ValueError('Immutable catalog KV/SSM prefix changed during profile capture')
        reconstructed = torch.tensor(scores.tolist(), dtype=torch.float32)
        # Exact native/cache parity admitted before pure features are stored.
        sanitized = checked_capture_proof(proof, reconstructed)
        return item, reconstructed, {'kind': 'fresh-native-neutral-profile-last16', 'cacheHit': False,
            'nativeBodyForwards': completed, 'attemptedNativeBodyForwards': attempted,
            'nativeLanguageModelProjectionCalls': 2, 'gradientEvaluationCalls': 1,
            'catalogPrefixCacheUsed': True, 'catalogPrefixCacheHit': catalog_hit,
            'catalogPrefixCreatedThisRequest': not catalog_hit,
            'catalogPrefixTokensReused': 721 if catalog_hit else 0,
            'catalogPrefixTokensPrefilled': 0 if catalog_hit else 721,
            'catalogPrefixPrefillForwards': catalog_prefills,
            'freshPublicPrefixTokens': len(checked['prefixTokenIds']) - 721,
            'nativeRankSuffixTokens': 16, 'targetFreeCapture': sanitized}

    def lexical_scores(self, row, *, cancelled=lambda: False):
        from .store_demo_engine import RequestCancelled
        if cancelled(): raise RequestCancelled('Cancelled before factorized profile projection')
        if self.active_profile is None or self.active_profile[1] != row['tokenIdsSha256']:
            raise ValueError('Exact neutral profile key/prompt binding required')
        key = self.active_profile[0]; started = time.perf_counter(); mx = self.bb.mx
        if key in self.profile_features:
            item = self.profile_features[key]; metadata = self.profile_metadata[key]
            if metadata['promptTokenIdsSha256'] != row['tokenIdsSha256'] or pure_feature_sha(item) != metadata['pureFeatureSha256']:
                raise ValueError('Pure profile cache tensor or prompt binding changed')
            with self.ownership.select('rank'):
                scores = scoring.score(self.bb, self.rank_adapter, item)
                mx.eval(scores); mx.synchronize(); self.bb.check_memory()
            scores = torch.tensor(scores.tolist(), dtype=torch.float32)
            self.profile_features.move_to_end(key)
            native = {'kind': 'factorized-profile-XHY-feature-reuse', 'cacheHit': True,
                'nativeBodyForwards': 0, 'nativeLanguageModelProjectionCalls': 1,
                'gradientEvaluationCalls': 0, 'catalogPrefixCacheUsed': False,
                'nativeCachedRouteScoresExactAtCapture': metadata['nativeCachedRouteScoresExact']}
        else:
            item, scores, native = self._capture_profile(row, cancelled=cancelled)
            item = deepcopy(scoring.public_item(item)); scoring.validate(item)
            metadata = {'promptTokenIdsSha256': row['tokenIdsSha256'], 'pureFeatureSha256': pure_feature_sha(item),
                'nativeCachedRouteScoresExact': native['targetFreeCapture']['nativeCachedRouteScoresExact']}
            self.profile_features[key] = item; self.profile_metadata[key] = metadata
            while len(self.profile_features) > 32:
                removed, _ = self.profile_features.popitem(last=False); self.profile_metadata.pop(removed)
        if tuple(scores.shape) != (16,) or not torch.isfinite(scores).all():
            raise ValueError('Fresh finite16 route scores required')
        native.update(elapsedMs=(time.perf_counter() - started) * 1000,
            adapterCheckpointSha256=self.manifest['adapter']['sha256'], promptTokenIdsSha256=row['tokenIdsSha256'],
            profileKeySha256=key, pureFeatureSha256=metadata['pureFeatureSha256'], profileCacheEntries=len(self.profile_features),
            conditioningPolicy=POLICY, actualQueryOrGuardFieldsUsedForContext=False,
            actualQueryUsedForFreshQueryHeadAndGuards=True, freshLearnedDecision=True,
            cachedScoringMode='plain', dynamicMode='gradient-public-XHY-last16',
            copyAdapterParticipated=False, generatedContinuationTokens=0, optimizerUpdates=0,
            targetWeightsProvided=False, scoresOrRankingsOrResponseWordsCached=False)
        return scores, native

    def attach(self, backbone, copy_adapter):
        super().attach(backbone, copy_adapter)
        color = deepcopy(self.preencoding); started = time.perf_counter(); rows = []
        for person_id, value in previous.public_browse_requests(self.public_people):
            row = self.public_row(value)
            _, proof = self.lexical_scores(row)
            rows.append({'publicPersonId': person_id, **proof})
        self.ownership.assert_idle(); self.assert_current(full=True)
        self.preencoding = {'version': VERSION, 'conditioningPolicy': POLICY, 'enabled': True,
            'publicPeopleSha256': previous.PUBLIC_PEOPLE_SHA, 'profileQuery': 'Browse the store',
            'profileItemBudget': None, 'profileHardRequirements': [], 'publicPersonaAndHistoryPreservedExactly': True,
            'colorPreencoding': color, 'rankProfileRows': rows,
            'rankProfilePreencodingMs': (time.perf_counter() - started) * 1000,
            'rankingWarmResponses': 0, 'copyWarmResponses': 0, 'responseWordsCached': False,
            'rankProfileCachePureXHYOnly': True, 'scoresOrRankingsCached': False,
            'cacheProducerSourceSha256': contract.sha(__file__),
            'scope': 'DIFFERENT profile-only conditioning experiment; no V2/V3 quality or logit parity claim'}
        self.preencoding['encodingMs'] = color['encodingMs'] + self.preencoding['rankProfilePreencodingMs']
        self.preencoding['nativeBodyForwards'] = sum(r['nativeBodyForwards'] for r in rows)
        self.preencoding['profileWarmupSha256'] = hashlib.sha256(json.dumps(self.preencoding,
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.encoding_calls.clear(); self.query_cache = None

    def rank(self, value, session, copy_adapter, *, cancelled=lambda: False):
        result = super().rank(value, session, copy_adapter, cancelled=cancelled)
        result[-1]['featureEncoding'].update(colorContextCacheIncludesActualQuery=False,
            rankXHYCacheIncludesActualWholePromptAndQuery=False,
            newUniqueQueryMayMissBothContextAndRankCaches=False,
            profileFeatureKeyExcludesQueryBudgetHardRequirements=True,
            profileCacheContainsPureFeaturesOnly=True)
        result[-1]['contextConditioning'] = {'version': POLICY,
            'query': 'Browse the store', 'itemBudget': None, 'hardRequirements': [],
            'publicPersonaAndObservedHistoryPreservedExactly': True,
            'actualRequestUsedForFreshQueryHeadAndHardGuards': True,
            'differentFromV2V3': True, 'modelPromotion': False, 'sourceSha256': contract.sha(__file__)}
        return result
