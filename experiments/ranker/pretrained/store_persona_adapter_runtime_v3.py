"""Measurement-only exact V2 math with public feature preencoding; no response cache."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from . import store_persona_adapter_runtime_v2 as previous
from . import store_persona_adapter_contract_v2 as contract

VERSION = 'persona-adapter-runtime.v3'
def tensor_sha(features):
    value = features.detach().cpu().contiguous()
    metadata = json.dumps([str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
    return hashlib.sha256(metadata + b'\0' + value.view(previous.torch.uint8).numpy().tobytes()).hexdigest()


PUBLIC_PEOPLE = Path(__file__).with_name('synthetic_history_v1') / 'demo-people.json'
PUBLIC_PEOPLE_SHA = '4737c785313b3200798133679ee914aae638d5b03f46b052ecf77451f75f5b1d'


def public_browse_requests(path=PUBLIC_PEOPLE):
    if contract.sha(path) != PUBLIC_PEOPLE_SHA:
        raise ValueError('Exact reviewed public ten-person source required')
    value = json.loads(Path(path).read_text())
    people = value['people']
    if len(people) != 10 or len({p['id'] for p in people}) != 10:
        raise ValueError('Exactly ten distinct public demonstration people required')
    result = []
    for person in people:
        request = deepcopy(person['request'])
        request['query'] = 'Browse the store'  # Exact bare-UI request, retaining all public context.
        result.append((person['id'], request))
    return result


class MultiColorBranch(previous.LearnedColorBranch):
    """Only finite tensors cached; exact full formatter text includes actual query."""
    @classmethod
    def from_branch(cls, branch, capacity=32):
        if type(capacity) is not int or not 1 <= capacity <= 128:
            raise ValueError('Color feature cache capacity must be 1..128')
        value = cls.__new__(cls)
        value.__dict__.update(branch.__dict__)
        value.capacity = capacity; value.encoded = OrderedDict(); value.dynamic = None
        value.origins = {}; value.feature_checksums = {}; value.warm_cache_sha = None
        return value

    def _remember(self, text, features, origin=None):
        self.encoded[text] = features.detach().clone()
        self.encoded.move_to_end(text)
        self.origins[text] = origin or self.origins.get(text, 'dynamic-request')
        self.feature_checksums[text] = tensor_sha(features)
        while len(self.encoded) > self.capacity:
            removed, _ = self.encoded.popitem(last=False)
            self.origins.pop(removed); self.feature_checksums.pop(removed)

    def prime(self, value, encode_base):
        text = self.formatter.public_context_text(value)
        from .store_persona_context_v1 import canonical
        key = hashlib.sha256(canonical(text).encode()).hexdigest()
        started = time.perf_counter()
        if key in self.cache['features']:
            features = self.cache['features'][key]['tensor']; calls = 0; source = 'original-public-color-feature'
        elif text in self.encoded:
            features = self.encoded[text]; calls = 0; source = 'exact-full-text-color-feature'
        else:
            features, _ = encode_base(text, cap=512); calls = 1; source = 'fresh-base-color-feature'
        if tuple(features.shape) != (1, 4096) or not previous.torch.isfinite(features).all():
            raise ValueError('Finite exact4096 pure color features required')
        self._remember(text, features, origin='startup-preencoding')
        return {'formatterTextSha256': hashlib.sha256(text.encode()).hexdigest(),
            'formatterTextBytes': len(text.encode()), 'featureTensorSha256': tensor_sha(features),
            'cacheKeySha256': hashlib.sha256(text.encode()).hexdigest(), 'nativeBackboneForwardCalls': calls,
            'encodingMs': (time.perf_counter() - started) * 1000, 'featureSource': source}

    def scores(self, value, encode_base):
        started = time.perf_counter()
        text = self.formatter.public_context_text(value)
        from .store_persona_context_v1 import canonical
        key = hashlib.sha256(canonical(text).encode()).hexdigest()
        was_cached = text in self.encoded
        if was_cached and tensor_sha(self.encoded[text]) != self.feature_checksums[text]:
            raise ValueError('Resident pure color feature tensor changed')
        self.dynamic = (key, self.encoded[text]) if was_cached else None
        try:
            scores, proof = super().scores(value, encode_base)
            if self.dynamic is not None and self.dynamic[0] == key:
                self._remember(text, self.dynamic[1])
            proof.update(cachePolicy='bounded exact full formatter text; query included',
                cacheEntries=len(self.encoded), cacheCapacity=self.capacity,
                formatterTextSha256=hashlib.sha256(text.encode()).hexdigest(),
                responseWordsCached=False,
                elapsedMs=(time.perf_counter() - started) * 1000,
                cacheKeySha256=hashlib.sha256(text.encode()).hexdigest(),
                featureTensorSha256=self.feature_checksums.get(text),
                cacheProducerSourceSha256=contract.sha(__file__),
                warmCacheSha256=self.warm_cache_sha,
                cacheOrigin=self.origins.get(text, 'original-public-color-context'))
            if proof['kind'] == 'exact-last-color-context':
                proof['kind'] = ('preencoded-public-color-context' if self.origins[text] == 'startup-preencoding'
                    else 'exact-full-text-multi-color-context')
            return scores, proof
        finally:
            self.dynamic = None  # Never retain an untracked second feature slot.


class AdapterPersonaRuntime(previous.AdapterPersonaRuntime):
    def __init__(self, *args, force_fresh_query=True, preencode_public_profiles=True,
                 color_cache_entries=32, public_people=PUBLIC_PEOPLE, **kwargs):
        super().__init__(*args, **kwargs)
        if type(force_fresh_query) is not bool or type(preencode_public_profiles) is not bool:
            raise ValueError('Explicit boolean measurement controls required')
        if preencode_public_profiles and color_cache_entries < 10:
            raise ValueError('Ten preencoded public contexts require capacity at least ten')
        self.force_fresh_query = force_fresh_query
        self.preencode_public_profiles = preencode_public_profiles
        self.public_people = Path(public_people)
        self.color = MultiColorBranch.from_branch(self.color, color_cache_entries)
        self.pins[str(Path(__file__))] = contract.sha(__file__)
        self.pins[str(self.public_people)] = contract.sha(self.public_people)
        self.encoding_calls = []; self.preencoding = None; self.integrity_checks = []

    def encode_base(self, text, *, cap=512):
        started = time.perf_counter()
        features, native = super().encode_base(text, cap=cap)
        self.encoding_calls.append({'textSha256': hashlib.sha256(text.encode()).hexdigest(),
            'textBytes': len(text.encode()), 'elapsedMs': (time.perf_counter() - started) * 1000,
            'nativeBackboneForwardCalls': native['backboneForwardCalls'], 'batchRows': native['batchRows'],
            'adaptersDisabled': True, 'truncatedRows': native['truncatedRows'],
            'nativeTokenizeMs': native['tokenizeMs'], 'nativeForwardMs': native['forwardMs'],
            'paddedTokensPerRow': native['paddedTokensPerRow']})
        return features, native

    def assert_current(self, *, full=False):
        started = time.perf_counter()
        result = super().assert_current(full=full)
        if hasattr(self, 'integrity_checks'):
            self.integrity_checks.append((time.perf_counter() - started) * 1000)
        return result

    def attach(self, backbone, copy_adapter):
        super().attach(backbone, copy_adapter)
        started = time.perf_counter(); calls_before = len(self.encoding_calls); rows = []
        if self.preencode_public_profiles:
            for person_id, value in public_browse_requests(self.public_people):
                contract.common_request(value, self.public_catalog, self.catalog['products'])
                rows.append({'publicPersonId': person_id, **self.color.prime(value, self.encode_base)})
        self.ownership.assert_idle(); self.assert_current()
        self.preencoding = {'version': VERSION, 'enabled': self.preencode_public_profiles,
            'publicPeopleSha256': contract.sha(self.public_people), 'requestQuery': 'Browse the store',
            'rows': rows, 'encodingMs': (time.perf_counter() - started) * 1000,
            'baseEncodingCalls': self.encoding_calls[calls_before:],
            'nativeBackboneForwardCalls': sum(r['nativeBackboneForwardCalls'] for r in rows),
            'rankingWarmResponses': 0, 'copyWarmResponses': 0, 'responseWordsCached': False,
            'rankWholePromptFeaturesPrecomputedHere': False,
            'cacheProducerSourceSha256': contract.sha(__file__),
            'scope': 'one-time authorized public profile color/context encoding, excluded from request latency'}
        self.preencoding['warmCacheSha256'] = hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.color.warm_cache_sha = self.preencoding['warmCacheSha256']
        self.encoding_calls.clear()

    def rank(self, value, session, copy_adapter, *, cancelled=lambda: False):
        started = time.perf_counter()
        self.encoding_calls.clear(); self.integrity_checks = []
        if self.force_fresh_query:
            self.query_cache = None
        try:
            result = super().rank(value, session, copy_adapter, cancelled=cancelled)
            details = result[-1]
            details['queryFeature']['forceFreshQuery'] = self.force_fresh_query
            fresh_query = details['queryFeature'].get('nativeBackboneForwardCalls', 0) == 1
            for index, call in enumerate(self.encoding_calls):
                call['role'] = 'query' if fresh_query and index == 0 else 'color-context'
            if fresh_query and self.encoding_calls:
                details['queryFeature']['encodingMeasurement'] = deepcopy(self.encoding_calls[0])
            details['featureEncoding'] = {'baseCalls': deepcopy(self.encoding_calls),
                'nativeBackboneForwardCalls': sum(r['nativeBackboneForwardCalls'] for r in self.encoding_calls),
                'encodingMs': sum(r['elapsedMs'] for r in self.encoding_calls),
                'publicProfilePreencodingEnabled': self.preencode_public_profiles,
                'totalRuntimeRankMs': (time.perf_counter() - started) * 1000,
                'runtimeIntegrityChecksMs': sum(self.integrity_checks),
                'runtimeIntegrityCheckCalls': len(self.integrity_checks),
                'baseB1EncoderForwardCalls': sum(r['nativeBackboneForwardCalls'] for r in self.encoding_calls),
                'rankDecisionNativeBodyForwards': details.get('rankDecision', {}).get('nativeBodyForwards', 0),
                'timingScope': 'runtime rank only; engine admission, outer guards, fresh copy/serialization and browser separately',
                'colorContextCacheIncludesActualQuery': True,
                'rankXHYCacheIncludesActualWholePromptAndQuery': True,
                'newUniqueQueryMayMissBothContextAndRankCaches': True,
                'responseWordsCached': False, 'rankingFormulaUnchanged': True}
            return result
        finally:
            if self.force_fresh_query:
                self.query_cache = None
