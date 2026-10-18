"""Exact immutable common-root reuse candidate; all output text stays fresh.

No model load, GPU launch, global patch or serving edit occurs on import. The
exclusive benchmark owner may temporarily bind prepare_root to this object.
Every title/body/badge/heading still uses its ordinary independent native fork.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def adapter_fingerprint(bb, adapter):
    """Hash only the installed adapter's real native A/B bytes, not the 9B base."""
    from mlx.utils import tree_flatten
    values = {}
    for name, value in tree_flatten(adapter.parameters()):
        if name.endswith(('lora_a', 'lora_b')):
            host = bb.np.array(value.astype(bb.mx.float32) if str(value.dtype) == 'mlx.core.bfloat16' else value)
            values[name] = hashlib.sha256((str(value.dtype) + str(value.shape)).encode() + host.tobytes()).hexdigest()
    if len(values) != 2:
        raise ValueError('Expected exact installed final-projection LoRA A/B tensors')
    return digest(values)


class ImmutableRootCache:
    """One bounded cache entry, bound to exact model/adapter/tokenizer/root IDs.

    This stores *prompt state*, never generated text or previous response data.
    The caller returns the pristine unconsumed source; downstream independent
    forks remain unchanged. Source mutation is rejected rather than repaired.
    """
    def __init__(self, bb, adapter, *, model_receipt, checkpoint_sha256, tokenizer_receipt,
                 _prepare=None, _state_hash=None, _adapter_hash=None, _guard=None,
                 _clock=time.perf_counter):
        if not model_receipt or not checkpoint_sha256 or not tokenizer_receipt:
            raise ValueError('Verified model/checkpoint/tokenizer identities required')
        if _prepare is None:
            from .branch_local_copy import prepare_root
            _prepare = prepare_root
        if _state_hash is None:
            from .native_batch_diagnosis import state_hash
            _state_hash = state_hash
        if _guard is None:
            from .store_copy_wide_v2 import guard_memory
            _guard = guard_memory
        self.bb = bb; self.adapter = adapter; self.tokenizer = bb.tokenizer; self.model = bb.model
        self._prepare = _prepare; self._state_hash = _state_hash; self._guard = _guard; self._clock = _clock
        self._adapter_hash = _adapter_hash or (lambda: adapter_fingerprint(bb, adapter))
        self.identity = {'modelRevision': bb.revision, 'modelReceipt': deepcopy(model_receipt),
            'adapterCheckpointSha256': checkpoint_sha256, 'tokenizerReceipt': deepcopy(tokenizer_receipt)}
        self.identity_sha = digest(self.identity)
        self.initial_adapter_sha = self._adapter_hash()
        self.entry = None; self.hits = 0; self.misses = 0; self.evictions = 0
        self.observations = []

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.bb, engine.adapter, model_receipt=engine.model_verification,
            checkpoint_sha256=engine.copy_sha,
            tokenizer_receipt=engine.model_verification['smallFileSha256'])

    def _key(self, bb, common):
        if bb is not self.bb or bb.model is not self.model or bb.tokenizer is not self.tokenizer:
            raise ValueError('Root cache belongs to the exact resident backbone/model/tokenizer objects')
        if bb.revision != self.identity['modelRevision']:
            raise ValueError('Resident model revision changed')
        if bb.model.layers[-1].mlp.down_proj is not self.adapter:
            raise ValueError('Root cache requires the explicitly bound active copy adapter')
        if self._adapter_hash() != self.initial_adapter_sha:
            raise ValueError('Installed copy adapter tensors changed; cached root refused')
        if digest(self.identity) != self.identity_sha:
            raise ValueError('Verified cache identity receipt was mutated')
        tokens = bb.tokenizer.apply_chat_template([common], tokenize=True, return_dict=False,
            add_generation_prompt=False, enable_thinking=False)
        if type(tokens) is not list or not 1 <= len(tokens) <= 2048 or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError('Complete actual common-root tokens required; no truncation')
        key = digest({'identitySha256': self.identity_sha, 'adapterNativeSha256': self.initial_adapter_sha,
            'common': common, 'exactTokens': tokens})
        return key, tokens

    def assert_immutable(self):
        if self.entry is None:
            return True
        entry = self.entry; state = entry['state']
        if (state.get('consumed') is not False or state.get('model') is not self.model
                or state.get('tokens') != entry['tokens'] or state.get('messages') != [entry['common']]
                or state.get('prefixSha256') != entry['prefixSha256']
                or self._state_hash(state['cache']) != entry['stateSha256']):
            raise ValueError('Immutable common source cache changed or was consumed; refuse reuse')
        return True

    def prepare_root(self, bb, common, *, cache_factory=None):
        started = self._clock()
        if cache_factory is not None:
            raise ValueError('Alternate cache factory cannot share the verified resident root')
        key, tokens = self._key(bb, common)
        self.assert_immutable()
        hit = self.entry is not None and self.entry['key'] == key
        original_prepare_ms = None
        if not hit:
            if self.entry is not None:
                self.evictions += 1
            state, preparation = self._prepare(bb, common)
            if (state.get('consumed') is not False or state.get('model') is not self.model
                    or state.get('tokens') != tokens or state.get('messages') != [common]):
                raise ValueError('Fresh prepared cache does not match exact key; no publication')
            self.entry = {'key': key, 'state': state, 'tokens': list(tokens), 'common': deepcopy(common),
                'prefixSha256': state['prefixSha256'], 'stateSha256': self._state_hash(state['cache']),
                'originalPreparation': deepcopy(preparation)}
            self.misses += 1; original_prepare_ms = preparation['rootPrepareMs']
        else:
            self.hits += 1
        memory = self._guard(bb)
        elapsed = (self._clock() - started) * 1000
        metadata = {**deepcopy(self.entry['originalPreparation']), 'rootPrepareMs': elapsed,
            'freshEmptyCache': not hit, 'immutableRootReuse': True, 'commonRootCacheHit': hit,
            'commonRootCacheKeySha256': key, 'commonRootSourceStateSha256': self.entry['stateSha256'],
            'commonRootSourceNativeAdapterSha256': self.initial_adapter_sha,
            'originalRootBuildMs': original_prepare_ms if not hit else self.entry['originalPreparation']['rootPrepareMs'],
            'generationReused': False, 'cacheScope': 'exact common prompt state only; every output field sampled freshly',
            'cacheLookupAndValidationMs': elapsed, 'memory': memory}
        self.observations.append({'hit': hit, 'keySha256': key, 'validationMs': elapsed})
        return self.entry['state'], metadata

    def receipt(self):
        self.assert_immutable()
        return {'version': 'store-common-root-reuse.v1', 'entries': int(self.entry is not None),
            'hits': self.hits, 'misses': self.misses, 'evictions': self.evictions,
            'modelTokenizerIdentitySha256': self.identity_sha, 'adapterNativeSha256': self.initial_adapter_sha,
            'sourceStateSha256': None if self.entry is None else self.entry['stateSha256'],
            'observations': deepcopy(self.observations), 'generatedTextCached': False,
            'callerBudgetBytes': getattr(self.bb, 'memory_budget', None),
            'memoryPolicy': 'caller-declared active+peak guard; cache never increases it'}

    def clear(self):
        self.assert_immutable(); self.entry = None
