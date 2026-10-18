"""Isolated V4 query encoder experiment; no query feature/result lookup cache.

Both arms preserve V4 conditioning. Prefix arm clones only the fixed literal
query prefix, forwards fresh whole-token-derived suffix, and proves source state.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import time

import torch

from . import store_persona_adapter_runtime_v4 as previous
from . import store_persona_adapter_runtime_v3 as measurements
from . import store_persona_adapter_contract_v2 as contract
from . import store_query_prefix_reuse_v1 as reuse

VERSION = 'persona-query-prefix-runtime.v1'
MODES = ('ordinary', 'prefix')


def assert_aliases(binding):
    aliases = binding.get('logicalAssetAliases', {})
    if binding.get('logicalAssetAliasesSha256', reuse.hashed({})) != reuse.hashed(aliases):
        raise ValueError('Prefix logical asset mapping changed')
    for logical, resolved in aliases.items():
        if str(Path(logical).resolve(strict=True)) != resolved:
            raise ValueError('Prefix asset symlink/regular-file resolution changed: ' + logical)


class AdapterPersonaRuntime(previous.AdapterPersonaRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_prefix = None; self.query_encoding_mode = 'ordinary'
        self.pins[str(Path(__file__))] = contract.sha(__file__)
        self.pins[str(Path(reuse.__file__))] = contract.sha(reuse.__file__)

    def prepare_query_prefix(self, binding, *, encoder_factory=None):
        if self.query_prefix is not None: raise ValueError('Prepare constant query prefix once per resident lifetime')
        self.ownership.assert_idle(); assert_aliases(binding)
        factory = encoder_factory or reuse.QueryPrefixReuse.prepare
        # Helper sees the pristine base directly. Never nest helper copy-disable
        # inside an ownership scope that already disabled that same adapter.
        with self.ownership.select('base'):
            self.query_prefix = factory(self.bb, binding, copy_adapter=None,
                native_owner='root', confirm_serialized=True)
        self.ownership.assert_idle()
        return deepcopy(self.query_prefix.preparation)

    def set_query_encoding_mode(self, mode):
        self.ownership.assert_idle()
        if mode not in MODES or self.query_prefix is None:
            raise ValueError('Prepared isolated encoder and explicit ordinary/prefix mode required')
        self.query_encoding_mode = mode

    def assert_current(self, *, full=False):
        result = super().assert_current(full=full)
        encoder = getattr(self, 'query_prefix', None)
        if encoder is not None:
            assert_aliases(encoder.binding); reuse.assert_assets(encoder.binding, full_sha=full)
            if full:
                with self.ownership.select('base'):
                    encoder.deps['parameter_check'](self.bb, encoder.binding)
                    if encoder.deps['state_hash'](encoder.cache) != encoder.source_hash:
                        raise ValueError('Constant query prefix changed during resident lifetime')
        return result

    def encode_base(self, text, *, cap=512):
        if not text.startswith(reuse.PREFIX) or self.query_prefix is None:
            return super().encode_base(text, cap=cap)  # Neutral color features remain ordinary V4.
        started = time.perf_counter(); mode = self.query_encoding_mode
        assert_aliases(self.query_prefix.binding)
        layout = reuse.query_layout(self.bb.tokenizer, None, formatter=lambda _: text)
        if len(layout['fullTokenIds']) > cap: raise ValueError('Complete query exceeds caller cap; no truncation')
        if mode == 'ordinary':
            features, native = super().encode_base(text, cap=cap)
            trace = self.encoding_calls[-1]
            native = {**native, 'prefixReused': False, 'reusedPrefixTokens': 0,
                'freshSuffixTokens': len(layout['fullTokenIds']), 'sourceUnchanged': True,
                'ordinaryCacheNone': True}
        elif mode == 'prefix':
            with self.ownership.select('base'):
                features, proof = self.query_prefix.encode(None, formatter=lambda _: text)
            if tuple(features.shape) != (1, 4096) or not torch.isfinite(features).all() or proof['backboneForwardCalls'] != 1:
                raise ValueError('Fresh finite exact B1 prefix-suffix feature required')
            native = {**proof, 'prefixReused': True, 'tokenizeMs': proof['tokenizeAndAssetStatMs'],
                'forwardMs': proof['freshSuffixForwardNormalizeMs'], 'ordinaryCacheNone': False}
            trace = {'textSha256': hashlib.sha256(text.encode()).hexdigest(), 'textBytes': len(text.encode()),
                'nativeBackboneForwardCalls': 1, 'batchRows': 1, 'adaptersDisabled': True, 'truncatedRows': 0,
                'nativeTokenizeMs': proof['tokenizeAndAssetStatMs'], 'nativeForwardMs': proof['freshSuffixForwardNormalizeMs'],
                'paddedTokensPerRow': proof['paddedTokensPerRow']}
            self.encoding_calls.append(trace)
        else: raise ValueError('Unknown explicit query encoding mode')
        trace.update(queryEncodingMode=mode, wholeTextSha256=layout['wholeTextSha256'],
            wholeTokenSha256=layout['wholeTokenSha256'], fullTokenIds=layout['fullTokenIds'],
            wholeRequestTokens=len(layout['fullTokenIds']), constantPrefixTokens=len(layout['prefixTokenIds']),
            prefixTokenSha256=reuse.hashed(layout['prefixTokenIds']),
            reusedPrefixTokens=native['reusedPrefixTokens'], freshSuffixTokens=native['freshSuffixTokens'],
            prefixProof=deepcopy(native), featureSha256=measurements.tensor_sha(features),
            featureVector4096=features[0].tolist(),
            elapsedMs=(time.perf_counter() - started) * 1000,
            noQueryFeatureOrScoreOrRankingOrAnswerCache=True)
        return features, native

    def rank(self, value, session, copy_adapter, *, cancelled=lambda: False):
        try:
            result = super().rank(value, session, copy_adapter, cancelled=cancelled)
            details = result[-1]; trace = details['queryFeature']['encodingMeasurement']
            if 'queryEncodingMode' not in trace:
                raise ValueError('Full actual query did not enter explicit measured encoding arm')
            audit = deepcopy(trace)
            # One copy of actual feature audit in the request receipt/events,
            # never a lookup or retained feature tensor for later inference.
            for call in details['featureEncoding']['baseCalls']:
                call.pop('featureVector4096', None)
            trace.pop('featureVector4096', None)
            details['queryEncoding'] = {'version': VERSION, 'mode': self.query_encoding_mode,
                **audit, 'featureAuditOnlyNeverInferenceReuse': True,
                'preparationExcludedFromRequest': self.query_prefix.preparation}
            return result
        finally:
            for call in self.encoding_calls: call.pop('featureVector4096', None)
