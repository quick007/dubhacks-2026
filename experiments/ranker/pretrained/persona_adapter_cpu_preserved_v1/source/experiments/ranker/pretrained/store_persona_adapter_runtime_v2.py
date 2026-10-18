"""Explicit unpromoted LoRA product decision + learned color-only runtime.

CPU import/load; native attachment requires the caller's sole resident worker.
All route logits are recomputed from public features each request. No labels,
teacher targets, old failed persona product rankings or response words are read.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

import torch

from . import store_persona_adapter_contract_v2 as contract
from . import store_persona_choice_scoring_v1 as scoring
from . import store_qwen_lexical_choice_inputs_v2 as formatter
from .store_persona_context_v1 import load_catalog_features, source_colors
from . import store_persona_choice_capture_v1 as inference

REVISION = '8b2b98c00a6b4d291155e4890773ca8f769aee53'
COLOR_SHA = '2ec600745b19086b4bedcc4f66c283b5870b65eebce43f8715f0e8cbb154ea56'
COLOR_CACHE_SHA = 'dbf8c335a533120f5220a19f88ad163e4984631e7d9a7e80e5894080d02af28e'
INTENT_TRAIN_MANIFEST_SHA = '25fdb0c976dc3b6e01e826e7f30ea63fc65e3fbb85f56f24f4c83e8538703232'


class VerifiedQueryIntent:
    def __init__(self, path, expected_sha, *, source_manifest_sha=None):
        from . import store_query_intent_gate_v3 as gate
        self.path = Path(path); self.expected_sha = contract.exact_sha(expected_sha)
        self.head, self.metadata = gate.load_intent_head(self.path, expected_sha,
            expected_source_manifest_sha256=source_manifest_sha or INTENT_TRAIN_MANIFEST_SHA)
        self.gate = gate; self.source_sha = contract.sha(gate.__file__)

    def assert_current(self):
        if contract.sha(self.path) != self.expected_sha or contract.sha(self.gate.__file__) != self.source_sha:
            raise ValueError('Learned query-intent gate changed')
        return True

    def p_specific(self, values):
        self.assert_current()
        return self.gate.specific_probability(self.head, values)

    def provenance(self):
        return {'checkpointSha256': self.expected_sha, 'sourceSha256': self.source_sha,
            'sourceManifestSha256': self.metadata['sourceManifestSha256'], 'feature': 'same exact base query4096',
            'phraseOrPersonRules': False}


def blended_scores(v5, lexical_logits, p_specific):
    """Root-declared learned gate formula; no query-string or person rules."""
    if tuple(v5.shape) != (16,) or tuple(lexical_logits.shape) != (16,):
        raise ValueError('Exactly16 source-bound scores required')
    if not torch.isfinite(v5).all() or not torch.isfinite(lexical_logits).all():
        raise ValueError('Finite query and LoRA scores required')
    if type(p_specific) not in (float, int) or not 0 <= p_specific <= 1:
        raise ValueError('Finite learned query-intent probability required')
    persona = lexical_logits.softmax(-1)
    return (1 - p_specific) * persona + p_specific * (v5 + .05 * persona)


def load_manifest(path, expected_sha):
    path = Path(path)
    if path.is_symlink() or contract.sha(path) != contract.exact_sha(expected_sha):
        raise ValueError('Exact caller-selected adapter serving manifest required')
    value = json.loads(path.read_text())
    required = {'version', 'adapter', 'publicCache', 'modelRevision', 'formatterSha256', 'catalogSha256',
        'routeTokenIds', 'productIds', 'catalogPrefixTokenIdsSha256', 'actualSuffixTokens', 'base927Unchanged',
        'trainedUpdates', 'postfitNativeControlsExact', 'scoreSourceSha256', 'inferenceTargetFieldsRequired'}
    if (not required <= set(value) or value['version'] != 'persona-choice-adapter-serving.v1' or
            value['actualSuffixTokens'] != 16 or value['base927Unchanged'] is not True or value['trainedUpdates'] != 3584 or
            value['postfitNativeControlsExact'] is not True or value['inferenceTargetFieldsRequired'] is not False or
            value['modelRevision'] != REVISION or value['formatterSha256'] != contract.sha(formatter.__file__) or
            value['catalogSha256'] != formatter.CATALOG_FILE_SHA256 or value['scoreSourceSha256'] != contract.sha(scoring.__file__)):
        raise ValueError('Completed exact last16/3584-update/native-parity/source serving proof required')
    parity_ref = value.get('inferenceParity')
    if type(parity_ref) is not dict or set(parity_ref) != {'file', 'sha256'}:
        raise ValueError('Explicit co-located sanitized inference parity certificate required')
    parity_path = contract.local_file(path.parent, parity_ref['file'])
    if contract.sha(parity_path) != contract.exact_sha(parity_ref['sha256']):
        raise ValueError('Inference parity certificate checksum changed')
    parity = json.loads(parity_path.read_text())
    parity_fields = {'version', 'adapterSha256', 'cachedScoringMode', 'dynamicMode',
        'cachedScoreMatchesTrainingFeaturesExact', 'dynamicMatchesCachedFeatureScoresExact',
        'zeroAndNonzeroControlPassed', 'learnedPostfitPassed', 'helperSourceSha256',
        'zeroNonzeroProofSha256', 'postfitProofSha256'}
    if (type(parity) is not dict or set(parity) != parity_fields or parity['version'] != 'persona-choice-inference-parity.v1' or
            parity.get('cachedScoringMode') != 'plain' or
            parity.get('cachedScoreMatchesTrainingFeaturesExact') is not True or
            parity.get('dynamicMatchesCachedFeatureScoresExact') is not True or parity.get('zeroAndNonzeroControlPassed') is not True or
            parity.get('learnedPostfitPassed') is not True or
            parity.get('dynamicMode') != 'gradient-public-XHY-last16' or
            parity.get('helperSourceSha256') != {'score': contract.sha(scoring.__file__), 'capture': contract.sha(inference.__file__)} or
            parity['adapterSha256'] != value['adapter']['sha256'] or len(parity['postfitProofSha256']) != 4 or
            len(set(parity['postfitProofSha256'])) != 4):
        raise ValueError('Measured cached/plain/dynamic inference parity at zero/control/learned weights required; gradient TRAIN gate alone is insufficient')
    contract.exact_sha(parity['zeroNonzeroProofSha256'])
    for checksum in parity['postfitProofSha256']: contract.exact_sha(checksum)
    for field in ('cachedScoringMode', 'dynamicMode'):
        if field in value and value[field] != parity[field]:
            raise ValueError('Serving mode differs from measured fixed inference policy')
    adapter = value['adapter']
    if (set(adapter) != {'file', 'sha256', 'rank', 'scale', 'target'} or adapter['rank'] != 4 or
            adapter['scale'] != 20 or adapter['target'] != contract.TARGET):
        raise ValueError('Exact separate final31 down rank4 scale20 adapter required')
    for item in (adapter, value['publicCache']):
        artifact = contract.local_file(path.parent, item['file'])
        if contract.sha(artifact) != contract.exact_sha(item['sha256']):
            raise ValueError('Pinned co-located adapter/cache artifact changed')
    return value, parity


class LearnedColorBranch:
    """Preserve trained within-product variant math; discard old product scores."""
    def __init__(self, head_path, cache_path, catalog):
        from . import store_persona_qwen_optimize_v2 as old
        from . import store_persona_qwen_v1 as context_formatter
        self.head_path = Path(head_path); self.cache_path = Path(cache_path)
        if contract.sha(self.head_path) != COLOR_SHA or contract.sha(self.cache_path) != COLOR_CACHE_SHA:
            raise ValueError('Explicit reviewed raw2500 color branch and original public20 context cache required')
        self.head = old.load_head(self.head_path)
        if self.head.centered:
            raise ValueError('Reviewed raw color branch required')
        self.cache = torch.load(self.cache_path, map_location='cpu', weights_only=True)
        if (self.cache['version'] != 'public-persona-native-context-cache.v1' or self.cache['teacherOrLabels'] is not False or
                self.cache['contextFormatterSha256'] != contract.sha(context_formatter.__file__) or len(self.cache['features']) != 20):
            raise ValueError('Exact target-free public20 frozen color-context features required')
        self.formatter = context_formatter; self.catalog = catalog
        self.colors, self.indices = source_colors(catalog)
        self.dynamic = None
        self.pins = {str(Path(old.__file__)): contract.sha(old.__file__),
            str(Path(context_formatter.__file__)): contract.sha(context_formatter.__file__),
            str(self.head_path): COLOR_SHA, str(self.cache_path): COLOR_CACHE_SHA}

    def assert_current(self):
        if any(contract.sha(path) != checksum for path, checksum in self.pins.items()):
            raise ValueError('Learned color branch or source cache changed')
        return True

    def scores(self, value, encode_base):
        text = self.formatter.public_context_text(value)
        from .store_persona_context_v1 import canonical
        key = __import__('hashlib').sha256(canonical(text).encode()).hexdigest()
        if key in self.cache['features']:
            features = self.cache['features'][key]['tensor'].clone(); source = 'precomputed-public-color-context'; calls = 0
        elif self.dynamic is not None and self.dynamic[0] == key:
            features = self.dynamic[1].clone(); source = 'exact-last-color-context'; calls = 0
        else:
            features, native = encode_base(text, cap=512)
            self.dynamic = key, features.detach().clone(); source = 'fresh-base-color-context'; calls = 1
        if tuple(features.shape) != (1, 4096) or not torch.isfinite(features).all():
            raise ValueError('Finite original4096 color-context feature required')
        from .store_persona_qwen_fit_v1 import ownership
        with torch.no_grad():
            # Keep original product+color arithmetic for exact within-product
            # argmax compatibility. Product outputs are discarded and never fused.
            _, variants = self.head(features, self.catalog['vectors'], self.colors, self.indices,
                ownership([{'request': value}], self.catalog))
        return variants[0], {'kind': source, 'cacheHit': calls == 0, 'nativeBackboneForwardCalls': calls,
            'checkpointSha256': COLOR_SHA, 'publicFeatureCacheSha256': COLOR_CACHE_SHA,
            'oldProductScoresUsedForProductRanking': False, 'withinProductVariantMathUnchanged': True}


class AdapterPersonaRuntime:
    def __init__(self, serving_manifest, serving_sha, catalog_feature_root, *, color_head, color_cache,
                 query_intent, adapter_factory=None):
        if query_intent is None:
            raise ValueError('Explicit pinned learned query-intent gate required; no phrase whitelist fallback')
        self.manifest_path = Path(serving_manifest); self.manifest_sha = contract.exact_sha(serving_sha)
        self.manifest, self.inference_policy = load_manifest(self.manifest_path, serving_sha)
        self.adapter_path = contract.local_file(self.manifest_path.parent, self.manifest['adapter']['file'])
        self.public_catalog = formatter.load_public_catalog()
        if self.manifest['productIds'] != [p['id'] for p in self.public_catalog]:
            raise ValueError('Canonical actual16 product-bound route order required')
        self.catalog = load_catalog_features(catalog_feature_root)
        if [p['productId'] for p in self.catalog['products']] != self.manifest['productIds']:
            raise ValueError('Original catalog feature binding differs from lexical routes')
        self.public_cache = contract.PublicFeatureCache(
            contract.local_file(self.manifest_path.parent, self.manifest['publicCache']['file']),
            self.manifest['publicCache']['sha256'], routes=self.manifest['routeTokenIds'], revision=REVISION,
            formatter_sha=self.manifest['formatterSha256'], score_sha=self.manifest['scoreSourceSha256'],
            feature_sha=contract.sha(Path(__file__).with_name('store_persona_choice_features_v2.py')))
        self.color = LearnedColorBranch(color_head, color_cache, self.catalog)
        self.query_intent = query_intent; self.adapter_factory = adapter_factory
        self.bb = None; self.catalog_state = None; self.query_cache = None
        files = [Path(__file__), Path(contract.__file__), Path(formatter.__file__), Path(scoring.__file__),
            Path(inference.__file__), Path(__file__).with_name('store_persona_choice_features_v2.py'), self.manifest_path,
            contract.local_file(self.manifest_path.parent, self.manifest['inferenceParity']['file'])]
        self.pins = {str(p): contract.sha(p) for p in files}

    def attach(self, backbone, copy_adapter):
        if self.bb is not None or backbone.revision != REVISION:
            raise ValueError('Attach once to exact resident frozen base')
        mx = backbone.mx
        def adapter_pins(adapter):
            import hashlib
            import numpy as np
            mx.eval(adapter.lora_a, adapter.lora_b); mx.synchronize()
            return {key: hashlib.sha256(str(getattr(adapter, key).dtype).encode() +
                str(tuple(getattr(adapter, key).shape)).encode() + np.array(getattr(adapter, key)).tobytes()).hexdigest()
                for key in ('lora_a', 'lora_b')}
        copy_before = adapter_pins(copy_adapter)
        if self.adapter_factory is None:
            from mlx_lm.tuner.lora import LoRALinear
            factory = LoRALinear.from_base
        else:
            factory = self.adapter_factory
        # Never wrap copy_adapter: both distinct A/B adapters share only pristine linear.
        rank_adapter = factory(copy_adapter.linear, r=4, scale=20, dropout=0)
        weights = mx.load(str(self.adapter_path))
        if set(weights) != {'lora_a', 'lora_b'}:
            raise ValueError('Rank checkpoint may contain only A/B, never base/copy weights')
        if (tuple(weights['lora_a'].shape) != (12288, 4) or tuple(weights['lora_b'].shape) != (4, 4096) or
                any(value.dtype != mx.float32 for value in weights.values())):
            raise ValueError('Exact FP32 rank4 decision weights required')
        rank_adapter.load_weights(list(weights.items()), strict=False); rank_adapter.freeze(); rank_adapter.eval()
        # Root's measured fixed policy: public sum16 autodiff for dynamic
        # feature capture, then plain reconstruction for both cache/miss.
        # A/B may participate in that transform but no optimizer exists here.
        rank_adapter.unfreeze(keys=['lora_a', 'lora_b'])
        mx.eval(rank_adapter.parameters()); mx.synchronize(); backbone.check_memory()
        if any(not bool(mx.all(mx.isfinite(value)).item()) for value in weights.values()):
            raise ValueError('Finite decision adapter required')
        self.ownership = contract.ProjectionOwnership(backbone.model, copy_adapter, rank_adapter, mx.synchronize)
        self.bb = backbone; self.rank_adapter = rank_adapter
        self.adapter_weights = {key: value for key, value in weights.items()}
        self._adapter_pins = adapter_pins
        self.copy_adapter_pins = copy_before; self.rank_adapter_pins = adapter_pins(rank_adapter)
        if adapter_pins(copy_adapter) != copy_before:
            raise ValueError('Rank adapter loading mutated the independent copy adapter')
        self.assert_current()

    def assert_current(self, *, full=False):
        if any(contract.sha(path) != checksum for path, checksum in self.pins.items()):
            raise ValueError('Adapter runtime/source/serving manifest changed')
        if contract.sha(self.adapter_path) != self.manifest['adapter']['sha256']:
            raise ValueError('Rank adapter checkpoint changed')
        self.public_cache.assert_current(full=full); self.color.assert_current(); self.query_intent.assert_current()
        if self.bb is not None:
            self.ownership.assert_idle()
            if full and (self._adapter_pins(self.rank_adapter) != self.rank_adapter_pins or
                    self._adapter_pins(self.ownership.copy) != self.copy_adapter_pins):
                raise ValueError('Resident rank/copy adapter tensors changed')
        return True

    def encode_base(self, text, *, cap=512):
        tokens = self.bb.tokenizer.encode(text, add_special_tokens=False)
        if not 0 < len(tokens) <= cap:
            raise ValueError(f'Complete base query/color context must fit{cap}tokens; no clipping')
        with self.ownership.select('base'):
            features, native = self.bb.encode([text])
        if (tuple(features.shape) != (1, 4096) or not torch.isfinite(features).all() or native['batchRows'] != 1 or
                native['backboneForwardCalls'] != 1 or native['truncatedRows'] != 0):
            raise ValueError('One exact finite B1 base feature forward required')
        return features, native

    def public_row(self, value):
        original = formatter.build_row(value, self.public_catalog, self.bb.tokenizer, 'runtime-public-request', 'canonical')
        row = {key: original[key] for key in ('prompt', 'tokenIds', 'catalogPrefixTokenIds')}
        row['tokenIdsSha256'] = contract.token_hash(row['tokenIds'])
        row['routeTokenIds'] = [choice['tokenId'] for choice in original['choices']]
        checked = contract.suffix_contract(row)
        if (row['routeTokenIds'] != self.manifest['routeTokenIds'] or
                contract.token_hash(checked['catalogPrefixTokenIds']) != self.manifest['catalogPrefixTokenIdsSha256']['canonical']):
            raise ValueError('Actual tokenizer/product routes/catalog prefix differ from learned export')
        return row

    def lexical_scores(self, row, *, cancelled=lambda: False):
        from .store_demo_engine import RequestCancelled
        if cancelled(): raise RequestCancelled('Cancelled before lexical decision')
        started = time.perf_counter(); item = self.public_cache.get(row); mx = self.bb.mx
        mode = self.inference_policy['cachedScoringMode']
        def score_item(public_item):
            return scoring.score(self.bb, self.rank_adapter, public_item)
        if item is not None:
            with self.ownership.select('rank'):
                scores = score_item(item)
                mx.eval(scores); mx.synchronize(); self.bb.check_memory()
            native = {'kind': 'public-XHY-feature-reuse', 'cacheHit': True, 'nativeBodyForwards': 0,
                'nativeLanguageModelProjectionCalls': 1, 'freshLearnedDecision': True,
                'gradientEvaluationCalls': 0, 'cachedScoringMode': mode}
        else:
            from mlx_lm.models.cache import make_prompt_cache
            from .parallel_decode import clone_prefix_state
            from .native_batch_diagnosis import state_hash
            checked = contract.suffix_contract(row)
            if self.bb.tokenizer.encode(row['prompt'], add_special_tokens=False) != row['tokenIds']:
                raise ValueError('Actual whole public prompt token identity differs')
            attempted = 0; completed = 0
            def prefill(tokens, cache):
                nonlocal attempted, completed
                for offset in range(0, len(tokens), 256):
                    if cancelled(): raise RequestCancelled('Cancelled between public prefix batches')
                    attempted += 1
                    hidden = self.bb.model.model(mx.array([tokens[offset:offset + 256]]), cache=cache)
                    mx.eval(hidden, [entry.state for entry in cache]); mx.synchronize(); self.bb.check_memory()
                    completed += 1
            with self.ownership.select('base'):
                if self.catalog_state is None:
                    cache = make_prompt_cache(self.bb.model)
                    prefill(checked['catalogPrefixTokenIds'], cache)
                    self.catalog_state = {'model': self.bb.model, 'cache': cache,
                        'tokens': checked['catalogPrefixTokenIds'], 'messages': [], 'consumed': False}
                    self.catalog_state_hash = state_hash(cache)
                if self.catalog_state['tokens'] != checked['catalogPrefixTokenIds'] or self.catalog_state.get('consumed'):
                    raise ValueError('Pristine exact catalog prefix required')
                branch = clone_prefix_state(self.bb, self.catalog_state)
                prefill(checked['prefixTokenIds'][721:], branch['cache'])
                branch['tokens'] = list(checked['prefixTokenIds'])
            with self.ownership.select('rank'):
                if cancelled(): raise RequestCancelled('Cancelled before rank final16 suffix')
                attempted += 1
                dynamic_mode = self.inference_policy['dynamicMode']
                public_item, proof = inference.capture(self.bb, self.rank_adapter, branch, row['tokenIds'], row['routeTokenIds'])
                scores = score_item(public_item)
                mx.eval(scores); mx.synchronize(); self.bb.check_memory()
                completed += 1
            if state_hash(self.catalog_state['cache']) != self.catalog_state_hash:
                raise ValueError('Shared base catalog KV/SSM cache mutated')
            native = {'kind': 'fresh-native-last16-rank-adapter', 'cacheHit': False,
                'nativeBodyForwards': completed, 'attemptedNativeBodyForwards': attempted,
                'nativeLanguageModelProjectionCalls': 2, 'freshLearnedDecision': True,
                'dynamicMode': dynamic_mode, 'cachedScoringMode': mode,
                'gradientEvaluationCalls': 1,
                'catalogPrefixTokensReused': 721, 'freshPublicPrefixTokens': len(checked['prefixTokenIds']) - 721,
                'nativeRankSuffixTokens': 16, 'wholePromptNoCacheBitParityClaim': False}
            native['targetFreeCapture'] = proof
        if tuple(scores.shape) != (16,) or not bool(mx.all(mx.isfinite(scores)).item()):
            raise ValueError('Finite exact16 pretrained route logits required')
        native.update(elapsedMs=(time.perf_counter() - started) * 1000,
            adapterCheckpointSha256=self.manifest['adapter']['sha256'], promptTokenIdsSha256=row['tokenIdsSha256'],
            copyAdapterParticipated=False, generatedContinuationTokens=0, optimizerUpdates=0, targetWeightsProvided=False)
        return torch.tensor(scores.tolist(), dtype=torch.float32), native

    def rank(self, value, session, copy_adapter, *, cancelled=lambda: False):
        from .store_demo_engine import RequestCancelled
        if cancelled(): raise RequestCancelled('Cancelled before adapter ranking')
        self.assert_current()
        if self.bb is None or self.bb is not session.bb or self.ownership.copy is not copy_adapter:
            raise ValueError('Exact attached resident backbone/copy adapter required')
        request, policy, context = contract.common_request(value, self.public_catalog, self.catalog['products'])
        from .store_demo_engine import ranking_input, guarded_order
        from .store_function_adaptation import request_text
        text = request_text(ranking_input(request))
        if self.query_cache is not None and self.query_cache[0] == text:
            query = self.query_cache[1].clone(); query_native = {'cacheHit': True, 'nativeBackboneForwardCalls': 0}
        else:
            query, timing = self.encode_base(text)
            self.query_cache = text, query.detach().clone()
            query_native = {'cacheHit': False, 'nativeBackboneForwardCalls': 1, 'native': timing}
        row = self.public_row(value); lexical, decision = self.lexical_scores(row, cancelled=cancelled)
        catalog = torch.cat([session.store.load(row)[0] for row in session.catalog_rows])
        with torch.no_grad():
            v5 = session.head.probabilities('catalog', catalog) @ session.head.probabilities('query', query)[0]
        p_specific = float(self.query_intent.p_specific(query))
        scores = blended_scores(v5, lexical, p_specific)
        order, guards = guarded_order(scores.tolist(), request, policy, session.products)
        if cancelled(): raise RequestCancelled('Cancelled before learned variant branch')
        variant_scores, color = self.color.scores(value, self.encode_base)
        cards = []
        for index in order[:6]:
            eligible = [(j, variant) for j, variant in enumerate(self.catalog['variants']) if variant['productIndex'] == index and
                all(r['attribute'] != 'color' or r['value'] == variant['colorName'] for r in request['hardRequirements'])]
            if not eligible:
                raise ValueError('No source variant satisfies all hard color constraints')
            _, variant = max(eligible, key=lambda item: (float(variant_scores[item[0]]), -item[0]))
            product = self.catalog['products'][index]
            cards.append({'productIndex': index, 'facts': {'name': product['name'], 'category': product['category'],
                'itemPrice': product['itemPrice'], 'selectedColor': variant['colorName']},
                'action': {'productId': product['productId'], 'variantId': variant['variantId'], 'image': deepcopy(variant['images'][0])}})
        self.assert_current()
        if cancelled(): raise RequestCancelled('Cancelled after learned ranking')
        return request, policy, order, guards, cards, {'scores16': scores.tolist(), 'v5Scores16': v5.tolist(),
            'lexicalRouteLogits16': lexical.tolist(), 'learnedQuerySpecificProbability': p_specific,
            'fusionFormula': '(1-p)*softmax(lora)+p*(v5+0.05*softmax(lora))', 'phraseWhitelistUsed': False,
            'queryFeature': query_native, 'rankDecision': decision, 'colorFeature': color,
            'failedOldProductRankingUsed': False, 'hiddenProfileInputs': False,
            'rankAdapterAndCopyAdapterMixed': False, 'defaultPromotionClaim': False}
