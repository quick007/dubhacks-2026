"""CPU persona ranking/variant runtime and exact public-query feature cache.

Only exact frozen-base query/public-context vectors may be reused. Learned
context scores, selected bindings and all neural copy words are recomputed for
every request. No labels are read.
"""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
from pathlib import Path
import time
import torch
from . import store_persona_context_v1 as context_model

VERSION = 'store-persona-runtime.v1'
# Fixed before hybrid native evaluation; neutral means public broad browse text.
BROAD_QUERIES = context_model.NEUTRAL | {'show me a few useful things for the next couple of weeks.',
    'help me find a practical addition for my everyday routine.', 'what should i browse next?',
    'show me useful products for my current plans.'}

FUSION_POLICY={'version':'public-persona-query-fusion.v1','specificQueryResidual':.05,
    'broadQueries':sorted(BROAD_QUERIES),'broadRanking':'softmax native persona logits',
    'specificRanking':'unchanged v5 probability plus fixed0.05 persona probability; not calibrated on outcomes',
    'constraintPrecedence':'merged most restrictive typed/query hardguards, then exact eligible identity',
    'freshIndependentGoldParsedWhenDeclared':False}

def fuse_scores(v5_scores,persona_logits,query):
    if tuple(v5_scores.shape)!=(16,) or tuple(persona_logits.shape)!=(16,):raise ValueError('Exact16 source roster required')
    if not torch.isfinite(v5_scores).all() or not torch.isfinite(persona_logits).all():raise ValueError('Finite fusion inputs required')
    broad=query.strip().casefold() in BROAD_QUERIES
    preference=persona_logits.softmax(-1)
    return (preference if broad else v5_scores+FUSION_POLICY['specificQueryResidual']*preference),broad

class ExactQueryFeatureCache:
    """One exact canonical public query, versioned by immutable model/source pins."""
    def __init__(self, provenance):
        self.provenance = deepcopy(provenance); self.entry = None
    def encode(self, text, backbone, *, copy_adapter=None):
        from .semantic_copy_training import disabled
        started = time.perf_counter(); key = hashlib.sha256(context_model.canonical({'text': text, 'baseAdapterDisabled': True, 'pins': self.provenance}).encode()).hexdigest()
        if self.entry is not None and self.entry[0] == key:
            values = self.entry[1].clone()
            return values, {'cacheHit': True, 'queryFeatureCacheKey': key, 'nativeBackboneForwardCalls': 0,
                'elapsedMs': (time.perf_counter()-started)*1000, 'originalNative': deepcopy(self.entry[2]),
                'scope': 'exact public request text frozen-base feature reuse; no context/rank/variant/copy cached'}
        active = backbone.model.layers[-1].mlp.down_proj
        if hasattr(active, 'lora_a') and copy_adapter is None: raise ValueError('Explicit copy adapter disable required for frozen-base query feature')
        if copy_adapter is not None and active is not copy_adapter: raise ValueError('Copy adapter does not belong to resident backbone')
        with disabled(backbone.model, copy_adapter) if copy_adapter is not None else nullcontext():
            values, timing = backbone.encode([text])
        if tuple(values.shape) != (1,4096) or not torch.isfinite(values).all() or timing['backboneForwardCalls'] != 1 or timing['batchRows'] != 1 or timing['truncatedRows'] != 0:
            raise ValueError('One complete fresh finite B1 Qwen query feature required')
        self.entry = (key, values.detach().clone(), deepcopy(timing))
        return values, {'cacheHit': False, 'queryFeatureCacheKey': key, 'nativeBackboneForwardCalls': 1,
            'elapsedMs': (time.perf_counter()-started)*1000, 'originalNative': timing,
            'scope': 'fresh frozen-base public query feature; no context/rank/variant/copy cached'}

class PersonaRuntime:
    """Load an explicit head plus its co-located public context cache.

    ``feature_root`` / launcher ``personaFeatureRoot`` is the original verified
    catalog feature run (actual-store-function-v3-attempt-1), not the new public
    context extraction directory. The public context cache always comes from
    the selected head's own directory, with the checkpoint's expected checksum.
    """
    def __init__(self, head_path, expected_sha, feature_root):
        if context_model.sha(head_path) != expected_sha: raise ValueError('Explicit persona checkpoint hash required')
        self.head_path = Path(head_path); self.head_sha = expected_sha
        self.native_context=False
        saved_type=torch.load(head_path,map_location='cpu',weights_only=True)['version']
        if saved_type == 'public-persona-qwen-head.v1':
            from .store_persona_qwen_fit_v1 import load
            from . import store_persona_qwen_v1 as formatter
            self.head,self.saved=load(head_path,expected_sha);self.native_context=True
            original_cache_path = self.saved['publicDemoFeatureCacheFile']
            cache_path = self.head_path.parent / Path(original_cache_path).name
            if not cache_path.is_file():
                raise FileNotFoundError('Co-located public context cache required at ' + str(cache_path) + '; saved original path is provenance only')
            if context_model.sha(cache_path)!=self.saved['publicDemoFeatureCacheSha256']:
                raise ValueError('Co-located public context cache checksum mismatch')
            self.context_cache_provenance = {'checkpointOriginalFile': original_cache_path,
                'runtimeFile': str(cache_path.resolve()), 'sha256': self.saved['publicDemoFeatureCacheSha256'],
                'resolution': 'selected checkpoint directory plus saved basename; no original-path fallback'}
            self.context_cache=torch.load(cache_path,map_location='cpu',weights_only=True)
            if self.context_cache['teacherOrLabels'] is not False or self.context_cache['contextFormatterSha256']!=context_model.sha(formatter.__file__):raise ValueError('Exact public nativecontext formatter/cache required')
            self.context_cache_path=cache_path;self.dynamic_context_cache=None
        else:self.head, self.saved = context_model.load_head(head_path)
        self.catalog_feature_root = Path(feature_root)
        self.catalog = context_model.load_catalog_features(self.catalog_feature_root)
        if self.saved['catalogPins'] != self.catalog['pins']: raise ValueError('Persona checkpoint and runtime real catalog features differ')
        self.pins = {str(Path(__file__)): context_model.sha(__file__), str(Path(context_model.__file__)): context_model.sha(context_model.__file__), str(self.head_path): expected_sha}
        if self.native_context:
            from . import store_persona_qwen_v1 as formatter
            from . import store_persona_qwen_fit_v1 as learner
            self.pins[str(self.context_cache_path)]=self.saved['publicDemoFeatureCacheSha256']
            for module in (formatter,learner):self.pins[str(Path(module.__file__))]=context_model.sha(module.__file__)
        self.colors, self.indices = context_model.source_colors(self.catalog)
        self.cache = None
    def assert_current(self):
        if any(context_model.sha(p) != checksum for p, checksum in self.pins.items()): raise ValueError('Persona runtime/checkpoint changed')
    def rank(self, value, session, adapter):
        from .store_demo_engine import ranking_input, guarded_order
        from .store_function_adaptation import request_text
        self.assert_current(); started = time.perf_counter()
        request, policy, context = context_model.checked_context_request(value, self.catalog['products'])
        text = request_text(ranking_input(request)); tokens = session.bb.tokenizer.encode(text, add_special_tokens=False)
        if not 0 < len(tokens) <= 512: raise ValueError('Whole query must fit512tokens')
        if self.cache is None:
            self.cache = ExactQueryFeatureCache({'featureProvenance': session.provenance, 'v5Head': session.head_sha,
                'personaRuntimeSourcePins': self.pins})
        query, native = self.cache.encode(text, session.bb, copy_adapter=adapter)
        catalog = torch.cat([session.store.load(row)[0] for row in session.catalog_rows])
        context_provenance={'kind':'public-hashed-context','nativeCalls':0}
        if self.native_context:
            tf,context_provenance=self.native_context_feature(value,session,adapter)
        else:tf,hf=context_model.context_features(context,self.catalog)
        with torch.no_grad():
            v5 = session.head.probabilities('catalog', catalog) @ session.head.probabilities('query', query)[0]
            if self.native_context:
                from .store_persona_qwen_fit_v1 import ownership
                persona_scores,variants=self.head(tf,self.catalog['vectors'],self.colors,self.indices,ownership([{'request':value}],self.catalog))
            else:persona_scores, variants = self.head(tf[None], hf[None], self.catalog['vectors'], self.colors, self.indices)
        # Preserve native query semantics on specific requests. Small bounded
        # learned residual breaks ties; broad requests use the persona head.
        scores,broad=fuse_scores(v5,persona_scores[0],request['query'])
        order, guards = guarded_order(scores.tolist(), request, policy, session.products)
        cards = []
        for i in order[:6]:
            choices = [(j,v) for j,v in enumerate(self.catalog['variants']) if v['productIndex'] == i and all(r['attribute'] != 'color' or r['value'] == v['colorName'] for r in request['hardRequirements'])]
            _, variant = max(choices, key=lambda item: (float(variants[0,item[0]]),-item[0]))
            product = self.catalog['products'][i]
            cards.append({'productIndex': i, 'facts': {'name': product['name'], 'category': product['category'], 'itemPrice': product['itemPrice'], 'selectedColor': variant['colorName']},
                'action': {'productId': product['productId'], 'variantId': variant['variantId'], 'image': deepcopy(variant['images'][0])}})
        return request, policy, order, guards, cards, {'scores16': scores.tolist(), 'v5Scores16': v5.tolist(),
            'personaScores16': persona_scores[0].tolist(), 'rawPersonaOrder': persona_scores[0].argsort(descending=True).tolist(),
            'queryFeature': native, 'contextFeature':context_provenance, 'queryTextSha256': hashlib.sha256(text.encode()).hexdigest(), 'queryTokens': len(tokens),
            'hybridPolicy': FUSION_POLICY,
            'broadPublicBrowse': broad, 'personaCheckpointSha256': self.head_sha,
            'contextRepresentation': self.saved['config']['contextRepresentation'], 'learnedVariantColor': True,
            'imagePolicy': 'first actual gallery image of learned source-bound color; image position deterministic',
            'freshContextRankVariant': True, 'elapsedMs': (time.perf_counter()-started)*1000}

    def native_context_feature(self,value,session,adapter):
        from . import store_persona_qwen_v1 as formatter
        from .semantic_copy_training import disabled
        started=time.perf_counter();text=formatter.public_context_text(value);key=hashlib.sha256(text.encode()).hexdigest()
        tokens=session.bb.tokenizer.encode(text,add_special_tokens=False)
        if not 0<len(tokens)<=formatter.CONFIG['maxTokens']:raise ValueError('Whole public context exceeds512tokens; no truncation')
        if self.saved['featureProvenance']!=session.provenance or self.context_cache['featureProvenance']!=session.provenance:raise ValueError('Native context and resident frozenbase provenance differ')
        record=self.context_cache['features'].get(key)
        if record is not None:
            if tokens!=record['tokenIds']:raise ValueError('Cached public context native tokens changed')
            values=record['tensor'].clone();source='precomputed-public-demo-feature';calls=0
        elif self.dynamic_context_cache is not None and self.dynamic_context_cache[0]==key:
            values=self.dynamic_context_cache[1].clone();source='exact-last-context-feature';calls=0
        else:
            with disabled(session.bb.model,adapter):values,timing=session.bb.encode([text])
            if timing['backboneForwardCalls']!=1 or timing['batchRows']!=1 or timing['truncatedRows']!=0:raise ValueError('Fresh wholeB1 public context required')
            self.dynamic_context_cache=(key,values.detach().clone());source='fresh-native-public-context';calls=1
        if tuple(values.shape)!=(1,4096) or not torch.isfinite(values).all():raise ValueError('Finite native context4096required')
        return values,{'kind':source,'cacheHit':calls==0,'nativeBackboneForwardCalls':calls,'publicContextTextSha256':key,'tokens':len(tokens),
            'elapsedMs':(time.perf_counter()-started)*1000,'precomputedContextCacheArtifact':deepcopy(self.context_cache_provenance),
            'scope':'only exact public frozen-base context feature reused; learned ownership/rank/color/fresh neural copy recomputed'}
