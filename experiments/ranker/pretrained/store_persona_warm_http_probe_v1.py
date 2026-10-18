"""CPU-only warm-profile/fresh-query factual analyzer; no HTTP or model launch.

Independent copy of the existing factual analyzer with new color cache kinds and
stronger startup/cache/query binding. No gold/quality gate or ranking change.
"""
from copy import deepcopy
import hashlib,json,math
from pathlib import Path
from . import store_persona_http_semantic_probe_v2 as previous
from . import store_persona_development_admission_v1 as development
from . import store_persona_adapter_runtime_v3 as measurement
from . import store_persona_adapter_runtime_v2 as runtime
from . import store_persona_adapter_contract_v2 as contract
from . import store_persona_combined_query_v1 as query
from . import store_qwen_lexical_choice_inputs_v2 as formatter
from . import store_persona_qwen_v1 as color_formatter
old=previous.old
HERE=Path(__file__).resolve().parent
NATIVE_COPY_SHA=previous.NATIVE_COPY_SHA
VERSION='persona-warm-http-factual-probe.v1'

def need(value,message):
    if not value:raise ValueError(message)
def text_sha(text):return hashlib.sha256(text.encode()).hexdigest()
def is_sha(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def selection(receipt_path,expected_sha):
    selected=development.selection(receipt_path,expected_sha)
    for name in ('store_persona_adapter_runtime_v3.py','store_persona_adapter_engine_v3.py','store_persona_warm_http_probe_v1.py'):
        path=str((HERE/name).resolve())
        need(selected['pins'].get(path)==old.sha(path),'Warm admission must explicitly pin current measurement source: '+name)
    validate_warm_startup(selected['expected']['startup'])
    return selected

def public_encoding_expectations(value,products,tokenizer):
    query_text=query.runtime_text(value,products);color_text=color_formatter.public_context_text(value)
    return {'queryTextSha256':text_sha(query_text),'queryTokens':len(tokenizer.encode(query_text,add_special_tokens=False)),
        'colorTextSha256':text_sha(color_text),'colorTokens':len(tokenizer.encode(color_text,add_special_tokens=False))}

def validate_warm_startup(startup,tokenizer=None):
    p=startup.get('persona',{});pre=p.get('featurePreencoding',{})
    need(p.get('measurementEngine')=='persona-adapter-engine.v3-measurement' and p.get('forceFreshQuery') is True and
        p.get('wholePromptRankCacheIncludesQuery') is True and p.get('noResponseWarmupOrTextCache') is True,'Fresh-query measurement engine/policy required')
    need(pre.get('version')=='persona-adapter-runtime.v3' and pre.get('enabled') is True and
        pre.get('publicPeopleSha256')==measurement.PUBLIC_PEOPLE_SHA and pre.get('requestQuery')=='Browse the store' and
        pre.get('rankingWarmResponses')==0 and pre.get('copyWarmResponses')==0 and pre.get('responseWordsCached') is False and
        pre.get('rankWholePromptFeaturesPrecomputedHere') is False and pre.get('cacheProducerSourceSha256')==old.sha(measurement.__file__),
        'One-time public profile encoding must remain separate from rank/copy response warmup')
    rows=pre.get('rows',[]);requests=measurement.public_browse_requests()
    need(len(rows)==10 and [r.get('publicPersonId') for r in rows]==[pid for pid,_ in requests],'Exact ordered ten public Browse preencoding rows required')
    calls=pre.get('baseEncodingCalls',[]);actual=[]
    for row,(_,value) in zip(rows,requests,strict=True):
        text=color_formatter.public_context_text(value);digest=text_sha(text);count=row.get('nativeBackboneForwardCalls')
        need(row.get('formatterTextSha256')==row.get('cacheKeySha256')==digest and row.get('formatterTextBytes')==len(text.encode()) and
            is_sha(row.get('featureTensorSha256')) and type(count) is int and count in (0,1),'Startup public color text/key/tensor/call binding differs')
        need(row.get('featureSource') in ('original-public-color-feature','exact-full-text-color-feature','fresh-base-color-feature') and
            (row.get('featureSource')=='fresh-base-color-feature')==(count==1),'Startup color source/actual call count differs')
        if count:actual.append((digest,len(tokenizer.encode(text,add_special_tokens=False)) if tokenizer else None))
    need(pre.get('nativeBackboneForwardCalls')==sum(r['nativeBackboneForwardCalls'] for r in rows)==len(calls),'Startup actual encoding total differs')
    for call,(digest,tokens) in zip(calls,actual,strict=True):
        need(call.get('textSha256')==digest and call.get('nativeBackboneForwardCalls')==call.get('batchRows')==1 and
            call.get('adaptersDisabled') is True and call.get('truncatedRows')==0 and
            (tokens is None or call.get('paddedTokensPerRow')==tokens),'Startup B1 source text/tokens/adapters differ')
    need(pre.get('warmCacheSha256')==hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'Exact startup warm-cache receipt hash differs')
    return {'publicProfiles':10,'startupNativeColorEncodes':len(calls),'startupEncodingMs':pre['encodingMs'],
        'rankingWarmResponses':0,'copyWarmResponses':0,'scope':'startup color/context feature encoding only; excluded from request latency, no generated response cache'}

def warm_request_errors(case,public,decision,expected):
    errors=[]
    def require(value,message):
        if not value:errors.append(message)
    q=decision.get('queryFeature',{});c=decision.get('colorFeature',{});f=decision.get('featureEncoding',{});n=decision.get('rankDecision',{})
    pre=expected.get('startup',{}).get('persona',{}).get('featurePreencoding',{})
    require(q.get('forceFreshQuery') is True and q.get('cacheHit') is False and q.get('nativeBackboneForwardCalls')==1,'Every request must perform fresh query B1 including reload')
    calls=f.get('baseCalls',[]);required=1+(c.get('nativeBackboneForwardCalls') if type(c.get('nativeBackboneForwardCalls')) is int else -99)
    require(len(calls)==required and f.get('nativeBackboneForwardCalls')==f.get('baseB1EncoderForwardCalls')==required and
        f.get('publicProfilePreencodingEnabled') is True and f.get('colorContextCacheIncludesActualQuery') is True and
        f.get('rankXHYCacheIncludesActualWholePromptAndQuery') is True and f.get('newUniqueQueryMayMissBothContextAndRankCaches') is True and
        f.get('responseWordsCached') is False and f.get('rankingFormulaUnchanged') is True and
        f.get('rankDecisionNativeBodyForwards')==n.get('nativeBodyForwards'),'Actual feature encoding split/cache/query policy differs')
    require(bool(calls) and q.get('encodingMeasurement')==calls[0],'Query measurement is not actual first base encode')
    for i,call in enumerate(calls):
        role='query' if i==0 else 'color-context';prefix='query' if i==0 else 'color'
        require(call.get('role')==role and call.get('textSha256')==public[prefix+'TextSha256'] and
            call.get('paddedTokensPerRow')==public[prefix+'Tokens'] and call.get('nativeBackboneForwardCalls')==call.get('batchRows')==1 and
            call.get('adaptersDisabled') is True and call.get('truncatedRows')==0,'Fresh base '+role+' actual text/token/B1 binding differs')
    require(c.get('formatterTextSha256')==c.get('cacheKeySha256')==public['colorTextSha256'] and is_sha(c.get('featureTensorSha256')) and
        c.get('cacheProducerSourceSha256')==old.sha(measurement.__file__) and c.get('warmCacheSha256')==pre.get('warmCacheSha256') and
        c.get('responseWordsCached') is False and c.get('cachePolicy')=='bounded exact full formatter text; query included' and
        c.get('cacheCapacity')==32 and type(c.get('cacheEntries')) is int and 1<=c['cacheEntries']<=32,'Color exact full-query text/hash/tensor/cache binding differs')
    if c.get('kind')=='preencoded-public-color-context':
        row=next((row for row in pre.get('rows',[]) if row.get('formatterTextSha256')==public['colorTextSha256']),None)
        require(row is not None and c.get('cacheOrigin')=='startup-preencoding' and c.get('cacheHit') is True and
            c.get('featureTensorSha256')==(row or {}).get('featureTensorSha256'),'Startup preencoded color hit is not this exact public request feature')
    elif c.get('kind')=='exact-full-text-multi-color-context':
        require(c.get('cacheOrigin')=='dynamic-request' and c.get('cacheHit') is True,'Repeated dynamic full-query color feature is not a real hit')
    elif c.get('kind')=='fresh-base-color-context':
        require(c.get('cacheOrigin')=='dynamic-request' and c.get('cacheHit') is False,'Fresh actual-query color feature falsely claims cache hit')
    return errors

def analyze_events(case, expected_public, events, headers, products, expected, seen):
    """Facts/guard/protocol audit, never target relevance or hidden persona labels."""
    import torch
    from .store_demo_engine import guarded_order
    from .store_parallel_text_data_v1 import field_rejections
    errors=[];quality=[]
    def require(condition,message):
        if not condition:errors.append(message)
    request_id=headers.get('x-request-id')
    require(type(request_id) is str and bool(request_id) and request_id not in seen,'missing or duplicate actual request ID')
    if request_id:seen.add(request_id)
    require(bool(events) and all(e.get('requestId')==request_id for e in events),'request ID mismatch')
    rank=next((e for e in events if e.get('type')=='rank'),None);complete=next((e for e in events if e.get('type')=='complete'),None)
    if rank is None or complete is None:
        return {'protocolPassed':False,'qualityPassed':False,'errors':errors+['missing rank/complete'],'remoteErrors':[e for e in events if e.get('type')=='error'],'requestId':request_id}
    provenance=complete.get('provenance',{});decision=provenance.get('personaRanking',{})
    require(all(e.get('provenance')==provenance for e in events),'provenance changed within request')
    require(provenance.get('rankCheckpointSha256')==expected['adapterSha256'] and
        provenance.get('v5QueryCheckpointSha256')==expected['v5HeadSha256'] and
        provenance.get('nativeCopyCheckpointSha256')==NATIVE_COPY_SHA,'adapter, v5 or original copy checkpoint differs')
    require(decision.get('phraseWhitelistUsed') is False and decision.get('failedOldProductRankingUsed') is False and
        decision.get('rankAdapterAndCopyAdapterMixed') is False and decision.get('hiddenProfileInputs') is False,
        'unexpected rank fallback, manual routing, hidden input or adapter mixing')
    lexical=old.scores(decision.get('lexicalRouteLogits16'),'LoRA route');v5=old.scores(decision.get('v5Scores16'),'v5');scores=old.scores(decision.get('scores16'),'combined')
    p=decision.get('learnedQuerySpecificProbability')
    try:recomputed=runtime.blended_scores(torch.tensor(v5),torch.tensor(lexical),p)
    except ValueError:require(False,'invalid learned query probability');recomputed=torch.full((16,),float('nan'))
    require(torch.equal(recomputed,torch.tensor(scores)),'actual combined math differs from learned gate policy')
    current,policy,_=contract.common_request(case['request'],formatter.load_public_catalog(),products)
    require(current==case['effectiveRequest'] and policy==case['effectivePolicy'],'effective guards changed')
    order,guards=guarded_order(scores,current,policy,products);v5_order,_=guarded_order(v5,current,policy,products)
    ids=[product['productId'] for product in products];wanted=[ids[i] for i in order[:6]]
    require(rank.get('productIds')==[ids[i] for i in order] and rank.get('selectedProductIds')==wanted,'actual selected learned order violates guards')
    native=decision.get('rankDecision',{});q=decision.get('queryFeature',{});color=decision.get('colorFeature',{})
    require(native.get('adapterCheckpointSha256')==expected['adapterSha256'] and
        native.get('promptTokenIdsSha256')==expected_public['promptTokenIdsSha256'],'rank feature is not bound to exact actual public prompt')
    require(native.get('freshLearnedDecision') is True and native.get('generatedContinuationTokens')==0 and
        native.get('copyAdapterParticipated') is False and native.get('optimizerUpdates')==0 and native.get('targetWeightsProvided') is False,
        'rank decision was not fresh target-free LoRA route inference')
    for label,feature in (('query',q),('color',color)):
        count=feature.get('nativeBackboneForwardCalls');hit=feature.get('cacheHit')
        require(type(hit) is bool and type(count) is int and count in (0,1) and count==int(not hit),label+' cache/native counters missing or contradictory')
    require(color.get('kind') in ('precomputed-public-color-context','preencoded-public-color-context','exact-full-text-multi-color-context','fresh-base-color-context') and
        (color.get('kind')=='fresh-base-color-context')==(color.get('cacheHit') is False),'color feature kind/cache flag differs')
    require(color.get('checkpointSha256')==expected['colorHeadSha256'] and color.get('oldProductScoresUsedForProductRanking') is False and
        color.get('withinProductVariantMathUnchanged') is True,'learned within-product color branch differs')
    require(native.get('cachedScoringMode')=='plain' and type(native.get('cacheHit')) is bool,'rank cache/mode receipt missing')
    if native.get('cacheHit') is True:
        require(native.get('kind')=='public-XHY-feature-reuse' and native.get('nativeBodyForwards')==0 and native.get('gradientEvaluationCalls')==0 and
            native.get('nativeLanguageModelProjectionCalls')==1 and native.get('catalogPrefixCacheUsed') is False,'public XHY cache did not report actual zero-body inference')
    else:
        hit=native.get('catalogPrefixCacheHit');created=native.get('catalogPrefixCreatedThisRequest');proof=native.get('targetFreeCapture',{})
        require(native.get('kind')=='fresh-native-last16-rank-adapter' and type(hit) is bool and created is (not hit) and native.get('catalogPrefixCacheUsed') is True,'explicit cold/warm catalog receipt missing')
        require(native.get('catalogPrefixTokensReused')==(721 if hit else 0) and native.get('catalogPrefixTokensPrefilled')==(0 if hit else 721) and
            native.get('catalogPrefixPrefillForwards')==(0 if hit else 3),'cold721-token prefill versus warm reuse is incorrect')
        prefix=expected_public['promptTokens']-16-721
        require(native.get('freshPublicPrefixTokens')==prefix and native.get('nativeRankSuffixTokens')==16 and
            native.get('nativeBodyForwards')==math.ceil(prefix/256)+1+(0 if hit else 3) and
            native.get('attemptedNativeBodyForwards')==native.get('nativeBodyForwards'),'actual native prefix/suffix forward counts differ')
        require(native.get('dynamicMode')=='gradient-public-XHY-last16' and native.get('gradientEvaluationCalls')==1 and
            native.get('nativeLanguageModelProjectionCalls')==2 and proof.get('targetOrLossProvided') is False and
            proof.get('optimizerUpdates')==0 and proof.get('scalar')=='sum16publicroutinglogits' and
            proof.get('sourceCacheUnchanged') is True and proof.get('adapterUnchanged') is True and
            proof.get('promptTokenIdsSha256')==expected_public['promptTokenIdsSha256'] and
            proof.get('nativeRouteScores')==lexical,'target-free dynamic capture proof missing')
    for issue in warm_request_errors(case,expected_public,decision,expected):require(False,issue)
    cards=[c for section in complete.get('sections',[]) for c in section.get('cards',[])];metadata=[c for section in rank.get('sections',[]) for c in section.get('cards',[])]
    require([c.get('productId') for c in cards]==wanted and [c.get('productId') for c in metadata]==wanted,'actual copy/action roster differs')
    require(rank.get('eligibleCount')==len(order) and complete.get('eligibleCount')==len(order) and
        complete.get('shortfall')==max(0,6-len(wanted)),'actual eligible/shortfall count differs')
    require([len(s.get('cards',[])) for s in complete.get('sections',[])]==[min(3,len(wanted)-i) for i in range(0,len(wanted),3)],'actual section grouping differs')
    require([e.get('type') for e in events]==(['rank','firstThree','complete'] if cards else ['rank','complete']),'event sequence/count differs')
    first=next((e for e in events if e.get('type')=='firstThree'),None)
    if cards:
        require(first is not None and first.get('sections')==complete.get('sections',[])[:1] and first.get('copySource')==complete.get('copySource'),'firstThree differs from completed first group')
        first_values=[v for section in (first or {}).get('sections',[]) for v in [section.get('heading'),*[c.get(key) for c in section.get('cards',[]) for key in ('title','badge','bodyCopy')]]]
        first_validation=(first or {}).get('validation',{})
        require((first or {}).get('requiredFields')==len(first_values)==1+3*min(3,len(cards)) and
            (first or {}).get('acceptedFields')==sum(v is not None for v in first_values) and
            (first or {}).get('withheldFields')==sum(v is None for v in first_values) and
            all((first or {}).get(k)==first_validation.get(k) for k in ('requiredFields','acceptedFields','withheldFields')),
            'actual firstThree field counts differ')
    by_id={p['productId']:p for p in products};bindings=[];values={}
    for section_index,section in enumerate(complete.get('sections',[])):
        heading=section.get('heading');values['S'+str(section_index+1)]=heading
        if heading is not None and field_rejections('section',heading):quality.append('unsupported section text')
        if heading is None and len(section.get('cards',[]))==3:quality.append('withheld full-group heading')
        for local,card in enumerate(section.get('cards',[])):
            index=section_index*3+local;product=by_id.get(card.get('productId'))
            variant=None if product is None else next((v for v in product['variants'] if v['variantId']==card.get('variantId')),None)
            if variant is None:quality.append('unknown source product/variant');continue
            facts={'name':product['name'],'category':product['category'],'itemPrice':product['itemPrice'],'selectedColor':variant['colorName']}
            if (card.get('originalName'),card.get('itemPrice'),card.get('selectedColor'))!=(facts['name'],facts['itemPrice'],facts['selectedColor']):quality.append('canonical identity/color/price mismatch')
            if any(r['attribute']=='color' and r['value']!=variant['colorName'] for r in current['hardRequirements']):quality.append('hard color violated')
            meta=metadata[index] if index<len(metadata) else {}
            image_ok=meta.get('image')==variant['images'][0]
            require(meta.get('variantId')==variant['variantId'] and rank.get('variants',{}).get(product['productId'])==variant['variantId'],'rank action variant mismatch')
            if not image_ok:quality.append('declared source variant/gallery image mismatch')
            if (meta.get('originalName'),meta.get('itemPrice'),meta.get('selectedColor'))!=(facts['name'],facts['itemPrice'],facts['selectedColor']):quality.append('rank canonical facts mismatch')
            if any(card.get(key) is not None for key in ('stock','size','shipping')):quality.append('unsupported availability claim')
            for prefix,key,kind in (('T','title','title'),('B','badge','badge'),('C','bodyCopy','copy')):
                text=card.get(key);values[prefix+str(index+1)]=text
                if text is None:quality.append('withheld required card field:'+key)
                elif field_rejections(kind,text,facts):quality.append('unsupported '+key)
            bindings.append({'productId':product['productId'],'variantId':variant['variantId'],'selectedColor':variant['colorName'],'canonicalImageBindingVerified':image_ok})
    validation=complete.get('validation',{});fields=validation.get('fields',[]);accepted=sum(v is not None for v in values.values());required=len(values)
    require(len(fields)==required and {r.get('field') for r in fields}==set(values) and
        all((r.get('accepted') is True)==(values.get(r.get('field')) is not None) for r in fields),'field validation identity/null receipt differs')
    require(validation.get('requiredFields')==required and validation.get('acceptedFields')==accepted and
        validation.get('withheldFields')==required-accepted and all(complete.get(k)==validation.get(k) for k in ('requiredFields','acceptedFields','withheldFields')),'actual field counts differ')
    source=complete.get('copySource',{})
    if cards:
        require(source.get('kind')=='fresh-parallel-head' and source.get('generatedEarlier') is False and source.get('freshPerRequestGeneration') is True and
            source.get('checkpointSha256')==expected['copyHeadSha256'] and source.get('feedforwardCalls')==1,'fresh actual neural word generation receipt differs')
    else:require(source.get('kind')=='no-generation' and source.get('noEligibleProducts') is True and source.get('generatedEarlier') is False and
        source.get('freshPerRequestGeneration') is False and required==0,'empty guarded result falsely claims generation')
    for event in events:
        timing=event.get('timing',{});require(type(timing.get('readyMs')) in (int,float) and math.isfinite(timing['readyMs']) and timing['readyMs']>=0,'finite serialized readiness timing missing')
    require(complete.get('timing',{}).get('fullReadyMs')==complete.get('timing',{}).get('readyMs'),'full readiness alias differs')
    if first:
        require(first.get('timing',{}).get('first3readyMs')==first.get('timing',{}).get('readyMs'),'firstThree readiness alias differs')
        require(rank['timing']['readyMs']<=first['timing']['readyMs']<=complete['timing']['readyMs'],'server readiness times went backwards')
    return {'protocolPassed':not errors,'qualityPassed':not errors and not quality,'errors':errors,'qualityFailures':quality,'requestId':request_id,
        'queryTextExpectedSha256':expected_public['queryTextSha256'],
        'queryTextBindingEvidence':'actual fresh query encodingMeasurement text SHA and padded token count equal independently formatted/tokenized public query; exact source/runtime pins',
        'learnedGateCheckpointSha256':expected['gateSha256'], 'gateBindingEvidence':'exact selected startup before/after and source pins',
        'sameV5Top1UnderSameGuards':order[:1]==v5_order[:1], 'bothAbstainUnderSameGuards':not order and not v5_order,
        'bothNonemptyUnderSameGuards':bool(order and v5_order),'learnedQuerySpecificProbability':p,
        'actualCards':len(cards),'eligibleCount':len(order),'shortfall':max(0,6-len(wanted)),
        'requiredFields':required,'acceptedFields':accepted,'withheldFields':required-accepted,
        'sixEligible':len(order)>=6,'all20Accepted':len(cards)==6 and required==20 and accepted==20,
        'queryFeature':deepcopy(q),'rankDecision':deepcopy(native),'colorFeature':deepcopy(color),'featureEncoding':deepcopy(decision.get('featureEncoding')),'startupPreencoding':deepcopy(expected['startup']['persona']['featurePreencoding']),
        'bindings':bindings,'rankTiming':rank.get('timing'),'firstThreeTiming':first.get('timing') if first else None,
        'completeTiming':complete.get('timing'),'imageScope':'declared source variant/gallery metadata only; no image fetch/pixel/paint claim'}

