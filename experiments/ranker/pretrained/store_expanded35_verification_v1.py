"""CPU-only source/receipt audit for explicit expanded35 practical serving.
No native model, browser, training packet or labels are loaded here.
"""
from copy import deepcopy
import hashlib,json,math
from pathlib import Path
VERSION='expanded35-verification.v1'
MAIN=Path('/Users/lukas/Documents/Projects/dubhacks-2026')
CATALOG=MAIN/'packages/contracts/data/expanded-store-catalog-v1.json'
CATALOG_SHA='2c3b4e95bff97eda5d8e4ef50e72739fa8083456f01b1e3041d966abee26389d'
PEOPLE=MAIN/'experiments/ranker/pretrained/synthetic_history_v1/demo-people.json'
PROFILE_ORDER=('maya','priya','lena','june','grace','sam','ana','rosa','felix','bob')
NOVEL_QUERIES=('Useful things for a short break next weekend','A few practical additions for my daily routine','Help me choose something for a quiet evening')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def load_catalog(path=CATALOG):
    if sha(path)!=CATALOG_SHA:raise ValueError('Exact reviewed35 source catalog SHA required')
    source=json.loads(Path(path).read_text());products=deepcopy(source['products'])
    if len(products)!=35 or sum(len(p['variants']) for p in products)!=105 or len({p['productId'] for p in products})!=35:raise ValueError('Exact35/105 source roster required')
    for p in products:p['title']=p['name'];p['colorVariants']=[{'colorName':v['colorName'],'images':deepcopy(v['images'])} for v in p['variants']]
    return products

def fixed_schedule():
    people={p['id']:p for p in json.loads(PEOPLE.read_text())['people']};rows=[]
    def row(kind,pid,query,**extra):
        request=deepcopy(people[pid]['request']);request['query']=query
        return {'kind':kind,'personId':pid,'displayName':people[pid]['displayName'],'query':query,'request':request,**extra}
    rows.extend(row('warm-public-Browse',pid,'Browse the store') for pid in PROFILE_ORDER)
    rows.extend([row('category-guard','lena','Rain jackets',expectedIntent='rain-jacket'),row('category-guard','sam','Work bags',expectedIntent='work-bag'),row('category-guard','bob','Rain jackets under $90 in Charcoal',expectedIntent='rain-jacket')])
    rows.extend(row('novel-query-miss','bob',q) for q in NOVEL_QUERIES)
    return [{**r,'index':i} for i,r in enumerate(rows)]

def distribution(values,n):
    finite=[v for v in values if type(v) in (int,float) and math.isfinite(v)];ok=len(values)==len(finite)==n
    def pct(q):
        s=sorted(values);x=(n-1)*q;l=math.floor(x);h=math.ceil(x);return s[l]+(s[h]-s[l])*(x-l)
    return {'fixedDenominator':n,'recorded':len(values),'valid':len(finite),'complete':ok,'p50':pct(.5) if ok else None,'p95':pct(.95) if ok else None,'samples':values}

def startup_evidence(startup_path,startup_sha,binding_path):
    if sha(startup_path)!=startup_sha:raise ValueError('Actual startup SHA changed')
    startup=json.loads(Path(startup_path).read_text());expanded=startup.get('expandedCatalog',{});binding=json.loads(Path(binding_path).read_text())
    if expanded.get('products')!=35 or expanded.get('variants')!=105 or expanded.get('catalogSha256')!=CATALOG_SHA or expanded.get('bindingManifestSha256')!=sha(binding_path):raise ValueError('Actual expanded35 startup/catalog/binding join failed')
    if expanded.get('modelPromotion') is not False or expanded.get('newSkuLoRATrainingClaim') is not False:raise ValueError('Expanded development/nontrainednewSKU disclosure required')
    if startup.get('sourceTensorsHashed')!=927:raise ValueError('Actual927 base tensor startup proof required')
    pins={str(Path(startup_path).resolve()):startup_sha,str(Path(binding_path).resolve()):sha(binding_path),str(CATALOG):CATALOG_SHA}
    for relative,digest in binding.get('sourceSha256',{}).items():
        path=Path(relative);path=path if path.is_absolute() else MAIN/path
        if sha(path)!=digest:raise ValueError('Current expanded UI/catalog source drift: '+str(path))
        pins[str(path)]=digest
    copy=expanded['copy'];head=next((p for p,h in copy['assetSha256'].items() if h==copy['checkpointSha256'] and Path(p).suffix=='.pt'),None)
    if head is None or sha(head)!=copy['checkpointSha256']:raise ValueError('Actual expanded copy checkpoint binding missing')
    import torch
    saved=torch.load(head,map_location='cpu',weights_only=True)
    if saved.get('catalogSha256')!=CATALOG_SHA or not isinstance(saved.get('vocabulary'),list):raise ValueError('CPU vocabulary must join actual catalog/checkpoint')
    pins[head]=copy['checkpointSha256']
    return {'startup':startup,'pins':pins,'vocabulary':saved['vocabulary'],'products':load_catalog(),'copyCheckpointSha256':copy['checkpointSha256'],'bindingManifestSha256':sha(binding_path),'qualityPromotion':False}

def slot_fields(cards):
    result=[]
    for start in range(0,len(cards),3):
        if len(cards[start:start+3])==3:result.append((f'S{start//3+1}','section',None))
        for i in range(start,min(start+3,len(cards))):
            for prefix,key,kind in [('T','title','title'),('B','badge','badge'),('C','bodyCopy','copy')]:result.append((f'{prefix}{i+1}',kind,cards[i]))
    return result

def audit_events(request,events,headers,evidence,server_receipt,seen=None):
    import torch
    from .store_expanded_rank_v1 import checked_request,strict_eligible_order,expanded_scores
    from .store_parallel_text_head_v1 import decode_ids
    from .store_parallel_text_data_v1 import field_rejections
    errors=[];withheld=[]
    def req(condition,message):
        if not condition:errors.append(message)
    rid=headers.get('x-request-id');seen=seen if seen is not None else set();req(bool(rid) and rid not in seen,'Missing or repeated request ID')
    if rid:seen.add(rid)
    req(all(e.get('requestId')==rid for e in events),'Actual stream request IDs disagree')
    rank=next((e for e in events if e.get('type')=='rank'),None);complete=next((e for e in events if e.get('type')=='complete'),None)
    if not rank or not complete:return {'passed':False,'errors':errors+['Missing actual rank/complete events'],'requestId':rid}
    req(server_receipt.get('requestId')==rid and server_receipt.get('input')==request and server_receipt.get('events')==events and server_receipt.get('status')=='completed','Actual completed server receipt/request/rawstream join failed')
    provenance=rank['provenance'];d=provenance['personaRanking'];products=evidence['products'];byid={p['productId']:p for p in products}
    req(all(e.get('provenance')==provenance for e in events),'Provenance changed across events')
    req(d.get('catalogProducts')==35 and d.get('catalogSha256')==CATALOG_SHA and d.get('newProductsTrainedByLoRA') is False and d.get('rankAdapterAndCopyAdapterMixed') is False,'Expanded35/family/context provenance differs')
    scores=d.get('scores',[]);v5=d.get('v5Scores',[]);lexical=d.get('lexicalRouteLogits16',[]);p=d.get('learnedQuerySpecificProbability')
    req(len(scores)==len(v5)==35 and len(lexical)==16 and all(type(x) in (int,float) and math.isfinite(x) for x in scores+v5+lexical),'Exact finite35scores/16family logits required')
    request_checked,policy,_=checked_request(request,products)
    req(server_receipt.get('effectiveRequest')==request_checked and server_receipt.get('policy')==policy,'Actual effective hardguard receipt differs')
    # Original16 family route order is pinned by actual serving export, never inferred from new SKU order.
    family_ids=evidence['startup']['persona'].get('productIds')
    if family_ids is None:
        from .store_qwen_lexical_choice_inputs_v2 import load_public_catalog
        family_ids=[x['id'] for x in load_public_catalog()]
    try:
        recomputed,family=expanded_scores(torch.tensor(v5),torch.tensor(lexical),p,products,family_ids)
        req(torch.equal(recomputed,torch.tensor(scores)) and torch.equal(family,torch.tensor(d['familyPriorScores'])),'Actual35 fusion/family probability math differs')
    except Exception as error:errors.append('Expanded fusion invalid: '+str(error))
    order,guards=strict_eligible_order(scores,request_checked,policy,products);wanted=[products[i]['productId'] for i in order[:6]]
    req(rank.get('productIds')==[products[i]['productId'] for i in order] and rank.get('selectedProductIds')==wanted and server_receipt.get('finalOrder')==order and server_receipt.get('guards')==guards,'Actual35 order/category/hardguard receipt differs')
    cards=[c for s in complete.get('sections',[]) for c in s.get('cards',[])];req([c.get('productId') for c in cards]==wanted,'Actual copy/action order differs')
    req(rank.get('eligibleCount')==complete.get('eligibleCount')==len(order) and complete.get('shortfall')==max(0,6-len(cards)),'Real eligibility/shortfall count differs')
    first=next((e for e in events if e.get('type')=='firstThree'),None)
    if cards:req(first is not None and first.get('sections')==complete.get('sections',[])[:1] and first.get('copySource')==complete.get('copySource'),'Actual firstThree differs from completed firstgroup')
    req([e.get('type') for e in events]==(['rank','firstThree','complete'] if cards else ['rank','complete']),'Unexpected actual event sequence')
    meta=[c for s in rank.get('sections',[]) for c in s.get('cards',[])];values={};factual=[]
    for si,section in enumerate(complete.get('sections',[])):
        values['S'+str(si+1)]=section.get('heading')
        if section.get('heading') is None and len(section.get('cards',[]))<3:withheld.append({'field':'S'+str(si+1),'reason':'declared-short-group-heading-abstention'})
        elif section.get('heading') is None:req(False,'Missing fullgroup fresh section heading')
        else:req(not field_rejections('section',section['heading']),'Unsupported fresh section heading')
        for card in section.get('cards',[]):
            i=len(factual);product=byid.get(card.get('productId'));variant=next((v for v in (product or {}).get('variants',[]) if v['variantId']==card.get('variantId')),None)
            if not variant:errors.append('Unknown actual product/variant');continue
            req((card.get('originalName'),card.get('selectedColor'),card.get('itemPrice'))==(product['name'],variant['colorName'],product['itemPrice']),'Actual canonical name/color/price mismatch')
            budget=request_checked['itemBudget'];req(budget is None or (product['itemPrice']<budget if policy['budgetExclusive'] else product['itemPrice']<=budget),'Actual item price exceeds hard budget')
            req(all(r['attribute']!='color' or r['value']==variant['colorName'] for r in request_checked['hardRequirements']),'Actual selected variant violates typed hard color')
            req(i<len(meta) and meta[i].get('variantId')==variant['variantId'] and meta[i].get('image')==variant['images'][0] and rank.get('variants',{}).get(product['productId'])==variant['variantId'],'Actual rank image/variant action binding differs')
            facts={'name':product['name'],'category':product['category'],'itemPrice':product['itemPrice'],'selectedColor':variant['colorName']}
            for prefix,key,kind in [('T','title','title'),('B','badge','badge'),('C','bodyCopy','copy')]:
                text=card.get(key);values[prefix+str(i+1)]=text;req(text is not None and not field_rejections(kind,text,facts),'Withheld/nonfactual card '+key)
            factual.append({'productId':product['productId'],'variantId':variant['variantId'],'itemPrice':product['itemPrice'],'sourcePrimaryImage':variant['images'][0]['src']})
    source=complete.get('copySource',{});generated=server_receipt.get('parallelGeneration',{});fields=generated.get('fields',[]);expected_fields=slot_fields(cards)
    if cards:
        req(source.get('kind')=='fresh-parallel-head' and source.get('generatedEarlier') is False and source.get('freshPerRequestGeneration') is True and source.get('feedforwardCalls')==generated.get('feedforwardCalls')==1 and source.get('checkpointSha256')==evidence['copyCheckpointSha256'],'Fresh current expanded neural wordhead/checkpoint proof missing')
        req(len(fields)==len(expected_fields) and generated.get('teacherStringsRead') is False,'Actual perfield slot generation receipt missing or teacher lookup')
        for (field,kind,card),observed in zip(expected_fields,fields):
            try:decoded=decode_ids(observed.get('slotIds'),evidence['vocabulary']);req(decoded.get('text')==observed.get('text')==values.get(field),'Actual wordIDs/vocabulary/serialized fresh text differ: '+field)
            except Exception as error:errors.append('Invalid actual slotIDs '+field+': '+str(error))
    else:req(source.get('kind')=='no-generation' and source.get('noEligibleProducts') is True and source.get('freshPerRequestGeneration') is False,'Empty result must explicitly abstain from generation')
    validation=complete.get('validation',{});accepted=sum(x is not None for x in values.values());req(complete.get('requiredFields')==validation.get('requiredFields')==len(values) and complete.get('acceptedFields')==accepted and complete.get('withheldFields')==len(values)-accepted,'Actual field counts mismatch')
    req({r.get('field') for r in validation.get('fields',[])}==set(values) and all((r.get('accepted') is True)==(values.get(r.get('field')) is not None) for r in validation.get('fields',[])),'Actual perfield validation/null identity differs')
    q=d.get('queryFeature',{});enc=d.get('queryEncoding',{});hit=q.get('cacheHit');req(type(hit) is bool and q.get('nativeBackboneForwardCalls')==(0 if hit else 1),'Actual exactquery cache/native counters contradictory')
    req(enc.get('queryEncodingMode')=='ordinary' and enc.get('queryFeatureCacheMode')=='lru32' and enc.get('queryFeatureCacheHit') is hit and enc.get('noScoreOrRankingOrAnswerCache') is True and enc.get('reusedPrefixTokens')==0,'Explicit ordinary queryfeatureLRU only; failedprefix/answercache forbidden')
    from .store_demo_engine import ranking_input
    from .store_function_adaptation import request_text
    from .store_query_prefix_reuse_v1 import hashed
    text=request_text(ranking_input(request_checked));text_digest=hashlib.sha256(text.encode()).hexdigest()
    req(enc.get('originalFormatterTextSha256')==enc.get('wholeTextSha256')==text_digest and enc.get('actualEncodingText')==text,'Original ordinary complete query text binding differs')
    ids=enc.get('fullTokenIds',[]);req(enc.get('wholeTokenSha256')==enc.get('originalFormatterTokenSha256')==hashed(ids) and enc.get('wholeRequestTokens')==enc.get('originalFormatterTokens')==len(ids),'Actual whole query token identity/SHA/count differs')
    if evidence.get('tokenizer') is not None:req(ids==evidence['tokenizer'].encode(text,add_special_tokens=False),'Actual tokenizer whole query IDs differ')
    req(enc.get('paddedTokensPerRow')==(0 if hit else len(ids)) and enc.get('freshSuffixTokens')==(0 if hit else len(ids)),'Actual cached/fresh query token accounting differs')
    n=d.get('rankDecision',{});req(n.get('freshLearnedDecision') is True and n.get('copyAdapterParticipated') is False and n.get('generatedContinuationTokens')==n.get('optimizerUpdates')==0,'Actual fresh family decision/adapter isolation proof missing')
    return {'passed':not errors,'errors':errors,'requestId':rid,'actualCards':len(cards),'eligibleCount':len(order),'shortfall':max(0,6-len(cards)),'requiredFields':len(values),'acceptedFields':accepted,'declaredShortHeadingAbstentions':withheld,'allAvailableCardFieldsAccepted':len(factual)==len(cards) and all(v is not None for k,v in values.items() if not k.startswith('S')),'all20Accepted':len(cards)==6 and accepted==20,'bindings':factual,'queryCacheHit':hit,'queryNativeForwards':q.get('nativeBackboneForwardCalls'),'rankDecision':n,'serverTimings':{e['type']:e.get('timing') for e in events},'qualityScope':'source factuality, strict public guards and serving freshness only; no heldout relevance claim'}
