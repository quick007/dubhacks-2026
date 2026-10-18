"""Bounded sequential HTTP semantics validation for an accepted persona head.

Default: create a CPU-only plan; zero HTTP requests. Live requests to local8767
require --run, a reviewed plan SHA, and exactly one explicit serving receipt:
--candidate-receipt for offline-passed/serving-pending validation, or
--accepted-profile after acceptance. This probe never writes acceptance state.
Uses only32 historical v5 PUBLIC query wordings and10 PUBLIC demo requests.
This is development replay/serving validation, not a new quality or UI benchmark.
No context vector is supplied by this client: the actual server must acquire
features for each exact query-conditioned request and disclose its native calls.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
VERSION='persona-http-semantic-probe.v1'
FIXED_CASES=320
MAX_BODY_BYTES=2*1024*1024
MAX_LINE_BYTES=1024*1024
SOURCE_NAMES=('store_persona_http_semantic_probe_v1.py','store_persona_context_v1.py',
    'store_persona_runtime_v1.py','store_persona_engine_v1.py','store_persona_qwen_v1.py',
    'store_demo_engine.py','store_catalog_v1.py','store_function_adaptation.py',
    'store_parallel_text_data_v1.py')


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def pins_current(pins):
    if any(sha(path)!=checksum for path,checksum in pins.items()):raise ValueError('Frozen probe/public/source file changed')


def public_catalog():
    from .store_catalog_v1 import load_snapshot
    snapshot=load_snapshot();products=[]
    for product,external in zip(snapshot['inputs']['products'],snapshot['externalMapping']['products'],strict=True):
        variants=[{'variantId':route['variantId'],'colorName':color['colorName'],'images':deepcopy(route['images'])}
            for color,route in zip(product['colorVariants'],external['colorVariants'],strict=True)]
        products.append({**deepcopy(product),'productId':external['productId'],'variants':variants})
    return products


def historical_queries(packet_path):
    from .store_demo_engine import checked_request,ranking_input
    from .store_function_adaptation import request_text
    packet=json.loads(Path(packet_path).read_text());queries=[]
    for row in packet['records']:
        if row['role']!='request' or row['split']!='evaluation':continue
        payload=json.loads(row['text'].split('\n',1)[1]);public=payload['request']
        if len(public['preferences'])!=1:raise ValueError('Exact historical public query shape required')
        query=public['preferences'][0]
        request,_=checked_request({'query':query,'itemBudget':public['itemBudget'],'hardRequirements':public['hardRequirements']})
        if request_text(ranking_input(request))!=row['text'] or digest(row['text'])!=row['textSha256']:
            raise ValueError('Historical public query text binding changed')
        queries.append({'caseId':row['caseId'],'query':query,'originalNativeQueryTextSha256':row['textSha256']})
    if len(queries)!=32 or len({q['caseId'] for q in queries})!=32:raise ValueError('Exactly32 historical public v5 queries required')
    return queries


def build_plan(packet_path,demos_path):
    from .store_persona_context_v1 import checked_context_request
    from .store_demo_engine import ranking_input
    from .store_function_adaptation import request_text
    from .store_persona_qwen_v1 import public_context_text
    products=public_catalog();queries=historical_queries(packet_path)
    demos=json.loads(Path(demos_path).read_text())['people']
    if len(demos)!=10 or len({p['id'] for p in demos})!=10:raise ValueError('Exactly10 public demo profiles required')
    cases=[]
    for query in queries:
        for person in demos:
            request=deepcopy(person['request']);request['query']=query['query']
            effective,policy,_=checked_context_request(request,products)
            cases.append({'probeCaseId':query['caseId']+'--'+person['id'],'queryCaseId':query['caseId'],
                'publicDemoId':person['id'],'request':request,'effectiveRequest':effective,'effectivePolicy':policy,
                'originalNativeQueryTextSha256':query['originalNativeQueryTextSha256'],
                'expectedServingQueryTextSha256':digest(request_text(ranking_input(effective))),
                'expectedServingContextTextSha256':digest(public_context_text(request))})
    if len(cases)!=FIXED_CASES or any(re.fullmatch('[A-Za-z0-9_-]+',c['probeCaseId']) is None for c in cases):raise ValueError('Fixed32by10 safe unique case roster required')
    pins={str((HERE/name).resolve()):sha(HERE/name) for name in SOURCE_NAMES}
    pins.update({str(Path(packet_path).resolve()):sha(packet_path),str(Path(demos_path).resolve()):sha(demos_path),
        str(HERE/'store_catalog_v1/runtime_export.json'):sha(HERE/'store_catalog_v1/runtime_export.json')})
    return {'version':VERSION,'scope':'32 previously inspected v5 public query wordings by10 public demo contexts; DEVELOPMENT REPLAY; no private judgments or unseen generalization claim',
        'fixedDenominator':FIXED_CASES,'historicalQueries':32,'publicDemoProfiles':10,'cases':cases,'products':products,
        'sourceSha256':pins,'publicOnly':True,'privateGoldParsed':False,'nativeOrHttpExecutedByPreparation':False,
        'execution':'sequential actual POSTs; no client feature injection, no retries, no browser/checkout/image fetch',
        'fieldPolicy':'20 expected only with6 eligible cards; preserve legitimate shortfalls, withheld fields and zero eligible cases',
        'timingScope':'client validation HTTP exchange including incremental raw-write overhead; not UI paint/image readiness or a throughput benchmark'}


def scores(value,label):
    if type(value) is not list or len(value)!=16 or any(type(v) not in (int,float) or not math.isfinite(v) for v in value):
        raise ValueError('Finite16 '+label+' scores required')
    return value


def analyze_events(case,events,headers,products,expected,seen_request_ids):
    """Validate actual payloads with source facts, no target/teacher comparison."""
    import torch
    from .store_demo_engine import guarded_order
    from .store_persona_context_v1 import checked_context_request
    from .store_persona_runtime_v1 import fuse_scores
    from .store_parallel_text_data_v1 import field_rejections
    errors=[];quality=[]
    def require(condition,message):
        if not condition:errors.append(message)
    request_id=headers.get('x-request-id')
    require(type(request_id) is str and bool(request_id),'missing actual X-Request-Id')
    require(request_id not in seen_request_ids,'duplicate actual request ID')
    if request_id:seen_request_ids.add(request_id)
    require(bool(events) and all(e.get('requestId')==request_id for e in events),'event/header request IDs differ')
    require(all(e.get('type') in ('rank','firstThree','complete','error') for e in events),'unexpected event kind')
    remote_errors=[e for e in events if e.get('type')=='error']
    require(not remote_errors,'server emitted error event')
    rank_rows=[e for e in events if e.get('type')=='rank'];complete_rows=[e for e in events if e.get('type')=='complete']
    require(len(rank_rows)==1 and len(complete_rows)==1,'need exactly one actual rank and complete event')
    if len(rank_rows)!=1 or len(complete_rows)!=1:
        return {'protocolPassed':False,'qualityPassed':False,'errors':errors,'remoteErrors':remote_errors,'requestId':request_id,'scoreComparisonAvailable':False}
    rank=rank_rows[0];complete=complete_rows[0]
    require(events[0].get('type')=='rank' and events[-1].get('type')=='complete','rank/complete stream ordering differs')
    provenance=complete.get('provenance',{})
    require(all(e.get('provenance')==provenance for e in events),'event provenance changed within request')
    decision=provenance.get('personaRanking',{})
    require(decision.get('personaCheckpointSha256')==expected['personaHeadSha256'],'persona head differs from explicit serving receipt')
    require(provenance.get('rankCheckpointSha256')==expected['v5HeadSha256'],'v5 head differs from pinned serving export')
    require(decision.get('queryTextSha256')==case['expectedServingQueryTextSha256'],'query-conditioned native text hash mismatch')
    context=decision.get('contextFeature',{});query=decision.get('queryFeature',{})
    require(context.get('publicContextTextSha256')==case['expectedServingContextTextSha256'],'context feature is not bound to exact query-conditioned public request')
    query_calls=query.get('nativeBackboneForwardCalls');context_calls=context.get('nativeBackboneForwardCalls')
    for label,feature,count in (('query',query,query_calls),('context',context,context_calls)):
        require(type(feature.get('cacheHit')) is bool and type(count) is int and count in (0,1),label+' cache/native counters missing or invalid')
        require((feature.get('cacheHit') is True and count==0) or (feature.get('cacheHit') is False and count==1),label+' cache flag/native forward count disagree')
    require(context.get('kind') in ('precomputed-public-demo-feature','exact-last-context-feature','fresh-native-public-context'),'unexpected native context source kind')
    require((context.get('kind')=='fresh-native-public-context')==(context.get('cacheHit') is False),'context source kind/cache flag disagree')
    encoding=provenance.get('featureEncoding',{})
    require(encoding.get('queryNativeForwards')==query_calls and encoding.get('contextNativeForwards')==context_calls,'aggregate native counters differ from actual feature receipts')
    if type(query_calls) is int and type(context_calls) is int:require(encoding.get('totalNativeForwards')==query_calls+context_calls,'aggregate native total differs')
    current,policy,_=checked_context_request(case['request'],products)
    require(current==case['effectiveRequest'] and policy==case['effectivePolicy'],'effective hard guard preparation changed')
    v5=scores(decision.get('v5Scores16'),'unmodified v5');persona=scores(decision.get('personaScores16'),'persona');fused=scores(decision.get('scores16'),'fused')
    recomputed,broad=fuse_scores(torch.tensor(v5),torch.tensor(persona),current['query'])
    require(torch.allclose(recomputed,torch.tensor(fused),rtol=1e-6,atol=1e-7),'fused scores differ from frozen composition policy')
    fused_order,guard_checks=guarded_order(fused,current,policy,products);v5_order,_=guarded_order(v5,current,policy,products)
    ids=[p['productId'] for p in products];expected_ids=[ids[i] for i in fused_order]
    require(rank.get('productIds')==expected_ids,'actual learned product ordering violates effective guards/composition')
    require(rank.get('eligibleCount')==len(fused_order) and complete.get('eligibleCount')==len(fused_order),'eligible count differs under same guards')
    full_cards=[c for section in complete.get('sections',[]) for c in section.get('cards',[])]
    metadata_cards=[c for section in rank.get('sections',[]) for c in section.get('cards',[])]
    wanted=expected_ids[:6]
    require([c.get('productId') for c in full_cards]==wanted,'complete card IDs differ from real selected roster')
    require(rank.get('selectedProductIds')==wanted and [c.get('productId') for c in metadata_cards]==wanted,'rank action roster differs')
    require(complete.get('shortfall')==max(0,6-len(wanted)),'shortfall differs from actual eligible roster')
    require(len(set(wanted))==len(wanted),'duplicate selected products')
    require([len(s.get('cards',[])) for s in complete.get('sections',[])]==[min(3,len(wanted)-start) for start in range(0,len(wanted),3)],'section grouping differs from selected card roster')
    expected_kinds=['rank','firstThree','complete'] if full_cards else ['rank','complete']
    require([e.get('type') for e in events]==expected_kinds,'actual event roster/order differs')
    if full_cards:
        first=next((e for e in events if e.get('type')=='firstThree'),{})
        require(first.get('sections')==complete.get('sections',[])[:1],'firstThree payload differs from complete first section')
        require(first.get('copySource')==complete.get('copySource'),'copy source changed between firstThree and complete')
        first_count=min(3,len(full_cards))*3+1
        require(first.get('requiredFields')==first_count,'firstThree required field count differs')
    by_id={p['productId']:p for p in products};metadata_by_id={c.get('productId'):c for c in metadata_cards};binding_rows=[]
    for card in full_cards:
        product=by_id.get(card.get('productId'));metadata=metadata_by_id.get(card.get('productId'),{})
        if product is None:quality.append('unknown product');continue
        variant=next((v for v in product['variants'] if v['variantId']==card.get('variantId')),None)
        if variant is None:quality.append('variant not owned by product');continue
        require(rank.get('variants',{}).get(product['productId'])==variant['variantId'],'rank/complete variant mismatch')
        facts={'name':product['name'],'category':product['category'],'itemPrice':product['itemPrice'],'selectedColor':variant['colorName']}
        if (card.get('originalName')!=facts['name'] or card.get('itemPrice')!=facts['itemPrice'] or card.get('selectedColor')!=facts['selectedColor']):quality.append('canonical product/color/price facts differ')
        if any(r['attribute']=='color' and r['value']!=variant['colorName'] for r in current['hardRequirements']):quality.append('chosen variant violates hard color')
        if (metadata.get('originalName')!=facts['name'] or metadata.get('itemPrice')!=facts['itemPrice'] or metadata.get('selectedColor')!=facts['selectedColor'] or metadata.get('variantId')!=variant['variantId']):quality.append('rank metadata canonical facts/variant differ')
        image_ok=metadata.get('image')==variant['images'][0]
        if not image_ok:quality.append('image is not canonical first gallery binding of chosen real variant')
        if any(card.get(k) is not None for k in ('stock','size','shipping')):quality.append('unsupported availability/size/shipping claim')
        for key,kind in (('title','title'),('badge','badge'),('bodyCopy','copy')):
            if card.get(key) is None:quality.append('withheld required card field:'+key)
            else:
                reasons=field_rejections(kind,card[key],facts)
                if reasons:quality.append(key+':'+','.join(reasons))
        binding_rows.append({'productId':product['productId'],'variantId':variant['variantId'],'selectedColor':variant['colorName'],'canonicalImageBindingVerified':image_ok})
    for section in complete.get('sections',[]):
        if section.get('heading') is None and len(section.get('cards',[]))==3:quality.append('withheld full-group heading')
        if section.get('heading') is not None:
            reasons=field_rejections('section',section['heading'])
            if reasons:quality.append('heading:'+','.join(reasons))
    validation=complete.get('validation',{});field_rows=validation.get('fields',[])
    observed_required=len(full_cards)*3+len(complete.get('sections',[]))
    field_values={}
    for section_index,section in enumerate(complete.get('sections',[])):
        field_values['S'+str(section_index+1)]=section.get('heading')
        for local_index,card in enumerate(section.get('cards',[])):
            index=section_index*3+local_index+1
            for prefix,key in (('T','title'),('B','badge'),('C','bodyCopy')):field_values[prefix+str(index)]=card.get(key)
    require(len({r.get('field') for r in field_rows})==len(field_rows) and {r.get('field') for r in field_rows}==set(field_values),'validation field identities differ or repeat')
    require(all((row.get('accepted') is True)==(field_values.get(row.get('field')) is not None) for row in field_rows),'accepted/withheld flags disagree with actual text/null values')
    require(validation.get('requiredFields')==observed_required and len(field_rows)==observed_required,'validation field roster/count differs from actual payload')
    accepted=sum(row.get('accepted') is True for row in field_rows);withheld=observed_required-accepted
    require(validation.get('acceptedFields')==accepted and validation.get('withheldFields')==withheld,'validation accepted/withheld counts differ')
    require(all(complete.get(k)==validation.get(k) for k in ('requiredFields','acceptedFields','withheldFields')),'top-level/validation counts disagree')
    source=complete.get('copySource',{})
    if full_cards:
        require(source.get('kind')=='fresh-parallel-head' and source.get('generatedEarlier') is False and source.get('freshPerRequestGeneration') is True,'copy did not use fresh neural generation')
        require(source.get('checkpointSha256')==expected['copyHeadSha256'],'copy head differs from pinned ALL48 serving manifest')
    else:require(source.get('kind')=='no-generation' and source.get('noEligibleProducts') is True and observed_required==0,'zero-eligible generation/field receipt differs')
    six_eligible=len(fused_order)>=6
    if six_eligible and (len(full_cards)!=6 or observed_required!=20 or accepted!=20 or withheld!=0):quality.append('six eligible products did not yield all20 useful validated fields')
    identity=guard_checks['exactNamePolicy'];identity_ok=True
    if identity.get('matchedProductIndices'):
        identity_ok=(fused_order[:1] and fused_order[0] in identity['matchedProductIndices']) if identity.get('applied') else not fused_order
        if not identity_ok:quality.append('eligible exact name identity or ineligible exact-name abstention violated')
    return {'protocolPassed':not errors,'qualityPassed':not errors and not quality,'errors':errors,'qualityFailures':quality,
        'requestId':request_id,'scoreComparisonAvailable':True,'fusedTop1':ids[fused_order[0]] if fused_order else None,
        'v5Top1UnderSameEffectiveGuards':ids[v5_order[0]] if v5_order else None,'sameV5Top1UnderSameGuards':fused_order[:1]==v5_order[:1],
        'bothAbstainUnderSameGuards':not fused_order and not v5_order,'bothNonemptyUnderSameGuards':bool(fused_order and v5_order),
        'broadPublicBrowse':broad,'eligibleCount':len(fused_order),'actualCards':len(full_cards),'shortfall':max(0,6-len(full_cards)),
        'expected20Fields':six_eligible,'requiredFields':observed_required,'acceptedFields':accepted,'withheldFields':withheld,
        'zeroEligible':not fused_order,'exactNameCheckApplicable':bool(identity.get('matchedProductIndices')),'exactNameCompliant':bool(identity_ok),
        'queryFeature':deepcopy(query),'contextFeature':deepcopy(context),'featureEncoding':deepcopy(encoding),'bindings':binding_rows,
        'serverTiming':complete.get('timing'), 'imageScope':'declared variant/gallery metadata only; no image HTTP fetch, pixel inspection or paint claim'}


class WallDeadline(TimeoutError):pass
@contextmanager
def deadline(seconds):
    if signal.getitimer(signal.ITIMER_REAL)[0]:raise RuntimeError('Standalone probe refuses nested active wall alarm')
    previous=signal.getsignal(signal.SIGALRM)
    def expired(_signum,_frame):raise WallDeadline('Bounded serving-probe lifecycle exceeded')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def exchange(opener,url,*,value=None,raw_path=None,timeout=10):
    headers={'Origin':'http://127.0.0.1:4313'};data=None
    if value is not None:data=canonical(value).encode();headers['Content-Type']='application/json'
    req=Request(url,data=data,headers=headers,method='POST' if value is not None else 'GET')
    response=None;body=bytearray();response_headers={};status=None;started=time.perf_counter()
    try:
        try:response=opener.open(req,timeout=timeout)
        except HTTPError as error:response=error
        status=response.status;response_headers={k.casefold():v for k,v in response.headers.items()}
        while True:
            line=response.readline(MAX_LINE_BYTES+1)
            if not line:break
            body.extend(line)
            if raw_path:Path(raw_path).write_bytes(body)
            if len(line)>MAX_LINE_BYTES or len(body)>MAX_BODY_BYTES:raise ValueError('Bounded raw response size exceeded')
        return {'httpStatus':status,'headers':response_headers,'rawBytes':bytes(body),'httpElapsedMs':(time.perf_counter()-started)*1000}
    except BaseException as error:
        if raw_path:Path(raw_path).write_bytes(body)
        error.http_partial={'httpStatus':status,'headers':response_headers,'rawBytes':len(body),'httpElapsedMs':(time.perf_counter()-started)*1000}
        raise
    finally:
        if response is not None:response.close()


def serving_selection(*,accepted_profile=None,candidate_receipt=None):
    """Read explicit root receipts; never create or promote an accepted profile."""
    if bool(accepted_profile)==bool(candidate_receipt):
        raise ValueError('Exactly one accepted profile or candidate receipt is required')
    if accepted_profile:
        path=Path(accepted_profile).resolve();profile=json.loads(path.read_text())
        if profile.get('version')!='personalized-local-profile.v1' or profile.get('status')!='accepted':
            raise ValueError('Explicit accepted public serving profile required')
        head=profile.get('personaHead',{}).get('sha256')
        if type(head) is not str or re.fullmatch('[0-9a-f]{64}',head) is None:raise ValueError('Accepted profile must name exact persona head SHA')
        return {'mode':'accepted-profile','personaHeadSha256':head,'receiptFile':str(path),
            'sourceSha256':{str(path):sha(path)},'acceptanceStateWritten':False}
    path=Path(candidate_receipt).resolve();receipt=json.loads(path.read_text())
    if receipt.get('version')!='persona-serving-candidate.v1' or receipt.get('status')!='offline-passed-serving-pending':
        raise ValueError('Explicit offline-passed/serving-pending candidate receipt required')
    head=receipt.get('personaHeadSha256');audit_sha=receipt.get('offlineAuditSha256');audit_name=receipt.get('offlineAuditFile')
    if any(type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None for value in (head,audit_sha)) or type(audit_name) is not str or not audit_name.strip():
        raise ValueError('Candidate receipt requires exact head/audit SHA and explicit offlineAuditFile')
    audit_path=Path(audit_name)
    if not audit_path.is_absolute():audit_path=path.parent/audit_path
    audit_path=audit_path.resolve()
    if sha(audit_path)!=audit_sha:raise ValueError('Candidate offline audit checksum mismatch')
    audit=json.loads(audit_path.read_text())
    if (audit.get('version')!='persona-qwen-independent-adjudicator.v1' or
            audit.get('status')!='offline-gate-passes-serving-pending' or audit.get('offlineGatePassed') is not True or
            audit.get('overallPromotionPassed') is not False):
        raise ValueError('Actual independent adjudicator must pass offline gate with serving acceptance pending')
    evidence=audit.get('evidence',{})
    if evidence.get('checkpointSha256')!=head:raise ValueError('Candidate head differs from actual offline adjudication')
    freeze_path=HERE/'PERSONA_QWEN_ADJUDICATOR_FROZEN_V1.json';gate_path=HERE/'PERSONA_QWEN_PROMOTION_GATE_V1.json'
    freeze=json.loads(freeze_path.read_text());gate=json.loads(gate_path.read_text())
    if (freeze.get('version')!='persona-qwen-adjudicator-freeze.v1' or freeze.get('status')!='frozen-before-outcomes' or
            evidence.get('adjudicatorFreezeSha256')!=sha(freeze_path) or audit.get('gateSha256')!=sha(gate_path) or
            freeze.get('gateSha256')!=sha(gate_path)):
        raise ValueError('Offline adjudicator/gate freeze provenance mismatch')
    pins={str(path):sha(path),str(audit_path):audit_sha,str(freeze_path):sha(freeze_path),str(gate_path):sha(gate_path)}
    for name,expected in freeze['sourceSha256'].items():
        source=HERE/name
        if sha(source)!=expected:raise ValueError('Frozen offline adjudicator source changed')
        pins[str(source.resolve())]=expected
    if evidence.get('adjudicatorSourceSha256')!=freeze['sourceSha256']['audit_persona_qwen_candidate_v1.py'] or evidence.get('sourceAndFeaturePinsVerified') is not True:
        raise ValueError('Offline audit must be tied to the actual frozen adjudicator and verified source/features')
    required_checks={'primary20','novel24','strongTop3_28','zeroFactConstraintViolations','devOwnedDemotions8','devNovelAfter8'}
    required_checks.update('primaryStrictlyExceeds_'+name for name in gate['freshIndependent32']['primaryTop1StrictlyExceeds'])
    checks=audit.get('checks',{})
    if set(checks)!=required_checks or any(value is not True for value in checks.values()):raise ValueError('Actual offline adjudicator checks are incomplete or failed')
    serving=audit.get('servingAcceptance',{})
    if (set(serving)!=set(gate['servingAcceptance']) or any(type(row) is not dict or row.get('status')!='pending' or
            row.get('required')!=gate['servingAcceptance'][name] for name,row in serving.items())):
        raise ValueError('Candidate adjudication must keep all declared serving checks pending')
    return {'mode':'candidate-offline-passed-serving-pending','personaHeadSha256':head,'receiptFile':str(path),
        'offlineAuditFile':str(audit_path),'offlineAuditSha256':audit_sha,'sourceSha256':pins,
        'offlineGatePassed':True,'overallPromotionPassed':False,'acceptanceStateWritten':False}


def run(plan,out,accepted_profile,rank_export,copy_manifest,base_url,max_seconds=240,request_timeout=10,*,candidate_receipt=None):
    parsed=urlsplit(base_url)
    if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost') or parsed.port!=8767 or parsed.path not in ('','/') or parsed.query or parsed.fragment or parsed.username:
        raise ValueError('Explicit local8767 base URL required; main8766 and external URLs forbidden')
    if plan['version']!=VERSION or len(plan['cases'])!=FIXED_CASES or len({c['probeCaseId'] for c in plan['cases']})!=FIXED_CASES or any(re.fullmatch('[A-Za-z0-9_-]+',c['probeCaseId']) is None for c in plan['cases']):raise ValueError('Exact frozen320 safe unique public plan required')
    if not 0<max_seconds<=600 or not 0<request_timeout<=30:raise ValueError('Bounded positive lifecycle/request timeouts required')
    selection=serving_selection(accepted_profile=accepted_profile,candidate_receipt=candidate_receipt)
    expected={'personaHeadSha256':selection['personaHeadSha256'],
        'v5HeadSha256':json.loads(Path(rank_export).read_text())['head']['sha256'],
        'copyHeadSha256':json.loads(Path(copy_manifest).read_text())['headSha256']}
    if any(re.fullmatch('[0-9a-f]{64}',x) is None for x in expected.values()):raise ValueError('Exact reviewed checkpoint hashes required')
    pins={**plan['sourceSha256'],**selection['sourceSha256'],str(Path(rank_export).resolve()):sha(rank_export),str(Path(copy_manifest).resolve()):sha(copy_manifest)}
    pins_current(pins);out=Path(out)
    if not out.is_absolute() or not out.resolve().is_relative_to(ROOT):raise ValueError('Explicit absolute output inside current code repository required')
    out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan);write(out/'expected-sources.json',{'checkpointSha256':expected,'filesSha256':pins,'servingSelection':selection})
    report={'version':VERSION,'status':'running','startedAt':utc(),'fixedDenominator':FIXED_CASES,'scope':plan['scope'],
        'execution':plan['execution'],'timingScope':plan['timingScope'],'servingSelection':selection,'acceptanceStateWritten':False,'cases':[{'probeCaseId':c['probeCaseId'],'status':'unattempted'} for c in plan['cases']],
        'privateGoldParsed':False,'browserOrCheckoutUsed':False,'nativeCallsMadeByClient':0,'httpRequestAttempts':0,'postRequestAttempts':0,'healthRequestAttempts':0,'completeHttpResponses':0}
    seen=set();started=time.perf_counter();opener=build_opener(ProxyHandler({}));active=None
    def save():write(out/'results.json',report)
    save()
    try:
        with deadline(max_seconds):
            report['httpRequestAttempts']+=1;report['healthRequestAttempts']+=1
            health=exchange(opener,base_url.rstrip('/')+'/health',timeout=min(request_timeout,max_seconds),raw_path=out/'health.json')
            report['completeHttpResponses']+=1
            health_value=json.loads(health['rawBytes'])
            if health['httpStatus']!=200 or health_value.get('ready') is not True:raise RuntimeError('Selected persona service is not ready')
            startup=health_value['startup']
            if startup['persona']['checkpointSha256']!=expected['personaHeadSha256'] or startup['rankCheckpointSha256']!=expected['v5HeadSha256'] or startup['copyCandidate']['checkpointSha256']!=expected['copyHeadSha256']:
                raise ValueError('Actual ready service does not match explicitly selected persona/v5/copy checkpoints')
            for case,entry in zip(plan['cases'],report['cases'],strict=True):
                pins_current(pins);active=entry;entry.update(status='running',startedAt=utc());save()
                directory=out/'cases'/case['probeCaseId'];directory.mkdir(parents=True,exist_ok=False);write(directory/'input.json',case['request'])
                try:
                    report['httpRequestAttempts']+=1;report['postRequestAttempts']+=1;entry['dispatchOutcome']='attempted-unconfirmed'
                    raw=exchange(opener,base_url.rstrip('/')+'/personalize',value=case['request'],raw_path=directory/'raw.ndjson',timeout=min(request_timeout,max_seconds-(time.perf_counter()-started)))
                    report['completeHttpResponses']+=1;entry['dispatchOutcome']='response-received'
                    write(directory/'http.json',{k:v for k,v in raw.items() if k!='rawBytes'})
                    if raw['httpStatus']!=200:raise RuntimeError('Actual POST returned HTTP'+str(raw['httpStatus']))
                    previous_seen=set(seen);actual_id=raw['headers'].get('x-request-id');entry['actualRequestId']=actual_id;entry['duplicateActualRequestId']=bool(actual_id and actual_id in seen)
                    if actual_id:seen.add(actual_id)
                    events=[json.loads(line) for line in raw['rawBytes'].splitlines() if line.strip()];write(directory/'events.json',events)
                    analysis=analyze_events(case,events,raw['headers'],plan['products'],expected,previous_seen)
                    write(directory/'analysis.json',analysis);entry.update(status='completed' if analysis['qualityPassed'] else 'failed-validation',analysis=analysis,httpElapsedMs=raw['httpElapsedMs'])
                except BaseException as error:
                    entry.update(status='timeout' if isinstance(error,(TimeoutError,WallDeadline)) else 'failed',error={'type':type(error).__name__,'message':str(error)},partialHttp=getattr(error,'http_partial',None))
                    partial=getattr(error,'http_partial',{}) or {}
                    if partial.get('httpStatus') is not None and entry.get('dispatchOutcome')!='response-received':entry['dispatchOutcome']='response-started-incomplete'
                    partial_id=partial.get('headers',{}).get('x-request-id')
                    if partial_id and not entry.get('actualRequestId'):
                        entry['actualRequestId']=partial_id;entry['duplicateActualRequestId']=partial_id in seen;seen.add(partial_id)
                    write(directory/'failure.json',entry)
                    if isinstance(error,(WallDeadline,KeyboardInterrupt,SystemExit)):raise
                finally:entry['finishedAt']=utc();save()
                active=None
            pins_current(pins);report['status']='completed-fixed320-validation'
    except BaseException as error:
        report.update(status='stopped',error={'type':type(error).__name__,'message':str(error)})
        if active and active['status']=='running':active.update(status='timeout' if isinstance(error,WallDeadline) else 'failed',error=report['error'])
    finally:
        analyses=[c['analysis'] for c in report['cases'] if 'analysis' in c];comparisons=[a for a in analyses if a.get('scoreComparisonAvailable')]
        counter_rows=[a for a in analyses if type(a.get('queryFeature',{}).get('nativeBackboneForwardCalls')) is int and type(a.get('contextFeature',{}).get('nativeBackboneForwardCalls')) is int]
        report['summary']={'attempted':sum(c['status']!='unattempted' for c in report['cases']),'unattempted':sum(c['status']=='unattempted' for c in report['cases']),
            'completed':sum(c['status']=='completed' for c in report['cases']),'failedOrTimeout':sum(c['status'] not in ('completed','unattempted') for c in report['cases']),
            'sameGuardV5ComparisonsAvailable':len(comparisons),'sameGuardV5Top1Agreements':sum(a['sameV5Top1UnderSameGuards'] for a in comparisons),
            'bothAbstainComparisons':sum(a['bothAbstainUnderSameGuards'] for a in comparisons),
            'bothNonemptyComparisons':sum(a['bothNonemptyUnderSameGuards'] for a in comparisons),
            'bothNonemptyV5Agreements':sum(a['bothNonemptyUnderSameGuards'] and a['sameV5Top1UnderSameGuards'] for a in comparisons),
            'fixed320V5AgreementNumerator':sum(a['sameV5Top1UnderSameGuards'] for a in comparisons),'fixedDenominator':FIXED_CASES,
            'zeroEligibleCases':sum(a.get('zeroEligible',False) for a in analyses),'sixEligibleCases':sum(a.get('expected20Fields',False) for a in analyses),
            'sixEligibleAll20Accepted':sum(a.get('expected20Fields',False) and a.get('acceptedFields')==20 and a.get('withheldFields')==0 for a in analyses),
            'protocolPassed':sum(a['protocolPassed'] for a in analyses),'uniqueActualRequestIds':len(seen),
            'casesWithReportedNativeCounters':len(counter_rows),
            'serverReportedQueryNativeForwards':sum(a['queryFeature']['nativeBackboneForwardCalls'] for a in counter_rows),
            'serverReportedContextNativeForwards':sum(a['contextFeature']['nativeBackboneForwardCalls'] for a in counter_rows),
            'reportedQueryFeatureCacheHits':sum(a['queryFeature'].get('cacheHit') is True for a in counter_rows),
            'reportedContextFeatureCacheHits':sum(a['contextFeature'].get('cacheHit') is True for a in counter_rows),
            'qualityOrBindingFailures':sum(bool(a.get('qualityFailures')) for a in analyses)}
        report['finishedAt']=utc();report['elapsedSeconds']=time.perf_counter()-started;save()
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--asset-root',type=Path,default=Path('/Users/lukas/Documents/Projects/dubhacks-2026'))
    parser.add_argument('--packet',type=Path);parser.add_argument('--demos',type=Path,default=HERE/'synthetic_history_v1/demo-people.json')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--plan',type=Path);parser.add_argument('--plan-sha256')
    receipt=parser.add_mutually_exclusive_group()
    receipt.add_argument('--accepted-profile',type=Path);receipt.add_argument('--candidate-receipt',type=Path)
    parser.add_argument('--base-url',default='http://127.0.0.1:8767')
    parser.add_argument('--max-seconds',type=float,default=240);parser.add_argument('--request-timeout',type=float,default=10);parser.add_argument('--run',action='store_true')
    args=parser.parse_args(argv)
    if args.run:
        if not args.plan or not args.plan_sha256 or not (args.accepted_profile or args.candidate_receipt):parser.error('--run requires reviewed --plan/--plan-sha256 and exactly one --candidate-receipt or --accepted-profile')
        if sha(args.plan)!=args.plan_sha256:raise ValueError('Reviewed HTTP plan checksum mismatch')
        plan=json.loads(args.plan.read_text());pins_current(plan['sourceSha256'])
        report=run(plan,args.out,args.accepted_profile,args.asset_root/'data/ranker/pretrained/actual-store-function-v5-attempt-1/serving-head.json',args.asset_root/'data/ranker/pretrained/store-parallel-text-train-all-demo-v1-attempt-2/serving-manifest.json',args.base_url,args.max_seconds,args.request_timeout,candidate_receipt=args.candidate_receipt)
        print(json.dumps({'status':report['status'],'summary':report['summary']}))
        if report['summary']['completed']!=FIXED_CASES:raise SystemExit(1)
    else:
        if not args.out.is_absolute() or not args.out.resolve().is_relative_to(ROOT):raise ValueError('Explicit absolute unique repository output required')
        plan=build_plan(args.packet or args.asset_root/'data/ranker/pretrained/actual-store-function-v5-attempt-1/packet.json',args.demos)
        args.out.mkdir(parents=True,exist_ok=False);write(args.out/'plan.json',plan)
        print(json.dumps({'status':'cpu-plan-only','httpRequests':0,'cases':len(plan['cases']),'planSha256':sha(args.out/'plan.json')}))
if __name__=='__main__':main()
