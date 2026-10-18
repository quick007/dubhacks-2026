"""Isolated adapter candidate HTTP validation; CPU prepare/no HTTP by default.

Requires actual offline admission/pass, independent TRAIN query replay and actual
query/v5/combined replay before root-selected serving-pending validation. Never
writes an accepted profile or starts a model, server, browser, checkout or optimizer.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

from . import store_persona_http_semantic_probe_v1 as old
from . import store_persona_adapter_contract_v2 as contract
from . import store_persona_adapter_runtime_v2 as runtime
from . import store_persona_combined_query_v1 as query
from . import store_qwen_lexical_choice_inputs_v2 as formatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VERSION = 'persona-adapter-http-semantic-probe.v2'
ROSTER_SHA = '39c4d2883dc58058b846d9a364fc6393d665cb6638064b128902173378cbcb1f'
FIXED_CASES = 320
INTEGRATION_MANIFEST_SHA = '3c9b3dca39d6dda8c72886f6ee6df973b7fee2c74126c0ec2800daea99433c9b'
NATIVE_COPY_SHA = '56c6c1f4199d8689f1920d871213a46d2d121fec4e5836ff70e33aaf98c7c7b8'
COPY_MANIFEST_SHA = '4bf86af880a59ffb420646bbaf24cd200b743dccb5d0cf9274190d3d0a4acfae'
SOURCE_NAMES = ('store_persona_http_semantic_probe_v2.py','store_persona_http_semantic_probe_v1.py',
    'store_persona_adapter_contract_v2.py','store_persona_adapter_runtime_v2.py','store_persona_adapter_engine_v2.py',
    'store_persona_choice_scoring_v1.py','store_persona_choice_capture_v1.py','store_qwen_lexical_choice_inputs_v2.py',
    'store_persona_context_v1.py','store_persona_qwen_v1.py','store_demo_engine.py','store_candidate_engine_v1.py',
    'store_candidate_image_service_v1.py','store_parallel_text_data_v1.py','store_query_intent_gate_v3.py',
    'store_query_intent_audit_v3.py','audit_persona_combined_query_replay_v1.py','audit_persona_choice_combined_v1.py')


def need(condition, message):
    if not condition: raise ValueError(message)


def prepare(roster_path, model_path):
    """Keep all original320 requests and original plan bytes; add exact public hashes."""
    from transformers import AutoTokenizer
    roster_path = Path(roster_path).resolve()
    need(old.sha(roster_path) == ROSTER_SHA, 'Exact original39c4d public320 plan required')
    original = json.loads(roster_path.read_text()); cases = deepcopy(original['cases'])
    need(len(cases) == 320 and len({c['probeCaseId'] for c in cases}) == 320, 'Fixed all320 unique roster required')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    catalog = formatter.load_public_catalog(); products = old.public_catalog(); expected = {}
    for case in cases:
        request, policy, _ = contract.common_request(case['request'], catalog, products)
        need(request == case['effectiveRequest'] and policy == case['effectivePolicy'], 'Original effective public guards changed')
        row = formatter.build_row(case['request'], catalog, tokenizer, 'runtime-public-request', 'canonical')
        public = {key: row[key] for key in ('prompt','tokenIds','catalogPrefixTokenIds')}
        public.update(tokenIdsSha256=contract.token_hash(public['tokenIds']), routeTokenIds=[c['tokenId'] for c in row['choices']])
        checked = contract.suffix_contract(public)
        expected[case['probeCaseId']] = {'publicRequestSha256': old.digest(old.canonical(case['request'])),
            'queryTextSha256': query.digest_text(query.runtime_text(case['request'], products)),
            'promptTokenIdsSha256': checked['promptTokenIdsSha256'], 'promptTokens': len(row['tokenIds']),
            'catalogPrefixTokenIdsSha256': contract.token_hash(checked['catalogPrefixTokenIds']),
            'routeTokenIds': checked['routeTokenIds']}
    pins = {str((HERE/name).resolve()): old.sha(HERE/name) for name in SOURCE_NAMES}
    pins[str(roster_path)] = ROSTER_SHA
    for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        path = (Path(model_path)/name).resolve(); pins[str(path)] = old.sha(path)
    return {'version': VERSION, 'originalRoster': {'file': str(roster_path), 'sha256': ROSTER_SHA},
        'cases': cases, 'expectedPublic': expected, 'products': products, 'sourceSha256': pins,
        'fixedDenominator': 320, 'publicOnly': True, 'privateGoldParsed': False, 'httpRequests': 0,
        'scope': 'same32 historical v5 public query wordings by10 public demo observations; DEVELOPMENT REPLAY',
        'execution': 'all320 sequential actual HTTP POSTs, no retries or response injection; failures/partials retained',
        'timingScope': 'server serialized rank/first3/full timings and measured client HTTP exchange; no UI paint/image readiness claim',
        'acceptanceStateWritten': False}


def artifact(record, base):
    need(type(record) is dict and set(record) == {'file','sha256'}, 'Explicit artifact file and SHA required')
    filename = Path(record['file'])
    if not filename.is_absolute(): filename = Path(base)/filename
    need(not filename.is_symlink(), 'Pinned regular artifact required')
    filename = filename.resolve()
    need(old.sha(filename) == contract.exact_sha(record['sha256']), 'Pinned candidate admission artifact changed')
    return filename


def selection(receipt_path, expected_sha):
    """Only existing audited evidence is admitted; no accepted/default state created."""
    path = Path(receipt_path).resolve()
    need(old.sha(path) == contract.exact_sha(expected_sha), 'Explicit serving-pending receipt checksum required')
    value = json.loads(path.read_text())
    need(value.get('version') == 'persona-adapter-serving-candidate.v1' and
        value.get('status') == 'offline-passed-query-mechanical-passed-serving-pending' and
        value.get('overallPromotionPass') is False, 'Explicit offline/mechanical-passed candidate with serving pending required')
    keys = ('offlineAudit','admissionBeforeGold','queryMechanicalAudit','queryCombinationReplay','adapterManifest',
        'gateCheckpoint','v5Export','copyManifest','integrationManifest','startup')
    paths = {key: artifact(value[key], path.parent) for key in keys}
    need(value['integrationManifest']['sha256'] == INTEGRATION_MANIFEST_SHA and value['copyManifest']['sha256'] == COPY_MANIFEST_SHA,
        'Explicit reviewed adapter integration and original public48 copy manifests required')
    pins = {str(path): expected_sha, **{str(p): value[key]['sha256'] for key,p in paths.items()}}
    source_pins = value.get('sourceSha256')
    need(type(source_pins) is dict and source_pins, 'Explicit server lifetime source inventory required')
    for name in SOURCE_NAMES:
        source = str((HERE/name).resolve())
        need(source_pins.get(source) == old.sha(source), 'Candidate does not bind exact runtime/probe sources: '+name)
    pins.update(source_pins)
    audit = json.loads(paths['offlineAudit'].read_text()); before = json.loads(paths['admissionBeforeGold'].read_text())
    audit_source = HERE/'audit_persona_choice_combined_v1.py'
    audit_freeze = HERE/'PERSONA_CHOICE_COMBINED_ADJUDICATOR_FROZEN_V1.json'
    frozen = json.loads(audit_freeze.read_text()); old.pins_current(frozen['sha256'])
    need(audit.get('version') == 'persona-choice-combined-independent-audit.v1' and
        audit.get('status') == 'completed-offline-adjudication' and audit.get('offlineCoreAndReplacementPass') is True and
        audit.get('overallPromotionPass') is False and audit.get('privateQualityJSONParsed') is True and audit.get('sourceSha256') == old.sha(audit_source) and
        audit.get('gateSha256') == frozen['sha256'][str((HERE/'PERSONA_CHOICE_COMBINED_PROMOTION_GATE_V2.json').resolve())] and
        before.get('privateQualityJSONParsed') is False, 'Completed independent offline pass and before-gold admission required')
    pins.update({str(audit_source): old.sha(audit_source), str(audit_freeze): old.sha(audit_freeze)})
    serving, parity = runtime.load_manifest(paths['adapterManifest'], value['adapterManifest']['sha256'])
    adapter_sha = serving['adapter']['sha256']; gate_sha = value['gateCheckpoint']['sha256']
    need(before.get('adapterCheckpointSha256') == adapter_sha, 'Offline adapter admission differs from selected serving checkpoint')
    mechanical = json.loads(paths['queryMechanicalAudit'].read_text())
    mechanical_source = HERE/'store_query_intent_audit_v3.py'
    need(mechanical.get('version') == 'query-intent-independent-TRAIN-audit.v3' and
        mechanical.get('status') == 'passed-independent-TRAIN-replay' and mechanical.get('headSha256') == gate_sha and
        mechanical.get('auditSourceSha256') == old.sha(mechanical_source) and mechanical.get('privateQualityLabelsRead') is False and
        mechanical.get('sourceManifestSha256') == runtime.INTENT_TRAIN_MANIFEST_SHA and
        mechanical.get('replay',{}).get('observedReplayOptimizerSteps') == 250 and
        mechanical.get('replay',{}).get('everyFinalParameterBitwiseMatches') is True and
        mechanical.get('twoReloadsExactLogitParity') is True and
        mechanical.get('reloadSavedParity',{}).get('rows') == 1272 and
        mechanical.get('reloadSavedParity',{}).get('exactSavedProbabilityParity') is True,
        'Actual independent query-head250-update TRAIN replay is incomplete or mismatched')
    need(gate_sha in mechanical['inputSha256'].values(), 'Mechanical audit did not pin this query head')
    pins.update(mechanical['inputSha256']); pins[str(mechanical_source)] = old.sha(mechanical_source)
    replay = json.loads(paths['queryCombinationReplay'].read_text())
    replay_source = HERE/'audit_persona_combined_query_replay_v1.py'
    export = json.loads(paths['v5Export'].read_text()); v5_sha = export['head']['sha256']
    need(replay.get('version') == 'persona-combined-query-replay-audit.v1' and replay.get('status') == 'passed' and
        replay.get('gateCheckpointSha256') == gate_sha and replay.get('v5HeadSha256') == v5_sha and
        replay.get('combinedFreezeSha256') == before.get('combinedFreezeSha256') and replay.get('rows') == 344 and
        replay.get('exactQueryProbabilityAndV5Parity') is True and replay.get('overallPromotionPass') is False and
        replay.get('servingPending') is True and replay.get('sourceSha256',{}).get(str(replay_source.resolve())) == old.sha(replay_source) and
        replay.get('adapterCheckpointSha256') == adapter_sha and replay.get('modelRowsChecked') == 1376 and
        replay.get('uniqueQueryVectorsReplayed') == 292 and replay.get('actualFloat32BlendAndGuardsExact') is True and
        replay.get('privateQualityLabelsRead') is False and replay.get('optimizerUpdates') == 0 and replay.get('nativeCalls') == 0 and
        replay.get('specific320BehavioralRegressionComputed') is False and
        replay.get('mechanicalQueryAuditSha256') == value['queryMechanicalAudit']['sha256'],
        'Independent exact query/v5/combined replay required before HTTP serving validation')
    need(replay.get('queryBundleSha256') in replay['inputSha256'].values() and
        before.get('combinedFreezeSha256') in replay['inputSha256'].values(), 'Replay did not pin exact query bundle and admitted model freeze')
    pins.update(replay['inputSha256']); pins.update(replay['sourceSha256']); pins[str(replay_source)] = old.sha(replay_source)
    copy = json.loads(paths['copyManifest'].read_text())
    need(copy.get('all48BindingsSupervised') is True and copy.get('runtimeTeacherLookup') is False,
        'Pinned public48 parallel copy manifest required')
    expected = {'adapterSha256': adapter_sha, 'adapterManifestSha256': value['adapterManifest']['sha256'],
        'gateSha256': gate_sha, 'gateSourceSha256': old.sha(HERE/'store_query_intent_gate_v3.py'),
        'v5HeadSha256': v5_sha, 'copyHeadSha256': copy['headSha256'], 'colorHeadSha256': runtime.COLOR_SHA,
        'gateTRAINManifestSha256': runtime.INTENT_TRAIN_MANIFEST_SHA,
        'rankExportSha256': value['v5Export']['sha256'], 'integrationManifestSha256': value['integrationManifest']['sha256'], 'startup': json.loads(paths['startup'].read_text())}
    old.pins_current(pins)
    return {'mode': 'offline-and-mechanical-passed-serving-pending', 'expected': expected, 'pins': pins,
        'overallPromotionPass': False, 'acceptanceStateWritten': False,
        'offlineAuditSha256': value['offlineAudit']['sha256'], 'queryMechanicalAuditSha256': value['queryMechanicalAudit']['sha256'],
        'queryCombinationReplaySha256': value['queryCombinationReplay']['sha256']}


def validate_health(value, expected):
    need(value.get('ready') is True and value.get('status') == 'ready', 'Actual adapter candidate is not ready')
    startup = value.get('startup',{})
    need(startup == expected['startup'], 'Live startup differs from exact selected lifetime receipt')
    persona = startup.get('persona',{}); intent = persona.get('queryIntent',{})
    need(persona.get('kind') == 'persona-adapter-engine.v2' and persona.get('checkpointSha256') == expected['adapterSha256'] and
        persona.get('servingManifestSha256') == expected['adapterManifestSha256'] and
        intent.get('checkpointSha256') == expected['gateSha256'] and intent.get('sourceSha256') == expected['gateSourceSha256'] and
        intent.get('sourceManifestSha256') == expected['gateTRAINManifestSha256'] and
        persona.get('color',{}).get('checkpointSha256') == expected['colorHeadSha256'] and
        persona.get('color',{}).get('failedOldProductOrderingUsed') is False and
        startup.get('rankCheckpointSha256') == expected['v5HeadSha256'] and startup.get('copyCheckpointSha256') == NATIVE_COPY_SHA and
        startup.get('copyCandidate',{}).get('checkpointSha256') == expected['copyHeadSha256'] and
        startup.get('rankSessionProvenance',{}).get('rankHead',{}).get('servingExportSha256') == expected['rankExportSha256'] and
        startup.get('sourceTensorsHashed') == 927 and startup.get('memory',{}).get('processBudgetBytes') == 7*2**30 and
        startup.get('imageDeliveryManifestSha256') == expected['integrationManifestSha256'],
        'Actual startup checkpoint/gate/color/copy/927/memory binding differs')
    return True


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
    require(color.get('kind') in ('precomputed-public-color-context','exact-last-color-context','fresh-base-color-context') and
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
        'queryTextBindingEvidence':'checked public input and pinned runtime source; live queryFeature does not expose a text SHA',
        'learnedGateCheckpointSha256':expected['gateSha256'], 'gateBindingEvidence':'exact selected startup before/after and source pins',
        'sameV5Top1UnderSameGuards':order[:1]==v5_order[:1], 'bothAbstainUnderSameGuards':not order and not v5_order,
        'bothNonemptyUnderSameGuards':bool(order and v5_order),'learnedQuerySpecificProbability':p,
        'actualCards':len(cards),'eligibleCount':len(order),'shortfall':max(0,6-len(wanted)),
        'requiredFields':required,'acceptedFields':accepted,'withheldFields':required-accepted,
        'sixEligible':len(order)>=6,'all20Accepted':len(cards)==6 and required==20 and accepted==20,
        'queryFeature':deepcopy(q),'rankDecision':deepcopy(native),'colorFeature':deepcopy(color),
        'bindings':bindings,'rankTiming':rank.get('timing'),'firstThreeTiming':first.get('timing') if first else None,
        'completeTiming':complete.get('timing'),'imageScope':'declared source variant/gallery metadata only; no image fetch/pixel/paint claim'}


def validate_plan(plan):
    need(plan.get('version')==VERSION and plan.get('fixedDenominator')==320 and plan.get('publicOnly') is True and
        plan.get('privateGoldParsed') is False and plan.get('acceptanceStateWritten') is False,'Explicit public-only adapter320 plan required')
    source=plan['originalRoster'];need(source['sha256']==ROSTER_SHA and old.sha(source['file'])==ROSTER_SHA,'Original320 roster identity changed')
    original=json.loads(Path(source['file']).read_text())
    need(plan['cases']==original['cases'],'Public320 cases/order/request/effective policies changed')
    need(len(plan['cases'])==320 and len({c['probeCaseId'] for c in plan['cases']})==320 and
        all(re.fullmatch('[A-Za-z0-9_-]+',c['probeCaseId']) for c in plan['cases']),'Fixed unique safe320 case roster required')
    need(set(plan['expectedPublic'])=={c['probeCaseId'] for c in plan['cases']},'Exact all320 public prompt preflights required')
    old.pins_current(plan['sourceSha256'])


def distribution(values):
    values=sorted(v for v in values if type(v) in (int,float) and math.isfinite(v))
    def percentile(p):
        index=(len(values)-1)*p;lower=int(index);upper=min(lower+1,len(values)-1)
        return values[lower]+(values[upper]-values[lower])*(index-lower)
    return {'count':len(values),'p50':percentile(.5) if values else None,'p95':percentile(.95) if values else None}


def run(plan,out,candidate_receipt,candidate_sha,*,base_url='http://127.0.0.1:8767',max_seconds=600,request_timeout=15,
        exchange_fn=None,selection_fn=None,opener_factory=None,deadline_context=None):
    parsed=urlsplit(base_url)
    need(parsed.scheme=='http' and parsed.hostname in ('127.0.0.1','localhost') and parsed.port==8767 and
        parsed.path in ('','/') and not (parsed.query or parsed.fragment or parsed.username or parsed.password),'Only explicit local adapter8767 endpoint allowed')
    need(0<max_seconds<=1200 and 0<request_timeout<=30,'Bounded HTTP lifecycle and request timeouts required')
    validate_plan(plan)
    selected=(selection_fn or selection)(candidate_receipt,candidate_sha)
    expected=selected['expected'];pins={**plan['sourceSha256'],**selected['pins']};old.pins_current(pins)
    out=Path(out)
    need(out.is_absolute() and out.resolve().is_relative_to(ROOT) and not out.exists(),'Unique absolute current-worktree HTTP output required')
    out.mkdir(parents=True);old.write(out/'plan.json',plan);old.write(out/'expected-sources.json',selected)
    report={'version':VERSION,'status':'running','startedAt':old.utc(),'fixedDenominator':320,'scope':plan['scope'],
        'timingScope':plan['timingScope'],'servingSelection':{k:v for k,v in selected.items() if k not in ('expected','pins')},
        'cases':[{'probeCaseId':case['probeCaseId'],'status':'unattempted'} for case in plan['cases']],
        'privateGoldParsed':False,'browserOrCheckoutUsed':False,'nativeCallsMadeByClient':0,'httpRequestAttempts':0,
        'healthRequestAttempts':0,'postRequestAttempts':0,'completeHttpResponses':0,'acceptanceStateWritten':False,'overallPromotionPass':False}
    save=lambda:old.write(out/'results.json',report)
    request=exchange_fn or old.exchange;opener=(opener_factory or (lambda:build_opener(ProxyHandler({}))))()
    seen=set();start=time.perf_counter();active=None;save()
    def health(name):
        report['httpRequestAttempts']+=1;report['healthRequestAttempts']+=1;save()
        result=request(opener,base_url.rstrip('/')+'/health',raw_path=out/name,timeout=min(request_timeout,max_seconds))
        report['completeHttpResponses']+=1
        need(result['httpStatus']==200,'Actual health did not return200')
        validate_health(json.loads(result['rawBytes']),expected)
    try:
        with (deadline_context or old.deadline)(max_seconds):
            health('health-before.json')
            for case,entry in zip(plan['cases'],report['cases'],strict=True):
                old.pins_current(pins);active=entry;entry.update(status='running',startedAt=old.utc());save()
                directory=out/'cases'/case['probeCaseId'];directory.mkdir(parents=True,exist_ok=False);old.write(directory/'input.json',case['request'])
                try:
                    report['httpRequestAttempts']+=1;report['postRequestAttempts']+=1;entry['dispatchOutcome']='attempted-unconfirmed';save()
                    result=request(opener,base_url.rstrip('/')+'/personalize',value=case['request'],raw_path=directory/'raw.ndjson',
                        timeout=min(request_timeout,max_seconds-(time.perf_counter()-start)))
                    report['completeHttpResponses']+=1;entry['dispatchOutcome']='response-received'
                    old.write(directory/'http.json',{k:v for k,v in result.items() if k!='rawBytes'})
                    need(result['httpStatus']==200,'Actual POST returnedHTTP'+str(result['httpStatus']))
                    events=[json.loads(line) for line in result['rawBytes'].splitlines() if line.strip()];old.write(directory/'events.json',events)
                    analysis=analyze_events(case,plan['expectedPublic'][case['probeCaseId']],events,result['headers'],plan['products'],expected,seen)
                    old.write(directory/'analysis.json',analysis)
                    entry.update(status='completed' if analysis['qualityPassed'] else 'failed-validation',analysis=analysis,httpElapsedMs=result['httpElapsedMs'])
                except BaseException as error:
                    entry.update(status='timeout' if isinstance(error,(TimeoutError,old.WallDeadline)) else 'failed',
                        error={'type':type(error).__name__,'message':str(error)},partialHttp=getattr(error,'http_partial',None))
                    partial=getattr(error,'http_partial',{}) or {}
                    if partial.get('httpStatus') is not None and entry.get('dispatchOutcome')!='response-received':entry['dispatchOutcome']='response-started-incomplete'
                    partial_id=partial.get('headers',{}).get('x-request-id')
                    if partial_id:entry['actualRequestId']=partial_id;entry['duplicateActualRequestId']=partial_id in seen;seen.add(partial_id)
                    old.write(directory/'failure.json',entry)
                    if isinstance(error,(old.WallDeadline,KeyboardInterrupt,SystemExit)):raise
                finally:entry['finishedAt']=old.utc();save()
                active=None
            health('health-after.json');old.pins_current(pins);report['status']='completed-fixed320-validation'
    except BaseException as error:
        report.update(status='stopped',error={'type':type(error).__name__,'message':str(error)})
        if active and active['status']=='running':active.update(status='timeout' if isinstance(error,old.WallDeadline) else 'failed',error=report['error'])
    finally:
        analyses=[row['analysis'] for row in report['cases'] if 'analysis' in row]
        valid=[a for a in analyses if 'rankDecision' in a]
        report['summary']={'attempted':sum(r['status']!='unattempted' for r in report['cases']),
            'unattempted':sum(r['status']=='unattempted' for r in report['cases']),
            'completed':sum(r['status']=='completed' for r in report['cases']),
            'failedOrTimeout':sum(r['status'] not in ('completed','unattempted') for r in report['cases']),
            'protocolPassed':sum(a['protocolPassed'] for a in analyses),'uniqueActualRequestIds':len(seen),
            'sixEligible':sum(a.get('sixEligible',False) for a in analyses),'sixEligibleAll20Accepted':sum(a.get('all20Accepted',False) for a in analyses),
            'sameGuardV5Comparisons':len(valid),'comparisonScope':'recomputed returned-score ordering; protocol/quality failures reported separately; fixed320 denominator retained','sameGuardV5Top1Agreements':sum(a['sameV5Top1UnderSameGuards'] for a in valid),
            'bothNonempty':sum(a['bothNonemptyUnderSameGuards'] for a in valid),'bothAbstain':sum(a['bothAbstainUnderSameGuards'] for a in valid),
            'reportedQueryBaseForwards':sum(a['queryFeature']['nativeBackboneForwardCalls'] for a in valid if type(a['queryFeature'].get('nativeBackboneForwardCalls')) is int),
            'reportedColorBaseForwards':sum(a['colorFeature']['nativeBackboneForwardCalls'] for a in valid if type(a['colorFeature'].get('nativeBackboneForwardCalls')) is int),
            'reportedRankBodyForwards':sum(a['rankDecision']['nativeBodyForwards'] for a in valid if type(a['rankDecision'].get('nativeBodyForwards')) is int),
            'reportedRankLMProjectionCalls':sum(a['rankDecision']['nativeLanguageModelProjectionCalls'] for a in valid if type(a['rankDecision'].get('nativeLanguageModelProjectionCalls')) is int),
            'reportedTargetFreeGradientCaptures':sum(a['rankDecision']['gradientEvaluationCalls'] for a in valid if type(a['rankDecision'].get('gradientEvaluationCalls')) is int),
            'reportedPublicXHYHits':sum(a['rankDecision'].get('cacheHit') is True for a in valid),
            'reportedCatalogColdBuilds':sum(a['rankDecision'].get('catalogPrefixCreatedThisRequest') is True for a in valid),
            'reportedCatalogWarmHits':sum(a['rankDecision'].get('catalogPrefixCacheHit') is True for a in valid),
            'learnedPSpecific':distribution([a['learnedQuerySpecificProbability'] for a in valid]),
            'serverRankMs':distribution([a['rankTiming'].get('rankMs') for a in valid]),
            'serverFirst3ReadyMs':distribution([a['firstThreeTiming'].get('first3readyMs') for a in valid if a.get('firstThreeTiming')]),
            'serverFullReadyMs':distribution([a['completeTiming'].get('fullReadyMs') for a in valid]),
            'clientHttpMs':distribution([r.get('httpElapsedMs') for r in report['cases']])}
        report.update(finishedAt=old.utc(),elapsedSeconds=time.perf_counter()-start);save()
    return report


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--original-plan',type=Path,default=HERE/'persona_http_semantic_prepare_v1/plan.json')
    parser.add_argument('--model-path',type=Path);parser.add_argument('--plan',type=Path);parser.add_argument('--plan-sha256')
    parser.add_argument('--candidate-receipt',type=Path);parser.add_argument('--candidate-sha256');parser.add_argument('--base-url',default='http://127.0.0.1:8767')
    parser.add_argument('--max-seconds',type=float,default=600);parser.add_argument('--request-timeout',type=float,default=15)
    args=parser.parse_args(argv)
    if args.run:
        if not all((args.plan,args.plan_sha256,args.candidate_receipt,args.candidate_sha256)):parser.error('Explicit frozen plan+SHA and serving-pending candidate receipt+SHA required')
        need(old.sha(args.plan)==contract.exact_sha(args.plan_sha256),'Reviewed adapter HTTP plan SHA differs')
        report=run(json.loads(args.plan.read_text()),args.out,args.candidate_receipt,args.candidate_sha256,
            base_url=args.base_url,max_seconds=args.max_seconds,request_timeout=args.request_timeout)
        print(json.dumps({'status':report['status'],'summary':report['summary'],'overallPromotionPass':False}))
        if report['status']!='completed-fixed320-validation' or report['summary']['completed']!=320:raise SystemExit(1)
    else:
        if not args.model_path:parser.error('CPU prepare requires explicit existing local --model-path')
        need(args.out.is_absolute() and args.out.resolve().is_relative_to(ROOT) and not args.out.exists(),'Unique absolute worktree preparation output required')
        plan=prepare(args.original_plan,args.model_path);args.out.mkdir(parents=True);old.write(args.out/'plan.json',plan)
        print(json.dumps({'status':'CPU-only-prepared','cases':320,'httpRequests':0,'planSha256':old.sha(args.out/'plan.json')}))


if __name__=='__main__':main()
