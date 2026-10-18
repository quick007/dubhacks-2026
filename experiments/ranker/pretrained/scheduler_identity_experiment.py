"""Bounded fresh-value protocol comparison; scheduler supplies field identity."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata

ROOT=Path('experiments/ranker/pretrained')
CASES=ROOT/'SCHEDULER_IDENTITY_CASES.json'
OUT=Path('data/ranker/pretrained/scheduler-identity-v1')
REPORT=ROOT/'SCHEDULER_IDENTITY_RESULTS.json'
FORBIDDEN_SEPARATORS='\r\n\v\f\x85\u2028\u2029'


def values_layout(plan,request,catalog):
    from .shared_evidence_experiment import shared_layout
    common,rows,evidence=shared_layout(plan,request,catalog)
    old='Each later request selects exactly one S/T/B field: return only its labeled plain line, then stop.'
    replacement=('Each later request selects one field whose identity is supplied by the scheduler. '
        'Return only the fresh value: no label, field ID or prefix. One plain line only, then stop.')
    if common['content'].count(old)!=1:raise ValueError('Frozen labeled layout changed')
    common['content']=common['content'].replace(old,replacement)
    for row in rows:
        index=int(row['fields'][0][1:]);target=f'section {index}' if row['kind']=='heading' else f'card {index}'
        row['messages']=[{'role':'user','content':f'Write only the fresh {row["kind"]} value for {target} using its shared evidence and {row["kind"]} rules. Do not output a label or field ID. End after this one line.'}]
    return common,rows,evidence


def encode_value_rows(rows):
    """Trusted scheduler prefix, untouched raw value; never repair old outputs."""
    result=[];failures=[];seen=set()
    for row in rows:
        if len(row['fields'])!=1 or not re.fullmatch(r'[STB]\d+',row['fields'][0]):raise ValueError('Invalid trusted field metadata')
        label=row['fields'][0]
        if label in seen:raise ValueError('Duplicate trusted field identity')
        seen.add(label);raw=row['text'];errors=[]
        if not raw or not raw.strip():errors.append('empty-value')
        if any(c in raw for c in FORBIDDEN_SEPARATORS):errors.append('multiline-or-line-separator')
        if re.search(r'(?<![A-Za-z0-9])[STB]\d+\s*:',unicodedata.normalize('NFKC',raw),re.I):
            errors.append('unexpected-generated-field-label')
        # Even invalid input is retained verbatim; fail-closed page status is
        # explicit, rather than stripping labels/newlines or inventing NONE.
        result.append({**row,'rawValue':raw,'text':f'{label}: {raw}',
            'schedulerFieldIdentity':label,'valueProtocolRejections':errors})
        failures.extend(row['rowId']+':'+error for error in errors)
    return result,failures


def assemble_values_legacy_v1(rows,plan,request,catalog):
    """Frozen historical accounting for explicit replay only, never new serving."""
    from .parallel_decode import assemble
    encoded,failures=encode_value_rows(rows);result=assemble(encoded,plan,request,catalog)
    if failures:
        result['quality']['allFieldsFulfilled']=False
        result['quality']['rejections']+=failures
    return {**result,'wireRows':encoded,'valueProtocolRejections':failures}


def assemble_values(rows,plan,request,catalog):
    """Reject malformed raw values in field provenance as well as page status.

    Keep the exact raw/wire text and original audit for diagnosis. The labeled
    parser may read a valid first line of a malformed value, so it cannot alone
    establish that this scheduler-owned field was generated successfully.
    """
    result=assemble_values_legacy_v1(rows,plan,request,catalog)
    sections=result['pagePlan']['sections']
    cards=[card for section in sections for card in section['products']]
    rejected=set()
    for row in result['wireRows']:
        if not row['valueProtocolRejections']:continue
        rejected.add(row['schedulerFieldIdentity'])
        # A malformed row can inject a syntactically valid different label.
        # Neither that injected value nor this row's first line has provenance.
        rejected.update(match[1] for line in row['text'].splitlines()
            if (match:=re.fullmatch(r'(S\d+|T\d+|B\d+): (.+)',line.strip())))
    for field in rejected:
        index=int(field[1:])-1
        if index<0:continue
        if field[0]=='S' and index<len(sections):
            sections[index]['heading']={'text':None,'source':'rejected'}
        elif field[0] in ('T','B') and index<len(cards):
            cards[index]['title' if field[0]=='T' else 'badge']={'text':None,'source':'rejected'}
    quality=result['quality']
    quality['newTitleCount']=sum(c['title']['source']=='generated' for c in cards)
    quality['generatedBadgeCount']=sum(c['badge']['source']=='generated' for c in cards)
    quality['generatedHeadingCount']=sum(s['heading']['source']=='generated' for s in sections)
    supported=[(i,c) for i,c in enumerate(cards,1) if c['approvedFactIds']]
    quality['titleObservedAlignmentRate']=(sum(c['title']['source']=='generated' for _,c in supported)/len(supported)
        if supported else None)
    no_support=[i for i,c in enumerate(cards,1) if not c['approvedFactIds']]
    if any('B'+str(i) in rejected for i in no_support):quality['correctNone']=False
    result['valueProtocolVersion']='strict-values-v2'
    return result


def mechanisms(rows,plan,catalog,values_only):
    """Diagnostics only; never change strict quality or silently accept fields."""
    from .fusion import _field_claim_errors,_normalized,_words
    cards=[c for s in plan['sections'] for c in s['products']];output=[]
    for row in rows:
        field=row['fields'][0];raw=row.get('rawValue',row['text']);kind=row['kind'];index=int(field[1:])-1
        if values_only:
            body=raw;protocol=not row.get('valueProtocolRejections')
        else:
            match=re.fullmatch(re.escape(field)+r': (.+)',raw)
            body=match[1] if match else raw;protocol=match is not None
        if kind=='heading':
            common=set.intersection(*(set(catalog[c['productId']]['tags']) for c in plan['sections'][index]['products']))
            product={'title':'Everyday travel choices','tags':sorted(common),'category':'backpacks'};support=[]
        else:
            card=cards[index];product=catalog[card['productId']];support=[f.rsplit(':',1)[1] for f in card['approvedFactIds']]
        claims=[] if kind=='badge' and body=='NONE' else _field_claim_errors(body,product,catalog)
        entry={'rowId':row['rowId'],'rawValueOrDiagnosticBody':body,'expectedField':field,'protocolValid':protocol,
            'finishReason':row['finishReason'],'factualClaimErrors':claims,'wordCount':len(body.split()),
            'contentDiagnosticsAreNotAcceptedFieldMetrics':True}
        if kind=='badge':entry.update(observedSupport=support,rawExactNone=body=='NONE',
            noneValuePolicyCorrect=(body=='NONE') if not support else True,
            noneAllowedWithSupport=True,noneRequiredWithoutSupport=not support)
        if kind=='title':
            approved={_normalized(product['title']),*(_normalized(v['title']) for v in product['presentationVariants'])}
            entry.update(copiesApprovedTitle=_normalized(body) in approved,
                observedSupport=support,supportedPreferenceMentioned=not support or any(
                re.search(r'\b'+re.escape(_normalized(t))+r'\b',_normalized(body)) for t in support),
                requestedFourToEightWords=4<=len(body.split())<=8)
        if kind=='heading':entry['requestedThreeToFiveWords']=3<=len(body.split())<=5
        output.append(entry)
    return output


def prepare_cases():
    """New SKU/fact-combination probes, frozen before any protocol generation."""
    from .fusion import observable_request
    from .boundary import guard_reasons
    original=next(c for c in json.loads(Path('data/ranker/pretrained/parallel-decode-v1/parallel-case-plan.json').read_text())['cases'] if c['caseId']=='evidence-page-scale-6')
    old_tags={tuple(sorted(p['tags'])) for p in original['catalog'].values()}
    old_names={p['title'].split()[0] for p in original['catalog'].values()}
    specifications=[('none-control',('Aster','Briar','Coral','Delta','Ember','Flint'),(18,22,27,16,24,30),
        [('compact','packable'),('durable','packable'),('organizer','packable'),('lightweight','packable'),('roomy','packable'),('water-resistant','packable')]),
        ('explicit-preferences',('Horizon','Islet','Jasper','Kestrel','Larch','Moraine'),(19,21,25,15,28,32),
        [('compact','padded'),('durable','reflective'),('compact','foldable'),('roomy','ventilated'),('lightweight','adjustable'),('durable','packable')])]
    heldouts=[]
    for label,names,capacities,tagsets in specifications:
        catalog={}
        for index,(name,capacity,tags) in enumerate(zip(names,capacities,tagsets,strict=True)):
            if name in old_names or tuple(sorted(tags)) in old_tags:raise ValueError('Holdout identity or tag combination is not new')
            pid='fresh-p-'+hashlib.sha256(f'scheduler-v1/{label}/{index}'.encode()).hexdigest()[:16]
            title=f'{name} {capacity}L Daypack';variants=[]
            for slot,style in enumerate(('detail-view','in-use-view')):
                vid='fresh-v-'+hashlib.sha256(f'{pid}/{slot}'.encode()).hexdigest()[:16]
                variants.append({'id':vid,'title':title if slot==0 else title+' — '+tags[0],
                    'imageDescription':f'The {title} shown in a {style} with its listed catalog attributes.',
                    'imageKey':'synthetic-text-only/'+vid,'facts':[f'{capacity}L',*tags],
                    'presentationStyle':style,'compatibleSurfaces':['compact-card','detail-panel']})
            catalog[pid]={'id':pid,'title':title,'category':'backpacks','description':f'{capacity}-litre daypack; '+', '.join(tags)+'.',
                'price':42.5+index*7,'shippingCost':5,'inStock':True,'tags':list(tags),'presentationVariants':variants}
        request=observable_request(copy.deepcopy(original['request']))
        request.update(recordId='scheduler-holdout-'+label,history=[],candidateProductIds=list(catalog))
        request['context'].update(task='Choose backpacks for everyday travel.',statedBudget=150,statedRequirements=[],
            statedPreferences=[] if label=='none-control' else ['compact','durable'],
            presentationSurface='compact-card',presentationPreferences=['detail-view'])
        request['presentationCompatibility']={pid:{v['id']:True for v in p['presentationVariants']} for pid,p in catalog.items()}
        if any(guard_reasons(request,p) for p in catalog.values()):raise ValueError('Heldout candidate is ineligible')
        heldouts.append({'caseId':request['recordId'],'cards':6,'source':'fresh-synthetic-protocol-holdout',
            'request':request,'catalog':catalog,'heldOutFromProtocolAdjustment':True,
            'rankingQualityLabelsAvailable':False,'newIdentityAndTagCombination':True})
    return {'schemaVersion':1,'frozenBeforeInference':True,'training':False,
        'diagnosticCases':[original],'freshSyntheticHoldouts':heldouts,
        'limitations':['Fresh synthetic protocol holdouts, not a sealed independent production test.',
            'New identities and tag combinations contain new attributes; frozen head ranking quality has no new oracle labels.',
            'No protocol adjustment or case dropping after observing either holdout. Existing inspected test cases are not relabeled held-out.']}


def value_arm(backbone,heads,case):
    import torch
    from .fusion import encode_shared_prefix,observable_request
    from .evidence_plan import all_actions
    from .page_plan import make_page_plan,validate_page_plan
    from .shared_evidence_experiment import append_evidence,message_suffixes
    from .parallel_decode import decode_rows,hashed
    from .native_batch_diagnosis import state_hash
    native=__import__('mlx_lm.generate',fromlist=['generation_stream'])
    begin=time.perf_counter();request=observable_request(case['request']);catalog=case['catalog']
    features,actions,rank_timing,rank=encode_shared_prefix(backbone,request,catalog,1536)
    with torch.inference_mode():decision=all_actions(request,catalog,actions,heads(features).tolist())
    del features
    plan=make_page_plan(request,catalog,decision,6);validate_page_plan(plan,request,catalog)
    structured=(time.perf_counter()-begin)*1000;rank_hash=state_hash(rank['cache'])
    common,rows,evidence=values_layout(plan,request,catalog);state,prep=append_evidence(backbone,rank,common)
    before=state_hash(state['cache'])
    with message_suffixes():outputs,timing=decode_rows(backbone,state,rows,3,prefill_step_size=256)
    backbone.mx.synchronize(native.generation_stream);backbone.check_memory();after=state_hash(state['cache'])
    if before!=after or state_hash(rank['cache'])!=rank_hash:raise AssertionError('Immutable native source changed')
    validation_start=time.perf_counter();result=assemble_values(outputs,plan,request,catalog)
    validation=(time.perf_counter()-validation_start)*1000
    return {'status':'completed-attempt','protocol':'scheduler-identity-raw-value','cards':6,'width':3,
        'rows':outputs,**result,'preparation':prep,'rankPrefixSha256':rank['prefixSha256'],
        'rankStateHash':rank_hash,'immutableSourceBefore':before,'immutableSourceAfter':after,
        'actualOutputTokens':sum(len(r['tokenIds']) for r in outputs),'actualRowArSteps':sum(r['arSteps'] for r in outputs),
        'totalOutputBudget':sum(r['maxTokens'] for r in rows),'mechanisms':mechanisms(result['wireRows'],plan,catalog,True),
        'timing':{'rankingPrefix':rank_timing,'structuredMs':structured,**timing,
            'assemblyAndStrictValidationMs':validation,'fullCompletionMs':(time.perf_counter()-begin)*1000},
        'memory':backbone.runtime_memory()}


def summarize(records,holdouts_only=False):
    selected=[r for r in records if r['freshHoldout'] or not holdouts_only];n=2 if holdouts_only else 3
    result={}
    for name in ('generatedLabels','schedulerIdentity'):
        values=[r['arms'][name] for r in selected]
        completed=[v for v in values if v['status']=='completed-attempt']
        valid=[v for v in completed if v['quality']['allFieldsFulfilled']]
        result[name]={'fixedCases':n,'completed':len(completed),'validPages':len(valid),
            'validRateFixedCases':len(valid)/n,'failed':sum(v['status']=='failed' for v in values),
            'unattempted':n-sum(v['status']!='not-attempted' for v in values),
            'fullCompletionMs':[v['externalFullCompletionMs'] for v in completed],
            'validFullCompletionMs':[v['externalFullCompletionMs'] for v in valid]}
    return result


def run(args):
    import gc
    import psutil
    import subprocess
    import torch
    from importlib.metadata import version
    from .backbone import FrozenBackbone
    from .evidence_plan import REVISION,load_corrected,dump,digest,all_actions
    from .parallel_benchmark import selected_ids,validate_gate
    from .shared_evidence_experiment import run_arm as labeled_arm,append_evidence,controls,same_outputs
    from .fusion import encode_shared_prefix
    from .page_plan import make_page_plan
    from .full_page_reliability import evidence_capsule
    packet=json.loads(CASES.read_text());cases=packet['diagnosticCases']+packet['freshSyntheticHoldouts']
    if len(cases)!=3 or len(packet['freshSyntheticHoldouts'])!=2:raise ValueError('Fixed one diagnostic and two newholdouts required')
    if args.out.exists() or args.report.exists():raise ValueError('Preserve existing experiment')
    expected_cases=subprocess.check_output(['git','show','HEAD:'+str(CASES)],text=True)
    if expected_cases!=CASES.read_text():raise ValueError('Freeze/commit inputs before inference')
    runtime={n:version(n) for n in ('mlx','mlx-lm')};gate_path=Path('data/ranker/pretrained/parallel-decode-v3/native-compatibility.json')
    validate_gate(json.loads(gate_path.read_text()),runtime,{n:digest(ROOT/n) for n in ('parallel_decode.py','test_parallel_decode.py')})
    prior_path=ROOT/'SHARED_EVIDENCE_RESULTS.json';prior=json.loads(prior_path.read_text())
    if digest(prior_path)!='72de2e0cf460f3e1f61ee37af70bdcf8ee1f4504062a78031b6f3be14da56f64':raise ValueError('Exact a7bdf19 baseline changed')
    before=psutil.virtual_memory()._asdict()
    if before['available']<7*2**30:raise MemoryError('Less than declared7GiB available')
    args.out.mkdir(parents=True);dump(args.out/'case-plan.json',{'casesSha256':digest(CASES),'sourceSha256':digest(Path(__file__)),
        'roster':[c['caseId'] for c in cases],'protocols':['generatedLabels','schedulerIdentity'],
        'fixedHoldoutCount':2,'width':3,'cardsEach':6,'fieldsEach':14,'outputStepCapEach':224,
        'prefillStepSize':256,'memoryGuardGiB':7,'validatorSha256':digest(ROOT/'evidence_plan.py'),
        'fullGridSha256':digest(ROOT/'FULL_GRID_WORKLOAD.json'),'baselineSha256':digest(prior_path),
        'order':'Labeled/values diagnostic; values/labeled heldoutNONE; labeled/values heldoutpreferences. One attempt each.'})
    torch.set_num_threads(1);heads,_,receipt=load_corrected(args.heads)
    class PeakBackbone(FrozenBackbone):
        def check_memory(self):
            super().check_memory()
            if self.mx.get_peak_memory()>self.memory_budget:raise MemoryError('Observed peak exceeded declared7GiB guard')
    started=time.perf_counter();backbone=PeakBackbone(args.model,REVISION,1536,7);load_ms=(time.perf_counter()-started)*1000
    first=cases[0];features,actions,_,rank=encode_shared_prefix(backbone,first['request'],first['catalog'],1536)
    with torch.inference_mode():decision=all_actions(first['request'],first['catalog'],actions,heads(features).tolist())
    del features
    plan=make_page_plan(first['request'],first['catalog'],decision,6)
    common,rows,evidence=values_layout(plan,first['request'],first['catalog']);state,prep=append_evidence(backbone,rank,common)
    control=controls(backbone,rank,state,common,rows,args.out);abort=control.get('abort')
    dump(args.out/'native-controls.json',control);del rank,state;gc.collect();backbone.mx.clear_cache()
    print(json.dumps({'controls':control['status'],'abort':abort}),flush=True)
    records=[];native=__import__('mlx_lm.generate',fromlist=['BatchGenerator'])
    for index,case in enumerate(cases):
        record={'caseId':case['caseId'],'freshHoldout':index>0,'arms':{}}
        order=('schedulerIdentity','generatedLabels') if index==1 else ('generatedLabels','schedulerIdentity')
        for name in order:
            value=None;raw=[];begin=time.perf_counter()
            if abort:value={'status':'not-attempted','reason':abort}
            else:
                original=native.BatchGenerator.next_generated
                def observed(gen):
                    responses=original(gen);raw.extend({'uid':r.uid,'tokenId':int(r.token),'finishReason':r.finish_reason} for r in responses)
                    return responses
                native.BatchGenerator.next_generated=observed
                try:
                    if name=='generatedLabels':
                        value=labeled_arm(backbone,heads,case,'sharedCached')
                        value['mechanisms']=mechanisms(value['rows'],value['pagePlan'],case['catalog'],False)
                    else:value=value_arm(backbone,heads,case)
                    response={'pagePlan':value['pagePlan'],'quality':value['quality'],
                        'rawRows':[{'rowId':r['rowId'],'text':r['text'],'rawSampledTokenIds':r['rawSampledTokenIds']} for r in value['rows']]}
                    serial_start=time.perf_counter();encoded=json.dumps(response,separators=(',',':')).encode()
                    value['responseSerializationMs']=(time.perf_counter()-serial_start)*1000
                    value['responseBytes']=len(encoded);value['externalFullCompletionMs']=(time.perf_counter()-begin)*1000
                    if len(value['rows'])!=14 or value['totalOutputBudget']!=224:raise AssertionError('Fields/card budget changed')
                    if index==0 and name=='generatedLabels':
                        ref=prior['arms']['sharedCached']
                        if not same_outputs(ref['rows'],value['rows']) or value['quality']!=ref['quality']:raise AssertionError('a7bdf19 labeled baseline replay differs')
                    if len(record['arms'])==1:
                        other=next(iter(record['arms'].values()))
                        if selected_ids(value['pagePlan'])!=selected_ids(other['pagePlan']):raise AssertionError('Product/image choices changed')
                        if evidence_capsule(value['pagePlan'],case['catalog'])!=evidence_capsule(other['pagePlan'],case['catalog']):raise AssertionError('Personalization support/evidence changed')
                    backbone.check_memory()
                except Exception as error:
                    abort=f'{case["caseId"]}/{name}: {type(error).__name__}: {error}'
                    value={'status':'failed','error':abort,'producedAttempt':value,'rawResponsesBeforeFailure':raw,
                        'failedWallMs':(time.perf_counter()-begin)*1000}
                finally:native.BatchGenerator.next_generated=original
                try:gc.collect();backbone.mx.synchronize(native.generation_stream);backbone.mx.clear_cache();backbone.check_memory()
                except Exception as error:abort=f'cleanup {type(error).__name__}: {error}';value.update(status='failed',cleanupError=abort)
            record['arms'][name]=value;dump(args.out/f'{index}-{name}.json',value)
            print(json.dumps({'caseId':case['caseId'],'arm':name,'status':value['status'],'quality':value.get('quality'),
                'fullMs':value.get('externalFullCompletionMs'),'abort':abort}),flush=True)
        records.append(record);dump(args.out/'progress.json',{'records':records,'abort':abort})
    report={'schemaVersion':1,'modelRevision':REVISION,'runtime':runtime,'records':records,'controls':control,'abort':abort,
        'summaryAll':summarize(records),'summaryFreshHoldouts':summarize(records,True),'training':False,'validatorChanges':False,
        'headCheckpoint':receipt['checkpoints']['heads'],'executionCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'sourceSha256':digest(Path(__file__)),'casesSha256':digest(CASES),'residentLoadMs':load_ms,
        'totalWallMs':(time.perf_counter()-started)*1000,'memoryBefore':before,'memoryAfter':psutil.virtual_memory()._asdict(),
        'memory':backbone.runtime_memory(),'limitations':packet['limitations']+[
            'Scheduler-owned canonical labels are transport metadata, not model-generated tokens or new wording.',
            'Raw generated values remain unchanged; empty/multiline/embedded-label output fails, no defaultNONE or retroactive old-output repair.',
            'Same strict lexical factuality/novelty/observed-support/global-completion policy; independent semantic audits remain necessary.',
            'One run per case/protocol; cachewarmth and order variation, not p95 or a statistical performance guarantee.',
            'All six products, fourteen fields and224sampled-step caps retained;12/24 workload unchanged and not run.']}
    dump(args.out/'report.json',report);report['evidence']={str(p):digest(p) for p in (args.out/'report.json',args.out/'case-plan.json',args.out/'native-controls.json',CASES)}
    dump(args.report,report)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('freeze','run'))
    parser.add_argument('--model',type=Path);parser.add_argument('--heads',type=Path,default=Path('data/ranker/pretrained/evidence-plan-v1'))
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--report',type=Path,default=REPORT);args=parser.parse_args()
    if args.action=='freeze':
        if CASES.exists():raise ValueError('Preserve already frozen holdouts')
        CASES.write_text(json.dumps(prepare_cases(),indent=2)+'\n');print('Frozen one diagnostic plus two new synthetic SKU/fact-combination holdouts; no inference')
    else:
        from .evidence_plan import REVISION
        expected=Path('.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots')/REVISION
        if args.model is None or args.model.resolve()!=expected.resolve():parser.error('Use existing pinned local snapshot')
        run(args)


if __name__=='__main__':main()
