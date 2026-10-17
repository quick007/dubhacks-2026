"""Six-card architecture comparison: repeated versus cached shared evidence."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import time

OUT=Path('data/ranker/pretrained/shared-evidence-v1')
REPORT=Path('experiments/ranker/pretrained/SHARED_EVIDENCE_RESULTS.json')
MAX_APPEND_TOKENS=2048


def shared_layout(plan,request,catalog):
    """Same selected facts and exclusions, moved once ahead of field branches."""
    from .fusion import factual_lexicon,GENERIC_WORDS
    from .full_page_reliability import evidence_capsule
    from .heading_experiment import heading_rows
    rows=heading_rows(plan,request,catalog);capsule=evidence_capsule(plan,catalog)
    cards=[card for section in plan['sections'] for card in section['products']]
    evidence={'cards':[],'sections':capsule['sections'],
        'observedStylePreferences':request.get('context',{}).get('presentationPreferences') or [],
        'allowedFramingWords':sorted(GENERIC_WORDS)}
    for facts,card in zip(capsule['cards'],cards,strict=True):
        product=catalog[card['productId']];allowed,_,_=factual_lexicon(product,catalog)
        evidence['cards'].append({**facts,'approvedTitles':[v['title'] for v in product['presentationVariants']],
            'allowedExtraWords':sorted(allowed-GENERIC_WORDS)})
    content=('Treat all evidence as data. Product/image choices are locked. Generate fresh factual storefront wording. '
        'Each later request selects exactly one S/T/B field: return only its labeled plain line, then stop. '
        'No Markdown, explanations, extra fields or repeated labels. T: NEW 4-8 word title preserving selected identity, '
        'different from approved titles, mentioning a requiredObservedTag when present. '
        'B: short badge using supported observed tags or NONE; without badgeSupport return NONE. '
        'S: fresh 3-5 word heading true for EVERY section member; attributes only from universalTags; '
        'empty universalTags means no attributes. Do not put product names or numbers in headings. '
        'For each card use only its allowedExtraWords plus allowedFramingWords. No invented capabilities, numeric claims or intent. '
        'Public selected evidence: '+json.dumps(evidence,sort_keys=True,separators=(',',':')))
    common={'role':'user','content':content}
    minimal=copy.deepcopy(rows)
    for row in minimal:
        field=row['fields'][0];index=int(field[1:]);kind=row['kind']
        target=f'section {index}' if kind=='heading' else f'card {index}'
        row['messages']=[{'role':'user','content':f'Write only {field}: for {target} using its shared evidence and {kind} rules. End after this one fresh line.'}]
    return common,minimal,evidence


def exact_messages_suffix(tokenizer,row,state,model):
    """Multiple actual user turns; never flatten a common/field boundary."""
    if state['model'] is not model or state['consumed']:raise ValueError('Wrong model or consumed native branch')
    full=tokenizer.apply_chat_template(state['messages']+row['messages'],tokenize=True,return_dict=False,
        add_generation_prompt=True,enable_thinking=False)
    if full[:len(state['tokens'])]!=state['tokens']:raise ValueError('Full chat does not extend exact complete prefix')
    suffix=full[len(state['tokens']):]
    if not 1<=len(suffix)<=MAX_APPEND_TOKENS:raise ValueError('Experimental append exceeds explicit2048token cap; no truncation')
    state['consumed']=True
    return suffix,state['cache']


@contextmanager
def message_suffixes():
    from . import parallel_decode
    original=parallel_decode.exact_row_suffix;parallel_decode.exact_row_suffix=exact_messages_suffix
    try:yield
    finally:parallel_decode.exact_row_suffix=original


def append_evidence(backbone,rank_state,common):
    from .parallel_decode import clone_prefix_state,hashed
    mx=backbone.mx;started=time.perf_counter()
    messages=rank_state['messages']+[common]
    full=backbone.tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,
        add_generation_prompt=False,enable_thinking=False)
    if full[:len(rank_state['tokens'])]!=rank_state['tokens']:raise ValueError('Shared userturn changed rank boundary')
    suffix=full[len(rank_state['tokens']):]
    if not 1<=len(suffix)<=MAX_APPEND_TOKENS:raise ValueError('Evidence exceeds declared2048token append; no truncation')
    # The original rank state stays immutable. A partial failure cannot publish
    # an old-length state with a mutated recurrent cache.
    branch=clone_prefix_state(backbone,rank_state)
    for offset in range(0,len(suffix),256):
        inputs=mx.array([suffix[offset:offset+256]])
        hidden=backbone.model.model(inputs,cache=branch['cache'])
        mx.eval([c.state for c in branch['cache']]);mx.synchronize();del hidden,inputs
        backbone.check_memory()
    state={**branch,'tokens':full,'messages':messages,'consumed':False,
        'prefixSha256':hashlib.sha256(json.dumps(full,separators=(',',':')).encode()).hexdigest(),
        'rankPrefixSha256':rank_state['prefixSha256'],'representation':'shared-selected-evidence-v1'}
    return state,{'sharedEvidenceTokens':len(suffix),'sharedCompletePrefixTokens':len(full),
        'commonMessageSha256':hashed(common),'completePrepareMs':(time.perf_counter()-started)*1000,
        'prefillStepSize':256,'unusedLogitsOrHiddenForced':False}


def token_contract(tokenizer,rank_state,shared_state,common,rows):
    """CPU proof that cached/uncached layouts supply identical full tokens."""
    from .parallel_decode import hashed
    records=[]
    for row in rows:
        a=tokenizer.apply_chat_template(rank_state['messages']+[common]+row['messages'],tokenize=True,
            return_dict=False,add_generation_prompt=True,enable_thinking=False)
        b=tokenizer.apply_chat_template(shared_state['messages']+row['messages'],tokenize=True,
            return_dict=False,add_generation_prompt=True,enable_thinking=False)
        if a!=b or a[:len(shared_state['tokens'])]!=shared_state['tokens']:raise AssertionError('Cached and repeated full token inputs differ')
        records.append({'rowId':row['rowId'],'kind':row['kind'],'maxTokens':row['maxTokens'],
            'fullInputTokens':len(a),'fullInputTokenIdsSha256':hashed(a),
            'repeatedSuffixTokens':len(a)-len(rank_state['tokens']),
            'cachedSuffixTokens':len(a)-len(shared_state['tokens'])})
    return records


def same_outputs(a,b):
    if len(a)!=len(b):return False
    return all(x['rowId']==y['rowId'] and x['rawSampledTokenIds']==y['rawSampledTokenIds'] and
        x['finishReason']==y['finishReason'] and x.get('distributionHashes')==y.get('distributionHashes')
        for x,y in zip(a,b,strict=True))


def controls(backbone,rank_state,shared,common,rows,out):
    import gc
    from .parallel_decode import decode_rows
    from .native_batch_diagnosis import state_hash
    from .evidence_plan import dump
    native=__import__('mlx_lm.generate',fromlist=['BatchGenerator'])
    chosen=[next(r for r in rows if r['rowId']==name) for name in ('heading-1','title-1','badge-1')]
    chosen=[{**r,'maxTokens':7} for r in chosen]
    reference=state_hash(shared['cache']);arms={};abort=None
    for mode in ('fork','independent','repeat'):
        arm={'status':'not-attempted'};state=None;probabilities=[];rebuilt=[]
        if abort:arm['reason']=abort
        else:
            original=native.BatchGenerator.next_generated
            def observed(gen):
                responses=original(gen)
                for response in responses:
                    vector=response.logprobs.astype(backbone.mx.float32)
                    probabilities.append((response.uid,hashlib.sha256(backbone.np.array(vector).tobytes()).hexdigest()))
                return responses
            native.BatchGenerator.next_generated=observed
            from . import parallel_decode
            original_clone=parallel_decode.clone_prefix_state
            def independent_clone(bb,source):
                if source['prefixSha256']==shared['prefixSha256']:
                    fresh,metadata=append_evidence(bb,rank_state,common)
                    actual=state_hash(fresh['cache']);rebuilt.append({'hash':actual,'preparation':metadata})
                    if actual!=reference:raise AssertionError('Independent per-lane evidence state differs')
                    return fresh
                return original_clone(bb,source)
            try:
                state=shared
                if mode=='independent':parallel_decode.clone_prefix_state=independent_clone
                before=state_hash(state['cache'])
                if before!=reference:raise AssertionError('Independent complete evidence state differs')
                with message_suffixes():outputs,timing=decode_rows(backbone,state,chosen,3,prefill_step_size=256)
                backbone.mx.synchronize(native.generation_stream);backbone.check_memory()
                for uid,row in enumerate(outputs):row['distributionHashes']=[h for u,h in probabilities if u==uid]
                after=state_hash(state['cache'])
                if before!=after or state_hash(shared['cache'])!=reference:raise AssertionError('Immutable evidence state mutated')
                arm={'status':'completed','before':before,'after':after,'independentlyRebuiltLanes':rebuilt,'rows':outputs,'timing':timing}
                if mode=='independent' and len(rebuilt)!=3:raise AssertionError('All three independent lane rebuilds required')
                if mode!='fork' and not same_outputs(arms['fork']['rows'],outputs):raise AssertionError('Matched evidence cache outputs/distributions differ')
            except Exception as error:
                abort=f'{type(error).__name__}: {error}';arm={'status':'failed','error':abort,'produced':arm,'distributionObservations':probabilities}
            finally:
                native.BatchGenerator.next_generated=original
                parallel_decode.clone_prefix_state=original_clone
        arms[mode]=arm;state=None;gc.collect();backbone.mx.clear_cache();dump(out/'controls-progress.json',{'arms':arms,'abort':abort})
    return {'status':'completed' if not abort else 'failed','arms':arms,'abort':abort,'pristineEvidenceStateHash':reference}


def run_arm(backbone,heads,case,name,trace=False):
    import torch
    from .fusion import encode_shared_prefix,observable_request
    from .evidence_plan import all_actions
    from .page_plan import make_page_plan,validate_page_plan
    from .heading_experiment import heading_rows
    from .parallel_decode import decode_rows,assemble,hashed
    from .native_batch_diagnosis import state_hash
    from .batch_stage_profile import StageTrace
    native=__import__('mlx_lm.generate',fromlist=['BatchGenerator'])
    started=time.perf_counter();request=observable_request(case['request']);catalog=case['catalog']
    embeddings,actions,rank_timing,rank_state=encode_shared_prefix(backbone,request,catalog,1536)
    with torch.inference_mode():decision=all_actions(request,catalog,actions,heads(embeddings).tolist())
    del embeddings
    plan=make_page_plan(request,catalog,decision,6);validate_page_plan(plan,request,catalog)
    structured_ms=(time.perf_counter()-started)*1000
    rank_hash=state_hash(rank_state['cache']);prepare=None;contract=None;state=rank_state
    if name=='baseline':rows=heading_rows(plan,request,catalog)
    else:
        common,rows,evidence=shared_layout(plan,request,catalog)
        if name=='sharedCached':state,prepare=append_evidence(backbone,rank_state,common)
        else:rows=[{**r,'messages':[common]+r['messages']} for r in rows]
    state_before=state_hash(state['cache']);diagnostic=StageTrace(backbone.mx.synchronize,backbone.check_memory) if trace else None
    def generate():
        if name=='baseline':return decode_rows(backbone,state,rows,3,prefill_step_size=256)
        with message_suffixes():return decode_rows(backbone,state,rows,3,prefill_step_size=256)
    if diagnostic:
        with diagnostic.install(native):outputs,timing=generate()
    else:outputs,timing=generate()
    backbone.mx.synchronize(native.generation_stream);backbone.check_memory()
    state_after=state_hash(state['cache'])
    if state_before!=state_after or state_hash(rank_state['cache'])!=rank_hash:raise AssertionError('Pristine rank/evidence state changed')
    validation_start=time.perf_counter();result=assemble(outputs,plan,request,catalog)
    validation_ms=(time.perf_counter()-validation_start)*1000
    # Returned full-input hashes permit identity comparison despite different cached boundaries.
    for output in outputs:
        complete=backbone.tokenizer.apply_chat_template(state['messages']+output['messages'],tokenize=True,
            return_dict=False,add_generation_prompt=True,enable_thinking=False)
        output['fullInputTokenIdsSha256']=hashed(complete);output['fullInputTokens']=len(complete)
    return {'status':'completed-attempt','name':name,'cards':6,'width':3,'rows':outputs,**result,
        'rankPrefixSha256':rank_state['prefixSha256'],'prefixSha256':state['prefixSha256'],
        'rankStateHash':rank_hash,'immutableSourceBefore':state_before,'immutableSourceAfter':state_after,
        'preparation':prepare,'actualOutputTokens':sum(len(r['tokenIds']) for r in outputs),
        'actualRowArSteps':sum(r['arSteps'] for r in outputs),'totalOutputBudget':sum(r['maxTokens'] for r in rows),
        'timing':{'rankingPrefix':rank_timing,'structuredMs':structured_ms,**timing,'assemblyAndStrictValidationMs':validation_ms,
            'fullCompletionMs':(time.perf_counter()-started)*1000},
        'stageTrace':diagnostic.result() if diagnostic else None,'memory':backbone.runtime_memory()}


def run(args):
    import gc
    import psutil
    import subprocess
    import torch
    from importlib.metadata import version
    from .backbone import FrozenBackbone
    from .evidence_plan import REVISION,load_corrected,dump,digest,all_actions
    from .parallel_benchmark import selected_ids,validate_gate
    from .native_batch_diagnosis import state_hash
    from .fusion import encode_shared_prefix
    from .page_plan import make_page_plan
    from .heading_experiment import heading_rows
    if args.out.exists() or args.report.exists():raise ValueError('Preserve prior experiment outputs')
    runtime={n:version(n) for n in ('mlx','mlx-lm')}
    gate_path=Path('data/ranker/pretrained/parallel-decode-v3/native-compatibility.json')
    validate_gate(json.loads(gate_path.read_text()),runtime,{n:digest(Path(__file__).with_name(n)) for n in ('parallel_decode.py','test_parallel_decode.py')})
    baseline_path=Path('experiments/ranker/pretrained/BATCH_STAGE_RESULTS.json')
    prior=json.loads(baseline_path.read_text());ref=next(r for r in prior['records'] if r['cards']==6)['arms']['plain256']
    if digest(baseline_path)!='751db3762e0d842a7ec7b99cce0c2d5ef1bded9f89efaf6bb938207d143cf77e':raise ValueError('Exact b6d2057 baseline changed')
    source=Path('data/ranker/pretrained/parallel-decode-v1/parallel-case-plan.json')
    case=next(c for c in json.loads(source.read_text())['cases'] if c['caseId']=='evidence-page-scale-6')
    memory_before=psutil.virtual_memory()._asdict()
    if memory_before['available']<7*2**30:raise MemoryError('Less than declared7GiB available')
    args.out.mkdir(parents=True)
    dump(args.out/'case-plan.json',{'case':case,'baselineReportSha256':digest(baseline_path),'baselineCommit':'b6d2057e3cfadee46be47dbc6c4cca1c7434a0fa',
        'arms':['baseline','sharedRepeated','sharedCached','sharedCachedTrace'],'width':3,'prefillStepSize':256,
        'outputStepCap':224,'requiredCards':6,'requiredFieldRows':14,'memoryGuardGiB':7,
        'maxAppendTokens':MAX_APPEND_TOKENS,'sourceSha256':digest(Path(__file__)),'fullGridWorkloadSha256':digest(Path(__file__).with_name('FULL_GRID_WORKLOAD.json'))})
    torch.set_num_threads(1);heads,_,receipt=load_corrected(args.heads)
    class PeakBackbone(FrozenBackbone):
        def check_memory(self):
            super().check_memory()
            if self.mx.get_peak_memory()>self.memory_budget:raise MemoryError('Observed peak exceeded declared7GiB guard')
    started=time.perf_counter();backbone=PeakBackbone(args.model,REVISION,1536,7);load_ms=(time.perf_counter()-started)*1000
    # Fixed preparation/control costs are diagnostics outside measured serving arms.
    features,actions,_,rank=encode_shared_prefix(backbone,case['request'],case['catalog'],1536)
    with torch.inference_mode():decision=all_actions(case['request'],case['catalog'],actions,heads(features).tolist())
    del features
    plan=make_page_plan(case['request'],case['catalog'],decision,6)
    common,rows,evidence=shared_layout(plan,case['request'],case['catalog'])
    evidence_state,prep=append_evidence(backbone,rank,common)
    contracts=token_contract(backbone.tokenizer,rank,evidence_state,common,rows)
    original_rows=heading_rows(plan,case['request'],case['catalog']);original_suffixes=[]
    from .parallel_decode import clone_prefix_state,exact_row_suffix
    for row in original_rows:
        opaque={**rank,'cache':[],'consumed':False}
        suffix,_=exact_row_suffix(backbone.tokenizer,row,opaque,backbone.model);original_suffixes.append(len(suffix))
    input_analysis={'sharedEvidence':evidence,'commonMessage':common,'minimalRows':rows,'preparation':prep,'contracts':contracts,
        'baselineAggregateSuffixTokens':sum(original_suffixes),'repeatedSharedLayoutAggregateSuffixTokens':sum(c['repeatedSuffixTokens'] for c in contracts),
        'cachedFieldAggregateSuffixTokens':sum(c['cachedSuffixTokens'] for c in contracts),
        'cachedTotalSuffixWorkTokens':prep['sharedEvidenceTokens']+sum(c['cachedSuffixTokens'] for c in contracts)}
    dump(args.out/'input-analysis.json',input_analysis)
    control=controls(backbone,rank,evidence_state,common,rows,args.out);abort=control.get('abort')
    print(json.dumps({'controls':control['status'],'inputTokenWork':{k:v for k,v in input_analysis.items() if 'Tokens' in k},'abort':abort}),flush=True)
    del rank,evidence_state;gc.collect();backbone.mx.clear_cache();records={}
    for name in ('baseline','sharedRepeated','sharedCached','sharedCachedTrace'):
        value=None;begin=time.perf_counter()
        if abort:value={'status':'not-attempted','reason':abort}
        else:
            try:
                value=run_arm(backbone,heads,case,'sharedCached' if name=='sharedCachedTrace' else name,trace=name.endswith('Trace'))
                if selected_ids(value['pagePlan'])!=selected_ids(ref['pagePlan']):raise AssertionError('Product/image selection changed')
                if value['totalOutputBudget']!=224 or len(value['rows'])!=14:raise AssertionError('Fields or budgets changed')
                if name=='baseline':
                    if not same_outputs(ref['rows'],value['rows']) or value['quality']!=ref['quality']:raise AssertionError('Exact b6d2057 baseline no longer reproduces')
                if name=='sharedCached':
                    repeated=records['sharedRepeated']
                    if any(a['fullInputTokenIdsSha256']!=b['fullInputTokenIdsSha256'] for a,b in zip(repeated['rows'],value['rows'],strict=True)):
                        raise AssertionError('Cached/uncached full model input differs')
                if name=='sharedCachedTrace':
                    plain=records['sharedCached']
                    if not same_outputs(plain['rows'],value['rows']) or plain['quality']!=value['quality']:raise AssertionError('Tracing changed shared output')
                    for key in ('admissions','finishOrder','nativeDecodeRounds','maximumRowsEmittingTogether'):
                        if plain['timing'][key]!=value['timing'][key]:raise AssertionError('Tracing changed scheduling')
            except Exception as error:
                abort=f'{name}: {type(error).__name__}: {error}';value={'status':'failed','error':abort,'producedAttempt':value,'failedWallMs':(time.perf_counter()-begin)*1000}
            try:gc.collect();backbone.mx.clear_cache();backbone.check_memory()
            except Exception as error:abort=f'cleanup {type(error).__name__}: {error}';value.update(status='failed',cleanupError=abort)
        records[name]=value;dump(args.out/f'{name}.json',value)
        print(json.dumps({'arm':name,'status':value['status'],'quality':value.get('quality'),'timing':value.get('timing'),'abort':abort}),flush=True)
    report={'schemaVersion':1,'modelRevision':REVISION,'runtime':runtime,'caseId':case['caseId'],'cards':6,'width':3,
        'sourceSha256':digest(Path(__file__)),'executionCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'baselineReportSha256':digest(baseline_path),'headCheckpoint':receipt['checkpoints']['heads'],'inputAnalysis':input_analysis,
        'controls':control,'arms':records,'abort':abort,'residentLoadMs':load_ms,'totalWallMs':(time.perf_counter()-started)*1000,
        'memoryBefore':memory_before,'memoryAfter':psutil.virtual_memory()._asdict(),'memory':backbone.runtime_memory(),
        'limitations':['One inspected six-card case and one run per arm; no held-out or statistical serving-speed claim.',
            'Baseline-to-shared changes prompt layout and cache strategy; repeated-to-cached uses identical full token inputs but different batch/prefill segmentation.',
            'Strict global page validation is unchanged; invalid attempts are not usable-page speedups.',
            'All fourteen fields, six product/image choices and224sampled-step cap retained. Larger12/24 workload is unchanged and not executed.',
            'Complete recurrent/convolution/KV cache snapshots only; no arbitrary trim/splice, MTP, model download, UI or externalQ001 changes.']}
    dump(args.out/'report.json',report)
    report['evidence']={str(p):digest(p) for p in (args.out/'case-plan.json',args.out/'input-analysis.json',args.out/'report.json',baseline_path)}
    dump(args.report,report)


def main():
    from .evidence_plan import REVISION
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--heads',type=Path,default=Path('data/ranker/pretrained/evidence-plan-v1'))
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--report',type=Path,default=REPORT)
    args=parser.parse_args();expected=Path('.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots')/REVISION
    if args.model.resolve()!=expected.resolve():parser.error('Use existing pinned local snapshot')
    run(args)


if __name__=='__main__':main()
