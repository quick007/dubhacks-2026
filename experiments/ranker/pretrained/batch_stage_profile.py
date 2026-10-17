"""Bounded width-three native stage profile and unchanged-prompt chunk control."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import time

CASE_IDS=('evidence-page-scale-1','evidence-page-scale-3','evidence-page-scale-6')
OUT=Path('data/ranker/pretrained/batch-stage-profile-v1')
REPORT=Path('experiments/ranker/pretrained/BATCH_STAGE_RESULTS.json')


class StageTrace:
    """Nested wall timers with exclusive attribution; no array materialization."""
    def __init__(self,synchronize,check_memory):
        self.synchronize=synchronize;self.check_memory=check_memory
        self.stack=[];self.events=[];self.totals={};self.started=time.perf_counter();self.stream=None

    @contextmanager
    def span(self,category,metadata=None):
        entry={'category':category,'metadata':metadata or {},'startMs':(time.perf_counter()-self.started)*1000}
        start=time.perf_counter();frame={'childrenMs':0};self.stack.append(frame)
        try:
            self.synchronize(self.stream)
            yield
            self.synchronize(self.stream)
            self.check_memory()
        except Exception as error:
            entry['error']=f'{type(error).__name__}: {error}';raise
        finally:
            elapsed=(time.perf_counter()-start)*1000;self.stack.pop()
            exclusive=elapsed-frame['childrenMs']
            entry.update(inclusiveMs=elapsed,exclusiveMs=exclusive)
            self.events.append(entry);self.totals[category]=self.totals.get(category,0)+exclusive
            if self.stack:self.stack[-1]['childrenMs']+=elapsed

    @contextmanager
    def install(self,native):
        patches=[]
        def patch(cls,name,category,metadata=None):
            original=getattr(cls,name)
            def wrapped(actor,*args,**kwargs):
                if cls is native.BatchGenerator:self.stream=actor.stream
                label=category(actor) if callable(category) else category
                info=metadata(actor,args) if metadata else {'uids':list(getattr(actor,'uids',[]))}
                with self.span(label,info):return original(actor,*args,**kwargs)
            patches.append((cls,name,original));setattr(cls,name,wrapped)
        try:
            patch(native.BatchGenerator,'insert_segments','queueInsertHost')
            patch(native.BatchGenerator,'_next','nativeSchedulerResidual',lambda a,x:{
                'activeUids':list(a._generation_batch.uids),'promptUids':list(a._prompt_batch.uids),
                'pendingCount':len(a._unprocessed_sequences)})
            patch(native.BatchGenerator,'_make_batch','newPrefixMergeDispatchHost')
            patch(native.PromptProcessingBatch,'extend','promptExtendDispatchHost')
            patch(native.PromptProcessingBatch,'split','promptSplitDispatchHost')
            patch(native.PromptProcessingBatch,'prompt','suffixPrefill',lambda a,x:{
                'uids':list(a.uids),'lengths':[len(t) for t in x[0]],
                'paddedPositions':max(map(len,x[0]),default=0)*len(x[0])-sum(map(len,x[0]))})
            patch(native.PromptProcessingBatch,'generate','promptToGenerationResidual')
            patch(native.GenerationBatch,'_step',lambda a:'lastPromptTokenFirstSample' if a._current_tokens is None else 'activeDecodeAndLookahead')
            patch(native.GenerationBatch,'next','responseBookkeepingResidual')
            patch(native.GenerationBatch,'extend','generationExtendDispatchHost')
            patch(native.GenerationBatch,'filter','generationFilterDispatchHost')
            patch(native.GenerationBatch,'extract_cache','finishedCacheExtractDispatchHost')
            yield self
        finally:
            for cls,name,original in reversed(patches):setattr(cls,name,original)

    def result(self):
        return {'exclusiveStageMs':self.totals,'sumExclusiveMs':sum(self.totals.values()),'events':self.events,
            'limitations':['Synchronized native method intervals include Python and synchronization overhead, not GPU kernel-only time.',
              'Cache merge/extend/filter timings are lazy dispatch; later materialization belongs to the consuming prefill/decode stage.',
              'Active decode includes normalization/sampling and native one-token lookahead, including terminal unused lookahead.',
              'No unused prefill logits or extra cache arrays are forced.']}


def fixed_cases(cases):
    lookup={c['caseId']:c for c in cases}
    if len(lookup)!=len(cases) or any(cid not in lookup for cid in CASE_IDS):raise ValueError('Missing or duplicate fixed cases')
    result=[lookup[cid] for cid in CASE_IDS]
    if [c['cards'] for c in result]!=[1,3,6]:raise ValueError('Fixed scale count changed')
    return result


def summarize(records):
    return {name:{'fixedCases':3,'completed':sum(r['arms'][name]['status']=='completed-attempt' for r in records),
        'failed':sum(r['arms'][name]['status']=='failed' for r in records),
        'unattempted':3-sum(r['arms'][name]['status']!='not-attempted' for r in records),
        'validPages':sum(r['arms'][name]['status']=='completed-attempt' and
            r['arms'][name].get('quality',{}).get('allFieldsFulfilled',False) for r in records)}
        for name in ('plain256','trace256','plain512')}


def matched_512(backbone,heads,case,out):
    """Four width-three arms; identical prefix/suffix shapes, rebuild and repeat."""
    import gc
    import numpy as np
    from .fusion import encode_shared_prefix
    from .parallel_decode import clone_prefix_state,exact_row_suffix,build_rows
    from .native_batch_diagnosis import state_hash,vector_comparison
    from .evidence_plan import all_actions,dump,digest
    from .page_plan import make_page_plan
    from mlx_lm.generate import BatchGenerator,StopSequences
    from mlx_lm.sample_utils import make_sampler
    embeddings,actions,_,base=encode_shared_prefix(backbone,case['request'],case['catalog'],1536)
    import torch
    with torch.inference_mode():decision=all_actions(case['request'],case['catalog'],actions,heads(embeddings).tolist())
    del embeddings
    rows=build_rows(make_page_plan(case['request'],case['catalog'],decision,1),case['request'],case['catalog'])
    pristine=state_hash(base['cache']);result={'width':3,'prefillStepSize':512,'memoryGuardBytes':backbone.memory_budget,
        'caseId':case['caseId'],'pristineSourceHash':pristine,'arms':{},'status':'running'}
    arrays={};abort=None;started=time.perf_counter()
    for repeat in (0,1):
        for mode in ('fork','independent'):
            key=f'{mode}-r{repeat}';arm={'mode':mode,'repeat':repeat,'status':'not-attempted','rows':{}}
            result['arms'][key]=arm;gen=None;caches=[];branch=None
            if abort:arm['reason']=abort;continue
            try:
                arm['baseBefore']=state_hash(base['cache']);prompts=[];branches=[]
                for row in rows:
                    if mode=='fork':branch=clone_prefix_state(backbone,base)
                    else:
                        features,_,_,branch=encode_shared_prefix(backbone,case['request'],case['catalog'],1536)
                        del features
                    branches.append(state_hash(branch['cache']))
                    suffix,cache=exact_row_suffix(backbone.tokenizer,row,branch,backbone.model)
                    prompts.append(suffix);caches.append(cache);branch=None
                arm['branchMatchesPristine']=[h==pristine for h in branches]
                if arm['branchMatchesPristine']!=[True]*3:raise AssertionError('Rebuilt complete state differs')
                gen=BatchGenerator(backbone.model,completion_batch_size=3,prefill_batch_size=3,prefill_step_size=512,
                    sampler=make_sampler(temp=0.0))
                uids=gen.insert(prompts,max_tokens=[7]*3,caches=caches,all_tokens=[list(base['tokens']) for _ in rows],
                    samplers=[make_sampler(temp=0.0) for _ in rows],logits_processors=[[] for _ in rows],
                    stop_sequences=[StopSequences([[int(t)] for t in backbone.tokenizer.eos_token_ids]) for _ in rows])
                mapping=dict(zip(uids,[r['rowId'] for r in rows],strict=True));del caches,cache;caches=[]
                arm['rows']={name:{'rawSampledTokenIds':[],'finishReason':None} for name in mapping.values()}
                while responses:=gen.next_generated():
                    backbone.mx.synchronize(gen.stream);backbone.check_memory()
                    if len(responses)!=3:raise AssertionError('Matched width-three control lost an active row')
                    for response in responses:
                        row=arm['rows'][mapping[response.uid]];row['rawSampledTokenIds'].append(int(response.token));row['finishReason']=response.finish_reason
                        if len(row['rawSampledTokenIds'])==7:
                            vector=np.array(response.logprobs.astype(backbone.mx.float32)).reshape(-1).copy()
                            arrays[key+'-'+mapping[response.uid]]=vector
                            top=np.argsort(vector)[-2:][::-1]
                            row['topTwoLogprobs']=[{'tokenId':int(t),'logprob':float(vector[t])} for t in top]
                    del response,responses
                arm['baseAfter']=state_hash(base['cache'])
                if arm['baseBefore']!=pristine or arm['baseAfter']!=pristine:raise AssertionError('Pristine source mutated')
                if any(len(r['rawSampledTokenIds'])!=7 for r in arm['rows'].values()):raise AssertionError('Control response cap changed')
                arm['status']='completed'
            except Exception as error:abort=f'{type(error).__name__}: {error}';arm.update(status='failed',error=abort)
            finally:
                if gen:
                    try:gen.close()
                    except Exception as error:abort=f'cleanup {type(error).__name__}: {error}';arm.update(status='failed',cleanupError=abort)
                gen=None;caches=[];branch=None
                gc.collect();backbone.mx.clear_cache()
                arm['memory']=backbone.runtime_memory();dump(out/'controls512-progress.json',result)
    if not abort:
        comparisons={}
        ref=result['arms']['fork-r0']
        for key,arm in result['arms'].items():
            comparisons[key]={}
            for rowid,row in arm['rows'].items():
                exact=row['rawSampledTokenIds']==ref['rows'][rowid]['rawSampledTokenIds'] and row['finishReason']==ref['rows'][rowid]['finishReason']
                dist=vector_comparison(arrays[key+'-'+rowid],arrays['fork-r0-'+rowid]);comparisons[key][rowid]={'exactTokensAndFinish':exact,'distribution':dist}
                if not exact or not dist['exact']:abort='Matched512 rebuild/repeat distribution differs'
        result['comparisons']=comparisons
    np.savez(out/'controls512-logprobs.npz',**arrays)
    result.update(status='completed' if not abort else 'failed',abort=abort,totalWallMs=(time.perf_counter()-started)*1000,
        arraysSha256=digest(out/'controls512-logprobs.npz'),memory=backbone.runtime_memory())
    dump(out/'controls512.json',result)
    return result


def run(args):
    import gc
    import psutil
    import subprocess
    import torch
    from importlib.metadata import version
    from .backbone import FrozenBackbone
    from .evidence_plan import REVISION,load_corrected,digest,dump
    from .parallel_benchmark import validate_gate,selected_ids,parity
    from .parallel_semantic_benchmark import validate_controls,comparison_policy
    from .parallel_decode import factorized_arm
    from .heading_experiment import heading_rows
    from .native_batch_diagnosis import state_hash
    native=importlib.import_module('mlx_lm.generate')
    runtime={n:version(n) for n in ('mlx','mlx-lm')}
    gate_path=Path('data/ranker/pretrained/parallel-decode-v3/native-compatibility.json')
    controls_path=Path('data/ranker/pretrained/native-batch-diagnosis-v5/probe.json')
    gate=json.loads(gate_path.read_text());controls=json.loads(controls_path.read_text())
    sources={n:digest(Path(__file__).with_name(n)) for n in ('parallel_decode.py','test_parallel_decode.py')}
    validate_gate(gate,runtime,sources);validate_controls(controls)
    if controls['runtime']!=runtime or controls['sourceSha256']!=digest(Path(__file__).with_name('native_batch_diagnosis.py')):
        raise ValueError('Matched256 source/runtime receipt changed')
    if controls['arraysSha256']!=digest(controls_path.parent/'position6-arrays.npz'):raise ValueError('Control arrays changed')
    if args.out.exists() or args.report.exists():raise ValueError('Preserve existing experiment; no overwrite')
    source=Path('data/ranker/pretrained/parallel-decode-v1/parallel-case-plan.json')
    all_cases=json.loads(source.read_text())['cases'];cases=fixed_cases(all_cases)
    before=psutil.virtual_memory()._asdict()
    if before['available']<7*2**30:raise MemoryError('Less than explicit7GiB available')
    prior_paths=[Path('experiments/ranker/pretrained/PARALLEL_SEMANTIC_RESULTS.json'),Path('experiments/ranker/pretrained/PARALLEL_SEMANTIC_SCALE_7G_RESULTS.json')]
    prior={r['caseId']:r['arms']['batch3'] for path in prior_paths for r in json.loads(path.read_text())['records'] if r['arms'].get('batch3',{}).get('status')=='completed-attempt'}
    native_source=Path(native.__file__)
    dump(args.out/'case-plan.json',{'cases':cases,'width':3,'cells':['plain256','trace256','plain512'],
        'memoryGuardGiB':7,'sameIndependentPrompts':'heading_rows compact; unchanged budgets/validator',
        'order':'plain256,trace256,plain512; reversed chunk order at3cards',
        'sourceSha256':digest(Path(__file__)),'nativeSourceSha256':digest(native_source),
        'compatibilitySha256':digest(gate_path),'matched256Sha256':digest(controls_path),
        'headingSourceSha256':digest(Path(__file__).with_name('heading_experiment.py'))})
    torch.set_num_threads(1);heads,_,receipt=load_corrected(args.heads)
    class PeakBackbone(FrozenBackbone):
        def check_memory(self):
            super().check_memory()
            if self.mx.get_peak_memory()>self.memory_budget:raise MemoryError('Observed peak exceeded declared7GiB guard')
    started=time.perf_counter();backbone=PeakBackbone(args.model,REVISION,1536,7);load_ms=(time.perf_counter()-started)*1000
    control=matched_512(backbone,heads,next(c for c in all_cases if c['caseId']=='evidence-r-e1f73b869dc810bc15'),args.out)
    abort=control.get('abort');records=[]
    print(json.dumps({'controls512':control['status'],'abort':abort,'memory':control['memory']}),flush=True)
    for index,case in enumerate(cases):
        record={'caseId':case['caseId'],'cards':case['cards'],'arms':{}}
        order=('plain256','trace256','plain512') if index!=1 else ('plain512','plain256','trace256')
        for name in order:
            trace=None;value=None;attempt_start=time.perf_counter();state_observation={};partial_responses=[]
            if abort:value={'status':'not-attempted','reason':abort}
            else:
                decoder=importlib.import_module('experiments.ranker.pretrained.parallel_decode');original_decode=decoder.decode_rows
                original_next=native.BatchGenerator.next_generated
                def observed_next(gen):
                    responses=original_next(gen)
                    partial_responses.extend({'uid':r.uid,'tokenId':int(r.token),'finishReason':r.finish_reason} for r in responses)
                    return responses
                def observed_decode(bb,state,*pargs,**kwargs):
                    state_observation['baseBefore']=state_hash(state['cache'])
                    try:return original_decode(bb,state,*pargs,**kwargs)
                    finally:state_observation['baseAfter']=state_hash(state['cache'])
                decoder.decode_rows=observed_decode
                native.BatchGenerator.next_generated=observed_next
                try:
                    step=512 if name=='plain512' else 256
                    if name=='trace256':
                        trace=StageTrace(backbone.mx.synchronize,backbone.check_memory)
                        with trace.install(native):
                            value=factorized_arm(backbone,heads,case['request'],case['catalog'],case['cards'],3,
                                prefill_step_size=step,row_builder=heading_rows)
                    else:value=factorized_arm(backbone,heads,case['request'],case['catalog'],case['cards'],3,
                        prefill_step_size=step,row_builder=heading_rows)
                    backbone.mx.synchronize(native.generation_stream);backbone.check_memory()
                    value['externalCompleteWallMs']=(time.perf_counter()-attempt_start)*1000
                    value['nativeResponseObservation']=partial_responses
                    value['completePrefixObservation']=state_observation
                    if state_observation.get('baseBefore')!=state_observation.get('baseAfter'):raise AssertionError('Pristine state changed')
                    if value['status']!='completed-attempt':raise RuntimeError(value.get('error','Native failure'))
                    value['limitations']=[x for x in value['limitations'] if not x.startswith('Six-GiB')]
                    value['experimentMemoryGuardBytes']=backbone.memory_budget
                    if selected_ids(value['pagePlan'])!=selected_ids(prior[case['caseId']]['pagePlan']):raise AssertionError('Frozen product/image selection changed')
                    comparison_policy(prior[case['caseId']],value)
                    if name=='plain256':value['historicalRawParity']=parity(prior[case['caseId']],value)
                    if name=='trace256':
                        baseline=record['arms']['plain256'];pair=parity(baseline,value)
                        if not pair['exact'] or baseline['quality']!=value['quality']:raise AssertionError('Instrumentation changed raw output or quality')
                        for key in ('admissions','finishOrder','maximumRowsEmittingTogether','nativeDecodeRounds','rowArSteps'):
                            if baseline['timing'][key]!=value['timing'][key]:raise AssertionError('Instrumentation changed scheduling: '+key)
                        if baseline['completePrefixObservation']!=value['completePrefixObservation']:raise AssertionError('Instrumentation changed source state')
                        value['exactInstrumentedPair']=True
                    if trace:value['stageTrace']=trace.result()
                except Exception as error:
                    abort=f'{name}: {type(error).__name__}: {error}'
                    value={'status':'failed','error':abort,'producedAttempt':value,'partialTrace':trace.result() if trace else None,
                        'completePrefixObservation':state_observation,'failedWallMs':(time.perf_counter()-attempt_start)*1000}
                    value['rawNativeResponsesBeforeFailure']=partial_responses
                finally:
                    decoder.decode_rows=original_decode
                    native.BatchGenerator.next_generated=original_next
                try:gc.collect();backbone.mx.synchronize(native.generation_stream);backbone.mx.clear_cache();backbone.check_memory()
                except Exception as error:
                    abort=f'cleanup {type(error).__name__}: {error}';value.update(status='failed',cleanupError=abort)
            record['arms'][name]=value;dump(args.out/f'{case["cards"]}-{name}.json',value)
            print(json.dumps({'cards':case['cards'],'arm':name,'status':value['status'],
                'quality':value.get('quality'),'wallMs':value.get('externalCompleteWallMs'),'abort':abort}),flush=True)
        if not abort:record['crossChunkParity']=comparison_policy(record['arms']['plain256'],record['arms']['plain512'])
        records.append(record);dump(args.out/'progress.json',{'records':records,'abort':abort})
    report={'schemaVersion':1,'synthetic':True,'training':False,'validatorChanges':False,'width':3,'fixedCases':3,
        'records':records,'summary':summarize(records),'abort':abort,'matched512':control,
        'runtime':runtime,'modelRevision':REVISION,'headCheckpoint':receipt['checkpoints']['heads'],
        'executionCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'sourceSha256':digest(Path(__file__)),'nativeSourceSha256':digest(native_source),
        'residentLoadMs':load_ms,'totalWallMs':(time.perf_counter()-started)*1000,
        'memoryBefore':before,'memoryAfter':psutil.virtual_memory()._asdict(),'memory':backbone.runtime_memory(),
        'limitations':['Three inspected cases and one run per cell; chunk order alternates but pressure/warmth bias remains.',
          'Instrumented wall times diagnose stages, not serving-speed improvements. Compare plain256/plain512 for speed.',
          'Different native chunk segmentation may change numerical output; raw differences remain and strict quality is unchanged.',
          'State hashing adds CPU/GPU transfer work identically in all cells; wall times include it.',
          'Deterministic section grouping; text descriptions only; no MTP/runtime/model/validator/UI changes.']}
    dump(args.out/'profile.json',report)
    report['evidence']={str(p):digest(p) for p in (args.out/'profile.json',args.out/'case-plan.json',args.out/'controls512.json',args.out/'controls512-logprobs.npz',gate_path,controls_path)}
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
