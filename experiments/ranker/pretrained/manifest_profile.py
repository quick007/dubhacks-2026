"""Paired retained-manifest profiling; no prompt/validator/runtime changes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def response_bytes(value):
    return json.dumps({'pagePlan':value['pagePlan'],'quality':value['quality'],
        'rawText':value['generation']['text']},sort_keys=True,separators=(',',':')).encode()


def field_delivery(text,events,prompt_done_ms):
    """Partition delivered lines after native prompt completion, not GPU fields."""
    if not text:return []
    if not events:raise ValueError('Nonempty text requires native delivery events')
    output=[];offset=0;previous=prompt_done_ms
    for line in text.splitlines(keepends=True):
        end=offset+len(line)
        complete=next((e['relativeMs'] for e in events if e['charEnd']>=end),events[-1]['relativeMs'])
        visible=next((e['relativeMs'] for e in events if e['charEnd']>offset),complete)
        match=re.fullmatch(r'(S\d+|T\d+|B\d+): (.+)',line.strip())
        output.append({'label':match[1] if match else None,'rawLine':line.rstrip('\n'),
            'firstVisibleMs':visible,'lineDeliveredMs':complete,
            'postPromptDeliveryWindowMs':max(0,complete-previous)})
        previous=complete;offset=end
    return output


def check_pair(baseline,profile,prior):
    from .parallel_benchmark import selected_ids
    if baseline['generation']['text']!=prior['generation']['text']:
        raise AssertionError('Retained baseline no longer replays historical raw text')
    for key in ('text','finishReason','generatedTokens'):
        if baseline['generation'][key]!=profile['generation'][key]:
            raise AssertionError('Profiling changed generation '+key)
    if baseline['rawSampledTokenIds']!=profile['rawSampledTokenIds']:
        raise AssertionError('Profiling changed sampled native tokens')
    if baseline['quality']!=profile['quality']:
        raise AssertionError('Profiling changed strict quality')
    if not selected_ids(baseline['pagePlan'])==selected_ids(profile['pagePlan'])==selected_ids(prior['pagePlan']):
        raise AssertionError('Selected products/images changed')


def profile_arm(backbone,heads,case):
    import numpy as np
    import torch
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from .fusion import observable_request,build_shared_prefix,append_generation_suffix
    from .evidence_plan import all_actions,validate_generated,CONFIG
    from .page_plan import make_page_plan,validate_page_plan
    from .generation_diagnosis import manifest_prompt,field_audit
    mx=backbone.mx;timing={};started=time.perf_counter()
    def measure(name,fn):
        mx.synchronize();start=time.perf_counter();value=fn();mx.synchronize()
        timing[name]=(time.perf_counter()-start)*1000;backbone.check_memory();return value
    request=measure('publicInputProjectionMs',lambda:observable_request(case['request']))
    catalog=case['catalog']
    tokens,actions,positions,messages=measure('inputConstructionAndTokenizationMs',
        lambda:build_shared_prefix(backbone.tokenizer,request,catalog,1536))
    def cache_inputs():
        cache=make_prompt_cache(backbone.model);inputs=mx.array([tokens]);mx.eval(inputs);return cache,inputs
    cache,inputs=measure('emptyCacheAndInputMaterializationMs',cache_inputs)
    def prefill():
        hidden=backbone.model.model(inputs,cache=cache)
        pooled=hidden[0,mx.array(positions)].astype(mx.float32)
        pooled=pooled/mx.maximum(mx.linalg.norm(pooled,axis=1,keepdims=True),1e-8)
        mx.eval(pooled,[c.state for c in cache]);return pooled
    pooled=measure('backbonePrefixPrefillAndReadoutMs',prefill)
    def host():
        if not bool(mx.all(mx.isfinite(pooled)).item()):raise FloatingPointError('Nonfinite features')
        return torch.from_numpy(np.array(pooled)).clone()
    embeddings=measure('readoutHostTransferMs',host)
    def state_metadata():
        return {'cache':cache,'tokens':tokens,'messages':messages,'consumed':False,'model':backbone.model,
            'prefixSha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()}
    state=measure('sharedPrefixStateMetadataMs',state_metadata)
    def head_scores():
        with torch.inference_mode():return heads(embeddings).tolist()
    scores=measure('decisionHeadsMs',head_scores)
    decision=measure('actionProjectionMs',lambda:all_actions(request,catalog,actions,scores))
    def page():
        plan=make_page_plan(request,catalog,decision,case['cards']);validate_page_plan(plan,request,catalog);return plan
    plan=measure('pagePlanAndStructuralValidationMs',page)
    prompt=measure('manifestPromptConstructionMs',lambda:manifest_prompt(plan,request,catalog,False))
    suffix,native_cache=measure('suffixBoundaryAndTokenizationMs',
        lambda:append_generation_suffix(backbone.tokenizer,prompt,state,backbone.model))
    if native_cache is not cache:raise AssertionError('Manifest must hand off same complete cache')
    progress=[];events=[];raw=[];text='';generation_start=time.perf_counter()
    def callback(done,total):
        mx.synchronize()
        progress.append({'processed':done,'total':total,'relativeMs':(time.perf_counter()-generation_start)*1000})
    mx.synchronize();generation_start=time.perf_counter();last=None
    for last in stream_generate(backbone.model,backbone.tokenizer,prompt=suffix,
        max_tokens=CONFIG['budgets'][str(case['cards'])],prompt_cache=native_cache,
        sampler=make_sampler(temp=0.0),prefill_step_size=512,prompt_progress_callback=callback):
        mx.synchronize();start_char=len(text);text+=last.text;raw.append(int(last.token))
        events.append({'rawIndex':len(raw)-1,'tokenId':int(last.token),
            'relativeMs':(time.perf_counter()-generation_start)*1000,'textFragment':last.text,
            'charStart':start_char,'charEnd':len(text),'finishReason':last.finish_reason})
        backbone.check_memory()
    mx.synchronize();timing['nativeGenerationWallMs']=(time.perf_counter()-generation_start)*1000
    if last is None or len(raw)!=last.generation_tokens:raise AssertionError('Native response accounting changed')
    completed,quality=measure('strictCopyValidationMs',lambda:validate_generated(text,last.finish_reason,plan,request,catalog))
    audit=measure('diagnosticAuditMs',lambda:field_audit(text,last.finish_reason,plan,request,catalog))
    value={'pagePlan':completed,'quality':quality,'audit':audit,
        'generation':{'text':text,'finishReason':last.finish_reason,'generatedTokens':last.generation_tokens,
            'promptTokens':last.prompt_tokens,'sharedPrefixTokens':len(tokens)},'rawSampledTokenIds':raw}
    encoded=measure('responseJSONSerializationMs',lambda:response_bytes(value))
    timing['endToEndWallMs']=(time.perf_counter()-started)*1000
    timing['unattributedAndObserverMs']=timing['endToEndWallMs']-sum(v for k,v in timing.items()
        if k not in ('endToEndWallMs','unattributedAndObserverMs'))
    chunk_intervals=[]
    for a,b in zip(progress,progress[1:]):
        chunk_intervals.append({'from':a['processed'],'to':b['processed'],'ms':b['relativeMs']-a['relativeMs'],
            'kind':'last-prompt-token-and-native-decode-lookahead' if b['processed']==b['total'] else 'suffix-prefill'})
    prompt_done=progress[-1]['relativeMs']
    value.update(status='completed-attempt',timing=timing,prefixSha256=state['prefixSha256'],
        prefixTokens=len(tokens),suffixTokens=len(suffix),candidateReadouts=len(actions),responseBytes=len(encoded),
        sameCompleteCacheHandedOff=True,cacheForkCount=0,nativePrefillStepSize=512,
        progress=progress,nativePromptIntervals=chunk_intervals,responseEvents=events,
        fieldDelivery=field_delivery(text,events,prompt_done),memory=backbone.runtime_memory(),
        streamCost={'suffixPrefillMs':sum(x['ms'] for x in chunk_intervals if x['kind']=='suffix-prefill'),
            'lastPromptTokenAndLookaheadMs':sum(x['ms'] for x in chunk_intervals if x['kind']!='suffix-prefill'),
            'afterPromptProgressMs':timing['nativeGenerationWallMs']-prompt_done,
            'afterLastTextDeliveryMs':timing['nativeGenerationWallMs']-max(
                (e['relativeMs'] for e in events if e['textFragment']),default=prompt_done)},
        limitations=['Prefix GPU timing includes original pooled readout; unused full logits/hidden states are never forced.',
            'Native callback at total includes final prompt token and already-dispatched one-token lookahead.',
            'Per-field windows describe line delivery after prompt completion, not independent field GPU latency.',
            'Synchronization/observation overhead is measured by paired baseline; no new speed claim.'])
    return value


def run(args):
    import gc
    import psutil
    import torch
    import mlx_lm
    from importlib.metadata import version
    from .backbone import FrozenBackbone
    from .evidence_plan import REVISION,dump,digest,load_corrected
    from .full_page_reliability import run_arm
    if args.out.exists() or args.report.exists():raise ValueError('Preserve existing profile')
    source=Path('data/ranker/pretrained/parallel-decode-v1/parallel-case-plan.json')
    cases=[c for c in json.loads(source.read_text())['cases'] if c['caseId'] in
        ('evidence-page-scale-1','evidence-page-scale-3','evidence-page-scale-6')]
    if [c['cards'] for c in cases]!=[1,3,6]:raise ValueError('All fixed1/3/6-scale cases required')
    prior_path=Path('data/ranker/pretrained/full-page-reliability-v1/benchmark.json')
    prior={r['caseId']:r['arms']['manifest'] for r in json.loads(prior_path.read_text())['records']}
    memory_before=psutil.virtual_memory()._asdict()
    if memory_before['available']<7*2**30:raise MemoryError('Less than explicit7GiB guard available')
    dump(args.out/'case-plan.json',{'cases':cases,'sourceSha256':digest(Path(__file__)),
        'priorSha256':digest(prior_path),'nativePrefillStepSize':512,'memoryGuardGiB':7,
        'order':'Baseline/profile at1and6cards; profile/baseline at3cards. One pair each.'})
    class PeakBackbone(FrozenBackbone):
        def check_memory(self):
            super().check_memory()
            if self.mx.get_peak_memory()>self.memory_budget:raise MemoryError('Profile peak exceeded7GiB guard')
    torch.set_num_threads(1);heads,_,receipt=load_corrected(args.heads)
    started=time.perf_counter();backbone=PeakBackbone(args.model,REVISION,1536,7)
    load_ms=(time.perf_counter()-started)*1000;records=[];abort=None
    original_stream=mlx_lm.stream_generate
    for index,case in enumerate(cases):
        record={'caseId':case['caseId'],'cards':case['cards'],'arms':{}}
        for name in (('baseline','profile') if index%2==0 else ('profile','baseline')):
            if abort:value={'status':'not-attempted','reason':abort}
            else:
                arm_start=time.perf_counter();value=None;raw=[]
                try:
                    if name=='profile':value=profile_arm(backbone,heads,case)
                    else:
                        def observed(*pargs,**kwargs):
                            for response in original_stream(*pargs,**kwargs):
                                raw.append(int(response.token));yield response
                        mlx_lm.stream_generate=observed
                        value=run_arm(backbone,heads,None,case,False)
                        backbone.mx.synchronize();serial_start=time.perf_counter();encoded=response_bytes(value)
                        backbone.mx.synchronize();serial_ms=(time.perf_counter()-serial_start)*1000
                        value.update(status='completed-attempt',rawSampledTokenIds=raw,responseBytes=len(encoded),
                            externalEndToEndMs=(time.perf_counter()-arm_start)*1000,
                            responseJSONSerializationMs=serial_ms,memory=backbone.runtime_memory())
                    backbone.check_memory()
                except Exception as error:
                    value={'status':'failed','error':f'{type(error).__name__}: {error}',
                        'partialAttempt':value,'rawObservedBeforeFailure':raw,'failedWallMs':(time.perf_counter()-arm_start)*1000}
                    abort=value['error']
                finally:mlx_lm.stream_generate=original_stream
            record['arms'][name]=value;dump(args.out/f'case-{case["cards"]}-{name}.json',value)
            print(json.dumps({'cards':case['cards'],'arm':name,'status':value['status'],
                'timing':value.get('timing'),'quality':value.get('quality'),'abort':abort}),flush=True)
            gc.collect();backbone.mx.synchronize();backbone.mx.clear_cache()
        if not abort:
            try:
                check_pair(record['arms']['baseline'],record['arms']['profile'],prior[case['caseId']])
                record['exactPairAndHistoricalReplay']=True
            except Exception as error:
                abort=f'{type(error).__name__}: {error}';record['pairFailure']=abort
        records.append(record);dump(args.out/'progress.json',{'records':records,'abort':abort})
    report={'schemaVersion':1,'synthetic':True,'training':False,'validatorChanges':False,
        'executionGitHead':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'residentLoadMs':load_ms,'totalWallMs':(time.perf_counter()-started)*1000,'records':records,'abort':abort,
        'memoryBefore':memory_before,'memoryAfter':psutil.virtual_memory()._asdict(),
        'processRssBytes':psutil.Process().memory_info().rss,
        'memory':backbone.runtime_memory(),'headCheckpoint':receipt['checkpoints']['heads'],
        'runtime':{n:version(n) for n in ('mlx','mlx-lm')},'modelRevision':REVISION,
        'sourceSha256':digest(Path(__file__)),'limitations':['Three inspected fixed cases and one pair each; order and machine pressure affect timings.',
            'Manifest is one autoregressive stream; field delivery windows do not imply independent field generation.',
            'Serialization measures pagePlan/quality/rawText UTF8 payload, excluding diagnostic report IO.']}
    dump(args.out/'profile.json',report)
    report['evidence']={str(p):digest(p) for p in (args.out/'profile.json',args.out/'case-plan.json',prior_path)}
    dump(args.report,report)


def main():
    from .evidence_plan import REVISION
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--heads',type=Path,default=Path('data/ranker/pretrained/evidence-plan-v1'))
    parser.add_argument('--out',type=Path,default=Path('data/ranker/pretrained/manifest-profile-v1'))
    parser.add_argument('--report',type=Path,default=Path('experiments/ranker/pretrained/MANIFEST_PROFILE_RESULTS.json'))
    args=parser.parse_args();expected=Path('.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots')/REVISION
    if args.model.resolve()!=expected.resolve():parser.error('Use existing pinned local snapshot')
    run(args)


if __name__=='__main__':main()
