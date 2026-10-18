"""Optional width10/20 contiguous actual-store ranking and fresh-copy benchmark.

Requires the completed source-conditioned v2 fit and its unchanged provenance.
Six development-replay requests, six cards, real bound actions, no UI mutations.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

from .catalog_fusion.generalization_v2 import native_owner_lock
from .store_catalog_v1 import canonical
from .store_rank_session import StoreRankSession
from .semantic_copy_feature_fit import MODEL, digest, verify_model_files


def completed_adapter(copy_run):
    copy_run=Path(copy_run);report=json.loads((copy_run/'results.json').read_text());checkpoint=copy_run/'adapter.safetensors'
    required=('all320NonzeroBGatesPassed','fitStartedFromExactFrozenBaseZeroB',
        'learnedOriginalArchivedNativeParityExact','frozenLanguageHashesUnchanged','sourceFilesUnchanged','datasetManifestUnchanged')
    if (report['status']!='completed' or report['nativeUpdates']!=320 or report['optimizerUpdateCalls']!=320
            or report['languageParameterTensorsHashed']!=927 or not report['checkpoint']['exact']
            or any(not report['controls'].get(k) for k in required) or digest(checkpoint)!=report['checkpoint']['sha256']):
        raise ValueError('Completed exact v2 fit with all fixed gates/updates/reload/base proofs required')
    for mode in ('frozen','failed-daypack','new-store'):
        if report['controls'][mode]['status']!='exact-controls-passed':raise ValueError('All completed width6 controls required')
    return report,checkpoint


def run(rank_run,copy_run,public_path,out,*,width,memory_gib=7):
    from .store_copy_fit_v2 import CONFIG, assert_source_fresh, deadline
    from .store_copy_wide_v2 import native_controls, run_case, validate_width
    validate_width(width)
    if memory_gib not in (7,8):raise ValueError('Bounded wide experiment requires explicit7 or8GiB process guard')
    from .store_copy_data_v2 import bind_case
    rank_run,copy_run,public_path,out=map(Path,(rank_run,copy_run,public_path,out))
    if out.exists():raise FileExistsError('Preserve prior hybrid run')
    fit,checkpoint=completed_adapter(copy_run);assert_source_fresh(fit['preflight'],Path('experiments/ranker/pretrained/store_copy_v2'))
    rank_report=json.loads((rank_run/'report.json').read_text())
    if digest(public_path)!=rank_report['publicCasesSha256']:raise ValueError('Exact audited public request file required')
    public=[x for x in json.loads(public_path.read_text()) if x['split']=='evaluation'][:6]
    if [x['caseId'] for x in public]!=[f'store-eval-{i:02d}' for i in range(6)]:raise ValueError('Fixed first6 development replay requests required')
    for case in public:
        if case['request']['hardRequirements']:raise ValueError('Fixed copy diagnostic only admits known six-eligible requests')
    with native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'),'root'):
        out.mkdir(parents=True);started=time.perf_counter();bb=None;arms=[{'caseId':x['caseId'],'status':'not-attempted'} for x in public]
        report={'status':'started','fixedCases':6,'arms':arms,'copyCheckpointSha256':digest(checkpoint),
          'executionCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'rankReportSha256':digest(rank_run/'report.json'),'copyReportSha256':digest(copy_run/'results.json'),
          'publicCasesSha256':digest(public_path),
          'sources':{n:digest(Path(__file__).with_name(n)) for n in ('store_hybrid_wide_benchmark_v2.py','store_copy_wide_v2.py','store_hybrid_benchmark_v2.py','store_rank_session.py')},
          'width':width,'memoryGiB':memory_gib,
          'scope':'complete model-side fresh ranking+native T/B/C/S generation, validated/serialized first3 and full6 actual-store action payloads; no browser paint',
          'controls':[],'optimizerUpdates':0}
        def save():
            report['wallSeconds']=time.perf_counter()-started
            done=[a for a in arms if a['status']=='completed-attempt']
            report['summary']={'completed':len(done),'fixedCases':6,'validFirstThree':sum(a['firstThreeReceipt']['valid'] for a in done),
                'validFullSix':sum(a['fullSixReceipt']['valid'] for a in done),
                'attemptFirstThreeMs':[a['firstThreeReceipt']['validatedAndSerializedAtMs'] for a in done],
                'attemptFullSixMs':[a['fullSixReceipt']['validatedAndSerializedAtMs'] for a in done],
                'validFirstThreeMs':[a['firstThreeReceipt']['validatedAndSerializedAtMs'] for a in done if a['firstThreeReceipt']['valid']],
                'validFullSixMs':[a['fullSixReceipt']['validatedAndSerializedAtMs'] for a in done if a['fullSixReceipt']['valid']]}
            if 'quality' in report:
                report['summary']['usefulFirstThreeProxy']=sum(x['usefulFirstThreeProxy'] for x in report['quality'].values())
                report['summary']['usefulFullSixProxy']=sum(x['usefulFullSixProxy'] for x in report['quality'].values())
            (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        save()
        try:
            report['modelVerification']=verify_model_files(fit['preflight']['modelFiles']);save()
            import mlx.core as mx
            from mlx_lm.generate import generation_stream
            from mlx_lm.tuner.lora import LoRALinear
            from .backbone import FrozenBackbone
            from .evidence_plan import REVISION
            from .semantic_copy_training import parameter_hashes
            context=mx.stream(generation_stream);context.__enter__()
            class Guarded(FrozenBackbone):
                def check_memory(self):
                    super().check_memory()
                    if max(self.mx.get_peak_memory(),self.mx.get_active_memory())>self.memory_budget:raise MemoryError('Hybrid wide peak exceeds caller-declared guard')
            import psutil
            available=psutil.virtual_memory();report['hostMemoryBeforeLoad']={'availableBytes':available.available,'totalBytes':available.total,'percent':available.percent,'swap':psutil.swap_memory()._asdict()}
            if available.available<(memory_gib+1)*2**30:raise MemoryError('Insufficient inspected headroom for bounded wide guard plus1GiB margin')
            bb=Guarded(MODEL,REVISION,4096,memory_gib);before=parameter_hashes(bb.model.language_model)
            adapter=LoRALinear.from_base(bb.model.layers[-1].mlp.down_proj,r=4,scale=20,dropout=0)
            bb.model.layers[-1].mlp.down_proj=adapter;weights=mx.load(str(checkpoint));adapter.load_weights(list(weights.items()),strict=False)
            adapter.freeze();bb.model.eval();mx.eval(adapter.parameters());mx.synchronize()
            session=StoreRankSession(bb,rank_run);report['residentLoadMs']=bb.load_ms
            # Actual request roots use different public text from fit evaluation;
            # prove exact native forks before each timed request, outside timing.
            for arm,case in zip(arms,public,strict=True):
                reference=session.report['results']['evaluation']['learned']['cases'][case['caseId']]
                control_bound=bind_case({'caseId':case['caseId'],'request':case['request'],
                  'selectedProductIndices':reference['topSix']['productIndices'],'selectedColorIndices':[0]*6},session.snapshot)
                with deadline(CONFIG['controlTimeoutSeconds'],'hybrid native controls'):
                    report['controls'].append({'caseId':case['caseId'],**native_controls(bb,control_bound,out/('controls-'+case['caseId']),width=width)})
                save()
                gc.collect();mx.clear_cache();arm['status']='running';save();request_started=time.perf_counter()
                with deadline(CONFIG['requestTimeoutSeconds'],'hybrid fresh ranking and copy request'):
                    arm['stage']='fresh-native-ranking';ranked=session.rank_request(case['request'],copy_adapter=adapter);arm['ranking']=ranked
                    if ranked['scores16']!=reference['scores']:raise ValueError('Integrated fresh rank scores differ from audited reference')
                    if ranked['productIndices']!=reference['topSix']['productIndices']:raise ValueError('Timed ordered rows differ from native control rows')
                    if ranked['shortfall']:raise ValueError('Do not pad copy request after shortfall')
                    color_indices=[next(i for i,c in enumerate(session.products[pi]['colorVariants']) if c['colorName']==card['facts']['selectedColor'])
                      for pi,card in zip(ranked['productIndices'],ranked['cards'],strict=True)]
                    bound=bind_case({'caseId':case['caseId'],'request':case['request'],'selectedProductIndices':ranked['productIndices'],
                        'selectedColorIndices':color_indices},session.snapshot);arm['bound']=bound
                    if [c['action'] for c in bound['cards']]!=[c['action'] for c in ranked['cards']]:raise ValueError('Actual variant/image binding differs across rank and copy')
                    bound['productSelection']='actual native learned ranking with hard guards; image/color metadata policy untrained'
                    arm.update(run_case(bb,bound,width=width,request_started=request_started,stage_callback=lambda stage:arm.update(stage=stage)),stage='completed',
                        requestScope='fresh observed request -> native query encoding -> learned rank -> real actions -> fresh native copy -> complete validated serialized payload; browser paint excluded')
                save();gc.collect();mx.clear_cache()
                print(canonical({'event':'hybrid-case','caseId':case['caseId'],'firstValid':arm['firstThreeReceipt']['valid'],'fullValid':arm['fullSixReceipt']['valid']}),flush=True)
            # External labels are scored after all timed requests; they never
            # enter rank features, selection, prompts or generation fields.
            report['quality']={}
            for arm in arms:
                reference=rank_report['results']['evaluation']['learned']['cases'][arm['caseId']]
                target=reference['targetProductIndex'];order=arm['ranking']['productIndices']
                first_valid=arm['firstThreeReceipt']['valid'];full_valid=arm['fullSixReceipt']['valid']
                report['quality'][arm['caseId']]={'targetProductIndex':target,'topOneCorrect':order[0]==target,
                    'targetInFirstThree':target in order[:3],'targetInFullSix':target in order,
                    'usefulFirstThreeProxy':bool(first_valid and target in order[:3]),
                    'usefulFullSixProxy':bool(full_valid and target in order),
                    'strictTopOneAndFirstThreeValid':bool(first_valid and order[0]==target),
                    'strictTopOneAndFullSixValid':bool(full_valid and order[0]==target),
                    'scope':'posthoc manual public-function target retrieval plus complete copy gate; no human usefulness/conversion claim'}
            after={k.replace('.31.mlp.down_proj.linear.','.31.mlp.down_proj.'):v for k,v in parameter_hashes(bb.model.language_model).items()}
            if before!=after or len(before)!=927 or digest(checkpoint)!=report['copyCheckpointSha256']:raise ValueError('Frozen base or copy checkpoint changed')
            assert_source_fresh(fit['preflight'],Path('experiments/ranker/pretrained/store_copy_v2'))
            report.update(status='completed',base927HashesUnchanged=True,memory=bb.runtime_memory());save()
        except BaseException as exc:
            for arm in arms:
                if arm['status']=='running':
                    partial=getattr(exc,'partial_result',{})
                    arm.update(partial,status='failed',rawRows=getattr(exc,'partial_rows',partial.get('rawRows',[])),
                        failedAtMs=(time.perf_counter()-request_started)*1000)
            report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
            if bb is not None:report['memory']=bb.runtime_memory()
            save();raise
        finally:
            if 'context' in locals():mx.synchronize(generation_stream);context.__exit__(None,None,None)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--rank-run',type=Path,required=True);p.add_argument('--copy-run',type=Path,required=True)
    p.add_argument('--public',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--width',type=int,choices=(10,20),required=True);p.add_argument('--memory-gib',type=int,choices=(7,8),default=7);a=p.parse_args()
    z=run(a.rank_run,a.copy_run,a.public,a.out,width=a.width,memory_gib=a.memory_gib);print(canonical(z['summary']))
