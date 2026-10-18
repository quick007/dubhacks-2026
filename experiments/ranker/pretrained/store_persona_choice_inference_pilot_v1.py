"""Bounded two TRAIN-row single-choice gate pilot, CPU preparation by default.

No optimizer exists here. Explicit root-only native action owns the shared main
lock through927 integrity and teardown; never a server or browser action.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack,contextmanager
import gc
import json
import os
from pathlib import Path
import time
from . import store_qwen_lexical_choice_pilot_v1 as lexical
from . import store_persona_choice_features_v2 as features
from . import store_persona_choice_capture_v1 as public_capture
from . import store_persona_choice_scoring_v1 as scoring

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
MAIN=lexical.MAIN
VERSION='persona-choice-inference-pilot.v1'
RECIPE={'rank':4,'scale':20.,'dropout':0.,'seed':19083,'initialB':'exactzero',
        'optimizer':'Adam','learningRate':.0001,'epochs':4,'trainProfilesExpected':448,
        'ordersPerProfile':2,'featuresExpected':896,'updatesExpected':3584,
        'loss':'route-conditional soft CE; equal mass all maximum-grade TRAIN products',
        'selection':'final fixed step only; no development tuning',
        'projection':'full frozen native quantized LM projection then16gather',
        'memoryGiB':7,'maxNativeSeconds':600,'pilotOptimizerUpdates':0,
        'pilotRows':'oldTRAIN original0 canonical + its contrast160 reversed',
        'sourcePrefixPath':'perprofile actual full prompt[:-16]; actual last16 input tokens; no sharedcatalog optimization in firstgate','actualSuffixTokens':16,'inferencePathProbe':True}

@contextmanager
def cwd(path):
    old=Path.cwd();os.chdir(path)
    try:yield
    finally:os.chdir(old)

def check_output(path):
    path=Path(path)
    if not path.is_absolute() or not path.is_relative_to(ROOT) or path.exists():raise ValueError('Unique absolute worktree output required')
    return path

def inference_comparison(bb,adapter,state,row,item,gate,path):
    import numpy as np
    from .semantic_copy_gradient_features import _host_tensor
    mx=bb.mx;pure=scoring.public_item(item)
    native_plain=public_capture.plain_scores(bb,adapter,state,row['tokenIds'],row['routeTokenIds'])
    cached_plain=scoring.score(bb,adapter,pure)
    cached_gradient=public_capture.gradient_scores(bb,adapter,pure)
    mx.eval(native_plain,cached_plain,cached_gradient);mx.synchronize();bb.check_memory()
    public_item,public_proof=public_capture.capture(bb,adapter,state,row['tokenIds'],row['routeTokenIds'])
    public_reconstructed=public_capture.gradient_scores(bb,adapter,public_item)
    mx.eval(public_reconstructed);mx.synchronize();bb.check_memory()
    reference=np.asarray(gate['arms']['direct']['routeLogits'],dtype=np.float32)
    values={'plainNative':np.array(native_plain),'cachedPlain':np.array(cached_plain),
            'cachedGradient':np.array(cached_gradient),'publicGradientNative':np.asarray(public_proof['nativeRouteScores'],dtype=np.float32),
            'publicGradientReconstructed':np.array(public_reconstructed)}
    metrics={}
    for name,value in values.items():
        metrics[name]={'exactVsTrainingGradientNative':bool(np.array_equal(reference,value)),
                       'maxAbsDifference':float(np.max(np.abs(reference-value))),
                       'top1Equal':int(np.argmax(reference))==int(np.argmax(value)),
                       'orderEqual':np.argsort(-reference,kind='stable').tolist()==np.argsort(-value,kind='stable').tolist(),
                       'routeLogits':value.tolist()}
    original_tensors={k:_host_tensor(mx,mx.array(item[k],dtype={str(t):t for t in (mx.float32,mx.float16,mx.bfloat16)}[item['dtypes'][k]]))[1] for k in ('x','h','y')}
    tensor_parity=original_tensors==public_proof['featureTensors']
    result={'version':'persona-choice-inference-parity.v1','caseId':row['caseId'],'noPrivateEvaluationLabels':True,
            'reference':'actual supervised TRAIN numerical gradient-mode native gate logits',
            'labelFreeCaptureFeaturesMatchTrainingExact':tensor_parity,'comparisons':metrics,
            'publicCaptureProof':public_proof,'referenceRouteLogits':reference.tolist(),
            'cachedPlainScoreMatchesTrainingFeaturesExact':metrics['cachedPlain']['exactVsTrainingGradientNative'],
            'cachedGradientScoreMatchesTrainingFeaturesExact':metrics['cachedGradient']['exactVsTrainingGradientNative'],
            'plainNativeMatchesTrainingGradientExact':metrics['plainNative']['exactVsTrainingGradientNative']}
    result['labelFreeGradientPathExact']=tensor_parity and all(metrics[n]['exactVsTrainingGradientNative'] for n in ('cachedGradient','publicGradientNative','publicGradientReconstructed'))
    result['dynamicMode']='plain-native-last16' if result['plainNativeMatchesTrainingGradientExact'] else 'gradient-public-XHY-last16'
    result['cachedScoringMode']='plain' if result['cachedPlainScoreMatchesTrainingFeaturesExact'] else 'gradient-public-sum16'
    lexical.dump(path,result)
    if not result['labelFreeGradientPathExact']:raise ValueError('Label-free gradient capture/scoring differs from TRAIN feature basis; serving remains blocked')
    return result

def prepare(out):
    from transformers import AutoTokenizer
    from . import store_qwen_lexical_choice_inputs_v2 as formatter
    from .semantic_copy_feature_fit import model_input_receipt
    out=check_output(out);data=HERE/'synthetic_history_train_contrasts_v2'
    manifest=json.loads((data/'manifest.json').read_text())
    if lexical.sha(data/'manifest.json')!='7ec75bc25321d3c235159a5b609893bbd428b1512c82ab1e2d07446d8247a017':raise ValueError('Frozen original320TRAIN manifest required')
    for name,checksum in manifest['files'].items():
        if lexical.sha(data/name)!=checksum:raise ValueError('Original320TRAIN changed')
    public=json.loads((data/'profiles.train.public.json').read_text())['profiles']
    labels=json.loads((data/'labels.train.json').read_text())['labels'];gold={r['profileId']:r for r in labels}
    if len(public)!=320 or set(gold)!={r['profileId'] for r in public}:raise ValueError('Exact320TRAIN source join required')
    tokenizer=AutoTokenizer.from_pretrained(lexical.MODEL,local_files_only=True,trust_remote_code=False)
    catalog=json.loads((HERE/'synthetic_history_v1/catalog-binding.json').read_text())['products'];rows=[]
    for index,order in [(0,'canonical'),(160,'reversed')]:
        p=public[index];row=formatter.build_row(p['request'],catalog,tokenizer,p['profileId']+'-'+order,order)
        row['routeTokenIds']=[c['tokenId'] for c in row['choices']]
        row['targetWeights']=features.max_grade_targets(gold[p['profileId']]['relevanceGrades'],[c['productId'] for c in row['choices']])
        row['originalTrainProfileId']=p['profileId'];features.choice_contract(row);rows.append(row)
    sources=[Path(__file__),Path(features.__file__),Path(formatter.__file__),Path(lexical.__file__),Path(public_capture.__file__),Path(scoring.__file__),
             HERE/'test_store_persona_choice_features_v2.py',HERE/'test_store_persona_choice_inference_pilot_v1.py',HERE/'test_store_persona_choice_capture_v1.py',HERE/'semantic_copy_gradient_features.py',
             HERE/'semantic_copy_feature_fit.py',HERE/'semantic_copy_training.py',HERE/'parallel_decode.py',
             HERE/'native_batch_diagnosis.py',HERE/'backbone.py',HERE/'store_copy_fit_v2.py',
             HERE/'catalog_fusion/generalization_v2.py',lexical.MODEL_SOURCE,
             MAIN/'.tools/pretrained-venv/lib/python3.14/site-packages/mlx_lm/tuner/lora.py']
    pins={str(p.resolve()):lexical.sha(p) for p in sources}
    pins.update({str((data/name).resolve()):checksum for name,checksum in manifest['files'].items()})
    pins[str(data/'manifest.json')]=lexical.sha(data/'manifest.json')
    pins.update({str(p):s for p,s in lexical.PUBLIC_INPUTS.items()})
    with cwd(MAIN):model=model_input_receipt()
    pins.update({str(lexical.MODEL/name):s for name,s in model['smallFileSha256'].items()})
    packet={'version':VERSION,'recipe':RECIPE,'rows':rows,'sha256':pins,'modelFiles':model,
            'nativeOwnerLock':str(lexical.LOCK),'maxNativeSeconds':600,'nativeCalls':0,'optimizerUpdates':0,
            'TRAINLabelsParsed':True,'developmentGoldParsed':False,'nativeExecuted':False,
            'trainingNotAuthorizedByPilot':True,'prefixCacheOptimization':'deferred; exactperprofileprefix first'}
    out.mkdir(parents=True);lexical.dump(out/'plan.json',packet)
    lexical.dump(out/'PLAN_FROZEN.json',{'version':VERSION,'planSha256':lexical.sha(out/'plan.json'),'optimizerUpdates':0,'nativeExecuted':False})
    return packet

def run(plan_path,plan_sha,out, *, execute=False,owner=None):
    if lexical.sha(plan_path)!=plan_sha:raise ValueError('Exact frozen pilot SHA required')
    packet=json.loads(Path(plan_path).read_text())
    if packet['version']!=VERSION or packet['recipe']!=RECIPE or len(packet['rows'])!=2:raise ValueError('Frozen2row recipe required')
    lexical.check_pins(packet['sha256'])
    if not execute:return {'status':'cpu-preflight-only','nativeExecuted':False,'optimizerUpdates':0}
    if owner!='root':raise ValueError('Root is the sole authorized native owner')
    if any(lexical.listening(p) for p in (8766,8767)):raise RuntimeError('Retire native serving before exclusive pilot')
    out=check_output(out);out.mkdir(parents=True);lexical.dump(out/'plan.json',packet)
    report={'version':VERSION,'status':'initializing','optimizerUpdates':0,'sourcePinsUnchanged':None,
            'frozen927HashesUnchanged':None,'shutdownCompleted':False,'completedGates':0,
            'roster':[{'caseId':r['caseId'],'status':'unattempted'} for r in packet['rows']]}
    save=lambda:lexical.dump(out/'report.json',report)
    started=time.monotonic();save();resources=ExitStack();bb=model=adapter=state=adapted=None;before=None;primary=None
    try:
        from .catalog_fusion.generalization_v2 import native_owner_lock
        from .store_copy_fit_v2 import deadline
        resources.enter_context(native_owner_lock(lexical.LOCK,'root'))
        if any(lexical.listening(p) for p in (8766,8767)):raise RuntimeError('Serving became active before lock acquisition')
        lexical.check_pins(packet['sha256'])
        with cwd(MAIN):
            from .semantic_copy_feature_fit import verify_model_files
            report['weightVerification']=verify_model_files(packet['modelFiles']);save()
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.generate import generation_stream
        from mlx_lm.tuner.lora import LoRALinear
        from .backbone import FrozenBackbone
        from .semantic_copy_training import parameter_hashes
        from .native_batch_diagnosis import state_hash
        import numpy as np
        class GuardedBackbone(FrozenBackbone):
            def check_memory(self):
                super().check_memory()
                if max(self.mx.get_peak_memory(),self.mx.get_active_memory())>self.memory_budget:raise MemoryError('Native active/peak exceeds7GiB')
        with deadline(600,'two TRAIN native choice gates'),mx.stream(generation_stream):
            mx.random.seed(RECIPE['seed']);bb=GuardedBackbone(lexical.MODEL,lexical.MODEL.name,max_length=2048,memory_gib=7)
            model=bb.model;model.freeze();model.eval();before=parameter_hashes(model.language_model)
            if len(before)!=927 or dict(tree_flatten(model.trainable_parameters())):raise ValueError('927 frozen base tensors required')
            report['frozen927Before']=before;save()
            adapter=LoRALinear.from_base(model.layers[-1].mlp.down_proj,r=4,scale=20,dropout=0)
            model.layers[-1].mlp.down_proj=adapter;adapter.lora_a=adapter.lora_a.astype(mx.float32);adapter.lora_b=mx.zeros((4,4096),dtype=mx.float32);model.eval()
            initial={'lora_a':adapter.lora_a,'lora_b':adapter.lora_b};mx.eval(initial);mx.synchronize()
            if np.any(np.array(adapter.lora_b)!=0):raise ValueError('B must initialize exactly zero')
            report['initialBExactlyZero']=True;report['trainableParameters']={k:int(v.size) for k,v in tree_flatten(model.trainable_parameters())}
            if len(report['trainableParameters'])!=2 or sum(report['trainableParameters'].values())!=65536:raise ValueError('Exactly65536 A/B floats required')
            report['status']='native-gates';save()
            for i,(row,item) in enumerate(zip(packet['rows'],report['roster'],strict=True)):
                item['status']='running';report['activeCaseId']=row['caseId'];save();lexical.check_pins(packet['sha256'])
                # One exact base prefix then a separate nonzero-adapter prefix
                # establishes cache independence for the complete current row.
                state,contract=features.prepare_state(bb,adapter,row)
                try:zero_gate,zero_feature=features.probe(bb,adapter,state,row,require_nonzero=False)
                except features.ChoiceFeatureGateError as error:
                    lexical.dump(out/f'zero-gate-{i}.json',error.report);raise
                lexical.dump(out/f'zero-gate-{i}.json',zero_gate)
                if zero_gate['status']!='exact-gate-passed':raise ValueError('Zero-B base exact gate failed')
                item['zeroBInference']=inference_comparison(bb,adapter,state,row,zero_feature,zero_gate,out/f'inference-zero-{i}.json')
                adapter.lora_b=mx.full((4,4096),1e-5,dtype=mx.float32)
                adapted,_=features.prepare_state(bb,adapter,row,adapter_enabled=True)
                base_cache=state_hash(state['cache']);adapted_cache=state_hash(adapted['cache'])
                if base_cache!=adapted_cache:raise ValueError('Final-down adapter changed complete native prefix cache')
                adapted=None;item['baseVsNonzeroBPrefixStateExact']=True;item['sourceStateHash']=base_cache
                try:
                    gate,feature=features.probe(bb,adapter,state,row)
                    if feature is not None:item['nonzeroBInference']=inference_comparison(bb,adapter,state,row,feature,gate,out/f'inference-nonzero-{i}.json')
                except features.ChoiceFeatureGateError as error:
                    lexical.dump(out/f'gate-{i}.json',error.report);raise
                finally:
                    for key,value in initial.items():setattr(adapter,key,value)
                    mx.eval(adapter.parameters());mx.synchronize()
                lexical.dump(out/f'gate-{i}.json',gate)
                if gate['status']!='exact-gate-passed' or feature is None:raise ValueError('New single-choice exact gate failed; zero updates allowed')
                archive=out/f'feature-{i}.npz';np.savez_compressed(archive,**{k:feature[k] for k in features.FEATURE_KEYS})
                metadata={k:v for k,v in feature.items() if k not in features.FEATURE_KEYS};lexical.dump(archive.with_suffix('.json'),metadata)
                with np.load(archive) as arrays:restored={**metadata,**{k:arrays[k] for k in features.FEATURE_KEYS}}
                features.validate_features(restored)
                if any(not np.array_equal(restored[k],feature[k]) for k in features.FEATURE_KEYS):raise ValueError('Exact feature archive reload failed')
                item.update(status='completed',zeroGateSha256=lexical.sha(out/f'zero-gate-{i}.json'),gateSha256=lexical.sha(out/f'gate-{i}.json'),archiveSha256=lexical.sha(archive),metadataSha256=lexical.sha(archive.with_suffix('.json')),zeroBRestored=not bool(np.any(np.array(adapter.lora_b)!=0)))
                report['completedGates']+=1;state=None;gc.collect();mx.clear_cache();save()
            report['status']='completed-two-inference-controls'
    except BaseException as error:
        primary=error
        for item in report['roster']:
            if item['status']=='running':item.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
    finally:
        try:
            if bb is not None:
                try:
                    if adapter is not None:model.layers[-1].mlp.down_proj=adapter.linear
                    after=parameter_hashes(model.language_model);report['frozen927After']=after
                    report['frozen927HashesUnchanged']=before is not None and len(after)==927 and before==after
                    report['memory']=bb.runtime_memory()
                    if not report['frozen927HashesUnchanged']:raise ValueError('Frozen927 integrity failed')
                except BaseException as error:
                    report.update(status='failed-shutdown-integrity',shutdownError={'type':type(error).__name__,'message':str(error)})
                    if primary is None:primary=error
                finally:
                    mx=bb.mx;state=adapted=adapter=model=bb=None;gc.collect()
                    try:mx.clear_cache();mx.synchronize();report['shutdownCompleted']=True
                    except BaseException as error:
                        report.update(status='failed-shutdown',shutdownError={'type':type(error).__name__,'message':str(error)})
                        if primary is None:primary=error
            try:lexical.check_pins(packet['sha256']);report['sourcePinsUnchanged']=True
            except BaseException as error:
                report.update(status='failed-source-integrity',sourceError={'type':type(error).__name__,'message':str(error)})
                if primary is None:primary=error
        finally:
            report['elapsedSeconds']=time.monotonic()-started;save();resources.close()
    if primary is not None:raise primary
    if report['status']!='completed-two-inference-controls' or report['completedGates']!=2 or not report['frozen927HashesUnchanged'] or not report['shutdownCompleted']:raise RuntimeError('Native pilot lifecycle incomplete')
    return report

def main(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--plan',type=Path);p.add_argument('--plan-sha256');p.add_argument('--run',action='store_true');p.add_argument('--native-owner',choices=['root']);a=p.parse_args(argv)
    if a.prepare:
        if a.run:p.error('prepare and native run are separate')
        packet=prepare(a.out);print(json.dumps({'status':'cpu-plan-frozen','planSha256':lexical.sha(a.out/'plan.json'),'rows':len(packet['rows']),'nativeCalls':0}))
    else:
        if not a.plan or not a.plan_sha256:p.error('plan and exact checksum required')
        print(json.dumps(run(a.plan,a.plan_sha256,a.out,execute=a.run,owner=a.native_owner)))

if __name__=='__main__':main()
