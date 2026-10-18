"""Root-owned lexical decision LoRA extraction/fit; CPU prepare by default.

No private evaluation annotation is parsed. All1240 rows are frozen before any
native action; 896TRAIN feature targets alone can reach the optimizer. The new
catalog-prefix path is gated before the full extraction. No serving is launched.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import json
from pathlib import Path
import time
from . import store_persona_choice_gate_pilot_v2 as pilot
from . import store_persona_choice_features_v2 as features
from . import store_persona_choice_catalog_cache_v1 as cache
from . import store_persona_choice_fit_v1 as fitter
from . import store_persona_choice_scoring_v1 as scoring
from . import store_persona_choice_capture_v1 as public_capture
from . import store_persona_choice_inference_pilot_v1 as inference_pilot
from . import store_persona_context_v1 as guards

HERE=pilot.HERE;ROOT=pilot.ROOT;MAIN=pilot.MAIN
VERSION='persona-choice-lora-training.v1'
COUNTS={'train':896,'evaluation-main':64,'evaluation-replacement':8,'evaluation-controls':192,
        'development-history':40,'demo':20,'demo-default-query':20}
RECIPE={**pilot.RECIPE,'pilotRows':None,'pilotOptimizerUpdates':None,
        'sourcePrefixPath':'two exact721token literalcatalog caches, perprofile continuation to[:-16], actual last16 suffix',
        'wholePromptNoCacheBitParityClaim':False,'maxNativeSeconds':2400,
        'cachedScoringMode':'plain','dynamicMode':'gradient-public-XHY-last16'}

def frozen_files(directory,expected, *, name='manifest.json'):
    directory=Path(directory).resolve();manifest=directory/name
    if pilot.lexical.sha(manifest)!=expected:raise ValueError('Exact reviewed data manifest checksum required')
    value=json.loads(manifest.read_text());pins={str(manifest):expected}
    for filename,checksum in value['files'].items():
        path=(directory/filename).resolve()
        if not path.is_relative_to(directory) or pilot.lexical.sha(path)!=checksum:raise ValueError('Immutable packet file mismatch')
        pins[str(path)]=checksum
    return pins

def prepare(curated,curated_sha,evaluation,evaluation_sha,out):
    from transformers import AutoTokenizer
    from . import store_qwen_lexical_choice_inputs_v2 as formatter
    from .semantic_copy_feature_fit import model_input_receipt
    out=pilot.check_output(out);old=HERE/'synthetic_history_train_contrasts_v2'
    inference_proof=ROOT/'data/ranker/pretrained/persona-choice-inference-pilot-v1-attempt-1/report.json'
    measured=json.loads(inference_proof.read_text())
    measured_plan_file=inference_proof.parent/'plan.json'
    measured_plan=json.loads(measured_plan_file.read_text());pilot.lexical.check_pins(measured_plan['sha256'])
    if (measured['status']!='completed-two-inference-controls' or measured['completedGates']!=2
        or measured['frozen927HashesUnchanged'] is not True or measured['shutdownCompleted'] is not True
        or measured['sourcePinsUnchanged'] is not True):raise ValueError('Actual completed inference-basis mechanical proof required')
    for item in measured['roster']:
        for arm in ('zeroBInference','nonzeroBInference'):
            if not item[arm]['cachedPlainScoreMatchesTrainingFeaturesExact'] or not item[arm]['labelFreeGradientPathExact']:raise ValueError('Measured fixed target-free serving path not exact')
    pins=frozen_files(old,'7ec75bc25321d3c235159a5b609893bbd428b1512c82ab1e2d07446d8247a017')
    pins[str(inference_proof)]=pilot.lexical.sha(inference_proof)
    pins[str(measured_plan_file)]=pilot.lexical.sha(measured_plan_file)
    pins.update(frozen_files(curated,curated_sha))
    pins.update(frozen_files(evaluation,evaluation_sha,name='MANIFEST.json'))
    public=[];labels=[]
    for directory,n in ((old,320),(Path(curated),128)):
        group=json.loads((directory/'profiles.train.public.json').read_text())['profiles']
        gold=json.loads((directory/'labels.train.json').read_text())['labels']
        if len(group)!=n or len(gold)!=n:raise ValueError('Exact320original+128curated TRAIN required')
        public.extend(group);labels.extend(gold)
    gold={r['profileId']:r for r in labels}
    if len(public)!=448 or len(gold)!=448 or set(gold)!={p['profileId'] for p in public}:raise ValueError('Exact448 TRAIN public/private join required')
    incoming=[(p,'train') for p in public]
    for name,split,n in [('main.public.json','evaluation-main',32),('replacement.public.json','evaluation-replacement',4),('controls.public.json','evaluation-controls',96)]:
        group=json.loads((Path(evaluation)/name).read_text())['profiles']
        if len(group)!=n:raise ValueError('Exact public future evaluation denominator required')
        incoming.extend((p,split) for p in group)
    known=HERE/'synthetic_history_v1'
    for name in ('history-only-probes.public.json','demo-people.json','catalog-binding.json'):
        pins[str(known/name)]=pilot.lexical.sha(known/name)
    probes=json.loads((known/'history-only-probes.public.json').read_text())['pairs']
    incoming.extend(({**pair[side],'userGroupId':pair['pairId']},'development-history') for pair in probes for side in ('profileA','profileB'))
    people=json.loads((known/'demo-people.json').read_text())['people']
    for p in people:
        incoming.append(({'profileId':p['id'],'userGroupId':p['id'],'request':p['request']},'demo'))
        q=deepcopy(p['request']);q['query']='Browse the store'
        incoming.append(({'profileId':p['id']+'-default-query','userGroupId':p['id'],'request':q},'demo-default-query'))
    if len({p['profileId'] for p,_ in incoming})!=620:raise ValueError('All620 public profile IDs must be disjoint')
    train_groups={p['userGroupId'] for p,s in incoming if s=='train'}
    if train_groups & {p['userGroupId'] for p,s in incoming if s.startswith('evaluation-')}:raise ValueError('New evaluation groups overlap TRAIN')
    tokenizer=AutoTokenizer.from_pretrained(pilot.lexical.MODEL,local_files_only=True,trust_remote_code=False)
    catalog=formatter.load_public_catalog();rows=[]
    for p,split in incoming:
        for order in ('canonical','reversed'):
            row=formatter.build_row(p['request'],catalog,tokenizer,p['profileId']+'-'+order,order)
            row.update(profileId=p['profileId'],userGroupId=p['userGroupId'],publicRequest=p['request'],split=split,
                       sourceProfileId=p.get('sourceProfileId'),controlName=p.get('controlName'))
            row['routeTokenIds']=[c['tokenId'] for c in row['choices']]
            if split=='train':
                row['targetWeights']=features.max_grade_targets(gold[p['profileId']]['relevanceGrades'],[c['productId'] for c in row['choices']]);row['targetPurpose']='maximum-grade-TRAIN-supervision'
            else:
                row['targetWeights']=[1.]+[0.]*15;row['targetPurpose']='arbitrary-first-route-numerical-control-not-evaluation-gold'
            features.choice_contract(row);cache.checked_prefix(row);rows.append(row)
    if {s:sum(r['split']==s for r in rows) for s in COUNTS}!=COUNTS:raise ValueError('Exact1240 feature roster required')
    for module in (features,pilot,cache,fitter,scoring,public_capture,inference_pilot,guards,formatter):pins[str(Path(module.__file__).resolve())]=pilot.lexical.sha(module.__file__)
    for path in [Path(__file__),HERE/'test_store_persona_choice_train_v1.py',HERE/'store_demo_engine.py',HERE/'store_catalog_v1.py',HERE/'store_function_adaptation.py']:
        pins[str(path)]=pilot.lexical.sha(path)
    pilot_plan=json.loads((ROOT/'data/ranker/pretrained/persona-choice-gate-pilot-prepare-v2/plan.json').read_text())
    # Pin all proven helper/model sources; data files retain their own manifests.
    for path,checksum in pilot_plan['sha256'].items():
        if Path(path).suffix=='.py':pins[path]=checksum
    with pilot.cwd(MAIN):model=model_input_receipt()
    pins.update({str(pilot.lexical.MODEL/name):s for name,s in model['smallFileSha256'].items()})
    pilot.lexical.check_pins(pins)
    packet={'version':VERSION,'recipe':RECIPE,'fitRecipe':pilot.RECIPE,'rows':rows,'expectedCounts':COUNTS,
            'sha256':pins,'modelFiles':model,'futurePublicEvaluationFrozenBeforeFit':True,
            'privateEvaluationLabelsParsed':False,'TRAINLabelsParsed':True,'nativeCalls':0,'optimizerUpdates':0,
            'canonicalOrderIsPrimary':True,'reversedOrderRobustnessOnly':True,
            'initialInferenceParityProofSha256':pilot.lexical.sha(inference_proof),
            'baselines':'frozen-base lexical native logits, TRAINprior, latest-owned-repeat, predeclaredpublic input controls',
            'variantScope':'source-factual binding only; this16product decision adapter does not train variant/image/copy heads'}
    out.mkdir(parents=True);pilot.lexical.dump(out/'plan.json',packet)
    pilot.lexical.dump(out/'PLAN_FROZEN.json',{'version':VERSION,'planSha256':pilot.lexical.sha(out/'plan.json'),'nativeCalls':0,'optimizerUpdates':0,'privateEvaluationLabelsParsed':False})
    return packet

@contextmanager
def owned_model(packet,report,save):
    """Shared native owner through final927 hashes, release and cleanup."""
    from .catalog_fusion.generalization_v2 import native_owner_lock
    from .store_copy_fit_v2 import deadline
    bb=model=adapter=None;before=None;primary=None
    with native_owner_lock(pilot.lexical.LOCK,'root'):
        try:
            if any(pilot.lexical.listening(p) for p in (8766,8767)):raise RuntimeError('Serving active during exclusive model lifecycle')
            pilot.lexical.check_pins(packet['sha256'])
            with pilot.cwd(MAIN):
                from .semantic_copy_feature_fit import verify_model_files
                report['weightVerification']=verify_model_files(packet['modelFiles']);save()
            import mlx.core as mx
            from mlx_lm.generate import generation_stream
            from mlx_lm.tuner.lora import LoRALinear
            from mlx.utils import tree_flatten
            from .backbone import FrozenBackbone
            from .semantic_copy_training import parameter_hashes
            class GuardedBackbone(FrozenBackbone):
                def check_memory(self):
                    super().check_memory()
                    if max(self.mx.get_peak_memory(),self.mx.get_active_memory())>self.memory_budget:raise MemoryError('Native active/peak exceeded7GiB')
            with deadline(RECIPE['maxNativeSeconds'],'choice extraction and fixed fit'),mx.stream(generation_stream):
                mx.random.seed(RECIPE['seed']);bb=GuardedBackbone(pilot.lexical.MODEL,pilot.lexical.MODEL.name,max_length=2048,memory_gib=7)
                model=bb.model;model.freeze();model.eval();before=parameter_hashes(model.language_model)
                if len(before)!=927 or dict(tree_flatten(model.trainable_parameters())):raise ValueError('927frozen language tensors required')
                report['frozen927Before']=before;save()
                adapter=LoRALinear.from_base(model.layers[-1].mlp.down_proj,r=4,scale=20,dropout=0)
                model.layers[-1].mlp.down_proj=adapter;adapter.lora_a=adapter.lora_a.astype(mx.float32);adapter.lora_b=mx.zeros((4,4096),dtype=mx.float32);model.eval()
                handle=[bb,adapter,before]
                try:yield handle
                finally:handle[:]=[None,None,None]
        except BaseException as error:primary=error;raise
        finally:
            cleanup_error=None
            if bb is not None:
                try:
                    if adapter is not None:model.layers[-1].mlp.down_proj=adapter.linear
                    after=parameter_hashes(model.language_model);report['frozen927After']=after
                    report['frozen927HashesUnchanged']=before is not None and len(after)==927 and before==after
                    report['memory']=bb.runtime_memory()
                    if not report['frozen927HashesUnchanged']:raise ValueError('Frozen927 tensor integrity failed')
                except BaseException as error:cleanup_error=error
                finally:
                    mx=bb.mx;bb=model=adapter=None;gc.collect()
                    try:mx.clear_cache();mx.synchronize();report['shutdownCompleted']=True
                    except BaseException as error:cleanup_error=cleanup_error or error
            try:pilot.lexical.check_pins(packet['sha256']);report['sourcePinsUnchanged']=True
            except BaseException as error:cleanup_error=cleanup_error or error
            if cleanup_error:
                report.update(status='failed-cleanup',cleanupError={'type':type(cleanup_error).__name__,'message':str(cleanup_error)})
            save()
            if cleanup_error and primary is None:raise cleanup_error

def extract(backbone,adapter,packet,out,report,save):
    import numpy as np
    from .native_batch_diagnosis import state_hash
    mx=backbone.mx;out=Path(out);(out/'features').mkdir();(out/'gates').mkdir()
    initial={'lora_a':adapter.lora_a,'lora_b':adapter.lora_b};catalog_states={};source_hashes={}
    control=mx.full((4,4096),1e-5,dtype=mx.float32)
    try:
        for order in ('canonical','reversed'):
            row=next(r for r in packet['rows'] if r['order']==order)
            source=cache.catalog_state(backbone,adapter,row);before=state_hash(source['cache'])
            adapter.lora_b=control;other=cache.catalog_state(backbone,adapter,row,enabled=True)
            if state_hash(other['cache'])!=before:raise ValueError('Complete catalog cache differs under nonzero adapter')
            other=None;catalog_states[order]=source;source_hashes[order]=before
        extraction={'status':'extracting','allGatesExact':False,'frozen927HashesUnchanged':False,'rows':[],'successfulExactGateSuffixForwards':0}
        report['extraction']=extraction;report['phase']='two-prefix-path-gates';save()
        for i,row in enumerate(packet['rows']):
            report['featureRoster'][i]['status']='running'
            report['activeCaseId']=row['caseId'];report['activeFeatureIndex']=i;save();pilot.lexical.check_pins(packet['sha256'])
            source=catalog_states[row['order']];adapter.lora_b=control
            state,timing=cache.profile_state(backbone,adapter,row,source)
            if i<2:
                other,_=cache.profile_state(backbone,adapter,row,source,enabled=True)
                if state_hash(other['cache'])!=state_hash(state['cache']):raise ValueError('Complete public profile prefix cache differs with nonzero adapter')
                other=None
            gate_path=out/'gates'/f'{i:04d}.json'
            try:gate,item=features.probe(backbone,adapter,state,row)
            except features.ChoiceFeatureGateError as error:pilot.lexical.dump(gate_path,error.report);raise
            pilot.lexical.dump(gate_path,gate)
            if gate['status']!='exact-gate-passed' or item is None:raise ValueError('Exact catalog-prefix feature gate failed before optimizer')
            extraction['successfulExactGateSuffixForwards']+=2
            if i==1:report['prefixPathTwoRowControlsPassed']=True;report['phase']='all-features';save()
            archive=out/'features'/f'{i:04d}.npz';metadata=archive.with_suffix('.json')
            np.savez_compressed(archive,**{k:item[k] for k in features.FEATURE_KEYS})
            pilot.lexical.dump(metadata,{k:v for k,v in item.items() if k not in features.FEATURE_KEYS})
            with np.load(archive) as arrays:
                if any(not np.array_equal(arrays[k],item[k]) for k in features.FEATURE_KEYS):raise ValueError('Exact NPZ archive reload failed')
            # Frozen-base lexical logits use precisely the same source features
            # and projection path, with pristine A/B restored, before training.
            for k,v in initial.items():setattr(adapter,k,v)
            base_scores=scoring.score(backbone,adapter,scoring.public_item(item));mx.eval(base_scores);mx.synchronize()
            entry={k:row[k] for k in ('caseId','profileId','order','split','targetPurpose','routeTokenIds','targetWeights')}
            entry.update(archive=str(archive.relative_to(out)),metadata=str(metadata.relative_to(out)),gate=str(gate_path.relative_to(out)),
                archiveSha256=pilot.lexical.sha(archive),metadataSha256=pilot.lexical.sha(metadata),gateSha256=pilot.lexical.sha(gate_path),
                frozenBaseRouteScores=base_scores.tolist(),prefixTiming=timing)
            extraction['rows'].append(entry)
            report['featureRoster'][i]['status']='completed'
            if state_hash(source['cache'])!=source_hashes[row['order']]:raise ValueError('Shared catalog source cache mutated')
            state=item=base_scores=None;gc.collect();mx.clear_cache();save()
        if len(extraction['rows'])!=1240:raise ValueError('Exact1240 gated immutable features required')
        from .semantic_copy_training import parameter_hashes
        current={k.replace('.31.mlp.down_proj.linear.','.31.mlp.down_proj.'):v for k,v in parameter_hashes(backbone.model.language_model).items()}
        if current!=report['frozen927Before']:raise ValueError('Frozen927 changed after extraction')
        extraction.update(status='completed-choice-feature-extraction',allGatesExact=True,frozen927HashesUnchanged=True)
        extraction['completedNativePrefillForwardCalls']=backbone.choice_prefill_completed
        extraction['attemptedNativePrefillForwardCalls']=backbone.choice_prefill_attempts
        pilot.lexical.dump(out/'EXTRACTION_FROZEN.json',extraction);return extraction
    finally:
        for k,v in initial.items():setattr(adapter,k,v)
        catalog_states.clear();gc.collect();mx.clear_cache()

def postfit_controls(backbone,adapter,packet,extraction,out):
    """Learned adapter native parity with ORIGINAL immutable feature archives."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from .semantic_copy_gradient_features import _host_tensor
    from .native_batch_diagnosis import state_hash
    by={r['caseId']:r for r in packet['rows']}
    chosen=[r for r in extraction['rows'] if r['split']=='train'][:2]+[r for r in extraction['rows'] if r['split']=='evaluation-main'][:2]
    if len(chosen)!=4:raise ValueError('Two TRAIN and two PUBLIC evaluation controls required')
    mx=backbone.mx;records=[]
    for i,entry in enumerate(chosen):
        row=by[entry['caseId']];source=cache.catalog_state(backbone,adapter,row)
        enabled=cache.catalog_state(backbone,adapter,row,enabled=True)
        if state_hash(source['cache'])!=state_hash(enabled['cache']):raise ValueError('Learned adapter changed catalog prefix state')
        enabled=None;state,_=cache.profile_state(backbone,adapter,row,source)
        enabled,_=cache.profile_state(backbone,adapter,row,source,enabled=True)
        if state_hash(state['cache'])!=state_hash(enabled['cache']):raise ValueError('Learned adapter changed full public profile prefix state')
        enabled=None
        try:gate,item=features.probe(backbone,adapter,state,row)
        except features.ChoiceFeatureGateError as error:pilot.lexical.dump(Path(out)/f'POSTFIT_NATIVE_GATE_{i}.json',error.report);raise
        pilot.lexical.dump(Path(out)/f'POSTFIT_NATIVE_GATE_{i}.json',gate)
        old_gate=json.loads((Path(out)/entry['gate']).read_text())
        if gate['status']!='exact-gate-passed' or gate['featureTensors']!=old_gate['featureTensors']:raise ValueError('Learned native feature tensor receipts differ from original immutable archive')
        original=fitter.load_feature(out,entry)
        (loss,scores),gradients=nn.value_and_grad(adapter,lambda:features.feature_value(backbone,adapter,original))()
        mx.eval(loss,scores,gradients);mx.synchronize();backbone.check_memory()
        actual={'lossTensor':_host_tensor(mx,loss)[1],'gradientTensors':{k:_host_tensor(mx,v)[1] for k,v in tree_flatten(gradients)},'routeLogitsTensor':_host_tensor(mx,scores)[1]}
        direct=gate['arms']['direct']
        if any(actual[k]!=direct[k] for k in actual):raise ValueError('Learned native loss/logits/A+B gradients differ from ORIGINAL archived features')
        inference_path=Path(out)/f'POSTFIT_INFERENCE_CONTROL_{i}.json'
        inference=inference_pilot.inference_comparison(backbone,adapter,state,row,original,gate,inference_path)
        # The helper was originally a TRAIN-only mechanical probe. This fresh
        # receipt explicitly describes the PUBLIC numerical controls as well.
        inference['reference']='single-decision gradient-mode numerical gate; TRAIN labels or fixed PUBLIC arbitrary control, no evaluation gold'
        inference['fixedServingPolicy']={'cachedScoringMode':'plain','dynamicMode':'gradient-public-XHY-last16'}
        pilot.lexical.dump(inference_path,inference)
        if not inference['cachedPlainScoreMatchesTrainingFeaturesExact'] or not inference['labelFreeGradientPathExact']:raise ValueError('Learned adapter violates fixed target-free cached/dynamic serving policy')
        records.append({'caseId':row['caseId'],'split':row['split'],'order':row['order'],'originalArchiveSha256':entry['archiveSha256'],
                        'freshNativeVsOriginalArchiveExact':True,'cacheBaseVsLearnedAdapterExact':True,
                        'inferenceProofSha256':pilot.lexical.sha(inference_path),'targetFreeLearnedInferenceExact':True,
                        'gateSha256':pilot.lexical.sha(Path(out)/f'POSTFIT_NATIVE_GATE_{i}.json')})
        source=state=item=original=None;gc.collect();mx.clear_cache()
    pilot.lexical.dump(Path(out)/'POSTFIT_ORIGINAL_ARCHIVE_PARITY.json',{'records':records,'privateEvaluationLabelsParsed':False,
                      'evaluationTargetPurpose':'fixed arbitrary first route numerical control, never evaluation gold'})
    return records

def export_public_feature(out,row,item):
    import numpy as np
    directory=Path(out)/'public-features';directory.mkdir(exist_ok=True)
    key=features.token_hash(row['tokenIds']);path=directory/(key+'.npz')
    np.savez_compressed(path,**{k:item[k] for k in features.FEATURE_KEYS})
    return key,{'archive':path.name,'archiveSha256':pilot.lexical.sha(path),
                'dtypes':item['dtypes'],'routeTokenIds':item['routeTokenIds'],
                'promptTokenIdsSha256':key,'promptTokens':len(row['tokenIds']),
                'prefixPath':RECIPE['sourcePrefixPath'],'nativeInputSuffixTokens':16}

def decision_prediction(row,scores,catalog):
    from .store_demo_engine import guarded_order
    request,policy,context=guards.checked_context_request(row['publicRequest'],catalog['products'])
    by={c['productId']:float(v) for c,v in zip(row['choices'],scores,strict=True)}
    vector=[by[p['productId']] for p in catalog['products']];order,checks=guarded_order(vector,request,policy,catalog['products'])
    return {'profileId':row['profileId'],'userGroupId':row['userGroupId'],'caseId':row['caseId'],'split':row['split'],
            'orderLabel':row['order'],'sourceProfileId':row.get('sourceProfileId'),'controlName':row.get('controlName'),
            'scores16':vector,'order':order,'rawLearnedOrder':sorted(range(16),key=lambda i:(-vector[i],i)),
            'rankedProductIds':[catalog['products'][i]['productId'] for i in order],
            'guardChecks':checks,'copyGenerated':False,'variantsOrImagesScored':False}

def run(plan_path,checksum,out, *, execute=False,owner=None):
    if pilot.lexical.sha(plan_path)!=checksum:raise ValueError('Exact frozen source/data plan SHA required')
    packet=json.loads(Path(plan_path).read_text())
    if packet['version']!=VERSION or packet['recipe']!=RECIPE or packet['expectedCounts']!=COUNTS:raise ValueError('Frozen1240-row recipe required')
    pilot.lexical.check_pins(packet['sha256'])
    if not execute:return {'status':'cpu-preflight-only','nativeCalls':0,'optimizerUpdates':0}
    if owner!='root':raise ValueError('Root is sole native owner')
    if any(pilot.lexical.listening(p) for p in (8766,8767)):raise RuntimeError('Serving must retire before exclusive fit')
    out=pilot.check_output(out);out.mkdir(parents=True);pilot.lexical.dump(out/'plan.json',packet)
    started=time.monotonic();report={'version':VERSION,'status':'initializing','phase':'initialization','actualUpdates':0,
        'planSha256':checksum,'privateEvaluationLabelsParsed':False,'frozen927HashesUnchanged':None,'shutdownCompleted':False,
        'featureRoster':[{'caseId':r['caseId'],'split':r['split'],'status':'unattempted'} for r in packet['rows']]}
    save=lambda:pilot.lexical.dump(out/'report.json',report);save()
    try:
        with owned_model(packet,report,save) as handle:
            extraction=extract(handle[0],handle[1],packet,out,report,save);report['phase']='fixed3584fit';save()
            fit_packet={**packet,'recipe':pilot.RECIPE}
            result=fitter.fit(handle[0],handle[1],fit_packet,out,extraction,out/'fit');report['actualUpdates']=result['actualUpdates'];report['fit']=result
            report['phase']='postfit-native-original-archive-controls';save()
            report['postfitNativeControls']=postfit_controls(handle[0],handle[1],packet,extraction,out)
            catalog=guards.load_catalog_features(MAIN/'data/ranker/pretrained/actual-store-function-v3-attempt-1')
            by={r['caseId']:r for r in packet['rows']};learned=[];base=[];public_cache={};prior_predictions=[];recent_predictions=[]
            product_ids=[p['productId'] for p in catalog['products']]
            train_canonical=[r for r in packet['rows'] if r['split']=='train' and r['order']=='canonical']
            if len(train_canonical)!=448:raise ValueError('Exactly448 distinct TRAIN profiles for fixed prior')
            prior={pid:0. for pid in product_ids}
            for row in train_canonical:
                for choice,weight in zip(row['choices'],row['targetWeights'],strict=True):prior[choice['productId']]+=weight/448
            pilot.lexical.dump(out/'BASELINE_DEFINITIONS_FROZEN.json',{'training-popularity':'mean448canonical TRAIN uniform maximum-grade target distributions',
                'latest-owned-repeat':'TRAINprior plus10 at most recent actual public purchasedSKU byoccurredAt; sameguards/exacteligibleidentity',
                'priorScoresByProductId':prior,'privateEvaluationLabelsParsed':False})
            report['phase']='freeze-all-nonfit-predictions';save()
            for entry in extraction['rows']:
                if entry['split']=='train':continue
                row=by[entry['caseId']];item=fitter.load_feature(out,entry)
                scores=scoring.score(handle[0],handle[1],scoring.public_item(item));handle[0].mx.eval(scores);handle[0].mx.synchronize();handle[0].check_memory()
                learned.append(decision_prediction(row,scores.tolist(),catalog));base.append(decision_prediction(row,entry['frozenBaseRouteScores'],catalog))
                prior_predictions.append(decision_prediction(row,[prior[c['productId']] for c in row['choices']],catalog))
                recent=dict(prior)
                purchases=[e for e in row['publicRequest']['publicHistory'] if e['eventType']=='purchase' and e.get('productId') in recent]
                if purchases:recent[max(purchases,key=lambda e:e['occurredAt'])['productId']]+=10
                recent_predictions.append(decision_prediction(row,[recent[c['productId']] for c in row['choices']],catalog))
                if row['split'] in ('demo','demo-default-query') and row['order']=='canonical':
                    key,metadata=export_public_feature(out,row,item)
                    if key in public_cache:raise ValueError('Public20 cache keys must be exact distinct prompt identities')
                    public_cache[key]=metadata
            if len(learned)!=344:raise ValueError('All344 nonfit order-specific predictions required')
            pilot.lexical.dump(out/'learned-nonfit-predictions.json',learned);pilot.lexical.dump(out/'frozen-base-nonfit-predictions.json',base)
            pilot.lexical.dump(out/'training-popularity-nonfit-predictions.json',prior_predictions);pilot.lexical.dump(out/'latest-owned-repeat-nonfit-predictions.json',recent_predictions)
            pilot.lexical.dump(out/'ALL_PREDICTIONS_FROZEN.json',{'version':VERSION,'checkpointSha256':result['checkpointSha256'],
                'learnedPredictionsSha256':pilot.lexical.sha(out/'learned-nonfit-predictions.json'),
                'frozenBasePredictionsSha256':pilot.lexical.sha(out/'frozen-base-nonfit-predictions.json'),
                'baselineFilesSha256':{name:pilot.lexical.sha(out/(name+'-nonfit-predictions.json')) for name in ('training-popularity','latest-owned-repeat')},
                'nonfitPromptRecords':344,'privateEvaluationLabelsParsed':False,'canonicalIsPrimary':True,'reversedRobustnessOnly':True,
                'copyOrPhysicalImageOrServingReadinessClaim':False})
            if len(public_cache)!=20:raise ValueError('Exact20 public canonical prompt cache entries required')
            pilot.lexical.dump(out/'public-features/index.json',{'version':'persona-choice-public-feature-cache.v1','entries':public_cache,
                 'teacherOrTargetsIncluded':False,'key':'exact rendered whole prompt token SHA; no case/person identifiers',
                 'modelRevision':pilot.lexical.MODEL.name,'formatterSha256':packet['sha256'][str((HERE/'store_qwen_lexical_choice_inputs_v2.py').resolve())],
                 'baseModelFiles':packet['modelFiles'],'scoreSourceSha256':pilot.lexical.sha(scoring.__file__),
                 'featureSourceSha256':pilot.lexical.sha(features.__file__)})
            report['completedNativePrefillForwardCalls']=handle[0].choice_prefill_completed
            report['attemptedNativePrefillForwardCalls']=handle[0].choice_prefill_attempts
            report['successfulExactGateSuffixForwards']=extraction['successfulExactGateSuffixForwards']+8
            report['successfulInferenceControlSuffixForwards']=8
            report['status']='completed-fixed-fit-and-predictions';save()
    except BaseException as error:
        for entry in report['featureRoster']:
            if entry['status']=='running':entry.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        fit_report=out/'fit/report.json'
        if fit_report.exists():
            progress=json.loads(fit_report.read_text());report['actualUpdates']=progress['actualUpdates'];report['fitProgress']=progress
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
    finally:report['elapsedSeconds']=time.monotonic()-started;save()
    if report['actualUpdates']!=3584 or not report['frozen927HashesUnchanged'] or not report['shutdownCompleted']:raise RuntimeError('Full native lifecycle incomplete')
    certificate={'version':'persona-choice-inference-parity.v1','adapterSha256':report['fit']['checkpointSha256'],
        'cachedScoringMode':'plain','dynamicMode':'gradient-public-XHY-last16',
        'cachedScoreMatchesTrainingFeaturesExact':True,'dynamicMatchesCachedFeatureScoresExact':True,
        'zeroAndNonzeroControlPassed':True,'learnedPostfitPassed':all(r['targetFreeLearnedInferenceExact'] for r in report['postfitNativeControls']),
        'helperSourceSha256':{'score':pilot.lexical.sha(scoring.__file__),'capture':pilot.lexical.sha(public_capture.__file__)},
        'zeroNonzeroProofSha256':packet['initialInferenceParityProofSha256'],
        'postfitProofSha256':[r['inferenceProofSha256'] for r in report['postfitNativeControls']]}
    pilot.lexical.dump(out/'inference-parity.json',certificate)
    canonical=next(r for r in packet['rows'] if r['order']=='canonical')
    pilot.lexical.dump(out/'serving.json',{'version':'persona-choice-adapter-serving.v1',
        'scope':'candidate research; offline and serving acceptance pending; no default promotion',
        'adapter':{'file':'fit/adapter.safetensors','sha256':report['fit']['checkpointSha256'],'rank':4,'scale':20,
                   'target':'language_model.model.layers.31.mlp.down_proj'},
        'publicCache':{'file':'public-features/index.json','sha256':pilot.lexical.sha(out/'public-features/index.json')},
        'modelRevision':pilot.lexical.MODEL.name,
        'formatterSha256':packet['sha256'][str((HERE/'store_qwen_lexical_choice_inputs_v2.py').resolve())],
        'catalogSha256':packet['sha256'][str(HERE/'synthetic_history_v1/catalog-binding.json')],
        'routeTokenIds':canonical['routeTokenIds'],'productIds':[c['productId'] for c in canonical['choices']],
        'catalogPrefixTokenIdsSha256':{order:features.token_hash(next(r for r in packet['rows'] if r['order']==order)['catalogPrefixTokenIds']) for order in ('canonical','reversed')},
        'actualSuffixTokens':16,'base927Unchanged':True,'trainedUpdates':3584,
        'cachedScoringMode':'plain','dynamicMode':'gradient-public-XHY-last16',
        'inferenceParity':{'file':'inference-parity.json','sha256':pilot.lexical.sha(out/'inference-parity.json')},
        'postfitNativeControlsExact':len(report['postfitNativeControls'])==4 and all(r['freshNativeVsOriginalArchiveExact'] and r['cacheBaseVsLearnedAdapterExact'] for r in report['postfitNativeControls']),
        'scoreSourceSha256':pilot.lexical.sha(scoring.__file__),'nativeReportSha256':pilot.lexical.sha(out/'report.json'),
        'inferenceTargetFieldsRequired':False})
    return report

def main(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--curated-train',type=Path);p.add_argument('--curated-manifest-sha256');p.add_argument('--evaluation',type=Path);p.add_argument('--evaluation-manifest-sha256')
    p.add_argument('--plan',type=Path);p.add_argument('--plan-sha256');p.add_argument('--run',action='store_true');p.add_argument('--native-owner',choices=['root']);a=p.parse_args(argv)
    if a.prepare:
        if a.run:p.error('Prepare and native run are separate')
        if not all((a.curated_train,a.curated_manifest_sha256,a.evaluation,a.evaluation_manifest_sha256)):p.error('Both frozen source manifests required')
        packet=prepare(a.curated_train,a.curated_manifest_sha256,a.evaluation,a.evaluation_manifest_sha256,a.out)
        print(json.dumps({'status':'cpu-plan-frozen','planSha256':pilot.lexical.sha(a.out/'plan.json'),'rows':len(packet['rows'])}))
    else:
        if not a.plan or not a.plan_sha256:p.error('Exact plan and SHA required')
        r=run(a.plan,a.plan_sha256,a.out,execute=a.run,owner=a.native_owner);print(json.dumps({k:r[k] for k in ('status','actualUpdates')} if a.run else r))

if __name__=='__main__':main()
