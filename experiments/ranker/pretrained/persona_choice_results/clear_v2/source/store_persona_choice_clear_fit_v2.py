"""One bounded clear128+frozen-base distillation objective; CPU imports only.

Legacy authored priorities never enter the optimizer. Source metadata preserves
their historical bytes; only pure X/H/Y and frozen-B0 soft targets are consumed.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time
from . import store_persona_choice_train_v1 as original
from . import store_persona_choice_train_diagnostic_v1 as diagnostic
from . import store_persona_choice_scoring_v1 as scoring

HERE=original.HERE;ROOT=original.ROOT;MAIN=original.MAIN
VERSION='persona-choice-clear-distillation-fit.v2'
RECIPE={**original.RECIPE,'trainProfilesExpected':128,'featuresExpected':256,'epochs':14,'updatesExpected':3584,
        'teacherProfiles':320,'teacherViews':640,'teacherWeight':.25,'teacherTemperature':1.,'teacherSeedOffset':1000,
        'loss':'CE(audited128 maximum-grade labels) + .25 CE(frozen-B0 legacy320 soft targets)',
        'oldAuthoredPriorityLabelsInOptimizer':False,'maxNativeSeconds':900,'selection':'one fixed final3584 checkpoint; no arms/earlystop',
        'oneNativeRunThenIntegrationQA':True}
OLD_PLAN_SHA=diagnostic.PLAN_SHA
sha=original.pilot.lexical.sha;dump=original.pilot.lexical.dump
def read(p):return json.loads(Path(p).read_text())
def teacher_weights(values):
    if len(values)!=16 or any(not math.isfinite(x) for x in values):raise ValueError('16finite actual B0 route scores required')
    peak=max(values);exp=[math.exp(v-peak) for v in values];total=sum(exp);return [v/total for v in exp]
def schedules():
    ce=original.features.fixed_orders(256,14,19083)
    teacher=original.features.fixed_orders(640,6,20083)[:3584]
    if len(ce)!=3584 or len(teacher)!=3584:raise ValueError('Fixed3584 paired schedule required')
    return ce,teacher

def prepare_packet(curated,curated_sha,evaluation,evaluation_sha,out,driver_file,test_file):
    from transformers import AutoTokenizer
    from . import store_qwen_lexical_choice_inputs_v2 as formatter
    out=original.pilot.check_output(out);run=ROOT/'data/ranker/pretrained/persona-choice-train-v1-attempt-1'
    oldplan=run/'plan.json'
    if sha(oldplan)!=OLD_PLAN_SHA:raise ValueError('Exact original basis plan required')
    prior=read(oldplan);original.pilot.lexical.check_pins(prior['sha256']);extraction=read(run/'EXTRACTION_FROZEN.json');report=read(run/'report.json')
    if report['status']!='completed-fixed-fit-and-predictions' or report['frozen927HashesUnchanged'] is not True or report['shutdownCompleted'] is not True or extraction['allGatesExact'] is not True:raise ValueError('Completed original native basis required')
    curated=Path(curated).resolve();evaluation=Path(evaluation).resolve()
    pins={**prior['sha256'],**original.frozen_files(curated,curated_sha),**original.frozen_files(evaluation,evaluation_sha,name='MANIFEST.json')}
    people=read(curated/'profiles.train.public.json')['profiles'];labels=read(curated/'labels.train.json')['labels'];gold={r['profileId']:r for r in labels}
    if len(people)!=128 or len(gold)!=128:raise ValueError('Exactly128 audited clear TRAIN profiles required')
    old_by={r['caseId']:r for r in prior['rows']};entries={r['caseId']:r for r in extraction['rows']}
    original128={p['profileId'] for p in read(HERE/'synthetic_history_train_expansion_v4_audited/profiles.train.public.json')['profiles']}
    if set(gold)!=original128 or {p['profileId'] for p in people}!=original128:raise ValueError('Same exact audited128 identities required')
    rows=[];reuse={}
    for p in people:
        for order in ('canonical','reversed'):
            case=p['profileId']+'-'+order;row=deepcopy(old_by[case])
            if row['publicRequest']!=p['request']:raise ValueError('Audited clear public bytes changed')
            weights=original.features.max_grade_targets(gold[p['profileId']]['relevanceGrades'],[c['productId'] for c in row['choices']])
            if weights!=row['targetWeights']:raise ValueError('Clear128 product grades changed')
            row.update(objective='clear-CE',targetPurpose='audited-clear128-maximum-grade-TRAIN-supervision')
            rows.append(row);reuse[case]=entries[case]
    for old in prior['rows']:
        if old['split']!='train' or old['profileId'] in original128:continue
        row=deepcopy(old);entry=entries[row['caseId']]
        row.update(objective='frozen-base-teacher',targetWeights=teacher_weights(entry['frozenBaseRouteScores']),targetPurpose='frozen-B0-legacy-soft-target-distillation-no-authored-priority')
        rows.append(row);reuse[row['caseId']]=entry
    if len(rows)!=896 or sum(r['objective']=='clear-CE' for r in rows)!=256:raise ValueError('Exactly256 CE+640 teacher views required')
    tokenizer=AutoTokenizer.from_pretrained(original.pilot.lexical.MODEL,local_files_only=True,trust_remote_code=False);catalog=formatter.load_public_catalog()
    future_groups=set();fresh=[]
    for filename,split,count in [('main.public.json','evaluation-main',32),('replacement.public.json','evaluation-replacement',4),('controls.public.json','evaluation-controls',96)]:
        group=read(evaluation/filename)['profiles']
        if len(group)!=count:raise ValueError('Fixed32+4+96 future PUBLIC roster required')
        for p in group:
            future_groups.add(p['userGroupId'])
            for order in ('canonical','reversed'):
                row=formatter.build_row(p['request'],catalog,tokenizer,p['profileId']+'-'+order,order)
                row.update(profileId=p['profileId'],userGroupId=p['userGroupId'],publicRequest=p['request'],split=split,sourceProfileId=p.get('sourceProfileId'),controlName=p.get('controlName'),
                    routeTokenIds=[c['tokenId'] for c in row['choices']],targetWeights=[1.]+[0.]*15,targetPurpose='arbitrary-first-route-numerical-control-not-evaluation-gold')
                original.features.choice_contract(row);original.cache.checked_prefix(row);fresh.append(row)
    if future_groups&{r['userGroupId'] for r in rows}:raise ValueError('Future evaluation groups overlap TRAIN')
    rows.extend(fresh)
    for old in prior['rows']:
        if old['split'] in ('development-history','demo','demo-default-query'):
            rows.append(deepcopy(old));reuse[old['caseId']]=entries[old['caseId']]
    if len(rows)!=1240 or len(reuse)!=976 or len({r['caseId'] for r in rows})!=1240:raise ValueError('Exact976 reuse+264 fresh unique rows required')
    for path in [Path(__file__),Path(driver_file),Path(test_file),oldplan,run/'EXTRACTION_FROZEN.json',run/'report.json',Path(diagnostic.__file__)]:pins[str(path.resolve())]=sha(path)
    reuse_files={}
    for case,entry in reuse.items():
        for key in ('archive','metadata','gate'):reuse_files[str((run/entry[key]).resolve())]=entry[key+'Sha256']
    ce,teacher=schedules();packet={'version':'persona-choice-clear-training.v2','recipe':RECIPE,'fitRecipe':RECIPE,'rows':rows,'expectedCounts':original.COUNTS,
        'sha256':pins,'reuseFileSha256':reuse_files,'modelFiles':prior['modelFiles'],'originalRun':str(run),'originalPlanSha256':OLD_PLAN_SHA,'reuseEntries':reuse,
        'futurePublicEvaluationFrozenBeforeFit':True,'privateEvaluationLabelsParsed':False,'TRAINLabelsParsed':True,'nativeCalls':0,'optimizerUpdates':0,
        'canonicalOrderIsPrimary':True,'reversedOrderRobustnessOnly':True,'initialInferenceParityProofSha256':prior['initialInferenceParityProofSha256'],
        'reuseScope':'original basis/gates only; changed teacher weights are not claimed as newly gated per-row targets',
        'ceOrder':ce,'teacherOrder':teacher,'variantScope':prior['variantScope'],'baselines':'frozen-base lexical native logits;128CE TRAINprior;latest-owned-repeat;public input controls'}
    original.pilot.lexical.check_pins(pins);out.mkdir(parents=True);dump(out/'plan.json',packet);dump(out/'PLAN_FROZEN.json',{'version':packet['version'],'planSha256':sha(out/'plan.json'),'nativeCalls':0,'optimizerUpdates':0,'privateEvaluationLabelsParsed':False})
    return packet

def objective_item(root,entry,row):
    # Critical one-way projection: original legacy targetWeights never returned.
    pure=diagnostic.load_public_feature(root,entry)
    return {**pure,'targetWeights':row['targetWeights'],'singleChoiceGradientGatePassed':True}

def pair_value(backbone,adapter,clear,teacher):
    left,left_scores=original.features.feature_value(backbone,adapter,clear)
    right,right_scores=original.features.feature_value(backbone,adapter,teacher)
    return left+.25*right,{'clearLoss':left,'teacherLoss':right,'clearScores':left_scores,'teacherScores':right_scores}

class PairedObjectiveGateError(RuntimeError):
    def __init__(self,report):self.report=report;super().__init__(report['error']['message'])

def paired_native_gate(bb,adapter,clear_row,teacher_row,clear_state,teacher_state,clear_item,teacher_item):
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten
    from .parallel_decode import clone_prefix_state
    from .native_batch_diagnosis import state_hash
    from .semantic_copy_gradient_features import _host_tensor
    mx=bb.mx;states=(clear_state,teacher_state);rows=(clear_row,teacher_row)
    before=[state_hash(s['cache']) for s in states];parameters={k:getattr(adapter,k) for k in original.features.GRADIENT_KEYS}
    report={'status':'running','clearCaseId':clear_row['caseId'],'teacherCaseId':teacher_row['caseId'],'optimizerUpdates':0,
        'sourceStateHashesBefore':before,'adapterBefore':{k:_host_tensor(mx,v)[1] for k,v in parameters.items()},
        'teacherTargetsSource':'immutable original B0 lexical logits; no legacy authored priorities'}
    def native():
        values=[]
        for state,row in zip(states,rows,strict=True):
            contract=original.features.choice_contract(row);branch=clone_prefix_state(bb,state)
            hidden=bb.model.model(mx.array([contract['suffixTokenIds']]),cache=branch['cache'])
            values.append(original.features.decision_loss(mx,original.features.native_projection(bb.model,hidden[:,-1,:]),row['routeTokenIds'],row['targetWeights']))
        return values[0][0]+.25*values[1][0],{'clearLoss':values[0][0],'teacherLoss':values[1][0],'clearScores':values[0][1],'teacherScores':values[1][1]}
    def evaluate(fn):
        (loss,aux),grad=nn.value_and_grad(adapter,fn)();mx.eval(loss,aux,grad);mx.synchronize();bb.check_memory()
        return {'loss':_host_tensor(mx,loss)[1],'aux':{k:_host_tensor(mx,v)[1] for k,v in aux.items()},'gradients':{k:_host_tensor(mx,v)[1] for k,v in tree_flatten(grad)}}
    try:
        direct=evaluate(native);report['native']=direct
        features=evaluate(lambda:pair_value(bb,adapter,clear_item,teacher_item));report['features']=features
        exact=direct==features
        if not exact:raise ValueError('Paired CE+.25teacher native/feature loss/logits/A+B gradients not EXACT')
        report.update(status='exact-paired-objective-gate-passed',exact=True,successfulNativeSuffixForwards=2)
        return report
    except BaseException as error:
        report.update(status='failed-paired-objective-gate',error={'type':type(error).__name__,'message':str(error)})
        raise PairedObjectiveGateError(report) from error
    finally:
        for k,v in parameters.items():setattr(adapter,k,v)
        report['sourceStateHashesAfter']=[state_hash(s['cache']) for s in states]
        report['adapterAfter']={k:_host_tensor(mx,v)[1] for k,v in parameters.items()}
        report['sourceStatePreserved']=report['sourceStateHashesAfter']==before;report['adapterPreserved']=report['adapterAfter']==report['adapterBefore']
        if not report['sourceStatePreserved'] or not report['adapterPreserved']:
            report.update(status='failed-paired-objective-gate-cleanup',error={'type':'ValueError','message':'Paired native cache/adapter changed'})
            raise PairedObjectiveGateError(report)

def initial_controls(bb,adapter,packet,extraction,out):
    from .native_batch_diagnosis import state_hash
    mx=bb.mx;by={r['caseId']:r for r in packet['rows']};entries={e['caseId']:e for e in extraction['rows']}
    clear=[r for r in packet['rows'] if r.get('objective')=='clear-CE'][:2];teacher=[r for r in packet['rows'] if r.get('objective')=='frozen-base-teacher'][:2]
    initial={k:getattr(adapter,k) for k in original.features.GRADIENT_KEYS};receipts=[]
    try:
        for index,(left,right) in enumerate(zip(clear,teacher,strict=True)):
            states=[]
            for row in (left,right):
                source=original.cache.catalog_state(bb,adapter,row);state,_=original.cache.profile_state(bb,adapter,row,source)
                adapter.lora_b=mx.full((4,4096),1e-5,dtype=mx.float32)
                enabled=original.cache.catalog_state(bb,adapter,row,enabled=True);other,_=original.cache.profile_state(bb,adapter,row,source,enabled=True)
                if state_hash(source['cache'])!=state_hash(enabled['cache']) or state_hash(state['cache'])!=state_hash(other['cache']):raise ValueError('New objective prefix independence failed')
                adapter.lora_b=initial['lora_b'];states.append(state)
            items=[objective_item(out,entries[row['caseId']],row) for row in (left,right)]
            for name,value in [('zero',initial['lora_b']),('nonzero',mx.full((4,4096),1e-5,dtype=mx.float32))]:
                adapter.lora_b=value
                path=Path(out)/f'INITIAL_PAIRED_GATE_{index}_{name}.json'
                try:receipt=paired_native_gate(bb,adapter,left,right,*states,*items)
                except PairedObjectiveGateError as error:dump(path,error.report);raise
                receipt.update(pairIndex=index,adapterControl=name);dump(path,receipt);receipts.append(receipt)
            states=items=None;mx.clear_cache()
        if len(receipts)!=4:raise ValueError('Exactly2paired rows×zero/control objective gates required')
        return receipts
    finally:
        for k,v in initial.items():setattr(adapter,k,v)

def fit(bb,adapter,packet,archive_root,extraction,out):
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    out=original.pilot.check_output(out);out.mkdir(parents=True);mx=bb.mx
    ce=[r for r in packet['rows'] if r.get('objective')=='clear-CE'];teacher=[r for r in packet['rows'] if r.get('objective')=='frozen-base-teacher'];by={r['caseId']:r for r in extraction['rows']}
    if len(ce)!=256 or len(teacher)!=640 or packet['recipe']!=RECIPE:raise ValueError('Exact256CE/640teacher recipe required')
    if (packet['ceOrder'],packet['teacherOrder'])!=schedules():raise ValueError('Frozen paired schedule changed')
    initial={k:np.array(getattr(adapter,k)) for k in original.features.GRADIENT_KEYS}
    if np.any(initial['lora_b']!=0):raise ValueError('Fresh zero B required')
    report={'version':VERSION,'status':'training','actualUpdates':0,'recipe':RECIPE,'nativeBodyForwardsDuringFit':0,'developmentGoldParsed':False,'losses':[]};save=lambda:dump(out/'report.json',report);save()
    dump(out/'FIT_FROZEN.json',{'version':VERSION,'recipe':RECIPE,'ceCases':[r['caseId'] for r in ce],'teacherCases':[r['caseId'] for r in teacher],
         'ceOrder':packet['ceOrder'],'teacherOrder':packet['teacherOrder'],'legacyAuthoredTargetsUsed':False,'optimizerUpdates':0})
    optimizer=optim.Adam(learning_rate=.0001);fn=nn.value_and_grad(adapter,lambda a,b:pair_value(bb,adapter,a,b));started=time.monotonic()
    try:
        for step,(i,j) in enumerate(zip(packet['ceOrder'],packet['teacherOrder'],strict=True)):
            a=objective_item(archive_root,by[ce[i]['caseId']],ce[i]);b=objective_item(archive_root,by[teacher[j]['caseId']],teacher[j]);report['activeCases']=[ce[i]['caseId'],teacher[j]['caseId']]
            (loss,aux),grads=fn(a,b);mx.eval(loss,aux,grads);mx.synchronize();bb.check_memory();flat=dict(tree_flatten(grads))
            if set(flat)!=original.features.GRADIENT_KEYS or not bool(mx.isfinite(loss).item()) or any(not bool(mx.all(mx.isfinite(v)).item()) for v in flat.values()):raise FloatingPointError('Finite exact A/B gradients required')
            optimizer.update(adapter,grads);mx.eval(adapter.parameters(),optimizer.state);mx.synchronize();bb.check_memory();report['actualUpdates']=step+1
            if step%25==0 or step==3583:
                report['losses'].append({'step':step+1,'jointCE':float(loss.item()),'clearCE':float(aux['clearLoss'].item()),'teacherCE':float(aux['teacherLoss'].item())});save()
            a=b=loss=aux=grads=flat=None
        if report['actualUpdates']!=3584:raise ValueError('Exactly3584 paired optimizer steps required')
        final={k:np.array(getattr(adapter,k)) for k in initial};report['changedAdapterTensors']=[k for k in initial if not np.array_equal(initial[k],final[k])]
        if set(report['changedAdapterTensors'])!=original.features.GRADIENT_KEYS:raise ValueError('A/B must actually change')
        checkpoint=out/'adapter.safetensors';mx.save_safetensors(str(checkpoint),{k:getattr(adapter,k) for k in initial});weights=mx.load(str(checkpoint))
        if set(weights)!=set(initial):raise ValueError('Exact A/B checkpoint required')
        for k,v in weights.items():setattr(adapter,k,v)
        mx.eval(adapter.parameters());mx.synchronize()
        if any(not np.array_equal(np.array(getattr(adapter,k)),final[k]) for k in final):raise ValueError('Exact final checkpoint reload failed')
        original.pilot.lexical.check_pins(packet['sha256']);report.update(status='completed-feature-fit',checkpointSha256=sha(checkpoint),reloadExact=True,allFitPinsUnchanged=True,legacyAuthoredTargetsUsed=False)
        return report
    except BaseException as error:report.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
    finally:report['elapsedSeconds']=time.monotonic()-started;save()
