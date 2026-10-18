"""Frozen TRAIN896 choice-score diagnostic; default CPU preflight, never a fit.

Scores original gated X/H/Y through the measured plain feature-scoring path.
Every base/final prediction is frozen before separate TRAIN label files open.
The original source plan/gate metadata already contains known TRAIN weights;
those bytes are not claimed unseen and are never passed to the score API.
No evaluation label path or inference case selection is accepted.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from . import store_persona_choice_train_v1 as training
from . import store_persona_choice_scoring_v1 as scoring

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; MAIN=training.MAIN
VERSION='persona-choice-TRAIN-diagnostic.v1'
PLAN_SHA='44a7bbd55ee6e662d23ecaf9cf2ed4fe4b73a78b3cea2c9193834691e512ec5b'
ADAPTER_SHA='1e88bdc919e8d8dede8cc54f8f39aff6c8af297c7e496504d7ab6cada63c148b'
RECIPE={'TRAINProfiles':448,'TRAINOrderRows':896,'scoreArms':['seeded-A-zero-B','learned-final'],
        'predictions':1792,'scoringMode':'plain-original-XHY-full16-norm-native-full-LM16-gather',
        'rank':4,'scale':20.,'seed':19083,'memoryGiB':7,'deadlineSeconds':300,
        'optimizerUpdates':0,'nativeBodyForwards':0,'lossScope':'post-freeze TRAIN-only diagnostic; no fit or selection'}
SOURCES=[Path(__file__),Path(training.__file__),Path(scoring.__file__),HERE/'test_store_persona_choice_train_diagnostic_v1.py']
sha=training.pilot.lexical.sha; dump=training.pilot.lexical.dump

def read(path):return json.loads(Path(path).read_text())
def local(root,name):
    p=(Path(root)/name).resolve()
    if Path(name).is_absolute() or not p.is_relative_to(Path(root).resolve()) or p.is_symlink():raise ValueError('Exact local immutable run artifact required')
    return p
def pincheck(pins):training.pilot.lexical.check_pins(pins)

def prepare(run,source_plan,out):
    """Hash receipts/source and project TRAIN public identities, not label files."""
    run=Path(run).resolve();source_plan=Path(source_plan).resolve();out=training.pilot.check_output(out)
    if sha(source_plan)!=PLAN_SHA:raise ValueError('Exact original44a7 frozen source plan required')
    full=read(source_plan); pincheck(full['sha256'])
    if sha(run/'plan.json')!=PLAN_SHA:raise ValueError('Run/source plan mismatch')
    report=read(run/'report.json');extraction=read(run/'EXTRACTION_FROZEN.json');fit=read(run/'fit/report.json');schedule=read(run/'fit/FIT_FROZEN.json')
    if (report['status']!='completed-fixed-fit-and-predictions' or report['actualUpdates']!=3584 or report['fit']['checkpointSha256']!=ADAPTER_SHA or
        any(report[k] is not True for k in ('frozen927HashesUnchanged','shutdownCompleted','sourcePinsUnchanged')) or
        extraction['status']!='completed-choice-feature-extraction' or extraction['allGatesExact'] is not True or
        fit['reloadExact'] is not True or fit['checkpointSha256']!=ADAPTER_SHA or sha(run/'fit/adapter.safetensors')!=ADAPTER_SHA):raise ValueError('Exact completed native run/checkpoint/cleanup required')
    by={r['caseId']:r for r in full['rows'] if r['split']=='train'}
    entries=[r for r in extraction['rows'] if r['split']=='train']
    if len(by)!=896 or len(entries)!=896 or {r['caseId'] for r in entries}!=set(by):raise ValueError('Exact896 TRAIN row roster required')
    roots={'old320':HERE/'synthetic_history_train_contrasts_v2','new128':HERE/'synthetic_history_train_expansion_v4_audited'}
    sources={};label_files={};pins={**full['sha256'],**{str(p.resolve()):sha(p) for p in SOURCES}}
    for name,directory in roots.items():
        public_file=directory/'profiles.train.public.json'; people=read(public_file)['profiles']
        expected=320 if name=='old320' else 128
        if len(people)!=expected:raise ValueError('Fixed TRAIN source counts required')
        sources.update({p['profileId']:name for p in people})
        label=directory/'labels.train.json';label_files[name]={'file':str(label),'sha256':sha(label)}
        if full['sha256'].get(str(label))!=label_files[name]['sha256']:raise ValueError('TRAIN labels must match original pre-fit pins')
    rows=[]
    for entry in entries:
        original=by[entry['caseId']]
        if any(entry[k]!=original[k] for k in ('profileId','order','routeTokenIds','targetPurpose')):raise ValueError('Original TRAIN feature join changed')
        rows.append({k:original[k] for k in ('caseId','profileId','userGroupId','publicRequest','choices','order','routeTokenIds')}
                    |{'source':sources[original['profileId']],**{k:entry[k] for k in ('archive','metadata','gate','archiveSha256','metadataSha256','gateSha256')}})
    if len(sources)!=448 or {r['profileId'] for r in rows}!=set(sources):raise ValueError('Exact448 TRAIN profile join required')
    for p in [source_plan,run/'plan.json',run/'report.json',run/'EXTRACTION_FROZEN.json',run/'fit/report.json',run/'fit/FIT_FROZEN.json',run/'fit/adapter.safetensors']:
        pins[str(p)]=sha(p)
    packet={'version':VERSION,'recipe':RECIPE,'run':str(run),'sourcePlanSha256':PLAN_SHA,'adapterSha256':ADAPTER_SHA,
            'canonicalProductIds':[c['productId'] for c in next(r for r in full['rows'] if r['order']=='canonical')['choices']],
            'rows':rows,'labelFiles':label_files,'sha256':pins,'modelFiles':full['modelFiles'],
            'initialAdapterSha256':schedule['initialAdapterSha256'],'privateEvaluationLabelsParsed':False,
            'separateTRAINLabelFilesParsed':False,'originalPlanContainsKnownTRAINTargets':True,
            'archiveValidation':'all NPZ/file/tensor/exact-gate hashes checked during scoring; CPU prepare binds frozen extraction inventory only'}
    pincheck(pins);out.mkdir(parents=True);dump(out/'plan.json',packet)
    dump(out/'PLAN_FROZEN.json',{'version':VERSION,'planSha256':sha(out/'plan.json'),'nativeCalls':0,'optimizerUpdates':0,'separateTRAINLabelFilesParsed':False})
    return packet

def load_public_feature(root,row):
    import numpy as np
    for name in ('archive','metadata','gate'):
        p=local(root,row[name])
        if sha(p)!=row[name+'Sha256']:raise ValueError('Original feature/gate checksum changed')
    metadata=read(local(root,row['metadata']));gate=read(local(root,row['gate']))
    if (gate['status']!='exact-gate-passed' or gate['sourceSha256']!=sha(training.features.__file__) or gate['sourceStatePreserved'] is not True or
        gate['adapterPreserved'] is not True or not gate['comparisons'] or any(c['exact'] is not True or c['routeLogitsExact'] is not True for c in gate['comparisons'].values()) or
        metadata['caseId']!=row['caseId'] or metadata['routeTokenIds']!=row['routeTokenIds'] or metadata['singleChoiceGradientGatePassed'] is not True):raise ValueError('Actual original exact gate required')
    with np.load(local(root,row['archive']),allow_pickle=False) as arrays:
        if set(arrays.files)!={'x','h','y'}:raise ValueError('Only actual X/H/Y archive arrays required')
        item={k:arrays[k].copy() for k in ('x','h','y')}
    item.update(dtypes=metadata['dtypes'],routeTokenIds=metadata['routeTokenIds']);scoring.validate(item)
    for key in ('x','h','y'):
        value=item[key];actual=hashlib.sha256(item['dtypes'][key].encode()+str(tuple(value.shape)).encode()+value.tobytes()).hexdigest()
        if value.dtype!=np.float32 or not np.isfinite(value).all() or gate['featureTensors'][key]['sha256']!=actual:raise ValueError('Original native tensor hash/finite mismatch')
    return item

@contextmanager
def owned_model(packet,report,save):
    """Same verified loader/adapter basis, one nonnested300sec alarm and main lock."""
    from .catalog_fusion.generalization_v2 import native_owner_lock
    from .store_copy_fit_v2 import deadline
    bb=model=adapter=None;before=None;primary=None
    with native_owner_lock(training.pilot.lexical.LOCK,'root'):
        try:
            if any(training.pilot.lexical.listening(p) for p in (8766,8767)):raise RuntimeError('Serving must retire before diagnostic')
            pincheck(packet['sha256'])
            with deadline(300,'TRAIN projection-only diagnostic'):
                with training.pilot.cwd(MAIN):
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
                with mx.stream(generation_stream):
                    mx.random.seed(19083);bb=GuardedBackbone(training.pilot.lexical.MODEL,training.pilot.lexical.MODEL.name,max_length=2048,memory_gib=7)
                    model=bb.model;model.freeze();model.eval();before=parameter_hashes(model.language_model)
                    if len(before)!=927 or dict(tree_flatten(model.trainable_parameters())):raise ValueError('927 frozen tensors required')
                    report['frozen927Before']=before;save()
                    adapter=LoRALinear.from_base(model.layers[-1].mlp.down_proj,r=4,scale=20,dropout=0)
                    model.layers[-1].mlp.down_proj=adapter;adapter.lora_a=adapter.lora_a.astype(mx.float32);adapter.lora_b=mx.zeros((4,4096),dtype=mx.float32);model.eval()
                    # Any accidental whole-model/attention/SSM forward is forbidden.
                    body_class=type(model.model);body_call=body_class.__call__
                    def forbidden_body(*args,**kwargs):
                        report['attemptedNativeBodyForwards']+=1;raise RuntimeError('TRAIN diagnostic may only call residual norm and frozen LM projection')
                    body_class.__call__=forbidden_body
                    try:yield bb,adapter
                    finally:body_class.__call__=body_call
        except BaseException as error:primary=error;raise
        finally:
            cleanup_error=None
            if bb is not None:
                try:
                    if adapter is not None:model.layers[-1].mlp.down_proj=adapter.linear
                    after=parameter_hashes(model.language_model);report['frozen927After']=after
                    report['frozen927HashesUnchanged']=len(after)==927 and before==after;report['memory']=bb.runtime_memory()
                    if not report['frozen927HashesUnchanged']:raise ValueError('Frozen927 integrity changed')
                except BaseException as error:cleanup_error=error
                finally:
                    mx=bb.mx;bb=model=adapter=None;gc.collect()
                    try:mx.clear_cache();mx.synchronize();report['shutdownCompleted']=True
                    except BaseException as error:cleanup_error=cleanup_error or error
            try:pincheck(packet['sha256']);report['sourcePinsUnchanged']=True
            except BaseException as error:cleanup_error=cleanup_error or error
            if cleanup_error:report.update(status='failed-cleanup',cleanupError={'type':type(cleanup_error).__name__,'message':str(cleanup_error)})
            save()
            if cleanup_error and primary is None:raise cleanup_error

def canonical_prediction(row,arm,values,catalog_order):
    by={c['productId']:float(v) for c,v in zip(row['choices'],values,strict=True)}
    if len(by)!=16 or not all(math.isfinite(v) for v in by.values()):raise ValueError('Finite16 product-bound scores required')
    if len(catalog_order)!=16 or set(catalog_order)!=set(by):raise ValueError('Exact canonical16 product order required')
    order=sorted(catalog_order,key=lambda key:(-by[key],catalog_order.index(key)))
    return {'caseId':row['caseId'],'profileId':row['profileId'],'userGroupId':row['userGroupId'],'source':row['source'],
            'order':row['order'],'arm':arm,'scoresByProductId':by,'rankedProductIds':order}

def subset_metrics(rows,labels):
    result={'rows':len(rows),'maximumGradeTop1':0,'maximumGradeTop3':0,'strongTop1':0,'strongTop3':0,'ownedTop1':0,'routeCE':0.,'targetEntropy':0.,'confusions':{}}
    for row in rows:
        grades=labels[row['profileId']]['relevanceGrades'];highest=max(grades.values());wanted={k for k,v in grades.items() if v==highest};order=row['rankedProductIds'];scores=row['scoresByProductId']
        result['maximumGradeTop1']+=order[0] in wanted;result['maximumGradeTop3']+=bool(wanted&set(order[:3]));result['strongTop1']+=grades[order[0]]>=2;result['strongTop3']+=any(grades[k]>=2 for k in order[:3])
        result['ownedTop1']+=order[0] in labels[row['profileId']]['alreadyOwnedProductIds']
        peak=max(scores.values());lse=peak+math.log(sum(math.exp(v-peak) for v in scores.values()))
        result['routeCE']+=lse-sum(scores[k] for k in wanted)/len(wanted);result['targetEntropy']+=math.log(len(wanted))
        key='|'.join(sorted(wanted))+' → '+order[0];result['confusions'][key]=result['confusions'].get(key,0)+1
    if rows:
        result['routeCE']/=len(rows);result['targetEntropy']/=len(rows)
    return result

def summarize(predictions,packet,labels):
    expected={r['profileId'] for r in packet['rows']}
    if set(labels)!=expected:raise ValueError('Exact448 TRAIN label join required')
    output={'version':VERSION,'scope':'TRAIN resubstitution diagnosis; no quality promotion or new evaluation','arms':{},'purchasePairs':[]}
    for arm in RECIPE['scoreArms']:
        rows=[r for r in predictions if r['arm']==arm]
        output['arms'][arm]={source:{order:subset_metrics([r for r in rows if (source=='all448' or r['source']==source) and r['order']==order],labels)
                                    for order in ('canonical','reversed')} for source in ('all448','old320','new128')}
        by={(r['profileId'],r['order']):r for r in rows}
        output['arms'][arm]['catalogOrderAgreement']={'profiles':448,'top1Agreements':sum(by[(p,'canonical')]['rankedProductIds'][0]==by[(p,'reversed')]['rankedProductIds'][0] for p in expected)}
    public={r['profileId']:r for r in packet['rows'] if r['order']=='canonical'};pairs=[]
    for pid,row in public.items():
        suffix='-purchase-contrast-v2'
        if pid.endswith(suffix):pairs.append((pid[:-len(suffix)],pid,'old320-purchase-contrast'))
    groups={}
    for pid,row in public.items():
        if row['source']=='new128':groups.setdefault(row['userGroupId'],[]).append(pid)
    pairs.extend((*sorted(ids),'new128-authored-pair') for ids in groups.values() if len(ids)==2)
    pred={(r['profileId'],r['order'],r['arm']):r for r in predictions}
    for left,right,kind in pairs:
        a=public[left]['publicRequest'];b=public[right]['publicRequest']
        owned=lambda q:{e['productId'] for e in q['publicHistory'] if e['eventType']=='purchase' and e.get('productId')}
        change={'pairKind':kind,'profileA':left,'profileB':right,'source':public[left]['source'],
                'nonHistoryContextEqual':{k:v for k,v in a.items() if k!='publicHistory'}=={k:v for k,v in b.items() if k!='publicHistory'},
                'historyChanged':a['publicHistory']!=b['publicHistory'],'newlyPurchasedA':sorted(owned(a)-owned(b)),'newlyPurchasedB':sorted(owned(b)-owned(a)),'arms':{}}
        for arm in RECIPE['scoreArms']:
            change['arms'][arm]={}
            for order in ('canonical','reversed'):
                aa=pred[(left,order,arm)]['rankedProductIds'];bb=pred[(right,order,arm)]['rankedProductIds'];new=change['newlyPurchasedB']
                change['arms'][arm][order]={'topA':aa[0],'topB':bb[0],'topChoiceChanged':aa[0]!=bb[0],
                    'newlyPurchasedBRankChanges':{p:{'before':aa.index(p)+1,'after':bb.index(p)+1} for p in new},
                    'gradeAtTopA':labels[left]['relevanceGrades'][aa[0]],'gradeAtTopB':labels[right]['relevanceGrades'][bb[0]]}
        output['purchasePairs'].append(change)
    return output

def run(plan_file,checksum,out,*,execute=False,owner=None):
    if sha(plan_file)!=checksum:raise ValueError('Exact frozen diagnostic plan SHA required')
    packet=read(plan_file)
    if packet['version']!=VERSION or packet['recipe']!=RECIPE or packet['sourcePlanSha256']!=PLAN_SHA or packet['adapterSha256']!=ADAPTER_SHA:raise ValueError('Frozen TRAIN diagnostic recipe required')
    pincheck(packet['sha256'])
    if not execute:return {'status':'cpu-preflight-only','nativeCalls':0,'optimizerUpdates':0,'separateTRAINLabelFilesParsed':False}
    if owner!='root':raise ValueError('Root is sole native owner')
    out=training.pilot.check_output(out);out.mkdir(parents=True);dump(out/'plan.json',packet)
    report={'version':VERSION,'status':'initializing','actualUpdates':0,'nativeBodyForwards':0,'attemptedNativeBodyForwards':0,
        'completedLMProjectionCalls':0,'separateTRAINLabelFilesParsed':False,'privateEvaluationLabelsParsed':False,
        'frozen927HashesUnchanged':None,'shutdownCompleted':False,'rows':[{'caseId':r['caseId'],'status':'unattempted'} for r in packet['rows']]}
    started=time.monotonic();save=lambda:dump(out/'report.json',report);save();predictions=[]
    try:
        with owned_model(packet,report,save) as (bb,adapter):
            import numpy as np
            mx=bb.mx
            initial={key:getattr(adapter,key) for key in ('lora_a','lora_b')};mx.eval(initial);mx.synchronize()
            actual={key:hashlib.sha256(np.array(value).tobytes()).hexdigest() for key,value in initial.items()}
            if actual!=packet['initialAdapterSha256'] or np.any(np.array(initial['lora_b'])!=0):raise ValueError('Pristine seeded A/B0 differs from original fit initialization')
            checkpoint=Path(packet['run'])/'fit/adapter.safetensors';final=mx.load(str(checkpoint));again=mx.load(str(checkpoint));mx.eval(final,again);mx.synchronize()
            if set(final)!={'lora_a','lora_b'} or any(not np.array_equal(np.array(final[k]),np.array(again[k])) or not np.isfinite(np.array(final[k])).all() for k in final):raise ValueError('Actual checkpoint reload is not exact/finite')
            report['checkpointReloadExact']=True;report['initialSeededAdapterExact']=True;report['status']='scoring-TRAIN-only';save()
            for index,(row,receipt) in enumerate(zip(packet['rows'],report['rows'],strict=True)):
                receipt['status']='running';report['activeCaseId']=row['caseId']
                item=load_public_feature(packet['run'],row)
                for arm,weights in zip(RECIPE['scoreArms'],(initial,final),strict=True):
                    for key,value in weights.items():setattr(adapter,key,value)
                    values=scoring.score(bb,adapter,scoring.public_item(item));mx.eval(values);mx.synchronize();bb.check_memory()
                    predictions.append(canonical_prediction(row,arm,values.tolist(),packet['canonicalProductIds']));report['completedLMProjectionCalls']+=1
                receipt['status']='completed';del item,values
                if index%25==0:save()
            if len(predictions)!=1792 or report['completedLMProjectionCalls']!=1792 or report['attemptedNativeBodyForwards']!=0:raise ValueError('Exact1792 zero-body projections required')
            dump(out/'all-TRAIN-predictions.json',predictions)
            dump(out/'ALL_TRAIN_PREDICTIONS_FROZEN.json',{'version':VERSION,'predictionSha256':sha(out/'all-TRAIN-predictions.json'),'predictions':1792,
                'adapterSha256':ADAPTER_SHA,'sourcePlanSha256':PLAN_SHA,'separateTRAINLabelFilesParsed':False,'privateEvaluationLabelsParsed':False})
            report['status']='completed-predictions-before-TRAIN-label-join';save()
        if report['frozen927HashesUnchanged'] is not True or report['shutdownCompleted'] is not True:raise RuntimeError('Native integrity/shutdown failed')
        # The only label-file parser is after the durable1792 prediction freeze and native teardown.
        labels={}
        for source,record in packet['labelFiles'].items():
            if sha(record['file'])!=record['sha256']:raise ValueError('Frozen TRAIN labels changed')
            for row in read(record['file'])['labels']:
                if row['profileId'] in labels:raise ValueError('Duplicate TRAIN label ID')
                labels[row['profileId']]=row
        report['separateTRAINLabelFilesParsed']=True;metrics=summarize(predictions,packet,labels);dump(out/'TRAIN_DIAGNOSIS.json',metrics)
        pincheck(packet['sha256']);report['status']='completed-TRAIN-diagnosis'
        return report
    except BaseException as error:
        for row in report['rows']:
            if row['status']=='running':row.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
    finally:report['elapsedSeconds']=time.monotonic()-started;save()

def main(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--native-owner',choices=['root']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-run',type=Path);p.add_argument('--source-plan',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--plan-sha256');a=p.parse_args(argv)
    if a.prepare:
        if a.run or not a.source_run or not a.source_plan:p.error('Separate CPU prepare requires exact original run/plan')
        packet=prepare(a.source_run,a.source_plan,a.out);print(json.dumps({'status':'prepared','rows':len(packet['rows']),'planSha256':sha(a.out/'plan.json'),'nativeCalls':0}));return 0
    if not a.plan or not a.plan_sha256:p.error('Exact diagnostic plan+SHA required')
    result=run(a.plan,a.plan_sha256,a.out,execute=a.run,owner=a.native_owner);print(json.dumps({k:v for k,v in result.items() if k not in ('rows','frozen927Before','frozen927After')}));return 0
if __name__=='__main__':raise SystemExit(main())
