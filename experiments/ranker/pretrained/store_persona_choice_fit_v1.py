"""Fixed4-pass MLX feature-only adapter fit API; no loader, CLI or native launch.

Root calls this inside its already-owned native lifecycle AFTER every extraction
gate and a sealed future-public evaluation packet. One NPZ host row is resident.
No evaluation annotation path is accepted. All updates use896 TRAIN rows only.
"""
from pathlib import Path
import json
import hashlib
import time
from . import store_persona_choice_features_v2 as features
from . import store_persona_choice_gate_pilot_v2 as pilot

VERSION='persona-choice-lora-feature-fit.v1'

def checked_roster(rows):
    if len(rows)!=896 or len({r['caseId'] for r in rows})!=896:raise ValueError('Exact896 unique TRAIN feature rows required')
    grouped={}
    for row in rows:
        if row.get('split')!='train' or row.get('targetPurpose')!='maximum-grade-TRAIN-supervision':raise ValueError('Only actual TRAIN target rows may enter optimizer')
        grouped.setdefault(row['profileId'],[]).append(row['order'])
    if len(grouped)!=448 or any(sorted(v)!=['canonical','reversed'] for v in grouped.values()):raise ValueError('448 TRAIN profiles in both fixed catalog orders required')
    return rows

def load_feature(root,row):
    import numpy as np
    for key in ('archive','metadata','gate'):
        path=(Path(root)/row[key]).resolve()
        if not path.is_relative_to(Path(root).resolve()) or pilot.lexical.sha(path)!=row[key+'Sha256']:raise ValueError('Exact local extracted file checksum required')
    metadata=json.loads((Path(root)/row['metadata']).read_text())
    gate=json.loads((Path(root)/row['gate']).read_text())
    if (gate['status']!='exact-gate-passed' or gate.get('sourceSha256')!=pilot.lexical.sha(features.__file__)
        or gate.get('sourceStatePreserved') is not True or gate.get('adapterPreserved') is not True
        or not gate.get('comparisons') or not all(c.get('exact') is True for c in gate['comparisons'].values())
        or metadata.get('singleChoiceGradientGatePassed') is not True
        or metadata['caseId']!=row['caseId'] or metadata['routeTokenIds']!=row['routeTokenIds']
        or metadata['targetWeights']!=row['targetWeights']):raise ValueError('Exact TRAIN choice gate/targets metadata required')
    with np.load(Path(root)/row['archive']) as arrays:item={**metadata,**{k:arrays[k] for k in features.FEATURE_KEYS}}
    features.validate_features(item)
    if any(not np.isfinite(item[k]).all() for k in features.FEATURE_KEYS):raise ValueError('Nonfinite serialized native features')
    for k in features.FEATURE_KEYS:
        host=item[k];dtype=item['dtypes'][k]
        expected=gate['featureTensors'][k]
        actual=hashlib.sha256(dtype.encode()+str(tuple(host.shape)).encode()+host.tobytes()).hexdigest()
        if host.dtype!=np.float32 or expected['sha256']!=actual:raise ValueError('Serialized feature differs from actual exact native gate tensor')
    return item

def fit(backbone,adapter,packet,archive_root,extraction,out):
    """Caller persists927/cache/shutdown proofs around this routine; no nested lock."""
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    import numpy as np
    out=pilot.check_output(out);out.mkdir(parents=True)
    mx=backbone.mx;recipe=pilot.RECIPE
    report={'version':VERSION,'status':'preflight','actualUpdates':0,'recipe':recipe,'developmentGoldParsed':False,'nativeBodyForwardsDuringFit':0,'losses':[]}
    save=lambda:pilot.lexical.dump(out/'report.json',report)
    save();started=time.monotonic()
    try:
        if (extraction['status']!='completed-choice-feature-extraction' or extraction['allGatesExact'] is not True
            or extraction['frozen927HashesUnchanged'] is not True):raise ValueError('Completed exact native extraction required')
        if packet['recipe']!=recipe or packet.get('futurePublicEvaluationFrozenBeforeFit') is not True:raise ValueError('Frozen recipe and future public evaluation packet required')
        pilot.lexical.check_pins(packet['sha256']);rows=checked_roster([r for r in extraction['rows'] if r['split']=='train'])
        expected={r['caseId']:r for r in packet['rows'] if r['split']=='train'}
        if set(expected)!={r['caseId'] for r in rows}:raise ValueError('Extracted TRAIN roster differs from frozen packet')
        for row in rows:
            reference=expected[row['caseId']]
            if any(row[k]!=reference[k] for k in ('targetWeights','routeTokenIds','profileId','order','targetPurpose')):raise ValueError('Extracted TRAIN row differs from predeclared exact target binding')
        roster=dict(tree_flatten(adapter.trainable_parameters()))
        if set(roster)!=features.GRADIENT_KEYS or tuple(adapter.lora_a.shape)!=(12288,4) or tuple(adapter.lora_b.shape)!=(4,4096):raise ValueError('Exact A/B adapter required')
        initial={k:np.array(v) for k,v in roster.items()}
        if np.any(initial['lora_b']!=0):raise ValueError('Fit must start from pristine exact zero B')
        order=features.fixed_orders(896,4,recipe['seed'])
        if len(order)!=3584:raise ValueError('Exact3584 optimizer schedule required')
        pilot.lexical.dump(out/'FIT_FROZEN.json',{'version':VERSION,'recipe':recipe,'order':order,
            'TRAINCases':[r['caseId'] for r in rows],'sourceAndDataSha256':packet['sha256'],
            'initialAdapterSha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in initial.items()},
            'developmentGoldParsed':False,'optimizerUpdates':0})
        optimizer=optim.Adam(learning_rate=recipe['learningRate'])
        loss_and_grad=nn.value_and_grad(adapter,lambda item:features.feature_loss(backbone,adapter,item))
        report['status']='training';save()
        for step,index in enumerate(order):
            report['activeCaseId']=rows[index]['caseId'];item=load_feature(archive_root,rows[index])
            loss,gradients=loss_and_grad(item);mx.eval(loss,gradients);mx.synchronize();backbone.check_memory()
            flat=dict(tree_flatten(gradients))
            if set(flat)!=features.GRADIENT_KEYS or not bool(mx.isfinite(loss).item()) or any(not bool(mx.all(mx.isfinite(v)).item()) for v in flat.values()):raise FloatingPointError('Nonfinite loss or wrong A/B gradient roster')
            optimizer.update(adapter,gradients);mx.eval(adapter.parameters(),optimizer.state);mx.synchronize();backbone.check_memory()
            report['actualUpdates']=step+1
            if step%25==0 or step==3583:
                report['losses'].append({'step':step+1,'caseId':rows[index]['caseId'],'preUpdateLoss':float(loss.item())});save()
            del item,loss,gradients,flat
        if report['actualUpdates']!=3584:raise ValueError('Exact3584 completed updates required')
        final={k:np.array(getattr(adapter,k)) for k in features.GRADIENT_KEYS}
        report['changedAdapterTensors']=[k for k in final if not np.array_equal(final[k],initial[k])]
        if set(report['changedAdapterTensors'])!=features.GRADIENT_KEYS:raise ValueError('Both A and B must actually learn')
        checkpoint=out/'adapter.safetensors';mx.save_safetensors(str(checkpoint),{k:getattr(adapter,k) for k in features.GRADIENT_KEYS})
        weights=mx.load(str(checkpoint))
        if set(weights)!=features.GRADIENT_KEYS:raise ValueError('Wrong serialized adapter roster')
        for k,v in weights.items():setattr(adapter,k,v)
        mx.eval(adapter.parameters());mx.synchronize()
        if any(not np.array_equal(np.array(getattr(adapter,k)),final[k]) for k in final):raise ValueError('Byte-exact adapter reload failed')
        # Report aggregate final CE instead of a misleading last-row loss.
        # Exact parameter reload above is not mislabeled native score parity.
        losses=[]
        for row in rows:
            item=load_feature(archive_root,row);loss,scores=features.feature_value(backbone,adapter,item)
            mx.eval(loss,scores);mx.synchronize();backbone.check_memory();losses.append(float(loss.item()))
            del item,loss,scores
        report.update(status='completed-feature-fit',checkpointSha256=pilot.lexical.sha(checkpoint),reloadExact=True,
                      finalMeanTrainRouteCE=sum(losses)/len(losses),trainRows896=True,
                      note='Prediction/native-after-fit parity and all nonfit freeze remain caller-owned, not claimed here')
        pilot.lexical.check_pins(packet['sha256']);report['allFitPinsUnchanged']=True
        return report
    except BaseException as error:
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
    finally:report['elapsedSeconds']=time.monotonic()-started;save()
