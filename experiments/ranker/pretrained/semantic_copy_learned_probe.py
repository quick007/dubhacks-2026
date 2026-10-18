"""One post-fit learned-parameter feature parity probe; never performs updates.

Parent-owned native invocation AFTER the fixed feature-fit worker has exited.
Imports are CPU-only. New output directories are mandatory; prior evidence is
read-only, including the initial nonzero-B and failed inference-mode probes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .semantic_copy_feature_fit import (ROOT,MODEL,dataset_packet,digest,dump,
    load_feature,serving_rows,verify_model_files)

DEFAULT_FIT=Path('data/ranker/pretrained/semantic-copy-feature-fit-v1')
DEFAULT_PREFLIGHT=ROOT/'SEMANTIC_COPY_FEATURE_PREFLIGHT_V2.json'


def completed_fit(fit,preflight):
    """No native imports: reject active, partial, changed or unverified fits."""
    report_path=fit/'results.json';report=json.loads(report_path.read_text())
    if (report['status']!='completed' or report['phase']!='completed' or
            report['nativeUpdates']!=504 or report['optimizerUpdateCalls']!=504 or
            len(report['steps'])!=504 or len(report['extraction'])!=504 or
            any(s['status']!='completed' or s['updateEvaluated'] is not True for s in report['steps'])):
        raise ValueError('Require completed 504-gate/504-update fit; active or incomplete workers are ineligible')
    if not all(report['controls'].get(k) is True for k in
            ('all504NonzeroBGatesPassed','fitStartedFromZeroB','frozenLanguageHashesUnchanged')):
        raise ValueError('Required extraction/reset/frozen-language receipts are missing')
    if report.get('languageParameterTensorsHashed',0)<=700:
        raise ValueError('Incomplete frozen language hash receipt')
    checkpoint=fit/'adapter.safetensors';receipt=report['checkpoint']
    if not receipt['exact'] or receipt['lossBeforeReload']!=receipt['lossAfterReload'] or digest(checkpoint)!=receipt['sha256']:
        raise ValueError('Require hash-bound learned checkpoint and exact same-mode reload')
    if digest(preflight)!=report['preflightSha256'] or json.loads(preflight.read_text())!=report['preflight']:
        raise ValueError('Original tokenizer preflight changed')
    for name,expected in report['preflight']['sources'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen fit source changed: '+name)
    _,packet=dataset_packet()
    first=report['extraction'][0];case=packet['train']['cases'][0]
    common,_,fields=serving_rows(case,[f for f in packet['train']['fields'] if f['caseId']==case['caseId']])
    field=fields[0]
    if first['index']!=0 or (first['caseId'],first['rowId'])!=(case['caseId'],field['rowId']):
        raise ValueError('First frozen case/field identity changed')
    archived=load_feature(first,fit)
    return report,first,archived,case,common,field


def run(out,fit=DEFAULT_FIT,preflight=DEFAULT_PREFLIGHT):
    import psutil
    train,entry,archived,case,common,field=completed_fit(fit,preflight)
    if out.exists():raise ValueError('Preserve previous probe; choose a new output directory')
    if psutil.virtual_memory().available<7*2**30:raise MemoryError('Less than declared 7 GiB available')
    out.mkdir(parents=True);started=time.perf_counter();source_hash=digest(Path(__file__))
    report={'status':'started','nativeUpdates':0,'fieldCount':1,'memoryGiB':7,
        'executionCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'sourceSha256':source_hash,'fitResultSha256':digest(fit/'results.json'),
        'fitExecutionCommit':train['executionCommit'],'learnedCheckpointSha256':train['checkpoint']['sha256'],
        'originalFeatureArchiveSha256':entry['archiveSha256'],'originalFeatureMetadataSha256':entry['metadataSha256'],
        'originalPerRowGateSha256':entry['gateSha256'],'caseId':case['caseId'],'rowId':field['rowId'],
        'scope':'One learned-parameter checkpoint/control row; no optimizer updates or quality claim',
        'controls':{}}
    def save():
        report['wallMs']=(time.perf_counter()-started)*1000;dump(out/'results.json',report)
    save()
    try:
        report['modelVerification']=verify_model_files(train['preflight']['modelFiles']);save()
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx.utils import tree_flatten
        from mlx_lm.tuner.lora import LoRALinear
        from .backbone import FrozenBackbone
        from .evidence_plan import REVISION
        from .fusion import encode_shared_prefix
        from .shared_evidence_experiment import append_evidence
        from .semantic_copy_training import disabled,parameter_hashes
        from .semantic_copy_gradient_features import (probe_gradient_features,GradientFeatureProbeError,
            build_native_contract,feature_loss)
        from .native_batch_diagnosis import state_hash
        class PeakBackbone(FrozenBackbone):
            def check_memory(self):
                super().check_memory()
                if self.mx.get_peak_memory()>self.memory_budget:raise MemoryError('Observed peak exceeded declared 7 GiB')
        bb=PeakBackbone(MODEL,REVISION,1792,7);model=bb.model;model.freeze();model.eval()
        if dict(tree_flatten(model.trainable_parameters())):raise AssertionError('Base must be fully frozen')
        base_hashes=parameter_hashes(model.language_model);bb.check_memory()
        adapter=LoRALinear.from_base(model.layers[-1].mlp.down_proj,r=4,scale=20,dropout=0)
        model.layers[-1].mlp.down_proj=adapter;model.eval()
        weights=mx.load(str(fit/'adapter.safetensors'))
        if set(weights)!={'lora_a','lora_b'}:raise ValueError('Only A/B checkpoint leaves may be loaded')
        adapter.load_weights(list(weights.items()),strict=False);mx.eval(adapter.parameters());mx.synchronize();bb.check_memory()
        roster=dict(tree_flatten(model.trainable_parameters()))
        if len(roster)!=2 or sum(v.size for v in roster.values())!=65536 or any(
                '.31.mlp.down_proj.' not in k or k.rsplit('.',1)[-1] not in ('lora_a','lora_b') for k in roster):
            raise AssertionError('Only final layer31 down-projection 65536 A/B trainables are allowed')
        with disabled(model,adapter):
            embeddings,_,rank_timing,rank=encode_shared_prefix(bb,case['request'],case['catalog'],1792);del embeddings
            state,preparation=append_evidence(bb,rank,common)
        original_state_hash=state_hash(state['cache'])
        if original_state_hash!=entry['sourceStateHash']:raise AssertionError('Fresh base common state differs from original extraction')
        contract=build_native_contract(bb.tokenizer,state,field)
        if {k:contract[k] for k in archived['contract']}!=archived['contract']:
            raise AssertionError('Fresh learned probe teacher/token/prompt contract differs from archived row')
        planned=train['preflight']['splits']['train']['rows'][0]
        if contract['modelInputSha256']!=planned['modelInputSha256'] or contract['promptSha256']!=planned['promptSha256']:
            raise AssertionError('Fresh learned probe differs from original tokenizer preflight')
        report['controls']['sourceAndTokensMatchOriginal']=True
        report['sourceStateHash']=original_state_hash;report['rankingContextTiming']=rank_timing;report['preparation']=preparation;save()
        try:gate,recaptured=probe_gradient_features(bb,adapter,state,field)
        except GradientFeatureProbeError as exc:
            report['learnedNativeGate']=exc.report;save();raise
        report['learnedNativeGate']=gate;save()
        if gate['status']!='exact-gate-passed' or recaptured is None:
            report['status']='strict-gate-failed';save();return report
        exact_features=gate['featureTensors']==entry['featureTensors']
        report['controls']['recapturedFeatureReceiptsExact']=exact_features
        report['recapturedFeatureTensors']=gate['featureTensors'];report['originalFeatureTensors']=entry['featureTensors']
        # Re-evaluate the ORIGINAL archived item at the LEARNED parameters;
        # compare both scalar and gradients to the freshly recaptured path.
        original_params={k:getattr(adapter,k) for k in ('lora_a','lora_b')}
        try:
            loss,grads=nn.value_and_grad(adapter,lambda:feature_loss(bb,adapter,archived))()
            mx.eval(loss,grads);mx.synchronize();bb.check_memory()
            flat=dict(tree_flatten(grads))
            if set(flat)!={'lora_a','lora_b'}:raise AssertionError('Wrong archived feature gradient roster')
            def host(value):
                array=np.array(value.astype(mx.float32))
                if not np.isfinite(array).all():raise FloatingPointError('Nonfinite archived learned loss/gradient')
                dtype=str(value.dtype)
                return array,{'dtype':dtype,'shape':list(value.shape),
                    'sha256':hashlib.sha256(dtype.encode()+str(tuple(value.shape)).encode()+array.tobytes()).hexdigest()}
            archived_result={'loss':host(loss),'gradients':{k:host(v) for k,v in flat.items()}}
        finally:
            for k,v in original_params.items():setattr(adapter,k,v)
        report['originalArchivedAtLearned']={'loss':float(archived_result['loss'][0].item()),
            'lossTensor':archived_result['loss'][1],'gradientTensors':{k:v[1] for k,v in archived_result['gradients'].items()}}
        # The existing native gate preserves raw loss and tensor-byte receipts;
        # equal receipts plus equal scalar are exact, without a tolerance.
        report['controls']['originalArchivedLearnedLossAndGradientsExact']=(report['originalArchivedAtLearned']==gate['arms']['direct'])
        if state_hash(state['cache'])!=original_state_hash:raise AssertionError('Probe changed original complete source cache')
        after={k.replace('.31.mlp.down_proj.linear.','.31.mlp.down_proj.'):v for k,v in parameter_hashes(model.language_model).items()}
        if after!=base_hashes:raise AssertionError('Frozen language weights changed during learned probe')
        report['controls']['frozenLanguageHashesUnchanged']=True;report['languageParameterTensorsHashed']=len(base_hashes)
        report['controls']['sourceStatePreserved']=True
        report['status']='exact-gate-passed' if exact_features and report['controls']['originalArchivedLearnedLossAndGradientsExact'] else 'strict-gate-failed'
        report['memory']=bb.runtime_memory();bb.check_memory();save();return report
    except Exception as exc:
        report['status']='blocked';report['blocker']={'type':type(exc).__name__,'message':str(exc)}
        if 'bb' in locals():report['memory']=bb.runtime_memory()
        save();raise
    finally:
        report['sourceImmutable']=digest(Path(__file__))==source_hash
        report['fitSourcesUnchanged']=all(digest(ROOT/name)==value for name,value in train['preflight']['sources'].items())
        report['fitResultUnchanged']=digest(fit/'results.json')==report['fitResultSha256']
        report['checkpointUnchanged']=digest(fit/'adapter.safetensors')==report['learnedCheckpointSha256']
        if not all(report[k] for k in ('sourceImmutable','fitSourcesUnchanged','fitResultUnchanged','checkpointUnchanged')):
            report['status']='blocked';report['blocker']={'type':'AssertionError','message':'Probe source, completed fit or checkpoint changed'}
            save();raise AssertionError(report['blocker']['message'])
        save()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fit',type=Path,default=DEFAULT_FIT);p.add_argument('--preflight',type=Path,default=DEFAULT_PREFLIGHT)
    args=p.parse_args();run(args.out,args.fit,args.preflight)


if __name__=='__main__':main()
