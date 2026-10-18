"""Direct native final-MLP QLoRA; the failed feature optimization is unused."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time
from .semantic_copy_training import (DATA,MODEL,ROOT,CONFIG as BASE_CONFIG,dump,digest,records,
    disabled,null_context,teacher_tokens,parameter_hashes,cache_proof,layout)

CONFIG={**BASE_CONFIG,'trainingMode':'direct native teacher-forced suffix; frozen upstream eval',
    'epochs':2,'updateOrder':'fixed case order; seeded row shuffle within each case, two passes per case',
    'featureOptimization':'retired after strict v2/v3/v4 mismatch; never used for updates'}


def run(args):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    import psutil
    import torch
    from importlib.metadata import version
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear
    from .backbone import FrozenBackbone
    from .fusion import encode_shared_prefix
    from .shared_evidence_experiment import append_evidence
    from .parallel_decode import clone_prefix_state
    from .native_batch_diagnosis import state_hash
    from .evidence_plan import REVISION,load_corrected
    if args.out.exists():raise ValueError('Preserve existing result directory')
    if psutil.virtual_memory().available<7*2**30:raise MemoryError('Less than declared7GiB available')
    data=args.dataset
    config={**CONFIG,'epochs':args.epochs,'dataset':str(data),
        'updateOrder':f'fixed case order; seeded row shuffle within each case, {args.epochs} passes per case'}
    manifest=json.loads((data/'manifest.json').read_text())
    for name,sha in manifest['fileSha256'].items():
        if digest(data/name)!=sha:raise ValueError('Frozen payload changed: '+name)
    args.out.mkdir(parents=True);start=time.perf_counter()
    report={'status':'started','config':config,'executionCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'sourceSha256':digest(Path(__file__)),'helperSourceSha256':digest(ROOT/'semantic_copy_training.py'),
        'inputHashesVerified':manifest['fileSha256'],'runtime':{n:version(n) for n in ('mlx','mlx-lm','torch','numpy')},
        'modelRevision':REVISION,'tokenizerSha256':digest(MODEL/'tokenizer.json'),
        'frozenLanguageWeights':True,'frozenDecisionHeads':True,'controls':{},'steps':[],'cases':[],'arms':[]}
    def save():
        report['wallMs']=(time.perf_counter()-start)*1000;dump(args.out/'results.json',report)
    save();torch.set_num_threads(1);mx.random.seed(CONFIG['seed'])
    class PeakBackbone(FrozenBackbone):
        def check_memory(self):
            super().check_memory()
            if self.mx.get_peak_memory()>self.memory_budget:raise MemoryError('Observed peak exceeded declared7GiB')
    try:
        bb=PeakBackbone(MODEL,REVISION,1792,7);model=bb.model;report['loadMs']=bb.load_ms
        base_hashes=parameter_hashes(model.language_model)
        adapter=LoRALinear.from_base(model.layers[-1].mlp.down_proj,r=4,scale=20,dropout=0)
        model.layers[-1].mlp.down_proj=adapter;model.eval();mx.eval(adapter.parameters())
        trainable=dict(tree_flatten(model.trainable_parameters()))
        if len(trainable)!=2 or sum(v.size for v in trainable.values())!=65536 or any(not k.endswith(('lora_a','lora_b')) for k in trainable):raise AssertionError('Wrong trainable roster')
        report['trainableParameters']={k:int(v.size) for k,v in trainable.items()};save()
        optimizer=optim.Adam(learning_rate=CONFIG['learningRate']);rng=np.random.default_rng(CONFIG['seed'])
        fields=records(data/'fields.train.jsonl');cases=records(data/'cases.train.jsonl')
        expected_fields={f'S{i}' for i in (1,2)}|{f'{kind}{i}' for kind in ('T','B') for i in range(1,7)}
        if not cases or len({c['caseId'] for c in cases})!=len(cases) or len(fields)!=14*len(cases):raise AssertionError('Wrong frozen training denominator')
        report['expectedTrainingCases']=len(cases);report['expectedTrainingSteps']=1 if args.smoke else len(fields)*args.epochs
        for case in cases:
            rows=[f for f in fields if f['caseId']==case['caseId']]
            if len(rows)!=14 or {f['field'] for f in rows}!=expected_fields:raise AssertionError('Wrong training field membership')
        if args.smoke:cases=cases[:1];fields=fields[:1]
        def make_state(case,common):
            with disabled(model,adapter):
                _,_,_,rank=encode_shared_prefix(bb,case['request'],case['catalog'],1792)
                state,preparation=append_evidence(bb,rank,common)
            return state,preparation
        def contract_for(state,field):
            contract=teacher_tokens(bb.tokenizer,state['messages']+field['messages'][-1:],field['targetValue'],field['maxTokens'])
            if contract['inputs'][:len(state['tokens'])]!=state['tokens']:raise AssertionError('Nonextending teacher prefix')
            return contract
        def direct_loss(state,contract):
            branch=clone_prefix_state(bb,state)
            suffix=contract['inputs'][len(state['tokens']):]
            if not 1<=len(suffix)<=256:raise ValueError('Direct native suffix exceeds explicit256cap; no truncation')
            hidden=model.model(mx.array([suffix]),cache=branch['cache'])
            logits=model.language_model.lm_head(hidden[0,contract['predictionStart']-len(state['tokens']):])
            return nn.losses.cross_entropy(logits.astype(mx.float32),mx.array(contract['targets']),reduction='mean')
        grad_fn=nn.value_and_grad(adapter,direct_loss);proof=None
        for ci,case in enumerate(cases):
            begin=time.perf_counter();rows=[f for f in fields if f['caseId']==case['caseId']]
            state,prep=make_state(case,rows[0]['messages'][0]);source_hash=state_hash(state['cache'])
            contracts=[contract_for(state,f) for f in rows]
            if ci==0:
                proof={'prompt':contracts[0]['promptTokens'],'forcedContinuation':contracts[0]['targets'][:3]}
                zero=adapter.lora_b;adapter.lora_b=mx.full(zero.shape,.0001,dtype=zero.dtype);mx.eval(adapter.lora_b)
                report['controls']['nonzeroCacheProof']=cache_proof(bb,adapter,proof)
                a,ga=grad_fn(state,contracts[0]);b,gb=grad_fn(state,contracts[0]);mx.eval(a,ga,b,gb);mx.synchronize()
                ga=dict(tree_flatten(ga));gb=dict(tree_flatten(gb))
                exact=set(ga)==set(gb)=={'lora_a','lora_b'} and float(a.item())==float(b.item()) and all(bool(mx.array_equal(ga[k],gb[k]).item()) for k in ga)
                if not exact:raise AssertionError('Direct native loss/gradient repeatability mismatch')
                report['controls']['directRepeatability']={'nonzeroB':True,'loss':float(a.item()),'gradientKeys':list(ga),'exact':True}
                adapter.lora_b=zero;mx.eval(adapter.parameters());del a,b,ga,gb
                report['firstTrainingProbeBefore']=float(direct_loss(state,contracts[0]).item())
            native_rows=[]
            for f,c in zip(rows,contracts):
                native_rows.append({'rowId':f['rowId'],'kind':f['kind'],'targetTokens':len(c['targets']),
                    **{k:v for k,v in c.items() if k not in ('inputs','targets','promptTokens')}})
            initial={k:np.array(v) for k,v in tree_flatten(adapter.trainable_parameters())}
            for epoch in range(1 if args.smoke else args.epochs):
                for idx in rng.permutation(len(rows)):
                    idx=int(idx);step_start=time.perf_counter();loss,grads=grad_fn(state,contracts[idx]);mx.eval(loss,grads)
                    flat=dict(tree_flatten(grads))
                    if set(flat)!={'lora_a','lora_b'}:raise AssertionError('Wrong gradient roster')
                    if not bool(mx.isfinite(loss).item()) or not all(bool(mx.all(mx.isfinite(v)).item()) for v in flat.values()):raise FloatingPointError('Nonfinite direct loss/gradient')
                    norm=sum(float(mx.sum(v.astype(mx.float32)**2).item()) for v in flat.values())**.5
                    optimizer.update(adapter,grads);mx.eval(adapter.parameters(),optimizer.state);mx.synchronize();bb.check_memory()
                    if not all(bool(mx.all(mx.isfinite(v)).item()) for _,v in tree_flatten(adapter.trainable_parameters())):raise FloatingPointError('Nonfinite adapter')
                    if not all(bool(mx.all(mx.isfinite(v)).item()) for _,v in tree_flatten(optimizer.state) if hasattr(v,'dtype')):raise FloatingPointError('Nonfinite optimizer state')
                    after_hash=state_hash(state['cache'])
                    if after_hash!=source_hash:raise AssertionError('Pristine shared training state mutated')
                    report['steps'].append({'caseId':case['caseId'],'rowId':rows[idx]['rowId'],'kind':rows[idx]['kind'],'epoch':epoch,
                        'loss':float(loss.item()),'gradientL2':norm,'sourceStateHash':after_hash,'ms':(time.perf_counter()-step_start)*1000})
            changes={k:float(np.abs(np.array(v)-initial[k]).sum()) for k,v in tree_flatten(adapter.trainable_parameters())}
            if not changes['lora_b']>0:raise AssertionError('No real LoRA-B update')
            report['cases'].append({'caseId':case['caseId'],'preparation':prep,'rows':native_rows,'parameterAbsoluteChanges':changes,
                'sourceStateHash':source_hash,'sourceStatePreserved':True,'wallMs':(time.perf_counter()-begin)*1000})
            print(json.dumps({'trainedCase':case['caseId'],'steps':len(report['steps']),'lastLoss':report['steps'][-1]['loss'],'memory':bb.runtime_memory()}),flush=True);save()
            del state,contracts,loss,grads;gc.collect();mx.clear_cache()
        report['controls']['learnedCacheProof']=cache_proof(bb,adapter,proof)
        state,_=make_state(cases[0],fields[0]['messages'][0]);first=contract_for(state,fields[0])
        report['firstTrainingProbeAfter']=float(direct_loss(state,first).item())
        checkpoint=args.out/'adapter.safetensors';mx.save_safetensors(str(checkpoint),dict(tree_flatten(adapter.trainable_parameters())))
        before_reload=float(direct_loss(state,first).item());saved=mx.load(str(checkpoint))
        if set(saved)!={'lora_a','lora_b'}:raise AssertionError('Wrong adapter checkpoint membership')
        adapter.load_weights(list(saved.items()),strict=False);mx.eval(adapter.parameters())
        after_reload=float(direct_loss(state,first).item())
        if before_reload!=after_reload:raise AssertionError('Reload prediction mismatch')
        report['checkpoint']={'sha256':digest(checkpoint),'exactDirectLossReload':True,'loss':after_reload}
        after_hashes=parameter_hashes(model.language_model)
        after_hashes={k.replace('.31.mlp.down_proj.linear.','.31.mlp.down_proj.'):v for k,v in after_hashes.items()}
        if base_hashes!=after_hashes:raise AssertionError('Frozen language weights changed')
        report['controls']['frozenLanguageHashesUnchanged']=True;report['languageParameterTensorsHashed']=len(base_hashes)
        del state;gc.collect();mx.clear_cache();save()
        if not args.smoke:
            from . import scheduler_identity_experiment as scheduler
            from . import fusion
            old_layout=scheduler.values_layout;old_encode=fusion.encode_shared_prefix
            heads,_,_=load_corrected(Path('data/ranker/pretrained/evidence-plan-v1'))
            report['headsCheckpointSha256']=digest(Path('data/ranker/pretrained/evidence-plan-v1/heads.pt'))
            def base_rank(backbone,request,catalog,*a,**kw):
                with disabled(model,adapter):return old_encode(backbone,request,catalog,1792)
            scheduler.values_layout=layout;fusion.encode_shared_prefix=base_rank
            try:
                for split in ('dev','test'):
                    for index,case in enumerate(records(data/f'cases.{split}.jsonl')):
                        for arm in (('learned','frozen') if index%2 else ('frozen','learned')):
                            begin=time.perf_counter()
                            with disabled(model,adapter) if arm=='frozen' else null_context():result=scheduler.value_arm(bb,heads,case)
                            result['pagePlan']['method']['headingsAndTitles']='same quantized backbone; final-MLP adapter '+('enabled' if arm=='learned' else 'disabled')+'; independently validated fields'
                            result['externalFullCompletionMs']=(time.perf_counter()-begin)*1000
                            report['arms'].append({'split':split,'caseId':case['caseId'],'arm':arm,**result})
                            print(json.dumps({'split':split,'caseId':case['caseId'],'arm':arm,'quality':result['quality'],'fullMs':result['externalFullCompletionMs']}),flush=True);save()
                            gc.collect();mx.clear_cache()
            finally:scheduler.values_layout=old_layout;fusion.encode_shared_prefix=old_encode
        if len(report['steps'])!=report['expectedTrainingSteps']:raise AssertionError('Incomplete fixed training step count')
        report['status']='smoke-passed' if args.smoke else 'completed';report['memory']=bb.runtime_memory();save()
        print(json.dumps({'status':report['status'],'steps':len(report['steps']),'firstProbeBefore':report['firstTrainingProbeBefore'],'firstProbeAfter':report['firstTrainingProbeAfter'],'memory':report['memory']}),flush=True)
    except Exception as exc:
        report['status']='blocked';report['blocker']={'type':type(exc).__name__,'message':str(exc)}
        if 'bb' in locals():report['memory']=bb.runtime_memory()
        save();raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--dataset',type=Path,default=DATA);p.add_argument('--epochs',type=int,choices=(1,2),default=2);run(p.parse_args())


if __name__=='__main__':main()
