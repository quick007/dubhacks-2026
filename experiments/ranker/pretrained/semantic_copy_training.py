"""One bounded final-MLP adapter fit; full public evidence, target-only loss.

The final down projection cannot affect earlier activations or any native cache.
Frozen teacher-forced features preserve the installed BF16 addition/cast order.
No teacher text is supplied to inference. All native work stays in one process.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('experiments/ranker/pretrained')
DATA = ROOT / 'semantic_copy_data'
MODEL = Path('.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53')
CONFIG = {'seed': 19074, 'rank': 4, 'scale': 20., 'learningRate': .0001,
          'epochs': 2, 'batchSize': 1, 'memoryGiB': 7, 'prefillStep': 256,
          'maximumContext': 4096, 'maximumRankingPrefix': 1792,
          'target': 'language_model.model.layers.31.mlp.down_proj',
          'selection': 'final fixed step; no dev/test checkpoint selection',
          'trainingMode': 'frozen upstream eval; exact cached final-MLP inputs; target-only CE'}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def teacher_tokens(tokenizer, messages, value, cap):
    from .semantic_copy_tokens import build_training_tokens
    result = build_training_tokens(tokenizer, messages[:-1],
        {'messages': messages[-1:], 'targetValue': value, 'maxTokens': cap})
    return {**{k:v for k,v in result.items() if k not in
        ('promptTokenIds','fullTokenIds','teacherInputTokenIds','targetTokenIds','promptTokens')},
        'promptTokenCount': result['promptTokens'], 'inputs': result['teacherInputTokenIds'],
        'promptTokens': result['promptTokenIds'], 'targets': result['targetTokenIds']}


def layout(plan, request, catalog):
    from .semantic_copy_data.generator import public_messages
    from .heading_experiment import heading_rows
    common, evidence = public_messages(plan, request, catalog)
    rows = heading_rows(plan, request, catalog)
    for row in rows:
        field = row['fields'][0]; number = int(field[1:])
        bound = 'section' if row['kind'] == 'heading' else 'card'
        row['messages'] = [{'role': 'user', 'content':
            f'Write only the fresh {row["kind"]} value for {bound} {number} using its public evidence and {row["kind"]} rules. End after this one line.'}]
        row['maxTokens'] = {'heading': 16, 'title': 24, 'badge': 8}[row['kind']]
    return common, rows, evidence


@contextmanager
def disabled(model, adapter):
    layer = model.layers[-1]
    original = layer.mlp.down_proj
    layer.mlp.down_proj = adapter.linear
    try:
        yield
    finally:
        layer.mlp.down_proj = original


def capture(backbone, adapter, state, field):
    """Exact base teacher features after full shared evidence; no approximate trim."""
    import mlx.nn as nn
    import numpy as np
    from .parallel_decode import clone_prefix_state
    mx = backbone.mx; model = backbone.model
    contract = teacher_tokens(backbone.tokenizer, state['messages'] + field['messages'][-1:], field['targetValue'], field['maxTokens'])
    full = contract['inputs']; offset = contract['predictionStart']
    if full[:len(state['tokens'])] != state['tokens']:
        raise AssertionError('Teacher input does not extend exact shared cache')
    branch = clone_prefix_state(backbone, state)
    suffix = full[len(state['tokens']):]
    start = offset - len(state['tokens'])
    if start < 0: raise AssertionError('Target prediction starts before branch')
    saved = {}; layer = model.layers[-1]; norm = layer.post_attention_layernorm
    class NormTap(nn.Module):
        def __init__(self):
            super().__init__(); self.base = norm
        def __call__(self, x):
            saved['h'] = x
            return self.base(x)
    class DownTap(nn.Module):
        def __init__(self):
            super().__init__(); self.base = adapter.linear
        def __call__(self, x):
            saved['x'] = x; saved['y'] = self.base(x)
            return saved['y']
    xs=[]; hs=[]; ys=[]
    if len(suffix)>CONFIG['prefillStep']:raise ValueError('Exact feature training requires one complete native row suffix, no shape substitution')
    original = layer.mlp.down_proj
    layer.mlp.down_proj = DownTap(); layer.post_attention_layernorm = NormTap()
    try:
        for pos in range(0, len(suffix), CONFIG['prefillStep']):
            end = min(pos + CONFIG['prefillStep'], len(suffix))
            hidden = model.model(mx.array([suffix[pos:end]]), cache=branch['cache'])
            begin = max(start - pos, 0)
            if begin < end - pos:
                # Preserve the ENTIRE [1,L,D] native field-branch shape. Slicing
                # before the adapter/norm changes floating-point kernel shapes.
                selected = [saved[k] for k in ('x', 'h', 'y')]
                mx.eval(selected, [c.state for c in branch['cache']]); mx.synchronize()
                for dest, value in zip((xs,hs,ys), selected):
                    # float32 exactly represents BF16; restore original dtype below.
                    dest.append(np.array(value.astype(mx.float32)))
            else:
                mx.eval([c.state for c in branch['cache']]); mx.synchronize()
            del hidden
            backbone.check_memory()
    finally:
        layer.mlp.down_proj = original; layer.post_attention_layernorm = norm
    features = {k: np.concatenate(parts,axis=1) for k, parts in (('x',xs),('h',hs),('y',ys))}
    if features['x'].shape[1]-start != len(contract['targets']): raise AssertionError('Target feature alignment')
    return {**features, 'targets': contract['targets'], 'dtype': str(saved['x'].dtype),
            'contract': {k:v for k,v in contract.items() if k not in ('inputs','targets','promptTokens')},
            'predictionStartInBranch':start,
            'caseId': field['caseId'], 'rowId': field['rowId'], 'kind': field['kind']}


def feature_loss(backbone, adapter, item):
    import mlx.nn as nn
    mx = backbone.mx
    dtype = mx.bfloat16 if item['dtype'] == 'mlx.core.bfloat16' else mx.float16 if item['dtype'] == 'mlx.core.float16' else mx.float32
    x,h,y = (mx.array(item[k], dtype=dtype) for k in ('x','h','y'))
    delta = (adapter.scale * ((x @ adapter.lora_a) @ adapter.lora_b)).astype(x.dtype)
    # Match LoRALinear and DecoderLayer's two additions, including rounding.
    hidden = backbone.model.model.norm(h + (y + delta))
    logits = backbone.model.language_model.lm_head(hidden[0,item['predictionStartInBranch']:])
    return nn.losses.cross_entropy(logits.astype(mx.float32), mx.array(item['targets']), reduction='mean')


def parameter_hashes(model):
    from mlx.utils import tree_flatten
    import numpy as np
    import mlx.core as mx
    out = {}
    for name, value in tree_flatten(model.parameters()):
        if name.endswith(('lora_a','lora_b')): continue
        # Reads actual native parameter bytes, without expanding quantized weights.
        host = np.array(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value)
        out[name] = hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+host.tobytes()).hexdigest()
    return out


def cache_proof(backbone, adapter, item):
    """Nonzero adapter: same complete native state, changed logits, reused continuation."""
    import copy
    from mlx_lm.models.cache import make_prompt_cache
    from .native_batch_diagnosis import state_hash
    mx = backbone.mx; model = backbone.model
    tokens = item['prompt']
    states=[]; last=[]
    for enabled in (False,True):
        cache = make_prompt_cache(model)
        with disabled(model,adapter) if not enabled else null_context():
            for offset in range(0,len(tokens),CONFIG['prefillStep']):
                hidden = model.model(mx.array([tokens[offset:offset+CONFIG['prefillStep']]]), cache=cache)
                mx.eval([c.state for c in cache]); mx.synchronize()
            logits=model.language_model.lm_head(hidden[:,-1:])
            mx.eval(logits); mx.synchronize()
            if not bool(mx.all(mx.isfinite(logits)).item()):raise FloatingPointError('Nonfinite prefix logits')
        states.append(cache); last.append(logits)
    hashes=[state_hash(c) for c in states]
    effect=float(mx.max(mx.abs(last[0].astype(mx.float32)-last[1].astype(mx.float32))).item())
    if hashes[0] != hashes[1] or not effect > 0: raise AssertionError('Final MLP cache proof or nonzero effect failed')
    continuations=[]
    for cache in states:
        logs=[]; sampled=[]
        for token in item['forcedContinuation']:
            logit=model(mx.array([[token]]),cache=cache)[:,-1]
            mx.eval(logit,[c.state for c in cache]);mx.synchronize()
            if not bool(mx.all(mx.isfinite(logit)).item()):raise FloatingPointError('Nonfinite continuation logits')
            sampled.append(int(mx.argmax(logit,axis=-1).item())); logs.append(logit)
        continuations.append((sampled,logs,state_hash(cache)))
    exact=continuations[0][0]==continuations[1][0] and continuations[0][2]==continuations[1][2] and all(bool(mx.array_equal(a,b).item()) for a,b in zip(continuations[0][1],continuations[1][1]))
    if not exact: raise AssertionError('Base-cache / fresh-enabled continuation mismatch')
    return {'adapterNonzero':True,'prefixTokens':len(tokens),'completeStateHashes':hashes,
            'maxAbsLogitEffect':effect,'matchedForcedContinuationLogitsExact':exact,
            'forcedContinuationTokens':item['forcedContinuation'],'predictions':continuations[0][0]}


@contextmanager
def null_context():
    yield


def execute(args):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    import psutil
    import torch
    import subprocess
    from importlib.metadata import version
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear
    from .backbone import FrozenBackbone
    from .evidence_plan import REVISION, load_corrected
    from .fusion import build_shared_prefix, encode_shared_prefix
    from .shared_evidence_experiment import append_evidence
    if args.out.exists(): raise ValueError('Preserve existing experiment directory')
    if psutil.virtual_memory().available < CONFIG['memoryGiB']*2**30: raise MemoryError('Less than declared7GiB available')
    manifest=json.loads((DATA/'manifest.json').read_text())
    for name,expected in manifest['fileSha256'].items():
        if digest(DATA/name)!=expected:raise ValueError('Frozen payload hash mismatch: '+name)
    args.out.mkdir(parents=True)
    started=time.perf_counter(); report={'status':'started','config':CONFIG,'sourceSha256':digest(Path(__file__)),
        'dataManifestSha256':digest(DATA/'manifest.json'),'frozenBase':True,'trainedHeadsUnchanged':True,
        'executionCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'runtime':{name:version(name) for name in ('mlx','mlx-lm','numpy','torch','transformers')},
        'inputHashesVerified':manifest['fileSha256'],
        'modelRevision':REVISION,'tokenizerSha256':digest(MODEL/'tokenizer.json'),
        'controls':{},'extraction':[],'steps':[],'arms':[]}
    def save():
        report.update(wallMs=(time.perf_counter()-started)*1000)
        dump(args.out/'results.json',report)
    save(); torch.set_num_threads(1); mx.random.seed(CONFIG['seed'])
    class PeakBackbone(FrozenBackbone):
        def check_memory(self):
            super().check_memory()
            if self.mx.get_peak_memory()>self.memory_budget: raise MemoryError('Observed peak exceeded declared7GiB')
    try:
        bb=PeakBackbone(MODEL,REVISION,1536,CONFIG['memoryGiB']); model=bb.model
        report['loadMs']=bb.load_ms
        before=parameter_hashes(model.language_model)
        adapter=LoRALinear.from_base(model.layers[-1].mlp.down_proj,r=CONFIG['rank'],scale=CONFIG['scale'],dropout=0)
        model.layers[-1].mlp.down_proj=adapter; model.eval()
        names=[(n,int(v.size)) for n,v in tree_flatten(model.trainable_parameters())]
        if len(names)!=2 or sum(n for _,n in names)!=65536 or any(not n.endswith(('lora_a','lora_b')) for n,_ in names): raise AssertionError(('Wrong trainable roster',names))
        report['trainableParameters']=names; mx.eval(adapter.parameters()); save()
        train_cases=records(DATA/'cases.train.jsonl'); train_fields=records(DATA/'fields.train.jsonl')
        selected=train_fields[:1] if args.smoke else train_fields
        features=[]; proof_item=None; direct_item=None
        for case in train_cases:
            fields=[f for f in selected if f['caseId']==case['caseId']]
            if not fields:continue
            begin=time.perf_counter()
            with disabled(model,adapter):
                _,_,_,rank=encode_shared_prefix(bb,case['request'],case['catalog'],CONFIG['maximumRankingPrefix'])
                state,_=append_evidence(bb,rank,fields[0]['messages'][0])
            for field in fields:
                value=capture(bb,adapter,state,field); features.append(value)
                report['extraction'].append({'caseId':field['caseId'],'rowId':field['rowId'],'kind':field['kind'],
                    'targetTokens':len(value['targets']),**value['contract']})
                if proof_item is None:
                    contract=teacher_tokens(bb.tokenizer,state['messages']+field['messages'][-1:],field['targetValue'],field['maxTokens'])
                    proof_item={'prompt':contract['promptTokens'],'forcedContinuation':contract['targets'][:3]}
                    direct_item=(state,field,value)
            if not args.smoke and case is not train_cases[0]: del rank,state
            gc.collect();mx.clear_cache();bb.check_memory()
            print(json.dumps({'extractedCase':case['caseId'],'rows':len(fields),'ms':(time.perf_counter()-begin)*1000,'memory':bb.runtime_memory()}),flush=True);save()
        # Nonzero B catches rounding-order mistakes hidden by zero initialization.
        original_b=adapter.lora_b
        adapter.lora_b=mx.full(adapter.lora_b.shape,0.0001,dtype=adapter.lora_b.dtype);mx.eval(adapter.lora_b)
        from .parallel_decode import clone_prefix_state
        state,field,feature=direct_item
        contract=teacher_tokens(bb.tokenizer,state['messages']+field['messages'][-1:],field['targetValue'],field['maxTokens'])
        def direct():
            branch=clone_prefix_state(bb,state)
            hidden=model.model(mx.array([contract['inputs'][len(state['tokens']):]]),cache=branch['cache'])
            selected=hidden[0,contract['predictionStart']-len(state['tokens']):]
            logits=model.language_model.lm_head(selected)
            return nn.losses.cross_entropy(logits.astype(mx.float32),mx.array(contract['targets']),reduction='mean')
        ld,gd=nn.value_and_grad(adapter,direct)();lf,gf=nn.value_and_grad(adapter,lambda:feature_loss(bb,adapter,feature))()
        mx.eval(ld,gd,lf,gf);mx.synchronize();bb.check_memory()
        direct_grads=dict(tree_flatten(gd)); feature_grads=dict(tree_flatten(gf))
        if set(direct_grads)!=set(feature_grads) or len(direct_grads)!=2:raise AssertionError('Gradient roster mismatch')
        gradients=[{'name':a,'maxAbsDifference':float(mx.max(mx.abs(x-feature_grads[a])).item()),'exact':bool(mx.array_equal(x,feature_grads[a]).item())}
            for a,x in direct_grads.items()]
        report['controls']['directFeatureEquivalence']={'nonzeroB':True,'directLoss':float(ld.item()),'featureLoss':float(lf.item()),'gradients':gradients}
        if float(ld.item())!=float(lf.item()) or not all(g['exact'] for g in gradients):raise AssertionError('Direct/features loss or gradient mismatch')
        report['controls']['nonzeroCacheProof']=cache_proof(bb,adapter,proof_item)
        adapter.lora_b=original_b;mx.eval(adapter.parameters());del direct_item,state,gd,gf,ld,lf
        optimizer=optim.Adam(learning_rate=CONFIG['learningRate'])
        value_grad=nn.value_and_grad(adapter,lambda item:feature_loss(bb,adapter,item))
        report['lossBefore']=float(np.mean([float(feature_loss(bb,adapter,f).item()) for f in features]))
        initial={name:np.array(value) for name,value in tree_flatten(adapter.trainable_parameters())}
        epochs=1 if args.smoke else CONFIG['epochs'];rng=np.random.default_rng(CONFIG['seed'])
        for epoch in range(epochs):
            for index in rng.permutation(len(features)):
                begin=time.perf_counter();item=features[int(index)];loss,grads=value_grad(item)
                mx.eval(loss,grads)
                if not bool(mx.isfinite(loss).item()) or not all(bool(mx.all(mx.isfinite(g)).item()) for _,g in tree_flatten(grads)):raise FloatingPointError('Nonfinite loss or gradient')
                norm=sum(float(mx.sum(g.astype(mx.float32)**2).item()) for _,g in tree_flatten(grads))**.5
                optimizer.update(adapter,grads);mx.eval(adapter.parameters(),optimizer.state);mx.synchronize();bb.check_memory()
                if not all(bool(mx.all(mx.isfinite(p)).item()) for _,p in tree_flatten(adapter.trainable_parameters())):raise FloatingPointError('Nonfinite updated adapter')
                if not all(bool(mx.all(mx.isfinite(p)).item()) for _,p in tree_flatten(optimizer.state) if hasattr(p,'dtype')):raise FloatingPointError('Nonfinite optimizer state')
                report['steps'].append({'epoch':epoch,'rowId':item['rowId'],'caseId':item['caseId'],'loss':float(loss.item()),'gradientL2':norm,'ms':(time.perf_counter()-begin)*1000})
                if len(report['steps'])%28==0:print(json.dumps({'step':len(report['steps']),'loss':float(loss.item()),'memory':bb.runtime_memory()}),flush=True);save()
        report['lossAfter']=float(np.mean([float(feature_loss(bb,adapter,f).item()) for f in features]))
        report['parameterAbsoluteChanges']={name:float(np.abs(np.array(value)-initial[name]).sum()) for name,value in tree_flatten(adapter.trainable_parameters())}
        if not report['parameterAbsoluteChanges'].get('lora_b',0)>0:raise AssertionError('No actual LoRA-B update')
        report['controls']['learnedCacheProof']=cache_proof(bb,adapter,proof_item)
        checkpoint=args.out/'adapter.safetensors';mx.save_safetensors(str(checkpoint),dict(tree_flatten(adapter.trainable_parameters())))
        saved=mx.load(str(checkpoint)); originals={k:np.array(v) for k,v in tree_flatten(adapter.trainable_parameters())}
        adapter.load_weights(list(saved.items()));mx.eval(adapter.parameters())
        report['checkpoint']={'sha256':digest(checkpoint),'exactParameterReload':all(np.array_equal(originals[k],np.array(v)) for k,v in tree_flatten(adapter.trainable_parameters()))}
        # Adapter wrapping nests base path by one '.linear'; canonicalize only that insertion.
        after=parameter_hashes(model.language_model)
        after={k.replace('.31.mlp.down_proj.linear.','.31.mlp.down_proj.'):v for k,v in after.items()}
        if before!=after:raise AssertionError('Frozen base parameter hash changed')
        report['controls']['frozenBaseHashesUnchanged']=True
        report['frozenParameterCount']=len(before);save()
        if not args.smoke:
            from . import scheduler_identity_experiment as scheduler
            from . import fusion
            original_layout=scheduler.values_layout; original_encode=fusion.encode_shared_prefix
            heads,_,_=load_corrected(Path('data/ranker/pretrained/evidence-plan-v1'))
            def base_rank(*a,**kw):
                # Separate experimental prefix bound, declared before this run.
                with disabled(model,adapter):return original_encode(*a[:3],CONFIG['maximumRankingPrefix'])
            scheduler.values_layout=layout;fusion.encode_shared_prefix=base_rank
            try:
                for split in ('dev','test'):
                    for ci,case in enumerate(records(DATA/f'cases.{split}.jsonl')):
                        order=('learned','frozen') if ci%2 else ('frozen','learned')
                        for arm in order:
                            begin=time.perf_counter()
                            with disabled(model,adapter) if arm=='frozen' else null_context():
                                result=scheduler.value_arm(bb,heads,case)
                            result['externalFullCompletionMs']=(time.perf_counter()-begin)*1000
                            report['arms'].append({'split':split,'caseId':case['caseId'],'arm':arm,**result})
                            print(json.dumps({'split':split,'caseId':case['caseId'],'arm':arm,'quality':result['quality'],'fullMs':result['externalFullCompletionMs']}),flush=True);save()
                            gc.collect();mx.clear_cache()
            finally:
                scheduler.values_layout=original_layout;fusion.encode_shared_prefix=original_encode
        report['status']='smoke-passed' if args.smoke else 'completed'
        report['memory']=bb.runtime_memory();save();print(json.dumps({'status':report['status'],'lossBefore':report['lossBefore'],'lossAfter':report['lossAfter'],'steps':len(report['steps']),'memory':report['memory']}),flush=True)
    except Exception as exc:
        report['status']='blocked';report['blocker']={'type':type(exc).__name__,'message':str(exc)}
        if 'bb' in locals():report['memory']=bb.runtime_memory()
        save();raise


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    execute(parser.parse_args())


if __name__=='__main__':main()
