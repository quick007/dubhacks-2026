"""Single-next-token final-down decision features; import is CPU-only.

No CLI/model loader/optimizer is supplied. Root owns the live backbone, adapter,
lock, deadlines,927 hashes, frozen TRAIN packet and lifecycle. Routing vocabulary
is caller-frozen. Loss uses the full native frozen LM projection then gathers
the sixteen product-bound routes; no unverified row-slicing optimization.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random

VERSION='persona-choice-single-token-features.v1'
FEATURE_KEYS=('x','h','y')
GRADIENT_KEYS={'lora_a','lora_b'}

def token_hash(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()

def choice_contract(row, *, cap=2048):
    ids=row.get('tokenIds');routes=row.get('routeTokenIds');weights=row.get('targetWeights')
    if (not isinstance(ids,list) or not 2<=len(ids)<=cap or
        any(type(t) is not int or not 0<=t<248320 for t in ids)):
        raise ValueError('Complete actual prompt token IDs required; no target/EOS append')
    if (not isinstance(routes,list) or len(routes)!=16 or len(set(routes))!=16 or
        any(type(t) is not int or not 0<=t<248320 for t in routes)):
        raise ValueError('Exactly16 unique frozen one-token routes required')
    if (not isinstance(weights,list) or len(weights)!=16 or
        any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in weights) or
        abs(sum(weights)-1)>1e-7):raise ValueError('Normalized TRAIN-only16-choice target distribution required')
    if row.get('tokenIdsSha256')!=token_hash(ids):raise ValueError('Prompt token checksum differs')
    return {'prefixTokenIds':ids[:-1],'suffixTokenIds':ids[-1:],'routeTokenIds':list(routes),
            'targetWeights':list(weights),'promptTokenIdsSha256':token_hash(ids),
            'prefixTokenIdsSha256':token_hash(ids[:-1]),'inputTargetTokens':0,'supervisedPositions':1,
            'loss':'soft-target route-conditional CE from full frozen native LM projection',
            'caseId':row.get('caseId')}

def max_grade_targets(grades, product_ids):
    if (len(product_ids)!=16 or set(grades)!=set(product_ids) or
        any(type(v) is not int or not 0<=v<=3 for v in grades.values())):raise ValueError('Exact16 TRAIN grades required')
    highest=max(grades.values())
    if highest<=0:raise ValueError('TRAIN choice needs supported positive utility')
    n=sum(v==highest for v in grades.values())
    return [float(grades[p]==highest)/n for p in product_ids]

def fixed_orders(count, epochs, seed):
    if type(count) is not int or count<=0 or type(epochs) is not int or epochs<=0:raise ValueError('Positive fixed training schedule required')
    result=[]
    for epoch in range(epochs):
        order=list(range(count));random.Random(seed+epoch).shuffle(order);result.extend(order)
    return result

def validate_features(item):
    shapes={k:tuple(item[k].shape) for k in FEATURE_KEYS}
    expected={'x':(1,1,12288),'h':(1,1,4096),'y':(1,1,4096)}
    if shapes!=expected:raise ValueError('Preserve complete actual B1 one-token suffix shape')
    if set(item['dtypes'])!=set(FEATURE_KEYS) or set(item['dtypes'].values())-{'mlx.core.bfloat16','mlx.core.float16','mlx.core.float32'}:
        raise ValueError('Exact native dtype roster required')
    choice_contract({'tokenIds':[1,2],'tokenIdsSha256':token_hash([1,2]),
                     'routeTokenIds':item['routeTokenIds'],'targetWeights':item['targetWeights']})
    return shapes

def native_projection(model, hidden):
    lm=model.language_model
    if bool(lm.args.tie_word_embeddings):raise ValueError('Pinned local9B has separate frozen LM head')
    return lm.lm_head(hidden)

def decision_loss(mx, logits, routes, weights):
    # Entire pretrained vocabulary is computed natively before gathering. The
    # conditional objective matches the actual constrained16-choice ranking.
    scores=logits.astype(mx.float32).reshape(-1)[mx.array(routes)]
    logprob=scores-mx.logsumexp(scores)
    return -mx.sum(mx.array(weights,dtype=mx.float32)*logprob), scores

def feature_value(backbone,adapter,item):
    validate_features(item);mx=backbone.mx
    dtypes={str(t):t for t in (mx.bfloat16,mx.float16,mx.float32)}
    x,h,y=(mx.array(item[k],dtype=dtypes[item['dtypes'][k]]) for k in FEATURE_KEYS)
    z=(adapter.dropout(x)@adapter.lora_a)@adapter.lora_b
    delta=(adapter.scale*z).astype(x.dtype)
    hidden=backbone.model.model.norm(h+(y+delta))
    logits=native_projection(backbone.model,hidden[:,0,:])
    return decision_loss(mx,logits,item['routeTokenIds'],item['targetWeights'])

def feature_loss(backbone,adapter,item):
    if item.get('singleChoiceGradientGatePassed') is not True:raise ValueError('New exact single-choice gate required')
    return feature_value(backbone,adapter,item)[0]

def prepare_state(backbone,adapter,row, *, prefill_step=256,adapter_enabled=False):
    """Only called within root-owned native lifecycle; no model is constructed."""
    from mlx_lm.models.cache import make_prompt_cache
    from contextlib import nullcontext
    from .semantic_copy_training import disabled
    contract=choice_contract(row);actual=backbone.tokenizer.encode(row['prompt'],add_special_tokens=False)
    if actual!=row['tokenIds']:raise ValueError('Actual source prompt differs from frozen tokenizer preparation')
    mx=backbone.mx;cache=make_prompt_cache(backbone.model)
    with nullcontext() if adapter_enabled else disabled(backbone.model,adapter):
        for offset in range(0,len(contract['prefixTokenIds']),prefill_step):
            hidden=backbone.model.model(mx.array([contract['prefixTokenIds'][offset:offset+prefill_step]]),cache=cache)
            mx.eval(hidden,[c.state for c in cache]);mx.synchronize();backbone.check_memory()
            del hidden
    return {'model':backbone.model,'cache':cache,'tokens':contract['prefixTokenIds'],
            'messages':[],'consumed':False},contract

class ChoiceFeatureGateError(RuntimeError):
    def __init__(self,report):self.report=report;super().__init__(report['error']['message'])

def probe(backbone,adapter,state,row, *, require_nonzero=True):
    """Exact native/auxiliary/serialized-feature loss+logit+A/B-gradient gate.

    Caller sets a nonzero-B numerical control and restores zero-B afterwards.
    This API does not update parameters, persist evidence, or relax tolerances.
    """
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten
    from .parallel_decode import clone_prefix_state
    from .native_batch_diagnosis import state_hash
    from .semantic_copy_gradient_features import _host_tensor,compare_host_results
    mx=backbone.mx;model=backbone.model;layer=model.layers[-1];contract=choice_contract(row)
    if state.get('model') is not model or state.get('consumed') or state['tokens']!=contract['prefixTokenIds']:
        raise ValueError('Exact unconsumed prompt-minus-final-token native prefix required')
    if len(model.layers)!=32 or layer.mlp.down_proj is not adapter or model.training or adapter.training or adapter.dropout.training or adapter.dropout._p_1!=1:
        raise ValueError('Pinned final-down adapter in eval mode with no dropout required')
    roster=dict(tree_flatten(model.trainable_parameters()))
    if len(roster)!=2 or any('.31.mlp.down_proj.' not in k or k.rsplit('.',1)[-1] not in GRADIENT_KEYS for k in roster):raise ValueError('Only final-down A/B may train')
    if tuple(adapter.lora_a.shape)!=(12288,4) or tuple(adapter.lora_b.shape)!=(4,4096) or adapter.scale!=20:raise ValueError('Rank4 scale20 required')
    parameters={k:getattr(adapter,k) for k in sorted(GRADIENT_KEYS)}
    initial={k:_host_tensor(mx,v)[1] for k,v in parameters.items()}
    nonzero=bool(np.any(_host_tensor(mx,adapter.lora_b)[0]!=0))
    if nonzero!=require_nonzero:raise ValueError('Exact requested zero/nonzero B numerical control required')
    before=state_hash(state['cache']);saved={};calls={k:0 for k in FEATURE_KEYS}
    original_norm=layer.post_attention_layernorm;original_linear=adapter.linear
    class NormTap(nn.Module):
        def __init__(self):super().__init__();self.base=original_norm
        def __call__(self,h):saved['h']=h;calls['h']+=1;return self.base(h)
    class DownTap(nn.Module):
        def __init__(self):super().__init__();self.base=original_linear
        def __call__(self,x):
            saved['x']=x;calls['x']+=1;saved['y']=self.base(x);calls['y']+=1;return saved['y']
    def native_value(capture=False):
        branch=clone_prefix_state(backbone,state)
        hidden=model.model(mx.array([contract['suffixTokenIds']]),cache=branch['cache'])
        loss,scores=decision_loss(mx,native_projection(model,hidden[:,0,:]),contract['routeTokenIds'],contract['targetWeights'])
        return loss,{'scores':scores,**({k:saved[k] for k in FEATURE_KEYS} if capture else {})}
    def evaluate(fn):
        try:
            (loss,aux),gradients=nn.value_and_grad(adapter,fn)();mx.eval(loss,aux,gradients);mx.synchronize();backbone.check_memory()
            flat=dict(tree_flatten(gradients))
            if set(flat)!=GRADIENT_KEYS:raise ValueError('Exactly A/B gradients required')
            result={'loss':_host_tensor(mx,loss),'gradients':{k:_host_tensor(mx,v) for k,v in flat.items()},'scores':_host_tensor(mx,aux['scores'])}
            return result,aux
        finally:
            for k,v in parameters.items():setattr(adapter,k,v)
    def receipt(result):
        return {'loss':float(result['loss'][0]),'lossTensor':result['loss'][1],
                'gradientTensors':{k:v[1] for k,v in result['gradients'].items()},'routeLogits':result['scores'][0].tolist(),'routeLogitsTensor':result['scores'][1]}
    report={'version':VERSION,'status':'running','caseId':row.get('caseId'),'sourceSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'contract':{k:v for k,v in contract.items() if k not in ('prefixTokenIds','suffixTokenIds')},
            'nativeUpdates':0,'nonzeroB':nonzero,'sourceStateHashBefore':before,'adapterBefore':initial,'arms':{},
            'projection':'full pinned native quantized LM head then16 gather; no dequantized row shortcut'}
    item=None;error=None
    try:
        direct,_=evaluate(native_value);report['arms']['direct']=receipt(direct)
        try:
            layer.post_attention_layernorm=NormTap();adapter.linear=DownTap()
            captured,aux=evaluate(lambda:native_value(True));report['arms']['auxiliaryNative']=receipt(captured)
        finally:layer.post_attention_layernorm=original_norm;adapter.linear=original_linear;saved.clear()
        features={k:_host_tensor(mx,aux[k]) for k in FEATURE_KEYS}
        item={k:v[0] for k,v in features.items()}
        item.update(dtypes={k:v[1]['dtype'] for k,v in features.items()},routeTokenIds=contract['routeTokenIds'],targetWeights=contract['targetWeights'],caseId=row.get('caseId'))
        validate_features(item)
        reconstructed,_=evaluate(lambda:(lambda pair:(pair[0],{'scores':pair[1]}))(feature_value(backbone,adapter,item)))
        report['arms']['frozenFeatures']=receipt(reconstructed)
        report['comparisons']={}
        for name,left,right in [('nativeVsAuxiliary',direct,captured),('auxiliaryVsFeatures',captured,reconstructed),('nativeVsFeatures',direct,reconstructed)]:
            comparison=compare_host_results(left,right)
            comparison['routeLogitsExact']=left['scores'][1]==right['scores'][1] and np.array_equal(left['scores'][0],right['scores'][0])
            comparison['exact']=comparison['exact'] and comparison['routeLogitsExact'];report['comparisons'][name]=comparison
        if calls!={k:1 for k in FEATURE_KEYS}:raise AssertionError('Exactly one complete suffix tap required')
        report['tapCalls']=calls;report['featureTensors']={k:v[1] for k,v in features.items()}
        report['status']='exact-gate-passed' if all(c['exact'] for c in report['comparisons'].values()) else 'strict-gate-failed'
    except BaseException as exc:error=exc
    finally:
        layer.post_attention_layernorm=original_norm;adapter.linear=original_linear
        for k,v in parameters.items():setattr(adapter,k,v)
        try:
            report['sourceStateHashAfter']=state_hash(state['cache'])
            report['sourceStatePreserved']=report['sourceStateHashAfter']==before and state['tokens']==contract['prefixTokenIds'] and not state.get('consumed')
            report['adapterAfter']={k:_host_tensor(mx,getattr(adapter,k))[1] for k in sorted(GRADIENT_KEYS)}
            report['adapterPreserved']=report['adapterAfter']==initial
            if not report['sourceStatePreserved'] or not report['adapterPreserved']:raise AssertionError('Native cache or adapter changed during gate')
            backbone.check_memory();report['memory']=backbone.runtime_memory()
        except BaseException as cleanup_error:
            if error is None:error=cleanup_error
            else:report['cleanupError']={'type':type(cleanup_error).__name__,'message':str(cleanup_error)}
    if error is not None:
        report.update(status='blocked',error={'type':type(error).__name__,'message':str(error)});raise ChoiceFeatureGateError(report) from error
    if report['status']=='exact-gate-passed':item['singleChoiceGradientGatePassed']=True;return report,item
    return report,None
