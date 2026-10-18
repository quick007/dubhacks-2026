"""Target-free gradient-mode auxiliary feature capture for final16 decisions.

Import is CPU-only. Caller owns the serial native model/adapter lifetime. Scalar
sum of16 public routing logits activates the same gradient transform; no target,
loss, labels, teacher tokens, or optimizer is accepted. Exact native parity is a
separate measured certificate, never inferred from this source.
"""
import hashlib
import json
from . import store_persona_choice_scoring_v1 as scoring

def token_hash(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()

def contract(state,prompt_ids,route_ids,model):
    if (not isinstance(prompt_ids,list) or not 16<len(prompt_ids)<=2048 or any(type(x) is not int or not 0<=x<248320 for x in prompt_ids)
        or not isinstance(route_ids,list) or len(route_ids)!=16 or len(set(route_ids))!=16
        or any(type(x) is not int or not 0<=x<248320 for x in route_ids)):
        raise ValueError('Complete actual public prompt and16 unique public routing tokens required')
    if state.get('model') is not model or state.get('consumed') or state['tokens']!=prompt_ids[:-16]:raise ValueError('Exact pristine same-model actual prompt[:-16] prefix required')
    return prompt_ids[-16:]

def plain_scores(backbone,adapter,state,prompt_ids,route_ids):
    from .parallel_decode import clone_prefix_state
    mx=backbone.mx;model=backbone.model;suffix=contract(state,prompt_ids,route_ids,model)
    branch=clone_prefix_state(backbone,state)
    hidden=model.model(mx.array([suffix]),cache=branch['cache'])
    if bool(model.language_model.args.tie_word_embeddings):raise ValueError('Pinned separate LM head required')
    logits=model.language_model.lm_head(hidden[:,-1,:]).astype(mx.float32).reshape(-1)[mx.array(route_ids)]
    mx.eval(logits,[c.state for c in branch['cache']]);mx.synchronize();backbone.check_memory()
    return logits

def gradient_scores(backbone,adapter,item):
    """Target-free same-gradient-mode norm/projection on cached public features."""
    scoring.validate(item)
    import mlx.nn as nn
    original={k:getattr(adapter,k) for k in ('lora_a','lora_b')}
    def value():
        scores=scoring.score(backbone,adapter,item)
        return backbone.mx.sum(scores),scores
    try:
        (scalar,scores),gradients=nn.value_and_grad(adapter,value)()
        backbone.mx.eval(scalar,scores,gradients);backbone.mx.synchronize();backbone.check_memory()
        return scores
    finally:
        for k,v in original.items():setattr(adapter,k,v)

def capture(backbone,adapter,state,prompt_ids,route_ids):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from .parallel_decode import clone_prefix_state
    from .native_batch_diagnosis import state_hash
    from .semantic_copy_gradient_features import _host_tensor
    mx=backbone.mx;model=backbone.model;layer=model.layers[-1];suffix=contract(state,prompt_ids,route_ids,model)
    if model.training or adapter.training or layer.mlp.down_proj is not adapter or adapter.dropout._p_1!=1:raise ValueError('Serial final-down eval adapter with no dropout required')
    roster=dict(tree_flatten(model.trainable_parameters()))
    if len(roster)!=2 or any('.31.mlp.down_proj.' not in k or k.rsplit('.',1)[-1] not in {'lora_a','lora_b'} for k in roster):raise ValueError('Only final-down A/B trainable transform required')
    if tuple(adapter.lora_a.shape)!=(12288,4) or tuple(adapter.lora_b.shape)!=(4,4096) or adapter.scale!=20:raise ValueError('Pinnedrank4scale20 adapter required')
    original={k:getattr(adapter,k) for k in ('lora_a','lora_b')}
    before_params={k:_host_tensor(mx,v)[1] for k,v in original.items()};before=state_hash(state['cache'])
    norm=layer.post_attention_layernorm;linear=adapter.linear;saved={};calls={k:0 for k in ('x','h','y')}
    class NormTap(nn.Module):
        def __init__(self):super().__init__();self.base=norm
        def __call__(self,h):saved['h']=h;calls['h']+=1;return self.base(h)
    class DownTap(nn.Module):
        def __init__(self):super().__init__();self.base=linear
        def __call__(self,x):saved['x']=x;calls['x']+=1;saved['y']=self.base(x);calls['y']+=1;return saved['y']
    def value():
        branch=clone_prefix_state(backbone,state)
        hidden=model.model(mx.array([suffix]),cache=branch['cache'])
        if bool(model.language_model.args.tie_word_embeddings):raise ValueError('Pinned separate LM head required')
        scores=model.language_model.lm_head(hidden[:,-1,:]).astype(mx.float32).reshape(-1)[mx.array(route_ids)]
        return mx.sum(scores),{**{k:saved[k] for k in ('x','h','y')},'scores':scores}
    try:
        layer.post_attention_layernorm=NormTap();adapter.linear=DownTap()
        (scalar,aux),gradients=nn.value_and_grad(adapter,value)()
        mx.eval(scalar,aux,gradients);mx.synchronize();backbone.check_memory()
        if calls!={k:1 for k in ('x','h','y')}:raise ValueError('Exactlyone actual16token suffix capture required')
        if not bool(mx.isfinite(scalar).item()):raise FloatingPointError('Nonfinite public logit sum')
        flat=dict(tree_flatten(gradients))
        if set(flat)!={'lora_a','lora_b'} or any(not bool(mx.all(mx.isfinite(v)).item()) for v in flat.values()):raise FloatingPointError('Nonfinite or wrong public-transform gradient roster')
        hosts={k:_host_tensor(mx,aux[k]) for k in ('x','h','y')}
        item={k:v[0] for k,v in hosts.items()};item.update(dtypes={k:v[1]['dtype'] for k,v in hosts.items()},routeTokenIds=list(route_ids));scoring.validate(item)
        proof={'version':'persona-choice-public-gradient-capture.v1','targetOrLossProvided':False,'optimizerUpdates':0,
               'scalar':'sum16publicroutinglogits','promptTokenIdsSha256':token_hash(prompt_ids),
               'featureTensors':{k:v[1] for k,v in hosts.items()},'nativeRouteScores':_host_tensor(mx,aux['scores'])[0].tolist(),
               'nativeRouteScoresTensor':_host_tensor(mx,aux['scores'])[1]}
    finally:
        layer.post_attention_layernorm=norm;adapter.linear=linear
        for k,v in original.items():setattr(adapter,k,v)
        after_params={k:_host_tensor(mx,getattr(adapter,k))[1] for k in original}
        if state_hash(state['cache'])!=before or after_params!=before_params:raise ValueError('Target-free capture mutated sourcecache or adapter')
    proof['sourceCacheUnchanged']=True;proof['adapterUnchanged']=True
    return item,proof
