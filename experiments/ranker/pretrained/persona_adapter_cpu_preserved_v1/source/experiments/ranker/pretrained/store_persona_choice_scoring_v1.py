"""Target-free final16 LoRA decision scoring, CPU-importable.

Serving must own serial adapter switching and source/model/checkpoint/cache pins.
No target/grade/teacher field is accepted or computed by this API.
"""
FORBIDDEN={'targetWeights','targets','relevanceGrades','primaryProductId','preferredVariantIds','caseId','personId','profileId'}

def validate(item):
    if set(item)&FORBIDDEN:raise ValueError('Target-free feature input required')
    if set(item)!={'x','h','y','dtypes','routeTokenIds'}:raise ValueError('Exact target-free feature field roster required')
    if any(tuple(item[k].shape)!=(1,16,d) for k,d in [('x',12288),('h',4096),('y',4096)]):raise ValueError('Complete native16-row feature shape required')
    if set(item['dtypes'])!={'x','h','y'} or set(item['dtypes'].values())-{'mlx.core.bfloat16','mlx.core.float16','mlx.core.float32'}:raise ValueError('Exact native dtype roster required')
    routes=item['routeTokenIds']
    if len(routes)!=16 or len(set(routes))!=16 or any(type(t) is not int or not 0<=t<248320 for t in routes):raise ValueError('16unique one-token routes required')
    return item

def score(backbone,adapter,item):
    validate(item);mx=backbone.mx
    dtypes={str(t):t for t in (mx.bfloat16,mx.float16,mx.float32)}
    x,h,y=(mx.array(item[k],dtype=dtypes[item['dtypes'][k]]) for k in ('x','h','y'))
    delta=(adapter.scale*((adapter.dropout(x)@adapter.lora_a)@adapter.lora_b)).astype(x.dtype)
    hidden=backbone.model.model.norm(h+(y+delta))
    lm=backbone.model.language_model
    if bool(lm.args.tie_word_embeddings):raise ValueError('Pinned separate native LM projection required')
    logits=lm.lm_head(hidden[:,-1,:]).astype(mx.float32)
    return logits.reshape(-1)[mx.array(item['routeTokenIds'])]

def public_item(training_item):
    """Explicit one-way projection; no IDs, labels, teacher arrays or loss terms."""
    return {k:training_item[k] for k in ('x','h','y','dtypes','routeTokenIds')}
