"""Exact literal catalog-prefix reuse for the new decision training/runtime path.

CPU-importable. Caller owns native lifecycle and frozen whole prompts. This is a
distinct path from full-prompt/no-cache pilot; no whole-shape bit-parity claim.
"""
from contextlib import nullcontext
from . import store_persona_choice_features_v2 as features

def checked_prefix(row):
    whole=row['tokenIds'];prefix=row.get('catalogPrefixTokenIds')
    if (not isinstance(prefix,list) or not prefix or len(prefix)>=len(whole)-16
        or whole[:len(prefix)]!=prefix):raise ValueError('Exact whole-prompt literal public-catalog token prefix required')
    return prefix

def prefill(backbone,adapter,tokens,cache, *, enabled=False,step=256):
    from .semantic_copy_training import disabled
    mx=backbone.mx
    with nullcontext() if enabled else disabled(backbone.model,adapter):
        for offset in range(0,len(tokens),step):
            backbone.choice_prefill_attempts=getattr(backbone,'choice_prefill_attempts',0)+1
            hidden=backbone.model.model(mx.array([tokens[offset:offset+step]]),cache=cache)
            mx.eval(hidden,[c.state for c in cache]);mx.synchronize();backbone.check_memory();del hidden
            backbone.choice_prefill_completed=getattr(backbone,'choice_prefill_completed',0)+1

def catalog_state(backbone,adapter,row, *, enabled=False):
    from mlx_lm.models.cache import make_prompt_cache
    prefix=checked_prefix(row);cache=make_prompt_cache(backbone.model)
    prefill(backbone,adapter,prefix,cache,enabled=enabled)
    return {'model':backbone.model,'cache':cache,'tokens':list(prefix),'messages':[],'consumed':False}

def profile_state(backbone,adapter,row,source, *, enabled=False):
    from .parallel_decode import clone_prefix_state
    contract=features.choice_contract(row);prefix=checked_prefix(row)
    if source.get('model') is not backbone.model or source.get('consumed') or source['tokens']!=prefix:raise ValueError('Pristine same-model exact catalog-prefix state required')
    actual=backbone.tokenizer.encode(row['prompt'],add_special_tokens=False)
    if actual!=row['tokenIds']:raise ValueError('Fresh whole prompt IDs differ from frozen preflight')
    branch=clone_prefix_state(backbone,source)
    suffix=contract['prefixTokenIds'][len(prefix):]
    if not suffix:raise ValueError('Actual public profile/query suffix required')
    prefill(backbone,adapter,suffix,branch['cache'],enabled=enabled)
    branch['tokens']=contract['prefixTokenIds']
    return branch,{'catalogPrefixTokensReused':len(prefix),'freshPublicSuffixTokens':len(suffix),
                   'finalNativeInputTokens':contract['suffixTokenIds'],
                   'prefixPath':'literalcatalog-cache then complete public profile prefix then actual last16tokens',
                   'wholePromptNoCacheBitParityClaim':False}
