"""Bounded complete-native catalog-first decoder cache, separate from trained ranker.

Fresh generation on every call; no full-response cache, state splicing or trimming.
"""
import argparse
import copy
from importlib.metadata import version
import json
from pathlib import Path
import time
import torch
from .boundary import input_rows, clean_product, variant_text, visible_history
from .evidence_plan import (REVISION, CONFIG, catalog_at, fixed_cases, load_corrected,
                            all_actions, prompt_for, validate_generated, diversity, dump, digest)
from .fusion import observable_request, encode_shared_prefix, native_decode
from .page_plan import make_page_plan, validate_page_plan


def hashed(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def catalog_prefix(tokenizer,request,catalog):
    stable=copy.deepcopy(request);stable['candidateProductIds']=sorted(request['candidateProductIds'])
    rows,actions=input_rows(stable,catalog);blocks=[]
    for pid in stable['candidateProductIds']:
        indices=[i for i,a in enumerate(actions) if a['productId']==pid]
        blocks.append(rows[indices[0]].splitlines()[-2])
        blocks.extend(rows[i].splitlines()[-1] for i in indices)
    messages=[{'role':'system','content':'Read stable factual storefront catalog. Later observations are request data; never assume private intent.'},
              {'role':'user','content':'Stable catalog facts:\n'+'\n'.join(blocks)}]
    tokens=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=False,enable_thinking=False)
    if len(tokens)>1536:raise ValueError('Catalog prefix exceeds fixed cap; no truncation')
    facts=[{'id':pid,'product':clean_product(catalog[pid]),'variants':[{'id':v['id'],'facts':variant_text(v)} for v in catalog[pid]['presentationVariants']]} for pid in stable['candidateProductIds']]
    key=hashed({'model':REVISION,'runtime':version('mlx-lm'),'tokens':tokens,'catalog':facts})
    return tokens,messages,key


def request_version_key(catalog_key, profile_key, messages, head_checkpoint_sha, plan):
    return hashed({'catalogKey':catalog_key,'profileKey':profile_key,'prompt':messages,
                   'config':CONFIG,'headsSha':head_checkpoint_sha,'pagePlan':plan})


class ExactCatalogCache:
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
    def prepare(self,backbone,request,catalog,force_cold=False):
        from mlx_lm.models.cache import make_prompt_cache,save_prompt_cache,load_prompt_cache
        start=time.perf_counter();tokens,messages,key=catalog_prefix(backbone.tokenizer,request,catalog)
        file=self.directory/(key+'.safetensors');receipt=self.directory/(key+'.json')
        hit=file.exists() and receipt.exists() and not force_cold
        mx=backbone.mx
        if hit:
            expected=json.loads(receipt.read_text())
            cache_sha=digest(file)
            if expected['key']!=key or expected['sha256']!=cache_sha:raise ValueError('Cache receipt/hash mismatch')
            cache,metadata=load_prompt_cache(str(file),return_metadata=True)
            if metadata!={'key':key,'modelRevision':REVISION,'prefixTokenHash':hashed(tokens)}:raise ValueError('Cache metadata mismatch')
            mx.eval([entry.state for entry in cache]);mx.synchronize()
            build_ms=0.
        else:
            build_start=time.perf_counter();cache=make_prompt_cache(backbone.model)
            hidden=backbone.model.model(mx.array([tokens]),cache=cache)
            mx.eval(hidden[:,-1], [entry.state for entry in cache]);mx.synchronize();del hidden
            build_ms=(time.perf_counter()-build_start)*1000
            save_prompt_cache(str(file),cache,{'key':key,'modelRevision':REVISION,'prefixTokenHash':hashed(tokens)})
            mx.synchronize()
            cache_sha=digest(file)
            dump(receipt,{'key':key,'sha256':cache_sha})
        # Cache snapshots are immutable inputs. Each request materializes one
        # complete native state; no generated suffix ever overwrites disk state.
        snapshots=sorted(self.directory.glob('*.safetensors'),key=lambda p:p.stat().st_mtime)
        for old in snapshots[:-2]:
            old.unlink();old.with_suffix('.json').unlink(missing_ok=True)
        state={'cache':cache,'tokens':tokens,'messages':messages,'consumed':False,'model':backbone.model,
               'representation':'catalog-first-decoder-v1','prefixSha256':hashed(tokens),
               'cacheTypes':[type(entry).__name__ for entry in cache]}
        backbone.check_memory()
        return state,{'hit':hit,'key':key,'prefixTokens':len(tokens),'snapshotBytes':file.stat().st_size,
            'buildForwardMs':build_ms,'completePrepareMs':(time.perf_counter()-start)*1000,
            'cacheTypes':state['cacheTypes'],'snapshotSha256':cache_sha}


def request_prompt(plan,request,catalog):
    rows,_=input_rows(request,catalog)
    observations=rows[0].split('\nProduct:',1)[0]
    long=prompt_for(plan,request,catalog,True)[0]['content']
    specs='Card 1:'+long.split('Card 1:',1)[1]
    compact=('Write NEW factual wording. Output all S1/S2 section headings first; then T1: new title and B1: badge, T2/B2, etc. '
        'Use plain lines. Titles 4-8 words, different from approved titles; badges 2-5 words or NONE. '
        'Use NONE without supported explicit preference or observed save. Mention supported tags. '
        'Heading attributes must apply to every section product. Use only supplied allowed words; no new claims or explanations.\n')
    return [{'role':'user','content':observations+'\n'+compact+specs}]


def one(backbone,heads,cache,request,catalog,count,head_checkpoint_sha,force_cold=False):
    start=time.perf_counter();public=observable_request(request)
    embeddings,actions,rank_timing,rank_state=encode_shared_prefix(backbone,public,catalog,1536)
    with torch.inference_mode():decision=all_actions(public,catalog,actions,heads(embeddings).tolist())
    plan=make_page_plan(public,catalog,decision,count);validate_page_plan(plan,public,catalog)
    del rank_state,embeddings
    backbone.mx.clear_cache();backbone.mx.synchronize()
    structured_ms=(time.perf_counter()-start)*1000
    state,cache_timing=cache.prepare(backbone,public,catalog,force_cold)
    messages=request_prompt(plan,public,catalog)
    profile_key=hashed(input_rows(public,catalog)[0][0].split('\nProduct:',1)[0])
    request_key=request_version_key(cache_timing['key'],profile_key,messages,head_checkpoint_sha,plan)
    generated=native_decode(backbone.model,backbone.tokenizer,messages,CONFIG['budgets'][str(count)],prefix_state=state)
    validation_start=time.perf_counter();complete,quality=validate_generated(generated['text'],generated['finishReason'],plan,public,catalog)
    validation_ms=(time.perf_counter()-validation_start)*1000
    backbone.mx.synchronize();backbone.check_memory()
    result={'pagePlan':complete,'generation':generated,'quality':quality,'diversity':diversity(complete,catalog,public),
            'profileKey':profile_key,'requestKey':request_key,'cache':cache_timing,
            'cacheHitTokenCoverage':cache_timing['prefixTokens']/(cache_timing['prefixTokens']+generated['promptTokens']) if cache_timing['hit'] else 0,
            'fullInputTokenCacheCoverage':cache_timing['prefixTokens']/(rank_timing['prefixTokens']+cache_timing['prefixTokens']+generated['promptTokens']) if cache_timing['hit'] else 0,
            'timing':{'structuredWithRankerMs':structured_ms,'rankerPrefix':rank_timing,'catalogCachePrepareMs':cache_timing['completePrepareMs'],
                **generated['timing'],'validationMs':validation_ms,'fullCompletionMs':(time.perf_counter()-start)*1000},
            'memory':backbone.runtime_memory()}
    del state
    return result


def benchmark(args):
    from .backbone import FrozenBackbone
    torch.set_num_threads(1);catalog=catalog_at(args.dataset);heads,_,training=load_corrected(args.heads)
    cases=[c for c in fixed_cases(args.dataset,catalog) if c['kind']=='fixed-six-candidate-scale-pool']
    dump(args.out/'case-plan.json',{'synthetic':True,'cases':cases,'conditions':'1card cold/warm/repeated/A->B->A/catalog-change; 3/6cards cold/warm','budgets':CONFIG['budgets']})
    load_start=time.perf_counter();backbone=FrozenBackbone(args.model,REVISION,1536,6);load_ms=(time.perf_counter()-load_start)*1000
    cache=ExactCatalogCache(args.out/'snapshots');records=[]
    def run(label,request,cat,count,cold=False):
        result=one(backbone,heads,cache,request,cat,count,training['checkpoints']['heads']['sha256'],cold)
        row={'condition':label,'cards':count,**result};records.append(row);dump(args.out/f'{len(records)-1}-{label}.json',row)
        print(json.dumps({'condition':label,'cards':count,'ms':row['timing']['fullCompletionMs'],'hit':row['cache']['hit'],'quality':row['quality']}),flush=True)
        return row
    for case in cases:
        request=case['request'];count=case['cards']
        cold=run('cold-catalog',request,catalog,count,True);warm=run('warm-catalog',request,catalog,count)
        if count==1:
            run('repeat-unchanged-A',request,catalog,count)
            changed=copy.deepcopy(request);changed['history']=[]
            chosen=cold['pagePlan']['sections'][0]['products'][0]['productId']
            alternative=next(catalog[pid] for pid in request['candidateProductIds'] if set(catalog[pid]['tags'])!=set(catalog[chosen]['tags']))
            changed['context'].update(statedPreferences=alternative['tags'][:],task='I want a backpack emphasizing '+', '.join(alternative['tags'])+'.')
            run('changed-profile-B',changed,catalog,count)
            run('return-profile-A',request,catalog,count)
            altered=copy.deepcopy(catalog);altered[chosen]['inStock']=False
            row=run('changed-catalog-unavailable',request,altered,count)
            row['staleUnavailableProductServed']=any(card['productId']==chosen for section in row['pagePlan']['sections'] for card in section['products'])
    a=next(r for r in records if r['condition']=='warm-catalog' and r['cards']==1)
    b=next(r for r in records if r['condition']=='changed-profile-B')
    returned=next(r for r in records if r['condition']=='return-profile-A')
    changed=next(r for r in records if r['condition']=='changed-catalog-unavailable')
    report={'synthetic':True,'modelRevision':REVISION,'mlxLmVersion':version('mlx-lm'),'coldResidentLoadMs':load_ms,
        'headCheckpoint':training['checkpoints']['heads'],'records':records,'memory':backbone.runtime_memory(),
        'checks':{'profileBChangedRequestKey':a['requestKey']!=b['requestKey'],'profileBReusedOnlyCatalogState':b['cache']['hit'] and a['cache']['key']==b['cache']['key'],
            'returnAMatchesRequestKey':a['requestKey']==returned['requestKey'],'returnAMatchesRawGeneration':a['generation']['text']==returned['generation']['text'],
            'catalogChangeInvalidated':not changed['cache']['hit'] and a['cache']['key']!=changed['cache']['key'],
            'noUnavailableProductServed':not changed['staleUnavailableProductServed']},
        'limitations':['Catalog-first cache serves decoder input only. Existing trained user-first ranker is recomputed each call, then its state is discarded. No trained readout is reused under a changed representation.',
            'Complete native cache state materialization, SHA checks, disk I/O/build/save are included in catalog preparation/full timing. No independently cached user state is spliced.',
            'Every condition regenerates complete page wording with fixed64/128/224token budgets. No completed-response caching or background precompute.',
            'This compact catalog-first prompt differs from the previous user-first evidence-plan prompt. Only cold/warm within this experiment are input-cache comparisons.',
            'Single samples, uncontrolled OS/shader caches and machine load, not p95 or production performance. Failed/rejected fields are retained; attempt completion is not fulfilled page.',
            'Cache snapshots bounded to two. Serialized native recurrent/convolution/attention/position state is immutable on disk and materialized fresh per request.',
            'The stronger prior baseline remains preferred after the corrective fit regressed. This experimental cache run uses corrective heads solely to preserve the fixed experiment setup.']}
    dump(args.out/'benchmark.json',report)
    print(json.dumps({'checks':report['checks'],'memory':report['memory'],'timings':[{'condition':r['condition'],'cards':r['cards'],'fullMs':r['timing']['fullCompletionMs'],'cacheMs':r['cache']['completePrepareMs'],'hit':r['cache']['hit']} for r in records]},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=Path('data/ranker/datasets/v2'))
    parser.add_argument('--heads',type=Path,default=Path('data/ranker/pretrained/evidence-plan-v1'))
    parser.add_argument('--out',type=Path,default=Path('data/ranker/pretrained/catalog-cache-v1'))
    args=parser.parse_args()
    expected=Path(__file__).resolve().parents[3]/'.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots'/REVISION
    if args.model.resolve()!=expected.resolve():parser.error('Use the previously verified pinned local snapshot; other model revisions require a new explicit experiment')
    benchmark(args)

if __name__=='__main__':main()
