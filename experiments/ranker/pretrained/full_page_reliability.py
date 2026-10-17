"""One locked-evidence prompt comparison against the preserved manifest baseline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from .catalog_cache import ExactCatalogCache,catalog_prefix
from .evidence_plan import CONFIG,REVISION,load_corrected,all_actions,dump,digest
from .fusion import observable_request,encode_shared_prefix,native_decode,GENERIC_WORDS
from .generation_diagnosis import (ROOT,manifest_prompt,expected_fields,cases_and_saved,field_audit)
from .page_plan import make_page_plan,validate_page_plan
from .evidence_plan import validate_generated

OUT=Path('data/ranker/pretrained/full-page-reliability-v1')
PRIOR=Path('data/ranker/pretrained/generation-diagnosis-v1/benchmark-reviewed.json')


def evidence_capsule(plan,catalog):
    """Derive universal section facts and selected-item anchors from public facts."""
    sections=[];cards=[]
    for section_index,section in enumerate(plan['sections'],1):
        members=[]
        for card in section['products']:
            index=len(cards)+1;product=catalog[card['productId']]
            supported=[value.rsplit(':',1)[1] for value in card['approvedFactIds']]
            cards.append({'index':index,'identity':product['title'],'tags':product['tags'][:],
                          'requiredObservedTags':supported,'badgeSupport':supported})
            members.append({'index':index,'tags':product['tags'][:]})
        common=set.intersection(*(set(member['tags']) for member in members)) if members else set()
        sections.append({'index':section_index,'members':members,'universalTags':sorted(common)})
    return {'sections':sections,'cards':cards}


def locked_prompt(plan,request,catalog,include_observations):
    """One compact evidence serialization; no generated wording is prewritten."""
    from .boundary import input_rows
    keys=expected_fields(plan);capsule=evidence_capsule(plan,catalog)
    instruction=(f"Write exactly {len(keys)} plain lines in order: {', '.join(keys)}. After {keys[-1]} stop. "
        "No blank lines, Markdown or explanations. Product selection is LOCKED: do not recommend or substitute another item. "
        "T is a NEW 4-8 word title: keep its selected identity and mention a required observed tag when listed. "
        "Do not copy any approved title unchanged. B is supported short badge text or NONE. "
        "S must be true for EVERY member: use ONLY universal tags, NEVER union member tags. "
        "EMPTY universal tags means ZERO product attributes in heading; use framing words only. "
        "Only selected identity/tags and allowed framing words may be asserted; no added capabilities or numeric claims.\n")
    bindings=[]
    for section in capsule['sections']:
        members=';'.join(str(member['index'])+':'+','.join(member['tags']) for member in section['members'])
        universal=','.join(section['universalTags']) or 'EMPTY'
        bindings.append(f"S{section['index']} evidence: members({members}); UNIVERSAL={universal}.")
    for card in capsule['cards']:
        i=card['index'];required=('title MUST mention='+','.join(card['requiredObservedTags'])) if card['requiredObservedTags'] else 'title=identity only, no personal preference claim'
        badge=','.join(card['badgeSupport']) or 'NONE ONLY'
        bindings.append(f"T{i}/B{i} LOCK: identity={card['identity']}; facts={','.join(card['tags'])}; {required}; badge allowed={badge}.")
    content=instruction+'\n'.join(bindings)+'\nAllowed framing words: '+' '.join(sorted(GENERIC_WORDS))
    if include_observations:
        rows,_=input_rows(request,catalog);content=rows[0].split('\nProduct:',1)[0]+'\n'+content
    return [{'role':'user','content':content}]


def run_arm(backbone,heads,cache,case,locked):
    start=time.perf_counter();request=observable_request(case['request']);catalog=case['catalog']
    embeddings,actions,rank_timing,state=encode_shared_prefix(backbone,request,catalog,1536)
    with torch.inference_mode():decision=all_actions(request,catalog,actions,heads(embeddings).tolist())
    plan=make_page_plan(request,catalog,decision,case['cards']);validate_page_plan(plan,request,catalog)
    structured_ms=(time.perf_counter()-start)*1000;cache_timing=None
    if case['source']=='cache':
        del state,embeddings;backbone.mx.clear_cache();backbone.mx.synchronize()
        state,cache_timing=cache.prepare(backbone,request,catalog,case['cold'])
    builder=locked_prompt if locked else manifest_prompt
    messages=builder(plan,request,catalog,case['source']=='cache')
    generated=native_decode(backbone.model,backbone.tokenizer,messages,CONFIG['budgets'][str(case['cards'])],prefix_state=state)
    validation_start=time.perf_counter()
    completed,quality=validate_generated(generated['text'],generated['finishReason'],plan,request,catalog)
    validation_ms=(time.perf_counter()-validation_start)*1000
    backbone.mx.synchronize();backbone.check_memory()
    result={'pagePlan':completed,'generation':generated,'quality':quality,'cache':cache_timing,
        'capsule':evidence_capsule(plan,catalog),'audit':field_audit(generated['text'],generated['finishReason'],plan,request,catalog),
        'timing':{'rankingPrefix':rank_timing,'structuredMs':structured_ms,
            'catalogCachePrepareMs':cache_timing['completePrepareMs'] if cache_timing else 0,
            **generated['timing'],'validationMs':validation_ms,'fullCompletionMs':(time.perf_counter()-start)*1000}}
    del state
    return result


def summarize(records):
    from collections import Counter
    from .routing import summary
    output={}
    for source in ('cache','evidence','all'):
        output[source]={}
        for arm in ('manifest','lockedEvidence'):
            selected=[r for r in records if source=='all' or r['source']==source]
            values=[row['arms'][arm] for row in selected]
            output[source][arm]={'n':len(values),'validPages':sum(v['quality']['allFieldsFulfilled'] for v in values),
                'validPageRate':sum(v['quality']['allFieldsFulfilled'] for v in values)/len(values),
                'newTitleCount':sum(v['quality']['newTitleCount'] for v in values),
                'requestedTitleCount':sum(len(s['products']) for v in values for s in v['pagePlan']['sections']),
                'generatedHeadingCount':sum(v['quality']['generatedHeadingCount'] for v in values),
                'requestedHeadingCount':sum(len(v['pagePlan']['sections']) for v in values),
                'generatedBadgeCount':sum(v['quality']['generatedBadgeCount'] for v in values),
                'fullLatencyMs':summary([v['timing']['fullCompletionMs'] for v in values]),
                'validOnlyLatencyMs':summary([v['timing']['fullCompletionMs'] for v in values if v['quality']['allFieldsFulfilled']]),
                'exactFieldOrderCount':sum(v['audit']['exactFieldOrder'] for v in values),
                'nonstopCount':sum(v['generation']['finishReason']!='stop' for v in values),
                'mechanismFlags':dict(Counter(flag for v in values for flag in v['audit']['categories']))}
    return output


def benchmark(args):
    from .backbone import FrozenBackbone
    from .fusion import build_shared_prefix,append_generation_suffix
    torch.set_num_threads(1);cases=cases_and_saved(args.dataset)
    prior=json.loads(PRIOR.read_text());previous={row['caseId']:row['arms']['manifest'] for row in prior['records']}
    dump(args.out/'case-plan.json',{'synthetic':True,'intervention':'locked-universal-and-item-evidence-v1','config':CONFIG,
        'priorSha256':digest(PRIOR),'cases':[{key:value for key,value in case.items() if key!='saved'} for case in cases],
        'validatorChanges':False,'training':False})
    heads,_,training=load_corrected(args.heads)
    load_start=time.perf_counter();backbone=FrozenBackbone(args.model,REVISION,1536,6);load_ms=(time.perf_counter()-load_start)*1000
    cache=ExactCatalogCache(args.out/'snapshots');records=[]
    for case in cases:
        if case['source']=='cache':tokens,messages,_=catalog_prefix(backbone.tokenizer,case['request'],case['catalog'])
        else:tokens,_,_,messages=build_shared_prefix(backbone.tokenizer,case['request'],case['catalog'],1536)
        dummy=object();state={'model':dummy,'consumed':False,'messages':messages,'tokens':tokens,'cache':[]}
        append_generation_suffix(backbone.tokenizer,locked_prompt(previous[case['caseId']]['pagePlan'],case['request'],case['catalog'],case['source']=='cache'),state,dummy)
    for index,case in enumerate(cases):
        row={'caseId':case['caseId'],'source':case['source'],'cards':case['cards'],'arms':{}}
        for arm in (('lockedEvidence','manifest') if index%2==0 else ('manifest','lockedEvidence')):
            value=run_arm(backbone,heads,cache,case,arm=='lockedEvidence')
            expected=[[c['productId'],c['imageCaptionId']] for s in previous[case['caseId']]['pagePlan']['sections'] for c in s['products']]
            observed=[[c['productId'],c['imageCaptionId']] for s in value['pagePlan']['sections'] for c in s['products']]
            if observed!=expected:raise AssertionError('Selected product/image changed')
            value['baselineRawMatchesPrior']=value['generation']['text']==previous[case['caseId']]['generation']['text'] if arm=='manifest' else None
            row['arms'][arm]=value;dump(args.out/f'case-{index}-{arm}.json',value)
            print(json.dumps({'case':index+1,'total':len(cases),'arm':arm,'fullMs':value['timing']['fullCompletionMs'],'quality':value['quality']}),flush=True)
        records.append(row);dump(args.out/'progress.json',{'records':records})
    report={'synthetic':True,'modelRevision':REVISION,'training':False,'validatorChanges':False,'config':CONFIG,
        'intervention':'locked-universal-and-item-evidence-v1 prompt serialization only','headCheckpoint':training['checkpoints']['heads'],
        'residentLoadMs':load_ms,'summary':summarize(records),'records':records,'memory':backbone.runtime_memory(),
        'manifestBaselineRawMatchesPrior':sum(r['arms']['manifest']['baselineRawMatchesPrior'] for r in records),
        'limitations':['Same previously inspected synthetic cases/repeated controls, not untouched generalization or human conversion evidence.',
            'One evidence serialization combines explicit universal membership and locked item identity/support; effects not separately ablated.',
            'Same model/heads/native prefix layout/greedy budgets and strict validator in paired arms. No retry, cache splicing, text rewrite or factuality relaxation.',
            'Every arm recomputes ranking and fresh wording. Catalog build/load included where applicable. Full attempt timing includes rejected fields.',
            'Selection, section grouping and evidence projection remain prior deterministic/structured policies. No learned layout or ranking improvement claim.',
            'Prefill reuse inspection is a separate read-only analysis; no different-order hidden state is substituted into trained heads.']}
    dump(args.out/'benchmark.json',report)
    compact={key:value for key,value in report.items() if key!='records'}
    compact['perCase']=[{'caseId':r['caseId'],'source':r['source'],'cards':r['cards'],
        'arms':{arm:{'quality':v['quality'],'generation':v['generation'],'timing':v['timing'],'capsule':v['capsule']} for arm,v in r['arms'].items()}} for r in records]
    compact['evidence']={str(path):digest(path) for path in (args.out/'benchmark.json',args.out/'case-plan.json',PRIOR)}
    dump(Path('experiments/ranker/pretrained/FULL_PAGE_RELIABILITY_RESULTS.json'),compact)
    print(json.dumps({'summary':report['summary'],'memory':report['memory'],'manifestReplayMatches':report['manifestBaselineRawMatchesPrior']},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=Path('data/ranker/datasets/v2'))
    parser.add_argument('--heads',type=Path,default=Path('data/ranker/pretrained/evidence-plan-v1'))
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    expected=ROOT/'.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots'/REVISION
    if args.model.resolve()!=expected.resolve():parser.error('Use the verified pinned model snapshot')
    benchmark(args)

if __name__=='__main__':main()
