"""One output-manifest prompt intervention; no training or validator changes."""
from __future__ import annotations
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import re
import time
import torch
from .boundary import input_rows
from .catalog_cache import ExactCatalogCache, request_prompt
from .evidence_plan import (REVISION, CONFIG, catalog_at, load_corrected, all_actions,
                            prompt_for, validate_generated, dump, digest)
from .fusion import (observable_request, encode_shared_prefix, native_decode,
                     factual_lexicon, GENERIC_WORDS, _field_claim_errors, _normalized)
from .page_plan import make_page_plan, validate_page_plan


ROOT=Path(__file__).resolve().parents[3]
CACHE_RUN=Path('data/ranker/pretrained/catalog-cache-v1')
PLAN_RUN=Path('data/ranker/pretrained/evidence-plan-v1')
OUT=Path('data/ranker/pretrained/generation-diagnosis-v1')


def expected_fields(plan):
    sections=plan['sections'];cards=[c for s in sections for c in s['products']]
    return [f'S{i}' for i in range(1,len(sections)+1)]+[f'{key}{i}' for i in range(1,len(cards)+1) for key in ('T','B')]


def manifest_prompt(plan,request,catalog,include_observations):
    """Exact label/card/section bindings replace open-ended protocol instructions.

    This supplies an output schema, not prewritten titles. Decision/control enum
    GENERATE is never rendered as a possible badge value.
    """
    keys=expected_fields(plan);cards=[c for s in plan['sections'] for c in s['products']]
    instruction=(f"Write exactly {len(keys)} plain lines with these labels in this order: {', '.join(keys)}. "
        f"After {keys[-1]} stop. No other labels, blank lines, Markdown, copied instructions or explanations. "
        "T is a NEW 4-8 word title, different from every approved catalog title. "
        "B is a 2-5 word badge using supported evidence, or exactly NONE. "
        "S is a short heading factual for ALL its bound products. "
        "Use only the allowed framing words and each bound product's factual words. No added claims or numeric claims.\n")
    bindings=[]
    offset=0
    for i,section in enumerate(plan['sections'],1):
        common=set.intersection(*(set(catalog[c['productId']]['tags']) for c in section['products']))
        indices=list(range(offset+1,offset+len(section['products'])+1))
        offset+=len(section['products'])
        bindings.append(f"S{i}: heading for cards {indices}; only shared factual tags={sorted(common)}; if empty use framing words only.")
    for i,card in enumerate(cards,1):
        product=catalog[card['productId']];allowed,_,_=factual_lexicon(product,catalog)
        supported=[fid.rsplit(':',1)[1] for fid in card['approvedFactIds']]
        bindings.append(f"T{i}/B{i} bind ONLY to {product['title']}; true tags={product['tags']}; factual words={' '.join(sorted(allowed-GENERIC_WORDS))}; title emphasis={supported}; badge support={supported if supported else 'NONE ONLY'}.")
    content=instruction+'\n'.join(bindings)+'\nAllowed framing words: '+' '.join(sorted(GENERIC_WORDS))
    if include_observations:
        rows,_=input_rows(request,catalog)
        content=rows[0].split('\nProduct:',1)[0]+'\n'+content
    return [{'role':'user','content':content}]


def cases_and_saved(dataset):
    """Same ten cache conditions and twelve original plan-arm cases; no selection by outcome."""
    catalog=catalog_at(dataset)
    cache_cases=json.loads((CACHE_RUN/'case-plan.json').read_text())['cases']
    saved_cache=json.loads((CACHE_RUN/'benchmark.json').read_text())['records']
    by_count={case['cards']:case['request'] for case in cache_cases}
    original_chosen=saved_cache[0]['pagePlan']['sections'][0]['products'][0]['productId']
    cases=[]
    for index,saved in enumerate(saved_cache):
        request=copy.deepcopy(by_count[saved['cards']]);cat=copy.deepcopy(catalog)
        if saved['condition']=='changed-profile-B':
            request['history']=[]
            other=next(catalog[pid] for pid in request['candidateProductIds'] if set(catalog[pid]['tags'])!=set(catalog[original_chosen]['tags']))
            request['context'].update(statedPreferences=other['tags'][:],task='I want a backpack emphasizing '+', '.join(other['tags'])+'.')
        elif saved['condition']=='changed-catalog-unavailable':
            cat[original_chosen]['inStock']=False
        cases.append({'caseId':f'cache-{index}-{saved["condition"]}', 'source':'cache','request':request,'catalog':cat,
            'cards':saved['cards'],'cold':not saved['cache']['hit'],'saved':saved})
    requests={case['caseId']:case for case in json.loads((PLAN_RUN/'case-plan.json').read_text())['cases']}
    original=json.loads((PLAN_RUN/'benchmark-revalidated.json').read_text())
    for row in original['records']:
        case=requests[row['caseId']]
        cases.append({'caseId':'evidence-'+row['caseId'],'source':'evidence','request':case['request'],'catalog':catalog,
            'cards':case['cards'],'cold':False,'saved':row['arms']['plan']})
    return cases


def field_audit(raw,finish,plan,request,catalog):
    """Overlapping mechanism labels; never changes acceptance or 'fixes' text."""
    categories=set();details=[];parsed={};seen=[];blank=0
    for line in raw.splitlines():
        if not line.strip():blank+=1;continue
        match=re.fullmatch(r'(S\d+|T\d+|B\d+): (.+)',line.strip())
        if match:
            seen.append(match[1]);parsed.setdefault(match[1],match[2])
            if re.search(r'\b[TB]\d+:',match[2]):categories.add('fused_fields')
        else:categories.add('unparseable_or_markdown_format')
    expected=expected_fields(plan)
    if set(parsed)!=set(expected):categories.add('wrong_field_set')
    if len(seen)!=len(set(seen)):categories.add('duplicate_field_labels')
    if set(parsed)==set(expected) and seen!=expected:categories.add('wrong_field_order')
    if blank:categories.add('blank_separator_strictness')
    if finish!='stop':categories.add('token_budget_truncation_or_nonstop')
    cards=[c for section in plan['sections'] for c in section['products']]
    products={f'{kind}{i}':catalog[card['productId']] for i,card in enumerate(cards,1) for kind in ('T','B')}
    for i,section in enumerate(plan['sections'],1):
        common=set.intersection(*(set(catalog[c['productId']]['tags']) for c in section['products']))
        products[f'S{i}']={'title':'Everyday travel choices','tags':sorted(common),'category':'backpacks'}
    semantic=dict(parsed);semantic_origins={key:'required-protocol' for key in parsed}
    section_index=None
    for line in raw.splitlines():
        section=re.fullmatch(r'# Section (\d+)',line.strip())
        heading=re.fullmatch(r'## (.+)',line.strip())
        card_line=re.fullmatch(r'Card (\d+): (.+)',line.strip())
        if section:section_index=int(section[1])
        elif heading and section_index is not None:
            key=f'S{section_index}';semantic.setdefault(key,heading[1]);semantic_origins[key]='nonprotocol-markdown-audit-only'
        elif card_line:
            key=f'T{card_line[1]}';semantic.setdefault(key,card_line[2]);semantic_origins[key]='nonprotocol-card-prose-audit-only'
    if 'additional factual words=' in raw or 'tags=[' in raw:
        categories.add('copied_input_instead_of_requested_output')
    for key,text in semantic.items():
        if key not in products:continue
        product=products[key];errors=[] if key.startswith('B') and text=='NONE' else _field_claim_errors(text,product,catalog)
        item={'field':key,'text':text,'origin':semantic_origins[key],'catalogTags':product['tags'],'claimErrors':errors}
        if any(e.startswith('attribute_not_in_selected_product_tags:') or e.startswith('number_not_in_selected_catalog_facts:') for e in errors):
            categories.add('validator_flagged_bound_fact_requires_context_review')
        if any(e.startswith('unsupported_claim_terms:') for e in errors):categories.add('forbidden_claim_terms')
        outside=[e for e in errors if e.startswith('outside_factual_or_generic_lexicon:')]
        if outside:categories.add('conservative_lexical_rejection_needs_review')
        if key.startswith('T'):
            support=[fid.rsplit(':',1)[1] for fid in cards[int(key[1:])-1]['approvedFactIds']]
            item['observedSupportedTags']=support
            if support and not any((' '+_normalized(tag)+' ') in (' '+_normalized(text)+' ') for tag in support):
                categories.add('title_missing_required_observed_attribute')
            names=[p['title'] for p in catalog.values() if _normalized(p['title']) in _normalized(text) and p['title']!=product['title']]
            if names:categories.add('other_catalog_product_identity_in_bound_title');item['otherCatalogTitles']=sorted(set(names))
            approved={_normalized(product['title']),*(_normalized(v['title']) for v in product['presentationVariants'])}
            if _normalized(text) in approved:categories.add('copied_approved_title_not_new')
        if key.startswith('B'):
            card=cards[int(key[1:])-1];support=[fid.rsplit(':',1)[1] for fid in card['approvedFactIds']]
            item['observedSupportedTags']=support
            if text=='GENERATE':categories.add('control_enum_copied_as_content')
            if text!='NONE' and (not support or not any((' '+_normalized(tag)+' ') in (' '+_normalized(text)+' ') for tag in support)):
                categories.add('badge_factuality_distinct_from_missing_personalization_evidence')
        details.append(item)
    return {'categories':sorted(categories),'expectedLabels':expected,'parsedLabels':seen,'exactFieldOrder':seen==expected,'blankLines':blank,
            'fieldDetails':details,'rawOutput':raw,'finishReason':finish}


def diagnose(args):
    catalog=catalog_at(args.dataset);cases=cases_and_saved(args.dataset);rows=[]
    # All 10 cache outputs and all 24 evidence arms, including the single success.
    for case in cases:
        if case['source']=='cache':
            saved=case['saved'];audit=field_audit(saved['generation']['text'],saved['generation']['finishReason'],saved['pagePlan'],case['request'],case['catalog'])
            rows.append({'caseId':case['caseId'],'source':'cache','quality':saved['quality'],**audit})
    requests={case['caseId']:case for case in json.loads((PLAN_RUN/'case-plan.json').read_text())['cases']}
    for row in json.loads((PLAN_RUN/'benchmark-revalidated.json').read_text())['records']:
        for arm,saved in row['arms'].items():
            audit=field_audit(saved['generation']['text'],saved['generation']['finishReason'],saved['pagePlan'],requests[row['caseId']]['request'],catalog)
            rows.append({'caseId':row['caseId']+'-'+arm,'source':'evidence','quality':saved['quality'],**audit})
    report={'synthetic':True,'rows':rows,'categoryDefinition':'Overlapping response-level validator/mechanism flags, NOT confirmed hallucination counts. Lexical/numeric/section-identity flags need context review; generic section proxies can reject actual one-product identities. Representative true errors and conservative false negatives are independently reviewed in GENERATION_DIAGNOSIS.md. Markdown/Card prose is inspected for semantic claims only, never rescued into valid protocol fields.',
        'counts':{source:dict(Counter(category for row in rows if row['source']==source for category in row['categories'])) for source in ('cache','evidence')},
        'population':{'cache':10,'evidenceArms':24},'intervention':'Replace ambiguous open-ended labels and control enums with one exact output-label/product/evidence manifest. Everything else fixed.',
        'generationSettings':{'sampler':'greedy temperature0','budgets':CONFIG['budgets'],'retries':0,'backboneAndHeads':'unchanged; no training','validator':'unchanged original strict validator; no vocabulary/claim acceptance relaxation'},
        'contextFinding':'Saved cold/warm and A-return equality, changed-B outputs/keys and catalog invalidation do not establish corrupted cache. Exact-field-to-product binding is incorrect in examples despite public catalog facts being present.',
        'inputs':{str(p):digest(p) for p in (CACHE_RUN/'benchmark.json',PLAN_RUN/'benchmark-revalidated.json',args.dataset/'catalog.json')}}
    dump(args.out/'diagnosis.json',report);dump(Path('experiments/ranker/pretrained/GENERATION_FAILURE_DIAGNOSIS.json'),report)
    print(json.dumps({'counts':report['counts'],'population':report['population']},indent=2))


def run_arm(backbone,heads,cache,case,intervention):
    start=time.perf_counter();request=observable_request(case['request']);catalog=case['catalog']
    embeddings,actions,rank_timing,state=encode_shared_prefix(backbone,request,catalog,1536)
    with torch.inference_mode():decision=all_actions(request,catalog,actions,heads(embeddings).tolist())
    plan=make_page_plan(request,catalog,decision,case['cards']);validate_page_plan(plan,request,catalog)
    structured_ms=(time.perf_counter()-start)*1000;cache_timing=None
    if case['source']=='cache':
        del state,embeddings;backbone.mx.clear_cache();backbone.mx.synchronize()
        state,cache_timing=cache.prepare(backbone,request,catalog,case['cold'])
        messages=manifest_prompt(plan,request,catalog,True) if intervention else request_prompt(plan,request,catalog)
    else:
        messages=manifest_prompt(plan,request,catalog,False) if intervention else prompt_for(plan,request,catalog,True)
    generated=native_decode(backbone.model,backbone.tokenizer,messages,CONFIG['budgets'][str(case['cards'])],prefix_state=state)
    validate_start=time.perf_counter();completed,quality=validate_generated(generated['text'],generated['finishReason'],plan,request,catalog)
    validation_ms=(time.perf_counter()-validate_start)*1000
    backbone.mx.synchronize();backbone.check_memory()
    result={'pagePlan':completed,'generation':generated,'quality':quality,'cache':cache_timing,
        'audit':field_audit(generated['text'],generated['finishReason'],plan,request,catalog),
        'timing':{'rankingPrefix':rank_timing,'structuredMs':structured_ms,'catalogCachePrepareMs':cache_timing['completePrepareMs'] if cache_timing else 0,
            **generated['timing'],'validationMs':validation_ms,'fullCompletionMs':(time.perf_counter()-start)*1000}}
    del state
    return result


def benchmark(args):
    from .backbone import FrozenBackbone
    from .fusion import build_shared_prefix,append_generation_suffix
    torch.set_num_threads(1);cases=cases_and_saved(args.dataset)
    # Fixed BEFORE loading/evaluating the model; every prior cache condition and
    # every prior plan arm is included, regardless of success/failure.
    dump(args.out/'case-plan.json',{'synthetic':True,'cases':[{k:v for k,v in c.items() if k!='saved'} for c in cases],
        'config':CONFIG,'intervention':'exact-output-manifest-v1','training':False,'validatorChanges':False})
    heads,_,training=load_corrected(args.heads);load_start=time.perf_counter()
    backbone=FrozenBackbone(args.model,REVISION,1536,6);load_ms=(time.perf_counter()-load_start)*1000
    cache=ExactCatalogCache(args.out/'snapshots');records=[]
    # Tokenizer-only all-case preflight before first generation. Use saved plans:
    # unchanged heads/requests must reproduce them, checked again during run.
    for case in cases:
        if case['source']=='cache':
            from .catalog_cache import catalog_prefix
            tokens,messages,_=catalog_prefix(backbone.tokenizer,case['request'],case['catalog'])
        else:tokens,_,_,messages=build_shared_prefix(backbone.tokenizer,case['request'],case['catalog'],1536)
        opaque=object();state={'model':opaque,'consumed':False,'messages':messages,'tokens':tokens,'cache':[]}
        append_generation_suffix(backbone.tokenizer,manifest_prompt(case['saved']['pagePlan'],case['request'],case['catalog'],case['source']=='cache'),state,opaque)
    for index,case in enumerate(cases):
        row={'caseId':case['caseId'],'source':case['source'],'cards':case['cards'],'arms':{}}
        for arm in (('manifest','baseline') if index%2==0 else ('baseline','manifest')):
            value=run_arm(backbone,heads,cache,case,arm=='manifest')
            old_ids=[[c['productId'],c['imageCaptionId']] for s in case['saved']['pagePlan']['sections'] for c in s['products']]
            ids=[[c['productId'],c['imageCaptionId']] for s in value['pagePlan']['sections'] for c in s['products']]
            if ids!=old_ids:raise AssertionError('Selection/image plan changed; refusing prompt-only comparison')
            value['rawMatchesSavedBaseline']=value['generation']['text']==case['saved']['generation']['text'] if arm=='baseline' else None
            row['arms'][arm]=value;dump(args.out/f'case-{index}-{arm}.json',value)
            print(json.dumps({'case':index+1,'total':len(cases),'arm':arm,'fullMs':value['timing']['fullCompletionMs'],'quality':value['quality']}),flush=True)
        records.append(row);dump(args.out/'progress.json',{'records':records})
    from .routing import summary
    summaries={}
    for source in ('cache','evidence','all'):
        summaries[source]={}
        for arm in ('baseline','manifest'):
            selected=[r for r in records if source=='all' or r['source']==source];vals=[r['arms'][arm] for r in selected]
            summaries[source][arm]={'n':len(vals),'validPages':sum(v['quality']['allFieldsFulfilled'] for v in vals),
                'validPageRate':sum(v['quality']['allFieldsFulfilled'] for v in vals)/len(vals),
                'newTitleCount':sum(v['quality']['newTitleCount'] for v in vals),
                'requestedTitleCount':sum(len(s['products']) for v in vals for s in v['pagePlan']['sections']),
                'headingCount':sum(v['quality']['generatedHeadingCount'] for v in vals),
                'badgeCount':sum(v['quality']['generatedBadgeCount'] for v in vals),
                'nonstopCount':sum(v['generation']['finishReason']!='stop' for v in vals),
                'fullLatencyMs':summary([v['timing']['fullCompletionMs'] for v in vals]),
                'failureMechanisms':dict(Counter(c for v in vals for c in v['audit']['categories']))}
    report={'synthetic':True,'modelRevision':REVISION,'training':False,'headCheckpoint':training['checkpoints']['heads'],
        'intervention':'exact-output-manifest-v1 prompt only','config':CONFIG,'validatorChanges':False,
        'residentLoadMs':load_ms,'summary':summaries,'records':records,'memory':backbone.runtime_memory(),
        'baselineRawReplayMatches':sum(r['arms']['baseline']['rawMatchesSavedBaseline'] for r in records),
        'limitations':['Same previously inspected synthetic cases used for diagnosis and comparison, not untouched generalization evidence.',
            'One manifest serialization change combines exact cardinality, field-to-product binding and control/content separation; effects not separately identified.',
            'Fresh full ranker prefill and fresh generation in every arm; catalog prepare/materialization included where applicable. No response cache, training, retry or validator relaxation.',
            'Greedy64/128/224token budgets identical within each pair. Alternating order, one resident model, original cache cold/warm conditions retained.',
            'Full attempt latency includes rejected fields; only validPages counts complete accepted responses. Single run per case; no production speed or human-lift claim.',
            'Structure/evidence projection remain deterministic policies; no learned grouping or architectural novelty claim.']}
    dump(args.out/'benchmark.json',report)
    compact={k:v for k,v in report.items() if k!='records'}
    compact['perCase']=[{'caseId':r['caseId'],'source':r['source'],'cards':r['cards'],'arms':{arm:{'quality':v['quality'],'timing':v['timing'],'generation':v['generation']} for arm,v in r['arms'].items()}} for r in records]
    compact['evidence']={str(args.out/'benchmark.json'):digest(args.out/'benchmark.json'),str(args.out/'case-plan.json'):digest(args.out/'case-plan.json')}
    dump(Path('experiments/ranker/pretrained/GENERATION_MANIFEST_RESULTS.json'),compact)
    print(json.dumps({'summary':summaries,'memory':report['memory'],'baselineRawReplayMatches':report['baselineRawReplayMatches']},indent=2),flush=True)


def review(args):
    """Refine mechanism labels on saved text only; acceptance/timing immutable."""
    diagnose(args)
    path=args.out/'benchmark.json';original=json.loads(path.read_text());report=copy.deepcopy(original)
    cases={case['caseId']:case for case in cases_and_saved(args.dataset)}
    for row in report['records']:
        case=cases[row['caseId']]
        for value in row['arms'].values():
            raw=value['generation']
            value['audit']=field_audit(raw['text'],raw['finishReason'],value['pagePlan'],case['request'],case['catalog'])
    for source,arms in report['summary'].items():
        for arm,metrics in arms.items():
            values=[r['arms'][arm] for r in report['records'] if source=='all' or r['source']==source]
            metrics['failureMechanisms']=dict(Counter(category for v in values for category in v['audit']['categories']))
            metrics['exactFieldOrderCount']=sum(v['audit']['exactFieldOrder'] for v in values)
    report['classificationReview']={'generationRepeated':False,'acceptanceOrTimingChanged':False,
        'originalBenchmarkSha256':digest(path),'generationSourceCommit':'afe7e20',
        'changes':['Record required observed title attributes separately from lexical rejection.','Check exact parsed field order separately from field set.','Label validator bound-fact flags as context-review signals; section proxy numeric/brand rejection is not confirmed hallucination.']}
    dump(args.out/'benchmark-reviewed.json',report)
    compact={k:v for k,v in report.items() if k!='records'}
    compact['perCase']=[{'caseId':r['caseId'],'source':r['source'],'cards':r['cards'],
        'arms':{arm:{'quality':v['quality'],'timing':v['timing'],'generation':v['generation'],
                      'exactFieldOrder':v['audit']['exactFieldOrder'],'mechanismFlags':v['audit']['categories']} for arm,v in r['arms'].items()}} for r in report['records']]
    compact['evidence']={str(p):digest(p) for p in (path,args.out/'benchmark-reviewed.json',args.out/'case-plan.json')}
    dump(Path('experiments/ranker/pretrained/GENERATION_MANIFEST_RESULTS.json'),compact)
    print(json.dumps({'summary':report['summary'],'classificationReview':report['classificationReview']},indent=2))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('command',choices=('diagnose','benchmark','review'))
    parser.add_argument('--model',type=Path);parser.add_argument('--dataset',type=Path,default=Path('data/ranker/datasets/v2'))
    parser.add_argument('--heads',type=Path,default=Path('data/ranker/pretrained/evidence-plan-v1'))
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    if args.command=='benchmark':
        expected=ROOT/'.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots'/REVISION
        if not args.model or args.model.resolve()!=expected.resolve():parser.error('Use previously verified pinned --model snapshot')
    globals()[args.command](args)

if __name__=='__main__':main()
