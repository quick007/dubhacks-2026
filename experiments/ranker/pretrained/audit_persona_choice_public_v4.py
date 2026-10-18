"""Public-only future packet/source/control/token audit; no gold JSON parser."""
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKET=HERE/'persona_choice_independent_eval_v4'
MANIFEST_SHA='5e5ef52f1d714e0502987b8de2de838af9e5d4bcab826c8d51c98496f080971f'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    if 'labels' in Path(p).name or 'author_persona' in Path(p).name:raise ValueError('Private annotation/authoring parser forbidden')
    return json.loads(Path(p).read_text())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def signature(p):return tuple(sorted(set((e['eventType'],e.get('productId')) for e in p['request']['publicHistory'] if e['eventType']!='purchase' and e.get('productId'))))


def run():
    assert sha(PACKET/'MANIFEST.json')==MANIFEST_SHA
    manifest=read(PACKET/'MANIFEST.json')
    # Hashing private bytes is not annotation parsing.
    for name,value in manifest['files'].items():assert sha(PACKET/name)==value
    for name,value in manifest['catalogSha256'].items():assert sha(name)==value
    main=read(PACKET/'main.public.json')['profiles'];replacement=read(PACKET/'replacement.public.json')['profiles'];controls=read(PACKET/'controls.public.json')['profiles']
    assert (len(main),len(replacement),len(controls))==(32,4,96)
    past=[p for name in ('synthetic_history_train_contrasts_v2/profiles.train.public.json','synthetic_history_train_expansion_v3/profiles.train.public.json','persona_choice_independent_eval_v3/main.public.json','persona_choice_independent_eval_v3/replacement.public.json') for p in read(HERE/name)['profiles']]
    assert len(past)==484
    pids={p['profileId'] for p in past};groups={p['userGroupId'] for p in past}
    requests={canonical(p['request']) for p in past};backgrounds={tuple(p['request']['persona']['background']) for p in past};signatures={signature(p) for p in past}
    source=read(HERE/'store_catalog_v1/model_inputs.json')['products'];mapping=read(HERE/'store_catalog_v1/external_mapping.json')['products']
    products={m['productId']:{**p,'productId':m['productId'],'variants':[{'variantId':v['variantId'],'colorName':c['colorName']} for v,c in zip(m['colorVariants'],p['colorVariants'],strict=True)]} for p,m in zip(source,mapping,strict=True)}
    from .store_persona_context_v1 import checked_context_request
    from . import store_qwen_lexical_choice_inputs_v2 as formatter
    catalog=formatter.load_public_catalog()
    for p in main+replacement+controls:
        assert p['profileId'] not in pids and p['userGroupId'] not in groups
        checked_context_request(p['request'],list(products.values()));formatter.checked_request(p['request'],catalog)
        times=[datetime.fromisoformat(e['occurredAt'].replace('Z','+00:00')) for e in p['request']['publicHistory']]
        assert times==sorted(times)
    for p in main+replacement:
        assert canonical(p['request']) not in requests and tuple(p['request']['persona']['background']) not in backgrounds and signature(p) not in signatures
    counts=Counter(p['userGroupId'] for p in main);assert len(counts)==16 and set(counts.values())=={2}
    assert len({p['profileId'] for p in main+replacement+controls})==132
    changed_products=[]
    for a,b in zip(main[::2],main[1::2],strict=True):
        assert a['userGroupId']==b['userGroupId'];qa=deepcopy(a['request']);qb=deepcopy(b['request']);ha=qa.pop('publicHistory');hb=qb.pop('publicHistory')
        assert qa==qb and [i for i,(x,y) in enumerate(zip(ha,hb,strict=True)) if x!=y]==[1]
        assert ha[1]['eventType'] in ('view','cart') and hb[1]['eventType']=='purchase'
        assert {k:v for k,v in ha[1].items() if k!='eventType'}=={k:v for k,v in hb[1].items() if k!='eventType'}
        changed_products.append(hb[1]['productId'])
    assert Counter(changed_products)==Counter(list(products))
    role_counts={str(j):dict(Counter(p['request']['publicHistory'][j]['productId'] for p in main[::2])) for j in (0,2,3)}
    assert all(set(v)==set(products) and set(v.values())=={1} for v in role_counts.values())
    by={p['profileId']:p for p in main};assert Counter(c['controlName'] for c in controls)=={n:32 for n in ('owned-only-public-context','shuffled-profile','purchase-history-removed')}
    for c in controls:
        source=by[c['sourceProfileId']];expected=deepcopy(source['request']);assert c['userGroupId']==source['userGroupId']
        if c['controlName']=='owned-only-public-context':expected['persona']={'background':[],'colorPreferences':[]};expected['publicHistory']=[e for e in expected['publicHistory'] if e['eventType']=='purchase']
        elif c['controlName']=='purchase-history-removed':expected['publicHistory']=[e for e in expected['publicHistory'] if e['eventType']!='purchase']
        else:
            donor=by[manifest['shuffledProfileDonors'][source['profileId']]];assert donor['userGroupId']!=source['userGroupId'];expected['persona']=donor['request']['persona'];expected['publicHistory']=donor['request']['publicHistory']
        assert c['request']==expected
    eligible=[]
    for p in main+replacement:
        q=p['request'];n=sum(v['itemPrice']<=q['itemBudget'] and all(any(c['colorName']==r['value'] for c in v['variants']) for r in q['hardRequirements']) for v in products.values());assert n>=4;eligible.append(n)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(formatter.pilot.MODEL,local_files_only=True,trust_remote_code=False)
    rows=[]
    for p in main+replacement+controls:
        for order in ('canonical','reversed'):
            row=formatter.build_row(p['request'],catalog,tokenizer,p['profileId']+'-'+order,order)
            assert len(row['catalogPrefixTokenIds'])==721
            assert not any(token in row['prompt'] for token in ('primaryProductId','relevanceGrades','preferredVariantIds',p['profileId'],p['userGroupId']))
            rows.append({'caseId':row['caseId'],'order':order,'tokens':len(row['tokenIds']),'tokenIdsSha256':hashlib.sha256(json.dumps(row['tokenIds'],separators=(',',':')).encode()).hexdigest()})
    assert len(rows)==264 and len({r['caseId'] for r in rows})==264
    return {'version':'persona-choice-public-v4-audit.v1','status':'pass','manifestSha256':MANIFEST_SHA,'privateAnnotationJSONParsed':False,
        'publicDenominators':{'main':32,'replacement':4,'controls':96,'twoOrderRecords':264},'publicDedupRows':484,
        'fullRequestBackgroundViewCartSignatureOverlaps':0,'sourceHistoryFactsAndChronologyVerified':True,'pairDeltaOnlyEventTypeAtIndex1':True,
        'all16ObservedChangedProductsOnce':True,'all16EachDistractorRoleOnce':role_counts,'controlTransformsExact':True,
        'eligibleProductCounts':dict(Counter(eligible)),'wholePromptTokensMin':min(r['tokens'] for r in rows),'wholePromptTokensMax':max(r['tokens'] for r in rows),
        'catalogPrefixTokens':721,'sourceSha256':sha(__file__),'formatterSha256':sha(formatter.__file__),
        'tokenizerSha256':sha(formatter.pilot.MODEL/'tokenizer.json'),'rows':rows,'nativeCalls':0,'optimizerUpdates':0,'modelScoringCalls':0}


if __name__=='__main__':
    result=run();path=HERE/'PERSONA_CHOICE_PUBLIC_V4_PREPARATION_AUDIT_V1.json'
    with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','all16EachDistractorRoleOnce')}))
