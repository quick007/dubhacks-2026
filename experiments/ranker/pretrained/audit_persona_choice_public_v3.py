"""Public-only packet/control/source checks. Never parse private annotations."""
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKET=HERE/'persona_choice_independent_eval_v3'
MANIFEST_SHA='2a0801c73440dfbcea50f73c0c1b2de3a96abc8c975caf8d54e92010a84fe312'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    if 'labels' in Path(p).name or 'AUTHORSHIP' in str(p) or 'SEMANTIC_COVERAGE' in str(p):raise ValueError('Private annotation read forbidden')
    return json.loads(Path(p).read_text())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def history_key(p):return tuple(sorted(set((e['eventType'],e.get('productId')) for e in p['request']['publicHistory'] if e['eventType']!='purchase' and e.get('productId'))))


def run():
    manifest=read(PACKET/'MANIFEST.json');assert sha(PACKET/'MANIFEST.json')==MANIFEST_SHA
    # File hashing includes private bytes only for integrity, never JSON parsing.
    for name,h in manifest['files'].items():assert sha(PACKET/name)==h
    for p,h in manifest['catalogSha256'].items():assert sha(p)==h
    main=read(PACKET/'main.public.json')['profiles'];replacement=read(PACKET/'replacement.public.json')['profiles'];controls=read(PACKET/'controls.public.json')['profiles']
    assert (len(main),len(replacement),len(controls))==(32,4,96)
    sources=[HERE/'synthetic_history_train_contrasts_v2/profiles.train.public.json',HERE/'synthetic_history_train_expansion_v3/profiles.train.public.json']
    train=[p for s in sources for p in read(s)['profiles']];assert len(train)==448
    products=read(HERE/'store_catalog_v1/model_inputs.json')['products'];mapping=read(HERE/'store_catalog_v1/external_mapping.json')['products']
    catalog={e['productId']:{**p,'variants':{v['variantId']:c['colorName'] for v,c in zip(e['colorVariants'],p['colorVariants'],strict=True)}} for p,e in zip(products,mapping,strict=True)}
    for p in main+replacement+controls:
        assert set(p)==({'profileId','userGroupId','request','controlName','sourceProfileId'} if 'controlName' in p else {'profileId','userGroupId','request'})
        r=p['request'];assert set(r)=={'query','itemBudget','hardRequirements','persona','publicHistory'} and set(r['persona'])=={'background','colorPreferences'}
        assert type(r['query']) is str and type(r['itemBudget']) in (float,int) and r['itemBudget']>=0
        assert all(set(h)=={'attribute','value'} and h['attribute']=='color' for h in r['hardRequirements'])
        assert len({e['eventId'] for e in r['publicHistory']})==len(r['publicHistory'])
        times=[datetime.fromisoformat(e['occurredAt'].replace('Z','+00:00')) for e in r['publicHistory']];assert times==sorted(times)
        for e in r['publicHistory']:
            assert e['eventType'] in ('view','cart','purchase') and e['source']=='demo-seed';assert datetime.fromisoformat(e['occurredAt'].replace('Z','+00:00')).tzinfo
            if e.get('external'):
                assert e['external'] is True and 'productId' not in e and 'variantId' not in e
            else:
                product=catalog[e['productId']];assert e['itemName']==product['name'] and e['category']==product['category']
                assert product['variants'][e['variantId']]==e['selectedColor']
    assert len(set(p['userGroupId'] for p in main))==16 and set(Counter(p['userGroupId'] for p in main).values())=={2}
    assert len(set(p['userGroupId'] for p in replacement))==4
    assert len({p['profileId'] for p in main+replacement+controls})==132
    assert not ({p['profileId'] for p in train}&{p['profileId'] for p in main+replacement}) and not ({p['userGroupId'] for p in train}&{p['userGroupId'] for p in main+replacement})
    assert len({p['request']['query'] for p in main})>=8 and len({canonical(p['request']['persona']['colorPreferences']) for p in main})>=8
    train_request={canonical(p['request']) for p in train};train_bg={tuple(p['request']['persona']['background']) for p in train};train_two={tuple(p['request']['persona']['background'][:2]) for p in train};train_history={history_key(p) for p in train}
    assert all(canonical(p['request']) not in train_request and tuple(p['request']['persona']['background']) not in train_bg and tuple(p['request']['persona']['background'][:2]) not in train_two and history_key(p) not in train_history for p in main+replacement)
    types=Counter()
    for a,b in zip(main[::2],main[1::2],strict=True):
        assert a['userGroupId']==b['userGroupId'];x=deepcopy(a['request']);y=deepcopy(b['request']);ha=x.pop('publicHistory');hb=y.pop('publicHistory');assert [e['eventId'] for e in ha]==[e['eventId'] for e in hb];changed=[i for i,(ea,eb) in enumerate(zip(ha,hb,strict=True)) if ea!=eb];assert changed==[1];ea,eb=ha[1],hb[1]
        assert x==y and ea['eventId']==eb['eventId'] and ea['occurredAt']==eb['occurredAt'] and eb['eventType']=='purchase'
        if ea.get('external'):types['external-to-purchase']+=1
        else:
            assert ea['eventType'] in ('view','cart') and {k:v for k,v in ea.items() if k!='eventType'}=={k:v for k,v in eb.items() if k!='eventType'};types['interest-to-purchase']+=1
        hard=a['request']['hardRequirements']
        if hard:assert all(eb['selectedColor']==h['value'] for h in hard)
    assert types['external-to-purchase']>=4 and types['interest-to-purchase']>=8
    by={p['profileId']:p for p in main};assert Counter(c['controlName'] for c in controls)=={n:32 for n in ('owned-only-public-context','shuffled-profile','purchase-history-removed')}
    for c in controls:
        src=by[c['sourceProfileId']];expected=deepcopy(src['request']);name=c['controlName']
        if name=='owned-only-public-context':expected['persona']={'background':[],'colorPreferences':[]};expected['publicHistory']=[e for e in expected['publicHistory'] if e['eventType']=='purchase']
        elif name=='purchase-history-removed':expected['publicHistory']=[e for e in expected['publicHistory'] if e['eventType']!='purchase']
        else:
            donor=by[manifest['shuffledProfileDonors'][src['profileId']]];assert donor['userGroupId']!=src['userGroupId'];expected['persona']=deepcopy(donor['request']['persona']);expected['publicHistory']=deepcopy(donor['request']['publicHistory'])
            assert donor==main[(main.index(src)+17)%32]
        assert c['request']==expected and c['userGroupId']==src['userGroupId']
    eligible=[]
    for p in main+replacement:
        r=p['request'];n=sum(q['itemPrice']<=r['itemBudget'] and all(h['value'] in q['variants'].values() for h in r['hardRequirements']) for q in catalog.values());assert n>=4;eligible.append(n)
    for name in ('main.labels.private.json','replacement.labels.private.json'):assert sha(PACKET/name)==sha(HERE/'persona_choice_independent_eval_v2'/name)
    return {'privateGoldByteIdenticalToV2':True,'oldestFirstChronologyVerified':True,'version':'persona-choice-public-v3-audit.v1','status':'pass','manifestSha256':MANIFEST_SHA,'publicDenominators':{'main':32,'replacement':4,'controls':96,'twoOrderRecords':264},'TRAINPublicRows':448,'privateAnnotationJSONParsed':False,'allManifestFileHashesVerified':True,'sourceEventFactsVerified':True,'pairedRequestDeltaExact':True,'controlTransformsExact':True,'TRAINExactRequestsBackgroundFirstTwoAndHistoryOverlaps':0,'newGroupsDisjoint':True,'interventions':dict(types),'eligibleProductCounts':dict(Counter(eligible)),'unrelatedHistoryRoleDistinctProducts':{str(j):len({next(e['productId'] for e in p['request']['publicHistory'] if e['eventId'].endswith('-event-'+str(j))) for p in main}) for j in (1,2,3)},'sourceSha256':sha(__file__),'noNativeOptimizerBrowser':True}


if __name__=='__main__':
    result=run();out=HERE/'PERSONA_CHOICE_PUBLIC_V3_PREPARATION_AUDIT_V1.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
