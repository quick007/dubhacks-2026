"""Offline independent authoring source: contains gold, NEVER an inference input.

Reads frozen PUBLIC TRAIN and catalog products only. It does not read TRAIN
annotations, model outputs, weights or native libraries. No optimizer exists.
"""
from copy import deepcopy
from collections import Counter
from itertools import combinations,permutations
import random
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
OUT=HERE/'persona_choice_independent_eval_v3'
PLAN=HERE/'PERSONA_CHOICE_INDEPENDENT_EVAL_PLAN_V1.json'
NEW=HERE/'synthetic_history_train_expansion_v3/profiles.train.public.json'
OLD=HERE/'synthetic_history_train_contrasts_v2/profiles.train.public.json'
EXPECTED_NEW='391c08b9cc80619d0c9d3723289d776f1aa70d02a86dae73d3c370f57fcc46f4'
EXPECTED_PLAN='cd66afd36c32ace7d81ac8dde4003355346196d81616a3cea27a9fbb72759704'

# Public-only semantic dedup notes, manually inferred from new TRAIN request
# backgrounds. These are not copied from TRAIN annotations or generator logic.
PUBLIC_NEED_SETS=[
 ['mug','notebook'],['mug','bottle','notebook'],['mug','blanket','lamp'],['mug','lamp'],
 ['pack','bottle'],['pack','lunchbox','bottle'],['pack','headphones','organizer'],['pack','sweatshirt'],
 ['notebook','mug'],['notebook','lamp','organizer'],['notebook','headphones','lamp'],['notebook','organizer'],
 ['jacket','sneakers'],['jacket','tote','pack'],['jacket','bottle','sneakers'],['jacket','pack'],
 ['bottle','towel'],['bottle','lunchbox','towel'],['bottle','mug','notebook'],['bottle','headphones'],
 ['tote','notebook'],['tote','lunchbox','bottle'],['tote','blanket','mug'],['tote','sneakers'],
 ['blanket','mug'],['blanket','lamp','mug'],['blanket','sweatshirt','lamp'],['blanket','notebook'],
 ['lamp','notebook'],['lamp','blanket','notebook'],['lamp','headphones','notebook'],['lamp','mug'],
 ['lunchbox','bottle'],['lunchbox','tote','bottle'],['lunchbox','organizer','bottle'],['lunchbox','mug'],
 ['headphones','organizer'],['headphones','notebook','organizer'],['headphones','sweatshirt','notebook'],['headphones','duffel'],
 ['sneakers','towel'],['sneakers','sunglasses','bottle'],['sneakers','jacket','organizer'],['sneakers','tote'],
 ['sweatshirt','mug'],['sweatshirt','headphones','mug'],['sweatshirt','blanket','lamp'],['sweatshirt','pack'],
 ['duffel','organizer'],['duffel','towel','organizer'],['duffel','notebook','organizer'],['duffel','headphones'],
 ['sunglasses','tote'],['sunglasses','sneakers','bottle'],['sunglasses','bottle','tote'],['sunglasses','organizer'],
 ['organizer','pack'],['organizer','headphones','notebook'],['organizer','duffel','notebook'],['organizer','notebook'],
 ['towel','bottle'],['towel','duffel','bottle'],['towel','sneakers','bottle'],['towel','lunchbox']]

# Fresh wording and combinations authored only after PUBLIC128 freeze. Product
# keys and grading logic here are private author metadata, never public prompts.
SPECS=[
 ('mug','jacket',150,['Ivory','Navy'],None,
  'Can you help me settle on one purchase from these plans?',
  ['Tea after the evening pottery class is my little ritual, and getting a cup to drink it from is the thing I would sort out first.',
   'On another part of the day I walk home through showers; a hooded outer rain layer is next on my list.']),
 ('pack','lamp',75,['Slate blue','Sand'],None,
  'I only want to buy one thing now; which gap should I fill?',
  ['I am tired of holding my computer and course books in my arms between buildings, so a hands-free everyday bag is the urgent gap.',
   'A light on the small table where I read at night would help too, after I have dealt with carrying those things.']),
 ('notebook','towel',35,['Terracotta','Clay stripe'],None,
  'Use these routines and my history to suggest the next useful item.',
  ['I sketch birds in the park and want pocket pages for the sightings I write down; that comes before my other shopping.',
   'I have also started visiting the pool and need a cloth to dry myself afterwards, which can be the following purchase.']),
 ('jacket','headphones',140,['Navy'],'Navy',
  'What would make the most sense to add this week?',
  ['The first problem is getting caught in rain on the walk to my evening classes; I want an outer shell with a hood.',
   'Later I study recordings in a shared room and would appreciate private over-ear listening, but the rainwear comes first.']),
 ('bottle','sweatshirt',80,['Graphite','Oatmeal'],None,
  'Please choose around the two jobs I have described.',
  ['At sports practice I need my own reusable container for drinking water, and that is the first job to cover.',
   'The studio where I work in the morning is cool, so an extra cotton clothing layer is my other wish, with less urgency.']),
 ('tote','duffel',90,['Olive','Natural'],None,
  'Which of these everyday gaps is worth addressing first?',
  ['The immediate need is an easy hand-carried bag for the loose produce I bring back from the weekly stalls.',
   'I also make one-night visits and would like a separate travel bag for folded clothes, but shopping trips have priority.']),
 ('blanket','organizer',60,['Rust','Ochre'],None,
  'I am making a small purchase, guided by these priorities.',
  ['When I sit reading, I want something to drape over my knees; that home comfort is what I would buy first.',
   'Loose charging leads are another annoyance, and a little zipped case to keep them together would be useful afterwards.']),
 ('lamp','bottle',70,['Ivory','Cream'],None,
  'Pick the addition that addresses my more pressing need.',
  ['The corner where I read after dark needs a tabletop light before I spend money on the rest of my list.',
   'Having a drinking-water container to take out the next day is also useful, but is the less pressing gap.']),
 ('lunchbox','sunglasses',50,['Navy band','Black'],None,
  'Help me make one sensible purchase rather than several.',
  ['I am taking yesterday\'s cooking to work and need a proper meal container first.',
   'The walk after lunch crosses a bright open square, so eyewear for the sunshine is the other thing I would eventually add.']),
 ('headphones','tote',95,['Navy'],'Navy',
  'Please take my stated priorities and purchases into account.',
  ['I practise a spoken-language course without disturbing my housemates; private over-ear audio is the first missing part.',
   'My weekend food shopping would also be easier with a simple hand-carrying bag, once the listening need is settled.']),
 ('sneakers','notebook',105,['Charcoal','Olive'],None,
  'What should come next for the activities below?',
  ['I have started going everywhere on foot and want ordinary low-top canvas footwear as the first addition.',
   'I also like writing down route ideas as I go, so a small paper notebook would be my next choice rather than the first.']),
 ('sweatshirt','lunchbox',75,['Slate blue','Rust band'],None,
  'Find the best next step in my shopping, using this context.',
  ['The workroom is chilly and I want a crewneck clothing layer to pull on; I would address that before anything else.',
   'I would then get a container for the lunches I cook at home and bring to that workspace.']),
 ('duffel','sneakers',110,['Sand','Off-white'],None,
  'I am choosing one addition for now, with the other need waiting.',
  ['My overnight visits need a hand-carried travel bag for clothes; that packing job matters most right now.',
   'Canvas low-top shoes for ordinary weekday walking are useful too, but I can leave that purchase until afterwards.']),
 ('sunglasses','blanket',65,['Amber','Ocean blue'],None,
  'Recommend the next useful thing from the needs I have explained.',
  ['Afternoon events in a sunny courtyard have made eyewear my first shopping priority.',
   'For a different part of the day I would like a lap cover while reading at home, but I would sort out the sunshine first.']),
 ('organizer','mug',35,['Sage','Cobalt'],None,
  'Can you select something useful without repeating a purchase?',
  ['The cables and little adapters around my repair desk need a zipped pouch first; keeping those pieces together is my priority.',
   'A cup for a hot drink at the same desk would be nice once that organizing job is done.']),
 ('towel','pack',75,['Sage stripe','Forest'],None,
  'Choose one item that fits the order of needs in my context.',
  ['Drying off after the early swim is the immediate gap, so I am looking for a cotton towel first.',
   'Separately, I would like a hands-free bag for the books I take to my daytime classes; that is the next job.'])]

REPLACEMENTS=[
 ('mug','jacket',150,['Cobalt','Navy'],'I need to renew something I bought, rather than add a duplicate.',
  ['The drinking cup in my purchase history has a broken handle and I have stopped using it. I want a new cup for my tea as the priority.',
   'Rainwear for my walk is still an unfilled secondary need; I am explicitly replacing the unusable cup first.']),
 ('notebook','towel',35,['Olive','Blue stripe'],'Please focus on the replacement that is actually needed.',
  ['Every page of the pocket notebook I purchased is full. I need another book with fresh writing pages before dealing with anything else.',
   'Something for drying off after the pool is also missing, but the new writing pages are what I am asking to replace first.']),
 ('headphones','tote',95,['Sand','Natural'],'Choose for my current replacement need, using the history as context.',
  ['The over-ear listening set I purchased no longer plays audio. I need to replace it so I can study recordings privately again.',
   'A bag to carry food purchases is a separate useful addition, but replacing the failed listening set comes before it.']),
 ('sneakers','notebook',105,['Off-white','Terracotta'],'I want to replace a worn-out item before adding other things.',
  ['The soles of the canvas shoes I purchased have worn through, and I have stopped wearing them. Replacement everyday footwear is my first need.',
   'A small paper book for route notes remains a second interest; I am deliberately replacing the unusable shoes.'])]

def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(name,v):
    with (OUT/name).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def signature(p):return tuple(sorted(set((e['eventType'],e.get('productId')) for e in p['request']['publicHistory'] if e['eventType']!='purchase' and e.get('productId'))))

def build():
    assert not OUT.exists() and sha(NEW)==EXPECTED_NEW and sha(PLAN)==EXPECTED_PLAN
    train=read(OLD)['profiles']+read(NEW)['profiles'];assert len(train)==448
    plan=read(PLAN)
    for p,h in plan['catalogSha256'].items():assert sha(p)==h
    source=read(HERE/'store_catalog_v1/model_inputs.json')['products'];external=read(HERE/'store_catalog_v1/external_mapping.json')['products']
    products={e['productId']:{**p,'productId':e['productId'],'variants':[{'variantId':v['variantId'],'colorName':c['colorName']} for c,v in zip(p['colorVariants'],e['colorVariants'],strict=True)]} for p,e in zip(source,external,strict=True)}
    ids=list(products);assert len(ids)==16 and [x[0] for x in SPECS]==ids and Counter(x[1] for x in SPECS)==Counter(ids)
    forbidden={frozenset(pair) for needs in PUBLIC_NEED_SETS for pair in combinations(needs,2)}
    assert all(frozenset((s[0],s[1])) not in forbidden for s in SPECS)
    seen_signatures={signature(p) for p in train};seen_requests={digest(p['request']) for p in train};seen_background={tuple(p['request']['persona']['background']) for p in train}
    profiles=[];gold=[];replacement_profiles=[];replacement_gold=[];pairs=[]
    def event(pid,index,event_type,event_id,color=None):
        p=products[pid];v=next(v for v in p['variants'] if v['colorName']==color) if color else p['variants'][index%3]
        return {'eventId':event_id,'eventType':event_type,'itemName':p['name'],'category':p['category'],'selectedColor':v['colorName'],'occurredAt':['2026-10-16T13:00:00.000Z','2026-10-12T10:00:00.000Z','2026-10-17T09:00:00.000Z','2026-10-17T18:00:00.000Z'][index%4],'source':'demo-seed','productId':pid,'variantId':v['variantId']}
    def label(profile,primary,secondary,*,replacement=False):
        req=profile['request'];owned={e['productId'] for e in req['publicHistory'] if e['eventType']=='purchase' and 'productId' in e}
        grades={pid:0 for pid in ids};grades[primary]=3
        if secondary:grades[secondary]=2
        for e in req['publicHistory']:
            if e['eventType'] in ('view','cart') and e.get('productId') not in (primary,secondary):grades[e['productId']]=1
        for pid,p in products.items():
            if p['itemPrice']>req['itemBudget'] or any(not any(v['colorName']==h['value'] for v in p['variants']) for h in req['hardRequirements']):grades[pid]=0
            if pid in owned and not (replacement and pid==primary):grades[pid]=0
        assert grades[primary]==3 and (replacement or primary not in owned)
        preferred=[];acceptable=[];policies={}
        for pid,g in grades.items():
            if not g:continue
            eligible=[v for v in products[pid]['variants'] if all(h['value']==v['colorName'] for h in req['hardRequirements'])]
            matched=next(([v for v in eligible if v['colorName'].casefold()==c.casefold()] for c in req['persona']['colorPreferences'] if any(v['colorName'].casefold()==c.casefold() for v in eligible)),[])
            chosen=matched or eligible;preferred.extend(v['variantId'] for v in chosen);acceptable.extend(v['variantId'] for v in chosen)
            policies[pid]={'policy':'first-explicit-offered-color' if matched else 'indifferent-among-source-eligible-variants','offeredEvidence':bool(matched),'variantIds':[v['variantId'] for v in chosen]}
        return {'profileId':profile['profileId'],'pairId':profile['userGroupId'],'primaryProductId':primary,'relevanceGrades':grades,'alreadyOwnedProductIds':sorted(owned),'preferredVariantIds':preferred,'acceptableVariantIds':acceptable,'variantPreferencePolicy':policies,'requestedReplacementProductIds':[primary] if replacement else [],'rationale':'Explicit public replacement need takes priority despite the recorded purchase.' if replacement else 'Respect the stated functional priority among unowned jobs; a completed usable purchase satisfies that job. View/cart interests may retain weak relevance.','judgmentSource':'independent-agent-authored-synthetic-not-observed-choice'}
    def make(spec,index,replacement=False):
        a,b,budget,colors,*rest=spec
        hard,query,bg=(None,*rest) if replacement else rest
        token=hashlib.sha256(('independent-choice-v1:'+str(index)+':'+str(replacement)).encode()).hexdigest()[:12]
        group='ind-choice-'+token;profile={'profileId':group+'-session-1','userGroupId':group,'request':{'query':query,'itemBudget':budget,'hardRequirements':[{'attribute':'color','value':hard}] if hard else [],'persona':{'background':bg+([] if replacement else ['Please cover the first job that is not already satisfied by a completed purchase; I want useful additions, not another usable copy.']),'colorPreferences':colors},'publicHistory':[]}}
        req=profile['request'];eligible=[pid for pid,p in products.items() if p['itemPrice']<=budget and (not hard or any(v['colorName']==hard for v in p['variants']))]
        assert a in eligible and b in eligible and len(eligible)>=4
        external=(not replacement and 8<=index<12)
        anchors=[('recipe postcards','Cooking supplies'),('pronunciation course subscription','Learning materials'),('walking-club membership','Activities'),('studio workbench pass','Workspace membership')]
        options=list(permutations([p for p in ids if p not in (a,b)],3));random.Random(20261018+index*104729).shuffle(options)
        primary_color=hard or next((c for c in colors if any(v['colorName']==c for v in products[a]['variants'])),None)
        for owned,view,cart in options:
            first=event(a,0,'purchase' if replacement else ('view' if index%2==0 else 'cart'),group+'-event-0',color=primary_color)
            if external:
                name,category=anchors[index-8];first={'eventId':first['eventId'],'eventType':'purchase','itemName':name,'category':category,'occurredAt':first['occurredAt'],'source':'demo-seed','external':True}
            history=[first,event(owned,1,'purchase',group+'-event-1'),event(view,2,'view',group+'-event-2'),event(cart,3,'cart',group+'-event-3')]
            req['publicHistory']=history;other=deepcopy(profile);other['profileId']=group+'-session-2';other['request']['publicHistory'][0]=event(a,0,'purchase',group+'-event-0',color=primary_color)
            if signature(profile) not in seen_signatures and (replacement or signature(other) not in seen_signatures):break
        else:raise ValueError('No independent public history combination')
        assert digest(req) not in seen_requests and tuple(req['persona']['background']) not in seen_background
        if replacement:return [profile],[label(profile,a,b,replacement=True)]
        pairs.append({'pairId':group,'profileAId':profile['profileId'],'profileBId':other['profileId'],'changedEventIndex':0,'intervention':'external-anchor-to-completed-purchase' if external else 'view-or-cart-to-completed-purchase'})
        return [profile,other],[label(profile,a,b),label(other,b,None)]
    for i,spec in enumerate(SPECS):
        ps,ls=make(spec,i);profiles.extend(ps);gold.extend(ls)
    for i,spec in enumerate(REPLACEMENTS):
        ps,ls=make(spec,i+100,replacement=True);replacement_profiles.extend(ps);replacement_gold.extend(ls)
    assert all(len(Counter(p['request']['publicHistory'][j]['productId'] for p in profiles))>=8 and max(Counter(p['request']['publicHistory'][j]['productId'] for p in profiles).values())<=6 for j in (1,2,3))
    assert len(profiles)==32 and len(replacement_profiles)==4 and Counter(r['primaryProductId'] for r in gold)=={pid:2 for pid in ids}
    assert len({p['request']['query'] for p in profiles})>=8 and len({tuple(p['request']['persona']['colorPreferences']) for p in profiles})>=8
    # Canonical public chronology required by the immutable lexical formatter.
    for profile in profiles+replacement_profiles:profile['request']['publicHistory'].sort(key=lambda e:e['occurredAt'])
    for pair in pairs:pair['changedEventIndex']=1
    controls=[];donors={}
    for name in ('owned-only-public-context','shuffled-profile','purchase-history-removed'):
        for i,p in enumerate(profiles):
            c=deepcopy(p);c['profileId']=p['profileId']+'--'+name;c['sourceProfileId']=p['profileId'];c['controlName']=name
            req=c['request']
            if name=='owned-only-public-context':req['persona']={'background':[],'colorPreferences':[]};req['publicHistory']=[e for e in req['publicHistory'] if e['eventType']=='purchase']
            elif name=='shuffled-profile':
                donor=profiles[(i+17)%32];assert donor['userGroupId']!=p['userGroupId'];req['persona']=deepcopy(donor['request']['persona']);req['publicHistory']=deepcopy(donor['request']['publicHistory']);donors[p['profileId']]=donor['profileId']
            else:req['publicHistory']=[e for e in req['publicHistory'] if e['eventType']!='purchase']
            controls.append(c)
    # Validate request/source facts using the existing public boundary, no model.
    from .store_persona_context_v1 import checked_context_request
    product_list=list(products.values())
    for p in profiles+replacement_profiles+controls:checked_context_request(p['request'],product_list)
    assert not ({p['profileId'] for p in train}&{p['profileId'] for p in profiles+replacement_profiles})
    assert not ({p['userGroupId'] for p in train}&{p['userGroupId'] for p in profiles+replacement_profiles})
    OUT.mkdir()
    wrapper=lambda ps:{'schemaVersion':'independent-public-persona-choice.v1','provenance':'Independent authored synthetic evaluation, same public catalog, no human behavior claim.','profiles':ps}
    write('main.public.json',wrapper(profiles));write('replacement.public.json',wrapper(replacement_profiles));write('controls.public.json',wrapper(controls))
    write('main.labels.private.json',{'schemaVersion':'independent-private-persona-choice-gold.v1','labels':gold});write('replacement.labels.private.json',{'schemaVersion':'independent-private-replacement-gold.v1','labels':replacement_gold})
    write('AUTHOR_ACCESS_LOG.json',{'actualNewCandidateOutputsRead':False,'newTRAINAnnotationsRead':False,'newTRAINGeneratorPrivateLogicRead':False,'publicInputsReadSha256':{str(p):sha(p) for p in (NEW,OLD,HERE/'synthetic_history_train_expansion_v3/PUBLIC_DEDUP.json',PLAN,HERE/'store_catalog_v1/model_inputs.json',HERE/'store_catalog_v1/external_mapping.json')},'catalogAccess':'products arrays only; fixture/persona/evaluator task contents not used','authoringSourceSha256':sha(__file__),'scope':'Author previously saw old32/40 development results, not this new candidate or its future cases.'})
    write('PUBLIC_DEDUP_AUDIT.json',{'TRAINPublicProfiles':448,'newTRAINPublicSha256':EXPECTED_NEW,'fullRequestExactOverlaps':0,'fullBackgroundBlockExactOverlaps':0,'viewCartSignatureOverlaps':0,'mainNeedPairs':16,'explicitNewTRAIN128NeedPairOverlaps':0,'semanticPairDedupMethod':'Manual product-function cooccurrence from public backgrounds only, all64new groups; no TRAIN annotations. Unseen wording/combinations, not unseen mechanisms or catalog.','publicNeedCooccurrencesFromTRAIN':PUBLIC_NEED_SETS,'naturalGroups':20,'mainPairs':pairs})
    card='''This frozen packet contains32 main requests in16matched pairs and4separate explicit replacement requests. It was authored only after frozen PUBLIC TRAIN128 inspection; no new TRAIN annotations/generator-private labels or new candidate outputs were read. All cases are synthetic judgments about the same16 public products, not real behavior. Main functional need pairs do not cooccur in the explicit newTRAIN128 background need lists. Words/mechanisms/catalog remain TRAIN-seen. Main thresholds remain20primary/24novel/28strongTop3/facts0; separate replacement target@1 minimum3/4. Ninety-six public input controls use the frozen definitions. Canonical prompt order determines primary scores; reverse order is robustness only. Private targets/grades/rationales and authoring source must never enter model prompts or Tiger public data. Variant preference is unique only where an explicitly preferred offered color supports it; otherwise all source-eligible variants are indifferent. Grade1 can preserve weaker public view/cart interests. Groups and complete requests/backgrounds are distinct from448TRAIN; future outputs must freeze before private gold access. All failures retain fixed denominators. These cases become development after outcomes are inspected. Source image/copy/browser readiness are separate from ranking quality.\n'''
    with (OUT/'DATA_CARD.md').open('x') as f:f.write(card)
    files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
    write('MANIFEST.json',{'version':'persona-choice-independent-packet.v3','status':'frozen-before-new-candidate-fit-or-scoring','mainRequests':32,'replacementRequests':4,'inputControlRequests':96,'naturalPlusControlRequests':132,'twoOrderPromptRecords':264,'primaryScoringOrder':'canonical','reverseOrder':'robustness-only','planSha256':EXPECTED_PLAN,'authoringSourceSha256':sha(__file__),'privateFiles':['main.labels.private.json','replacement.labels.private.json'],'files':files,'catalogSha256':plan['catalogSha256'],'shuffledProfileDonors':donors,'main32QualityGate':plan['qualityGate'],'createdAfterTRAINPublicSha256':EXPECTED_NEW,'v2ChronologyRepair':'V2preserved. V3stablysortsunchangedeventsoldest-first beforeformatter; needs/productgrades/variantpreferences unchanged, allprivategoldbytes identical.', 'draftV1PreservedAndNotUsed':'Initialpre-outcomedraft usedlexicographicdistractors; finalv2seededvariedhistories and desiredcolorcompletedpurchasefix before anyfit/scoring. Targets/thresholds unchanged.','allNewCandidateOutputsUnread':True,'labelsNeverInPrompts':True})
    print(json.dumps({'public32Sha256':sha(OUT/'main.public.json'),'public4Sha256':sha(OUT/'replacement.public.json'),'controls96Sha256':sha(OUT/'controls.public.json'),'manifestSha256':sha(OUT/'MANIFEST.json')},indent=2))


if __name__=='__main__':build()
