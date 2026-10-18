"""PRIVATE offline authoring source. Never a trainer/runtime/prompt input.

Source-only public dedup; no TRAIN gold, model scoring, optimization or native.
Author has inspected prior DEV outcomes, disclosed; new cases never scored.
"""
from collections import Counter
from copy import deepcopy
from itertools import combinations
import hashlib
import json
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent
OUT=HERE/'persona_choice_independent_eval_v4'
PLAN=HERE/'PERSONA_CHOICE_INDEPENDENT_EVAL_PLAN_V3.json'
PLAN_SHA='ec3cd18f18fd176c3444ad21818ee447071ca0106a2a6a5f4242642ca4355cac'

# AllA classes once; unique undirected public functions, Bcap2 (parent approved).
# Each sentence expresses an observable need, not a catalog hard specification.
SPECS=[
 ('mug','sneakers',110,['Cobalt','Off-white'],None,
  'Which purchase should I make next, following my priorities?',
  ['The first thing I want to sort out is a proper cup for the herbal tea I brew at the community kitchen.',
   'After that I want low-top canvas shoes for my daily trips around town; the drinking cup comes first.']),
 ('pack','mug',75,['Forest','Ivory'],None,
  'Help me pick the next missing piece of my routine.',
  ['I need a bag that I can wear on my back to carry my everyday belongings; getting those things out of my hands is the priority.',
   'Once that carrying need is covered, I would get a cup for the tea I have when I arrive.']),
 ('notebook','jacket',140,['Navy'],'Navy',
  'Please use the order of needs below to choose one item.',
  ['First I need a small paper book in which to write the observations I collect while volunteering; fresh note-taking pages are the immediate gap.',
   'A hooded rain shell for the wet journey home would be useful afterwards, but the writing book is the first purchase.']),
 ('jacket','lamp',135,['Ochre','Ivory'],None,
  'What is the next sensible purchase for these plans?',
  ['My main task is finding a waterproof outer layer with a hood for standing outside in the rain.',
   'A cordless tabletop lamp for my work corner is also on the list, after I have covered the rainwear need.']),
 ('bottle','blanket',65,['Steel blue','Moss'],None,
  'Choose one addition, taking account of what I have already bought.',
  ['The immediate gap is a reusable bottle for the water I take along during the day.',
   'Later I want a cotton throw to put across my legs while sitting at home; the water bottle is the earlier priority.']),
 ('tote','towel',50,['Natural','Blue stripe'],None,
  'Which unmet need is first in line for a purchase?',
  ['I would first buy a canvas hand-carrying bag for my loose fruit and vegetables from the produce stand.',
   'My next purchase would be a cotton towel for drying myself at the pool, once the shopping bag is dealt with.']),
 ('blanket','pack',75,['Ocean blue','Sand'],None,
  'Recommend the next single item for my stated sequence of needs.',
  ['I want a cotton blanket to pull over my lap during evening crossword puzzles, and that comfort item takes priority.',
   'Then I would look for a day bag worn on the back to carry my everyday things.']),
 ('lamp','tote',70,['Terracotta','Olive'],None,
  'Please identify which useful purchase is still due next.',
  ['The first job is adding a cordless table lamp beside the chair where I do close reading.',
   'A cotton canvas bag for carrying my market purchases is the second job, after I have a suitable light.']),
 ('lunchbox','headphones',100,['Rust band','Sand'],None,
  'Follow the priorities I described and suggest one purchase.',
  ['I need a lunch container to take a portion of home cooking with me; a box for that meal is the most immediate purchase.',
   'After the meal-carrying need is met, over-ear headphones would let me listen to audio on the journey.']),
 ('headphones','sunglasses',100,['Graphite','Amber'],None,
  'What should I buy now rather than leave until later?',
  ['My earlier priority is a pair of over-ear headphones for listening to the recordings I practise with.',
   'The other item I would add afterwards is sunglasses for bright walks outside.']),
 ('sneakers','lunchbox',105,['Olive','Olive band'],None,
  'Pick the next item using my needs and completed purchases.',
  ['First I want a pair of low-top canvas shoes for ordinary walks to the shops.',
   'Once that footwear is sorted, I would get a box for the lunch I make before leaving home.']),
 ('sweatshirt','duffel',95,['Oatmeal','Charcoal'],None,
  'Which gap should this purchase cover first?',
  ['A cotton crewneck sweatshirt to pull on in the cool rehearsal room is the immediate need.',
   'A canvas bag to pack clothes for short trips comes later, after I have the clothing layer.']),
 ('duffel','lunchbox',90,['Olive','Navy band'],None,
  'Suggest the next outstanding purchase from my priorities.',
  ['I need a travel bag for the clothes I take on brief visits, so that packing bag is first on my list.',
   'Afterwards I would buy a steel lunch container for bringing my cooked meal on workdays.']),
 ('sunglasses','jacket',135,['Tortoise','Forest'],None,
  'Select one item while respecting which job comes first.',
  ['Sunglasses for the sunny walk across the open grounds are the first thing I want to buy.',
   'A waterproof hooded jacket for rainy outings is another need, but it is next after the eyewear.']),
 ('organizer','sweatshirt',80,['Navy','Slate blue'],None,
  'Use my stated priorities to choose the next addition.',
  ['My first purchase would be a zipped pouch to gather the charging leads and small adapters in my work bag.',
   'A cotton crewneck for the cool room where I work can follow once those cables have a place.']),
 ('towel','lamp',70,['Clay stripe','Ivory'],None,
  'Which of these needs is the next one I should fill?',
  ['I need a cotton towel for drying off after the lunchtime swim; I would buy that before the rest of my list.',
   'Then I would get a cordless lamp for the side table I use in the evening.'])]

REPLACEMENTS=[
 ('pack',80,['Slate blue'],
  'I am replacing an item I can no longer use. What should I choose?',
  ['The day bag recorded in my purchases has torn shoulder straps and I have stopped carrying it.',
   'I want another bag worn on my back for my everyday belongings. This is an explicit replacement of the unusable bag.']),
 ('lamp',75,['Slate blue'],
  'Choose the replacement that will restore this part of my routine.',
  ['The tabletop light in my purchase history no longer turns on, so I cannot use it for reading.',
   'I am asking for another cordless table lamp to replace that failed light.']),
 ('bottle',50,['Cream'],
  'My recorded purchase is no longer available to use; help me replace it.',
  ['I lost the water bottle that appears in my purchases and have no bottle to take out with me.',
   'I want a replacement reusable drinking-water bottle, even though that product appears as a past purchase.']),
 ('duffel',95,['Sand'],
  'Please prioritize the item I explicitly need to replace.',
  ['The travel bag I purchased has a failed fastening and I have taken it out of use.',
   'I want another canvas bag for the clothes I take on short overnight visits, replacing the unusable bag.'])]


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def read(p):
    if 'labels' in Path(p).name:raise ValueError('No annotation input permitted')
    return json.loads(Path(p).read_text())
def write(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def signature(p):return tuple(sorted(set((e['eventType'],e.get('productId')) for e in p['request']['publicHistory'] if e['eventType']!='purchase' and e.get('productId'))))


def build():
    assert not OUT.exists() and sha(PLAN)==PLAN_SHA
    plan=read(PLAN);coverage=read(HERE/'PERSONA_CHOICE_PUBLIC_SEMANTIC_COVERAGE_V2.json')
    assert sha(HERE/'PERSONA_CHOICE_PUBLIC_SEMANTIC_COVERAGE_V2.json')==plan['coverageSha256']
    for path,d in coverage['inputsSha256'].items():assert sha(path)==d
    for path,d in plan['catalogSha256'].items():assert sha(path)==d
    source=read(HERE/'store_catalog_v1/model_inputs.json')['products'];mapping=read(HERE/'store_catalog_v1/external_mapping.json')['products']
    products={m['productId']:{**p,'productId':m['productId'],'variants':[{'variantId':v['variantId'],'colorName':c['colorName']} for v,c in zip(m['colorVariants'],p['colorVariants'],strict=True)]} for p,m in zip(source,mapping,strict=True)}
    ids=list(products);assert [s[0] for s in SPECS]==ids and max(Counter(s[1] for s in SPECS).values())<=2
    forbidden={frozenset(pair) for sets in (coverage['TRAIN128PublicNeedSets'],coverage['old320PublicMotifSets']) for group in sets for pair in combinations(group,2)}
    forbidden|={frozenset(p) for p in coverage['retiredDEV32PublicNeedPairs']}
    assert len({frozenset(s[:2]) for s in SPECS})==16 and all(frozenset(s[:2]) not in forbidden for s in SPECS)
    paths=[HERE/'synthetic_history_train_contrasts_v2/profiles.train.public.json',HERE/'synthetic_history_train_expansion_v3/profiles.train.public.json',HERE/'persona_choice_independent_eval_v3/main.public.json',HERE/'persona_choice_independent_eval_v3/replacement.public.json']
    past=[p for path in paths for p in read(path)['profiles']];assert len(past)==484
    seen_signature={signature(p) for p in past};seen_request={digest(p['request']) for p in past};seen_bg={tuple(p['request']['persona']['background']) for p in past}
    # Each distractor role uses each catalog product exactly once across16pairs.
    def role_permutation(excluded,rng):
        options={i:[p for p in ids if p not in excluded[i]] for i in range(16)}
        for values in options.values():rng.shuffle(values)
        order=sorted(options,key=lambda i:len(options[i]));chosen={}
        def search(k,used):
            if k==16:return True
            i=order[k]
            for p in options[i]:
                if p not in used:
                    chosen[i]=p
                    if search(k+1,used|{p}):return True
            return False
        assert search(0,set());return chosen
    def event(pid,index,kind,eid,color=None):
        p=products[pid];variant=next((v for v in p['variants'] if v['colorName']==color),p['variants'][index%3])
        return {'eventId':eid,'eventType':kind,'itemName':p['name'],'category':p['category'],'selectedColor':variant['colorName'],
                'occurredAt':['2026-10-09T08:20:00.000Z','2026-10-14T16:45:00.000Z','2026-10-16T12:10:00.000Z','2026-10-17T20:25:00.000Z'][index],
                'source':'demo-seed','productId':pid,'variantId':variant['variantId']}
    def label(profile,primary,secondary=None,replacement=False):
        q=profile['request'];owned={e['productId'] for e in q['publicHistory'] if e['eventType']=='purchase' and e.get('productId')}
        grades={p:0 for p in ids};grades[primary]=3
        if secondary:grades[secondary]=2
        for e in q['publicHistory']:
            if e['eventType'] in ('view','cart') and e.get('productId') not in (primary,secondary):grades[e['productId']]=1
        for pid,p in products.items():
            if p['itemPrice']>q['itemBudget'] or any(not any(v['colorName']==r['value'] for v in p['variants']) for r in q['hardRequirements']):grades[pid]=0
            if pid in owned and not (replacement and pid==primary):grades[pid]=0
        assert grades[primary]==3
        preferred=[];policies={}
        for pid,g in grades.items():
            if not g:continue
            eligible=[v for v in products[pid]['variants'] if all(v['colorName']==r['value'] for r in q['hardRequirements'])]
            offered=next(([v for v in eligible if v['colorName'].casefold()==c.casefold()] for c in q['persona']['colorPreferences'] if any(v['colorName'].casefold()==c.casefold() for v in eligible)),[])
            chosen=offered or eligible;preferred.extend(v['variantId'] for v in chosen)
            policies[pid]={'policy':'first-explicit-offered-color' if offered else 'indifferent-among-source-eligible-variants','offeredEvidence':bool(offered),'variantIds':[v['variantId'] for v in chosen]}
        return {'profileId':profile['profileId'],'pairId':profile['userGroupId'],'primaryProductId':primary,'relevanceGrades':grades,'alreadyOwnedProductIds':sorted(owned),
                'requestedReplacementProductIds':[primary] if replacement else [],'preferredVariantIds':preferred,'acceptableVariantIds':list(preferred),'variantPreferencePolicy':policies,
                'rationale':'Explicit unusable/lost item replacement takes priority.' if replacement else 'First explicitly prioritized need not already satisfied by a completed usable purchase; other view/cart interests are weaker.',
                'judgmentSource':'independent-authored-synthetic-not-real-behavior'}
    profiles=[];gold=[];pairs=[];role_maps=None
    for attempt in range(1000):
        rng=random.Random(2026101940+attempt);ex=[set(s[:2]) for s in SPECS]
        owned=role_permutation(ex,rng);view=role_permutation([ex[i]|{owned[i]} for i in range(16)],rng)
        cart=role_permutation([ex[i]|{owned[i],view[i]} for i in range(16)],rng)
        candidates=[];candidate_gold=[];candidate_pairs=[]
        for i,(a,b,budget,colors,hard,query,bg) in enumerate(SPECS):
            group='fresh-choice-'+hashlib.sha256(('future-public-v4:'+str(i)).encode()).hexdigest()[:12]
            q={'query':query,'itemBudget':budget,'hardRequirements':[{'attribute':'color','value':hard}] if hard else [],
               'persona':{'background':bg+['For this purchase, address the earlier need only if it is still unmet; a completed usable purchase covers that job. Then move to the next need.'],'colorPreferences':colors},'publicHistory':[]}
            color=hard or next((c for c in colors if any(v['colorName']==c for v in products[a]['variants'])),None)
            q['publicHistory']=[event(owned[i],0,'purchase',group+'-old'),event(a,1,'view' if i%2==0 else 'cart',group+'-interest',color),
                                event(view[i],2,'view',group+'-view'),event(cart[i],3,'cart',group+'-cart')]
            pa={'profileId':group+'-a','userGroupId':group,'request':q};pb=deepcopy(pa);pb['profileId']=group+'-b';pb['request']['publicHistory'][1]['eventType']='purchase'
            eligible=[pid for pid,p in products.items() if p['itemPrice']<=budget and all(any(v['colorName']==r['value'] for v in p['variants']) for r in q['hardRequirements'])]
            assert len(eligible)>=4 and a in eligible and b in eligible
            candidates.extend([pa,pb]);candidate_gold.extend([label(pa,a,b),label(pb,b)])
            candidate_pairs.append({'pairId':group,'profileAId':pa['profileId'],'profileBId':pb['profileId'],'changedEventIndex':1,'intervention':'interest-to-completed-usable-purchase'})
        if all(signature(p) not in seen_signature for p in candidates) and len({signature(p) for p in candidates})==32:
            profiles=candidates;gold=candidate_gold;pairs=candidate_pairs;role_maps={'purchase':owned,'view':view,'cart':cart,'attempt':attempt};break
    assert len(profiles)==32 and role_maps is not None
    replacements=[];replacement_gold=[]
    for i,(a,budget,colors,query,bg) in enumerate(REPLACEMENTS):
        group='fresh-replace-'+hashlib.sha256(('replacement-v4:'+str(i)).encode()).hexdigest()[:12]
        others=[p for p in ids if p!=a];random.Random(2026101970+i).shuffle(others)
        p={'profileId':group+'-a','userGroupId':group,'request':{'query':query,'itemBudget':budget,'hardRequirements':[],
           'persona':{'background':bg,'colorPreferences':colors},'publicHistory':[event(a,0,'purchase',group+'-old',colors[0]),
             event(others[0],2,'view',group+'-view'),event(others[1],3,'cart',group+'-cart')]}}
        assert signature(p) not in seen_signature
        replacements.append(p);replacement_gold.append(label(p,a,replacement=True))
    controls=[];donors={}
    for name in ('owned-only-public-context','shuffled-profile','purchase-history-removed'):
        for i,p in enumerate(profiles):
            c=deepcopy(p);c['profileId']+='--'+name;c.update(sourceProfileId=p['profileId'],controlName=name);q=c['request']
            if name=='owned-only-public-context':q['persona']={'background':[],'colorPreferences':[]};q['publicHistory']=[e for e in q['publicHistory'] if e['eventType']=='purchase']
            elif name=='shuffled-profile':
                donor=profiles[(i+17)%32];assert donor['userGroupId']!=p['userGroupId'];q['persona']=deepcopy(donor['request']['persona']);q['publicHistory']=deepcopy(donor['request']['publicHistory']);donors[p['profileId']]=donor['profileId']
            else:q['publicHistory']=[e for e in q['publicHistory'] if e['eventType']!='purchase']
            controls.append(c)
    from .store_persona_context_v1 import checked_context_request
    from .store_qwen_lexical_choice_inputs_v2 import checked_request,load_public_catalog
    for p in profiles+replacements+controls:
        checked_context_request(p['request'],list(products.values()));checked_request(p['request'],load_public_catalog())
        assert p['request']['publicHistory']==sorted(p['request']['publicHistory'],key=lambda e:e['occurredAt'])
    for p in profiles+replacements:assert digest(p['request']) not in seen_request and tuple(p['request']['persona']['background']) not in seen_bg
    assert len({p['profileId'] for p in profiles+replacements+controls})==132
    assert not ({p['userGroupId'] for p in past}&{p['userGroupId'] for p in profiles+replacements})
    OUT.mkdir();wrapper=lambda p:{'schemaVersion':'independent-public-persona-choice.v1','provenance':'Authored synthetic future evaluation; same public catalog, no real-behavior claim.','profiles':p}
    for name,values in [('main',profiles),('replacement',replacements),('controls',controls)]:write(name+'.public.json',wrapper(values))
    write('main.labels.private.json',{'schemaVersion':'independent-private-persona-choice-gold.v1','labels':gold});write('replacement.labels.private.json',{'schemaVersion':'independent-private-replacement-gold.v1','labels':replacement_gold})
    write('AUTHOR_ACCESS_LOG.json',{'newTRAINAnnotationsRead':False,'newCandidateModelsScored':False,'nativeCalls':0,'optimizerUpdates':0,
        'priorDEVOutcomesKnownToAuthor':True,'caseSelectionUsesModelOutputs':False,'sourceOnlyCoverageAndRoleSeedsFixed':True,
        'publicInputsReadSha256':{str(p.resolve()):sha(p) for p in paths+[PLAN,HERE/'PERSONA_CHOICE_PUBLIC_SEMANTIC_COVERAGE_V2.json',HERE/'store_catalog_v1/model_inputs.json',HERE/'store_catalog_v1/external_mapping.json']},
        'scope':'Only catalog products arrays, public TRAIN448 and retiredDEV36 requests used; no hidden shopper/evaluator truth or newTRAINgold. Public role histories are authored synthetic observations, not observed behavior.'})
    write('PUBLIC_DEDUP_AUDIT.json',{'TRAINPublicProfiles':448,'retiredDEVProfiles':36,'fullRequestExactOverlaps':0,'backgroundExactOverlaps':0,'viewCartSignatureOverlaps':0,
        'semanticNeedPairOverlapAgainstConservativePUBLICSets':0,'uniqueUndirectedNeedPairs':16,'allAClassesOnce':True,'BCounts':dict(Counter(s[1] for s in SPECS)),
        'primaryClassCounts':dict(Counter(r['primaryProductId'] for r in gold)),'mainPairs':pairs,
        'distractorRoleCounts':{role:dict(Counter(mapping.values())) for role,mapping in role_maps.items() if role!='attempt'},'roleSelectionAttempt':role_maps['attempt'],
        'scope':'Explicit publicfunctionpair dedup; not proof of zero latentsemantic overlap. All16products/vocabulary/rulemechanisms remain seen.'})
    card='''Fresh32 known-catalog synthetic requests in16whole paired groups plus4explicit replacements,96public input controls. No futuremodel scores used to author/select cases. Every product is onceA; sixteen distinct undirected needpairs avoid conservative publicTRAIN448+retiredDEV32 cooccurrences. Three degree-one vertices make allBonce impossible without duplicate reversepairs; parent approved Bcap2 beforefit. Aggregate primarycounts disclosed in publicaudit. Distractorpurchase/view/cart roles each useall16products once; mug is not overrepresented. All32 core targets are unowned;4separate replacementtargets are legitimately owned and unusable/lost, no runtimeownershipban implied. Allrequests have chronologicalsource-bound history, factualbudgets/colors and at least4eligible products. Publicqueries/fields contain no target/grade/label IDs; observed cataloghistory identity is legitimate publiccontext. Labels are explicit authored judgments, not real behavior; uniqueofferedcolor preference only whenevidenced, otherwise indifferent allsourceeligible variants. Canonicalprimary/reverse robustness, fixed20primary24novel28top3/0facts, five strictcontributioncontrols, replacement3/4 andknownDEVhistory8/8 unchanged. Labels/sourceauthoring never go to modelprompts, publicTiger or runtime. Allmodel/controlpredictions must durablyfreeze before privategoldparse. Previous32 nowDEV; future36 becomesDEV after outcomes. No unseenhuman/catalog/mechanism claim.\n'''
    with (OUT/'DATA_CARD.md').open('x') as f:f.write(card)
    files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
    write('MANIFEST.json',{'version':'persona-choice-independent-packet.v4','status':'frozen-before-new-candidate-fit-or-scoring','mainRequests':32,'replacementRequests':4,'inputControlRequests':96,'naturalPlusControlRequests':132,'twoOrderPromptRecords':264,
        'primaryScoringOrder':'canonical','reverseOrder':'robustness-only','planSha256':PLAN_SHA,'rootOriginalGateSha256':plan['rootGateSha256'],'authoringSourceSha256':sha(__file__),
        'privateFiles':['main.labels.private.json','replacement.labels.private.json'],'files':files,'catalogSha256':plan['catalogSha256'],'shuffledProfileDonors':donors,'main32QualityGate':plan['qualityGate'],
        'allNewCandidateOutputsUnread':True,'labelsNeverInPrompts':True,'allAProductsOnce':True,'BProductMaximumCount':2,'primaryClassCounts':dict(Counter(r['primaryProductId'] for r in gold))})
    print(json.dumps({'manifestSha256':sha(OUT/'MANIFEST.json'),'public32Sha256':sha(OUT/'main.public.json'),'replacement4Sha256':sha(OUT/'replacement.public.json'),'controls96Sha256':sha(OUT/'controls.public.json'),'primaryClassCounts':dict(Counter(r['primaryProductId'] for r in gold))}))


if __name__=='__main__':build()
