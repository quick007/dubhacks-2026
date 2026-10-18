"""CPU validation of this TRAIN-only packet; no private packets, model, MLX or fitting."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib,json,re
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def read(name):return json.loads((HERE/name).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def public_context_text(request):
 anchor=max(datetime.fromisoformat(e['occurredAt'].replace('Z','+00:00')) for e in request['publicHistory'])
 events=[[e['eventType'],e['itemName'],e['category'],e.get('selectedColor'),round((anchor-datetime.fromisoformat(e['occurredAt'].replace('Z','+00:00'))).total_seconds()/86400,3)] for e in request['publicHistory']]
 facts={'query':request['query'],'itemBudget':request['itemBudget'],'hardRequirements':request['hardRequirements'],'background':request['persona']['background'],'colorPreferences':request['persona']['colorPreferences'],'recentEvents_type_item_category_color_ageDays':events}
 return ('Encode public shopping observations. Purchases are already owned possessions; suggest useful complements rather than duplicates. '
  'Views and carts are interests, not ownership. Respect query, budget and hard requirements.\n'+json.dumps(facts,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False))

def check_request(request,products):
 assert set(request)=={'query','itemBudget','hardRequirements','persona','publicHistory'}
 assert 0<len(request['query'])<=200 and isinstance(request['itemBudget'],(int,float)) and request['itemBudget']>0
 assert set(request['persona'])=={'background','colorPreferences'}
 assert 1<=len(request['persona']['background'])<=6 and len(request['persona']['colorPreferences'])<=4
 assert all(isinstance(s,str) and 0<len(s)<=500 for a in request['persona'].values() for s in a)
 assert 4<=len(request['publicHistory'])<=6
 seen=set();last=None
 for event in request['publicHistory']:
  assert set(event)=={'eventId','eventType','itemName','category','selectedColor','occurredAt','source','productId','variantId'}
  assert event['eventId'] not in seen;seen.add(event['eventId'])
  assert event['eventType'] in ['view','cart','purchase'] and event['source']=='demo-seed'
  at=datetime.fromisoformat(event['occurredAt'].replace('Z','+00:00'));assert at.tzinfo and (last is None or at>=last);last=at
  product=products[event['productId']];variant=next(v for v in product['variants'] if v['id']==event['variantId'])
  assert event['itemName']==product['name'] and event['category']==product['category'] and event['selectedColor']==variant['colorName']
 for constraint in request['hardRequirements']:
  assert set(constraint)=={'attribute','value'} and constraint['attribute']=='color'
  assert any(v['colorName']==constraint['value'] for p in products.values() for v in p['variants'])
 # UUIDs, exact SKU IDs, user/session/group IDs and absolute dates are excluded from semantic input.
 text=public_context_text(request)
 for event in request['publicHistory']:
  assert event['eventId'] not in text and event['variantId'] not in text and event['occurredAt'] not in text
 return text

def check_label(profile,label,products):
 request=profile['request'];grades=label['relevanceGrades'];assert set(grades)==set(products)
 assert list(grades.values()).count(3)==1 and list(grades.values()).count(2)>=1 and set(grades.values())<={0,2,3}
 owned={e['productId'] for e in request['publicHistory'] if e['eventType']=='purchase'}
 assert set(label['alreadyOwnedProductIds'])==owned
 replacements=set(label['replacementAllowedProductIds'])
 assert replacements<=owned
 for pid in replacements:
  assert products[pid]['name'] in ' '.join(request['persona']['background'])
  assert re.search(r'replac', ' '.join(request['persona']['background']), re.I)
 for pid in owned-replacements:assert grades[pid]==0
 preferred=label['preferredVariantIds'];acceptable=label['acceptableVariantIds']
 assert len(set(preferred))==len(preferred) and len(set(acceptable))==len(acceptable)
 positive={pid for pid,grade in grades.items() if grade}
 assert len(preferred)==len(positive)
 for pid in positive:
  product=products[pid];assert product['price']<=request['itemBudget']
  choices=[v for v in product['variants'] if all(c['value']==v['colorName'] for c in request['hardRequirements'])]
  assert choices
  assert len(set(preferred)&{v['id'] for v in choices})==1
  assert {v['id'] for v in choices}==set(acceptable)&{v['id'] for v in product['variants']}
 for pid,grade in grades.items():
  if not grade:assert pid in label['negativeReasons']
 return positive

def audit():
 freeze=read('PLAN_FREEZE.json')
 for name,checksum in freeze['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==checksum
 source=ROOT/'experiments/ranker/pretrained/synthetic_history_v1/catalog-binding.json'
 assert hashlib.sha256(source.read_bytes()).hexdigest()==freeze['publicCatalogSourceSha256']
 products={p['id']:p for p in json.loads(source.read_text())['products']}
 profiles=read('profiles.train.public.json')['profiles'];labels=read('labels.train.json')['labels'];pairs=read('pairs.train.json')['pairs']
 assert len(profiles)==128 and len(labels)==128 and len(pairs)==64
 byid={p['profileId']:p for p in profiles};bylabel={l['profileId']:l for l in labels};assert len(byid)==128 and set(byid)==set(bylabel)
 groups=Counter(p['userGroupId'] for p in profiles);assert len(groups)==64 and set(groups.values())=={2}
 for profile in profiles:
  assert set(profile)=={'profileId','userGroupId','sessionId','request'}
  check_request(profile['request'],products);check_label(profile,bylabel[profile['profileId']],products)
 kinds=Counter();changes=Counter()
 for pair in pairs:
  kinds[pair['kind']]+=1;a=byid[pair['baseProfileId']];b=byid[pair['counterfactualProfileId']]
  la=bylabel[a['profileId']];lb=bylabel[b['profileId']];assert a['userGroupId']==b['userGroupId']==pair['pairId']
  assert a['request']['query']==b['request']['query']
  if pair['kind']=='view-to-owned-history-only':
   aa=deepcopy(a['request']);bb=deepcopy(b['request']);diff=[]
   for i,(ea,eb) in enumerate(zip(aa['publicHistory'],bb['publicHistory'],strict=True)):
    if ea!=eb:
     assert ea['eventType']=='view' and eb['eventType']=='purchase'
     eb['eventType']='view';assert ea==eb;diff.append(i)
   assert len(diff)==1 and aa==bb
   pid=b['request']['publicHistory'][diff[0]]['productId'];assert la['relevanceGrades'][pid]==3 and lb['relevanceGrades'][pid]==0
  elif pair['kind']=='priority-order-swap':
   positives=[pid for pid,g in la['relevanceGrades'].items() if g];assert len(positives)==2
   for pid in positives:assert la['relevanceGrades'][pid]+lb['relevanceGrades'][pid]==5
   assert a['request']['publicHistory']==b['request']['publicHistory']
  elif pair['kind']=='explicit-owned-replacement':
   assert a['request']['publicHistory']==b['request']['publicHistory'] and not la['replacementAllowedProductIds'] and len(lb['replacementAllowedProductIds'])==1
   pid=lb['replacementAllowedProductIds'][0];assert la['relevanceGrades'][pid]==0 and lb['relevanceGrades'][pid]==3
  else:
   assert la['relevanceGrades']==lb['relevanceGrades']
   assert la['preferredVariantIds']!=lb['preferredVariantIds']
  changes.update(pair['publicChangedFields'])
 assert set(kinds.values())=={16}
 queries={p['request']['query'] for p in profiles};backgrounds={digest(p['request']['persona']['background']) for p in profiles}
 assert len(queries)>=61 and 'Browse the store' in queries
 baseids={x['baseProfileId'] for x in pairs};assert len({digest(byid[i]['request']['persona']['background']) for i in baseids})==64
 colorlists={digest(p['request']['persona']['colorPreferences']) for p in profiles}
 prior=Counter(pid for l in labels for pid,g in l['relevanceGrades'].items() if g==3);assert set(prior)==set(products)
 owned=Counter(e['productId'] for p in profiles for e in p['request']['publicHistory'] if e['eventType']=='purchase')
 variants={v for l in labels for v in l['preferredVariantIds']}
 # This comparison reads only the original TRAIN-only projection, never old evaluation/public holdout packets.
 oldpath=ROOT/'experiments/ranker/pretrained/synthetic_history_train_contrasts_v2/profiles.train.public.json'
 old=json.loads(oldpath.read_text())['profiles']
 oldbackgrounds={digest(p['request']['persona']['background']) for p in old}
 overlap=backgrounds&oldbackgrounds;assert not overlap
 return {'status':'cpu-validation-pass','trainRows':128,'newGroups':64,'pairKinds':dict(kinds),'uniqueBroadQueries':len(queries),'uniqueBackgroundBlocks':len(backgrounds),'uniqueColorLists':len(colorlists),'baseBackgroundBlocks':64,'historyCountHistogram':dict(Counter(len(p['request']['publicHistory']) for p in profiles)),'purchasePositions':dict(Counter(i for p in profiles for i,e in enumerate(p['request']['publicHistory']) if e['eventType']=='purchase')),'grade3Counts':dict(prior),'globalPriorTop1Ceiling':max(prior.values())/128,'ownedNegativeClassCounts':dict(owned),'preferredVariantCoverage':len(variants),'preferredVariants':sorted(variants),'hardColorCases':sum(bool(p['request']['hardRequirements']) for p in profiles),'publicChangedFields':dict(changes),'oldTrainBackgroundExactOverlap':0,'privateEvaluationRead':False,'modelLoaded':False,'gpuUsed':False,'limitations':['Author-curated synthetic known16catalog; not human preference outcomes','Grade0 means outside requested priorities, owned or ineligible for this request','Replacement-aware behavior required: an unconditional ownership mask would invalidate16positive replacement rows','Token preflight is separate; no truncation permitted','Independent semantic audit required before fit']}

if __name__=='__main__':print(json.dumps(audit(),indent=2))
