import json,hashlib,shutil,datetime
from pathlib import Path
root=Path(__file__).resolve().parents[4]
packet=root/'experiments/ranker/pretrained/store_shopper_eval_v1'
base=root/'experiments/ranker/pretrained/store_shopper_eval_smoke_v1/shopper-pair-02'
load=lambda p:json.loads(p.read_text())
plan=load(packet/'pair-plan.json');schedule=next(r for r in plan['schedule'] if r['caseId']=='shopper-pair-02')
case=next(r for r in load(packet/'shopper-instructions.json') if r['caseId']=='shopper-pair-02')
catalog=load(packet/'catalog-binding.json')['catalogSnapshot']
query='a bottle for drinking water under $40'
now=datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z')
for arm in ['generic','local-model']:
 d=base/arm;raw=load(d/'result.json');run_id=next(x['runId'] for x in schedule['runs'] if x['arm']==arm)
 observed=raw['selection']
 product=next(p for p in catalog['products'] if p['originalName']==observed['name'])
 variant=next(v for v in product['variants'] if v['selectedColor']==observed['variant'])
 assert observed['quantity']==1 and product['itemPrice']==observed.get('unitPriceUSD',observed.get('itemPrice'))
 entry='http://127.0.0.1:4312/'+('?model=local' if arm=='local-model' else '')
 artifacts=[]
 def artifact(name,kind='result',mime='application/json',missing=None):
  p=d/name;aid=run_id+'-'+name.replace('.','-')
  if missing:
   value={'status':'missing','sha256':None,'bytes':0,'error':missing}
  else:
   value={'status':'captured','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'error':None}
  artifacts.append({'artifact':{'id':aid,'runId':run_id,'kind':kind,'path':name,'mimeType':mime,'createdAt':now},**value});return aid
 raw_id=artifact('result.json')
 target=d/'catalog-binding.json'
 if target.exists():raise RuntimeError('Preserve prior normalized artifact')
 shutil.copyfile(packet/'catalog-binding.json',target);catalog_id=artifact('catalog-binding.json')
 artifact('screenshot.jpg','screenshot','image/jpeg','CUA screenshot exists only in native tool transcript; durable file unavailable')
 artifact('trace.zip','trace','application/zip','No Playwright trace produced by this Chrome CUA session')
 if arm=='generic':
  raw_actions=[raw['actions'][i] for i in [0,1,2,3,4,7,8,9,10]]
  acts=[{'type':'navigate','url':entry},{'type':'open-cart','target':'Initial cart empty'},
   {'type':'navigate','url':entry+'#collection'},{'type':'search','query':query},
   {'type':'inspect','target':'Visible bottle heading; no detail navigation'}, {'type':'inspect','target':'In use bottle image'},
   {'type':'add-to-cart','productId':product['productId'],'variantId':variant['variantId']},
   {'type':'open-cart','target':'Observed selected bottle Graphite quantity1 subtotal$36'},
   {'type':'confirm-demo-order','target':'Observed Demo order confirmed; no real payment'}]
  elapsed=38456;started='2026-10-18T03:48:45Z';precision='upper-bound';timing_source='Measured pre-browser and post-confirmation monotonic checkpoints; per-action clocks not recorded'
 else:
  raw_actions=raw['actions']
  acts=[{'type':'navigate','url':entry},{'type':'open-cart','target':'Initial cart empty'},
   {'type':'navigate','url':entry+'#collection'},{'type':'inspect','target':'Entered exact assigned public query'},
   {'type':'search','query':query},{'type':'inspect','target':'Visible bottle heading and partial model copy'},
   {'type':'add-to-cart','productId':product['productId'],'variantId':variant['variantId']},
   {'type':'open-cart','target':'Observed selected bottle Graphite quantity1 subtotal$36'},
   {'type':'confirm-demo-order','target':'Observed Demo order confirmed; no real payment'}]
  elapsed=35000;started=None;precision='estimated';timing_source='Approximately35s from agent report; first-action clock missing. Recorded intermediate/end checkpoints and call durations retained in result.json. Not a measured upper bound.'
 steps=[]
 for i,(action,source) in enumerate(zip(acts,raw_actions,strict=True)):
  msg=source.get('observed') or source.get('action') or source.get('api')
  kind='navigation' if action['type']=='navigate' else 'observation' if action['type']=='inspect' else 'action'
  steps.append({'event':{'id':run_id+'-step-'+str(i),'runId':run_id,'sequence':i,'at':None,'kind':kind,'message':msg},'elapsedMs':None,'action':action})
 checks={}
 for kind in ['hardConstraints','preferences']:
  checks[kind]=[]
  for need in case['privateShopperNeeds'][kind]:
   value=product['itemPrice'] if need['attribute']=='itemPrice' else product['facts'].get(need['attribute'])
   op=need['operator'];met=(value<need['value']) if op=='less-than' else (value<=need['value']) if op=='at-most' else need['value'] in value if op=='includes' else value==need['value']
   checks[kind].append({'needId':need['id'],'status':'met' if met else 'not-met','observed':json.dumps(value,separators=(',',':')),'evidenceArtifactIds':[catalog_id,raw_id]})
 readiness=[]
 if arm=='local-model':
  readiness_id=artifact('readiness-attribute.txt',mime='text/plain')
  for e in load(d/'readiness-attribute.txt'):
   detail=e['detail']
   if detail['requestId']!='a516a44d-1cd6-4dae-b962-6f537fcac17b':continue
   readiness.append({'name':e['name'],'stage':detail['stage'],'requestId':detail['requestId'],'startTimeMs':e['startTime'],'durationMs':e['duration'],'capturedAtElapsedMs':None,'acceptedFields':detail['acceptedFields'],'withheldFields':detail['withheldFields'],'requiredFields':detail['requiredFields'],'source':'data-local-model-readiness','artifactId':readiness_id,'scope':'after-paint-opportunity','physicalPaintMeasured':False,'imageTransferMeasured':False})
 out={'contractVersion':'shopper-session.v1','caseId':'shopper-pair-02','arm':arm,
  'run':{'id':run_id,'experimentId':'actual-store-shopper-smoke-v1','personaId':'shopper-pair-02','personaSplit':schedule['split'],'seed':schedule['seed']},
  'browser':{'name':'Chrome','headed':True,'actualNavigation':True,'version':None,'automation':'cua'},
  'isolation':{'kind':'fresh-tab','tabId':raw.get('tabId') or raw['freshSession']['tabId'],'contextId':None,'sharedStorage':True,'storageCleared':False,'initialCartEmpty':True,'evidenceArtifactIds':[raw_id],'note':'Separate agent-created Chrome tab with observed empty cart/fresh page; origin storage shared and untouched; provider context ID/version unknown'},
  'entryUrl':entry,'startedAt':started,'elapsedMs':elapsed,'timingPrecision':{'sessionElapsed':precision,'stepElapsed':'not-recorded','source':timing_source},
  'protocol':{'passed':False,'deviations':['Exploratory run began before final framework freeze','Provider-default viewport used; required1280x960 viewport not established','No durable screenshot or trace; choice observation remains agent-reported','Per-action timestamps unavailable']+(['Session start timestamp and exact elapsed unknown; approximately35s only'] if arm=='local-model' else [])},
  'status':'completed','error':None,'realPurchase':False,'fakeOrderConfirmed':True,'steps':steps,'artifacts':artifacts,
  'choice':{'productId':product['productId'],'variantId':variant['variantId'],'originalName':product['originalName'],'catalogPrice':product['itemPrice'],'selectedColor':variant['selectedColor'],'quantity':1,'observationSource':'agent-reported','sourceStepIds':[steps[-3]['event']['id'],steps[-2]['event']['id'],steps[-1]['event']['id']],'evidenceArtifactIds':[raw_id]},
  'verification':{'verifierId':'root-independent-catalog-check-v1','independentOfShopperAgent':True,'catalogSha256':catalog['sha256'],**checks},
  'model':{'status':'complete' if arm=='local-model' else 'not-enabled','readiness':readiness},'fallback':{'used':False,'kind':'none','reason':None}}
 p=d/'session.json'
 p.write_text(json.dumps(out,indent=2)+'\n')
 print(p)
