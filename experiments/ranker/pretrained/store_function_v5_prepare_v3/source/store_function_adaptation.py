"""Actual16-store public-function adaptation. CPU prep; root-only native CLI.

All16 catalog identities are TRAIN-seen. Evaluation is independent request
wording, never an unseen-catalog or sealed human-quality claim. No copy/images.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import torch
from torch import nn
from torch.nn import functional as F

from .store_catalog_v1 import canonical, load_snapshot
from .store_model_bridge import prepare_current_catalog
from .catalog_fusion.inputs import digest, tensor_digest
from .catalog_fusion.preflight import file_digest, DEFAULT_MODEL
from .catalog_fusion.runtime import FeatureStore, NativeEncoder, assert_model_files, verify_model, write_json

ROOT = Path(__file__).resolve().parent
VERSION = 'actual-store-function-adaptation.v1'
DATA = ROOT / 'store_function_v3'
EVAL = ROOT / 'store_function_eval_v1.json'
CONFIG = {'seed':728, 'steps':80, 'learningRate':.03, 'weightDecay':.001,
          'epsilon':1e-8, 'threads':1, 'memoryGiB':6, 'maxTokens':512,
          'nativeForwards':64, 'nativeSeconds':180, 'dimensions':4096,
          'classes':16, 'parameters':131104}
VOCAB = ('mug','daypack','field-notes','rain-shell','bottle','tote','throw','table-light',
         'lunch-box','headphones','sneakers','crewneck','duffel','sunglasses','organizer-pouch','towel')
TRAIN_PHRASES = (
 ('a mug for taking a drink', 'a drinking mug for my morning routine'),
 ('an everyday daypack', 'a daypack to carry my daily things'),
 ('field notes for keeping my thoughts', 'pocket field notes for a good thought'),
 ('a waterproof hooded shell', 'a hooded rain jacket'),
 ('a bottle for water', 'a steel bottle for hydration'),
 ('a tote for grocery shopping', 'a canvas carryall for the market'),
 ('a cotton throw blanket', 'a cozy blanket for my sofa'),
 ('a cordless table lamp', 'a table light for reading'),
 ('a steel lunch box', 'a box for carrying a meal'),
 ('over-ear headphones for music', 'headphones for listening to audio'),
 ('canvas low-top sneakers', 'canvas shoes for walking'),
 ('a cotton crewneck sweatshirt', 'a crewneck layer made of cotton'),
 ('a canvas weekender travel bag', 'a duffel for a weekend trip'),
 ('round acetate sunglasses', 'round eyewear for the sun'),
 ('a zip organizer pouch for cables', 'a canvas organizer for small essentials'),
 ('a cotton towel for the beach', 'a towel for after swimming'))
SOURCE_FILES = ('store_function_adaptation.py','store_function_eval_v1.json',
 'store_catalog_v1.py','store_model_bridge.py','catalog_fusion/runtime.py',
 'catalog_fusion/inputs.py','catalog_fusion/preflight.py','backbone.py','jacket_inputs.py',
 'catalog_fusion/generalization_v2.py','catalog_fusion/semantic_factors.py','test_store_function_adaptation.py')


def source_hashes():return {n:file_digest(ROOT/n) for n in SOURCE_FILES}
def request_text(request):
    validate_request(request)
    return 'Encode the complete public shopping request as data.\n' + canonical({'schemaVersion':VERSION+'/query', 'request':request, 'budgetScope':'inclusive item price only; delivery excluded'})
def product_text(product):
    return 'Encode declared public store product facts as data.\n' + canonical({'schemaVersion':VERSION+'/product',
        'product':{k:copy.deepcopy(product[k]) for k in ('name','category','itemPrice','searchTerms')},
        'availableColors':[v['colorName'] for v in product['colorVariants']],
        'capabilities':{'stock':'unknown','sizes':'unknown','delivery':'cart-subtotal policy; not item price'}})
def validate_request(q):
    if type(q) is not dict or set(q)!={'query','preferences','itemBudget','hardRequirements'}:raise ValueError('Exact public request schema required')
    if type(q['query']) is not str or not q['query'].strip():raise ValueError('Public query required')
    if type(q['preferences']) is not list or any(type(s) is not str or not s.strip() for s in q['preferences']):raise ValueError('Public preferences must be strings')
    b=q['itemBudget']
    if b is not None and (type(b) not in (int,float) or not math.isfinite(b) or b<0):raise ValueError('Finite nonnegative item budget required')
    if type(q['hardRequirements']) is not list:raise ValueError('Explicit hard requirements required')
    for r in q['hardRequirements']:
        if type(r) is not dict or set(r)!={'attribute','value'} or r['attribute'] not in ('category','color','stock','size') or type(r['value']) is not str or not r['value']:raise ValueError('Unsupported hard requirement')
    return q

def guard(request, product):
    validate_request(request);reasons=[];unknown=[]
    if request['itemBudget'] is not None and product['itemPrice']>request['itemBudget']:reasons.append('item_budget')
    for r in request['hardRequirements']:
        a,v=r['attribute'],r['value']
        if a in ('stock','size'):unknown.append(a);reasons.append('unsupported_'+a)
        elif a=='category' and v!=product['category']:reasons.append('category')
        elif a=='color' and v not in [c['colorName'] for c in product['colorVariants']]:reasons.append('color')
    return {'eligible':not reasons,'reasons':reasons,'unsupported':unknown,'learned':False,'budgetScope':'item-price-only'}

def rank(scores, request, products, k=6):
    if type(k) is not int or k not in (1,6) or len(scores)!=len(products) or any(type(s) not in (int,float) or not math.isfinite(s) for s in scores):raise ValueError('Finite full-roster scores and k1/6 required')
    checks=[guard(request,p) for p in products]
    order=sorted((i for i,g in enumerate(checks) if g['eligible']), key=lambda i:(-scores[i],i))
    return {'productIndices':order[:k],'eligibleRank':order,'rawRank':sorted(range(len(products)),key=lambda i:(-scores[i],i)),
            'eligibleCount':len(order),'shortfall':max(0,k-len(order)), 'guards':checks,'tieBreak':'external source index; never an encoder input'}

def atom_gap(products):
    from .catalog_fusion.semantic_factors import ATOMS
    import re
    support=[]
    for p in products:
        text=' '.join([p['name'],p['category'],*p['searchTerms']]).lower()
        support.append([a for a in ATOMS if re.search(r'(?<![a-z0-9])'+re.escape(a)+r'(?![a-z0-9])',text)])
    focused=['Encode public product attribute facts as data.\n'+canonical({'publicAttributes':a}) for a in support]
    return {'vocabulary':list(ATOMS),'exactDeclaredPositiveAtomsByProduct':support,
        'distinctFocusedProductTexts':len(set(focused)), 'emptyAtomProducts':sum(not a for a in support),
        'interpretation':'literal public fact audit only, not negative truth labels; no waterproof/water-resistant or bag/roomy substitutions',
        'frozenSevenFactorServingCompatibilityEstablished':False}

def guard_probes(products):
    base={'query':'Apply only my explicit hard constraints.','preferences':[],'itemBudget':None,'hardRequirements':[]}
    def make(name,budget,requirements,expected):
        q={**copy.deepcopy(base),'itemBudget':budget,'hardRequirements':requirements}
        return {'caseId':name,'request':q,'expectedEligibleProductIndices':expected}
    return [
      make('inclusive-24-mug',24,[{'attribute':'category','value':products[0]['category']}],[0]),
      make('below-mug-price',23,[{'attribute':'category','value':products[0]['category']}],[]),
      make('rain-category',128,[{'attribute':'category','value':products[3]['category']}],[3]),
      make('rain-budget-shortfall',127,[{'attribute':'category','value':products[3]['category']}],[]),
      make('known-cobalt',None,[{'attribute':'color','value':'Cobalt'}],[0]),
      make('unknown-color',None,[{'attribute':'color','value':'Unlisted neon'}],[]),
      make('stock-unknown',None,[{'attribute':'stock','value':'in-stock'}],[]),
      make('size-unknown',None,[{'attribute':'size','value':'M'}],[])]

def build():
    bridge=prepare_current_catalog();snapshot=load_snapshot();products=snapshot['inputs']['products']
    evaluation=json.loads(EVAL.read_text())
    if evaluation['version']!=VERSION or len(evaluation['cases'])!=16:raise ValueError('Independent16-case evaluation packet required')
    cases=[];labels=[]
    for i,phrases in enumerate(TRAIN_PHRASES):
        for j,phrase in enumerate(phrases):
            rid=f'train-{i:02d}-{j}'
            cases.append({'caseId':rid,'split':'train','request':{'query':'Choose a product for my stated use.','preferences':[phrase],'itemBudget':None,'hardRequirements':[]}})
            labels.append({'caseId':rid,'targetProductIndex':i,'publicSupport':{k:products[i][k] for k in ('name','category','searchTerms')}})
    for c in evaluation['cases']:
        validate_request(c['request'])
        if c['split']!='evaluation' or type(c['targetProductIndex']) is not int or not 0<=c['targetProductIndex']<16:raise ValueError('Invalid audited evaluation label')
        cases.append({k:c[k] for k in ('caseId','split','request')})
    if len({c['caseId'] for c in cases})!=48 or len({request_text(c['request']) for c in cases})!=48:raise ValueError('Request wording/identities must be disjoint')
    pairs=evaluation['pairs']
    if len(pairs)!=8 or sorted(x for p in pairs for x in p)!=sorted(c['caseId'] for c in cases if c['split']=='evaluation'):raise ValueError('All16 eval cases must have8 disjoint counterfactual pairs')
    byid={c['caseId']:c for c in cases}
    for a,b in pairs:
        qa,qb=byid[a]['request'],byid[b]['request']
        if qa['query']!=qb['query'] or qa['itemBudget']!=qb['itemBudget'] or qa['hardRequirements']!=qb['hardRequirements'] or qa['preferences']==qb['preferences']:raise ValueError('Preference pair must hold task/guards fixed')
    return products,cases,labels,evaluation,bridge

def freeze(out=DATA):
    products,cases,labels,evaluation,bridge=build();out=Path(out);out.mkdir(exist_ok=False,parents=True)
    files={'catalog.json':products,'public-cases.json':cases,'train-labels.json':labels,
      'evaluation-labels.json':{'cases':[{k:c[k] for k in ('caseId','targetProductIndex','publicSupport')} for c in evaluation['cases']], 'pairs':evaluation['pairs'],'audit':evaluation['audit']},
      'guard-probes.json':guard_probes(products),
      'external-actions.json':bridge['rows']}
    for name,value in files.items():write_json(out/name,value)
    receipt={'version':VERSION,'sourceHashes':source_hashes(),'files':{n:file_digest(out/n) for n in files},
      'counts':{'products':16,'trainQueries':32,'evaluationQueries':16,'nativeForwards':64,'evaluationPairs':8},
      'schedule':CONFIG,'classVocabulary':list(VOCAB),'atomSupportGap':atom_gap(products),
      'bridge':{k:bridge[k] for k in ('modelInputSha256','bridgeSha256','currentSourcesMatchPinnedInputs','sourceReceipts')},
      'limitations':['All16 catalog identities and all16 public functions TRAIN-seen; request paraphrase adaptation only.',
       'Hand-authored synthetic public preference cases; independent reviewer supplied evaluation wording before features, not sealed human tasks.',
       'Product labels encode public function only; no image attractiveness, latent persona, conversion or hidden preference labels.',
       'Stock/size unknown; item price does not establish delivery price; unsupported hard requirements abstain.',
       'No image decision, copy generation, learned grouping, complete-page or UI evidence.'], 'nativeExecuted':False,'trainingExecuted':False}
    receipt['limitations'].extend(['Price and product wording are class-correlated; no nuisance decorrelation claim.',
       'Closed16-function classifier; arbitrary unmatched intent has no validated abstention threshold.',
       'Evaluation request timer includes fresh native encode, immutable query receipt, verified16 catalog reads, learned scoring, guards and serialized top6 payload; no copy/image/browser latency.'])
    write_json(out/'FREEZE.json',receipt);return receipt

def read_public(data):
    data=Path(data);z=json.loads((data/'FREEZE.json').read_text())
    if z['version']!=VERSION or z['sourceHashes']!=source_hashes():raise ValueError('Source freeze changed')
    for n,s in z['files'].items():
        if file_digest(data/n)!=s:raise ValueError('Frozen artifact changed: '+n)
    prepare_current_catalog()
    return z,json.loads((data/'catalog.json').read_text()),json.loads((data/'public-cases.json').read_text())

class FunctionHead(nn.Module):
    def __init__(self,means):
        super().__init__();g=torch.Generator().manual_seed(CONFIG['seed'])
        for role in ('query','catalog'):
            self.register_buffer(role+'_mean',means[role].clone())
            setattr(self,role+'_weight',nn.Parameter(torch.randn(16,4096,generator=g)*.05))
            setattr(self,role+'_bias',nn.Parameter(torch.zeros(16)))
    def logits(self,role,x):
        x=x-getattr(self,role+'_mean');x=x/x.norm(dim=-1,keepdim=True).clamp_min(CONFIG['epsilon'])
        return x@getattr(self,role+'_weight').T+getattr(self,role+'_bias')
    def probabilities(self,role,x):return self.logits(role,x).softmax(-1)

def fit_inputs(store,rows,cases,labels):
    train=[c for c in cases if c['split']=='train'];lookup={(r['role'],r['caseId']):r for r in rows}
    q=torch.cat([store.load(lookup['request',c['caseId']])[0] for c in train])
    c=torch.cat([store.load(lookup['catalog',str(i)])[0] for i in range(16)])
    targets={x['caseId']:x['targetProductIndex'] for x in labels}
    t=torch.tensor([targets[x['caseId']] for x in train],dtype=torch.long)
    means={'query':q.double().mean(0).float(),'catalog':c.double().mean(0).float()}
    metadata={'queryMembers':[x['caseId'] for x in train],'catalogMembers':list(range(16)),
      'meanSha256':{k:tensor_digest(v) for k,v in means.items()},'vocabulary':list(VOCAB), 'config':CONFIG}
    return q,c,t,means,metadata

def save_head(path,model,metadata):
    with Path(path).open('xb') as h:torch.save({'version':VERSION,'metadata':metadata,'state':model.state_dict()},h)
    return file_digest(path)
def load_head(path,sha,metadata):
    if file_digest(path)!=sha:raise ValueError('Head checksum changed')
    z=torch.load(path,map_location='cpu',weights_only=True)
    if z['version']!=VERSION or z['metadata']!=metadata:raise ValueError('Head contract changed')
    means={r:z['state'][r+'_mean'] for r in ('query','catalog')}
    if any(tensor_digest(v)!=metadata['meanSha256'][r] for r,v in means.items()):raise ValueError('TRAIN means changed')
    model=FunctionHead(means);model.load_state_dict(z['state'],strict=True)
    if any(not torch.isfinite(x).all() for x in model.state_dict().values()):raise ValueError('Nonfinite head')
    return model.eval()

def prepare(data,model,prior):
    from tokenizers import Tokenizer
    frozen,products,cases=read_public(data);model=Path(model).resolve();old=json.loads((Path(prior)/'preflight.json').read_text())
    from .catalog_fusion.preflight import REVISION
    if model.name!=REVISION or str(model)!=old['modelPath']:raise ValueError('Pinned model path required')
    for name,key in (('tokenizer.json','tokenizerSha256'),('config.json','configSha256'),('chat_template.jinja','chatTemplateSha256')):
        if file_digest(model/name)!=old[key]:raise ValueError('Pinned tokenizer/config/template changed')
    provenance_preflight={k:old[k] for k in ('quantization','tokenizerSha256','configSha256','chatTemplateSha256')}
    provenance_preflight['modelPath']=str(model)
    tokenizer=Tokenizer.from_file(str(model/'tokenizer.json'));rows=[]
    for i,p in enumerate(products):rows.append({'role':'catalog','member':[VERSION,'catalog',i],'caseId':str(i),'split':'train','text':product_text(p)})
    for c in cases:rows.append({'role':'request','member':[VERSION,c['caseId']],'caseId':c['caseId'],'split':c['split'],'text':request_text(c['request'])})
    for r in rows:
        ids=tokenizer.encode(r['text'],add_special_tokens=False).ids
        if not 0<len(ids)<=512:raise ValueError('Whole input exceeds512; no truncation')
        r.update(textSha256=digest(r['text']),tokenIds=ids,tokenSha256=digest(canonical(ids)),tokens=len(ids),batchShape=[1,len(ids)])
    if len(rows)!=64:raise ValueError('Exactly64 native inputs required')
    packet={'version':VERSION,'sourceHashes':source_hashes(),'dataFreezeSha256':file_digest(Path(data)/'FREEZE.json'),
      'model':provenance_preflight,'priorPacketSha256':file_digest(Path(prior)/'preflight.json'),'config':CONFIG,'records':rows,
      'tokenRange':[min(r['tokens'] for r in rows),max(r['tokens'] for r in rows)]}
    return packet,frozen,products,cases,rows

def predict(models,prior_scores,q,c,cases,products):
    """No labels accepted. All four model/control arms are frozen before scoring."""
    results={};qids=[x['caseId'] for x in cases];query={x['caseId']:x for x in cases}
    with torch.no_grad():
        for arm in ('learned','prior','random','permuted-query'):
            cases_out={};model=models['random' if arm=='random' else 'learned'];catp=model.probabilities('catalog',c)
            for i,rid in enumerate(qids):
                offset=2 if len(qids)==32 else 1
                source=qids[(i+offset)%len(qids)] if arm=='permuted-query' else rid;begin=time.perf_counter()
                qp=model.probabilities('query',q[qids.index(source):qids.index(source)+1])[0]
                scores=prior_scores.tolist() if arm=='prior' else (catp@qp).tolist()
                head_ms=(time.perf_counter()-begin)*1000;begin=time.perf_counter()
                top=rank(scores,query[rid]['request'],products,6);best=top['productIndices'][0] if top['productIndices'] else None
                cases_out[rid]={'querySource':None if arm=='prior' else source,'scores':scores,
                  'queryProbabilities':None if arm=='prior' else qp.tolist(), 'catalogProbabilities':None if arm=='prior' else catp.tolist(),
                  'topSix':top,'chosenProductIndex':best,'headScoreMs':head_ms,'guardsAndTopOneTopSixMs':(time.perf_counter()-begin)*1000}
            results[arm]={'fixedCases':len(cases),'cases':cases_out}
    return results

def label_predictions(predictions,targets,pairs,products,cases):
    """Postfreeze evaluator; oracle is a nonlearned public-label ceiling only."""
    results=copy.deepcopy(predictions);results['public-oracle']={'cases':{},'fixedCases':len(cases),'learned':False,'serving':False}
    for case in cases:
        rid=case['caseId'];target=targets[rid];scores=[float(i==target) for i in range(16)];top=rank(scores,case['request'],products,6)
        results['public-oracle']['cases'][rid]={'scores':scores,'topSix':top,'chosenProductIndex':top['productIndices'][0] if top['productIndices'] else None}
    for arm,result in results.items():
        for rid,row in result['cases'].items():
            row.update(targetProductIndex=targets[rid],topOneCorrect=row['chosenProductIndex']==targets[rid],targetInTopSix=targets[rid] in row['topSix']['productIndices'])
        validpairs=[p for p in pairs if all(x in result['cases'] for x in p)]
        result.update(topOneCorrect=sum(x['topOneCorrect'] for x in result['cases'].values()),
          targetInTopSix=sum(x['targetInTopSix'] for x in result['cases'].values()),fixedPairs=len(validpairs),
          pairsBothCorrect=sum(all(result['cases'][x]['topOneCorrect'] for x in p) for p in validpairs),
          pairChoiceChanges=sum(len({result['cases'][x]['chosenProductIndex'] for x in p})==2 for p in validpairs))
        if arm in ('learned','random','permuted-query'):
            result['queryFunctionClassCorrect']=sum(max(range(16),key=lambda i:r['queryProbabilities'][i])==targets[rid] for rid,r in result['cases'].items())
            cp=next(iter(result['cases'].values()))['catalogProbabilities']
            result['catalogFunctionClassCorrect']=sum(max(range(16),key=lambda j:cp[i][j])==i for i in range(16))
    return results

def check_elapsed(started):
    if time.perf_counter()-started>CONFIG['nativeSeconds']:raise TimeoutError('Fixed native lifecycle budget exceeded')

def write_failure(out,error,phase,forwards,observations,endpoints,started):
    write_json(Path(out)/'FAILED.json',{'status':'failed','phase':phase,'errorType':type(error).__name__,'error':str(error),
      'completedNativeForwards':forwards,'fixedNativeForwards':64,'serializedFeatures':len(observations),
      'completedEvaluationEndpoints':len(endpoints),'fixedEvaluationCases':16,'fixedTrainingCases':32,
      'elapsedSeconds':time.perf_counter()-started,'promotable':False})

def run(out,data,packet_path,model,prior,owner):
    """Keep a failure receipt for every newly created run, including postprocessing."""
    out=Path(out);was_present=out.exists();started=time.perf_counter()
    try:return _run(out,data,packet_path,model,prior,owner)
    except BaseException as error:
        if not was_present and out.exists() and not (out/'FAILED.json').exists():
            log=out/'native-observations.jsonl'
            observations=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            path=out/'FRESH_REQUEST_ENDPOINTS.json'
            endpoints=json.loads(path.read_text()) if path.exists() else {}
            write_failure(out,error,'preparation-or-postprocessing',len(observations),observations,endpoints,started)
        raise

def _run(out,data,packet_path,model,prior,owner):
    from .catalog_fusion.generalization_v2 import native_owner_lock
    started=time.perf_counter();execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    prepared=prepare(data,model,prior);packet,frozen,products,cases,rows=prepared
    if json.loads(Path(packet_path).read_text())!=packet:raise ValueError('Prepared packet changed')
    out=Path(out)
    if out.exists():raise FileExistsError('Preserve previous run')
    with native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'),owner):
        out.mkdir(parents=True,exist_ok=False);write_json(out/'packet.json',packet)
        provenance=verify_model(model,packet['model']);write_json(out/'model-verification.json',provenance);assert_model_files(model,provenance)
        store=FeatureStore(out/'features',provenance);native_start=time.perf_counter();encoder=None;observations=[];endpoints={};completed_forwards=0;phase='model-load'
        try:
            encoder=NativeEncoder(model)
            def extract(row):
                nonlocal completed_forwards
                check_elapsed(native_start)
                values,timing=encoder.encode(row['text'],row);completed_forwards+=1
                check_elapsed(native_start)
                _,receipt=store.save(row,values,timing)
                observation={'member':row['member'],'timing':timing,'artifactSha256':receipt['artifactSha256']}
                observations.append(observation)
                with (out/'native-observations.jsonl').open('a') as log:log.write(json.dumps(observation)+'\n')
                print(json.dumps({'event':'actual-store-feature','completed':len(observations),'fixed':64}),flush=True)
                check_elapsed(native_start)
                return values
            phase='train-extraction'
            train_rows=[r for r in rows if r['split']=='train']
            for row in train_rows:extract(row)
            if len(observations)!=48:raise ValueError('Exactly48 TRAIN features required before fitting')
            write_json(out/'TRAIN_FEATURES_FROZEN.json',{'nativeForwards':48,'evaluationFeaturesRead':False,
              'files':{str(p.relative_to(out)):file_digest(p) for row in train_rows for p in store.paths(row['role'],row['member'])}})
            phase='fit'
            torch.set_num_threads(1);torch.manual_seed(CONFIG['seed']);fit_start=time.perf_counter()
            labels=json.loads((Path(data)/'train-labels.json').read_text());q,c,t,means,meta=fit_inputs(store,rows,cases,labels)
            meta.update(datasetFreezeSha256=packet['dataFreezeSha256'],packetSha256=file_digest(out/'packet.json'),
                representation=VERSION+'/full-public-facts',featureProvenanceSha256=digest(canonical(provenance)),
                trainTargetsSha256=digest(canonical(t.tolist())))
            learned=FunctionHead(means);random_head=FunctionHead(means)
            if sum(p.numel() for p in learned.parameters())!=CONFIG['parameters']:raise ValueError('Parameter contract changed')
            optimizer=torch.optim.AdamW(learned.parameters(),lr=CONFIG['learningRate'],weight_decay=CONFIG['weightDecay']);steps=[]
            for step in range(80):
                optimizer.zero_grad();loss=(F.cross_entropy(learned.logits('query',q),t)+F.cross_entropy(learned.logits('catalog',c),torch.arange(16)))/2
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();optimizer.step();steps.append({'step':step,'loss':loss.item()})
            prior_logits=nn.Parameter(torch.zeros(16));po=torch.optim.AdamW([prior_logits],lr=.03,weight_decay=.001);prior_steps=[]
            for step in range(80):
                po.zero_grad();loss=F.cross_entropy(prior_logits.expand(len(t),-1),t);loss.backward();po.step();prior_steps.append({'step':step,'loss':loss.item()})
            hashes={};restored={}
            for name,m in (('learned',learned),('random',random_head)):
                path=out/(name+'.pt');sha=save_head(path,m,meta);reload=load_head(path,sha,meta)
                if any(not torch.equal(v,reload.state_dict()[k]) for k,v in m.state_dict().items()):raise ValueError('Exact state reload failed')
                hashes[name]={'file':path.name,'sha256':sha,'metadata':meta};restored[name]=reload
            with (out/'prior.pt').open('xb') as h:torch.save({'scores':prior_logits.detach(),'steps':80,'metadata':meta},h)
            prior_sha=file_digest(out/'prior.pt');prior_saved=torch.load(out/'prior.pt',weights_only=True)
            if not torch.equal(prior_saved['scores'],prior_logits.detach()):raise ValueError('Prior reload changed')
            write_json(out/'HEADS_FROZEN.json',{'heads':hashes,'priorSha256':prior_sha,'packetSha256':file_digest(out/'packet.json'),
                'trainMetadata':meta,'evaluationLabelsRead':False,'trainingSteps':steps,'priorSteps':prior_steps})
            fit_seconds=time.perf_counter()-fit_start
            phase='fresh-evaluation-endpoints'
            lookup={(r['role'],r['caseId']):r for r in rows}
            for case in [x for x in cases if x['split']=='evaluation']:
                row=lookup['request',case['caseId']];begin=time.perf_counter()
                query_value=extract(row)
                verified_catalog=torch.cat([store.load(lookup['catalog',str(i)])[0] for i in range(16)])
                if not torch.equal(verified_catalog,c):raise ValueError('Warm catalog bytes changed')
                with torch.no_grad():
                    model_head=restored['learned'];scores=(model_head.probabilities('catalog',verified_catalog)@model_head.probabilities('query',query_value)[0]).tolist()
                payload={'scores':scores,'topSix':rank(scores,case['request'],products,6)}
                serialized=canonical(payload).encode()
                endpoints[case['caseId']]={'freshEncodeToSerializedRankMs':(time.perf_counter()-begin)*1000,
                  'payloadSha256':hashlib.sha256(serialized).hexdigest(),'payloadBytes':len(serialized),'payload':payload}
                check_elapsed(native_start)
            memory=encoder.memory();load_seconds=encoder.load_seconds;native_seconds=time.perf_counter()-native_start
            check_elapsed(native_start)
        except BaseException as error:
            write_failure(out,error,phase,completed_forwards,observations,endpoints,started)
            raise
        finally:
            if encoder is not None:encoder.close()
        write_json(out/'native-observations.json',observations)
        write_json(out/'FRESH_REQUEST_ENDPOINTS.json',endpoints)
        artifacts={str(p.relative_to(out)):file_digest(p) for row in rows for p in store.paths(row['role'],row['member'])}
        write_json(out/'RAW_FEATURES_FROZEN.json',{'files':artifacts,'nativeForwards':64,'evaluationFeaturesEncodedAfterHeadsFreeze':True})
        if completed_forwards!=64 or len(endpoints)!=16:raise ValueError('Fixed64 forwards/16 endpoints required')
        eval_start=time.perf_counter()
        lookup={(r['role'],r['caseId']):r for r in rows};predictions={};warm_reads={}
        for split in ('train','evaluation'):
            selected=[x for x in cases if x['split']==split];query_values=[]
            for case in selected:
                begin=time.perf_counter()
                verified_catalog=torch.cat([store.load(lookup['catalog',str(i)])[0] for i in range(16)])
                if not torch.equal(verified_catalog,c):raise ValueError('Warm catalog bytes changed')
                catalog_read_ms=(time.perf_counter()-begin)*1000;begin=time.perf_counter()
                query_values.append(store.load(lookup['request',case['caseId']])[0])
                warm_reads[case['caseId']]={'verified16WarmCatalogReadMs':catalog_read_ms,'verifiedQueryFeatureReadMs':(time.perf_counter()-begin)*1000}
            qs=torch.cat(query_values)
            predictions[split]=predict(restored,prior_saved['scores'],qs,c,selected,products)
        for rid,endpoint in endpoints.items():
            if endpoint['payload']['scores']!=predictions['evaluation']['learned']['cases'][rid]['scores']:raise ValueError('Contiguous endpoint/cached replay scores differ')
        write_json(out/'PREDICTIONS_FROZEN.json',{'predictions':predictions,'headsFreezeSha256':file_digest(out/'HEADS_FROZEN.json'),
            'rawFeatureFreezeSha256':file_digest(out/'RAW_FEATURES_FROZEN.json'),'evaluationLabelsRead':False})
        evaluation=json.loads((Path(data)/'evaluation-labels.json').read_text());alltargets={x['caseId']:x['targetProductIndex'] for x in labels+evaluation['cases']}
        results={}
        for split in ('train','evaluation'):
            selected=[x for x in cases if x['split']==split]
            results[split]=label_predictions(predictions[split],alltargets,evaluation['pairs'],products,selected)
        final_train_metadata=fit_inputs(store,rows,cases,labels)[-1]
        if any(final_train_metadata[k]!=meta[k] for k in ('queryMembers','catalogMembers','meanSha256','vocabulary','config')):raise ValueError('TRAIN means changed')
        constraint_checks=[]
        for probe in json.loads((Path(data)/'guard-probes.json').read_text()):
            check=rank([0.]*16,probe['request'],products,6)
            if check['eligibleRank']!=probe['expectedEligibleProductIndices']:raise ValueError('Hard guard probe failed')
            constraint_checks.append({**probe,'actual':check,'learned':False})
        read_public(data)
        if source_hashes()!=packet['sourceHashes'] or file_digest(out/'prior.pt')!=prior_sha or any(file_digest(out/v['file'])!=v['sha256'] for v in hashes.values()):raise ValueError('Source/head changed')
        query_ms=[x['timing']['encodeTotalMs'] for x,r in zip(observations,rows) if r['role']=='request']
        report={'status':'completed','version':VERSION,'executionCommit':execution_commit,
          'packetSha256':file_digest(out/'packet.json'),'modelInputSha256':frozen['bridge']['modelInputSha256'],
          'heads':hashes,'priorSha256':prior_sha,'predictionsFreezeSha256':file_digest(out/'PREDICTIONS_FROZEN.json'),
          'publicCasesSha256':file_digest(Path(data)/'public-cases.json'),'featureProvenance':provenance,
          'rawFeaturesFreezeSha256':file_digest(out/'RAW_FEATURES_FROZEN.json'),'nativeForwards':64,'modelLoadSeconds':load_seconds,'nativeExtractionSecondsIncludingLoad':native_seconds,
          'fitSeconds':fit_seconds,'evaluationSeconds':time.perf_counter()-eval_start,'totalWallSecondsIncludingPreparation':time.perf_counter()-started,
          'memory':memory,'queryEncodeMedianMs':statistics.median(query_ms),'warmFeatureReadTimings':warm_reads,
          'freshQueryTimings':{r['caseId']:x['timing'] for x,r in zip(observations,rows) if r['role']=='request'},
          'freshRequestEndpoints':endpoints,'freshRequestMedianMs':statistics.median(x['freshEncodeToSerializedRankMs'] for x in endpoints.values()),
          'latencyScope':'Measured contiguous fresh query encode, serialized immutable query receipt, verified16 warm catalog reads, learned head, guards and serialized top6 ranking payload. No generated copy, images or browser rendering.',
          'scope':'known16-store function ranking only; top1/top6 share one query feature; no copy/image/UI evidence','guardProbes':constraint_checks,'results':results,'limitations':frozen['limitations']}
        write_json(out/'report.json',report);return report

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('freeze','prepare','run'));p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--packet',type=Path);p.add_argument('--out',type=Path);p.add_argument('--model',type=Path,default=Path(DEFAULT_MODEL));p.add_argument('--prior',type=Path,default=Path('data/ranker/pretrained/catalog-user-fusion-v1-attempt-1'));p.add_argument('--owner')
    a=p.parse_args()
    if a.action=='freeze':z=freeze(a.data);print(canonical(z['counts']))
    elif a.action=='prepare':
        if not a.packet:raise ValueError('New --packet required')
        z=prepare(a.data,a.model,a.prior)[0];write_json(a.packet,z);print(canonical({'nativeForwards':64,'tokenRange':z['tokenRange']}))
    else:
        if not a.packet or not a.out or a.owner!='root':raise ValueError('Root --owner root/new --out/frozen --packet required')
        z=run(a.out,a.data,a.packet,a.model,a.prior,a.owner);print(canonical({'status':z['status'],'nativeForwards':z['nativeForwards']}))
if __name__=='__main__':main()
