"""Public10 Qwen LM product-bound lexical routing pilot. Preparation is CPU-only; native run is root-only.

One fresh full-prompt body forward and one frozen LM-head operation score all16
one-token choices. No autoregressive continuation, labels, teacher lookup or fit.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import socket
import statistics
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MAIN = Path('/Users/lukas/Documents/Projects/dubhacks-2026')
MODEL = MAIN/'.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53'
MODEL_SOURCE = MAIN/'.tools/pretrained-venv/lib/python3.14/site-packages/mlx_lm/models/qwen3_5.py'
LOCK = MAIN/'data/ranker/pretrained/native-owner.lock'
VERSION = 'public-demo-qwen-lm-lexical-choice-pilot.v1'
CONFIG = {'people':10,'orders':['canonical','reversed'],'expectedForwards':20,'maxTokens':2048,
    'memoryGiB':7,'maxNativeSeconds':600,'batchRows':1,'cache':None,'enableThinking':False,
    'assistantPrefix':'Choice:','routingCodes':'16unique single tokens bound toPRODUCT acrossorders:15naturalnouns plus1exactpublicmateriallabel cotton',
    'materialLabelCaveat':'cotton labels the sweatshirt row by its exact public material word; it is not a unique product noun/function and cotton also occurs in other product facts',
    'catalogRendering':'readable factual lines; stable label to full canonical name/category/itemprice/availablecolors',
    'hidden':'raw final post-RMSNorm last prompt token, no L2 normalization',
    'scoring':'all16 routing logits from one full frozen LM-head projection; no autoregressive token appended',
    'scope':'Known public demo diagnostic and position-bias probe; no unseen quality or promotion claim'}
PUBLIC_INPUTS = {HERE/'synthetic_history_v1/demo-people.json':'4737c785313b3200798133679ee914aae638d5b03f46b052ecf77451f75f5b1d',
    HERE/'synthetic_history_v1/catalog-binding.json':'bdd7e2f5f5bc41c147830b5ecd03b550794a237ad633d4e4233688ec223b8450'}
SOURCE_FILES = [Path(__file__),HERE/'backbone.py',HERE/'semantic_copy_training.py',HERE/'store_copy_fit_v2.py',
    HERE/'catalog_fusion/generalization_v2.py',MODEL_SOURCE]
SYSTEM = ('Choose one useful product from this public catalog for this shopper. Respect the explicit query, item budget and all hard requirements. '
    'Purchase events mean the item is already owned: prefer useful unowned complements for an addition request, unless replacement or repeat purchase is explicitly requested. '
    'View and cart events show interests, not ownership. Stated color preferences are soft unless made a hard requirement. '
    'Use only the listed public product facts; do not invent functionality. Each routing label identifies the full product row that follows it. The label cotton is a factual material word identifying the cotton crewneck row, not a claim that other rows lack cotton. Answer with Choice: followed by exactly one listed routing label, without explanation.')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def dump(path,value): Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def check_pins(pins):
    if any(sha(path)!=expected for path,expected in pins.items()): raise ValueError('Frozen public/source/model-text pin changed')

PRODUCT_LABELS = {'mug':'mug','pack':'pack','notebook':'notes','jacket':'jacket','bottle':'bottle',
    'tote':'tote','blanket':'throw','lamp':'lamp','lunchbox':'box','headphones':'headphones',
    'sneakers':'sneakers','sweatshirt':'cotton','duffel':'bag','sunglasses':'sunglasses','organizer':'pouch','towel':'towel'}

def routing_tokens(tokenizer,catalog):
    import re
    if set(PRODUCT_LABELS)!={p['id'] for p in catalog}:raise ValueError('Exact16 public catalog IDs required')
    routes=[]
    for product in catalog:
        label=PRODUCT_LABELS[product['id']]
        words=re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*",product['name']+' '+product['category'])
        if label not in [w.lower() for w in words]:raise ValueError('Routing label is not an exact public name/category word')
        continuation=' '+label;ids=tokenizer.encode(continuation,add_special_tokens=False)
        if len(ids)!=1 or tokenizer.decode(ids)!=continuation:raise ValueError('Natural factual routing label is not one whole token: '+label)
        routes.append({'code':label,'continuation':continuation,'tokenId':ids[0],'productId':product['id'],
            'labelKind':'public-material-noun' if product['id']=='sweatshirt' else 'public-product/category-noun'})
    if len({r['tokenId'] for r in routes})!=16:raise ValueError('Routing labels collide')
    return routes

def public_profile(person,catalog):
    if set(person['request'])-{'query','itemBudget','hardRequirements','persona','publicHistory'}: raise ValueError('Only public request fields')
    request=deepcopy(person['request']); persona=request['persona']
    if set(persona)!={'background','colorPreferences'} or not isinstance(persona['background'],list) or not isinstance(persona['colorPreferences'],list): raise ValueError('Only public persona context')
    if not isinstance(request['query'],str) or not request['query'].strip():raise ValueError('Public query required')
    budget=request.get('itemBudget')
    if budget is not None and (type(budget) not in (int,float) or not math.isfinite(budget) or budget<=0):raise ValueError('Finite public item budget')
    if not 4<=len(request['publicHistory'])<=12: raise ValueError('Expected4-12 public demo events')
    products={p['id']:p for p in catalog};history=[]
    for event in request['publicHistory']:
        if set(event)-{'eventId','eventType','itemName','category','selectedColor','occurredAt','source','productId','variantId','external'}:raise ValueError('Unknown/private event field')
        if event['eventType'] not in ('view','cart','purchase'):raise ValueError('Unknown history event type')
        if event.get('external'):
            if 'productId' in event or 'variantId' in event:raise ValueError('External item claims a storeSKU')
        else:
            product=products.get(event['productId']);variant=next((v for v in product['variants'] if v['id']==event['variantId']),None) if product else None
            if not variant or (event['itemName'],event['category'],event['selectedColor'])!=(product['name'],product['category'],variant['colorName']):raise ValueError('Stale public product/variant binding')
        # Opaque IDs/display names are excluded; factual history is unchanged.
        history.append({k:event[k] for k in ('eventType','itemName','category','selectedColor','occurredAt','external') if k in event})
    return {'query':request['query'],'itemBudget':budget,'hardRequirements':request.get('hardRequirements',[]),
        'background':persona['background'],'colorPreferences':persona['colorPreferences'],'recentHistory':history}

def build_rows(people,catalog,tokenizer):
    if len(people)!=10 or len(catalog)!=16 or len({p['id'] for p in people})!=10 or len({p['id'] for p in catalog})!=16:raise ValueError('Exactly10public people/16catalog products')
    routes=routing_tokens(tokenizer,catalog);by_product={r['productId']:r for r in routes};rows=[]
    for person in people:
        profile=public_profile(person,catalog)
        for order in CONFIG['orders']:
            products=catalog if order=='canonical' else list(reversed(catalog))
            choices=[dict(by_product[p['id']]) for p in products]
            public_catalog=[f"{c['code']}: {p['name']} | category: {p['category']} | item price: ${p['price']:.2f} | available colors: {', '.join(v['colorName'] for v in p['variants'])}" for c,p in zip(choices,products,strict=True)]
            observations=["Query: "+profile['query'],"Item budget: "+(f"${profile['itemBudget']:.2f}" if profile['itemBudget'] is not None else 'not specified'),
                "Hard requirements: "+canonical(profile['hardRequirements']),"Public background:"]
            observations += ['- '+line for line in profile['background']]
            observations += ["Stated soft color preferences: "+', '.join(profile['colorPreferences']),"Recent public history, oldest first:"]
            for event in profile['recentHistory']:
                line=f"- {event['eventType']} | {event['itemName']} | category: {event['category']}"
                if 'selectedColor' in event:line+=' | color: '+event['selectedColor']
                line+=' | occurred: '+event['occurredAt']+' | '+('external item, no store SKU' if event.get('external') else 'actual store item')
                observations.append(line)
            user='Public catalog (routing labels are product-bound, independent of order):\n'+'\n'.join(public_catalog)+'\nPublic shopper observations:\n'+'\n'.join(observations)+'\nSelect the best listed product routing label.'
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':user}]
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)+CONFIG['assistantPrefix']
            ids=tokenizer.encode(prompt,add_special_tokens=False)
            if not 0<len(ids)<=CONFIG['maxTokens']:raise ValueError(f'Whole prompt {person["id"]}/{order} has{len(ids)}tokens; no truncation')
            for route in routes:
                if tokenizer.encode(prompt+route['continuation'],add_special_tokens=False)!=ids+[route['tokenId']]:raise ValueError('Routing code changes prompt boundary; not a true next single token')
            rows.append({'caseId':person['id']+'-'+order,'personId':person['id'],'order':order,'publicRequestSha256':digest(person['request']),
                'prompt':prompt,'promptSha256':digest(prompt),'tokenIds':ids,'tokenIdsSha256':digest(ids),'tokens':len(ids),'choices':choices,
                'publicPurchaseProductIds':[e['productId'] for e in person['request']['publicHistory'] if e['eventType']=='purchase' and not e.get('external')]})
    return rows,routes

def prepare(out,*,tokenizer=None):
    out=Path(out)
    if out.exists():raise FileExistsError('Unique pilot plan directory required')
    check_pins(PUBLIC_INPUTS)
    config=json.loads((MODEL/'config.json').read_text());quant=config.get('quantization') or config.get('quantization_config')
    if quant.get('bits')!=4 or config['model_type']!='qwen3_5':raise ValueError('Pinned localQwen3.5 four-bit model required')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    people=json.loads((HERE/'synthetic_history_v1/demo-people.json').read_text())['people']
    catalog=json.loads((HERE/'synthetic_history_v1/catalog-binding.json').read_text())['products']
    rows,routes=build_rows(people,catalog,tokenizer)
    model_text={str(MODEL/name):sha(MODEL/name) for name in ('config.json','tokenizer.json','tokenizer_config.json','vocab.json','chat_template.jinja','model.safetensors.index.json')}
    shards={name:{'bytes':(MODEL/name).stat().st_size,'mtimeNs':(MODEL/name).stat().st_mtime_ns,'inode':(MODEL/name).stat().st_ino} for name in sorted(set(json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'].values()))}
    packet={'version':VERSION,'config':CONFIG,'modelPath':str(MODEL),'modelRevision':MODEL.name,'modelSourcePath':str(MODEL_SOURCE),
        'nativeOwnerLock':str(LOCK),'nativeOwner':'root','sourceSha256':{str(p):sha(p) for p in SOURCE_FILES},
        'publicInputSha256':{str(p):v for p,v in PUBLIC_INPUTS.items()},'modelTextSha256':model_text,'modelShardStats':shards,
        'nativeExpectedLanguageTensors':927,'expectedTieWordEmbeddings':bool(config.get('text_config',{}).get('tie_word_embeddings',config.get('tie_word_embeddings',False))),
        'routes':routes,'productBoundLabels':PRODUCT_LABELS,'materialLabelProducts':['sweatshirt'],'rows':rows,'tokenRange':[min(r['tokens'] for r in rows),max(r['tokens'] for r in rows)],
        'privateGoldRead':False,'generatedAnswersInput':False,'tokenizerPreparationOnly':True,'nativeExecuted':False,
        'weightExport':'none;16 dequantized routing rows are not needed for this full-LM projection pilot',
        'runtime':{k:importlib.metadata.version(k) for k in ('transformers','tokenizers','mlx','mlx-lm')}}
    out.mkdir(parents=True);dump(out/'plan.json',packet);dump(out/'PLAN_FROZEN.json',{'version':VERSION,'planSha256':sha(out/'plan.json'),'expectedNativeForwards':20,'privateGoldRead':False,'nativeExecuted':False})
    return packet

def model_text_and_stat_pins(packet):
    for group in ('sourceSha256','publicInputSha256','modelTextSha256'):check_pins(packet[group])
    for name,stats in packet['modelShardStats'].items():
        stat=(Path(packet['modelPath'])/name).stat()
        if {'bytes':stat.st_size,'mtimeNs':stat.st_mtime_ns,'inode':stat.st_ino}!=stats:raise ValueError('Frozen native model shard identity changed')

def listening(port):
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=.5):return True
    except ConnectionRefusedError:return False
    except OSError as error:raise RuntimeError('Unknown model listener state; refuse native pilot') from error

def project_last(lm,last):
    # Mirrors installed qwen3_5.TextModel.__call__, using raw final hidden states.
    return lm.model.embed_tokens.as_linear(last) if lm.args.tie_word_embeddings else lm.lm_head(last)

def score_row(bb,row):
    mx=bb.mx;started=time.perf_counter();actual=bb.tokenizer.encode(row['prompt'],add_special_tokens=False)
    if actual!=row['tokenIds'] or not 0<len(actual)<=CONFIG['maxTokens']:raise ValueError('Fresh native whole-prompt tokens differ')
    token_ms=(time.perf_counter()-started)*1000;bb.check_memory();begin=time.perf_counter()
    inputs=mx.array([actual]);mx.eval(inputs)
    hidden=bb.model.model(inputs,cache=None)
    last=hidden[:,-1:,:]  # No normalization/pooling that would alter pretrained LM logits.
    logits=project_last(bb.model.language_model,last)[:,0,:].astype(mx.float32)
    if logits.shape[0]!=1:raise ValueError('B1 LM output required')
    route_ids=mx.array([c['tokenId'] for c in row['choices']]);selected=logits[0,route_ids]
    norm=mx.logsumexp(logits[0]);top=mx.argmax(logits[0]);finite=mx.all(mx.isfinite(logits))
    mx.eval(selected,norm,top,finite);mx.synchronize()
    if not bool(finite.item()):raise FloatingPointError('Nonfinite frozen LM logits')
    forward_ms=(time.perf_counter()-begin)*1000
    values=selected.tolist();normalizer=float(norm.item());top_id=int(top.item());route_norm=max(values)+math.log(sum(math.exp(x-max(values)) for x in values))
    scores=[{**c,'logit':value,'fullVocabularyLogProbability':value-normalizer,'routingConditionalProbability':math.exp(value-route_norm)} for c,value in zip(row['choices'],values,strict=True)]
    ranking=sorted(scores,key=lambda x:(-x['logit'],x['tokenId']));choice=ranking[0]
    bb.check_memory()
    if mx.get_peak_memory()>bb.memory_budget:raise MemoryError('Observed peak exceeded fixed7GiB')
    return {'caseId':row['caseId'],'personId':row['personId'],'order':row['order'],'tokens':len(actual),'batchRows':1,
        'bodyForwardCalls':1,'lmHeadCalls':1,'autoregressiveSteps':0,'cache':None,'truncatedRows':0,
        'lastHiddenShape':list(last.shape),'lastHiddenDtype':str(last.dtype),'logitsShape':list(logits.shape),'logitsDtype':str(logits.dtype),
        'lmProjection':'tied embed_tokens.as_linear' if bb.model.language_model.args.tie_word_embeddings else 'language_model.lm_head',
        'scores':scores,'rankedProductIds':[s['productId'] for s in ranking],'selectedProductId':choice['productId'],'selectedCode':choice['code'],
        'selectedAlreadyOwned':choice['productId'] in row['publicPurchaseProductIds'],'fullVocabularyTopTokenId':top_id,
        'fullVocabularyTopTokenText':bb.tokenizer.decode([top_id]),'fullVocabularyTopIsRoutingCode':top_id in [c['tokenId'] for c in row['choices']],
        'routingTokenProbabilityMass':math.exp(route_norm-normalizer),'tokenizationAndVerificationMs':token_ms,'fullPromptBodyPlusLmScoresMs':forward_ms,
        'completeChoiceMs':(time.perf_counter()-started)*1000,'memory':bb.runtime_memory()}

def run(plan,out,*,execute=False,owner=None):
    plan=Path(plan);packet=json.loads(plan.read_text());frozen=json.loads((plan.parent/'PLAN_FROZEN.json').read_text())
    if sha(plan)!=frozen['planSha256'] or len(packet['rows'])!=20 or packet['config']!=CONFIG:raise ValueError('Frozen20-forward plan/config differs')
    model_text_and_stat_pins(packet)
    if not execute:return {'status':'cpu-preflight-only','nativeExecuted':False,'expectedNativeForwards':20,'tokenRange':packet['tokenRange']}
    if owner!='root':raise ValueError('Only explicit native-owner root may execute')
    if any(listening(p) for p in (8766,8767)):raise RuntimeError('Model serving listener active; release before exclusive pilot')
    out=Path(out)
    if not out.is_absolute() or not out.is_relative_to(ROOT) or out.exists():raise ValueError('Unique absolute output in isolated worktree required')
    out.mkdir(parents=True);dump(out/'plan.json',packet)
    report={'version':VERSION,'status':'starting','expectedNativeForwards':20,'completedForwards':0,'privateGoldRead':False,
        'roster':[{'caseId':r['caseId'],'status':'unattempted'} for r in packet['rows']],'timings':[],'frozen927BeforeAfterUnchanged':None,'shutdownCompleted':False}
    save=lambda:dump(out/'report.json',report)
    save();bb=None;before=None;started=time.perf_counter();resources=ExitStack()
    try:
        from .catalog_fusion.generalization_v2 import native_owner_lock
        from .store_copy_fit_v2 import deadline
        from .semantic_copy_training import parameter_hashes
        from .backbone import FrozenBackbone
        resources.enter_context(native_owner_lock(Path(packet['nativeOwnerLock']),owner))
        with deadline(CONFIG['maxNativeSeconds'],'public20 frozenQwen LM choice pilot'):
            if any(listening(p) for p in (8766,8767)):raise RuntimeError('Model listener became active')
            model_text_and_stat_pins(packet)
            bb=FrozenBackbone(packet['modelPath'],packet['modelRevision'],max_length=CONFIG['maxTokens'],memory_gib=CONFIG['memoryGiB'])
            if bool(bb.model.language_model.args.tie_word_embeddings)!=packet['expectedTieWordEmbeddings']:raise ValueError('Native tied-head argument mismatch')
            before=parameter_hashes(bb.model.language_model)
            if len(before)!=927:raise ValueError('Exactly927 frozen language tensors including LM projection required')
            from mlx.utils import tree_flatten
            if tree_flatten(bb.model.trainable_parameters()) or len(tree_flatten(bb.model.language_model.parameters()))!=927:raise ValueError('Exactly927 frozen native tensors with no adapters/trainables required')
            report['beforeParameterSha256']=before;report['lmProjectionTensorNames']=[k for k in before if k.startswith('lm_head.') or k.startswith('model.embed_tokens.')]
            report['loadMs']=bb.load_ms;report['status']='running';save()
            for row,item in zip(packet['rows'],report['roster'],strict=True):
                item['status']='running';report['activeCaseId']=row['caseId'];save();model_text_and_stat_pins(packet)
                result=score_row(bb,row);report['completedForwards']+=1
                dump(out/(row['caseId']+'.json'),result);item.update(status='completed',receiptFile=row['caseId']+'.json')
                report['timings'].append({k:result[k] for k in ('caseId','tokens','completeChoiceMs','fullPromptBodyPlusLmScoresMs')});save()
            completed=[json.loads((out/item['receiptFile']).read_text()) for item in report['roster']]
            grouped={person:{r['order']:r for r in completed if r['personId']==person} for person in {r['personId'] for r in completed}}
            report['diagnosticSummary']={'fixedPeople':10,'fixedForwards':20,'selectedOwned':sum(r['selectedAlreadyOwned'] for r in completed),
                'catalogOrderAgreementPeople':sum(v['canonical']['selectedProductId']==v['reversed']['selectedProductId'] for v in grouped.values()),
                'completeChoiceMedianMs':statistics.median(r['completeChoiceMs'] for r in completed),
                'people':{person:{order:{k:r[k] for k in ('selectedProductId','selectedCode','selectedAlreadyOwned','routingTokenProbabilityMass')} for order,r in by_order.items()} for person,by_order in grouped.items()},
                'qualityGoldUsed':False,'scope':CONFIG['scope']}
            report['status']='completed20-forward-diagnostic'
    except BaseException as error:
        for item in report['roster']:
            if item['status']=='running':item.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        try:
            if bb is not None:
                try:
                    after=parameter_hashes(bb.model.language_model);report['afterParameterSha256']=after
                    report['frozen927BeforeAfterUnchanged']=before is not None and len(after)==927 and before==after
                    report['memory']=bb.runtime_memory()
                    if not report['frozen927BeforeAfterUnchanged']:report['status']='failed-frozen-integrity'
                except BaseException as error:report.update(status='failed-shutdown-integrity',shutdownError={'type':type(error).__name__,'message':str(error)})
                finally:
                    mx=bb.mx;bb=None
                    import gc
                    try:
                        gc.collect();mx.clear_cache();mx.synchronize();report['shutdownCompleted']=True
                    except BaseException as error:report.update(status='failed-shutdown',shutdownError={'type':type(error).__name__,'message':str(error)})
            try:model_text_and_stat_pins(packet)
            except BaseException as error:report.update(status='failed-source-integrity',sourceIntegrityError={'type':type(error).__name__,'message':str(error)})
        finally:
            try:
                report['elapsedSeconds']=time.perf_counter()-started;report['receiptSha256']={item['receiptFile']:sha(out/item['receiptFile']) for item in report['roster'] if item['status']=='completed'};save()
            finally:resources.close()
    if report['status']!='completed20-forward-diagnostic' or not report['frozen927BeforeAfterUnchanged'] or not report['shutdownCompleted']:raise RuntimeError('Pilot did not complete native integrity/shutdown')
    return report

def main(argv=None):
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--out',type=Path,required=True)
    b=sub.add_parser('run');b.add_argument('--plan',type=Path,required=True);b.add_argument('--out',type=Path,required=True);b.add_argument('--run',action='store_true');b.add_argument('--native-owner',choices=['root'])
    args=p.parse_args(argv)
    result=prepare(args.out) if args.command=='prepare' else run(args.plan,args.out,execute=args.run,owner=args.native_owner)
    print(json.dumps({'status':'cpu-plan-frozen','rows':len(result['rows']),'tokenRange':result['tokenRange']} if args.command=='prepare' else result,indent=2))
if __name__=='__main__':main()
