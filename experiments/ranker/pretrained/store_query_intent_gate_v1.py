"""Learned broad/specific gate preparation. No native/fit without explicit CLI.

Only approved448 broad TRAIN public requests and96 v5 specific TRAIN features.
Inference reuses exact existing v5 query4096 feature; no persona feature or rule.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from . import store_persona_context_v1 as context
from .store_demo_engine import ranking_input
from .store_function_adaptation import request_text

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
MAIN=Path('/Users/lukas/Documents/Projects/dubhacks-2026')
V5=MAIN/'data/ranker/pretrained/actual-store-function-v5-attempt-1'
VERSION='query-intent-linear.v1'
RECIPE={'seed':20261019,'dimensions':4096,'hiddenLayers':0,'steps':250,'learningRate':.03,'weightDecay':.001,'threads':1,
    'loss':'0.5 mean broad BCE +0.5 mean specific BCE','selection':'fixed final250steps only; no evaluation/threshold tuning',
    'feature':'existing v5 final4096 L2-normalized raw query vector before learned ranking head',
    'probability':'sigmoid linear output, specific=1; no hard routing threshold',
    'runtimeText':'request_text(ranking_input(checked_context_request(public_request, actual_catalog)[0]))',
    'memoryGiB':7,'maxTokens':512,'maxNativeSeconds':300}
BROAD_INPUTS={HERE/'synthetic_history_train_contrasts_v2/profiles.train.public.json':'88e202b7225b5e3d9686a798ec418cbe9df785c8b24b09581e925016e934df1b',
    HERE/'synthetic_history_train_expansion_v3/profiles.train.public.json':'391c08b9cc80619d0c9d3723289d776f1aa70d02a86dae73d3c370f57fcc46f4'}
CATALOG=HERE/'synthetic_history_v1/catalog-binding.json'
SOURCES=[Path(__file__),Path(context.__file__),HERE/'store_demo_engine.py',HERE/'store_function_adaptation.py',HERE/'backbone.py',
    HERE/'catalog_fusion/runtime.py',HERE/'catalog_fusion/inputs.py',HERE/'catalog_fusion/generalization_v2.py',HERE/'store_copy_fit_v2.py',HERE/'semantic_copy_training.py']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def dump(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def pincheck(pins):
    if any(sha(p)!=expected for p,expected in pins.items()):raise ValueError('Frozen gate input/source/feature pin changed')

def public_products():
    if sha(CATALOG)!='bdd7e2f5f5bc41c147830b5ecd03b550794a237ad633d4e4233688ec223b8450':raise ValueError('Actual public catalog changed')
    return [{**p,'productId':p['id'],'title':p['name'],'variants':[{'variantId':v['id'],'colorName':v['colorName']} for v in p['variants']]} for p in json.loads(CATALOG.read_text())['products']]

def runtime_text(public_request,products=None):
    request=context.checked_context_request(public_request,public_products() if products is None else products)[0]
    return request_text(ranking_input(request))

def _feature(value):
    if isinstance(value,torch.Tensor) and value.device.type!='cpu':raise ValueError('CPU existing query feature required')
    value=torch.as_tensor(value,dtype=torch.float32)
    if value.ndim==1:value=value[None,:]
    if value.shape!=(1,4096) or not torch.isfinite(value).all() or not torch.allclose(value.norm(dim=-1),torch.ones(1),atol=.002,rtol=0):raise ValueError('Finite unit-L2 existing v5 query feature shape1x4096 required')
    return value

class QueryIntentHead(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():torch.manual_seed(RECIPE['seed']);self.linear=nn.Linear(4096,1)
    def forward(self,values):return self.linear(values).squeeze(-1)

def specific_probability(head,raw_feature4096):
    head.eval()
    with torch.inference_mode():value=head(_feature(raw_feature4096)).sigmoid().item()
    if not 0<=value<=1:raise FloatingPointError('Invalid intent probability')
    return float(value)

def load_intent_head(checkpoint_path,expected_sha256,*,expected_source_manifest_sha256=None):
    if sha(checkpoint_path)!=expected_sha256:raise ValueError('Pinned intent checkpoint checksum differs')
    value=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if value.get('version')!=VERSION or set(value)!={'version','state','metadata'}:raise ValueError('Unknown intent checkpoint schema')
    metadata=value['metadata']
    if metadata['recipe']!=RECIPE or metadata['dimensions']!=4096 or metadata['specificClass']!=1 or metadata['featureContract']!='exact-v5-runtime-query4096-L2':raise ValueError('Intent feature/recipe contract differs')
    if expected_source_manifest_sha256 is not None and metadata['sourceManifestSha256']!=expected_source_manifest_sha256:raise ValueError('Intent TRAIN manifest mismatch')
    head=QueryIntentHead();head.load_state_dict(value['state'],strict=True)
    if any(not torch.isfinite(x).all() for x in head.state_dict().values()):raise ValueError('Nonfinite intent checkpoint')
    head.eval().requires_grad_(False)
    return head,metadata

def class_balanced_bce(logits,labels):
    losses=F.binary_cross_entropy_with_logits(logits,labels,reduction='none')
    if set(labels.tolist())!={0.,1.}:raise ValueError('Both intent classes required')
    return .5*losses[labels==0].mean()+.5*losses[labels==1].mean()

def prepare(out):
    from tokenizers import Tokenizer
    from .catalog_fusion.runtime import FeatureStore
    out=Path(out)
    if out.exists():raise FileExistsError('Unique intent preparation directory required')
    pincheck(BROAD_INPUTS);products=public_products();broad={};source_profiles=0
    for path in BROAD_INPUTS:
        profiles=json.loads(path.read_text())['profiles'];source_profiles+=len(profiles)
        for profile in profiles:
            text=runtime_text(profile['request'],products)
            entry=broad.setdefault(text,{'sourceProfileIds':[],'query':context.checked_context_request(profile['request'],products)[0]['query']})
            entry['sourceProfileIds'].append(profile['profileId'])
    if source_profiles!=448 or len(broad)!=118:raise ValueError('Exact approved448broadTRAIN/118unique runtime texts required')
    packet=json.loads((V5/'packet.json').read_text());report=json.loads((V5/'report.json').read_text())
    if report['status'] not in ('completed','completed-not-promoted') or report['nativeForwards']!=96 or report['frozen927HashesUnchanged'] is not True or report['packetSha256']!=sha(V5/'packet.json'):raise ValueError('Verified v5 native source report required')
    originals=[r for r in packet['records'] if r['role']=='request' and r['split']=='train']
    if len(originals)!=96:raise ValueError('Only96 v5 TRAIN request rows allowed')
    tokenizer=Tokenizer.from_file(str(Path(packet['model']['modelPath'])/'tokenizer.json'))
    store=FeatureStore.open_existing(V5/'features',packet['featureProvenance']);featurepins={};rows=[];norms=[]
    for row in originals:
        request=json.loads(row['text'].split('\n',1)[1])['request']
        if len(request['preferences'])!=1:raise ValueError('Specific native row does not hold exactly one actual query in preferences')
        source={'query':request['preferences'][0],'itemBudget':request['itemBudget'],'hardRequirements':request['hardRequirements']}
        if runtime_text(source,products)!=row['text']:raise ValueError('Specific source is not byte-exact runtime ranking normalization')
        tensor,_=store.load(row);_feature(tensor);norms.append(float(tensor.norm()))
        paths=store.paths(row['role'],row['member']);featurepins.update({str(p):sha(p) for p in paths})
        rows.append({'caseId':'specific-'+row['caseId'],'intent':'specific','labelSpecific':1,'text':row['text'],'textSha256':digest(row['text']),
            'query':request['preferences'][0],'tokens':row['tokens'],'tokenIds':row['tokenIds'],'origin':'verified-v5-TRAIN-reuse',
            'originalFeatureRecord':row,'featureTensorPath':str(paths[0]),'featureReceiptPath':str(paths[1])})
    if set(broad)&{r['text'] for r in rows}:raise ValueError('Cross-class exact-text intent conflict')
    for index,(text,source) in enumerate(sorted(broad.items())):
        ids=tokenizer.encode(text,add_special_tokens=False).ids
        if not 0<len(ids)<=512:raise ValueError('Whole runtime intent text exceeds512tokens; no truncation')
        rows.append({'caseId':f'broad-{index:03d}','intent':'broad','labelSpecific':0,'text':text,'textSha256':digest(text),
            'query':source['query'],'sourceProfileIds':source['sourceProfileIds'],'tokens':len(ids),'tokenIds':ids,'tokenSha256':digest(canonical(ids)),
            'batchShape':[1,len(ids)],'role':'request','member':[VERSION,'TRAIN-broad',index],'split':'train','origin':'fresh-native-required'})
    pins={str(p):sha(p) for p in SOURCES};pins.update({str(p):v for p,v in BROAD_INPUTS.items()});pins.update({str(p):sha(p) for p in (CATALOG,V5/'packet.json',V5/'report.json')});pins.update(featurepins)
    audit={'broadSourceProfiles':448,'broadUniqueQueryStrings':len({r['query'] for r in rows if r['intent']=='broad'}),'broadUniqueRuntimeTexts':118,
        'specificTRAINRows':96,'specificUniqueRuntimeTexts':96,'crossClassExactTextConflicts':0,'specificReusedByteExactRuntimeRows':96,
        'reusedFeatureNormRange':[min(norms),max(norms)],'newNativeForwardsRequired':118,'totalTRAINVectors':214,
        'profileBackgroundOrHistoryInFeatureText':False,'futureEvaluationFilesRead':False,'privateQualityLabelsRead':False,
        'labelDefinition':'broad=authored under-specified addition request benefiting public context; specific=existing v5 explicit product/function request',
        'semanticAmbiguity':'Some broad prompts name an activity (reading/writing/travel) without uniquely selecting a product. Dataset-source classes are clean and exact-text-disjoint; semantic boundary remains graded, so output is a soft probability rather than a manual keyword rule.'}
    out.mkdir(parents=True)
    plan={'version':VERSION,'recipe':RECIPE,'rows':rows,'sha256':pins,'model':packet['model'],'featureProvenance':packet['featureProvenance'],
        'specificFeatureRoot':str(V5/'features'),'nativeOwnerLock':str(MAIN/'data/ranker/pretrained/native-owner.lock'),'audit':audit,
        'sourceClassLabelsAuthoredFromTRAINInputs':True,'nativeExecuted':False,'optimizerUpdates':0}
    dump(out/'plan.json',plan);dump(out/'PLAN_FROZEN.json',{'version':VERSION,'planSha256':sha(out/'plan.json'),'recipe':RECIPE,'nativeExecuted':False,'optimizerUpdates':0})
    return plan

def extract(plan_path,out,*,execute=False,owner=None):
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text())
    if sha(plan_path)!=json.loads((plan_path.parent/'PLAN_FROZEN.json').read_text())['planSha256'] or plan['recipe']!=RECIPE:raise ValueError('Frozen intent plan differs')
    pincheck(plan['sha256'])
    if not execute:return {'status':'cpu-preflight-only','newNativeForwards':118,'reusedVectors':96,'nativeExecuted':False}
    if owner!='root':raise ValueError('Explicit native-owner root required')
    from .store_qwen_choice_pilot_v1 import listening
    if any(listening(p) for p in (8766,8767)):raise RuntimeError('Model serving active; exclusive native intent extraction required')
    out=Path(out)
    if not out.is_absolute() or not out.is_relative_to(ROOT) or out.exists():raise ValueError('Unique absolute worktree native output required')
    out.mkdir(parents=True);dump(out/'plan.json',plan)
    from .catalog_fusion.generalization_v2 import native_owner_lock
    from .catalog_fusion.runtime import FeatureStore,verify_model,assert_model_files
    from .semantic_copy_training import parameter_hashes
    from .store_copy_fit_v2 import deadline
    from .backbone import FrozenBackbone
    report={'version':VERSION,'status':'starting','completedNativeForwards':0,'expectedNativeForwards':118,'reusedVectors':96,
        'roster':[{'caseId':r['caseId'],'status':'unattempted'} for r in plan['rows'] if r['origin']=='fresh-native-required'],
        'frozen927Unchanged':None,'shutdownCompleted':False,'privateQualityLabelsRead':False,'sourceManifestSha256':sha(plan_path)}
    bb=None;before=None;resources=ExitStack();vectors=[];started=time.perf_counter();save=lambda:dump(out/'report.json',report);save()
    try:
        resources.enter_context(native_owner_lock(Path(plan['nativeOwnerLock']),owner))
        with deadline(RECIPE['maxNativeSeconds'],'intent118broad native extraction'):
            if any(listening(p) for p in (8766,8767)):raise RuntimeError('Model listener became active')
            verify_model(plan['model']['modelPath'],plan['model']);assert_model_files(plan['model']['modelPath'],plan['featureProvenance'])
            bb=FrozenBackbone(plan['model']['modelPath'],plan['featureProvenance']['modelRevision'],max_length=512,memory_gib=7)
            before=parameter_hashes(bb.model.language_model)
            if len(before)!=927:raise ValueError('Expected927frozen tensors with no adapter')
            report['beforeParameterSha256']=before
            oldstore=FeatureStore.open_existing(plan['specificFeatureRoot'],plan['featureProvenance']);newstore=FeatureStore(out/'features',plan['featureProvenance']);native=iter(report['roster'])
            for row in plan['rows']:
                if row['origin']=='verified-v5-TRAIN-reuse':values,_=oldstore.load(row['originalFeatureRecord'])
                else:
                    receipt=next(native);receipt['status']='running';save()
                    if bb.tokenizer.encode(row['text'],add_special_tokens=False)!=row['tokenIds']:raise ValueError('Complete native runtime tokens differ')
                    values,timing=bb.encode([row['text']])
                    if timing['batchRows']!=1 or timing['truncatedRows']!=0 or timing['backboneForwardCalls']!=1 or timing['paddedTokensPerRow']!=row['tokens']:raise ValueError('Exact whole B1 native feature receipt required')
                    newstore.save(row,values,timing);receipt.update(status='completed',timing=timing);report['completedNativeForwards']+=1;save()
                _feature(values);vectors.append(values)
                if bb.mx.get_peak_memory()>bb.memory_budget:raise MemoryError('Peak exceeds fixed7GiB')
            pincheck(plan['sha256']);bundle={'version':VERSION,'caseIds':[r['caseId'] for r in plan['rows']],'features':torch.cat(vectors),'sourceManifestSha256':sha(plan_path)}
            torch.save(bundle,out/'query-features.pt');report['queryFeaturesSha256']=sha(out/'query-features.pt');report['status']='completed-native-features'
    except BaseException as error:
        for item in report['roster']:
            if item['status']=='running':item.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
    finally:
        try:
            if bb is not None:
                try:
                    after=parameter_hashes(bb.model.language_model);report['afterParameterSha256']=after
                    report['frozen927Unchanged']=before is not None and len(after)==927 and before==after;report['memory']=bb.runtime_memory()
                    if not report['frozen927Unchanged']:report['status']='failed-frozen-integrity'
                except BaseException as error:report.update(status='failed-shutdown-integrity',shutdownError=str(error))
                finally:
                    mx=bb.mx;bb=None
                    import gc
                    try:gc.collect();mx.clear_cache();mx.synchronize();report['shutdownCompleted']=True
                    except BaseException as error:report.update(status='failed-shutdown',shutdownError=str(error))
        finally:
            try:report['elapsedSeconds']=time.perf_counter()-started;save()
            finally:resources.close()
    if report['status']!='completed-native-features' or not report['frozen927Unchanged'] or not report['shutdownCompleted']:raise RuntimeError('Intent extraction integrity/shutdown failed')
    return report

def fit(feature_run,out):
    feature_run=Path(feature_run);out=Path(out)
    if out.exists():raise FileExistsError('Unique CPU intent fit output required')
    plan=json.loads((feature_run/'plan.json').read_text());report=json.loads((feature_run/'report.json').read_text())
    if plan['recipe']!=RECIPE or report['status']!='completed-native-features' or report['completedNativeForwards']!=118 or not report['frozen927Unchanged'] or not report['shutdownCompleted'] or sha(feature_run/'query-features.pt')!=report['queryFeaturesSha256']:raise ValueError('Verified118fresh+96reused native bundle required')
    pincheck(plan['sha256']);value=torch.load(feature_run/'query-features.pt',map_location='cpu',weights_only=True)
    if value['caseIds']!=[r['caseId'] for r in plan['rows']] or value['sourceManifestSha256']!=sha(feature_run/'plan.json') or value['features'].shape!=(214,4096):raise ValueError('Exact frozen214TRAIN feature join required')
    x=value['features'].float()
    for row in x:_feature(row)
    y=torch.tensor([r['labelSpecific'] for r in plan['rows']],dtype=torch.float32);old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        head=QueryIntentHead();optimizer=torch.optim.AdamW(head.parameters(),lr=.03,weight_decay=.001);losses=[]
        for step in range(250):
            optimizer.zero_grad();loss=class_balanced_bce(head(x),y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite TRAIN intent loss')
            loss.backward();optimizer.step()
            if step in (0,49,99,249):losses.append({'step':step+1,'loss':float(loss.detach())})
        metadata={'recipe':RECIPE,'dimensions':4096,'specificClass':1,'featureContract':'exact-v5-runtime-query4096-L2','sourceManifestSha256':sha(feature_run/'plan.json'),
            'featureBundleSha256':report['queryFeaturesSha256'],'sourceSha256':sha(__file__),'trainingOnly':True,'qualityEvaluationLabelsRead':False}
        out.mkdir(parents=True);torch.save({'version':VERSION,'state':head.state_dict(),'metadata':metadata},out/'head.pt')
        probabilities=head(x).sigmoid().detach().tolist()
        dump(out/'HEAD_FROZEN.json',{'version':VERSION,'headSha256':sha(out/'head.pt'),'sourceManifestSha256':metadata['sourceManifestSha256'],'optimizerUpdates':250,'qualityEvaluationLabelsRead':False})
        dump(out/'train-results.json',{'version':VERSION,'lossSnapshots':losses,'rows':[{'caseId':r['caseId'],'labelSpecific':r['labelSpecific'],'pSpecific':p} for r,p in zip(plan['rows'],probabilities,strict=True)],
            'thresholdSelection':'none;runtimeusescontinuousprobability','evaluation':'none;futurepublicpredictionsmustfreeze beforeprivatepipelinequalitylabelsread'})
    finally:torch.set_num_threads(old_threads)
    return {'status':'completed-TRAIN-only','headSha256':sha(out/'head.pt'),'sourceManifestSha256':metadata['sourceManifestSha256']}

def main(argv=None):
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--out',type=Path,required=True)
    b=sub.add_parser('extract');b.add_argument('--plan',type=Path,required=True);b.add_argument('--out',type=Path,required=True);b.add_argument('--run',action='store_true');b.add_argument('--native-owner',choices=['root'])
    c=sub.add_parser('fit');c.add_argument('--features',type=Path,required=True);c.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv)
    result=prepare(args.out) if args.command=='prepare' else extract(args.plan,args.out,execute=args.run,owner=args.native_owner) if args.command=='extract' else fit(args.features,args.out)
    print(json.dumps({'status':'cpu-TRAIN-plan-frozen','audit':result['audit']} if args.command=='prepare' else result,indent=2))
if __name__=='__main__':main()
