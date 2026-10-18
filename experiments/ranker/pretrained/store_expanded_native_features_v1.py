"""Root-only explicit base-Qwen extraction:105 facts +35 catalog vectors.

Default prepare/check is CPU tokenizer only. No adapters, targets, optimizer or
serving process manipulation; shared lock retained through927 and cleanup.
"""
import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from . import store_expanded_parallel_copy_v1 as copy
from . import store_function_adaptation as function
VERSION='expanded-public-native-features.v1'
MAIN=Path('/Users/lukas/Documents/Projects/dubhacks-2026')
LOCK=MAIN/'data/ranker/pretrained/native-owner.lock'

def catalog_product(product):
 return {k:product[k] for k in ('name','category','itemPrice','searchTerms')}|{'colorVariants':[{'colorName':v['colorName']} for v in product['variants']]}
def source_pins():
 names=('store_expanded_native_features_v1.py','store_expanded_parallel_copy_v1.py','store_function_adaptation.py','store_parallel_text_data_v1.py','store_parallel_text_head_v1.py','store_copy_data_v2.py','store_catalog_v1.py','backbone.py','semantic_copy_training.py','semantic_copy_feature_fit.py','store_copy_fit_v2.py','catalog_fusion/generalization_v2.py')
 root=Path(__file__).resolve().parent
 return {str(root/n):copy.sha(root/n) for n in names}
def check_pins(pins):
 for p,h in pins.items():
  if copy.sha(p)!=h:raise ValueError('Native extraction immutable input/source changed: '+p)
def prepare(packet_dir,out):
 packet_dir=Path(packet_dir).resolve();out=Path(out).resolve()
 if out.exists():raise FileExistsError('Unused extraction prepare directory required')
 prepared=json.loads((packet_dir/'PREPARED.json').read_text());public=json.loads((packet_dir/'public-records.json').read_text());copy.assert_sources(prepared['sourceSha256'])
 if prepared['productCount']!=35 or prepared['bindingCount']!=105 or copy.sha(prepared['catalogFile'])!=prepared['catalogSha256'] or copy.sha(packet_dir/'public-records.json')!=prepared['publicRecordsSha256']:raise ValueError('Final frozen35/105 public packet required')
 catalog=json.loads(Path(prepared['catalogFile']).read_text());rows=[]
 for record in public['featureRecords']:rows.append({'role':'fact','member':record['featureKey'],'text':record['text'],'textSha256':record['textSha256']})
 for p in catalog['products']:
  text=function.product_text(catalog_product(p));rows.append({'role':'catalog','member':p['productId'],'text':text,'textSha256':hashlib.sha256(text.encode()).hexdigest()})
 from transformers import AutoTokenizer
 from .semantic_copy_feature_fit import model_input_receipt
 previous=Path.cwd()
 try:
  os.chdir(MAIN);model_files=model_input_receipt()
 finally:os.chdir(previous)
 tokenizer=AutoTokenizer.from_pretrained(model_files['modelPath'],local_files_only=True,trust_remote_code=False)
 for row in rows:
  ids=tokenizer.encode(row['text'],add_special_tokens=False)
  if not 0<len(ids)<=512:raise ValueError('Exact complete public text exceeds512token cap')
  row.update(tokenIds=ids,tokens=len(ids),tokenIdsSha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())
 pins={**source_pins(),str(packet_dir/'PREPARED.json'):copy.sha(packet_dir/'PREPARED.json'),str(packet_dir/'public-records.json'):copy.sha(packet_dir/'public-records.json'),prepared['catalogFile']:prepared['catalogSha256']}
 plan={'version':VERSION,'assetRoot':str(MAIN),'packetDir':str(packet_dir),'publicCatalogSha256':prepared['catalogSha256'],'publicRecords':public,'modelFiles':model_files,'sourceAndInputSha256':pins,'rows':rows,'counts':{'fact':105,'catalog':35,'nativeForwards':140},'config':{'tokenCap':512,'memoryGiB':7,'maxNativeSeconds':600,'B':1},'adaptersInstalled':False,'teacherOrPrivateLabelsRead':False}
 out.mkdir(parents=True);copy.dump(out/'plan.json',plan);return plan

def extract(plan_path,expected_sha,out,*,run=False,owner=None):
 plan_path=Path(plan_path).resolve();out=Path(out).resolve()
 if copy.sha(plan_path)!=expected_sha:raise ValueError('Exact native plan SHA required')
 plan=json.loads(plan_path.read_text());check_pins(plan['sourceAndInputSha256'])
 if plan['version']!=VERSION or plan['counts']!={'fact':105,'catalog':35,'nativeForwards':140} or len(plan['rows'])!=140:raise ValueError('Exact complete140-row extraction roster required')
 if not run:return {'status':'CPU-plan-verified','nativeCalls':0,'counts':plan['counts']}
 if owner!='root':raise ValueError('Root is sole native owner; explicit --owner root required')
 if out.exists() or not out.is_relative_to(MAIN):raise ValueError('Unique absolute MAIN output required')
 from .store_qwen_choice_pilot_v1 import listening
 if any(listening(port) for port in (8766,8767)):raise RuntimeError('Serving active; extraction refuses competing GPU worker')
 out.mkdir(parents=True);copy.dump(out/'plan.json',plan);(out/'rows').mkdir()
 report={'version':VERSION,'status':'starting','planSha256':expected_sha,'counts':plan['counts'],'attemptedNativeForwards':0,'completedNativeForwards':0,'nativeForwardCounter':0,'adaptersInstalled':False,'optimizerUpdates':0,'teacherOrPrivateLabelsRead':False,'frozen927HashesUnchanged':None,'shutdownCompleted':False,'roster':[{'role':r['role'],'member':r['member'],'status':'unattempted'} for r in plan['rows']]}
 bb=before=None;resources=ExitStack();start=time.perf_counter();values=[];primary=None
 def save():copy.dump(out/'report.json',report)
 save()
 try:
  from .catalog_fusion.generalization_v2 import native_owner_lock
  from .store_copy_fit_v2 import deadline
  from .semantic_copy_feature_fit import verify_model_files
  from .semantic_copy_training import parameter_hashes
  from .backbone import FrozenBackbone
  resources.enter_context(native_owner_lock(LOCK,'root'))
  with deadline(600,'expanded140 base-frozen native extraction'):
   if any(listening(port) for port in (8766,8767)):raise RuntimeError('Serving became active')
   previous=Path.cwd()
   try:os.chdir(MAIN);report['modelVerification']=verify_model_files(plan['modelFiles'])
   finally:os.chdir(previous)
   import mlx.core as mx
   from mlx_lm.generate import generation_stream
   resources.enter_context(mx.stream(generation_stream))
   bb=FrozenBackbone(plan['modelFiles']['modelPath'],plan['modelFiles']['revision'],max_length=512,memory_gib=7)
   bb.model.freeze();bb.model.eval();before=parameter_hashes(bb.model.language_model)
   if len(before)!=927:raise ValueError('Exactly927 pristine language tensors required')
   report['beforeParameterSha256']=before;save()
   for i,(row,receipt) in enumerate(zip(plan['rows'],report['roster'],strict=True)):
    check_pins(plan['sourceAndInputSha256']);receipt['status']='running';report['activeRow']=i;save()
    if bb.tokenizer.encode(row['text'],add_special_tokens=False)!=row['tokenIds']:raise ValueError('Actual whole input tokenizer differs')
    report['attemptedNativeForwards']+=1;save();vector,timing=bb.encode([row['text']]);report['completedNativeForwards']+=1;report['nativeForwardCounter']=bb.forward_calls
    if timing['batchRows']!=1 or timing['backboneForwardCalls']!=1 or timing['truncatedRows']!=0 or timing['paddedTokensPerRow']!=row['tokens']:raise ValueError('Exact complete freshB1 feature required')
    vector=vector.detach().cpu().float().clone();copy.model.validate_vectors(vector,1)
    memory=bb.runtime_memory()
    if max(memory['mlxPeakBytes'],memory['mlxActiveBytes'])>7*2**30:raise MemoryError('Native peak/active7GiB guard exceeded')
    torch.save(vector,out/'rows'/f'{i:03d}.pt');row_receipt={**row,'timing':timing,'featureTensorSha256':copy.model.tensor_sha(vector),'tensorFileSha256':copy.sha(out/'rows'/f'{i:03d}.pt'),'memory':memory,'adaptersInstalled':False};copy.dump(out/'rows'/f'{i:03d}.json',row_receipt);values.append(vector);receipt.update(status='completed',receiptSha256=copy.sha(out/'rows'/f'{i:03d}.json'));save()
    print(json.dumps({'event':'expanded-native-feature','completed':i+1,'total':140}),flush=True)
   if bb.forward_calls!=140:raise ValueError('Actual native forward count differs140')
   report['status']='completed-features'
 except BaseException as error:
  primary=error;report.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
  for row in report['roster']:
   if row['status']=='running':row.update(status='failed',error=report['error'])
 finally:
  try:
   if bb is not None:
    report['nativeForwardCounter']=bb.forward_calls
    try:
     from .semantic_copy_training import parameter_hashes
     after=parameter_hashes(bb.model.language_model);report['afterParameterSha256']=after;report['frozen927HashesUnchanged']=before is not None and before==after and len(after)==927;report['memory']=bb.runtime_memory()
     if not report['frozen927HashesUnchanged']:report['status']='failed-frozen-integrity'
    except BaseException as error:report.update(status='failed-integrity-check',integrityError=str(error))
    finally:
     mx=bb.mx;bb=None;gc.collect()
     try:mx.clear_cache();mx.synchronize();report['shutdownCompleted']=True
     except BaseException as error:report.update(status='failed-shutdown',shutdownError=str(error))
   try:check_pins(plan['sourceAndInputSha256']);report['sourcePinsUnchanged']=True
   except BaseException as error:report.update(status='failed-source-change',sourceError=str(error))
  finally:
   try:resources.close()
   except BaseException as error:report.update(status='failed-resource-shutdown',resourceError=str(error))
   report['elapsedSeconds']=time.perf_counter()-start;save()
 if primary is not None:raise primary
 if report['status']!='completed-features' or report['frozen927HashesUnchanged'] is not True or report['shutdownCompleted'] is not True or len(values)!=140:raise RuntimeError('Actual expanded extraction lifecycle incomplete')
 # Only after frozen hashes and shutdown may successful native provenance exist.
 prov={'evidenceSource':'native-frozen-qwen','modelRevision':plan['modelFiles']['revision'],'copyAdapterEnabled':False,'rankAdapterEnabled':False,'nativeForwards':105,'truncatedRows':0,'catalogSha256':plan['publicCatalogSha256'],'frozenTensorCount':927,'frozen927HashesUnchanged':True,'shutdownVerified':True}
 cats=torch.cat(values[105:]);catids=[r['member'] for r in plan['rows'][105:]];catsha=[r['textSha256'] for r in plan['rows'][105:]]
 report['copyFeatureProvenance']=prov;report['catalogFeatureProvenance']={**prov,'nativeForwards':35,'productIds':catids,'textSha256':catsha,'catalogTensorSha256':copy.model.tensor_sha(cats)};save()
 nativeprov={**prov,'nativeReceiptFile':str(out/'report.json'),'nativeReceiptSha256':copy.sha(out/'report.json')}
 bundle=copy.make_expanded_feature_bundle(plan['publicRecords']['featureRecords'],torch.cat(values[:105]),nativeprov,catalog_sha=plan['publicCatalogSha256']);bundle.update(catalogVectors=cats,catalogTensorSha256=copy.model.tensor_sha(cats),catalogProductIds=catids,catalogTextSha256=catsha,catalogNativeForwards=35)
 copy.validate_bundle(bundle,plan['publicRecords']);torch.save(bundle,out/'features.pt');copy.dump(out/'FEATURES_FROZEN.json',{'version':VERSION,'reportSha256':copy.sha(out/'report.json'),'featureBundleSha256':copy.sha(out/'features.pt'),'factVectors105':True,'catalogVectors35':True,'teacherOrPrivateLabelsRead':False,'all927Unchanged':True,'shutdownVerified':True})
 return {'status':'completed','featureBundleFile':str(out/'features.pt'),'featureBundleSha256':copy.sha(out/'features.pt'),'nativeReceiptSha256':copy.sha(out/'report.json')}
def main(argv=None):
 p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='command',required=True);a=sub.add_parser('prepare');a.add_argument('--packet',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a=sub.add_parser('extract');a.add_argument('--plan',type=Path,required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--run',action='store_true');a.add_argument('--owner');args=p.parse_args(argv)
 result=prepare(args.packet,args.out) if args.command=='prepare' else extract(args.plan,args.plan_sha256,args.out,run=args.run,owner=args.owner);print(json.dumps({'status':'prepared','rows':len(result['rows'])} if 'rows' in result else result))
if __name__=='__main__':main()
