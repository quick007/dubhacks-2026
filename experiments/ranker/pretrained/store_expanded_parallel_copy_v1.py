"""Dynamic public-catalog Qwen fact-feature copy head, CPU fit and pure loader.

All bindings are supervised synthetic TRAIN. Runtime predicts fresh slots; it
never loads TRAIN labels or invokes teacher(). Old fixed48 assets are untouched.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from . import store_parallel_text_data_v1 as data
from . import store_parallel_text_head_v1 as model

VERSION='expanded-public-fact-parallel-copy.v1'
SCOPE='TRAIN-ALL expanded public catalog synthetic copy; no heldout quality claim'
CONFIG={'maxSteps':2000,'checkEvery':100,'learningRate':.004,'weightDecay':.0001,'width':128,'seed':7211,'cpuThreads':1,'stop':'all supervised fields exact including learned EOS, at fixed100-step checks; otherwise final2000'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sources():
 return {str(Path(p).resolve()):sha(p) for p in (__file__,data.__file__,model.__file__,Path(__file__).with_name('store_copy_data_v2.py'),Path(__file__).with_name('store_catalog_v1.py'))}
def state_sha(head):
 h=hashlib.sha256()
 for name,tensor in sorted(head.state_dict().items()):
  h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(json.dumps(list(tensor.shape)).encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def assert_sources(pins):
 for p,h in pins.items():
  if sha(p)!=h:raise ValueError('Expanded copy source changed: '+p)
def build_packet(catalog,catalog_sha):
 if type(catalog) is not dict or type(catalog.get('products')) is not list or not catalog['products']:raise ValueError('Public normalized catalog required')
 records=[];bindings=[];pids=set();vids=set()
 for pi,p in enumerate(catalog['products']):
  if p['productId'] in pids:raise ValueError('Duplicate public product ID')
  pids.add(p['productId'])
  if not p['variants']:raise ValueError('Nonempty declared variants required')
  for vi,v in enumerate(p['variants']):
   if v['variantId'] in vids or not v.get('images'):raise ValueError('Unique actual variant with source images required')
   vids.add(v['variantId'])
   record=data.public_record({'name':p['name'],'category':p['category'],'itemPrice':p['itemPrice'],'selectedColor':v['colorName']})
   records.append(record);bindings.append({'productIndex':pi,'variantIndex':vi,'productId':p['productId'],'variantId':v['variantId'],'featureKey':record['featureKey']})
 if len(set(r['featureKey'] for r in records))!=len(records):raise ValueError('Fact bindings must be unique')
 rows=[]
 for i,record in enumerate(records):
  for style in data.STYLES:
   for kind in data.KINDS[1:]:
    rows.append({'input':data.serving_row(kind,style,record['facts']),'featureKeys':[record['featureKey']],'targetText':data.teacher(kind,style,record['facts'])})
 # Diverse source means from distinct products, all headings supervised offline.
 if len(pids)>=3:
  for i in range(len(records)):
   group=[records[i]['featureKey']];used={bindings[i]['productIndex']}
   for j in list(range(i+1,len(records)))+list(range(i)):
    if bindings[j]['productIndex'] not in used:group.append(records[j]['featureKey']);used.add(bindings[j]['productIndex'])
    if len(group)==3:break
   for style in data.STYLES:
    for section in (0,1):rows.append({'input':data.serving_row('section',style,section_index=section),'featureKeys':group,'targetText':data.teacher('section',style,section_index=section)})
 vocabulary=[data.EOS,data.PAD]+sorted({w for row in rows for w in data.tokens(row['targetText'])})
 model.encode_labels(rows,vocabulary)
 if any(data.field_rejections(r['input']['kind'],r['targetText'],r['input'].get('source',{}).get('facts')) for r in rows):raise ValueError('Offline source teacher violates bounded factual contract')
 return {'version':VERSION,'scope':SCOPE,'catalogSha256':catalog_sha,'productCount':len(pids),'bindingCount':len(records),'featureRecords':records,'bindings':bindings,'vocabulary':vocabulary,'train':rows,'heldoutBindings':0}
def prepare(catalog_file,out,*,expected_products=35,expected_bindings=105):
 catalog_file=Path(catalog_file).resolve();out=Path(out)
 if out.exists():raise FileExistsError('Unused packet output required')
 packet=build_packet(json.loads(catalog_file.read_text()),sha(catalog_file))
 if packet['productCount']!=expected_products or packet['bindingCount']!=expected_bindings:raise ValueError('Final complete catalog count required; partial catalog not prepared')
 out.mkdir(parents=True);dump(out/'packet.TRAIN.json',packet)
 public={k:packet[k] for k in ('version','scope','catalogSha256','productCount','bindingCount','featureRecords','bindings')};public['teacherOrLabelContent']=False;dump(out/'public-records.json',public)
 manifest={'version':VERSION,'catalogFile':str(catalog_file),'catalogSha256':packet['catalogSha256'],'packetSha256':sha(out/'packet.TRAIN.json'),'publicRecordsSha256':sha(out/'public-records.json'),'productCount':packet['productCount'],'bindingCount':packet['bindingCount'],'config':CONFIG,'sourceSha256':sources(),'allBindingsTrain':True,'heldoutQualityClaim':False}
 dump(out/'PREPARED.json',manifest);return manifest

def validate_native_provenance(provenance,count,catalog_sha):
 expected={'evidenceSource':'native-frozen-qwen','modelRevision':model.MODEL_REVISION,'copyAdapterEnabled':False,'rankAdapterEnabled':False,'nativeForwards':count,'truncatedRows':0,'catalogSha256':catalog_sha,'frozenTensorCount':927,'frozen927HashesUnchanged':True,'shutdownVerified':True}
 if type(provenance) is not dict or any(provenance.get(k)!=v or (isinstance(v,bool) and type(provenance.get(k)) is not bool) for k,v in expected.items()):raise ValueError('Actual complete native frozen-base extraction/927 cleanup provenance required')
 p=provenance.get('nativeReceiptFile');h=provenance.get('nativeReceiptSha256')
 if not p or not Path(p).is_absolute() or sha(p)!=h:raise ValueError('Pinned actual native extraction receipt required')
 receipt=json.loads(Path(p).read_text())
 # Producer writes explicit copy provenance in its durable final receipt.
 if receipt.get('copyFeatureProvenance')!=expected:raise ValueError('Native receipt does not independently bind exact extraction provenance')
 return provenance

def make_expanded_feature_bundle(records,values,provenance,*,catalog_sha):
 if any(data.public_record(r['facts'])!=r for r in records):raise ValueError('Pure exact public fact records required')
 validate_native_provenance(provenance,len(records),catalog_sha)
 values=values.detach().cpu().to(torch.float32).clone();model.validate_vectors(values,len(records))
 return {'schemaVersion':VERSION,'featureKeys':[r['featureKey'] for r in records],'textSha256':[r['textSha256'] for r in records],'features':values,'featureTensorSha256':model.tensor_sha(values),'provenance':deepcopy(provenance),'publicCatalogSha256':catalog_sha}
def validate_bundle(bundle,public):
 if public.get('version')!=VERSION or public.get('scope')!=SCOPE:raise ValueError('Explicit expanded public contract version/scope required')
 records=public['featureRecords']
 if any(data.public_record(r['facts'])!=r for r in records):raise ValueError('Pure bound source records required')
 if bundle.get('schemaVersion')!=VERSION or bundle.get('publicCatalogSha256')!=public['catalogSha256'] or bundle.get('featureKeys')!=[r['featureKey'] for r in records] or bundle.get('textSha256')!=[r['textSha256'] for r in records]:raise ValueError('Complete public source feature keys/order/catalog differ')
 model.validate_vectors(bundle['features'],len(records))
 if model.tensor_sha(bundle['features'])!=bundle.get('featureTensorSha256'):raise ValueError('Fact feature tensor SHA differs')
 validate_native_provenance(bundle['provenance'],len(records),public['catalogSha256'])
 if 'catalogVectors' in bundle:
  model.validate_vectors(bundle['catalogVectors'],public['productCount'])
  pids=list(dict.fromkeys(x['productId'] for x in public['bindings']))
  if model.tensor_sha(bundle['catalogVectors'])!=bundle.get('catalogTensorSha256'):raise ValueError('Separate catalog tensor checksum differs')
  if bundle.get('catalogProductIds')!=pids or bundle.get('catalogNativeForwards')!=len(pids) or len(bundle.get('catalogTextSha256',[]))!=len(pids):raise ValueError('Separate complete native catalog feature evidence required')
 return {r['featureKey']:bundle['features'][i] for i,r in enumerate(records)}

def replay(head,rows,feature_map):
 generated=model.generate(head,model.inference_rows(rows),feature_map)
 return {'fields':len(rows),'accepted':sum(x['accepted'] for x in generated['fields']),'exact':sum(x['text']==r['targetText'] for x,r in zip(generated['fields'],rows,strict=True)),'slotIdsSha256':hashlib.sha256(json.dumps([x['slotIds'] for x in generated['fields']],separators=(',',':')).encode()).hexdigest()}

def fit_expanded(packet_dir,bundle_file,out):
 packet_dir=Path(packet_dir).resolve();bundle_file=Path(bundle_file).resolve();out=Path(out).resolve()
 if out.exists():raise FileExistsError('Unused fit directory required; preserve failed fits')
 prep=json.loads((packet_dir/'PREPARED.json').read_text());assert_sources(prep['sourceSha256'])
 if sha(prep['catalogFile'])!=prep['catalogSha256'] or sha(packet_dir/'packet.TRAIN.json')!=prep['packetSha256'] or sha(packet_dir/'public-records.json')!=prep['publicRecordsSha256']:raise ValueError('Frozen packet/catalog changed')
 packet=json.loads((packet_dir/'packet.TRAIN.json').read_text());public=json.loads((packet_dir/'public-records.json').read_text());bundle=torch.load(bundle_file,map_location='cpu',weights_only=True);feature_map=validate_bundle(bundle,public)
 if packet['featureRecords']!=public['featureRecords'] or packet['bindings']!=public['bindings']:raise ValueError('Public/TRAIN fact binding mismatch')
 input_pins={str(p):sha(p) for p in (packet_dir/'PREPARED.json',packet_dir/'packet.TRAIN.json',packet_dir/'public-records.json',Path(prep['catalogFile']),bundle_file,Path(bundle['provenance']['nativeReceiptFile']))}
 out.mkdir(parents=True);report={'version':VERSION,'status':'started','scope':SCOPE,'config':CONFIG,'packetPreparedSha256':sha(packet_dir/'PREPARED.json'),'featureBundleSha256':sha(bundle_file),'nativeFeatureProvenance':bundle['provenance'],'heldoutQualityClaim':False,'nativeForwardsDuringFit':0,'optimizerUpdates':0,'sourceSha256':sources(),'inputSha256':input_pins};dump(out/'preflight.json',report)
 try:
  with model.bounded_cpu_threads(1):
   head=model.ParallelTextHead(packet['vocabulary'],width=CONFIG['width'],seed=CONFIG['seed']);report['initialHeadStateSha256']=state_sha(head);x,c=model.serving_tensors(model.inference_rows(packet['train']),feature_map);targets=model.encode_labels(packet['train'],packet['vocabulary']);optimizer=torch.optim.AdamW(head.parameters(),lr=CONFIG['learningRate'],weight_decay=CONFIG['weightDecay']);started=time.perf_counter();checks=[]
   for step in range(1,CONFIG['maxSteps']+1):
    optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(head(x,c).flatten(0,1),targets.flatten(),ignore_index=-100)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite expanded copy loss')
    loss.backward();optimizer.step();report['optimizerUpdates']=step
    if step%CONFIG['checkEvery']==0:
     check=replay(head,packet['train'],feature_map);checks.append({'step':step,'loss':float(loss.detach()),**check});dump(out/'progress.json',checks)
     if check['exact']==check['fields'] and check['accepted']==check['fields']:break
   report.update(fitSeconds=time.perf_counter()-started,checks=checks,finalHeadStateSha256=state_sha(head))
   report['headWeightsChanged']=report['initialHeadStateSha256']!=report['finalHeadStateSha256']
   if not report['headWeightsChanged'] or any(not torch.isfinite(p).all() for p in head.parameters()):raise ValueError('Actual finite changed head weights required')
   torch.save({'version':VERSION,'vocabulary':list(head.vocabulary),'width':head.width,'state_dict':head.state_dict(),'featureBundleSha256':sha(bundle_file),'catalogSha256':public['catalogSha256']},out/'head.pt')
   saved=torch.load(out/'head.pt',map_location='cpu',weights_only=True);restored=model.ParallelTextHead(saved['vocabulary'],width=saved['width']);restored.load_state_dict(saved['state_dict'],strict=True);restored.eval()
   final=replay(restored,packet['train'],feature_map);live=replay(head,packet['train'],feature_map)
   if final!=live:raise ValueError('Reload exact slot parity failed')
   complete=final['exact']==final['fields'] and final['accepted']==final['fields'];report.update(status='completed-supervised-catalog-fit' if complete else 'failed-supervised-completeness-gate',trainReplay=final,reloadExactSlotParity=True)
   # Serving export is pure public facts/features, never target strings/labels.
   dump(out/'public-runtime-features.json',public)
   manifest={'version':VERSION,'scope':SCOPE,'ready':complete,'catalogSha256':public['catalogSha256'],'catalogFile':prep['catalogFile'],'headFile':'head.pt','headSha256':sha(out/'head.pt'),'publicFile':'public-runtime-features.json','publicSha256':sha(out/'public-runtime-features.json'),'featureBundleFile':str(bundle_file),'featureBundleSha256':sha(bundle_file),'nativeReceiptFile':bundle['provenance']['nativeReceiptFile'],'nativeReceiptSha256':bundle['provenance']['nativeReceiptSha256'],'sourceSha256':sources(),'allBindingsSupervised':True,'heldoutQualityClaim':False,'runtimeTeacherLookup':False,'productCount':public['productCount'],'bindingCount':public['bindingCount'],'optimizerUpdates':report['optimizerUpdates'],'reloadExactSlotParity':True,'trainReplay':final}
   dump(out/'serving-manifest.json',manifest)
  assert_sources(prep['sourceSha256'])
  for p,h in input_pins.items():
   if sha(p)!=h:raise ValueError('Immutable expanded fit input changed: '+p)
 except BaseException as error:
  report.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
  mp=out/'serving-manifest.json'
  if mp.exists():
   failed=json.loads(mp.read_text());failed['ready']=False;dump(mp,failed)
 dump(out/'results.json',report);return report

def load_expanded_parallel_assets(run,expected_manifest_sha):
 """Pure serving loader: never opens packet.TRAIN.json, targets or teacher()."""
 run=Path(run).resolve();mp=run/'serving-manifest.json'
 if sha(mp)!=expected_manifest_sha:raise ValueError('Expanded copy manifest SHA differs')
 manifest=json.loads(mp.read_text());assert_sources(manifest['sourceSha256'])
 if manifest['version']!=VERSION or manifest['ready'] is not True or manifest['runtimeTeacherLookup'] is not False or manifest['heldoutQualityClaim'] is not False or manifest['allBindingsSupervised'] is not True or manifest['reloadExactSlotParity'] is not True:raise ValueError('Complete explicitly supervised non-promoted expanded copy assets required')
 pins={str(mp):expected_manifest_sha}
 for key,hkey in (('headFile','headSha256'),('publicFile','publicSha256'),('featureBundleFile','featureBundleSha256'),('nativeReceiptFile','nativeReceiptSha256'),('catalogFile','catalogSha256')):
  p=Path(manifest[key]);p=p if p.is_absolute() else run/p
  if key in ('headFile','publicFile') and p.parent.resolve()!=run:raise ValueError('Head/public file must be co-located')
  if sha(p)!=manifest[hkey]:raise ValueError('Expanded copy asset checksum differs: '+key)
  pins[str(p)]=manifest[hkey]
 public=json.loads((run/manifest['publicFile']).read_text())
 if set(public)!={'version','scope','catalogSha256','productCount','bindingCount','featureRecords','bindings','teacherOrLabelContent'}:raise ValueError('Exact pure-public runtime contract required; no label extras')
 records=public['featureRecords'];catalog=json.loads(Path(manifest['catalogFile']).read_text())
 # Independent public facts reconstruction, no teacher/label constructor.
 expected=[];expected_bindings=[]
 for pi,p in enumerate(catalog['products']):
  for vi,v in enumerate(p['variants']):
   rec=data.public_record({'name':p['name'],'category':p['category'],'itemPrice':p['itemPrice'],'selectedColor':v['colorName']});expected.append(rec);expected_bindings.append({'productIndex':pi,'variantIndex':vi,'productId':p['productId'],'variantId':v['variantId'],'featureKey':rec['featureKey']})
 if records!=expected or public['bindings']!=expected_bindings or public['teacherOrLabelContent'] is not False or public['catalogSha256']!=manifest['catalogSha256'] or public['bindingCount']!=len(records) or public['productCount']!=len(catalog['products']):raise ValueError('Actual expanded public source facts differ')
 bundle=torch.load(manifest['featureBundleFile'],map_location='cpu',weights_only=True);feature_map=validate_bundle(bundle,public);saved=torch.load(run/manifest['headFile'],map_location='cpu',weights_only=True)
 if saved['version']!=VERSION or saved['featureBundleSha256']!=manifest['featureBundleSha256'] or saved['catalogSha256']!=manifest['catalogSha256']:raise ValueError('Head/native/public linkage differs')
 head=model.ParallelTextHead(saved['vocabulary'],width=saved['width']);head.load_state_dict(saved['state_dict'],strict=True);head.eval()
 if any(not torch.isfinite(p).all() for p in head.parameters()):raise ValueError('Nonfinite expanded head')
 return head,feature_map,records,{'version':VERSION,'kind':'fresh-parallel-head','assetSha256':pins,'checkpointSha256':manifest['headSha256'],'featuresSha256':manifest['featureBundleSha256'],'featureBundleFile':manifest['featureBundleFile'],'publicFactsSha256':manifest['publicSha256'],'catalogSha256':manifest['catalogSha256'],'nativeProvenance':bundle['provenance'],'trainingScope':SCOPE,'heldoutBindings':0,'teacherStringsReadAtServing':False,'headSlots':data.SLOTS,'vocabularySize':len(head.vocabulary),'style':'plain','productCount':public['productCount'],'bindingCount':len(records),'manifest':manifest}

def main(argv=None):
 parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='command',required=True);p=sub.add_parser('prepare');p.add_argument('--catalog',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p=sub.add_parser('fit-cpu');p.add_argument('--packet',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=parser.parse_args(argv)
 result=prepare(args.catalog,args.out) if args.command=='prepare' else fit_expanded(args.packet,args.bundle,args.out);print(json.dumps({'status':result.get('status','prepared'),'bindings':result.get('bindingCount'),'optimizerUpdates':result.get('optimizerUpdates')}))
 if args.command=='fit-cpu' and result['status']!='completed-supervised-catalog-fit':raise SystemExit(1)
if __name__=='__main__':main()
