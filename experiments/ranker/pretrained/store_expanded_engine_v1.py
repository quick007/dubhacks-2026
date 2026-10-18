"""Explicit expanded35 public store, strict intent eligibility and fresh copy.

Original16 LoRA routes are family priors, not newly trained SKU logits. Root owns
native extraction/execution. No old fixed16/fixed48 loaders are modified.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import torch

from . import store_persona_practical_engine_v1 as practical
from . import store_persona_adapter_contract_v2 as contract
from .store_demo_engine import ResidentDemoEngine, RequestCancelled, public_sections, NEUTRAL_COPY_REQUEST
from .store_expanded_binding_v1 import ExpandedBinding
from .store_expanded_rank_v1 import ExpandedRankAdapter, checked_request, validate_catalog
from .store_candidate_engine_v1 import parallel_rows,parallel_payload

VERSION='expanded-personalized-resident-engine.v1'


def load_catalog_vectors(metadata,binding,family_ids):
    from .store_function_adaptation import product_text
    filename=Path(metadata['featureBundleFile'])
    if contract.sha(filename)!=metadata['assetSha256'][str(filename)]: raise ValueError('Expanded feature bundle checksum changed')
    bundle=torch.load(filename,map_location='cpu',weights_only=True)
    products=deepcopy(binding.products);ids=[p['productId'] for p in products]
    expected=[hashlib.sha256(product_text(p).encode()).hexdigest() for p in products]
    if (bundle.get('catalogProductIds')!=ids or bundle.get('publicCatalogSha256')!=binding.catalog_sha or
        bundle.get('catalogNativeForwards')!=len(ids) or bundle.get('catalogTextSha256')!=expected or
        bundle.get('provenance',{}).get('frozen927HashesUnchanged') is not True or
        bundle.get('provenance',{}).get('shutdownVerified') is not True):
        raise ValueError('Actual expanded public catalog native provenance/text/order required')
    vectors=bundle['catalogVectors']
    from .store_parallel_text_head_v1 import tensor_sha,MODEL_REVISION
    checksum=tensor_sha(vectors)
    receipt=json.loads(Path(bundle['provenance']['nativeReceiptFile']).read_text())
    catalog_proof={'evidenceSource':'native-frozen-qwen','modelRevision':MODEL_REVISION,'copyAdapterEnabled':False,'rankAdapterEnabled':False,
        'nativeForwards':len(ids),'truncatedRows':0,'catalogSha256':binding.catalog_sha,'productIds':ids,'textSha256':expected,
        'catalogTensorSha256':checksum,'frozenTensorCount':927,'frozen927HashesUnchanged':True,'shutdownVerified':True}
    if checksum!=bundle.get('catalogTensorSha256') or receipt.get('catalogFeatureProvenance')!=catalog_proof:
        raise ValueError('Actual catalog vectors/text/order must join final native tensor/927/shutdown receipt')
    variants=[{**v,'productIndex':i} for i,p in enumerate(products) for v in p['variants']]
    return validate_catalog({'products':products,'variants':variants,'vectors':vectors},family_ids)


def engine_class(*,copy_loader=None,resident_cls=ResidentDemoEngine,practical_factory=practical.engine_class,binding_cls=ExpandedBinding):
    if copy_loader is None:
        from .store_expanded_parallel_copy_v1 import load_expanded_parallel_assets
        copy_loader=load_expanded_parallel_assets
    class Base(resident_cls):
        def __init__(self,*,mode,candidate_run,expanded_copy_manifest_sha256,**kwargs):
            if mode!='parallel': raise ValueError('Expanded store requires actual fresh neural copy')
            self.head,self.feature_map,self.public_records,self.copy_metadata=copy_loader(candidate_run,expanded_copy_manifest_sha256)
            self.expanded_sources={str(Path(__file__)):contract.sha(__file__)}
            for name in ('store_expanded_binding_v1.py','store_expanded_rank_v1.py','store_expanded_parallel_copy_v1.py'):
                path=Path(__file__).with_name(name);self.expanded_sources[str(path)]=contract.sha(path)
            try:
                super().__init__(catalog_binding_factory=binding_cls,**kwargs)
                self._assert_candidate()
            except BaseException as error:
                from .store_candidate_image_service_v1 import cleanup_failed_start
                cleanup_failed_start(self,error)
                raise
        def _assert_candidate(self):
            for filename,expected in {**self.expanded_sources,**self.copy_metadata['assetSha256']}.items():
                if contract.sha(filename)!=expected: raise ValueError('Expanded source/copy artifact changed')
            self.binding.assert_current_sources(self.session.snapshot['receipt'])
            if hasattr(self.session,'integrity_receipt'): self.session.integrity_receipt()
        def close(self):
            primary=None
            try:self._assert_candidate()
            except BaseException as error:primary=error
            try:result=super().close()
            except BaseException as error:
                if primary:primary.add_note('Native shutdown also failed: '+str(error));raise primary
                raise
            if primary:raise primary
            self.binding.assert_full_integrity()
            result.update(expandedCatalogSourceAndCopyAssetsUnchanged=True,originalAssetsFullShutdownSHAUnchanged=True)
            return result
    factory=practical_factory(base_cls=Base)
    class Engine:
        def __init__(self,**kwargs):
            self.engine=factory(**kwargs)
            try:
                runtime=self.engine.adapter_runtime
                self.catalog=load_catalog_vectors(self.engine.copy_metadata,self.engine.binding,runtime.manifest['productIds'])
                self.expanded_rank=ExpandedRankAdapter(runtime,self.catalog,catalog_sha256=self.engine.binding.catalog_sha,
                    source_pins={**self.engine.expanded_sources,**self.engine.copy_metadata['assetSha256']})
                from .store_parallel_text_data_v1 import public_record
                expected={json.dumps(public_record({'name':p['name'],'category':p['category'],'itemPrice':p['itemPrice'],'selectedColor':v['colorName']}),sort_keys=True)
                    for p in self.catalog['products'] for v in p['variants']}
                if expected!={json.dumps(r,sort_keys=True) for r in self.engine.public_records}:
                    raise ValueError('Every actual expanded source product/color must have exact public copy features')
                # Historical catalog_rows/feature files remain original16; only
                # active products/action mapping are expanded for receipts/UI.
                self.engine.session.products=self.catalog['products']
                self.engine.session.snapshot=deepcopy(self.engine.session.snapshot)
                self.engine.session.snapshot['externalMapping']['products']=[{'productId':p['productId']} for p in self.catalog['products']]
                self.engine.startup['expandedCatalog']={'version':VERSION,'products':len(self.catalog['products']),
                    'variants':len(self.catalog['variants']),'catalogSha256':self.engine.binding.catalog_sha,
                    'bindingManifestSha256':self.engine.binding.manifest_sha,'originalLoRAFamilies':16,'newSkuLoRATrainingClaim':False,
                    'strictSpecificSearchEligibilityBeforePersonalization':True,'copy':self.engine.copy_metadata,
                    'modelPromotion':False,'qualityUnvalidated':True,'querySuggestionPreencoding':self._prewarm_queries()}
                self.engine._assert_candidate();self._save_startup()
            except BaseException as error:
                try:self.engine.close()
                except BaseException as cleanup:error.add_note('Expanded initialization cleanup: '+str(cleanup))
                raise
        def __getattr__(self,name):return getattr(self.engine,name)
        def _save_startup(self):(self.output/'startup.json').write_text(json.dumps(self.startup,indent=2,allow_nan=False)+'\n')
        def _prewarm_queries(self):
            from .store_persona_adapter_runtime_v3 import public_browse_requests
            from .store_demo_engine import ranking_input
            from .store_function_adaptation import request_text
            started=time.perf_counter();rows={};runtime=self.adapter_runtime
            for person,value in public_browse_requests():
                for query in ('Rain jackets','Work bags','Browse the store'):
                    value=deepcopy(value);value['query']=query
                    effective,_,_=checked_request(value,self.catalog['products']);text=request_text(ranking_input(effective))
                    key=hashlib.sha256(text.encode()).hexdigest()
                    if key not in rows:
                        _,native=runtime.encode_base(text)
                        rows[key]={'wholeQueryTextSha256':key,'query':query,'actualItemBudget':effective['itemBudget'],
                            'native':native,'publicSourcePersonIds':[]}
                    rows[key]['publicSourcePersonIds'].append(person)
            runtime.encoding_calls.clear();runtime.query_cache=None
            if len(rows)!=24:raise ValueError('Exactly24 actual query/budget/guard keys required for three public suggestions')
            return {'queries':['Rain jackets','Work bags','Browse the store'],'keys':list(rows.values()),'uniqueKeys':len(rows),
                'elapsedMs':(time.perf_counter()-started)*1000,'scoresRankingsOrAnswersCached':False,'featureCacheCapacity':32}
        def _assert_candidate(self):self.engine._assert_candidate();self.expanded_rank.assert_current()
        def close(self):return self.engine.close()
        def personalize(self,value,request_id,started,emit,cancelled):
            from . import store_parallel_text_head_v1 as words
            from .store_copy_wide_v2 import guard_memory
            request,policy,_=checked_request(value,self.catalog['products'])
            directory=self.output/'requests'/request_id;directory.mkdir(parents=True,exist_ok=False)
            receipt={'version':VERSION,'requestId':request_id,'input':deepcopy(value),'effectiveRequest':request,'policy':policy,
                'status':'running','copyConditioning':deepcopy(NEUTRAL_COPY_REQUEST),'copyShopperQueryUsed':False,'events':[]}
            def save():
                temp=directory/'receipt.tmp';temp.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');temp.replace(directory/'receipt.json')
            def event(kind,**body):
                if cancelled():raise RequestCancelled('Cancelled before expanded event')
                self._assert_candidate();item={'type':kind,'requestId':request_id,**body}
                item.setdefault('timing',{})['readyMs']=(time.perf_counter()-started)*1000
                if kind in ('firstThree','complete'):item['timing']['first3readyMs' if kind=='firstThree' else 'fullReadyMs']=item['timing']['readyMs']
                receipt['events'].append(deepcopy(item));emit(item)
            save()
            try:
                request,policy,order,guards,cards,ranked=self.expanded_rank.rank(value,self.session,self.adapter,cancelled=cancelled)
                rankms=(time.perf_counter()-started)*1000
                provenance={'ranking':'strict source product-type eligibility, original16 learned LoRA family priors and learned expanded public-product projections',
                    'personaRanking':ranked,'rankCheckpointSha256':self.adapter_runtime.manifest['adapter']['sha256'],
                    'v5QueryCheckpointSha256':self.session.head_sha,'copy':self.copy_metadata,
                    'images':'source-bound learned within-product color; actual distinct gallery metadata; no pixel model claim',
                    'qualityScope':'expanded known-public-catalog development demo; quality unvalidated; no model promotion','hiddenProfileInputs':False}
                receipt.update(ranking=ranked,finalOrder=order,guards=guards,cards=cards,provenance=provenance)
                shared={'eligibleCount':len(order),'shortfall':max(0,6-len(cards)),'provenance':provenance}
                metadata,_=public_sections(cards,[],self.catalog['products'])
                event('rank',productIds=[self.catalog['products'][i]['productId'] for i in order],selectedProductIds=[c['action']['productId'] for c in cards],
                    variants={c['action']['productId']:c['action']['variantId'] for c in cards},sections=metadata,timing={'rankMs':rankms},**shared);save()
                self.adapter_runtime.ownership.assert_idle()
                if cards:
                    if cancelled():raise RequestCancelled('Cancelled before fresh expanded word slots')
                    rows=parallel_rows(cards)
                    with words.bounded_cpu_threads(1):generated=words.generate(self.head,words.inference_rows(rows),self.feature_map)
                    receipt.update(runtimeRows=rows,parallelGeneration=generated)
                    source={**self.copy_metadata,'kind':'fresh-parallel-head','generatedEarlier':False,'freshPerRequestGeneration':True,
                        'autoregressiveSteps':0,'feedforwardCalls':generated['feedforwardCalls']}
                    payload=lambda limit:parallel_payload(cards,rows,generated,limit)
                else:
                    source={'kind':'no-generation','generatedEarlier':False,'freshPerRequestGeneration':False,'noEligibleProducts':True}
                    payload=lambda limit:public_sections([],[],self.catalog['products'],0)
                for kind,limit in ([('firstThree',min(3,len(cards))),('complete',len(cards))] if cards else [('complete',0)]):
                    sections,validation=payload(limit)
                    event(kind,sections=sections,validation=validation,copySource=source,
                        **{k:validation[k] for k in ('acceptedFields','withheldFields','requiredFields')},
                        timing={'rankMs':rankms,'copyMs':(time.perf_counter()-started)*1000-rankms},**shared)
                receipt.update(status='completed',memory=guard_memory(self.bb));save()
            except BaseException as error:
                receipt.update(status='cancelled' if isinstance(error,RequestCancelled) else 'failed',error={'type':type(error).__name__,'message':str(error)},failedAtMs=(time.perf_counter()-started)*1000);save();raise
    return Engine


def handler_for(worker,products):
    import queue,uuid
    from .store_persona_engine_v1 import persona_handler_for,MAX_REQUEST_BYTES
    from .store_persona_portless_http_v1 import handler_for as portless
    class Handler(persona_handler_for(worker)):
        def do_POST(self):
            started=time.perf_counter()
            if not self.origin_allowed():return self.respond(403,{'error':'Origin is not an allowed local storefront'})
            if self.path!='/personalize':return self.respond(404,{'error':'Unknown endpoint'})
            if self.headers.get('Transfer-Encoding'):return self.respond(400,{'error':'Bounded Content-Length required'})
            try:
                size=int(self.headers.get('Content-Length','-1'))
                if not 1<=size<=MAX_REQUEST_BYTES:return self.respond(413,{'error':'Bounded public request required'})
                self.connection.settimeout(10);raw=self.rfile.read(size)
                if len(raw)!=size:raise ValueError('Incomplete public request')
                value=json.loads(raw);checked_request(value,products)
            except (ValueError,UnicodeError,TimeoutError) as error:return self.respond(400,{'error':str(error)})
            request_id=str(uuid.uuid4())
            try:events,cancel,future=worker.submit(value,request_id,started)
            except queue.Full:return self.respond(429,{'error':'Local model queue full'})
            except RuntimeError as error:return self.respond(503,{'error':str(error),'health':worker.health()})
            self.send_response(200);self.cors();self.send_header('Content-Type','application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Connection','close');self.send_header('X-Request-Id',request_id);self.end_headers();self.close_connection=True
            try:
                while True:
                    if self.disconnected():cancel.set();break
                    try:item=events.get(timeout=.2)
                    except queue.Empty:continue
                    if item is None:
                        error=future.exception()
                        if error is not None and not isinstance(error,RequestCancelled):
                            self.wfile.write((json.dumps({'type':'error','requestId':request_id,'message':f'{type(error).__name__}: {error}'})+'\n').encode());self.wfile.flush()
                        break
                    self.wfile.write((json.dumps(item,allow_nan=False)+'\n').encode());self.wfile.flush()
            except (ConnectionError,OSError,TimeoutError):cancel.set()
            finally:cancel.set()
    return portless(worker,lambda _:Handler)


def main(argv=None):
    import argparse,os,signal,threading
    from http.server import ThreadingHTTPServer
    parser=argparse.ArgumentParser(__doc__)
    for name in ('asset-root','output','serving-manifest','catalog-feature-root','color-head','color-cache','query-intent-head','candidate-run','integration-manifest','rank-head-export'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('serving-manifest-sha256','query-intent-head-sha256','rank-head-export-sha256','expanded-copy-manifest-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--query-intent-source-manifest-sha256')
    args=parser.parse_args(argv);root=Path(__file__).resolve().parents[3]
    if not args.output.is_absolute() or not args.output.resolve().is_relative_to(root) or args.output.exists():parser.error('Unique absolute output in current code root required')
    if not args.asset_root.is_absolute() or args.asset_root.resolve()!=practical.MAIN_ASSET_ROOT.resolve():parser.error('Actual MAIN local assets required')
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    os.chdir(args.asset_root)
    # CPU validation precedes native worker, including actual active35 UI export.
    binding=ExpandedBinding(args.integration_manifest)
    from .store_demo_service import ModelWorker
    from .store_rank_session_v5 import make_v5_rank_session_factory
    klass=engine_class();server=ThreadingHTTPServer(('127.0.0.1',8767),handler_for(None,binding.products))
    worker=ModelWorker(lambda:klass(mode='parallel',serving_manifest=args.serving_manifest,serving_sha=args.serving_manifest_sha256,
        catalog_feature_root=args.catalog_feature_root,color_head=args.color_head,color_cache=args.color_cache,query_intent_head=args.query_intent_head,
        query_intent_sha=args.query_intent_head_sha256,query_intent_source_manifest_sha=args.query_intent_source_manifest_sha256,
        candidate_run=args.candidate_run,expanded_copy_manifest_sha256=args.expanded_copy_manifest_sha256,
        integration_manifest=args.integration_manifest,output=args.output,width=20,
        rank_session_factory=make_v5_rank_session_factory(args.rank_head_export,args.rank_head_export_sha256)))
    server.RequestHandlerClass=handler_for(worker,binding.products);server.daemon_threads=True
    def stop(*_):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        if not worker.close():raise RuntimeError('Expanded worker teardown integrity unconfirmed')

if __name__=='__main__':main()
