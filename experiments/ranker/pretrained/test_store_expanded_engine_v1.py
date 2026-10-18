from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
from . import store_expanded_engine_v1 as engine
from .test_store_expanded_rank_v1 import catalog


class Tests(unittest.TestCase):
    def fake(self,directory):
        c=catalog();runtime=NS(manifest={'productIds':[p['productId']for p in c['products'][:16]],'adapter':{'sha256':'a'*64}},ownership=NS(assert_idle=lambda:None))
        value=NS(output=Path(directory),adapter_runtime=runtime,binding=NS(catalog_sha='b'*64,manifest_sha='c'*64),copy_metadata={'assetSha256':{},'scope':'TRAIN fixture'},
            expanded_sources={},session=NS(products=[],snapshot={'externalMapping':{'products':[]}},head_sha='d'*64),startup={},closed=0,adapter=object())
        def close():value.closed+=1;return {'closed':True}
        value.close=close;value._assert_candidate=lambda:None
        return value,c
    def test_actual_expansion_replaces_only_action_metadata_and_requires_every_copybinding(self):
        from .store_parallel_text_data_v1 import public_record
        with tempfile.TemporaryDirectory()as directory:
            base,c=self.fake(directory);base.public_records=[public_record({'name':p['name'],'category':p['category'],'itemPrice':p['itemPrice'],'selectedColor':v['colorName']})for p in c['products']for v in p['variants']]
            klass=engine.engine_class(copy_loader=lambda*a:None,practical_factory=lambda **kw:lambda **args:base)
            with patch.object(engine,'load_catalog_vectors',return_value=c),patch.object(klass,'_prewarm_queries',return_value={'uniqueKeys':24}):
                instance=klass()
            self.assertEqual(len(base.session.products),21);self.assertEqual(len(base.session.snapshot['externalMapping']['products']),21)
            self.assertEqual(base.startup['expandedCatalog']['originalLoRAFamilies'],16);self.assertFalse(base.startup['expandedCatalog']['newSkuLoRATrainingClaim'])
            self.assertTrue((Path(directory)/'startup.json').is_file());self.assertIs(instance.engine,base)
    def test_missing_copybinding_fails_closed_and_tears_down(self):
        with tempfile.TemporaryDirectory()as directory:
            base,c=self.fake(directory);base.public_records=[]
            klass=engine.engine_class(copy_loader=lambda*a:None,practical_factory=lambda **kw:lambda **args:base)
            with patch.object(engine,'load_catalog_vectors',return_value=c):
                with self.assertRaisesRegex(ValueError,'Every actual expanded'):klass()
            self.assertEqual(base.closed,1)
    def test_handler_uses_actual_expanded_source_and_no_measurement_controls(self):
        c=catalog();Handler=engine.handler_for(NS(),c['products'])
        def call(value,path='/personalize',origin='http://shop.dubhacks.localhost:1356'):
            h=Handler.__new__(Handler);h.path=path;data=json.dumps(value).encode();h.headers={'Origin':origin,'Content-Length':str(len(data))};h.rfile=io.BytesIO(data);h.connection=NS(settimeout=lambda x:None)
            replies=[];h.respond=lambda status,body:replies.append((status,body));h.do_POST();return replies
        self.assertEqual(call({'query':'Work bags'},'/measurement/route-projection')[0][0],404)
        self.assertEqual(call({'query':'Work bags'},origin='https://other.example')[0][0],403)
        self.assertEqual(call({'query':'Work bags','publicHistory':[{'eventId':'bad'}]})[0][0],400)
    def test_native_catalog_vectors_must_join_final_receipt_not_bundle_assertion(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);file=root/'features.pt';c=catalog();binding=NS(products=c['products'],catalog_sha='e'*64)
            from .store_function_adaptation import product_text
            from .store_persona_adapter_contract_v2 import sha
            from .store_parallel_text_head_v1 import tensor_sha,MODEL_REVISION
            import hashlib
            texts=[hashlib.sha256(product_text(p).encode()).hexdigest()for p in c['products']];ids=[p['productId']for p in c['products']]
            native=root/'native.json';native.write_text(json.dumps({'catalogFeatureProvenance':{'evidenceSource':'native-frozen-qwen','modelRevision':MODEL_REVISION,'copyAdapterEnabled':False,'rankAdapterEnabled':False,'nativeForwards':len(ids),'truncatedRows':0,'catalogSha256':binding.catalog_sha,'productIds':ids,'textSha256':texts,'catalogTensorSha256':tensor_sha(c['vectors']),'frozenTensorCount':927,'frozen927HashesUnchanged':True,'shutdownVerified':True}}))
            bundle={'catalogVectors':c['vectors'],'catalogProductIds':ids,'catalogNativeForwards':len(ids),'publicCatalogSha256':binding.catalog_sha,'catalogTextSha256':texts,'catalogTensorSha256':tensor_sha(c['vectors']),'provenance':{'nativeReceiptFile':str(native),'frozen927HashesUnchanged':True,'shutdownVerified':True}}
            torch.save(bundle,file);meta={'featureBundleFile':str(file),'assetSha256':{str(file):sha(file)}};families=ids[:16]
            self.assertEqual(len(engine.load_catalog_vectors(meta,binding,families)['products']),21)
            native.write_text('{}')
            with self.assertRaisesRegex(ValueError,'join final native'):engine.load_catalog_vectors(meta,binding,families)

if __name__=='__main__':unittest.main()
