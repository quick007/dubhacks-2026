"""CPU protocol fixtures only; fake tensors are not training evidence."""
import copy
from types import SimpleNamespace
import unittest
import torch
from .store_persona_runtime_v1 import ExactQueryFeatureCache

class CacheTests(unittest.TestCase):
    def backbone(self):
        class Fake:
            def __init__(self):
                self.calls=0;self.model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=object()))])
            def encode(self, rows):
                self.calls+=1
                return torch.ones(1,4096)*self.calls, {'backboneForwardCalls':1,'batchRows':1,'truncatedRows':0}
        return Fake()
    def test_hit_is_exact_source_bound_and_returns_isolated_tensor(self):
        b=self.backbone();c=ExactQueryFeatureCache({'revision':'pinned','sources':{'x':'sha'}})
        x,r=c.encode('actual canonical text',b);x.fill_(17)
        y,s=c.encode('actual canonical text',b)
        self.assertEqual(b.calls,1);self.assertTrue(torch.equal(y,torch.ones(1,4096)))
        self.assertFalse(r['cacheHit']);self.assertTrue(s['cacheHit']);self.assertEqual(s['nativeBackboneForwardCalls'],0)
        c.encode('actual canonical text ',b);self.assertEqual(b.calls,2)
        c.provenance['revision']='changed';c.encode('actual canonical text ',b);self.assertEqual(b.calls,3)
    def test_wrong_native_shape_not_cached(self):
        b=self.backbone();b.encode=lambda rows:(torch.zeros(1,2),{'backboneForwardCalls':1,'batchRows':1,'truncatedRows':0})
        c=ExactQueryFeatureCache({})
        with self.assertRaises(ValueError):c.encode('x',b)
        self.assertIsNone(c.entry)
    def test_copy_adapter_cannot_be_silently_active(self):
        b=self.backbone();b.model.layers[-1].mlp.down_proj=SimpleNamespace(lora_a=1)
        with self.assertRaises(ValueError):ExactQueryFeatureCache({}).encode('x',b)

if __name__=='__main__':unittest.main()

class FusionTests(unittest.TestCase):
    def test_specific_semantic_margin_greater_than_bound_is_preserved(self):
        from .store_persona_runtime_v1 import fuse_scores
        v5=torch.zeros(16);v5[3]=.6;v5[8]=.5
        persona=torch.full((16,),-100.);persona[8]=100
        result,broad=fuse_scores(v5,persona,'a waterproof jacket for rain')
        self.assertFalse(broad);self.assertEqual(int(result.argmax()),3)
        self.assertTrue(bool(((result-v5)>=0).all()));self.assertLessEqual(float((result-v5).max()),.050001)
    def test_default_browse_ranking_is_genuinely_profile_sensitive(self):
        from .store_persona_runtime_v1 import fuse_scores
        v5=torch.ones(16)/16;a=torch.zeros(16);b=torch.zeros(16);a[0]=10;b[6]=10
        ar,broad=fuse_scores(v5,a,'Browse the store');br,_=fuse_scores(v5,b,'Browse the store')
        self.assertTrue(broad);self.assertEqual(int(ar.argmax()),0);self.assertEqual(int(br.argmax()),6)

class ContextRepresentationTests(unittest.TestCase):
    def rank_fixture(self,native):
        from unittest.mock import patch
        from .store_persona_runtime_v1 import PersonaRuntime
        from . import store_persona_context_v1 as legacy
        from . import store_persona_qwen_v1 as qwen
        obj=PersonaRuntime.__new__(PersonaRuntime);obj.native_context=native
        obj.saved={'config':qwen.CONFIG if native else legacy.CONFIG};obj.head_sha='fixture-head'
        obj.assert_current=lambda:None;obj.catalog={'products':[],'vectors':torch.zeros(16,4096)}
        obj.colors=torch.zeros(48,256);obj.indices=torch.arange(16).repeat_interleave(3)
        obj.head=lambda *args:(torch.zeros(1,16),torch.zeros(1,48))
        query_receipt={'cacheHit':True,'nativeBackboneForwardCalls':0,'scope':'fixture exact query feature'}
        context_receipt={'kind':'precomputed-public-demo-feature','cacheHit':True,'nativeBackboneForwardCalls':0,'tokens':123}
        obj.cache=SimpleNamespace(encode=lambda *args,**kwargs:(torch.zeros(1,4096),copy.deepcopy(query_receipt)))
        obj.native_context_feature=lambda *args:(torch.zeros(1,4096),copy.deepcopy(context_receipt))
        session=SimpleNamespace(bb=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *args,**kwargs:[1])),
            store=SimpleNamespace(load=lambda row:(torch.zeros(1,4096),{})),catalog_rows=list(range(16)),products=[],
            head=SimpleNamespace(probabilities=lambda role,values:torch.eye(16) if role=='catalog' else torch.ones(1,16)/16))
        with patch('experiments.ranker.pretrained.store_demo_engine.guarded_order',return_value=([],{})),patch.object(legacy,'context_features',return_value=(torch.zeros(2048),torch.zeros(4096))):
            result=obj.rank({'query':'Browse the store'},session,adapter=None)[-1]
        self.assertEqual(result['queryFeature'],query_receipt)
        if native:self.assertEqual(result['contextFeature'],context_receipt)
        else:self.assertEqual(result['contextFeature'],{'kind':'public-hashed-context','nativeCalls':0})
        return result
    def test_rank_reports_loaded_native_or_legacy_representation_without_changing_features_or_scores(self):
        from . import store_persona_context_v1 as legacy
        from . import store_persona_qwen_v1 as qwen
        native=self.rank_fixture(True);hashed=self.rank_fixture(False)
        self.assertEqual(native['contextRepresentation'],qwen.CONFIG['contextRepresentation'])
        self.assertEqual(hashed['contextRepresentation'],legacy.CONFIG['contextRepresentation'])
        self.assertNotEqual(native['contextRepresentation'],hashed['contextRepresentation'])
        self.assertEqual(native['scores16'],hashed['scores16'])
