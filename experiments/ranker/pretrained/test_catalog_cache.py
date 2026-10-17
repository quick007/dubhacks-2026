"""Exact-state serialization proof on a tiny native hybrid, no pretrained weights."""
import tempfile
import unittest
from pathlib import Path
from .catalog_cache import hashed


class CacheKeyTests(unittest.TestCase):
    def test_model_catalog_profile_keys(self):
        self.assertEqual(hashed({'a':1,'b':2}),hashed({'b':2,'a':1}))
        self.assertNotEqual(hashed({'model':'v1','catalog':'a'}),hashed({'model':'v2','catalog':'a'}))
        self.assertNotEqual(hashed({'model':'v1','catalog':'a'}),hashed({'model':'v1','catalog':'b'}))


class NativeCacheTests(unittest.TestCase):
    def test_native_roundtrip_full_state_and_A_B_A(self):
        import mlx.core as mx
        from mlx_lm.models.qwen3_5 import Qwen3_5TextModel,TextModelArgs
        from mlx_lm.models.cache import ArraysCache,KVCache,save_prompt_cache,load_prompt_cache
        args=TextModelArgs(model_type='qwen3_5',hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,vocab_size=32,linear_num_value_heads=2,
            linear_num_key_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,full_attention_interval=2)
        model=Qwen3_5TextModel(args);model.eval();model.freeze();cache=[ArraysCache(2),KVCache()]
        hidden=model(mx.array([[1,2,3]]),cache=cache);mx.eval(hidden,[c.state for c in cache]);mx.synchronize()
        self.assertIsNotNone(cache[0][0]);self.assertIsNotNone(cache[0][1]);self.assertEqual(cache[1].offset,3)
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'catalog.safetensors';save_prompt_cache(str(file),cache,{'version':'test'})
            from .evidence_plan import digest
            original=digest(file);results=[]
            for suffix in ([4,5],[6,7],[4,5]):
                restored,meta=load_prompt_cache(str(file),return_metadata=True)
                self.assertEqual(meta,{'version':'test'});self.assertEqual(restored[1].offset,3)
                self.assertIsNotNone(restored[0][0]);self.assertIsNotNone(restored[0][1])
                mx.eval([c.state for c in restored]);mx.synchronize()
                appended=model(mx.array([suffix]),cache=restored)
                fresh=model(mx.array([[1,2,3,*suffix]]));mx.eval(appended,fresh,[c.state for c in restored]);mx.synchronize()
                self.assertLess(float(mx.max(mx.abs(appended[:,-1]-fresh[:,-1])).item()),1e-4)
                results.append(appended[:,-1]);self.assertEqual(digest(file),original)
            self.assertTrue(bool(mx.array_equal(results[0],results[2]).item()))

if __name__=='__main__':unittest.main()
