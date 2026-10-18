"""CPU unit fixtures only. No native feature/training result claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import store_persona_qwen_v1 as native
from .store_persona_qwen_fit_v1 import NativePersonaHead

class NativePersonaTests(unittest.TestCase):
    def request(self):
        return {'query':'Browse','persona':{'background':['Reads on a new sofa'],'colorPreferences':['Ivory']},
            'publicHistory':[{'eventId':'opaque','eventType':'purchase','itemName':'sofa','category':'Furniture','occurredAt':'2026-10-01T00:00:00Z','source':'demo-seed','external':True}]}
    def test_text_is_target_free_id_and_absolute_date_invariant(self):
        a=self.request();b=copy.deepcopy(a);b['publicHistory'][0]['eventId']='different';b['publicHistory'][0]['occurredAt']='2025-01-01T00:00:00Z'
        self.assertEqual(native.public_context_text(a),native.public_context_text(b))
        a['targetProductId']='blanket'
        with self.assertRaises(ValueError):native.public_context_text(a)
    def test_owned_history_is_semantically_distinct_from_view(self):
        a=self.request();b=copy.deepcopy(a);b['publicHistory'][0]['eventType']='view'
        self.assertNotEqual(native.public_context_text(a),native.public_context_text(b))
    def test_ownership_is_zero_initial_learned_and_differentiable(self):
        h=NativePersonaHead();g=torch.Generator().manual_seed(9)
        x=torch.randn(2,4096,generator=g);c=torch.randn(16,4096,generator=g);colors=torch.randn(48,256,generator=g);indices=torch.arange(16).repeat_interleave(3)
        own=torch.zeros(2,16);own[:,0]=1
        a,_=h(x,c,colors,indices,own);b,_=h(x,c,colors,indices,torch.zeros_like(own))
        self.assertTrue(torch.equal(a,b));a[:,0].sum().backward();self.assertEqual(float(h.ownership.grad),2.)
        with torch.no_grad():h.ownership.fill_(-1)
        after,_=h(x,c,colors,indices,own);self.assertTrue(torch.allclose(after[:,0],a[:,0]-1));self.assertTrue(torch.equal(after[:,1:],a[:,1:]))
    def test_no_run_does_not_import_or_construct_native_backbone(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'preflight.json';p.write_text(json.dumps({'expectedNativeForwards':402,'sourceSha256':{},'publicFilesSha256':{},'originalCatalogPins':{}}))
            with patch.dict('sys.modules',{'mlx':None,'mlx.core':None}):
                result=native.extract(p,Path(d)/'out',run=False)
            self.assertFalse(result['nativeExecuted']);self.assertFalse((Path(d)/'out').exists())

if __name__=='__main__':unittest.main()

class ExtractionFailureTests(unittest.TestCase):
    def test_final_integrity_failure_is_nonzero_and_lock_covers_shutdown(self):
        from contextlib import contextmanager, nullcontext
        from types import SimpleNamespace
        from . import store_persona_context_v1 as base
        events=[];calls=[]
        @contextmanager
        def lock(*args):
            events.append('lock-enter')
            try:yield
            finally:events.append('lock-exit')
        def hashes(model):
            calls.append(1);events.append('hash-'+str(len(calls)))
            return {str(i):('before' if len(calls)==1 else 'after') for i in range(927)}
        class FakeBackbone:
            def __init__(self,*args,**kwargs):
                self.model=SimpleNamespace(language_model=object());self.mx=SimpleNamespace(clear_cache=lambda:events.append('clear'),synchronize=lambda:events.append('sync'))
            def runtime_memory(self):return {'fixture':True}
        with tempfile.TemporaryDirectory(dir=base.HERE.parents[2]) as directory:
            root=Path(directory);p=root/'preflight.json';out=root/'out'
            p.write_text(json.dumps({'assetRoot':str(base.HERE.parents[2]),'expectedNativeForwards':0,'sourceSha256':{},'publicFilesSha256':{},'originalCatalogPins':{},'rows':[],'model':{'modelPath':'fixture'},'featureProvenance':{'modelRevision':'fixture'}}))
            with patch('experiments.ranker.pretrained.store_ar_comparison_run_v1.demo_listening',return_value=False),patch('experiments.ranker.pretrained.catalog_fusion.generalization_v2.native_owner_lock',lock),patch('experiments.ranker.pretrained.catalog_fusion.runtime.verify_model'),patch('experiments.ranker.pretrained.catalog_fusion.runtime.assert_model_files'),patch('experiments.ranker.pretrained.semantic_copy_training.parameter_hashes',hashes),patch('experiments.ranker.pretrained.backbone.FrozenBackbone',FakeBackbone),patch('experiments.ranker.pretrained.store_copy_fit_v2.deadline',lambda *args:nullcontext()):
                with self.assertRaisesRegex(RuntimeError,'integrity/shutdown'):native.extract(p,out,run=True)
            receipt=json.loads((out/'report.json').read_text())
            self.assertEqual(receipt['status'],'failed-frozen-integrity');self.assertFalse(receipt['frozen927HashesUnchanged'])
            self.assertLess(events.index('hash-2'),events.index('lock-exit'));self.assertLess(events.index('sync'),events.index('lock-exit'))
