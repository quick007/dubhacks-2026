"""CPU-only query gate contract tests, no native extraction or training."""
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
import torch
from . import store_query_intent_gate_v1 as gate

class Tests(unittest.TestCase):
    def test_runtime_text_excludes_persona_history_and_keeps_query_budget_guards(self):
        p={'query':'something useful','itemBudget':60,'hardRequirements':[{'attribute':'color','value':'Navy'}],
           'persona':{'background':['Reads in a chair'],'colorPreferences':['blue']},'publicHistory':[]}
        q=deepcopy(p);q['persona']['background']=['Travels on trains'];q['persona']['colorPreferences']=['green']
        self.assertEqual(gate.runtime_text(p),gate.runtime_text(q))
        text=gate.runtime_text(p);self.assertIn('something useful',text);self.assertIn('60',text);self.assertIn('Navy',text)
        self.assertNotIn('Reads in a chair',text);self.assertNotIn('background',text)
    def test_exact_v5_ranking_wrapper_retains_actual_intent_in_preferences(self):
        request={'query':'a mug for taking a drink','itemBudget':None,'hardRequirements':[]}
        value=json.loads(gate.runtime_text(request).split('\n',1)[1])['request']
        self.assertEqual(value['query'],'Choose a product for my stated use.');self.assertEqual(value['preferences'],[request['query']])
    def test_linear_contract_and_probability_requires_real_query_shape(self):
        head=gate.QueryIntentHead();self.assertEqual(sum(v.numel() for v in head.parameters()),4097)
        x=torch.zeros(4096);x[0]=1
        self.assertTrue(0<gate.specific_probability(head,x)<1)
        for invalid in (torch.zeros(4096),torch.ones(3),torch.full((4096,),float('nan'))):
            with self.assertRaises(ValueError):gate.specific_probability(head,invalid)
    def test_balanced_bce_equalizes_class_total_weight_and_has_real_gradients(self):
        logits=torch.zeros(6,requires_grad=True);labels=torch.tensor([0.,0.,1.,1.,1.,1.])
        loss=gate.class_balanced_bce(logits,labels);loss.backward()
        self.assertAlmostEqual(float(loss.detach()),.693147,places=5)
        self.assertTrue(torch.allclose(logits.grad[:2],torch.tensor([.125,.125])))
        self.assertTrue(torch.allclose(logits.grad[2:],torch.tensor([-.0625]*4)))
        with self.assertRaises(ValueError):gate.class_balanced_bce(torch.zeros(3),torch.zeros(3))
    def test_pinned_loader_return_metadata_and_source_manifest_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'head.pt';head=gate.QueryIntentHead()
            metadata={'recipe':gate.RECIPE,'dimensions':4096,'specificClass':1,'featureContract':'exact-v5-runtime-query4096-L2','sourceManifestSha256':'frozen-source'}
            torch.save({'version':gate.VERSION,'state':head.state_dict(),'metadata':metadata},path)
            loaded,receipt=gate.load_intent_head(path,gate.sha(path),expected_source_manifest_sha256='frozen-source')
            self.assertEqual(receipt,metadata);self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            with self.assertRaises(ValueError):gate.load_intent_head(path,'wrong')
            with self.assertRaises(ValueError):gate.load_intent_head(path,gate.sha(path),expected_source_manifest_sha256='wrong')
    def test_recipe_is_fixed_no_validation_selection(self):
        self.assertEqual(gate.RECIPE['steps'],250);self.assertEqual(gate.RECIPE['seed'],20261019)
        self.assertEqual(gate.RECIPE['hiddenLayers'],0);self.assertIn('no evaluation',gate.RECIPE['selection'])
if __name__=='__main__':unittest.main()
