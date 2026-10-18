"""CPU protocol/math unit fixtures only, never model-quality evidence."""
import tempfile
import unittest
from pathlib import Path
import torch
from . import store_persona_qwen_optimize_v2 as candidate
from . import store_persona_qwen_fit_v1 as original

class OptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.train=torch.randn(4,4096)
        self.catalog=torch.randn(16,4096)
        self.colors=torch.randn(48,256)
        self.indices=torch.arange(16).repeat_interleave(3)
        self.owned=torch.zeros(4,16)

    def test_raw_head_exactly_matches_original_seeded_math(self):
        cm,pm=candidate.centers(self.train,self.catalog,False)
        head=candidate.OptimizationHead(cm,pm,False);prior=original.NativePersonaHead()
        args=(self.train,self.catalog,self.colors,self.indices,self.owned)
        for a,b in zip(head(*args),prior(*args),strict=True):self.assertTrue(torch.equal(a,b))

    def test_centering_uses_only_supplied_train_and_public_catalog(self):
        cm,pm=candidate.centers(self.train,self.catalog,True)
        self.assertTrue(torch.equal(cm,self.train.mean(0)))
        self.assertTrue(torch.equal(pm,self.catalog.mean(0)))
        head=candidate.OptimizationHead(cm,pm,True)
        unseen=self.train[:1]+10
        actual_context,actual_catalog=head.transform(unseen,self.catalog)
        self.assertTrue(torch.equal(actual_context,torch.nn.functional.normalize(unseen-cm,dim=-1,eps=1e-12)))
        self.assertTrue(torch.equal(actual_catalog,torch.nn.functional.normalize(self.catalog-pm,dim=-1,eps=1e-12)))
        self.assertTrue(torch.equal(head.context_mean,cm))

    def test_transform_is_checkpointed_and_reload_exact(self):
        cm,pm=candidate.centers(self.train,self.catalog,True)
        head=candidate.OptimizationHead(cm,pm,True).eval()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'head.pt'
            torch.save({'version':candidate.VERSION,'config':candidate.CONFIG,
                        'arm':candidate.ARMS[1],'state_dict':head.state_dict()},path)
            loaded=candidate.load_head(path)
            args=(self.train,self.catalog,self.colors,self.indices,self.owned)
            for a,b in zip(head(*args),loaded(*args),strict=True):self.assertTrue(torch.equal(a,b))

    def test_entropy_floor_and_gradient_are_real_math(self):
        pt=torch.full((4,16),1/16);vt=torch.full((4,48),1/48)
        ps=torch.zeros_like(pt,requires_grad=True);vs=torch.zeros_like(vt,requires_grad=True)
        diagnostics=candidate.loss_diagnostics(ps,vs,pt,vt)
        self.assertAlmostEqual(diagnostics['loss'],diagnostics['targetEntropyLossFloor'],places=6)
        self.owned[0,0]=1
        head=candidate.OptimizationHead(*candidate.centers(self.train,self.catalog,True),True)
        products,variants=head(self.train,self.catalog,self.colors,self.indices,self.owned)
        loss=-(pt*products.log_softmax(-1)).sum(-1).mean()-.5*(vt*variants.log_softmax(-1)).sum(-1).mean()
        loss.backward()
        self.assertIsNotNone(head.ownership.grad)
        self.assertTrue(torch.isfinite(head.ownership.grad))
        self.assertNotEqual(float(head.ownership.grad),0)

    def test_pin_drift_and_invalid_features_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.py';path.write_text('original')
            pins={str(path):candidate.base.sha(path)};candidate.validate_pins(pins)
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Immutable'):candidate.validate_pins(pins)
        for value in (torch.zeros(4,8),torch.full((4,4096),float('nan'))):
            with self.assertRaises(ValueError):candidate.checked_matrix(value)

    def test_exact_schedules_and_unique_output_protected(self):
        self.assertEqual([(a['centered'],a['steps']) for a in candidate.ARMS],[(False,2500),(True,250),(True,1000)])
        with tempfile.TemporaryDirectory() as directory:
            plan=Path(directory)/'plan.json'
            candidate.base.dump(plan,{'version':candidate.VERSION,'config':candidate.CONFIG})
            with self.assertRaises(FileExistsError):candidate.run(plan,candidate.base.sha(plan),directory)

if __name__=='__main__':unittest.main()
