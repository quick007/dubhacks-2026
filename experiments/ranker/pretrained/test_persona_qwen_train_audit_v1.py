"""Fabricated TRAIN target and geometry checks; no actual labels or model."""
from copy import deepcopy
import unittest

from . import audit_persona_qwen_train_v1 as diagnosis
from .audit_persona_qwen_candidate_v1 import AuditError


class TrainDiagnosisTests(unittest.TestCase):
    def fixture(self):
        products=[{'productId':f'item{i}'} for i in range(16)]
        catalog={'products':products,'variants':[{'productIndex':i,'variantId':f'variant{i}'} for i in range(16)]}
        profiles=[{'profileId':f'p{i}','request':{'publicHistory':[{'eventType':'purchase','productId':'item15'}]}} for i in range(320)]
        labels=[]
        for i in range(320):
            grades={f'item{j}':0 for j in range(16)};grades['item0']=3;grades['item1']=3 if i<16 else 2
            labels.append({'profileId':f'p{i}','relevanceGrades':grades,'alreadyOwnedProductIds':['item15'],'preferredVariantIds':['variant0','variant1']})
        return profiles,labels,catalog

    def test_multigrade3_and_label_order_are_preserved(self):
        import torch
        profiles,labels,catalog=self.fixture()
        pt,vt,_=diagnosis.independent_targets(profiles,labels,catalog)
        self.assertEqual(pt[0,:2].tolist(),[.5,.5])
        self.assertTrue(torch.equal(pt,vt))
        reverse_pt,reverse_vt,_=diagnosis.independent_targets(profiles,list(reversed(labels)),catalog)
        self.assertTrue(torch.equal(pt,reverse_pt) and torch.equal(vt,reverse_vt))
        self.assertTrue(torch.allclose(pt.sum(1),torch.ones(320)))

    def test_duplicate_join_and_ownership_mismatch_fail(self):
        profiles,labels,catalog=self.fixture();bad=deepcopy(labels);bad[-1]=deepcopy(bad[0])
        with self.assertRaises(AuditError):diagnosis.independent_targets(profiles,bad,catalog)
        bad=deepcopy(labels);bad[0]['alreadyOwnedProductIds']=[]
        with self.assertRaises(AuditError):diagnosis.independent_targets(profiles,bad,catalog)

    def test_geometry_uses_shared_positive_sets(self):
        import torch
        x=torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
        value=diagnosis.geometry(x,[[1,0],[1,1],[0,1]])
        self.assertEqual(value['withinPrimaryClassCosines']['count'],4)
        self.assertEqual(value['betweenPrimaryClassCosines']['count'],2)
        self.assertEqual(value['offDiagonalCosines']['count'],6)
        self.assertAlmostEqual(value['offDiagonalCosines']['mean'],1/3)


if __name__=='__main__':unittest.main()
