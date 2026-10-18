"""CPU-only action/name semantics; no native model claim."""
from copy import deepcopy
import unittest
from .store_catalog_v1 import load_snapshot
from .store_function_adaptation import rank
from .store_rank_session import action_bindings, exact_name_order

class StoreRankSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.snapshot=load_snapshot();cls.products=cls.snapshot['inputs']['products']
    def request(self):return {'query':'Trail mug','preferences':[],'itemBudget':None,'hardRequirements':[]}
    def test_exact_name_after_hard_guards_preserves_scores(self):
        request=self.request();scores=list(map(float,range(16)));guarded=rank(scores,request,self.products)
        before=deepcopy(guarded);order,policy=exact_name_order(request,self.products,guarded)
        self.assertEqual(order[0],0);self.assertTrue(policy['applied']);self.assertEqual(guarded,before)
        request['itemBudget']=23;guarded=rank(scores,request,self.products);order,policy=exact_name_order(request,self.products,guarded)
        self.assertNotIn(0,order);self.assertFalse(policy['applied'])
    def test_inexact_query_does_not_override_learned_order(self):
        request=self.request();request['query']='Something like a Trail mug'
        guarded=rank(list(map(float,range(16))),request,self.products)
        order,policy=exact_name_order(request,self.products,guarded)
        self.assertEqual(order,guarded['eligibleRank']);self.assertFalse(policy['applied'])
    def test_color_constraint_binds_actual_matching_variant(self):
        request=self.request();request['hardRequirements']=[{'attribute':'color','value':'Cobalt'}]
        card=action_bindings(request,[0],self.snapshot)[0]
        self.assertEqual(card['facts']['selectedColor'],'Cobalt')
        route=self.snapshot['externalMapping']['products'][0]['colorVariants'][1]
        color_index=next(i for i,c in enumerate(self.products[0]['colorVariants']) if c['colorName']=='Cobalt')
        route=self.snapshot['externalMapping']['products'][0]['colorVariants'][color_index]
        self.assertEqual(card['action']['variantId'],route['variantId']);self.assertEqual(card['action']['image'],route['images'][0])
    def test_conflicting_colors_fail_closed(self):
        request=self.request();request['hardRequirements']=[{'attribute':'color','value':v} for v in ['Cobalt','Sage']]
        with self.assertRaises(ValueError):action_bindings(request,[0],self.snapshot)
    def test_no_padding_and_no_snapshot_alias(self):
        request=self.request();card=action_bindings(request,[0],self.snapshot)[0]
        card['action']['image']['src']='/mutated'
        self.assertNotEqual(action_bindings(request,[0],self.snapshot)[0]['action']['image']['src'],'/mutated')
        self.assertEqual(action_bindings(request,[],self.snapshot),[])
if __name__=='__main__':unittest.main()
