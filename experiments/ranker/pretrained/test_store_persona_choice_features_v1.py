"""CPU contract fixtures only; no native parity or model quality claims."""
import unittest
import numpy as np
from . import store_persona_choice_features_v1 as f

class ChoiceContractTests(unittest.TestCase):
    def row(self):
        ids=[4,5,25]
        return {'caseId':'unit-only','tokenIds':ids,'tokenIdsSha256':f.token_hash(ids),
                'routeTokenIds':list(range(100,116)),'targetWeights':[.5,.5]+[0.]*14}
    def test_actual_last_token_boundary_no_teacher_append(self):
        row=self.row();c=f.choice_contract(row)
        self.assertEqual(c['prefixTokenIds'],[4,5]);self.assertEqual(c['suffixTokenIds'],[25])
        self.assertEqual(c['inputTargetTokens'],0);self.assertEqual(c['supervisedPositions'],1)
        self.assertEqual(row['tokenIds'],[4,5,25])
    def test_invalid_routes_labels_hashes_caps_fail(self):
        for key,value in [('routeTokenIds',[100]*16),('targetWeights',[1.]*16),('targetWeights',[float('nan')]+[0.]*15),('tokenIdsSha256','wrong')]:
            row=self.row();row[key]=value
            with self.assertRaises(ValueError):f.choice_contract(row)
        with self.assertRaises(ValueError):f.choice_contract(self.row(),cap=2)
    def test_two_valid_primary_products_preserved(self):
        ids=[str(i) for i in range(16)];grades={i:0 for i in ids};grades['1']=3;grades['7']=3;grades['4']=2
        values=f.max_grade_targets(grades,ids)
        self.assertEqual(values[1],.5);self.assertEqual(values[7],.5);self.assertEqual(values[4],0)
        self.assertEqual(sum(values),1)
    def test_complete_feature_shape_and_native_dtypes(self):
        item={k:np.zeros((1,1,d),dtype='float32') for k,d in [('x',12288),('h',4096),('y',4096)]}
        item.update(dtypes={k:'mlx.core.bfloat16' for k in f.FEATURE_KEYS},routeTokenIds=list(range(16)),targetWeights=[1.]+[0.]*15)
        f.validate_features(item)
        item['x']=np.zeros((1,2,12288),dtype='float32')
        with self.assertRaises(ValueError):f.validate_features(item)
    def test_ungated_feature_api_refuses_before_mlx_import(self):
        with self.assertRaisesRegex(ValueError,'gate'):f.feature_loss(None,None,{})
    def test_fixed_schedule_has_each_row_each_epoch(self):
        actual=f.fixed_orders(896,2,19083)
        self.assertEqual(actual,f.fixed_orders(896,2,19083))
        self.assertEqual(len(actual),1792)
        self.assertEqual(sorted(actual[:896]),list(range(896)))
        self.assertEqual(sorted(actual[896:]),list(range(896)))
        self.assertNotEqual(actual[:896],actual[896:])

if __name__=='__main__':unittest.main()
