"""CPU last16 real-token contracts, no native kernel/quality evidence."""
import unittest
import numpy as np
from . import store_persona_choice_features_v2 as f

class ChoiceLast16Tests(unittest.TestCase):
    def row(self):
        ids=list(range(24))
        return {'caseId':'unit-only','tokenIds':ids,'tokenIdsSha256':f.token_hash(ids),
                'routeTokenIds':list(range(100,116)),'targetWeights':[.5,.5]+[0.]*14}
    def test_actual_last16_boundary_only_final_position_supervised(self):
        row=self.row();c=f.choice_contract(row)
        self.assertEqual(c['prefixTokenIds'],list(range(8)))
        self.assertEqual(c['suffixTokenIds'],list(range(8,24)))
        self.assertEqual(c['prefixTokenIds']+c['suffixTokenIds'],row['tokenIds'])
        self.assertEqual(c['supervisedPositions'],1);self.assertEqual(c['inputTargetTokens'],0)
        self.assertEqual(row['tokenIds'],list(range(24)))
    def test_no_synthetic_padding_short_prompts_refused(self):
        row=self.row();row['tokenIds']=list(range(16));row['tokenIdsSha256']=f.token_hash(row['tokenIds'])
        with self.assertRaises(ValueError):f.choice_contract(row)
    def test_invalid_routes_soft_targets_and_checksum_refused(self):
        for key,value in [('routeTokenIds',[100]*16),('targetWeights',[1.]*16),('targetWeights',[float('nan')]+[0.]*15),('tokenIdsSha256','wrong')]:
            row=self.row();row[key]=value
            with self.assertRaises(ValueError):f.choice_contract(row)
    def test_full16_feature_shapes_preserved_until_adapter_and_norm(self):
        item={k:np.zeros((1,16,d),dtype='float32') for k,d in [('x',12288),('h',4096),('y',4096)]}
        item.update(dtypes={k:'mlx.core.bfloat16' for k in f.FEATURE_KEYS},routeTokenIds=list(range(16)),targetWeights=[1.]+[0.]*15)
        f.validate_features(item)
        self.assertEqual(sum(item[k].nbytes for k in f.FEATURE_KEYS),1310720)
        item['x']=np.zeros((1,1,12288),dtype='float32')
        with self.assertRaises(ValueError):f.validate_features(item)
    def test_multimax_grade_distribution(self):
        ids=[str(i) for i in range(16)];grades={k:0 for k in ids};grades['0']=3;grades['8']=3
        self.assertEqual(f.max_grade_targets(grades,ids),[.5]+[0.]*7+[.5]+[0.]*7)
    def test_ungated_feature_loss_refuses_without_native_import(self):
        with self.assertRaisesRegex(ValueError,'gate'):f.feature_loss(None,None,{})

if __name__=='__main__':unittest.main()
