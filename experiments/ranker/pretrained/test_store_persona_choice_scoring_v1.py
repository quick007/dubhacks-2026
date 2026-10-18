"""CPU public-cache whitelist tests; no native math parity claims."""
import unittest
import numpy as np
from . import store_persona_choice_scoring_v1 as scoring

class PublicFeatureTests(unittest.TestCase):
    def item(self):
        return {**{k:np.zeros((1,16,d),dtype='float32') for k,d in [('x',12288),('h',4096),('y',4096)]},
                'dtypes':{k:'mlx.core.bfloat16' for k in ('x','h','y')},'routeTokenIds':list(range(16))}
    def test_explicit_projection_strips_every_training_label_and_identity(self):
        value={**self.item(),'targetWeights':[1.]*16,'targets':[1],'profileId':'private-map','caseId':'case','relevanceGrades':{'p':3}}
        public=scoring.public_item(value);scoring.validate(public)
        self.assertEqual(set(public),{'x','h','y','dtypes','routeTokenIds'})
        self.assertFalse(set(public)&scoring.FORBIDDEN)
    def test_target_fields_refused_before_native_call(self):
        for key in scoring.FORBIDDEN:
            item={**self.item(),key:'forbidden'}
            with self.assertRaisesRegex(ValueError,'Target-free'):scoring.score(None,None,item)
    def test_full16_shape_and_unique_routes_required(self):
        item=self.item();item['x']=item['x'][:,:1]
        with self.assertRaises(ValueError):scoring.validate(item)
        item=self.item();item['routeTokenIds']=[0]*16
        with self.assertRaises(ValueError):scoring.validate(item)

if __name__=='__main__':unittest.main()
