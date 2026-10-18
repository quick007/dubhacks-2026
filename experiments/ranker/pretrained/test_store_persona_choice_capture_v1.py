"""Pure target-free boundary tests; no native evidence or model calls."""
import unittest
import numpy as np
from . import store_persona_choice_capture_v1 as capture

class PublicCaptureTests(unittest.TestCase):
    def test_exact_actual_last16_tokens_and_same_pristine_model(self):
        model=object();ids=list(range(24));state={'model':model,'tokens':ids[:-16],'consumed':False}
        self.assertEqual(capture.contract(state,ids,list(range(100,116)),model),ids[-16:])
        with self.assertRaises(ValueError):capture.contract(state,ids,list(range(100,116)),object())
        state['consumed']=True
        with self.assertRaises(ValueError):capture.contract(state,ids,list(range(100,116)),model)
    def test_synthetic_padding_and_colliding_routes_refused(self):
        model=object();ids=list(range(24));state={'model':model,'tokens':ids[:-16],'consumed':False}
        with self.assertRaises(ValueError):capture.contract(state,ids,[100]*16,model)
        with self.assertRaises(ValueError):capture.contract(state,ids[:16],list(range(16)),model)
    def test_target_fields_fail_before_gradient_runtime_import(self):
        item={k:np.zeros((1,16,d),dtype='float32') for k,d in [('x',12288),('h',4096),('y',4096)]}
        item.update(dtypes={k:'mlx.core.bfloat16' for k in ('x','h','y')},routeTokenIds=list(range(16)),targetWeights=[1.]*16)
        with self.assertRaisesRegex(ValueError,'Target-free'):capture.gradient_scores(None,None,item)

if __name__=='__main__':unittest.main()
