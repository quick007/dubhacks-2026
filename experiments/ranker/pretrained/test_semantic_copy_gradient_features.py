"""CPU-only contracts; these tests do not establish native numerical parity."""
import copy
import subprocess
import sys
import unittest

import numpy as np

from .semantic_copy_gradient_features import (
    build_native_contract, compare_host_results, feature_loss,
    unpack_auxiliary_value, validate_feature_contract)


class ShapeOnly:
    def __init__(self, shape):
        self.shape = shape


class StubTokenizer:
    eos_token_ids = {99}
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, return_dict=False,
                              add_generation_prompt=True, enable_thinking=False)
        return [10] + [20 + len(m['content']) for m in messages] + [11, 12]

    def encode(self, text, add_special_tokens=False):
        return [99] if text == '<|im_end|>' else [100 + ord(c) for c in text]


class GradientFeatureContracts(unittest.TestCase):
    def fixture(self):
        return ({k: ShapeOnly((1, 39, width)) for k, width in (('x',12288),('h',4096),('y',4096))},
                {'x':'mlx.core.bfloat16','h':'mlx.core.bfloat16','y':'mlx.core.float16'})

    def token_fixture(self):
        common = {'role':'user','content':'shared facts'}
        state = {'messages':[{'role':'system','content':'rank'},common],
                 'tokens':[10,24,32], 'consumed':False}
        field = {'messages':[common,{'role':'user','content':'title'}],
                 'targetValue':'new', 'maxTokens':4}
        return state, field

    def test_import_is_runtime_free(self):
        code = ('import sys; import experiments.ranker.pretrained.semantic_copy_gradient_features; '
                'assert not any(k == "mlx" or k.startswith("mlx.") for k in sys.modules)')
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)

    def test_full_shape_and_individual_dtypes(self):
        arrays, dtypes = self.fixture()
        receipt = validate_feature_contract(arrays,dtypes,35,[1,2,3,99])
        self.assertEqual(receipt['shapes']['x'],[1,39,12288])
        self.assertEqual(receipt['dtypes']['y'],'mlx.core.float16')
        for patch in ({'x':ShapeOnly((39,12288))}, {'y':ShapeOnly((1,4,4096))},
                      {'h':ShapeOnly((1,39,4095))}, {'x':ShapeOnly((1,257,12288))}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_feature_contract({**arrays,**patch},dtypes,35,[1,2,3,99])
        with self.assertRaisesRegex(ValueError,'dtype'):
            validate_feature_contract(arrays,{**dtypes,'x':'float32'},35,[1,2,3,99])
        for start,targets in ((True,[1]),(39,[1]),(35,[1,2]),(35,[1,2,3,-1]),(35,[1,2,3,248320])):
            with self.subTest(start=start,targets=targets), self.assertRaises(ValueError):
                validate_feature_contract(arrays,dtypes,start,targets)

    def test_only_transformed_auxiliary_tuple_is_accepted(self):
        arrays,_ = self.fixture();loss = ShapeOnly(())
        self.assertIs(unpack_auxiliary_value((loss,arrays))[1],arrays)
        for value in ((loss,),[loss,arrays],(ShapeOnly((1,)),arrays),(loss,{'x':arrays['x']})):
            with self.subTest(value=value), self.assertRaises(ValueError):
                unpack_auxiliary_value(value)

    def test_native_contract_exact_common_prefix_and_causal_targets(self):
        state,field = self.token_fixture();before = copy.deepcopy((state,field))
        contract = build_native_contract(StubTokenizer(),state,field)
        self.assertEqual((state,field),before)
        self.assertEqual(contract['teacherInputTokenIds'][:len(state['tokens'])],state['tokens'])
        self.assertEqual(contract['suffixTokenIds'][contract['predictionStartInBranch']+1:],
                         contract['targetTokenIds'][:-1])
        self.assertEqual(len(contract['suffixTokenIds'])-contract['predictionStartInBranch'],4)
        # Metadata does not change the model-input receipt.
        other = build_native_contract(StubTokenizer(),state,{**field,'caseId':'other'})
        self.assertEqual(contract['modelInputSha256'],other['modelInputSha256'])
        with self.assertRaisesRegex(ValueError,'common evidence'):
            build_native_contract(StubTokenizer(),state,{**field,'messages':[
                {'role':'user','content':'changed'},field['messages'][-1]]})
        with self.assertRaisesRegex(ValueError,'unconsumed'):
            build_native_contract(StubTokenizer(),{**state,'consumed':True},field)
        with self.assertRaisesRegex(ValueError,'exact source token'):
            build_native_contract(StubTokenizer(),{**state,'tokens':[1]},field)
        with self.assertRaisesRegex(ValueError,'context exceeds'):
            build_native_contract(StubTokenizer(),state,field,maximum_context=5)
        with self.assertRaisesRegex(ValueError,'1-256'):
            build_native_contract(StubTokenizer(),{**state,'tokens':[10]},
                {**field,'targetValue':'x'*255,'maxTokens':256})

    def result(self,loss=1.,gradient=2.,dtype='mlx.core.float32'):
        def tensor(value):
            a = np.array(value,dtype=np.float32)
            return a,{'shape':list(a.shape),'dtype':dtype}
        return {'loss':tensor(loss),'gradients':{'lora_a':tensor([gradient]),'lora_b':tensor([3.])}}

    def test_exact_gate_rejects_tiny_differences_dtype_and_nonfinite(self):
        left = self.result()
        self.assertTrue(compare_host_results(left,self.result())['exact'])
        mismatch = compare_host_results(left,self.result(gradient=2.0000002))
        self.assertFalse(mismatch['exact'])
        self.assertGreater(mismatch['gradients']['lora_a']['maxAbsDifference'],0)
        self.assertFalse(compare_host_results(left,self.result(dtype='mlx.core.float16'))['exact'])
        right = self.result();right['gradients'].pop('lora_b')
        with self.assertRaisesRegex(ValueError,'gradients'):compare_host_results(left,right)
        with self.assertRaises(FloatingPointError):compare_host_results(left,self.result(loss=float('nan')))

    def test_unpassed_features_cannot_enter_public_loss(self):
        for item in ({},{'gradientFeatureGatePassed':False}):
            with self.assertRaisesRegex(ValueError,'has not passed'):feature_loss(None,None,item)


if __name__ == '__main__':
    unittest.main()
