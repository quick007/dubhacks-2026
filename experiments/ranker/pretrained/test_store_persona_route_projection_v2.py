"""CPU packed-affine protocol fixtures, never native projection evidence."""
from copy import deepcopy
from types import SimpleNamespace as NS
import subprocess
import sys
import unittest

import numpy as np

from . import store_persona_route_projection_v2 as candidate


class Dtype(str):
    def __str__(self):
        return 'mlx.core.' + super().__str__()


class CPUBackend:
    uint32 = np.uint32
    float32 = Dtype('float32')
    float16 = Dtype('float16')
    bfloat16 = Dtype('float64')  # Deliberately distinct key, unused in these fixtures.
    array = staticmethod(np.array)
    take = staticmethod(np.take)
    eval = staticmethod(lambda *args: None)
    synchronize = staticmethod(lambda: None)
    calls = []

    @classmethod
    def quantized_matmul(cls, x, weight, *, scales, biases, transpose, group_size, bits, mode):
        cls.calls.append({'rows': weight.shape[0], 'transpose': transpose, 'group_size': group_size,
            'bits': bits, 'mode': mode, 'input_dtype': str(x.dtype)})
        if mode != 'affine' or not transpose:
            raise AssertionError('Exact original affine transpose protocol required')
        count = weight.shape[1] * 32 // bits
        # Independent packed-word bit extraction, not a selected-head stub.
        indices = np.arange(count) * bits
        words = weight[:, indices // 32]
        quantized = ((words >> (indices % 32).astype(np.uint32)) & ((1 << bits) - 1)).astype(scales.dtype)
        groups = np.arange(count) // group_size
        dense = quantized * scales[:, groups] + biases[:, groups]
        return x @ dense.T


class QuantizedLinear(dict):
    group_size = 4
    bits = 4
    mode = 'affine'

    def __call__(self, x):
        out = CPUBackend.quantized_matmul(x, self['weight'], scales=self['scales'], biases=self['biases'],
            transpose=True, group_size=self.group_size, bits=self.bits, mode=self.mode)
        return out + self['bias'] if 'bias' in self else out


def fixture(bias=True, dtype=np.float32):
    rows, inputs = 32, 8
    quantized = ((np.arange(rows * inputs).reshape(rows, inputs) * 7 + np.arange(rows)[:, None]) % 16).astype(np.uint32)
    packed = np.sum(quantized.astype(np.uint64) << (4 * np.arange(inputs)).astype(np.uint64), axis=1).astype(np.uint32)[:, None]
    head = QuantizedLinear(weight=packed, scales=(np.arange(rows * 2).reshape(rows, 2) % 5 + 1).astype(dtype) / 8,
        biases=(np.arange(rows * 2).reshape(rows, 2) % 3 - 2).astype(dtype) / 4)
    if bias:
        head['bias'] = np.arange(rows, dtype=dtype) / 16
    backbone = NS(mx=CPUBackend, np=np, check_memory=lambda: None,
        model=NS(language_model=NS(args=NS(tie_word_embeddings=False, vocab_size=rows), lm_head=head)))
    routes = [31, 0, 29, 2, 27, 4, 25, 6, 23, 8, 21, 10, 19, 12, 17, 14]
    return backbone, head, routes


class ProjectionTests(unittest.TestCase):
    def test_import_does_not_import_mlx_or_torch(self):
        subprocess.run([sys.executable, '-B', '-c',
            'import sys;import experiments.ranker.pretrained.store_persona_route_projection_v2;'
            'assert not any(n=="mlx" or n.startswith("mlx.") or n=="torch" for n in sys.modules)'], check=True)

    def test_selected_packed_rows_equal_full_head_gather_with_and_without_ordinary_bias(self):
        for bias in (False, True):
            for dtype in (np.float16, np.float32):
                bb, head, routes = fixture(bias, dtype); before = deepcopy(head)
                projector = candidate.SelectedRouteProjection.prepare(bb, routes, _quantized_class=QuantizedLinear)
                hidden = np.arange(8, dtype=dtype)[None, :] / 4
                expected = head(hidden).astype(np.float32).reshape(-1)[routes]
                CPUBackend.calls.clear(); actual = projector.project(hidden)
                self.assertTrue(np.array_equal(expected, actual))
                self.assertEqual(CPUBackend.calls[-1]['rows'], 16)
                self.assertEqual(CPUBackend.calls[-1]['input_dtype'], str(dtype().dtype))
                self.assertEqual(CPUBackend.calls[-1]['group_size'], 4)
                for key in before:
                    self.assertTrue(np.array_equal(head[key], before[key]))
                    self.assertTrue(np.array_equal(projector.selected[key], head[key][routes]))
                    self.assertEqual(projector.selected[key].dtype, head[key].dtype)
                self.assertTrue(projector.assert_current(full=True))

    def test_bfloat16_receipt_uses_exact_float32_host_values_without_changing_native_rows(self):
        import hashlib
        class BF16:
            dtype = 'mlx.core.bfloat16'
            shape = (16, 2)
            def __init__(self): self.casts = []
            def __array__(self, *args, **kwargs): raise AssertionError('RawBF16 NumPy conversion unsupported')
            def astype(self, dtype):
                self.casts.append(dtype)
                return np.arange(32, dtype=np.float32).reshape(16, 2) / 8
        tensor = BF16(); proof = candidate._tensor_receipt(tensor, mx=CPUBackend)
        host = np.arange(32, dtype=np.float32).reshape(16, 2) / 8
        expected = hashlib.sha256(b'mlx.core.bfloat16' + str((16,2)).encode() + host.tobytes()).hexdigest()
        self.assertEqual(proof, {'shape':[16,2], 'dtype':'mlx.core.bfloat16', 'sha256':expected})
        self.assertEqual(tensor.casts, [CPUBackend.float32]); self.assertEqual(tensor.dtype, 'mlx.core.bfloat16')

    def test_route_order_is_not_sorted_and_public_route_validation_is_strict(self):
        bb, _, routes = fixture()
        for bad in (routes[:-1], [0] * 16, [True] + routes[1:], [32] + routes[1:]):
            with self.assertRaises(ValueError):
                candidate.SelectedRouteProjection.prepare(bb, bad, _quantized_class=QuantizedLinear)

    def test_tied_dense_wrapped_modes_and_bad_metadata_refused(self):
        for change in ('tie', 'wrapped', 'mode', 'bits', 'scales', 'bias', 'dtype'):
            bb, head, routes = fixture()
            if change == 'tie': bb.model.language_model.args.tie_word_embeddings = True
            elif change == 'wrapped': bb.model.language_model.lm_head = NS(linear=head)
            elif change == 'mode': head.mode = 'mxfp8'
            elif change == 'bits': head.bits = 7
            elif change == 'scales': head['scales'] = np.zeros((32, 1), np.float32)
            elif change == 'bias': head['bias'] = np.zeros(31, np.float32)
            elif change == 'dtype': head['weight'] = head['weight'].astype(np.float32)
            with self.assertRaises(ValueError):
                candidate.SelectedRouteProjection.prepare(bb, routes, _quantized_class=QuantizedLinear)

    def test_source_replacement_and_inplace_source_or_selected_corruption_refused(self):
        for change in ('replace', 'source-content', 'selected-content', 'config'):
            bb, head, routes = fixture(); projector = candidate.SelectedRouteProjection.prepare(bb, routes, _quantized_class=QuantizedLinear)
            if change == 'replace': head['scales'] = head['scales'].copy()
            elif change == 'source-content': head['scales'][routes[0], 0] += 1
            elif change == 'selected-content': projector.selected['scales'][0, 0] += 1
            else: head.group_size = 8
            with self.assertRaises(ValueError): projector.assert_current(full=True)

    def test_score_keeps_the_original_adapter_residual_norm_math_and_native_comparison_hook(self):
        bb, head, routes = fixture(); projector = candidate.SelectedRouteProjection.prepare(bb, routes, _quantized_class=QuantizedLinear)
        # Production dimensions are validated by the untouched ordinary validator.
        # Use width4096 LM inputs and zero packed rows for a full actual shape case.
        head['weight'] = np.zeros((32, 512), np.uint32); head['scales'] = np.ones((32, 1024), np.float32)
        head['biases'] = np.zeros((32, 1024), np.float32)
        projector = candidate.SelectedRouteProjection.prepare(bb, routes, _quantized_class=QuantizedLinear)
        bb.model.model = NS(norm=lambda hidden: hidden / np.sqrt(np.mean(hidden * hidden, axis=-1, keepdims=True) + 1e-6))
        item = {'x': np.ones((1, 16, 12288), np.float32), 'h': np.ones((1, 16, 4096), np.float32),
            'y': np.ones((1, 16, 4096), np.float32) * 2,
            'dtypes': {k: 'mlx.core.float32' for k in ('x', 'h', 'y')}, 'routeTokenIds': routes}
        adapter = NS(scale=20, dropout=lambda x: x, lora_a=np.ones((12288, 4), np.float32) * .001,
            lora_b=np.ones((4, 4096), np.float32) * .001)
        result = candidate.native_comparison(bb, adapter, item, projector, order=('selected', 'ordinary'))
        self.assertTrue(result['exactNativeProjectionGatePassed'])
        self.assertEqual(result['bodyForwards'], 0)
        self.assertIs(candidate.ordinary.score, candidate.REFERENCE_SCORE)
        self.assertTrue(np.array_equal(projector.score(bb, adapter, item), candidate.REFERENCE_SCORE(bb, adapter, item)))
        with self.assertRaises(ValueError): projector.score(bb, adapter, {**item, 'targetWeights': [0]})
        with self.assertRaises(ValueError): projector.score(bb, adapter, {**item, 'routeTokenIds': routes[::-1]})

    def test_exact_comparison_preserves_all_magnitudes_and_does_not_admit_order_only(self):
        a = np.arange(16, dtype=np.float32); b = a.copy(); b[-1] += np.float32(.125)
        proof = candidate.compare_vectors(a, b)
        self.assertTrue(proof['all16RawOrderEqual']); self.assertFalse(proof['exactNativeProjectionGatePassed'])
        self.assertEqual(proof['maximumAbsoluteDifference'], .125)
        self.assertEqual(proof['referenceRouteScores'], a.tolist())
        signed = np.zeros(16, np.float32); negative = signed.copy(); negative[0] = -0.
        self.assertFalse(candidate.compare_vectors(signed, negative)['bitwiseEqual'])
        with self.assertRaises(ValueError): candidate.compare_vectors(a, [float('nan')] * 16)


if __name__ == '__main__': unittest.main()
