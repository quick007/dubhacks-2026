"""Selected affine-quantized LM rows; explicit preparation, no model on import.

Only the frozen output projection changes shape. The final-MLP adapter, residual,
normalization and input dtype retain the ordinary scorer's arithmetic. Native
exactness is unknown until the caller runs native_comparison under its GPU lock.
"""
from copy import deepcopy
import hashlib
import inspect
import time
from pathlib import Path

from . import store_persona_choice_scoring_v1 as ordinary

VERSION = 'persona-selected-route-projection.v2'
REFERENCE_SCORE = ordinary.score


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _routes(values, vocabulary):
    if (type(values) is not list or len(values) != 16 or len(set(values)) != 16 or
            any(type(x) is not int or not 0 <= x < vocabulary for x in values)):
        raise ValueError('Exactly16 distinct in-vocabulary public route IDs required')
    return tuple(values)


def _head(backbone, quantized_class):
    lm = backbone.model.language_model
    if bool(lm.args.tie_word_embeddings):
        raise ValueError('Separate untied LM head required; tied embedding is not silently replaced')
    head = lm.lm_head
    if type(head) is not quantized_class:
        raise ValueError('Exact QuantizedLinear required; adapted/wrapped/dense LM heads are unsupported')
    if head.mode != 'affine':
        raise ValueError('Only affine quantization is supported; never substitute another quantizer')
    if type(head.bits) is not int or head.bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError('Supported affine packed bit width required')
    if type(head.group_size) is not int or head.group_size <= 0:
        raise ValueError('Explicit positive quantization group size required')
    weight, scales, biases = head['weight'], head['scales'], head.get('biases')
    if len(weight.shape) != 2 or len(scales.shape) != 2 or biases is None:
        raise ValueError('Affine packed weight/scales/quantization biases required')
    rows, packed = weight.shape
    if packed * 32 % head.bits:
        raise ValueError('Packed weight shape does not encode complete input dimensions')
    inputs = packed * 32 // head.bits
    if (inputs % head.group_size or tuple(scales.shape) != (rows, inputs // head.group_size) or
            tuple(biases.shape) != tuple(scales.shape) or weight.dtype != backbone.mx.uint32 or
            scales.dtype != biases.dtype or scales.dtype not in (backbone.mx.bfloat16, backbone.mx.float16, backbone.mx.float32)):
        raise ValueError('Packed weight and affine group metadata shapes/dtypes disagree')
    if getattr(lm.args, 'vocab_size', rows) != rows:
        raise ValueError('LM vocabulary and output row count differ')
    if 'bias' in head and tuple(head['bias'].shape) != (rows,):
        raise ValueError('Ordinary additive bias must have one value per output row')
    fields = {'weight': weight, 'scales': scales, 'biases': biases}
    if 'bias' in head:
        fields['bias'] = head['bias']
    return head, fields, {'bits': head.bits, 'groupSize': head.group_size,
        'mode': head.mode, 'inputDimensions': inputs, 'vocabularyRows': rows,
        'ordinaryBias': 'bias' in fields}


def _tensor_receipt(value, *, mx):
    import numpy as np
    host_value = value.astype(mx.float32) if str(value.dtype) == 'mlx.core.bfloat16' else value
    array = np.ascontiguousarray(np.array(host_value))
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
        'sha256': hashlib.sha256(str(value.dtype).encode() + str(tuple(value.shape)).encode() + array.tobytes()).hexdigest()}


class SelectedRouteProjection:
    """One bound frozen head and route order; stores weights, never responses."""
    @classmethod
    def prepare(cls, backbone, route_ids, *, _quantized_class=None):
        started = time.perf_counter()
        fixture = _quantized_class is not None
        if _quantized_class is None:
            from mlx.nn import QuantizedLinear
            _quantized_class = QuantizedLinear
        head, fields, specification = _head(backbone, _quantized_class)
        routes = _routes(route_ids, specification['vocabularyRows'])
        mx = backbone.mx; index = mx.array(list(routes))
        selected = {key: mx.take(value, index, axis=0) for key, value in fields.items()}
        mx.eval(*selected.values()); mx.synchronize(); backbone.check_memory()
        obj = cls(); obj.backbone = backbone; obj.model = backbone.model; obj.head = head; obj.quantized_class = _quantized_class
        obj.fields = fields; obj.selected = selected; obj.specification = specification; obj.routes = routes
        obj.selected_receipts = {k: _tensor_receipt(v, mx=mx) for k, v in selected.items()}
        obj.receipt = {'version': VERSION, 'sourceSha256': _sha(__file__),
            'quantizedClassSourceSha256': _sha(inspect.getfile(_quantized_class)),
            'routeTokenIds': list(routes), 'specification': deepcopy(specification),
            'selectedTensorReceipts': deepcopy(obj.selected_receipts),
            'preparationMs': (time.perf_counter() - started) * 1000,
            'headRequantizedOrDequantized': False, 'modelOrHeadReplaced': False,
            'queryOrResponseFeaturesCached': False, 'CPUFixture': fixture,
            'nativeExactnessMeasured': False,
            'sourceInvariance': 'identity/config checks per call; selected source/cache content hashes on explicit full check; caller retains full927 before/after proof'}
        return obj

    def assert_current(self, *, full=False):
        if self.backbone.model is not self.model:
            raise ValueError('Prepared projection belongs to a different resident model')
        head, fields, specification = _head(self.backbone, self.quantized_class)
        if (head is not self.head or specification != self.specification or set(fields) != set(self.fields) or
                any(fields[k] is not self.fields[k] for k in fields)):
            raise ValueError('Bound frozen output head/parameter identity or quantization configuration changed')
        if full:
            mx = self.backbone.mx; index = mx.array(list(self.routes))
            current = {key: mx.take(value, index, axis=0) for key, value in fields.items()}
            mx.eval(*current.values(), *self.selected.values()); mx.synchronize()
            if ({k: _tensor_receipt(v, mx=mx) for k, v in current.items()} != self.selected_receipts or
                    {k: _tensor_receipt(v, mx=mx) for k, v in self.selected.items()} != self.selected_receipts):
                raise ValueError('Selected frozen source rows or prepared projection rows changed')
        return True

    def project(self, hidden):
        self.assert_current()
        if tuple(hidden.shape) != (1, self.specification['inputDimensions']):
            raise ValueError('Exact B1 last-token hidden vector required')
        mx = self.backbone.mx
        scores = mx.quantized_matmul(hidden, self.selected['weight'],
            scales=self.selected['scales'], biases=self.selected['biases'], transpose=True,
            group_size=self.specification['groupSize'], bits=self.specification['bits'], mode=self.specification['mode'])
        if 'bias' in self.selected:
            scores = scores + self.selected['bias']
        return scores.astype(mx.float32).reshape(-1)

    def score(self, backbone, adapter, item):
        ordinary.validate(item)
        if backbone is not self.backbone or tuple(item['routeTokenIds']) != self.routes:
            raise ValueError('Exact bound backbone and public route order required')
        mx = backbone.mx
        dtypes = {str(t): t for t in (mx.bfloat16, mx.float16, mx.float32)}
        x, h, y = (mx.array(item[k], dtype=dtypes[item['dtypes'][k]]) for k in ('x', 'h', 'y'))
        delta = (adapter.scale * ((adapter.dropout(x) @ adapter.lora_a) @ adapter.lora_b)).astype(x.dtype)
        hidden = backbone.model.model.norm(h + (y + delta))
        return self.project(hidden[:, -1, :])


def compare_vectors(reference, candidate):
    """CPU evidence from evaluated sixteen FP32 magnitudes; strict bit parity."""
    import numpy as np
    a, b = (np.asarray(value, dtype=np.float32).reshape(-1) for value in (reference, candidate))
    if a.shape != (16,) or b.shape != (16,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Two finite evaluated16 route score vectors required')
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    denominator = np.maximum(np.abs(a.astype(np.float64)), 1e-30)
    exact = a.tobytes() == b.tobytes()
    order = lambda values: sorted(range(16), key=lambda i: (-float(values[i]), i))
    return {'version': VERSION, 'referenceRouteScores': a.tolist(), 'selectedRouteScores': b.tolist(),
        'bitwiseEqual': exact, 'maximumAbsoluteDifference': float(difference.max()),
        'maximumRelativeDifference': float((difference / denominator).max()),
        'all16RawOrderEqual': order(a) == order(b), 'referenceRawOrder': order(a),
        'selectedRawOrder': order(b), 'exactNativeProjectionGatePassed': exact,
        'qualityOrFinalDecisionClaim': False}


def native_comparison(backbone, adapter, item, projector, *, order=('ordinary', 'selected')):
    """Root's hook: same X/H/Y, ordinary full head versus prepared sixteen rows.

    Caller owns native lock/adapter scope. No body forward, optimizer, serving
    patch, parameter mutation, relaxed threshold or checkpoint save occurs here.
    Pair with prefix-off/on separately in the external fixed-row driver.
    """
    if tuple(order) not in (('ordinary', 'selected'), ('selected', 'ordinary')):
        raise ValueError('Predeclared AB or BA pair required')
    started = time.perf_counter()
    projector.assert_current(full=True); values = {}; timings = {}; mx = backbone.mx
    for arm in order:
        begin = time.perf_counter()
        result = REFERENCE_SCORE(backbone, adapter, item) if arm == 'ordinary' else projector.score(backbone, adapter, item)
        mx.eval(result); mx.synchronize(); backbone.check_memory()
        values[arm] = backbone.np.array(result).copy()
        timings[arm] = (time.perf_counter() - begin) * 1000
    projector.assert_current(full=True)
    return {**compare_vectors(values['ordinary'], values['selected']), 'order': list(order),
        'armMsIncludingFeatureTransferNormProjectionSyncMemoryAndHostCopy': timings,
        'pairedComparisonIncludingSourceChecksMs': (time.perf_counter() - started) * 1000,
        'preparationExcludedAndReportedSeparately': True,
        'projectionPreparation': deepcopy(projector.receipt), 'bodyForwards': 0,
        'scope': 'same evaluated public X/H/Y projection diagnostic; not complete rank/copy/UI latency'}
