"""Gated auxiliary-output feature probe; imports never load MLX or a model.

This does not replace the retained failed inference-mode feature experiments.
Call ``probe_gradient_features`` with an already prepared, pristine native
state and a nonzero final-down LoRA adapter. There are no optimizer updates.
Only a passing exact three-way comparison returns a usable feature item.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path


FEATURE_KEYS = ('x', 'h', 'y')
GRADIENT_KEYS = {'lora_a', 'lora_b'}
DTYPES = {'mlx.core.bfloat16', 'mlx.core.float16', 'mlx.core.float32'}
SOURCE_BASIS = {
    'auxiliaryContract': 'mlx/core/__init__.pyi:4695-4735; scalar first, arbitrary auxiliary outputs',
    'nnWrapper': 'mlx/nn/utils.py:27-35; preserves mx.value_and_grad value tree',
    'gradientModeBoundary': 'https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/fast.cpp#L871-L885',
    'auxiliaryStopGradient': 'https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/transforms.cpp#L694-L706',
    'interpretation': 'Source motivates this new probe; it does not establish the cause of prior mismatches.',
}


class GradientFeatureProbeError(RuntimeError):
    """A blocked native probe, with its partial JSON-safe receipt attached."""
    def __init__(self, report):
        self.report = report
        super().__init__(report['blocker']['type'] + ': ' + report['blocker']['message'])


def _digest_tokens(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def build_native_contract(tokenizer, state, field, maximum_context=4096):
    """Pure tokenization: one complete suffix, exact cache boundary, no trim."""
    from .semantic_copy_tokens import build_training_tokens
    if state.get('consumed'):
        raise ValueError('A pristine, unconsumed source state is required')
    messages = field.get('messages')
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError('An explicit field message is required')
    # Training records can repeat the declared common evidence turn. It must
    # already be the exact end of this state's messages, never silently ignored.
    common = list(messages[:-1])
    if common and state['messages'][-len(common):] != common:
        raise ValueError('Declared common evidence differs from the prepared state')
    contract = build_training_tokens(tokenizer, state['messages'],
        {**field, 'messages': [messages[-1]]}, maximum_context)
    prefix = state['tokens']
    inputs = contract['teacherInputTokenIds']
    if inputs[:len(prefix)] != prefix:
        raise ValueError('Teacher input does not extend the exact source token prefix')
    suffix = inputs[len(prefix):]
    start = contract['predictionStart'] - len(prefix)
    if not 1 <= len(suffix) <= 256:
        raise ValueError('One complete native suffix of 1-256 tokens is required; no truncation')
    if start < 0 or len(suffix) - start != len(contract['targetTokenIds']):
        raise ValueError('Target positions do not align within the complete suffix')
    return {**contract, 'suffixTokenIds': suffix, 'predictionStartInBranch': start,
            'sourceTokenSha256': _digest_tokens(prefix), 'sourceTokens': len(prefix)}


def validate_feature_contract(features, dtypes, start, targets):
    """CPU-testable shape/dtype/alignment guard; no tensor execution."""
    if set(features) != set(FEATURE_KEYS) or set(dtypes) != set(FEATURE_KEYS):
        raise ValueError('Exactly x, h and y features and dtype receipts are required')
    shapes = {k: tuple(features[k].shape) for k in FEATURE_KEYS}
    length = shapes['x'][1] if len(shapes['x']) == 3 else 0
    if not 1 <= length <= 256 or any(shapes[k] != (1, length, width)
            for k, width in (('x', 12288), ('h', 4096), ('y', 4096))):
        raise ValueError('Preserve the entire native [1,L,D] suffix shape')
    if set(dtypes.values()) - DTYPES:
        raise ValueError('Unsupported per-tensor feature dtype')
    if isinstance(start, bool) or not isinstance(start, Integral) or not 0 <= start < length:
        raise ValueError('Invalid first target prediction position')
    if not isinstance(targets, (list, tuple)) or not targets or any(
            isinstance(t, bool) or not isinstance(t, Integral) or t < 0 or t >= 248320 for t in targets):
        raise ValueError('Invalid pinned-vocabulary target IDs')
    if length - start != len(targets):
        raise ValueError('Full suffix and target-only projection do not align')
    return {'shapes': {k: list(shapes[k]) for k in FEATURE_KEYS},
            'dtypes': dict(dtypes), 'suffixTokens': length, 'targetTokens': len(targets),
            'predictionStartInBranch': int(start)}


def unpack_auxiliary_value(value):
    """Require the actual transformed return value, rather than escaped taps."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError('Expected transformed (scalar loss, auxiliary feature tree)')
    loss, features = value
    if tuple(loss.shape) != () or not isinstance(features, dict) or set(features) != set(FEATURE_KEYS):
        raise ValueError('Invalid scalar/auxiliary output contract')
    return loss, features


def _host_tensor(mx, value):
    import numpy as np
    if str(value.dtype) not in DTYPES:
        raise ValueError('Unsupported native floating-point tensor dtype')
    # Float32 is an exact host representation of each supported source dtype.
    host = np.array(value.astype(mx.float32))
    if not np.isfinite(host).all():
        raise FloatingPointError('Nonfinite loss, gradient, parameter or feature')
    dtype = str(value.dtype)
    receipt = {'shape': list(value.shape), 'dtype': dtype,
               'sha256': hashlib.sha256(dtype.encode() + str(tuple(value.shape)).encode() + host.tobytes()).hexdigest()}
    return host, receipt


def compare_host_results(left, right):
    """Strict numerical gate, including gradient roster/shape/dtype, no tolerance."""
    import numpy as np
    if set(left['gradients']) != GRADIENT_KEYS or set(right['gradients']) != GRADIENT_KEYS:
        raise ValueError('Exactly lora_a and lora_b gradients are required')
    def one(a, b):
        av, ar = a; bv, br = b
        if not np.isfinite(av).all() or not np.isfinite(bv).all():
            raise FloatingPointError('Nonfinite comparison input')
        shape_match = ar['shape'] == br['shape'] and av.shape == bv.shape
        dtype_match = ar['dtype'] == br['dtype']
        return {'shapeExact': shape_match, 'dtypeExact': dtype_match,
                'arrayExact': bool(shape_match and np.array_equal(av, bv)),
                'maxAbsDifference': float(np.max(np.abs(av.astype('float64') - bv.astype('float64')))) if shape_match else None}
    loss = one(left['loss'], right['loss'])
    gradients = {k: one(left['gradients'][k], right['gradients'][k]) for k in sorted(GRADIENT_KEYS)}
    exact = all(v['shapeExact'] and v['dtypeExact'] and v['arrayExact'] for v in [loss, *gradients.values()])
    return {'exact': exact, 'loss': loss, 'gradients': gradients}


def _feature_loss(backbone, adapter, item):
    import mlx.nn as nn
    mx = backbone.mx
    dtype_map = {str(t): t for t in (mx.bfloat16, mx.float16, mx.float32)}
    validate_feature_contract({k: item[k] for k in FEATURE_KEYS}, item['dtypes'],
                              item['predictionStartInBranch'], item['targets'])
    x, h, y = (mx.array(item[k], dtype=dtype_map[item['dtypes'][k]]) for k in FEATURE_KEYS)
    # Preserve LoRALinear's scale/cast and DecoderLayer's two additions. Keep
    # full suffix shape until after the final native RMSNorm.
    z = (adapter.dropout(x) @ adapter.lora_a) @ adapter.lora_b
    delta = (adapter.scale * z).astype(x.dtype)
    hidden = backbone.model.model.norm(h + (y + delta))
    logits = backbone.model.language_model.lm_head(hidden[0, item['predictionStartInBranch']:])
    return nn.losses.cross_entropy(logits.astype(mx.float32), mx.array(item['targets']), reduction='mean')


def feature_loss(backbone, adapter, item):
    """Only a feature item returned by a passing probe may enter this API."""
    if item.get('gradientFeatureGatePassed') is not True:
        raise ValueError('Native exact auxiliary feature gate has not passed')
    return _feature_loss(backbone, adapter, item)


def probe_gradient_features(backbone, adapter, state, field):
    """Return ``(report, item_or_None)``; no optimizer, parameter or cache updates.

    Caller owns the sole native worker, memory guard, result persistence and a
    separately verified final-MLP cache proof. The adapter must have nonzero B.
    A failed numerical gate returns its measured evidence and no usable item.
    Execution/contract errors raise; caller must persist them as blocked runs.
    """
    import mlx.nn as nn
    import numpy as np
    from importlib.metadata import version
    from mlx.utils import tree_flatten
    from .native_batch_diagnosis import state_hash
    from .parallel_decode import clone_prefix_state
    mx = backbone.mx; model = backbone.model; layer = model.layers[-1]
    if state.get('model') is not model or state.get('consumed'):
        raise ValueError('Pristine native state must belong to this exact model object')
    if len(model.layers) != 32 or layer.mlp.down_proj is not adapter:
        raise ValueError('Only the pinned layer31 final-down adapter is supported')
    if model.training or adapter.training or adapter.dropout.training or adapter.dropout._p_1 != 1:
        raise ValueError('Frozen eval model and zero adapter dropout are required')
    roster = dict(tree_flatten(model.trainable_parameters()))
    if len(roster) != 2:
        raise ValueError('Unexpected model trainable roster')
    if {k.rsplit('.', 1)[-1] for k in roster} != GRADIENT_KEYS or any(
            '.31.mlp.down_proj.' not in k for k in roster):
        raise ValueError('Only layer31 down-projection A/B may be trainable')
    if set(dict(tree_flatten(adapter.trainable_parameters()))) != GRADIENT_KEYS:
        raise ValueError('Unexpected adapter trainable roster')
    if tuple(adapter.lora_a.shape) != (12288, 4) or tuple(adapter.lora_b.shape) != (4, 4096) or adapter.scale != 20:
        raise ValueError('Pinned rank4/scale20 final-down adapter is required')
    original_params = {k: getattr(adapter, k) for k in sorted(GRADIENT_KEYS)}
    mx.eval(list(original_params.values())); mx.synchronize()
    initial = {k: _host_tensor(mx, v) for k, v in original_params.items()}
    if not np.any(initial['lora_b'][0] != 0):
        raise ValueError('A nonzero B adapter is required; this probe never changes parameters')
    contract = build_native_contract(backbone.tokenizer, state, field)
    source_hash = state_hash(state['cache'])
    state_tokens = list(state['tokens'])
    source_messages = json.dumps(state['messages'], sort_keys=True, separators=(',', ':'))
    original_norm = layer.post_attention_layernorm; original_linear = adapter.linear
    saved = {}; calls = {'x': 0, 'h': 0, 'y': 0}

    class NormTap(nn.Module):
        def __init__(self):
            super().__init__(); self.base = original_norm
        def __call__(self, h):
            calls['h'] += 1; saved['h'] = h
            return self.base(h)

    class BaseDownTap(nn.Module):
        def __init__(self):
            super().__init__(); self.base = original_linear
        def __call__(self, x):
            calls['x'] += 1; saved['x'] = x
            y = self.base(x)
            calls['y'] += 1; saved['y'] = y
            return y

    def native_loss(with_auxiliary=False):
        branch = clone_prefix_state(backbone, state)
        hidden = model.model(mx.array([contract['suffixTokenIds']]), cache=branch['cache'])
        logits = model.language_model.lm_head(hidden[0, contract['predictionStartInBranch']:])
        loss = nn.losses.cross_entropy(logits.astype(mx.float32), mx.array(contract['targetTokenIds']), reduction='mean')
        if with_auxiliary:
            if calls != {'x': 1, 'h': 1, 'y': 1}:
                raise AssertionError('Feature taps must observe exactly one complete suffix')
            return loss, {k: saved[k] for k in FEATURE_KEYS}
        return loss

    def evaluate(fn, auxiliary=False):
        try:
            value, gradients = nn.value_and_grad(adapter, fn)()
            if auxiliary:
                loss, aux = unpack_auxiliary_value(value)
            else:
                loss, aux = value, None
            # All evaluation/host conversion happens AFTER value_and_grad has
            # returned. Only returned auxiliary arrays leave the trace.
            mx.eval(value, gradients); mx.synchronize(); backbone.check_memory()
            flat = dict(tree_flatten(gradients))
            if set(flat) != GRADIENT_KEYS:
                raise AssertionError('Unexpected native gradient roster')
            result = {'loss': _host_tensor(mx, loss),
                      'gradients': {k: _host_tensor(mx, flat[k]) for k in sorted(flat)}}
            return result, aux
        finally:
            # Keep caller-owned parameter objects; never retain traced module
            # assignments or any adapter changes made by a transformation.
            for k, v in original_params.items(): setattr(adapter, k, v)

    def arm_receipt(result):
        return {'loss': float(result['loss'][0].item()), 'lossTensor': result['loss'][1],
                'gradientTensors': {k: v[1] for k, v in result['gradients'].items()}}

    report = {'status': 'started', 'mechanism': 'native value_and_grad auxiliary outputs; full suffix features',
              'sourceSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'sourceBasis': SOURCE_BASIS, 'runtime': {k: version(k) for k in ('mlx', 'mlx-lm', 'numpy')},
              'modelRevision': getattr(backbone, 'revision', None),
              'rowId': field.get('rowId'), 'caseId': field.get('caseId'),
              'nativeUpdates': 0, 'nonzeroB': True, 'trainingPermission': 'probe only; no full fit',
              'earlierInferenceFeatureFailures': 'retained strict failures; not superseded by this probe',
              'contract': {k: v for k, v in contract.items() if k not in
                           ('teacherInputTokenIds', 'fullTokenIds', 'promptTokenIds', 'suffixTokenIds', 'targetTokenIds')},
              'sourceStateHashBefore': source_hash,
              'sourceStateHashScope': 'native cache class and complete .state tree; metadata compatibility proof remains separately required',
              'adapterBefore': {k: v[1] for k, v in initial.items()}, 'arms': {}}
    error = None
    try:
        direct, _ = evaluate(native_loss)
        report['arms']['direct'] = arm_receipt(direct)
        # Install taps only around the capture arm. LoRALinear.__call__, both
        # original native modules and every tensor shape remain unchanged.
        try:
            layer.post_attention_layernorm = NormTap(); adapter.linear = BaseDownTap()
            captured, auxiliary = evaluate(lambda: native_loss(True), auxiliary=True)
            report['arms']['auxiliaryNative'] = arm_receipt(captured)
        finally:
            layer.post_attention_layernorm = original_norm; adapter.linear = original_linear
            saved.clear()
        features = {k: _host_tensor(mx, auxiliary[k]) for k in FEATURE_KEYS}
        item = {k: features[k][0] for k in FEATURE_KEYS}
        item.update(dtypes={k: features[k][1]['dtype'] for k in FEATURE_KEYS},
                    predictionStartInBranch=contract['predictionStartInBranch'],
                    targets=list(contract['targetTokenIds']), rowId=field.get('rowId'), caseId=field.get('caseId'),
                    contract=report['contract'])
        report['featureContract'] = validate_feature_contract({k: item[k] for k in FEATURE_KEYS},
            item['dtypes'], item['predictionStartInBranch'], item['targets'])
        report['featureTensors'] = {k: features[k][1] for k in FEATURE_KEYS}
        del auxiliary
        reconstructed, _ = evaluate(lambda: _feature_loss(backbone, adapter, item))
        report['arms']['frozenFeatures'] = arm_receipt(reconstructed)
        report['comparisons'] = {
            'nativeVsAuxiliaryNative': compare_host_results(direct, captured),
            'auxiliaryNativeVsFrozenFeatures': compare_host_results(captured, reconstructed),
            'nativeVsFrozenFeatures': compare_host_results(direct, reconstructed)}
        report['tapCalls'] = dict(calls)
        passed = all(comparison['exact'] for comparison in report['comparisons'].values())
    except Exception as exc:
        error = exc
    finally:
        layer.post_attention_layernorm = original_norm; adapter.linear = original_linear
        for k, v in original_params.items(): setattr(adapter, k, v)
        saved.clear()
        try:
            report['sourceStateHashAfter'] = state_hash(state['cache'])
            report['adapterAfter'] = {k: _host_tensor(mx, getattr(adapter, k))[1] for k in sorted(GRADIENT_KEYS)}
            report['sourceStatePreserved'] = (report['sourceStateHashAfter'] == source_hash
                and state['tokens'] == state_tokens and not state.get('consumed')
                and json.dumps(state['messages'], sort_keys=True, separators=(',', ':')) == source_messages)
            report['adapterPreserved'] = report['adapterBefore'] == report['adapterAfter']
            if not report['sourceStatePreserved'] or not report['adapterPreserved']:
                raise AssertionError('Probe changed the pristine state or adapter')
            backbone.check_memory()
            report['memory'] = backbone.runtime_memory()
        except Exception as exc:
            if error is not None:
                report['priorBlocker'] = {'type': type(error).__name__, 'message': str(error)}
            error = exc
    if error is not None:
        report['status'] = 'blocked'
        report['blocker'] = {'type': type(error).__name__, 'message': str(error)}
        raise GradientFeatureProbeError(report) from error
    report['status'] = 'exact-gate-passed' if passed else 'strict-gate-failed'
    item['gradientFeatureGatePassed'] = bool(passed)
    return report, item if passed else None
