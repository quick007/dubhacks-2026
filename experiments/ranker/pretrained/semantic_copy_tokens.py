"""Pure token contract for exact inference prompts and target-only copy loss."""
from __future__ import annotations

import copy
import hashlib
import json
from numbers import Integral


STOP_TOKEN = '<|im_end|>'


def _ids(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f'{name} must contain token IDs')
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in values):
        raise ValueError(f'{name} contains an invalid token ID')
    return [int(v) for v in values]


def _eos_values(value, name):
    if value is None:
        return []
    if isinstance(value, Integral) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(f'{name} must be an ID or collection of IDs')
    if not value:
        return []
    return sorted(set(_ids(list(value), name)))


def _hash_ids(values):
    return hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def build_training_tokens(tokenizer, rank_messages, row, max_context_tokens=4096):
    """Return exact causal alignment; this function never imports a model runtime.

    ``row['messages']`` already contains the declared common evidence and field
    turns. Ranking messages are supplied separately, rather than reconstructed
    from target metadata. The sampled cap includes the terminal native stop.
    """
    if isinstance(max_context_tokens, bool) or not isinstance(max_context_tokens, Integral) or max_context_tokens < 2:
        raise ValueError('A positive complete-context cap of at least two is required')
    cap = row.get('maxTokens')
    if isinstance(cap, bool) or not isinstance(cap, Integral) or cap < 1:
        raise ValueError('A positive sampled field-token cap is required')
    target_value = row.get('targetValue')
    if not isinstance(target_value, str) or not target_value.strip():
        raise ValueError('A nonempty fresh target value is required')
    field_messages = row.get('messages')
    if not isinstance(rank_messages, (list, tuple)) or not isinstance(field_messages, (list, tuple)) or not field_messages:
        raise ValueError('Supply explicit ranking messages and nonempty field messages')
    messages = copy.deepcopy(list(rank_messages) + list(field_messages))
    if any(not isinstance(m, dict) or not isinstance(m.get('role'), str) or not isinstance(m.get('content'), str)
           for m in messages):
        raise ValueError('Model messages need explicit role and text content')
    prompt = _ids(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False), 'Inference prompt')
    value_tokens = _ids(tokenizer.encode(target_value, add_special_tokens=False), 'Fresh target')
    stop_ids = _ids(tokenizer.encode(STOP_TOKEN, add_special_tokens=False), 'Stop marker')
    if len(stop_ids) != 1:
        raise ValueError('Native im_end stop marker must encode to exactly one token')
    stop = stop_ids[0]
    declared_many = _eos_values(getattr(tokenizer, 'eos_token_ids', None), 'eos_token_ids')
    declared_one = _eos_values(getattr(tokenizer, 'eos_token_id', None), 'eos_token_id')
    accepted_eos = sorted(set(declared_many + declared_one))
    if stop not in accepted_eos:
        raise ValueError('Native im_end is not a declared tokenizer EOS token')
    if any(token in accepted_eos for token in value_tokens):
        raise ValueError('Fresh target contains an embedded stop token')
    target = value_tokens + [stop]
    if len(target) > cap:
        raise ValueError('Fresh target plus native EOS exceeds sampled field cap; refusing truncation')
    full = prompt + target
    if len(full) > max_context_tokens:
        raise ValueError('Complete teacher-forced context exceeds cap; refusing truncation')
    inputs = full[:-1]
    prediction_start = len(prompt) - 1
    # These are structural checks, independent of model/loss implementations.
    if full[:len(prompt)] != prompt or inputs[prediction_start + 1:] != target[:-1]:
        raise AssertionError('Teacher forcing changed the inference boundary')
    if len(inputs) - prediction_start != len(target):
        raise AssertionError('Causal hidden positions and supervised targets do not align')
    return {
        'promptTokenIds': prompt, 'fullTokenIds': full,
        'teacherInputTokenIds': inputs, 'targetTokenIds': target,
        'predictionStart': prediction_start, 'supervisedTokenCount': len(target),
        'promptTokens': len(prompt), 'teacherInputTokens': len(inputs), 'fullTokens': len(full),
        'stopTokenId': stop, 'stopToken': STOP_TOKEN,
        'eosReceipt': {'eosTokenIds': declared_many, 'eosTokenId': declared_one,
            'acceptedEosTokenIds': accepted_eos, 'stopDeclaredBy': [name for name, values in
                (('eos_token_ids', declared_many), ('eos_token_id', declared_one)) if stop in values]},
        # Only actual model input token sequences are hashed, never row metadata.
        'promptSha256': _hash_ids(prompt), 'modelInputSha256': _hash_ids(inputs),
    }
