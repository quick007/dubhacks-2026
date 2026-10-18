"""Short raw focused-query features plus complete same-backbone copy cache.

CPU import only. Native execution requires an explicitly supplied loaded model;
this module never loads a model, downloads weights, or patches global functions.
The new serialization is raw focused text followed by genuine native chat text.
It changes the copy distribution and requires its own quality/cache controls.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

VERSION = "focused-raw-fused-prefix.v1"
RAW_ROLE = "raw_prefix"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def _ids(tokens):
    if type(tokens) is not list or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("Require nonempty actual tokenizer IDs")
    return list(tokens)


def _mode(value):
    if type(value) is not str or not value.strip():
        raise ValueError("Explicit verified model/adapter feature mode identity required")
    return value


def focused_query_text(request, catalog):
    """Delegate exact frozen v2 formatting; never add chat framing to readout."""
    from .catalog_fusion.generalization_v2 import query_text_v2
    return query_text_v2(request, catalog, "focused")


class RawPrefixTokenizer:
    """Transparent tokenizer with one explicit synthetic serialization role.

    Only apply_chat_template changes. encode/decode/EOS and other properties
    delegate to the original tokenizer. Synthetic raw_prefix is stripped before
    the native template sees any messages. Full input is genuinely re-encoded;
    token IDs are never made up or concatenated as a substitute for tokenization.
    """
    def __init__(self, native_tokenizer):
        if isinstance(native_tokenizer, RawPrefixTokenizer):
            raise ValueError("Do not nest raw-prefix tokenizer wrappers")
        self._native = native_tokenizer

    def __getattr__(self, name):
        return getattr(self._native, name)

    @property
    def native_tokenizer(self):
        return self._native

    def apply_chat_template(self, messages, *, tokenize=True, return_dict=False,
                            add_generation_prompt=False, enable_thinking=False, **kwargs):
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("Explicit nonempty message sequence required")
        synthetic = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == RAW_ROLE]
        if not synthetic:
            return self._native.apply_chat_template(messages, tokenize=tokenize, return_dict=return_dict,
                add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking, **kwargs)
        if synthetic != [0] or set(messages[0]) != {"role", "content"}:
            raise ValueError("Exactly one leading raw_prefix role is allowed")
        if any(type(v) is not bool for v in (tokenize, return_dict, add_generation_prompt, enable_thinking)):
            raise ValueError("Explicit boolean template controls required")
        if return_dict or enable_thinking or kwargs:
            raise ValueError("Raw-prefix v1 requires plain IDs/text, thinking off and no extra template options")
        raw = messages[0]["content"]
        if type(raw) is not str or not raw:
            raise ValueError("Raw prefix must be exact nonempty text")
        real_messages = copy.deepcopy(list(messages[1:]))
        if any(type(m) is not dict or set(m) != {"role", "content"}
               or type(m["role"]) is not str or m["role"] not in {"system", "user", "assistant"}
               or type(m["content"]) is not str for m in real_messages):
            raise ValueError("Raw-prefix v1 accepts actual text-only native chat turns")
        raw_tokens = _ids(self._native.encode(raw, add_special_tokens=False))
        if not real_messages:
            if add_generation_prompt:
                raise ValueError("A raw ranking prefix alone is not a generation instruction")
            return raw_tokens if tokenize else raw
        rendered = self._native.apply_chat_template(real_messages, tokenize=False,
            return_dict=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
        if type(rendered) is not str or not rendered:
            raise ValueError("Native chat template must render actual nonempty text")
        # Both paths use the native tokenizer's actual encoding. Compare native
        # template tokenization as an extra guard against hidden added tokens.
        rendered_tokens = _ids(self._native.encode(rendered, add_special_tokens=False))
        native_tokens = self._native.apply_chat_template(real_messages, tokenize=True,
            return_dict=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
        if _ids(native_tokens) != rendered_tokens:
            raise ValueError("Native template text/tokenized paths differ")
        full = raw + rendered
        full_tokens = _ids(self._native.encode(full, add_special_tokens=False))
        if full_tokens[:len(raw_tokens)] != raw_tokens:
            raise ValueError("Combined raw/native-chat text changes the exact cached token boundary")
        return full_tokens if tokenize else full


def _memory(backbone):
    backbone.check_memory()
    if hasattr(backbone, "runtime_memory"):
        receipt = backbone.runtime_memory()
        budget = receipt["processBudgetBytes"]
        if receipt["mlxActiveBytes"] > budget or receipt["mlxPeakBytes"] > budget:
            raise MemoryError("Active/observed peak exceeds declared process guard")


def _complete_cache(cache, tokens):
    if len(cache) != 32:
        raise ValueError("Pinned Qwen3.5 requires all32 cache entries")
    types = [type(c).__name__ for c in cache]
    if types.count("ArraysCache") != 24 or types.count("KVCache") != 8:
        raise ValueError("Require native24 recurrent/convolution +8 KV cache entries")
    if types != ["KVCache" if i % 4 == 3 else "ArraysCache" for i in range(32)]:
        raise ValueError("Native hybrid cache entries differ from pinned layer order")
    for entry in cache:
        state = entry.state
        if not isinstance(state, (list, tuple)) or len(state) != 3:
            raise ValueError("Incomplete native cache metadata")
        if type(entry).__name__ == "KVCache":
            if state[0] is None or state[1] is None or type(state[2]) is not int or state[2] != tokens:
                raise ValueError("KV history/offset differs from complete raw/chat prefix")
        elif (not isinstance(state[0], (list, tuple)) or len(state[0]) != 2
              or any(v is None for v in state[0]) or state[1] is not None or state[2] is not None):
            raise ValueError("Require full unpadded B1 convolution/recurrent state")


def prepare_fused_prefix(backbone, request, catalog, *, feature_mode, cache_factory=None,
                         host_converter=None, max_root_tokens=512):
    """One exact B1 full raw forward, last-token FP32 L2 feature + full cache.

    Cached-mode feature equivalence to FrozenBackbone.encode(cache=None) remains
    an explicit native gate. feature_mode must identify the actually active
    weights/adapter state; the caller must bind it to its verified checkpoint.
    """
    feature_mode = _mode(feature_mode)
    if type(max_root_tokens) is not int or not 1 <= max_root_tokens <= 512:
        raise ValueError("Raw root cap must be within fixed512; no truncation")
    if getattr(backbone.model, "training", False) is not False:
        raise ValueError("Query features require ordinary eval-forward mode")
    _memory(backbone)
    started = time.perf_counter()
    text = focused_query_text(request, catalog)
    tokens = _ids(backbone.tokenizer.encode(text, add_special_tokens=False))
    if len(tokens) > max_root_tokens:
        raise ValueError("Raw focused root exceeds token cap; no truncation")
    if cache_factory is None:
        from mlx_lm.models.cache import make_prompt_cache
        cache_factory = make_prompt_cache
    cache = cache_factory(backbone.model)
    mx = backbone.mx
    inputs = mx.array([tokens]); mx.eval(inputs)
    tokenize_ms = (time.perf_counter() - started) * 1000
    forward_start = time.perf_counter()
    hidden = backbone.model.model(inputs, cache=cache)
    # Match FrozenBackbone.encode's operation order, dtype and whole B1 shape.
    pooled = hidden[mx.arange(1), mx.array([len(tokens)]) - 1].astype(mx.float32)
    pooled = pooled / mx.maximum(mx.linalg.norm(pooled, axis=1, keepdims=True), 1e-8)
    mx.eval(pooled, [entry.state for entry in cache]); mx.synchronize()
    if not bool(mx.all(mx.isfinite(pooled)).item()):
        raise FloatingPointError("Nonfinite fused-prefix query feature")
    _complete_cache(cache, len(tokens))
    if host_converter is None:
        import torch
        features = torch.from_numpy(backbone.np.array(pooled)).clone()
    else:
        features = host_converter(pooled)
    backbone.forward_calls += 1
    _memory(backbone)
    state = {"model": backbone.model, "cache": cache, "tokens": list(tokens),
             "messages": [{"role": RAW_ROLE, "content": text}], "consumed": False,
             "prefixSha256": token_hash(tokens), "rawPrefixText": text,
             "rawPrefixTextSha256": hashlib.sha256(text.encode()).hexdigest(),
             "representation": VERSION, "featureMode": feature_mode, "cacheMode": feature_mode,
             "featureTokenSha256": token_hash(tokens), "cacheTypes": [type(c).__name__ for c in cache],
             "featureCacheNoneEquivalence": "requires separate exact native gate"}
    timing = {"tokenizeMs": tokenize_ms, "forwardMs": (time.perf_counter() - forward_start) * 1000,
              "rootPrepareMs": (time.perf_counter() - started) * 1000, "batchRows": 1,
              "prefixTokens": len(tokens), "backboneForwardCalls": 1, "truncatedRows": 0,
              "featureRepresentation": "B1 whole raw focused text; final-token FP32 L2",
              "cacheMaterializedCompletely": True, "lmHeadExecuted": False}
    return features, timing, state


prepare_raw_prefix = prepare_fused_prefix


def append_common_message(backbone, state, common_message, *, current_mode, max_append_tokens=512):
    """Fork complete raw state, then append one shared observable copy turn.

    The original cache is never modified or trimmed. Hidden/LM logits are not
    forced: every native cache component is materialized after each256-token
    chunk. The full request/style/history belongs here, before independent rows.
    """
    from .parallel_decode import fork_complete_state
    current_mode = _mode(current_mode)
    if (state.get("representation") != VERSION or state.get("model") is not backbone.model
            or state.get("consumed") or state.get("cacheMode") != current_mode):
        raise ValueError("Require pristine same-model/same-mode fused raw source")
    if len(state["messages"]) != 1 or state["messages"][0]["role"] != RAW_ROLE:
        raise ValueError("Append the common copy turn exactly once, immediately after raw root")
    if type(common_message) is not dict or set(common_message) != {"role", "content"} or common_message["role"] != "user" or type(common_message["content"]) is not str or not common_message["content"].strip():
        raise ValueError("One explicit nonempty observable user common turn required")
    if type(max_append_tokens) is not int or not 1 <= max_append_tokens <= 512:
        raise ValueError("Common append cap must be within512")
    if not isinstance(backbone.tokenizer, RawPrefixTokenizer):
        raise ValueError("New runner must explicitly own the versioned tokenizer wrapper")
    _memory(backbone)
    start = time.perf_counter()
    messages = copy.deepcopy(state["messages"] + [common_message])
    full = backbone.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
        add_generation_prompt=False, enable_thinking=False)
    prefix = state["tokens"]
    if state["prefixSha256"] != token_hash(prefix) or full[:len(prefix)] != prefix:
        raise ValueError("Common chat does not extend the complete exact raw prefix")
    suffix = full[len(prefix):]
    if not 1 <= len(suffix) <= max_append_tokens or len(full) > 4096:
        raise ValueError("Common append exceeds explicit cap; no truncation")
    def materialize(values):
        backbone.mx.eval(values); backbone.mx.synchronize()
    branch = fork_complete_state(state, materialize)
    _memory(backbone)
    calls = 0
    for offset in range(0, len(suffix), 256):
        hidden = backbone.model.model(backbone.mx.array([suffix[offset:offset + 256]]), cache=branch["cache"])
        backbone.mx.eval([entry.state for entry in branch["cache"]]); backbone.mx.synchronize()
        del hidden
        calls += 1; backbone.forward_calls += 1
        _memory(backbone)
    _complete_cache(branch["cache"], len(full))
    branch.update(tokens=list(full), messages=messages, prefixSha256=token_hash(full),
                  commonMessageSha256=hashlib.sha256(canonical(common_message).encode()).hexdigest())
    return branch, {"commonAppendTokens": len(suffix), "copyPrefixTokens": len(full),
                    "commonAppendMs": (time.perf_counter() - start) * 1000,
                    "backboneForwardCalls": calls, "prefillStepSize": 256,
                    "lmHeadExecuted": False, "unusedHiddenForced": False,
                    "sourceCacheMutation": False, "cacheFork": "complete native states; exact deep copy"}


def assert_feature_compatibility(state, head_feature_mode, copy_mode, *, feature_gate, copy_cache_gate=None):
    """Fail closed on stale head mode or an unproved final-adapter cache seam.

    Receipts must come from root-owned native controls, not CPU contract tests.
    This validates their binding; it does not manufacture experimental evidence.
    """
    _mode(head_feature_mode); _mode(copy_mode)
    if state.get("featureMode") != head_feature_mode:
        raise ValueError("Factor head and actual query feature weights/mode differ")
    if (type(feature_gate) is not dict or feature_gate.get("featureMode") != head_feature_mode
            or feature_gate.get("featureTokenSha256") != state.get("featureTokenSha256")
            or feature_gate.get("cacheVsNoCacheFeaturesExact") is not True):
        raise ValueError("Unchanged v2 head requires its exact cache=None/cached feature gate")
    if copy_mode != state.get("cacheMode"):
        if (type(copy_cache_gate) is not dict
                or copy_cache_gate.get("sourceMode") != state.get("cacheMode")
                or copy_cache_gate.get("copyMode") != copy_mode
                or copy_cache_gate.get("prefixSha256") != state.get("prefixSha256")
                or any(copy_cache_gate.get(key) is not True for key in
                       ("adapterNonzero", "completeCacheStatesExact", "forcedContinuationLogitsExact"))):
            raise ValueError("Changed copy adapter requires complete-prefix native cache/continuation proof")
    return {"featureMode": head_feature_mode, "copyMode": copy_mode,
            "featureGateRequired": True, "cacheAdapterSeamProved": copy_mode != state.get("cacheMode")}


def matched_controls(backbone, request, catalog, common, rows, mode_receipt, out, widths=(3, 6)):
    """Root-only bounded same-width fork/rebuild/repeat distribution controls.

    No native work occurs on import. Each width admits exactly N independent
    fields with their original caps. All full response distributions are hashed,
    including EOS responses. These instrumented diagnostics are not latency
    measurements. No native class, suffix function, sampler or global is patched.
    """
    from pathlib import Path
    if (not isinstance(widths, (list, tuple)) or not widths
            or any(type(w) is not int or w not in (1, 3, 6) for w in widths) or len(set(widths)) != len(widths)):
        raise ValueError("Unique bounded widths1/3/6 required")
    if (type(mode_receipt) is not dict or set(mode_receipt) != {"featureMode", "currentMode"}
            or _mode(mode_receipt["featureMode"]) != _mode(mode_receipt["currentMode"])):
        raise ValueError("Controls require the actually active single feature/cache/copy mode")
    if not isinstance(rows, (list, tuple)) or len(rows) < max(widths):
        raise ValueError("Enough genuine independent field rows for every actual width required")
    selected = list(rows[:max(widths)])
    if any(type(r) is not dict or type(r.get("rowId")) is not str or not r["rowId"] for r in selected):
        raise ValueError("Explicit distinct field row IDs required")
    if len({r["rowId"] for r in selected}) != len(selected):
        raise ValueError("Distinct actual field identities required")
    for row in selected:
        if type(row["maxTokens"]) is not int or row["maxTokens"] < 1 or len(row["messages"]) != 1 or row["messages"][0]["role"] != "user":
            raise ValueError("Controls retain full original caps and one actual field user turn")
    out = Path(out)
    if out.exists():
        raise FileExistsError("Preserve previous native control artifacts")
    from mlx_lm.generate import BatchGenerator, StopSequences, generation_stream
    from mlx_lm.sample_utils import make_sampler
    from .native_batch_diagnosis import state_hash
    from .parallel_decode import clone_prefix_state, exact_row_suffix
    out.mkdir(parents=True, exist_ok=False)
    report = {"version": VERSION, "status": "running", "modeReceipt": copy.deepcopy(mode_receipt),
              "widths": list(widths), "arms": [], "qualityOrSpeedupClaim": False,
              "controlPolicy": "samewidth originalcaps; fork / independent per-lane raw+common rebuild / repeat"}
    with (out / "control-inputs.json").open("x") as handle:
        json.dump({"rawPrefixText": focused_query_text(request, catalog), "commonMessage": common,
            "rows": [{key: row[key] for key in ("rowId", "messages", "maxTokens")} for row in selected]},
            handle, indent=2, allow_nan=False); handle.write("\n")
    mx = backbone.mx
    mode = mode_receipt["currentMode"]

    def prepare():
        features, raw_timing, raw = prepare_fused_prefix(backbone, request, catalog, feature_mode=mode)
        state, common_timing = append_common_message(backbone, raw, common, current_mode=mode)
        return features, state, {"raw": raw_timing, "common": common_timing}

    def host_feature(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy().copy()
        return backbone.np.array(value).copy()

    try:
        features, source, preparation = prepare()
        reference_features = host_feature(features)
        source_hash = state_hash(source["cache"])
        report.update(sourceCacheSha256=source_hash, prefixSha256=source["prefixSha256"],
                      featureTokenSha256=source["featureTokenSha256"], preparation=preparation)
        for width in widths:
            width_rows = selected[:width]
            reference = None
            for label in ("fork", "independent-rebuild", "fork-repeat"):
                before = state_hash(source["cache"])
                caches, prompts, prompt_receipts = [], [], []
                report["activeArm"] = {"width": width, "label": label, "prompts": prompt_receipts, "rows": []}
                for row in width_rows:
                    if label == "independent-rebuild":
                        rebuilt_features, branch, _ = prepare()
                        if not backbone.np.array_equal(reference_features, host_feature(rebuilt_features)):
                            raise AssertionError("Independent same-mode raw feature repeat differs")
                        del rebuilt_features
                    else:
                        branch = clone_prefix_state(backbone, source)
                    branch_hash = state_hash(branch["cache"])
                    if branch_hash != source_hash:
                        raise AssertionError("Complete fork/independently rebuilt raw+common native state differs")
                    suffix, cache = exact_row_suffix(backbone.tokenizer, row, branch, backbone.model)
                    if not 1 <= len(suffix) <= 256 or len(source["tokens"]) + len(suffix) + row["maxTokens"] > 4096:
                        raise ValueError("Control field exceeds prospective short suffix/context cap")
                    caches.append(cache); prompts.append(suffix)
                    prompt_receipts.append({"rowId": row["rowId"], "maxTokens": row["maxTokens"],
                        "suffixTokens": len(suffix), "fullPromptSha256": token_hash(source["tokens"] + suffix),
                        "completeCacheSha256": branch_hash})
                _memory(backbone)
                generator = BatchGenerator(backbone.model, completion_batch_size=width,
                    prefill_batch_size=width, prefill_step_size=256, sampler=make_sampler(temp=0.0))
                records, maximum, rounds = {}, 0, 0
                eos = [[int(token)] for token in backbone.tokenizer.eos_token_ids]
                try:
                    uids = generator.insert(prompts, max_tokens=[r["maxTokens"] for r in width_rows], caches=caches,
                        all_tokens=[list(source["tokens"]) for _ in width_rows],
                        samplers=[make_sampler(temp=0.0) for _ in width_rows], logits_processors=[[] for _ in width_rows],
                        stop_sequences=[StopSequences(eos) for _ in width_rows])
                    records = {uid: {"rowId": row["rowId"], "rawSampledTokenIds": [],
                        "distributionSha256": [], "finishReason": None, "stopTokenId": None}
                        for uid, row in zip(uids, width_rows, strict=True)}
                    report["activeArm"]["rows"] = list(records.values())
                    del caches, branch, cache
                    while responses := generator.next_generated():
                        maximum = max(maximum, len(responses)); rounds += 1
                        for response in responses:
                            if records[response.uid]["finishReason"] is not None:
                                raise AssertionError("Completed UID emitted again")
                            probabilities = response.logprobs.astype(mx.float32)
                            mx.eval(probabilities); mx.synchronize(generation_stream)
                            vector = backbone.np.array(probabilities).copy()
                            if not backbone.np.isfinite(vector).all():
                                raise FloatingPointError("Nonfinite native full response distribution")
                            record = records[response.uid]
                            record["distributionSha256"].append(hashlib.sha256(vector.tobytes()).hexdigest())
                            record["rawSampledTokenIds"].append(int(response.token))
                            if response.finish_reason is not None:
                                record["finishReason"] = response.finish_reason
                                if response.finish_reason == "stop":
                                    if int(response.token) not in backbone.tokenizer.eos_token_ids:
                                        raise AssertionError("Stop response is not native EOS")
                                    record["stopTokenId"] = int(response.token)
                        del response, responses, probabilities, vector
                        _memory(backbone)
                finally:
                    generator.close()
                    mx.synchronize(generation_stream)
                ordered = [records[uid] for uid in uids]
                if len(ordered) != width or maximum != width or any(r["finishReason"] is None for r in ordered):
                    raise AssertionError("Control did not execute every row at its actual declared width")
                for row, record in zip(width_rows, ordered, strict=True):
                    record["arSteps"] = len(record["rawSampledTokenIds"])
                    if not 1 <= record["arSteps"] <= row["maxTokens"]:
                        raise AssertionError("Actual sampled steps exceed original field cap")
                after = state_hash(source["cache"])
                arm = {"width": width, "label": label, "prompts": prompt_receipts, "rows": ordered,
                    "nativeDecodeRounds": rounds, "maximumRowsEmittingTogether": maximum,
                    "sourceBefore": before, "sourceAfter": after, "memory": backbone.runtime_memory()}
                report["arms"].append(arm)
                report.pop("activeArm", None)
                with (out / f"width{width}-{label}.json").open("x") as handle:
                    json.dump(arm, handle, indent=2, allow_nan=False); handle.write("\n")
                if before != source_hash or after != source_hash:
                    raise AssertionError("Pristine complete source state mutated")
                if reference is None:
                    reference = ordered
                elif reference != ordered:
                    raise AssertionError("Strict samewidth token/EOS/full-distribution control mismatch")
                _memory(backbone)
        report["status"] = "exact-controls-passed"
        return report
    except Exception as exc:
        report.update(status="failed-strict-control", errorType=type(exc).__name__, error=str(exc))
        raise
    finally:
        with (out / "results.json").open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False); handle.write("\n")
