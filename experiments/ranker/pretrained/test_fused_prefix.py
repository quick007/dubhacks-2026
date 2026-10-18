"""CPU-only raw/native serialization, feature/cache construction and mode gates."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from . import fused_prefix as fused
from .semantic_copy_tokens import build_training_tokens
from .semantic_copy_gradient_features import build_native_contract


class NativeTokenizer:
    eos_token_id = 3
    eos_token_ids = {3, 4}
    pad_token_id = 0
    def __init__(self): self.calls = []
    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        tokens = []
        while text:
            if text.startswith("<|im_end|>"):
                tokens.append(3); text = text[len("<|im_end|>"):]
            elif text.startswith("<|im_start|>"):
                tokens.append(2); text = text[len("<|im_start|>"):]
            else:
                tokens.append(ord(text[0]) + 100); text = text[1:]
        return tokens
    def decode(self, values): return "decoded:" + str(values)
    def apply_chat_template(self, messages, *, tokenize=True, return_dict=False,
                            add_generation_prompt=False, enable_thinking=False, **kwargs):
        assert all(m["role"] != fused.RAW_ROLE for m in messages)
        self.calls.append(copy.deepcopy(messages))
        text = "".join("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n" for m in messages)
        if add_generation_prompt: text += "<|im_start|>assistant\n"
        value = self.encode(text) if tokenize else text
        return {"input_ids": value} if return_dict else value


class ArraysCache:
    def __init__(self): self.state = ([None, None], None, None)
    @classmethod
    def from_state(cls, state):
        value = cls(); value.state = state; return value


class KVCache(ArraysCache):
    def __init__(self): self.state = (None, None, 0)


def cache_factory(model):
    return [KVCache() if i % 4 == 3 else ArraysCache() for i in range(32)]


class FakeMx:
    array = staticmethod(np.array)
    arange = staticmethod(np.arange)
    maximum = staticmethod(np.maximum)
    isfinite = staticmethod(np.isfinite)
    all = staticmethod(np.all)
    float32 = np.float32
    linalg = np.linalg
    def __init__(self): self.eval_calls = []; self.sync_calls = 0
    def eval(self, *values): self.eval_calls.append(values)
    def synchronize(self, *stream): self.sync_calls += 1


class Model:
    training = False
    layers = [None] * 32
    def __init__(self): self.calls = []
    def __call__(self, *args, **kwargs): raise AssertionError("Unused full LM logits must never execute")
    def model(self, inputs, *, cache):
        self.calls.append((inputs.copy(), cache))
        length = inputs.shape[1]
        if cache is not None:
            for entry in cache:
                if isinstance(entry, KVCache):
                    offset = entry.state[2] + length
                    entry.state = (np.full((1, 1, offset, 2), offset, np.float32),
                                   np.full((1, 1, offset, 2), offset + 1, np.float32), offset)
                else:
                    old = 0 if entry.state[0][0] is None else entry.state[0][0][0, 0, 0]
                    entry.state = ([np.full((1, 1, 2), old + length, np.float32),
                                    np.full((1, 1, 2), old + length + 1, np.float32)], None, None)
        values = inputs.astype(np.float32)
        return np.stack((values + 1, values * 2 + 1, np.ones_like(values) * 3, np.ones_like(values) * 4), axis=-1)


class Backbone:
    np = np
    def __init__(self):
        self.model = Model(); self.tokenizer = fused.RawPrefixTokenizer(NativeTokenizer())
        self.mx = FakeMx(); self.forward_calls = 0; self.memory_checks = 0; self.peak = 0
    def check_memory(self): self.memory_checks += 1
    def runtime_memory(self):
        return {"processBudgetBytes": 100, "mlxActiveBytes": 0, "mlxPeakBytes": self.peak}


def request():
    return {"recordId": "external-only", "context": {"statedPreferences": ["durable", "compact"],
            "task": "Full task must appear separately in common copy context."}, "history": []}


def common(): return {"role": "user", "content": "Full public task, style and observed history. Independent factual fields."}


def prepare(bb):
    return fused.prepare_fused_prefix(bb, request(), {}, feature_mode="verified-base-sha",
        cache_factory=cache_factory, host_converter=lambda value: np.array(value).copy())


def snapshot_cache(cache):
    result = []
    for c in cache:
        def visit(v):
            if isinstance(v, np.ndarray): return (str(v.dtype), v.shape, v.tobytes())
            if isinstance(v, (list, tuple)): return [visit(x) for x in v]
            return v
        result.append((type(c).__name__, visit(c.state)))
    return result


class FusedPrefixTests(unittest.TestCase):
    def test_exact_focused_bytes_and_external_metadata_invariance(self):
        from .catalog_fusion.generalization_v2 import query_text_v2
        q = request(); expected = query_text_v2(q, {}, "focused")
        self.assertEqual(fused.focused_query_text(q, {}), expected)
        other = copy.deepcopy(q); other.update(recordId="different", hiddenScore=999, personaBio="hidden")
        other["context"]["task"] = "Different observable task still belongs in common turn."
        self.assertEqual(fused.focused_query_text(other, {}), expected)
        self.assertNotIn("external-only", expected)
        self.assertIn('["compact","durable"]', expected)

    def test_renderer_native_boundary_no_synthetic_role_or_fake_ids(self):
        native = NativeTokenizer(); tokenizer = fused.RawPrefixTokenizer(native)
        raw = fused.focused_query_text(request(), {})
        messages = [{"role": fused.RAW_ROLE, "content": raw}, common()]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        self.assertEqual(rendered, raw + native.apply_chat_template([common()], tokenize=False, add_generation_prompt=True))
        self.assertEqual(tokens, native.encode(rendered))
        prefix = native.encode(raw)
        self.assertEqual(tokens[:len(prefix)], prefix)
        self.assertTrue(all(all(m["role"] != fused.RAW_ROLE for m in call) for call in native.calls))
        self.assertEqual(tokenizer.encode(raw), native.encode(raw))
        self.assertEqual(tokenizer.decode([1]), native.decode([1]))
        self.assertIs(tokenizer.eos_token_ids, native.eos_token_ids)

    def test_native_only_messages_pass_through_unchanged(self):
        native = NativeTokenizer(); tokenizer = fused.RawPrefixTokenizer(native)
        for tokenize in (True, False):
            self.assertEqual(tokenizer.apply_chat_template([common()], tokenize=tokenize, return_dict=True),
                             native.apply_chat_template([common()], tokenize=tokenize, return_dict=True))

    def test_raw_only_has_exact_b1_tokens_and_no_generation_header(self):
        native = NativeTokenizer(); tokenizer = fused.RawPrefixTokenizer(native)
        messages = [{"role": fused.RAW_ROLE, "content": "RAW"}]
        self.assertEqual(tokenizer.apply_chat_template(messages), native.encode("RAW"))
        self.assertEqual(tokenizer.apply_chat_template(messages, tokenize=False), "RAW")
        self.assertEqual(native.calls, [])
        with self.assertRaises(ValueError): tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def test_fail_closed_raw_protocol_options_roles_and_boundary_changes(self):
        tokenizer = fused.RawPrefixTokenizer(NativeTokenizer())
        raw = {"role": fused.RAW_ROLE, "content": "RAW}"}
        for messages in ([common(), raw], [raw, raw], [raw, {"role": [], "content": "bad"}],
                         [raw, {"role": "user", "content": ["image"]}]):
            with self.assertRaises(ValueError): tokenizer.apply_chat_template(messages)
        for kwargs in ({"return_dict": True}, {"enable_thinking": True}, {"tools": []}, {"tokenize": 1}):
            with self.assertRaises(ValueError): tokenizer.apply_chat_template([raw, common()], **kwargs)
        class MergingTokenizer(NativeTokenizer):
            def encode(self, text, **kwargs):
                result = super().encode(text, **kwargs)
                if "}<|im_start|>" in text: result[0] += 1
                return result
        with self.assertRaisesRegex(ValueError, "cached token boundary"):
            fused.RawPrefixTokenizer(MergingTokenizer()).apply_chat_template([raw, common()])

    def test_one_b1_forward_feature_pooling_and_complete_hybrid_materialization(self):
        bb = Backbone(); features, timing, state = prepare(bb)
        tokens = bb.tokenizer.encode(fused.focused_query_text(request(), {}))
        expected_hidden = Model().model(np.array([tokens]), cache=None)
        pooled = expected_hidden[np.arange(1), np.array([len(tokens)]) - 1].astype(np.float32)
        pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-8)
        self.assertTrue(np.array_equal(features, pooled))
        self.assertEqual(len(bb.model.calls), 1); self.assertEqual(bb.forward_calls, 1)
        self.assertEqual(bb.model.calls[0][0].shape, (1, len(tokens)))
        self.assertEqual(state["tokens"], tokens)
        self.assertEqual(state["messages"], [{"role": fused.RAW_ROLE, "content": fused.focused_query_text(request(), {})}])
        self.assertEqual(state["cacheTypes"].count("ArraysCache"), 24)
        self.assertEqual(state["cacheTypes"].count("KVCache"), 8)
        self.assertEqual(len(bb.mx.eval_calls[-1][1]), 32)
        self.assertFalse(timing["lmHeadExecuted"])
        self.assertEqual(timing["truncatedRows"], 0)

    def test_common_append_deep_fork_source_unchanged_and_exact_chat(self):
        bb = Backbone(); _, _, raw = prepare(bb)
        before = snapshot_cache(raw["cache"])
        state, timing = fused.append_common_message(bb, raw, common(), current_mode="verified-base-sha")
        expected = bb.tokenizer.apply_chat_template(raw["messages"] + [common()], add_generation_prompt=False)
        self.assertEqual(state["tokens"], expected)
        self.assertEqual(state["messages"], raw["messages"] + [common()])
        self.assertEqual(state["featureTokenSha256"], raw["featureTokenSha256"])
        self.assertEqual(snapshot_cache(raw["cache"]), before)
        for source, branch in zip(raw["cache"], state["cache"], strict=True):
            self.assertIsNot(source, branch)
            source_array = source.state[0] if isinstance(source, KVCache) else source.state[0][0]
            branch_array = branch.state[0] if isinstance(branch, KVCache) else branch.state[0][0]
            self.assertFalse(np.shares_memory(source_array, branch_array))
        self.assertFalse(timing["unusedHiddenForced"])
        self.assertEqual(len(bb.mx.eval_calls[-1]), 1)
        with self.assertRaises(ValueError): fused.append_common_message(bb, state, common(), current_mode="verified-base-sha")

    def test_existing_teacher_contract_and_decoder_suffix_are_identical(self):
        from .fusion import append_generation_suffix
        bb = Backbone(); _, _, raw = prepare(bb)
        state, _ = fused.append_common_message(bb, raw, common(), current_mode="verified-base-sha")
        field = {"messages": [{"role": "user", "content": "Selected bound facts. Write fresh title."}],
                 "targetValue": "Fresh", "maxTokens": 8}
        contract = build_native_contract(bb.tokenizer, state, field)
        normal = build_training_tokens(bb.tokenizer, state["messages"], field)
        self.assertEqual(contract["promptTokenIds"], normal["promptTokenIds"])
        self.assertEqual(contract["teacherInputTokenIds"][:len(state["tokens"])], state["tokens"])
        self.assertEqual(contract["stopTokenId"], bb.tokenizer.eos_token_id)
        branch = {**state, "consumed": False}
        suffix, cache = append_generation_suffix(bb.tokenizer, field["messages"], branch, bb.model)
        self.assertEqual(state["tokens"] + suffix, contract["promptTokenIds"])
        self.assertIs(cache, state["cache"]); self.assertFalse(state["consumed"])
        self.assertTrue(branch["consumed"])

    def test_mode_feature_and_copy_seam_gates_fail_closed(self):
        bb = Backbone(); _, _, raw = prepare(bb)
        state, _ = fused.append_common_message(bb, raw, common(), current_mode="verified-base-sha")
        feature = {"featureMode": "verified-base-sha", "featureTokenSha256": state["featureTokenSha256"],
                   "cacheVsNoCacheFeaturesExact": True}
        fused.assert_feature_compatibility(state, "verified-base-sha", "verified-base-sha", feature_gate=feature)
        with self.assertRaises(ValueError): fused.assert_feature_compatibility(state, "wrong-mode", "verified-base-sha", feature_gate=feature)
        with self.assertRaises(ValueError): fused.assert_feature_compatibility(state, "verified-base-sha", "learned-sha", feature_gate=feature)
        proof = {"sourceMode": "verified-base-sha", "copyMode": "learned-sha", "prefixSha256": state["prefixSha256"],
                 "adapterNonzero": True, "completeCacheStatesExact": True, "forcedContinuationLogitsExact": True}
        fused.assert_feature_compatibility(state, "verified-base-sha", "learned-sha", feature_gate=feature, copy_cache_gate=proof)
        for key in ("completeCacheStatesExact", "forcedContinuationLogitsExact"):
            bad = {**proof, key: False}
            with self.assertRaises(ValueError): fused.assert_feature_compatibility(state, "verified-base-sha", "learned-sha", feature_gate=feature, copy_cache_gate=bad)
        for bad in ({**feature, "featureTokenSha256": "stale"}, {**feature, "cacheVsNoCacheFeaturesExact": False}):
            with self.assertRaises(ValueError): fused.assert_feature_compatibility(state, "verified-base-sha", "verified-base-sha", feature_gate=bad)

    def test_caps_bad_modes_cache_layout_memory_and_failure_do_not_mutate_source(self):
        bb = Backbone()
        for cap in (True, 0, 513):
            with self.assertRaises(ValueError): fused.prepare_fused_prefix(bb, request(), {}, feature_mode="base", max_root_tokens=cap)
        with self.assertRaises(ValueError): fused.prepare_fused_prefix(bb, request(), {}, feature_mode="")
        with self.assertRaises(ValueError): fused.prepare_fused_prefix(bb, request(), {}, feature_mode="base", max_root_tokens=1)
        bb.model.training = True
        with self.assertRaises(ValueError): prepare(bb)
        bb.model.training = False
        with self.assertRaises(ValueError): fused.prepare_fused_prefix(bb, request(), {}, feature_mode="base", cache_factory=lambda model: [], host_converter=np.array)
        bb = Backbone(); _, _, raw = prepare(bb); before = snapshot_cache(raw["cache"])
        with self.assertRaises(ValueError): fused.append_common_message(bb, raw, common(), current_mode="different")
        with self.assertRaises(ValueError): fused.append_common_message(bb, raw, common(), current_mode="verified-base-sha", max_append_tokens=1)
        self.assertEqual(snapshot_cache(raw["cache"]), before)
        bb.peak = 101
        with self.assertRaises(MemoryError): fused.append_common_message(bb, raw, common(), current_mode="verified-base-sha")

    def runtime_stubs(self, *, mismatch_arm=None, fail_arm=None):
        """Unit-test dependency mocks only; production controls patch no globals."""
        class Generator:
            created = []
            def __init__(self, model, **kwargs):
                self.index = len(self.created); self.created.append(self)
                self.width = kwargs["completion_batch_size"]; self.step = 0; self.closed = False
            def insert(self, prompts, **kwargs):
                self.caches = kwargs["caches"]
                return list(range(len(prompts)))
            def next_generated(self):
                if fail_arm == self.index and self.step == 1:
                    raise RuntimeError("synthetic mid-arm failure")
                if self.step == 2: return []
                self.step += 1
                result = []
                for uid in range(self.width):
                    # Mutate only branch caches, testing true source isolation.
                    self.caches[uid][0].state[0][0][:] += 1
                    probabilities = np.array([-.1, -1., -2.], np.float32)
                    if mismatch_arm == self.index: probabilities[0] += .001
                    result.append(types.SimpleNamespace(uid=uid, token=3 if self.step == 2 else 201 + uid,
                        finish_reason="stop" if self.step == 2 else None, logprobs=probabilities))
                return result
            def close(self): self.closed = True
        package = types.ModuleType("mlx_lm"); package.__path__ = []
        models = types.ModuleType("mlx_lm.models"); models.__path__ = []
        generate = types.ModuleType("mlx_lm.generate")
        generate.BatchGenerator = Generator; generate.StopSequences = lambda eos: eos; generate.generation_stream = object()
        sample = types.ModuleType("mlx_lm.sample_utils"); sample.make_sampler = lambda **kwargs: kwargs
        cache = types.ModuleType("mlx_lm.models.cache"); cache.make_prompt_cache = cache_factory
        return {"mlx_lm": package, "mlx_lm.models": models, "mlx_lm.generate": generate,
                "mlx_lm.sample_utils": sample, "mlx_lm.models.cache": cache}, Generator

    def control_rows(self):
        return [{"rowId": f"field-{i}", "messages": [{"role": "user", "content": f"Field {i} selected facts."}],
                 "maxTokens": 8} for i in range(6)]

    def run_fake_controls(self, directory, **kwargs):
        from . import native_batch_diagnosis
        modules, generator = self.runtime_stubs(**kwargs)
        def state_hash(cache): return hashlib.sha256(repr(snapshot_cache(cache)).encode()).hexdigest()
        with patch.dict(sys.modules, modules), patch.object(native_batch_diagnosis, "state_hash", state_hash):
            result = fused.matched_controls(Backbone(), request(), {}, common(), self.control_rows(),
                {"featureMode": "verified-base-sha", "currentMode": "verified-base-sha"}, directory, widths=(1, 3, 6))
        return result, generator

    def test_cpu_control_accounting_full_distributions_eos_actual_width_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "controls"
            result, generator = self.run_fake_controls(out)
            self.assertEqual(result["status"], "exact-controls-passed")
            self.assertEqual(len(result["arms"]), 9)
            self.assertTrue(all(g.closed for g in generator.created))
            self.assertEqual(json.loads((out / "results.json").read_text())["status"], result["status"])
            for arm in result["arms"]:
                self.assertEqual(arm["maximumRowsEmittingTogether"], arm["width"])
                self.assertEqual(arm["sourceBefore"], arm["sourceAfter"])
                self.assertEqual(len(arm["rows"]), arm["width"])
                for row in arm["rows"]:
                    self.assertEqual(row["stopTokenId"], 3)
                    self.assertEqual(row["arSteps"], 2)
                    self.assertEqual(len(row["distributionSha256"]), 2)
                self.assertTrue(all(p["maxTokens"] == 8 for p in arm["prompts"]))
            with self.assertRaises(FileExistsError): self.run_fake_controls(out)

    def test_cpu_distribution_only_mismatch_saved_and_strict_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "controls"
            with self.assertRaisesRegex(AssertionError, "full-distribution"):
                self.run_fake_controls(out, mismatch_arm=2)
            result = json.loads((out / "results.json").read_text())
            self.assertEqual(result["status"], "failed-strict-control")
            a, _, b = result["arms"]
            self.assertEqual(a["rows"][0]["rawSampledTokenIds"], b["rows"][0]["rawSampledTokenIds"])
            self.assertNotEqual(a["rows"][0]["distributionSha256"], b["rows"][0]["distributionSha256"])
            self.assertTrue((out / "width1-fork-repeat.json").is_file())

    def test_cpu_partial_failure_keeps_active_arm_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "controls"
            with self.assertRaisesRegex(RuntimeError, "mid-arm"):
                self.run_fake_controls(out, fail_arm=1)
            result = json.loads((out / "results.json").read_text())
            self.assertEqual(result["status"], "failed-strict-control")
            self.assertEqual(result["activeArm"]["label"], "independent-rebuild")
            self.assertEqual(len(result["activeArm"]["rows"][0]["rawSampledTokenIds"]), 1)
            self.assertEqual(len(result["activeArm"]["rows"][0]["distributionSha256"]), 1)


if __name__ == "__main__":
    unittest.main()
