"""CPU-only boundary, full-state clone, invalidation and gate tests."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from . import store_query_prefix_reuse_v1 as candidate

def request(query='new words'):
    return {'query': query, 'preferences': [], 'itemBudget': None, 'hardRequirements': []}

def formatter(value):
    return candidate.INSTRUCTION + candidate.canonical({'schemaVersion': 'actual-store-function-adaptation.v1/query',
        'request': value, 'budgetScope': 'inclusive item price only; delivery excluded'})

class BoundaryTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        ids = list(map(ord, text))
        # Deliberately context-sensitive token at held-back boundary. Whole text
        # must be encoded; standalone PREFIX + suffix encodings would differ.
        if text != candidate.PREFIX and text.startswith(candidate.PREFIX): ids[len(candidate.PREFIX) - 1] = 9999
        return ids

class KVCache:
    def __init__(self, state): self.state = state
    @classmethod
    def from_state(cls, state): return cls(state)

class ArraysCache(KVCache): pass

class PrefixReuseTests(unittest.TestCase):
    def binding(self, root):
        groups = {}
        for group in candidate.REQUIRED_ASSET_GROUPS:
            path = root / group; path.write_text(group); groups[group] = [path]
        return candidate.freeze_assets(groups, frozen927_hashes_sha256='a' * 64, model_revision='test-only')

    def test_module_import_is_stdlib_and_never_loads_mlx(self):
        subprocess.run([sys.executable, '-c', 'import sys; import experiments.ranker.pretrained.store_query_prefix_reuse_v1; '
            'assert not any(n=="mlx" or n.startswith("mlx.") or n=="torch" for n in sys.modules)'], check=True)

    def test_whole_tokens_checked_not_standalone_suffix_concatenation(self):
        tok = BoundaryTokenizer(); layout = candidate.query_layout(tok, request(), formatter=formatter)
        self.assertEqual(layout['fullTokenIds'], layout['prefixTokenIds'] + layout['suffixTokenIds'])
        self.assertEqual(layout['suffixTokenIds'][0], 9999)
        self.assertNotEqual(layout['prefixTokenIds'][-1], 9999)
        changed = request('"quoted" café'); changed['itemBudget'] = 0
        changed['hardRequirements'] = [{'attribute': 'color', 'value': 'Blue'}]
        other = candidate.query_layout(tok, changed, formatter=formatter)
        self.assertEqual(layout['prefixTokenIds'], other['prefixTokenIds'])
        self.assertNotEqual(layout['suffixTokenIds'], other['suffixTokenIds'])

    def test_formatter_prefix_token_mismatch_and_overlength_refuse_without_fallback(self):
        with self.assertRaisesRegex(ValueError, 'formatter'): candidate.query_layout(BoundaryTokenizer(), request(), formatter=lambda _: 'wrong')
        with self.assertRaisesRegex(ValueError, '512'): candidate.query_layout(BoundaryTokenizer(), request('x' * 1000), formatter=formatter)
        class Changed(BoundaryTokenizer):
            def encode(self, text, **kwargs):
                ids = super().encode(text, **kwargs)
                if text != candidate.PREFIX: ids[0] += 1
                return ids
        with self.assertRaisesRegex(ValueError, 'exactly'): candidate.query_layout(Changed(), request(), formatter=formatter)

    def test_full_native_state_clone_keeps_kv_offsets_and_both_recurrent_arrays(self):
        cache = [ArraysCache(([[1], [2]], None, None)), KVCache(([3], [4], 26))]
        materialized = []
        fork = candidate.clone_complete_cache(cache, materialize=materialized.append)
        fork[0].state[0][0][0] = 999; fork[1].state[0][0] = 888
        self.assertEqual(cache[0].state, ([[1], [2]], None, None)); self.assertEqual(cache[1].state, ([3], [4], 26))
        self.assertEqual(fork[1].state[2], 26); self.assertEqual(len(materialized), 1)

    def test_asset_invalidations_cover_source_model_catalog_and_head(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); binding = self.binding(root)
            self.assertTrue(candidate.assert_assets(binding, full_sha=True))
            (root / 'catalogFeatures').write_text('new feature')
            with self.assertRaisesRegex(ValueError, 'changed'): candidate.assert_assets(binding)

    def test_copy_adapter_must_be_owned_and_explicitly_disabled(self):
        adapter = SimpleNamespace(lora_a='weights', linear=object())
        bb = SimpleNamespace(model=SimpleNamespace(training=False, layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=adapter))]))
        with self.assertRaisesRegex(ValueError, 'explicitly'): 
            with candidate.frozen_base(bb): pass
        with candidate.frozen_base(bb, adapter): self.assertIs(bb.model.layers[-1].mlp.down_proj, adapter.linear)
        self.assertIs(bb.model.layers[-1].mlp.down_proj, adapter)

    def test_native_prepare_owner_gate_precedes_runtime_dependencies(self):
        with self.assertRaisesRegex(ValueError, 'Root'): candidate.QueryPrefixReuse.prepare(None, {}, native_owner=None)

    def test_fake_b1_runtime_prepares_once_and_runs_fresh_suffix_for_each_query(self):
        import numpy as np
        def state_hash(cache):
            def host(x):
                if isinstance(x, (tuple, list)): return [host(v) for v in x]
                return x.tolist() if hasattr(x, 'tolist') else x
            return candidate.hashed([host(c.state) for c in cache])
        calls = []
        class Model:
            training = False
            layers = [SimpleNamespace(mlp=SimpleNamespace(down_proj=object()))]
            def model(inner, inputs, cache=None):
                calls.append(inputs.shape[1])
                previous = int(cache[0].state[0][0].item()) if cache else 0
                cumulative = np.cumsum(inputs, axis=1) + previous
                hidden = (np.arange(4096, dtype=np.float32)[None, None, :] + cumulative[:, :, None]).astype(np.float32)
                if cache:
                    total = int(cumulative[0, -1])
                    for entry in cache:
                        if type(entry).__name__ == 'ArraysCache': entry.state = ([np.array([total]), np.array([total])], None, None)
                        else: entry.state = (np.array([total]), np.array([total]), entry.state[2] + inputs.shape[1])
                return hidden
        def factory(_):
            return [KVCache((np.array([0]), np.array([0]), 0)) if i % 4 == 3
                else ArraysCache(([np.array([0]), np.array([0])], None, None)) for i in range(32)]
        def validate(cache, expected):
            self.assertEqual(len(cache), 32)
            self.assertTrue(all(e.state[2] == expected for e in cache if type(e).__name__ == 'KVCache'))
        mx = SimpleNamespace(array=np.array, arange=np.arange, eval=lambda *args: None, synchronize=lambda: None,
            float32=np.float32, maximum=np.maximum, linalg=np.linalg, all=np.all, isfinite=np.isfinite)
        bb = SimpleNamespace(model=Model(), tokenizer=BoundaryTokenizer(), mx=mx, np=np,
            memory_budget=7 * 2**30, revision='test-only', forward_calls=0)
        deps = {'cache_factory': factory, 'validate_cache': validate, 'memory_guard': lambda bb: None,
            'state_hash': state_hash, 'parameter_check': lambda bb, binding: {'mockOnly': True}}
        with tempfile.TemporaryDirectory() as temp:
            binding = self.binding(Path(temp))
            encoder = candidate.QueryPrefixReuse.prepare(bb, binding, native_owner='root', confirm_serialized=True, _dependencies=deps)
            initial_hash = encoder.source_hash
            for value in (request('one fresh query'), request('a different fresh query')):
                feature, timing = encoder.encode(value, formatter=formatter, host_converter=np.array)
                layout = candidate.query_layout(bb.tokenizer, value, formatter=formatter)
                hidden = Model().model(np.array([layout['fullTokenIds']]))
                baseline = hidden[np.arange(1), np.array([len(layout['fullTokenIds'])]) - 1].astype(np.float32)
                baseline /= np.maximum(np.linalg.norm(baseline, axis=1, keepdims=True), 1e-8)
                self.assertTrue(np.array_equal(feature, baseline))
                self.assertEqual(timing['backboneForwardCalls'], 1); self.assertTrue(timing['sourceUnchanged'])
                self.assertEqual(timing['freshSuffixTokens'] + timing['reusedPrefixTokens'], len(layout['fullTokenIds']))
            self.assertEqual(bb.forward_calls, 3); self.assertEqual(state_hash(encoder.cache), initial_hash)
            self.assertFalse(any(key in encoder.__dict__ for key in ('query', 'features', 'responses')))

    def test_feature_cosine_is_distinct_from_bitwise_equality_and_guard_gate_is_strict(self):
        import torch
        reference = torch.ones(1, 4096); reference /= reference.norm()
        same = candidate.compare_features(reference, reference.clone())
        self.assertTrue(same['bitwiseEqual']); self.assertTrue(same['numericalGate'])
        bad = reference.clone(); bad[0, 0] += .1
        self.assertFalse(candidate.compare_features(reference, bad)['numericalGate'])
        decision = {'scores16': [0.] * 16, 'learnedRank': {'rawRank': list(range(16)), 'guards': [{'eligible': True}] * 16},
            'productIndices': list(range(6)), 'eligibleCount': 16, 'shortfall': 0, 'exactNamePolicy': {'applied': False}, 'cards': []}
        altered = copy.deepcopy(decision); altered['learnedRank']['guards'][0] = {'eligible': False}
        self.assertFalse(candidate.decision_parity(decision, altered)['decisionGate'])
        self.assertTrue(candidate.decision_parity(decision, copy.deepcopy(decision))['decisionGate'])

    def test_exact32_public_only_gate_rejects_labels_before_any_forward_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = [{'caseId': str(i), 'request': request(), 'target': 0} for i in range(32)]
            out = Path(temp) / 'controls'
            with self.assertRaisesRegex(ValueError, 'no gold'): candidate.development_controls(None, rows, None, out=out)
            self.assertFalse(out.exists())

if __name__ == '__main__': unittest.main()
