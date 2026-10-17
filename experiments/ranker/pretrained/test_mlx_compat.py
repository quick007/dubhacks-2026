"""Tiny native hybrid compatibility test; no pretrained weights or downloads."""
import time
import unittest
import numpy as np
import torch
try:
    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import Qwen3_5TextModel, TextModelArgs
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
from experiments.ranker.pretrained.backbone import ParallelHeads
from experiments.ranker.pretrained.cli import fit


@unittest.skipUnless(MLX_AVAILABLE, "Native MLX/Metal device required for this optional hardware test")
class MLXCompatibilityTests(unittest.TestCase):
    def test_hybrid_hidden_states_right_padding_and_trainable_heads(self):
        args = TextModelArgs(model_type="qwen3_5", hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, vocab_size=32, linear_num_value_heads=2,
                             linear_num_key_heads=2, linear_key_head_dim=8,
                             linear_value_head_dim=8, full_attention_interval=2)
        model = Qwen3_5TextModel(args)
        model.eval()
        model.freeze()
        short = model(mx.array([[1, 2, 3]]))[:, 2]
        pooled = model(mx.array([[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]]))[:, 2]
        mx.eval(short, pooled)
        self.assertTrue(bool(mx.all(mx.isfinite(pooled)).item()))
        self.assertLess(float(mx.max(mx.abs(short[0] - pooled[0])).item()), 1e-5)
        embeddings = torch.from_numpy(np.array(pooled)).clone()
        heads = ParallelHeads(32)
        before = heads(embeddings).detach().clone()
        report = fit(heads, embeddings, torch.zeros((2, 2)), 1, time.perf_counter() + 5)
        self.assertGreater(report["firstGradientL2"], 0)
        self.assertFalse(torch.equal(before, heads(embeddings)))

    def test_complete_recurrent_conv_and_kv_cache_append_matches_fresh_prefix(self):
        from mlx_lm.models.cache import ArraysCache, KVCache
        args = TextModelArgs(model_type="qwen3_5", hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, vocab_size=32, linear_num_value_heads=2,
                             linear_num_key_heads=2, linear_key_head_dim=8,
                             linear_value_head_dim=8, full_attention_interval=2)
        model = Qwen3_5TextModel(args)
        model.eval(); model.freeze()
        cache = [ArraysCache(2), KVCache()]
        identity = [id(entry) for entry in cache]
        prefix = model(mx.array([[1, 2, 3]]), cache=cache)
        mx.eval(prefix, [entry.state for entry in cache])
        self.assertIsNotNone(cache[0][0])  # convolution state
        self.assertIsNotNone(cache[0][1])  # recurrent DeltaNet state
        appended = model(mx.array([[4, 5]]), cache=cache)
        fresh = model(mx.array([[1, 2, 3, 4, 5]]))
        mx.eval(appended, fresh, [entry.state for entry in cache])
        self.assertEqual(identity, [id(entry) for entry in cache])
        self.assertLess(float(mx.max(mx.abs(appended[:, -1] - fresh[:, -1])).item()), 1e-4)


if __name__ == "__main__":
    unittest.main()
