"""CPU tests by default; tiny native tests only after root explicitly sets the flag."""
import copy
import os
from types import SimpleNamespace
import unittest
from .parallel_decode import collect_native, field_rows, fork_complete_state


class FakeCache:
    def __init__(self):
        self.state = ([[[1, 2]], [[3, 4]]], [0], [7])
    @classmethod
    def from_state(cls, state):
        result = cls()
        result.state = state
        return result


class ParallelDecodeTests(unittest.TestCase):
    def test_full_state_fork_has_no_mutable_alias_or_model_copy(self):
        model = object()
        base = {"cache": [FakeCache()], "tokens": [1, 2], "messages": [], "model": model, "consumed": False}
        a, b = fork_complete_state(base), fork_complete_state(base)
        a["cache"][0].state[0][0][0][0] = 99
        a["cache"][0].state[2][0] = 99
        self.assertEqual(b["cache"][0].state, base["cache"][0].state)
        self.assertIs(a["model"], model)
        a["consumed"] = True
        with self.assertRaises(ValueError):
            fork_complete_state(a)

    def test_row_budget_and_heading_universal_evidence(self):
        for count in (1, 3, 6):
            catalog = {}; cards = []
            for i in range(count):
                pid = str(i)
                catalog[pid] = {"id": pid, "title": f"Travel bag {i}", "tags": ["durable", "compact"] if i % 2 == 0 else ["durable", "roomy"],
                    "presentationVariants": [{"title": f"Approved bag {i}"}]}
                cards.append({"productId": pid, "approvedFactIds": [], "observedHistoryReasonIds": []})
            sections = [{"products": cards[i:i+3]} for i in range(0, count, 3)]
            plan = {"requestedCardCount": count, "shortfall": 0, "sections": sections}
            rows = field_rows(plan, {"context": {}, "history": []}, catalog)
            self.assertEqual(sum(r["maxTokens"] for r in rows), {1: 64, 3: 128, 6: 224}[count])
            self.assertEqual(len(rows), count + len(sections))
            if count > 1:
                self.assertIn('"universalTags": ["durable"]', rows[0]["messages"][0]["content"])
            self.assertEqual([r["rowId"] for r in rows if r["kind"] == "card"], [f"card-{i}" for i in range(1, count+1)])

    def test_actual_token_collection_early_stop_and_compaction(self):
        class Generator:
            def __init__(self):
                self.steps = iter([[SimpleNamespace(uid=0, token=9, finish_reason="stop"), SimpleNamespace(uid=1, token=4, finish_reason=None)],
                    [SimpleNamespace(uid=1, token=5, finish_reason="length")], []])
            def next_generated(self):
                return next(self.steps)
        tokenizer = SimpleNamespace(decode=lambda ids: ",".join(map(str, ids)))
        records, stats = collect_native(Generator(), [0, 1], tokenizer)
        self.assertEqual(records[0]["tokenIds"], [])
        self.assertEqual(records[1]["tokenIds"], [4, 5])
        self.assertEqual(records[1]["arSteps"], 2)
        self.assertEqual(stats["maximumRowsEmittingTogether"], 2)

    @unittest.skipUnless(os.environ.get("RUN_NATIVE_PARALLEL_DECODE") == "1", "Requires root go for tiny native Metal parity")
    def test_tiny_hybrid_fork_unequal_suffix_batch_and_early_finish_token_parity(self):
        import mlx.core as mx
        from mlx_lm.models.qwen3_5 import Model, ModelArgs
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.generate import BatchGenerator, StopSequences
        from mlx_lm.sample_utils import make_sampler
        config = dict(model_type="qwen3_5", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=32, linear_num_value_heads=2,
            linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=8, full_attention_interval=2)
        mx.random.seed(97)
        model = Model(ModelArgs(model_type="qwen3_5", text_config=config))
        model.eval(); model.freeze()
        cache = make_prompt_cache(model)
        hidden = model(mx.array([[1, 2, 3]]), cache=cache)
        mx.eval(hidden, [c.state for c in cache]); mx.synchronize()
        base = {"cache": cache, "tokens": [1, 2, 3], "messages": [], "model": model, "consumed": False}
        def materialize(values):
            mx.eval(values); mx.synchronize()
        a = fork_complete_state(base, materialize)
        b = fork_complete_state(base, materialize)
        first = model(mx.array([[4, 5]]), cache=a["cache"])
        independent = model(mx.array([[6]]), cache=b["cache"])
        fresh = model(mx.array([[1, 2, 3, 6]]), cache=make_prompt_cache(model))
        mx.eval(first, independent, fresh); mx.synchronize()
        self.assertLess(float(mx.max(mx.abs(independent[:, -1] - fresh[:, -1])).item()), 1e-4)
        prompts = [[4], [6, 7, 8]]
        def run(prompts, budgets, forced_eos=False):
            generator = BatchGenerator(model, completion_batch_size=len(prompts), prefill_batch_size=len(prompts), prefill_step_size=2)
            branches = [fork_complete_state(base, materialize)["cache"] for _ in prompts]
            samplers = [lambda logits: mx.full((logits.shape[0],), 0, dtype=mx.int32) if forced_eos and i == 0 else mx.argmax(logits, axis=-1) for i in range(len(prompts))]
            # Bind sampler closures independently to avoid late-bound row selection.
            samplers = [(lambda logits: mx.zeros((logits.shape[0],), dtype=mx.int32)) if forced_eos and i == 0 else make_sampler(temp=0.0) for i in range(len(prompts))]
            try:
                uids = generator.insert(prompts, max_tokens=budgets, caches=branches, samplers=samplers,
                    logits_processors=[[] for _ in prompts], stop_sequences=[StopSequences([[0]]) if forced_eos and i == 0 else StopSequences() for i in range(len(prompts))])
                records, _ = collect_native(generator, uids, SimpleNamespace(decode=str))
                return [records[uid]["tokenIds"] for uid in uids]
            finally:
                generator.close()
        sequential = [run([prompt], [budget])[0] for prompt, budget in zip(prompts, [1, 3])]
        self.assertEqual(sequential, run(prompts, [1, 3]))
        eos_batch = run(prompts, [3, 3], True)
        self.assertEqual(eos_batch[0], [])
        self.assertEqual(eos_batch[1], run([prompts[1]], [3])[0])
        # Original prefix remains reusable after independent branch mutations.
        self.assertEqual(run([prompts[1]], [3])[0], sequential[1])


if __name__ == "__main__":
    unittest.main()
