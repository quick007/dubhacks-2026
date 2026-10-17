"""Independent CPU/stub checks; no model/MLX import or GPU execution."""
import unittest
from .prefill_analysis import exact_append, lcp, stage_row


class PrefillAnalysisTests(unittest.TestCase):
    def test_lcp_does_not_make_entire_prefix_equivalent(self):
        self.assertEqual(lcp([1, 2, 3], [1, 2, 4, 5]), 2)
        self.assertEqual(lcp([1], [1, 2]), 1)
        self.assertEqual(lcp([], [1]), 0)

    def test_native_append_guard_exact_identity_and_single_consumption(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return [1, 2, 3, 4]
        result = exact_append(Tokenizer(), [1, 2], [], [{"content": "suffix"}])
        self.assertTrue(result["consumedOnce"])
        self.assertEqual(result["fullTokens"], 4)
        with self.assertRaises(ValueError):
            exact_append(Tokenizer(), [1, 9], [], [{"content": "suffix"}])
        from .fusion import append_generation_suffix
        model = object()
        state = {"model": model, "consumed": True}
        with self.assertRaises(ValueError):
            append_generation_suffix(Tokenizer(), [], state, model)

    def test_warm_and_cold_token_accounting_avoids_timing_overlap(self):
        timing = {"rankingPrefix": {"prefixTokens": 10}, "structuredMs": 2., "catalogCachePrepareMs": 1.,
                  "promptPreparationMs": .1, "prefillAndFirstTokenMs": 3., "autoregressiveAfterFirstTokenMs": 4.,
                  "generationMs": 7., "nativePromptMs": 2.5, "validationMs": .2, "fullCompletionMs": 10.5}
        arm = {"timing": timing, "generation": {"promptTokens": 5, "generatedTokens": 6},
               "cache": {"prefixTokens": 8, "hit": True}, "quality": {"allFieldsFulfilled": False}}
        row = {"caseId": "r", "source": "cache", "cards": 1, "arms": {"baseline": arm}}
        warm = stage_row(row, "baseline")
        self.assertEqual(warm["freshPrefillTokens"], 15)
        self.assertEqual(warm["logicalInputWorkTokens"], 23)
        self.assertAlmostEqual(warm["stagesMs"]["accountedNonoverlappingMs"], 10.3)
        arm["cache"]["hit"] = False
        self.assertEqual(stage_row(row, "baseline")["freshPrefillTokens"], 23)
        arm["cache"] = None
        self.assertEqual(stage_row(row, "baseline")["freshPrefillTokens"], 15)


if __name__ == "__main__":
    unittest.main()
