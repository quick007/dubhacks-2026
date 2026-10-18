"""CPU-only invariants for a heading-only native experiment; no model load."""
import copy
import unittest

from .heading_experiment import DIAGNOSIS_CASE_IDS, diagnosis_cases, heading_rows, heading_diagnostics, heading_arm
from .parallel_decode import build_rows


class HeadingExperimentTests(unittest.TestCase):
    def fixture(self, count):
        catalog = {}
        cards = []
        for index in range(count):
            key = str(index)
            catalog[key] = {"id": key, "title": f"Example {index + 10}L Daypack",
                            "tags": ["compact", "durable"] if index % 2 == 0 else ["compact", "roomy"],
                            "presentationVariants": [{"title": f"Approved daypack {index}"}]}
            cards.append({"productId": key, "approvedFactIds": [], "observedHistoryReasonIds": []})
        plan = {"requestedCardCount": count, "shortfall": 0,
                "sections": [{"products": cards[start:start + 3]} for start in range(0, count, 3)]}
        return plan, {"context": {}, "history": []}, catalog

    def test_only_heading_messages_change_for_all_existing_scales(self):
        for count in (1, 3, 6):
            plan, request, catalog = self.fixture(count)
            original_inputs = copy.deepcopy((plan, request, catalog))
            baseline = build_rows(plan, request, catalog)
            self.assertEqual(heading_rows(plan, request, catalog, compact=False), baseline)
            compact = heading_rows(plan, request, catalog)
            self.assertEqual((plan, request, catalog), original_inputs)
            self.assertEqual(sum(row["maxTokens"] for row in compact), {1: 64, 3: 128, 6: 224}[count])
            for old, new in zip(baseline, compact, strict=True):
                if old["kind"] != "heading":
                    self.assertEqual(old, new)
                else:
                    self.assertNotEqual(old["messages"], new["messages"])
                    self.assertEqual({k: v for k, v in old.items() if k != "messages"},
                                     {k: v for k, v in new.items() if k != "messages"})
                    prompt = new["messages"][0]["content"]
                    self.assertIn("3-5 words", prompt)
                    self.assertNotIn("Allowed framing words:", prompt)
                    self.assertIn("do not put product names or numbers", prompt)
            if count > 1:
                self.assertIn('"sharedTags": ["compact"]', compact[0]["messages"][0]["content"])
            self.assertIn('"selectedMemberNames": ["Example 10L Daypack"', compact[0]["messages"][0]["content"])

    def test_no_common_tags_remains_empty_not_union(self):
        plan, request, catalog = self.fixture(3)
        catalog["0"]["tags"] = ["durable"]
        catalog["1"]["tags"] = ["compact"]
        catalog["2"]["tags"] = ["roomy"]
        prompt = heading_rows(plan, request, catalog)[0]["messages"][0]["content"]
        self.assertIn('"sharedTags": []', prompt)

    def test_diagnostic_bounds_use_fixed_case_identity(self):
        cases = [{"caseId": key, "cards": 1} for key in reversed(DIAGNOSIS_CASE_IDS)]
        cases.append({"caseId": "later-unattempted", "cards": 6})
        self.assertEqual([c["caseId"] for c in diagnosis_cases(cases)], list(DIAGNOSIS_CASE_IDS))
        with self.assertRaises(ValueError):
            diagnosis_cases(cases[:-2])
        with self.assertRaises(ValueError):
            diagnosis_cases(cases + [cases[0]])

    def test_diagnostics_preserve_failed_raw_text_and_do_not_assert_acceptance(self):
        _, _, catalog = self.fixture(1)
        rows = [{"kind": "heading", "rowId": "heading-1", "text": "S1: Example 10L Daypack", "finishReason": "length"},
                {"kind": "title", "rowId": "title-1", "text": "T1: compact", "finishReason": "stop"}]
        original = copy.deepcopy(rows)
        diagnostic = heading_diagnostics(rows, catalog)[0]
        self.assertEqual(rows, original)
        self.assertEqual(diagnostic["finishReason"], "length")
        self.assertEqual(diagnostic["catalogNamesFound"], ["Example 10L Daypack"])
        self.assertTrue(diagnostic["containsNumber"])
        self.assertNotIn("accepted", diagnostic)
        rows[0]["text"] += "\nS2: extra"
        self.assertFalse(heading_diagnostics(rows, catalog)[0]["singleLineProtocol"])

    def test_arm_rejects_scope_expansion_before_using_backbone(self):
        with self.assertRaises(ValueError):
            heading_arm(None, None, {}, {}, 1, width=3)
        with self.assertRaises(ValueError):
            heading_arm(None, None, {}, {}, 3)


if __name__ == "__main__":
    unittest.main()
