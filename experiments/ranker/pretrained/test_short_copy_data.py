"""Fast CPU checks; teacher fixtures are supervision, never generated output."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from .catalog_fusion.generalization_v2 import query_text_v2
from .catalog_fusion.inputs import canonical
from .semantic_copy_pairs.generator import TAGS
from .short_copy_data import (LIMITS, build, build_layout, expected_fields,
                              freeze, load_dataset, public_context, validate_rows)


class ShortCopyDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.fields, cls.coverage = build()

    def fixture(self, split="train", index=0):
        case = copy.deepcopy(self.cases[split][index])
        rows = [{**copy.deepcopy(f), "text": f["targetValue"], "finishReason": "stop"}
                for f in self.fields[split] if f["caseId"] == case["caseId"]]
        return case, rows

    def validate(self, case, rows):
        return validate_rows(rows, case["pendingPagePlan"], case["request"], case["catalog"])

    def test_fixed_complete_rosters_caps_and_all18_supported_body_atoms(self):
        self.assertEqual([len(self.fields[s]) for s in ("train", "dev", "test")], [240, 60, 60])
        self.assertEqual(set(self.coverage["train"]["positiveObservedBodyCountsByAtom"]), set(TAGS))
        self.assertEqual(self.coverage["train"]["bodyTargets"], 72)
        for split in self.cases:
            for index in range(len(self.cases[split])):
                _, rows = self.fixture(split, index)
                self.assertEqual([r["field"] for r in rows], expected_fields())
                self.assertEqual(len(set(r["field"] for r in rows)), 20)
                self.assertEqual(sum(r["maxTokens"] for r in rows[:10]), LIMITS["firstThreeOutputCap"])
                self.assertEqual(sum(r["maxTokens"] for r in rows), LIMITS["fullSixOutputCap"])

    def test_all_teacher_views_pass_unchanged_contract_and_remain_raw(self):
        for split in self.cases:
            for index in range(len(self.cases[split])):
                case, rows = self.fixture(split, index); before = copy.deepcopy(rows)
                result = self.validate(case, rows)
                self.assertTrue(result["firstThreeValid"])
                self.assertTrue(result["fullSixValid"])
                self.assertEqual(result["rawRowsUnmodified"], before)
                self.assertEqual(rows, before)

    def test_fullsix_failure_does_not_drop_field_or_corrupt_firstthree(self):
        case, rows = self.fixture()
        next(r for r in rows if r["field"] == "C6")["text"] += "\n"
        result = self.validate(case, rows)
        self.assertTrue(result["firstThreeValid"])
        self.assertFalse(result["fullSixValid"])
        self.assertIsNone(result["fullSixPayload"])
        self.assertEqual(result["rawRowsUnmodified"], rows)
        result = self.validate(case, [r for r in rows if r["field"] != "C1"])
        self.assertFalse(result["firstThreeValid"])
        self.assertFalse(result["all20RowsPresent"])

    def test_strict_none_identity_and_independent_body_support_are_not_repaired(self):
        case, original = self.fixture()
        none = next(r["field"] for r in original if r["kind"] == "badge" and r["text"] == "NONE")
        for bad in ("NONE ", "NONE\n", "B1: NONE", "none"):
            rows = copy.deepcopy(original);next(r for r in rows if r["field"] == none)["text"] = bad
            self.assertFalse(self.validate(case, rows)["fullSixValid"], bad)
        rows = copy.deepcopy(original)
        next(r for r in rows if r["field"] == "T1")["text"] = "Meet the compact everyday travel choice"
        self.assertFalse(self.validate(case, rows)["firstThreeValid"])
        card = next(c for c in case["publicEvidence"]["cards"] if c["supportedObservedTags"])
        rows = copy.deepcopy(original)
        next(r for r in rows if r["field"] == f'C{card["index"]}')["text"] = "A daypack for everyday travel."
        self.assertFalse(self.validate(case, rows)["fullSixValid"])

    def test_scheduler_injection_duplicate_and_unknown_fields_fail_closed(self):
        case, rows = self.fixture()
        for extra in (rows[0], {**rows[0], "fields": ["S9"]}, {**rows[0], "fields": ["S1", "C1"]}):
            with self.assertRaises(ValueError):self.validate(case, rows + [extra])
        next(r for r in rows if r["field"] == "C1")["text"] = "C1: A compact daypack for travel."
        self.assertFalse(self.validate(case, rows)["firstThreeValid"])

    def test_full_public_context_once_focused_root_exact_no_teacher_or_private_ids(self):
        for split in self.cases:
            for c in self.cases[split]:
                before = copy.deepcopy(c)
                common, rows, evidence = build_layout(c["pendingPagePlan"], c["request"], c["catalog"])
                self.assertEqual(c, before)
                self.assertEqual(c["rawPrefixText"], query_text_v2(c["request"], c["catalog"], "focused"))
                self.assertIn(canonical(public_context(c["request"], c["catalog"])), common["content"])
                inputs = canonical([c["rawPrefixText"], common, [r["messages"] for r in rows]])
                for forbidden in (c["caseId"], "personaFamilyId", "targetValue", "bestProductId"):
                    self.assertNotIn(forbidden, inputs)
                self.assertTrue(all(len(r["messages"]) == 1 for r in rows))
                self.assertTrue(all(common not in r["messages"] for r in rows))
                self.assertEqual(evidence, c["publicEvidence"])

    def test_headings_require_universal_fact_or_neutral_and_body_positive_support(self):
        for split in self.cases:
            for c in self.cases[split]:
                targets = {f["field"]: f["targetValue"] for f in self.fields[split] if f["caseId"] == c["caseId"]}
                for index, section in enumerate(c["publicEvidence"]["sections"], 1):
                    tags = section["universalTags"]
                    if tags:self.assertIn(tags[0], targets[f"S{index}"].lower())
                    else:self.assertEqual(targets[f"S{index}"], "Everyday travel choices")
                for card in c["publicEvidence"]["cards"]:
                    if card["supportedObservedTags"]:
                        self.assertIn(card["supportedObservedTags"][0], targets[f'C{card["index"]}'])

    def test_freeze_exclusive_and_hash_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "packet"; manifest = freeze(data)
            self.assertFalse(manifest["nativeExecution"])
            self.assertFalse(manifest["trainingExecuted"])
            self.assertEqual(load_dataset(data)[0], manifest)
            with self.assertRaises(FileExistsError):freeze(data)
            payload = data / "fields.train.jsonl"
            payload.write_text(payload.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "payload changed"):load_dataset(data)


if __name__ == "__main__":unittest.main()
