"""Leakage, cutoff, uncertainty, and determinism checks for the synthetic fixture."""
import hashlib
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from .generate import SPLITS, generate, write_dataset


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate(2026)
        cls.truth = {row["recordId"]: row for row in cls.data["truths"]}
        cls.products = {p["id"]: p for p in cls.data["catalog"]["products"]}

    def test_split_before_examples_has_no_family_or_user_overlap(self):
        families, users = {}, {}
        for split, records in self.data["public"].items():
            families[split] = {self.truth[r["recordId"]]["personaFamilyId"] for r in records}
            users[split] = {r["userId"] for r in records}
            self.assertEqual(len(records), 240 if split == "train" else 60)
        for index, left in enumerate(SPLITS):
            for right in SPLITS[index + 1:]:
                self.assertFalse(families[left] & families[right])
                self.assertFalse(users[left] & users[right])
        self.assertEqual([len(families[s]) for s in SPLITS], [40, 10, 10])
        self.assertEqual(len({r["recordId"] for r in self.data["truths"]}), 360)

    def test_public_whitelist_and_cutoff(self):
        allowed = {"schemaVersion", "synthetic", "recordId", "userId", "asOf", "context",
                   "history", "candidateProductIds", "missingFields", "provenance"}
        private_keys = {"hiddenPreferences", "hiddenConstraints", "personaFamilyId", "split",
                        "relevance", "bestProductId", "futureOutcome", "observationMode",
                        "counterfactualTwinGroup", "labelProvenance"}
        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(v) for v in value.values()), set())
            if isinstance(value, list):
                return set().union(*(keys(v) for v in value), set())
            return set()
        for records in self.data["public"].values():
            for row in records:
                self.assertEqual(set(row), allowed)
                self.assertFalse(keys(row) & private_keys)
                self.assertTrue(row["synthetic"])
                self.assertEqual(len(set(row["candidateProductIds"])), 6)
                self.assertEqual(set(row["context"]), {"task", "statedBudget", "statedRequirements", "statedPreferences"})
                for event in row["history"]:
                    self.assertLessEqual(event["at"], row["asOf"])
                    self.assertIn(event["productId"], self.products)
                    self.assertGreaterEqual(event["confidence"], 0)
                    self.assertLessEqual(event["confidence"], 1)
                self.assertGreater(self.truth[row["recordId"]]["futureOutcome"]["at"], row["asOf"])

    def test_supervision_is_separate_and_consistent(self):
        self.assertNotIn("test", self.data["labels"])
        for split, labels in self.data["labels"].items():
            self.assertEqual({r["recordId"] for r in labels},
                             {r["recordId"] for r in self.data["public"][split]})
            for label in labels:
                truth = self.truth[label["recordId"]]
                self.assertEqual(label["relevance"], truth["relevance"])
                self.assertEqual(label["bestProductId"], truth["bestProductId"])
                self.assertEqual(set(label), {"recordId", "relevance", "bestProductId"})
                self.assertTrue(set(label["relevance"].values()) <= {0, 1, 2, 3})
        self.assertTrue(any(r["bestProductId"] is None for r in self.data["truths"]))

    def test_hidden_truth_never_fills_missing_observations(self):
        rows = [r for records in self.data["public"].values() for r in records]
        self.assertEqual(sum(not r["history"] for r in rows), 120)
        self.assertTrue(any(r["context"]["statedBudget"] is None and
                            self.truth[r["recordId"]]["hiddenConstraints"]["totalBudget"] is not None for r in rows))
        self.assertTrue(any(not r["context"]["statedRequirements"] and
                            self.truth[r["recordId"]]["hiddenConstraints"]["requiredTags"] for r in rows))
        self.assertTrue(any(event["confidence"] < .5 for r in rows for event in r["history"]))
        self.assertTrue(any("dwellSeconds" not in event for r in rows for event in r["history"]))
        self.assertTrue(any(event.get("dwellSeconds", 1) is None for r in rows for event in r["history"]))
        self.assertTrue(any(len(r["history"]) == 1 for r in rows))

    def test_identical_observations_can_have_different_hidden_best(self):
        twins = defaultdict(list)
        for records in self.data["public"].values():
            for row in records:
                truth = self.truth[row["recordId"]]
                if truth["counterfactualTwinGroup"]:
                    twins[truth["counterfactualTwinGroup"]].append(row)
        self.assertEqual(len(twins), 60)
        for pair in twins.values():
            self.assertEqual(len(pair), 2)
            a, b = pair
            for field in ("context", "history", "candidateProductIds", "asOf", "missingFields", "provenance"):
                self.assertEqual(a[field], b[field])
            self.assertNotEqual(self.truth[a["recordId"]]["bestProductId"], self.truth[b["recordId"]]["bestProductId"])

    def test_catalog_variants_only_repeat_factual_tags(self):
        self.assertEqual(len(self.products), 18)
        for product in self.products.values():
            self.assertGreaterEqual(product["shippingCost"], 0)
            for variant in product["presentationVariants"]:
                self.assertTrue(set(variant["facts"]) <= set(product["tags"]))
                self.assertTrue(variant["title"].startswith(product["title"]))
                self.assertTrue(variant["imageKey"].startswith("fixture/"))

    def test_reproducible_files_and_seed_sensitive_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary) / "left", Path(temporary) / "right"
            a, b = write_dataset(left, 2026), write_dataset(right, 2026)
            self.assertEqual(a, b)
            self.assertFalse((left / "labels/test.jsonl").exists())
            for relative, expected_hash in a["sha256"].items():
                self.assertEqual((left / relative).read_bytes(), (right / relative).read_bytes())
                self.assertEqual(hashlib.sha256((left / relative).read_bytes()).hexdigest(), expected_hash)
            self.assertEqual(json.loads((left / "manifest.json").read_text()), a)
            stages = a["trainStageRecordIds"]
            self.assertEqual([len(stages[s]) for s in ("initial", "adaptation")], [120, 120])
            initial_families = {self.truth[r]["personaFamilyId"] for r in stages["initial"]}
            adaptation_families = {self.truth[r]["personaFamilyId"] for r in stages["adaptation"]}
            self.assertFalse(initial_families & adaptation_families)
            self.assertEqual(initial_families | adaptation_families,
                             {r["personaFamilyId"] for r in self.data["truths"] if r["split"] == "train"})
        self.assertNotEqual(self.data["families"], generate(2027)["families"])


if __name__ == "__main__":
    unittest.main()
