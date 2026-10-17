"""Determinism, held-out groups, input boundaries and presentation validity."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .generate import COUNTS, SPLITS, generate, write_dataset


class DatasetV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate(2026)
        cls.truth = {r["recordId"]: r for r in cls.data["truths"]}
        cls.products = {p["id"]: p for p in cls.data["catalog"]["products"]}

    def test_split_groups_products_users_and_combinations_are_disjoint(self):
        groups = {s: {k: set() for k in ("users", "families", "combinations", "products", "facts")} for s in SPLITS}
        for split, rows in self.data["public"].items():
            self.assertEqual(len(rows), COUNTS[split])
            for row in rows:
                truth = self.truth[row["recordId"]]
                groups[split]["users"].add(row["userId"])
                groups[split]["families"].add(truth["personaFamilyId"])
                groups[split]["combinations"].add(tuple(truth["preferenceCombination"]))
                groups[split]["products"].update(row["candidateProductIds"])
                for pid in row["candidateProductIds"]:
                    p = self.products[pid]
                    groups[split]["facts"].add(json.dumps({k: p[k] for k in ("title", "description", "price", "shippingCost", "tags")}, sort_keys=True))
                self.assertTrue({h["productId"] for h in row["history"]} <= groups[split]["products"])
        for i, left in enumerate(SPLITS):
            for right in SPLITS[i + 1:]:
                for key in groups[left]:
                    self.assertFalse(groups[left][key] & groups[right][key], (left, right, key))
        self.assertEqual(sum(len(groups[s]["users"]) for s in SPLITS), 42)
        self.assertEqual([len(groups[s]["families"]) for s in SPLITS], [12, 3, 6])

    def test_public_whitelist_no_future_or_private_fields(self):
        allowed = {"schemaVersion", "synthetic", "recordId", "userId", "asOf", "context", "history",
                   "candidateProductIds", "presentationCompatibility", "missingFields", "provenance"}
        forbidden = {"hiddenPreferences", "hiddenConstraints", "hiddenPresentationPreferences", "split",
                     "personaFamilyId", "preferenceCombination", "relevance", "bestProductId",
                     "presentationScores", "bestPresentationAction", "futureOutcome", "observationMode"}
        def all_keys(value):
            result = set()
            if isinstance(value, dict):
                result |= set(value)
                for child in value.values():
                    result |= all_keys(child)
            if isinstance(value, list):
                for child in value:
                    result |= all_keys(child)
            return result
        for rows in self.data["public"].values():
            for row in rows:
                self.assertEqual(set(row), allowed)
                self.assertFalse(all_keys(row) & forbidden)
                self.assertEqual(row["schemaVersion"], 2)
                self.assertTrue(row["synthetic"])
                self.assertEqual(len(set(row["candidateProductIds"])), 4)
                for event in row["history"]:
                    self.assertLessEqual(event["at"], row["asOf"])
                    self.assertIsInstance(event["description"], str)
                self.assertGreater(self.truth[row["recordId"]]["futureOutcome"]["at"], row["asOf"])

    def test_presentation_fact_compatibility_and_labels(self):
        incompatibles = 0
        for rows in self.data["public"].values():
            for row in rows:
                truth = self.truth[row["recordId"]]
                surface = row["context"]["presentationSurface"]
                for pid in row["candidateProductIds"]:
                    p = self.products[pid]
                    self.assertEqual(len(p["presentationVariants"]), 2)
                    for v in p["presentationVariants"]:
                        self.assertIn(v["presentationStyle"], {"detail-view", "in-use-view"})
                        self.assertTrue(v["imageKey"].startswith("fictional-descriptor/"))
                        self.assertIn(p["title"], v["title"])
                        self.assertTrue(set(p["tags"]) <= set(v["facts"]))
                        self.assertEqual(row["presentationCompatibility"][pid][v["id"]], surface in v["compatibleSurfaces"])
                        if not row["presentationCompatibility"][pid][v["id"]]:
                            self.assertEqual(truth["presentationScores"][pid][v["id"]], 0)
                            incompatibles += 1
                action = truth["bestPresentationAction"]
                self.assertEqual(action["productId"], truth["bestProductId"])
                self.assertTrue(row["presentationCompatibility"][action["productId"]][action["variantId"]])
        self.assertGreater(incompatibles, 0)

    def test_supervision_separate_and_train_stages_family_disjoint(self):
        self.assertNotIn("test", self.data["labels"])
        for split, labels in self.data["labels"].items():
            self.assertEqual({x["recordId"] for x in labels}, {x["recordId"] for x in self.data["public"][split]})
            for label in labels:
                self.assertEqual(set(label), {"recordId", "relevance", "bestProductId", "presentationScores", "bestPresentationAction"})
                self.assertEqual(label["relevance"], self.truth[label["recordId"]]["relevance"])
        stages = self.data["trainStageRecordIds"]
        self.assertEqual([len(stages[s]) for s in ("initial", "adaptation")], [12, 12])
        groups = [{self.truth[r]["personaFamilyId"] for r in stages[s]} for s in ("initial", "adaptation")]
        self.assertFalse(groups[0] & groups[1])
        self.assertEqual([len(g) for g in groups], [6, 6])

    def test_missing_noise_uncertainty_and_position_diagnostics(self):
        rows = [r for split in self.data["public"].values() for r in split]
        self.assertTrue(any(not r["history"] for r in rows))
        self.assertTrue(any(r["context"]["statedBudget"] is None for r in rows))
        self.assertTrue(any(e["confidence"] == .35 for r in rows for e in r["history"]))
        self.assertTrue(any("dwellSeconds" not in e for r in rows for e in r["history"]))
        self.assertTrue(any(e.get("dwellSeconds", 0) is None for r in rows for e in r["history"]))
        self.assertTrue(any(not r["context"]["presentationPreferences"] and self.truth[r["recordId"]]["hiddenPresentationPreferences"]["preferredStyle"] for r in rows))
        for split, records in self.data["public"].items():
            positions = Counter(r["candidateProductIds"].index(self.truth[r["recordId"]]["bestProductId"]) for r in records)
            self.assertLessEqual(max(positions.values()) - min(positions.values()), 1)
            self.assertEqual({str(i): positions[i] for i in range(4)}, self.data["positionCounts"][split])
            # Random variant positions are disclosed rather than falsely claimed balanced.
            self.assertTrue(all(n > 0 for n in self.data["actionPositionCounts"][split].values()))

    def test_counterfactual_preferences_change_choice_with_fixed_catalog_and_history(self):
        rows = {r["recordId"]: r for r in self.data["diagnostics"]}
        truth = {r["recordId"]: r for r in self.data["diagnosticTruths"]}
        self.assertEqual(len(rows), 6)
        self.assertFalse(set(rows) & set(self.truth))
        for pair in self.data["diagnosticPairs"]:
            a, b = [rows[r] for r in pair["recordIds"]]
            for field in ("candidateProductIds", "history", "asOf", "presentationCompatibility", "provenance"):
                self.assertEqual(a[field], b[field])
            self.assertNotEqual(a["context"]["statedPreferences"], b["context"]["statedPreferences"])
            self.assertNotEqual(truth[a["recordId"]]["bestProductId"], truth[b["recordId"]]["bestProductId"])

    def test_budget_stock_requirements_include_real_hard_negatives_and_prices_do_not_encode_roles(self):
        failures = Counter()
        for rows in self.data["public"].values():
            for row in rows:
                suitable = 0
                for pid in row["candidateProductIds"]:
                    p = self.products[pid]
                    stock = p["inStock"]
                    budget = row["context"]["statedBudget"]
                    affordable = budget is None or p["price"] + p["shippingCost"] <= budget
                    meets = set(row["context"]["statedRequirements"]) <= set(p["tags"])
                    failures["stock"] += not stock
                    failures["budget"] += not affordable
                    failures["requirements"] += not meets
                    suitable += stock and affordable and meets
                self.assertGreater(suitable, 0)
        self.assertTrue(all(failures[k] > 0 for k in ("stock", "budget", "requirements")))
        # Regression: cents previously encoded construction ordinal and hence role.
        cents = [round(self.products[a["productId"]]["price"] * 100) % 100
                 for a in self.data["productAssignments"]]
        self.assertLess(sum(c == (i + 1) % 100 for i, c in enumerate(cents)), len(cents) // 4)

    def test_deterministic_manifest_file_hashes_and_seed_variation(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / "a", Path(temp) / "b"
            a, b = write_dataset(left, 2026), write_dataset(right, 2026)
            self.assertEqual(a, b)
            self.assertFalse((left / "labels/test.jsonl").exists())
            self.assertEqual(json.loads((left / "manifest.json").read_text()), a)
            for name, digest in a["sha256"].items():
                self.assertEqual((left / name).read_bytes(), (right / name).read_bytes())
                self.assertEqual(hashlib.sha256((left / name).read_bytes()).hexdigest(), digest)
        self.assertNotEqual(self.data["familyAssignments"], generate(2027)["familyAssignments"])


if __name__ == "__main__":
    unittest.main()
