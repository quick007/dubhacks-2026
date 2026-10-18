"""Small CPU-only checks of the prospective jacket boundary and frozen import."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("jacket_prepare", HERE / "prepare.py")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class PublicImportTest(unittest.TestCase):
    def test_multivalue_exact_membership(self):
        p = {"price": 120, "attributes": {"size": ["S", "L"], "waterproof": ["yes"]}}
        t = {"budget": 120, "requirements": [{"kind": "hard", "attribute": "size", "value": "M"}]}
        self.assertFalse(PREPARE.check_product(t, p)["eligible"])
        t["requirements"][0]["value"] = "L"
        self.assertTrue(PREPARE.check_product(t, p)["eligible"])

    def test_no_weatherproof_synonym_or_case_coercion(self):
        p = {"price": 1, "attributes": {"water-resistant": ["yes"], "color": ["navy"]}}
        t = {"requirements": [{"kind": "hard", "attribute": "waterproof", "value": "yes"}]}
        self.assertFalse(PREPARE.check_product(t, p)["eligible"])
        t["requirements"] = [{"kind": "hard", "attribute": "color", "value": "Navy"}]
        self.assertFalse(PREPARE.check_product(t, p)["eligible"])

    def test_preference_soft_budget_inclusive(self):
        p = {"price": 100, "attributes": {"color": ["olive"]}}
        t = {"budget": 100, "requirements": [{"kind": "preference", "attribute": "color", "value": "navy"}]}
        self.assertTrue(PREPARE.check_product(t, p)["eligible"])
        t["budget"] = 99
        self.assertFalse(PREPARE.check_product(t, p)["eligible"])

    def test_frozen_receipts_inputs_and_shortfalls(self):
        receipt = json.loads((HERE / "IMPORT_RECEIPT.json").read_text())
        for name, row in receipt["outputs"].items():
            value = json.loads((HERE / name).read_text())
            self.assertEqual(PREPARE.sha(PREPARE.canonical(value)), row["canonicalSha256"])
        self.assertEqual(receipt["preparerSha256"], hashlib.sha256((HERE / "prepare.py").read_bytes()).hexdigest())
        inputs = json.loads((HERE / "model_inputs.json").read_text())
        self.assertEqual(len(inputs["products"]), 6)
        self.assertEqual(len(inputs["tasks"]), 10)
        self.assertTrue(all(set(p) == {"name", "price", "attributes"} for p in inputs["products"]))
        self.assertTrue(all(set(t) <= {"query", "budget", "requirements"} for t in inputs["tasks"]))
        self.assertTrue(all(set(r) == {"label", "kind", "attribute", "value"}
                            for t in inputs["tasks"] for r in t["requirements"]))
        guards = json.loads((HERE / "guard_receipt.json").read_text())
        self.assertEqual([r["eligibleCount"] for r in guards["tasks"]], [1, 2, 2, 1, 2, 2, 1, 1, 2, 1])
        self.assertTrue(all(r["shortfallToThree"] > 0 and not r["rankingPerformed"] for r in guards["tasks"]))


if __name__ == "__main__":
    unittest.main()
