"""CPU-only typed jacket inputs, exact guards and pinned import integrity."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from . import jacket_inputs as ji


def product():
    return {"name": "Harbor rain jacket", "price": 120,
            "attributes": {"size": ["S", "L"], "color": ["navy"], "waterproof": ["yes"]}}


def task():
    return {"query": "waterproof jacket in large, olive if possible", "budget": 120,
            "requirements": [
                {"label": "Size L", "kind": "hard", "attribute": "size", "value": "L"},
                {"label": "Olive", "kind": "preference", "attribute": "color", "value": "olive"}]}


class JacketInputTests(unittest.TestCase):
    def test_actual_snapshot_all_ten_counts_and_id_binding(self):
        loaded = ji.load_snapshot()
        self.assertEqual(len(loaded["products"]), 6)
        self.assertEqual(len(loaded["tasks"]), 10)
        counts = [ji.eligibility(t, loaded["products"])["eligibleCount"] for t in loaded["tasks"]]
        self.assertEqual(counts, [1, 2, 2, 1, 2, 2, 1, 1, 2, 1])
        ids = loaded["externalMapping"]["productIdsByInputIndex"]
        self.assertEqual([ids[i] for i in ji.eligibility(loaded["tasks"][1], loaded["products"])["eligibleProductIndices"]],
                         ["harbor", "coastline"])
        self.assertEqual(loaded["receipt"]["sourceCommit"], "65b5896b28d4901b95a691fd68436e79ad264e61")

    def test_exact_multivalue_hard_and_soft_preference(self):
        receipt = ji.guard_product(task(), product())
        self.assertTrue(receipt["eligible"])
        self.assertTrue(receipt["requirements"][0]["met"])
        self.assertFalse(receipt["requirements"][1]["met"])
        self.assertEqual(receipt["requirements"][0]["observedValues"], ["S", "L"])
        changed = task(); changed["requirements"][0]["value"] = "M"
        self.assertFalse(ji.guard_product(changed, product())["eligible"])
        changed["requirements"][0]["value"] = "l"
        self.assertFalse(ji.guard_product(changed, product())["eligible"])

    def test_waterproof_not_water_resistant_and_missing_attribute_fails(self):
        t = {"query": "rain jacket", "requirements": [
            {"label": "Waterproof", "kind": "hard", "attribute": "waterproof", "value": "yes"}]}
        p = product(); p["attributes"] = {"water-resistant": ["yes"]}
        receipt = ji.guard_product(t, p)
        self.assertFalse(receipt["eligible"])
        self.assertEqual(receipt["requirements"][0]["observedValues"], [])
        p["attributes"] = {"waterproof": ["no"]}
        self.assertFalse(ji.guard_product(t, p)["eligible"])

    def test_budget_inclusive_item_only_optional_zero_and_no_query_parsing(self):
        t = task(); self.assertTrue(ji.guard_product(t, product())["eligible"])
        t["budget"] = 119.99; self.assertFalse(ji.guard_product(t, product())["eligible"])
        del t["budget"]
        receipt = ji.guard_product(t, product())
        self.assertTrue(receipt["eligible"]); self.assertFalse(receipt["budgetSpecified"])
        t = {"query": "under $0 and must be waterproof", "requirements": []}
        self.assertTrue(ji.guard_product(t, product())["eligible"])
        t["budget"] = 0
        self.assertFalse(ji.guard_product(t, product())["eligible"])
        p = product(); p["price"] = 0
        self.assertTrue(ji.guard_product(t, p)["eligible"])
        self.assertEqual(ji.capabilities(), {"shipping": "unknown", "stock": "unknown", "imageSelection": "unsupported"})

    def test_versioned_text_preserves_all_values_catalog_task_independent(self):
        p = product(); before = copy.deepcopy(p)
        text = ji.catalog_text(p)
        encoded = json.loads(text)
        self.assertEqual(encoded["schemaVersion"], ji.CATALOG_TEXT_VERSION)
        self.assertEqual(encoded["product"], p)
        for t in (task(), {"query": "a different shopper", "requirements": []}):
            ji.task_text(t)
            self.assertEqual(text, ji.catalog_text(p))
        self.assertEqual(p, before)
        reordered = {"attributes": dict(reversed(list(p["attributes"].items()))), "price": 120, "name": p["name"]}
        self.assertEqual(text, ji.catalog_text(reordered))
        self.assertEqual(json.loads(ji.task_text(task()))["task"], task())

    def test_external_id_changes_leave_encoder_outputs_identical(self):
        loaded = ji.load_snapshot()
        before = ([ji.catalog_text(p) for p in loaded["products"]], [ji.task_text(t) for t in loaded["tasks"]])
        mapping = copy.deepcopy(loaded["externalMapping"])
        mapping["productIdsByInputIndex"] = [f"opaque-p-{i}" for i in range(6)]
        for i, entry in enumerate(mapping["tasksByInputIndex"]):
            entry["personaId"] = f"opaque-person-{i}"
            entry["requirementIds"] = [f"opaque-r-{j}" for j in range(len(entry["requirementIds"]))]
        ji.validate_mapping(loaded["products"], loaded["tasks"], mapping)
        after = ([ji.catalog_text(p) for p in loaded["products"]], [ji.task_text(t) for t in loaded["tasks"]])
        self.assertEqual(before, after)
        self.assertNotIn("opaque-", "".join(after[0] + after[1]))

    def test_metadata_is_rejected_instead_of_projected_into_prediction_text(self):
        for key in ("id", "productId", "imageKey", "shippingCost", "inStock", "presentationVariants", "score"):
            p = product(); p[key] = "HIDDEN_SENTINEL"
            with self.subTest(productExtra=key), self.assertRaises(ji.JacketInputError):
                ji.catalog_text(p)
        for key in ("id", "personaId", "bio", "goal", "traits", "history", "split", "outcome", "purchase", "optimalSteps"):
            t = task(); t[key] = "HIDDEN_SENTINEL"
            with self.subTest(taskExtra=key), self.assertRaises(ji.JacketInputError):
                ji.task_text(t)
        t = task(); t["requirements"][0]["id"] = "not-a-model-feature"
        with self.assertRaises(ji.JacketInputError): ji.task_text(t)

    def test_bad_numbers_bools_and_containers_fail_closed(self):
        for bad in (True, False, -1, float("nan"), float("inf"), -float("inf"), "120", None, [], 10 ** 10000):
            p = product(); p["price"] = bad
            t = task(); t["budget"] = bad
            with self.subTest(numberType=type(bad).__name__):
                with self.assertRaises(ji.JacketInputError): ji.validate_product(p)
                with self.assertRaises(ji.JacketInputError): ji.validate_task(t)
        for bad in (None, [], True, "jacket"):
            with self.assertRaises(ji.JacketInputError): ji.validate_product(bad)
            with self.assertRaises(ji.JacketInputError): ji.validate_task(bad)
        for attrs in ([], {1: ["M"]}, {"size": "M"}, {"size": ("M",)}, {"size": [True]},
                      {"size": [""]}, {"size": ["M", "M"]}):
            p = product(); p["attributes"] = attrs
            with self.subTest(attributes=attrs), self.assertRaises(ji.JacketInputError): ji.validate_product(p)

    def test_requirement_kinds_types_and_missing_fields_fail_closed(self):
        for bad in ("required", "HARD", "", None, True, [], 1):
            t = task(); t["requirements"][0]["kind"] = bad
            with self.subTest(kind=bad), self.assertRaises(ji.JacketInputError): ji.validate_task(t)
        for key, bad in (("query", True), ("query", " "), ("requirements", {}), ("requirements", [None])):
            t = task(); t[key] = bad
            with self.assertRaises(ji.JacketInputError): ji.validate_task(t)
        for field in ("label", "kind", "attribute", "value"):
            t = task(); del t["requirements"][0][field]
            with self.assertRaises(ji.JacketInputError): ji.validate_task(t)

    def test_mapping_duplicates_bad_ids_counts_and_extra_fields(self):
        loaded = ji.load_snapshot(); p, t, original = loaded["products"], loaded["tasks"], loaded["externalMapping"]
        mutations = [
            lambda m: m["productIdsByInputIndex"].pop(),
            lambda m: m["tasksByInputIndex"].pop(),
            lambda m: m["productIdsByInputIndex"].__setitem__(1, m["productIdsByInputIndex"][0]),
            lambda m: m["tasksByInputIndex"][1].__setitem__("personaId", m["tasksByInputIndex"][0]["personaId"]),
            lambda m: m["tasksByInputIndex"][0]["requirementIds"].__setitem__(1, "color"),
            lambda m: m["tasksByInputIndex"][0]["requirementIds"].pop(),
            lambda m: m["productIdsByInputIndex"].__setitem__(0, True),
            lambda m: m["tasksByInputIndex"][0].__setitem__("personaId", ""),
            lambda m: m.__setitem__("outcome", "hidden"),
        ]
        for index, mutate in enumerate(mutations):
            m = copy.deepcopy(original); mutate(m)
            with self.subTest(index=index), self.assertRaises(ji.JacketInputError): ji.validate_mapping(p, t, m)

    def test_defensive_copies_all_public_results(self):
        p, t = product(), task()
        clean = ji.validate_product(p); clean["attributes"]["size"].append("M")
        self.assertEqual(p["attributes"]["size"], ["S", "L"])
        clean_task = ji.validate_task(t); clean_task["requirements"][0]["value"] = "M"
        self.assertEqual(t["requirements"][0]["value"], "L")
        guard = ji.guard_product(t, p); guard["requirements"][0]["observedValues"].append("M")
        self.assertEqual(p["attributes"]["size"], ["S", "L"])
        loaded = ji.load_snapshot(); mapped = ji.validate_mapping(loaded["products"], loaded["tasks"], loaded["externalMapping"])
        mapped["productIdsByInputIndex"][0] = "changed"
        self.assertEqual(loaded["externalMapping"]["productIdsByInputIndex"][0], "harbor")
        loaded["products"][0]["attributes"]["size"].append("XXL")
        self.assertEqual(ji.load_snapshot()["products"][0]["attributes"]["size"], ["S", "L"])
        capability = ji.capabilities(); capability["stock"] = "in stock"
        self.assertEqual(ji.capabilities()["stock"], "unknown")

    def copy_snapshot(self, directory):
        for name in ("model_inputs.json", "external_mapping.json", "guard_receipt.json", "IMPORT_RECEIPT.json"):
            shutil.copyfile(ji.DEFAULT_SNAPSHOT / name, directory / name)

    def test_hash_tamper_and_matching_receipt_rewrite_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); self.copy_snapshot(directory)
            p = directory / "model_inputs.json"
            changed = json.loads(p.read_text()); changed["products"][0]["price"] = 0
            p.write_text(json.dumps(changed))
            receipt_path = directory / "IMPORT_RECEIPT.json"
            receipt = json.loads(receipt_path.read_text())
            import hashlib
            forged_hash = hashlib.sha256(ji.canonical(changed).encode()).hexdigest()
            receipt["modelInputSha256"] = forged_hash
            receipt["outputs"]["model_inputs.json"]["canonicalSha256"] = forged_hash
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaises(ji.JacketInputError): ji.load_snapshot(directory)

    def test_mapping_reorder_receipt_tamper_and_missing_artifacts_rejected(self):
        for name in ("external_mapping.json", "guard_receipt.json", "IMPORT_RECEIPT.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp); self.copy_snapshot(directory)
                path = directory / name; changed = json.loads(path.read_text())
                if name == "external_mapping.json": changed["productIdsByInputIndex"].reverse()
                elif name == "guard_receipt.json": changed["tasks"][0]["eligibleCount"] = 6
                else: changed["sourceCommit"] = "0" * 40
                path.write_text(json.dumps(changed))
                with self.assertRaises(ji.JacketInputError): ji.load_snapshot(directory)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ji.JacketInputError): ji.load_snapshot(temp)

    def test_duplicate_json_nonfinite_and_formatting_only_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); self.copy_snapshot(directory)
            path = directory / "model_inputs.json"; original = path.read_text()
            path.write_text('{"schemaVersion":"jacket-public-inputs.v1",' + original.lstrip()[1:])
            with self.assertRaisesRegex(ji.JacketInputError, "duplicate JSON key"): ji.load_snapshot(directory)
            path.write_text(original.replace('"price": 120', '"price": NaN', 1))
            with self.assertRaisesRegex(ji.JacketInputError, "nonfinite"): ji.load_snapshot(directory)
            path.write_text(json.dumps(json.loads(original), ensure_ascii=True))
            self.assertEqual(len(ji.load_snapshot(directory)["products"]), 6)


if __name__ == "__main__":
    unittest.main()
