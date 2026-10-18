"""Bounded CPU runtime-export and actual-store snapshot integrity controls."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from . import store_catalog_v1 as store


class ActualStoreCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.frozen = Path(cls.temp.name) / "frozen"
        cls.receipt = store.freeze(cls.frozen)
        cls.loaded = store.load_snapshot(cls.frozen)
        cls.raw = cls.loaded["rawExport"]
        cls.inputs = cls.loaded["inputs"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def copy_snapshot(self, directory):
        for path in self.frozen.iterdir():
            shutil.copyfile(path, directory / path.name)

    def test_actual_roster_views_assets_and_version_receipt(self):
        counts = self.receipt["counts"]
        self.assertEqual(counts, {"products": 16, "colorVariants": 48, "baseImageRecords": 65,
            "variantGalleryRecords": 195, "optionalColorPreviewRecords": 32,
            "uniqueAssetFiles": 81, "publicApiFixtureTasks": 10})
        self.assertEqual(self.receipt["sourceCommit"], store.COMMIT)
        self.assertEqual(store.digest(self.raw), store.RAW_EXPORT_SHA256)
        self.assertTrue(all(len(p["colorVariants"]) == 3 for p in self.inputs["products"]))
        for asset in self.receipt["assets"]:
            self.assertTrue((store.ROOT / "apps/storefront/public" / asset["src"].lstrip("/")).is_file())
        self.assertEqual(self.receipt["modelCalls"], 0)
        self.assertFalse(self.receipt["trainingPerformed"])

    def test_exact_default_and_nondefault_asset_crop_binding(self):
        for product, group, mapped in zip(self.raw["products"], self.raw["variants"], self.loaded["externalMapping"]["products"], strict=True):
            self.assertEqual(group["colorVariants"][0]["gallery"], self.raw["baseImages"][product["id"]])
            self.assertEqual(mapped["productId"], product["id"])
            for color, route in zip(group["colorVariants"], mapped["colorVariants"], strict=True):
                self.assertEqual(route["variantId"], color["variant"]["id"])
                for image, binding in zip(color["gallery"], route["images"], strict=True):
                    self.assertEqual(binding, {key: image[key] for key in ("id", "src", "position", "tile") if key in image})
                    if "tile" in binding:
                        tile = binding["tile"]
                        self.assertLess(tile["index"], tile["columns"] * tile["rows"])
                        self.assertGreater(tile["sheetAspect"], 0)
        mug = self.loaded["externalMapping"]["products"][0]
        tabletop = [color["images"][-1] for color in mug["colorVariants"]]
        self.assertEqual(len({image["src"] for image in tabletop}), 1)
        self.assertEqual([image["tile"]["index"] for image in tabletop], [0, 1, 2])

    def test_all_gallery_candidates_have_id_free_declared_text(self):
        total = 0
        for pi, product in enumerate(self.inputs["products"]):
            for ci, color in enumerate(product["colorVariants"]):
                for ii, image in enumerate(color["images"]):
                    text = store.catalog_text(self.inputs, pi, ci, ii)
                    encoded = json.loads(text)
                    self.assertEqual(encoded["product"]["name"], product["name"])
                    self.assertEqual(encoded["colorName"], color["colorName"])
                    self.assertEqual(encoded["image"], image)
                    for forbidden in ('"id":', '"src":', '"tile":', '"position":', '"variantId":', '"swatch":'):
                        self.assertNotIn(forbidden, text)
                    self.assertIn("pixels unexamined", text)
                    total += 1
        self.assertEqual(total, 195)

    def test_metadata_not_promoted_into_product_capabilities_or_persona_intent(self):
        for source, public in zip(self.raw["products"], self.inputs["products"], strict=True):
            self.assertEqual(set(public), {"name", "category", "itemPrice", "searchTerms", "colorVariants"})
            self.assertEqual(public["category"], source["category"])
            if "scenario" in source:
                for color in public["colorVariants"]:
                    for image in color["images"]:
                        self.assertNotIn(source["scenario"], image["tags"])
        for task in self.inputs["publicApiFixtureTasks"]:
            self.assertTrue(set(task) <= {"query", "budget", "requirements"})
            for requirement in task["requirements"]:
                self.assertEqual(set(requirement), {"label", "kind", "attribute", "value"})
        self.assertEqual(self.inputs["capabilities"]["sizes"], "unknown")
        self.assertEqual(self.inputs["capabilities"]["stock"], "unknown")
        self.assertIn("not established", self.inputs["taskProvenance"])

    def test_routing_identity_renaming_keeps_predictive_inputs_identical(self):
        raw = copy.deepcopy(self.raw)
        old = raw["products"][0]["id"]; new = "opaque-product-routing"
        raw["products"][0]["id"] = new
        raw["baseImages"][new] = raw["baseImages"].pop(old)
        group = raw["variants"][0]; group["productId"] = new
        image_renames = {}
        def rename(image):
            identifier = image["id"]
            image["id"] = image_renames.setdefault(identifier, f"opaque-image-{len(image_renames)}")
        for image in raw["baseImages"][new]: rename(image)
        for index, color in enumerate(group["colorVariants"]):
            color["variant"]["id"] = f"opaque-color-{index}"
            color["variant"]["productId"] = new
            for image in color["gallery"]: rename(image)
            if "image" in color["variant"]: rename(color["variant"]["image"])
        a, _ = store.project_export(self.raw)
        b, external = store.project_export(raw)
        self.assertEqual(a, b)
        self.assertEqual(external["products"][0]["productId"], new)

    def test_shipping_threshold_visibility_and_cart_scope(self):
        probes = self.raw["shipping"]["probes"]
        for subtotal, amount in ((49.99, 5), (50, 5), (50.01, 0)):
            visible = next(p for p in probes if p["subtotal"] == subtotal and p["showShipping"])
            hidden = next(p for p in probes if p["subtotal"] == subtotal and not p["showShipping"])
            self.assertEqual(visible["quote"]["amount"], amount)
            self.assertIsNone(hidden["quote"])
            self.assertEqual(hidden["displayedTotal"], subtotal)
        self.assertEqual(self.inputs["shippingTerms"]["scope"], "cart-subtotal")
        self.assertIn("not established", self.inputs["shippingTerms"]["hiddenQuote"])
        self.assertEqual(self.inputs["capabilities"]["perItemShippingCost"], "unsupported")

    def test_strict_bad_types_owner_duplicates_and_crop_bounds(self):
        def set_price(raw, value): raw["products"][0]["price"] = value
        mutations = [lambda raw: set_price(raw, True), lambda raw: set_price(raw, float("inf")),
            lambda raw: raw["products"][0].__setitem__("inStock", True),
            lambda raw: raw["products"][1].__setitem__("id", raw["products"][0]["id"]),
            lambda raw: raw["variants"][0].__setitem__("productId", "wrong-owner"),
            lambda raw: raw["variants"][0]["colorVariants"][0]["variant"].__setitem__("productId", "wrong-owner"),
            lambda raw: raw["variants"][0]["colorVariants"][1]["gallery"][-1]["tile"].__setitem__("index", 999),
            lambda raw: raw["variants"][0]["colorVariants"][1]["gallery"][-1]["tile"].__setitem__("rows", True),
            lambda raw: raw["variants"][0]["colorVariants"][1]["gallery"][0].__setitem__("src", "https://external.invalid/a.jpg"),
            lambda raw: raw["variants"][0]["colorVariants"][1]["gallery"][0].__setitem__("context", []),
            lambda raw: raw["shipping"]["probes"][0].__setitem__("showShipping", 0)]
        for index, mutate in enumerate(mutations):
            raw = copy.deepcopy(self.raw); mutate(raw)
            with self.subTest(index=index), self.assertRaises(store.StoreCatalogError): store.validate_export(raw)

    def test_encoder_rejects_id_injection_bad_versions_and_bad_indices(self):
        for field in ("id", "src", "tile", "score", "personaId"):
            inputs = copy.deepcopy(self.inputs)
            inputs["products"][0]["colorVariants"][0]["images"][0][field] = "hidden-metadata"
            with self.subTest(field=field), self.assertRaises(store.StoreCatalogError): store.catalog_text(inputs, 0, 0, 0)
        inputs = copy.deepcopy(self.inputs); inputs["schemaVersion"] = "future-unsupported"
        with self.assertRaises(store.StoreCatalogError): store.catalog_text(inputs, 0, 0, 0)
        for index in (True, -1, 1000, "0"):
            with self.assertRaises(store.StoreCatalogError): store.catalog_text(self.inputs, index, 0, 0)

    def test_defensive_projection_and_source_immutability(self):
        before = copy.deepcopy(self.raw)
        inputs, routes = store.project_export(self.raw)
        inputs["products"][0]["colorVariants"][0]["images"][0]["tags"].append("changed")
        routes["products"][0]["colorVariants"][0]["images"][0]["src"] = "changed"
        self.assertEqual(self.raw, before)
        self.assertEqual(store._source_receipts(require_worktree=True), self.receipt["sources"])

    def test_refuse_overwrite_before_runtime_execution(self):
        with patch.object(store, "capture_runtime", side_effect=AssertionError("must not execute")):
            with self.assertRaises(FileExistsError): store.freeze(self.frozen)

    def test_raw_route_and_projection_tampering_even_with_forged_receipt(self):
        for name in ("runtime_export.json", "model_inputs.json", "external_mapping.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp); self.copy_snapshot(directory)
                path = directory / name; value = json.loads(path.read_text())
                if name == "runtime_export.json": value["variants"][0]["colorVariants"][1]["gallery"][-1]["tile"]["index"] = 2
                elif name == "model_inputs.json": value["products"][0]["itemPrice"] = 0
                else: value["products"][0]["productId"] = "forged-id"
                path.write_text(json.dumps(value))
                receipt_path = directory / "IMPORT_RECEIPT.json"
                receipt = json.loads(receipt_path.read_text()); receipt["outputs"][name]["canonicalSha256"] = store.digest(value)
                receipt_path.write_text(json.dumps(receipt))
                with self.assertRaises(store.StoreCatalogError): store.load_snapshot(directory)

    def test_receipt_assets_counts_training_calls_and_source_corruption(self):
        mutations = [lambda r: r["counts"].__setitem__("variantGalleryRecords", 48),
            lambda r: r["assets"][0].__setitem__("sha256", "0" * 64),
            lambda r: r.__setitem__("modelCalls", 1), lambda r: r.__setitem__("modelCalls", False),
            lambda r: r.__setitem__("trainingPerformed", True), lambda r: r.__setitem__("importerSha256", "0" * 64),
            lambda r: r.__setitem__("exportProgramSha256", "0" * 64),
            lambda r: r["sources"][0].__setitem__("sha256", "0" * 64)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp); self.copy_snapshot(directory)
                path = directory / "IMPORT_RECEIPT.json"; receipt = json.loads(path.read_text())
                mutate(receipt); path.write_text(json.dumps(receipt))
                with self.assertRaises(store.StoreCatalogError): store.load_snapshot(directory)

    def test_asset_bytes_corruption_rejected_without_mutating_storefront(self):
        asset = self.receipt["assets"][0]
        target = store.ROOT / "apps/storefront/public" / asset["src"].lstrip("/")
        original = Path.read_bytes
        def changed(path):
            data = original(path)
            return data + b"tampered" if path == target else data
        with patch.object(Path, "read_bytes", changed):
            with self.assertRaisesRegex(store.StoreCatalogError, "asset differs"):
                store.load_snapshot(self.frozen)

    def test_duplicate_json_and_nonfinite_content_rejected(self):
        for injected in ('{"products":[],', '{"bad":NaN,'):
            with tempfile.TemporaryDirectory() as temp:
                directory = Path(temp); self.copy_snapshot(directory)
                path = directory / "runtime_export.json"; original = path.read_text()
                path.write_text(injected + original.lstrip()[1:])
                with self.assertRaises(store.StoreCatalogError): store.load_snapshot(directory)


if __name__ == "__main__":
    unittest.main()
