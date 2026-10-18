"""Frozen actual-store public catalog projection via installed pure TS modules.

No TS evaluator/parser is invented and no dependency is installed. Capture uses
the installed Node+tsx runtime for reviewed configuration modules only. Loading
and predictive encoders are stdlib-only. Image semantics are fixture declarations;
pixels are neither loaded nor interpreted. IDs/asset routes/crops stay external.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

from .jacket_inputs import validate_task

VERSION = "actual-store-public-inputs.v1"
COMMIT = "e172f6057aa76c58089f5e9b86b214a1543052a6"
DEFAULT_OUT = Path(__file__).resolve().parent / "store_catalog_v1"
ROOT = Path(__file__).resolve().parents[3]
# Pinned after independently capturing the reviewed runtime export, before freeze.
RAW_EXPORT_SHA256 = "fa440e919e1f36890a1c97d8e05ea01bd4ce46811eaed57c0dc61db7b049a0d8"
MODEL_INPUTS_SHA256 = "205572ed535b6f40cd56ede51208c348ad05998ad4d205f193a8b5bfe3e79fb7"
MAPPING_SHA256 = "4c5caa231e70945ac65ae0804db7c9ea37a1c05387262909f6a930f660224fee"
ASSET_RECEIPTS_SHA256 = "a4e7f5289b362d565cffd3d41e6e8a6d7af35135334c41e13178fc70b6519b90"
SOURCES = (
    "apps/storefront/src/config.ts", "apps/storefront/src/catalog-expansion.ts",
    "apps/storefront/src/variants.ts", "apps/storefront/src/images.ts",
    "apps/storefront/src/color-gallery-layouts.json", "apps/storefront/src/personalization.ts",
    "apps/storefront/src/shipping.ts", "apps/storefront/src/ProductPhoto.tsx",
    "apps/storefront/src/ProductGallery.tsx", "apps/storefront/src/ProductCard.tsx",
    "packages/contracts/src/index.ts", "packages/contracts/src/local.ts",
    "packages/contracts/package.json", "apps/api/fixtures/catalog.json",
    "apps/storefront/fixtures/shopper-personas.json",
)
_EXPORT_JS = r'''
import {PRODUCTS} from "./apps/storefront/src/config.ts";
import {PRODUCT_VARIANTS,getVariantImages} from "./apps/storefront/src/variants.ts";
import {PRODUCT_IMAGES} from "./apps/storefront/src/images.ts";
import {FREE_SHIPPING_OVER,FLAT_SHIPPING_COST,shippingQuote,orderTotal} from "./apps/storefront/src/shipping.ts";
const shippingProbes=[0,49.99,50,50.01,100].flatMap(subtotal=>[false,true].map(showShipping=>{
 const quote=shippingQuote(subtotal,showShipping);
 return {subtotal,showShipping,quote,displayedTotal:orderTotal(subtotal,quote)};
}));
console.log(JSON.stringify({products:PRODUCTS,baseImages:PRODUCT_IMAGES,
 variants:PRODUCTS.map(product=>({productId:product.id,colorVariants:PRODUCT_VARIANTS[product.id].map(variant=>({variant,gallery:getVariantImages(variant)}))})),
 shipping:{freeShippingOver:FREE_SHIPPING_OVER,flatShippingCost:FLAT_SHIPPING_COST,probes:shippingProbes}}));
'''


class StoreCatalogError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _object(value, required, optional=()):
    if (type(value) is not dict or any(type(key) is not str for key in value)
            or not set(required) <= value.keys() or not value.keys() <= set(required) | set(optional)):
        raise StoreCatalogError("missing, unexpected or non-object fields")


def _text(value):
    if type(value) is not str or not value.strip():
        raise StoreCatalogError("expected nonempty string")
    return value


def _number(value, *, positive=False, integer=False):
    if type(value) not in ((int,) if integer else (int, float)):
        raise StoreCatalogError("expected numeric value excluding bool")
    try:
        valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        valid = False
    if not valid:
        raise StoreCatalogError("expected finite nonnegative number")
    return value


def _list(value):
    if type(value) is not list:
        raise StoreCatalogError("expected list")
    return value


def _strings(value):
    return [_text(v) for v in _list(value)]


def _unique(values):
    if len(values) != len(set(values)):
        raise StoreCatalogError("duplicate routing identity")


def _asset_path(src):
    _text(src)
    path = Path(src)
    if not src.startswith("/products/") or any(p in {"..", "."} for p in src.split("/")) or "?" in src or "#" in src:
        raise StoreCatalogError("image source must be a local /products asset")
    if path.suffix not in {".jpg", ".png", ".webp"}:
        raise StoreCatalogError("unsupported local asset extension")
    return src


def _image(image):
    _object(image, {"id", "src", "label", "alt", "context", "presentation", "tags", "position"}, {"tile"})
    for key in ("id", "label", "alt", "position"):
        _text(image[key])
    _asset_path(image["src"])
    _declared_image(image)
    if "tile" in image:
        tile = image["tile"]
        _object(tile, {"columns", "rows", "index", "sheetAspect"})
        for key in ("columns", "rows"):
            _number(tile[key], positive=True, integer=True)
        _number(tile["index"], integer=True)
        _number(tile["sheetAspect"], positive=True)
        if tile["index"] >= tile["columns"] * tile["rows"]:
            raise StoreCatalogError("image crop tile out of bounds")
    return copy.deepcopy(image)


def _declared_image(image):
    for key in ("label", "alt"):
        _text(image[key])
    if type(image["context"]) is not str or image["context"] not in {"studio", "outdoors", "city", "campus", "family", "everyday"}:
        raise StoreCatalogError("unsupported declared image context")
    presentation = image["presentation"]
    _object(presentation, {"subject", "background", "tone", "composition"})
    vocabulary = {"subject": {"product", "in-use"}, "background": {"plain", "textured", "wood", "location"},
                  "tone": {"light", "dark", "warm", "natural"},
                  "composition": {"full-product", "detail", "flat-lay", "lifestyle"}}
    for key, allowed in vocabulary.items():
        if type(presentation[key]) is not str or presentation[key] not in allowed:
            raise StoreCatalogError("unsupported declared presentation value")
    _strings(image["tags"])


def validate_export(value):
    """Validate exact runtime shapes/bindings before projecting any model text."""
    _object(value, {"products", "baseImages", "variants", "shipping"})
    products = _list(value["products"]); groups = _list(value["variants"])
    ids = []
    for product in products:
        _object(product, {"id", "name", "category", "price", "color", "kind"},
                {"searchTerms", "stillLife", "lifestyle", "context", "scenario"})
        for key in ("id", "name", "category", "color", "kind"):
            _text(product[key])
        _number(product["price"])
        optional = {"searchTerms", "stillLife", "lifestyle", "context", "scenario"}
        if set(product) & optional and not optional <= product.keys():
            raise StoreCatalogError("partial additional-product metadata")
        if "searchTerms" in product:
            _strings(product["searchTerms"])
            for key in ("stillLife", "lifestyle", "context", "scenario"):
                _text(product[key])
        ids.append(product["id"])
    _unique(ids)
    if type(value["baseImages"]) is not dict or set(value["baseImages"]) != set(ids):
        raise StoreCatalogError("base image owners differ from products")
    if len(groups) != len(products):
        raise StoreCatalogError("variant owners differ from products")
    variant_ids, image_ids, base_ids = [], [], []
    for product, group in zip(products, groups, strict=True):
        _object(group, {"productId", "colorVariants"})
        if group["productId"] != product["id"]:
            raise StoreCatalogError("variant owner/order mismatch")
        base = _list(value["baseImages"][product["id"]])
        if not base:
            raise StoreCatalogError("product has no base image views")
        for image in base:
            _image(image); base_ids.append(image["id"])
        colors = _list(group["colorVariants"])
        if not colors:
            raise StoreCatalogError("product has no color variants")
        color_names = []
        for index, color in enumerate(colors):
            _object(color, {"variant", "gallery"})
            variant = color["variant"]
            _object(variant, {"id", "productId", "colorName", "swatch"}, {"image"})
            for key in ("id", "colorName", "swatch"):
                _text(variant[key])
            if variant["productId"] != product["id"]:
                raise StoreCatalogError("color variant owner mismatch")
            variant_ids.append(variant["id"]); color_names.append(variant["colorName"])
            gallery = _list(color["gallery"])
            if len(gallery) != len(base):
                raise StoreCatalogError("color gallery differs from base view count")
            if index == 0 and ("image" in variant or gallery != base):
                raise StoreCatalogError("default color must preserve exact base gallery")
            for image, original in zip(gallery, base, strict=True):
                _image(image); image_ids.append(image["id"])
                if (image["label"], image["context"], image["presentation"]) != (original["label"], original["context"], original["presentation"]):
                    raise StoreCatalogError("color gallery semantic binding differs from base")
            if "image" in variant:
                _image(variant["image"])
        _unique(color_names)
    _unique(variant_ids); _unique(image_ids); _unique(base_ids)
    shipping = value["shipping"]
    _object(shipping, {"freeShippingOver", "flatShippingCost", "probes"})
    _number(shipping["freeShippingOver"]); _number(shipping["flatShippingCost"])
    if shipping["freeShippingOver"] != 50 or shipping["flatShippingCost"] != 5:
        raise StoreCatalogError("shipping policy changed; freeze another version")
    expected = []
    for probe in _list(shipping["probes"]):
        _object(probe, {"subtotal", "showShipping", "quote", "displayedTotal"})
        _number(probe["subtotal"]); _number(probe["displayedTotal"])
        if type(probe["showShipping"]) is not bool:
            raise StoreCatalogError("shipping visibility must be boolean")
        if probe["quote"] is not None:
            _object(probe["quote"], {"label", "amount"})
            _text(probe["quote"]["label"]); _number(probe["quote"]["amount"])
    for subtotal in (0, 49.99, 50, 50.01, 100):
        for visible in (False, True):
            cost = 0 if subtotal > 50 else 5
            quote = {"label": "Free" if cost == 0 else "$5", "amount": cost} if visible else None
            expected.append({"subtotal": subtotal, "showShipping": visible, "quote": quote,
                             "displayedTotal": subtotal + (cost if visible else 0)})
    if shipping["probes"] != expected:
        raise StoreCatalogError("runtime shipping visibility/threshold behavior changed")
    return copy.deepcopy(value)


def _semantics(image, scenario):
    return {"label": image["label"], "alt": image["alt"], "context": image["context"],
            "presentation": copy.deepcopy(image["presentation"]),
            "tags": [tag for tag in image["tags"] if tag != scenario],
            "evidenceProvenance": "fixture-declared image text; pixels unexamined"}


def _route(image):
    return {key: copy.deepcopy(image[key]) for key in ("id", "src", "position", "tile") if key in image}


def project_export(value, public_tasks=(), task_mapping=()):
    """Separate observable predictive metadata from exact external action routing."""
    value = validate_export(value)
    tasks = [validate_task(t) for t in public_tasks]
    if len(tasks) != len(task_mapping):
        raise StoreCatalogError("public task mapping cardinality mismatch")
    persona_ids = []
    for task, row in zip(tasks, task_mapping, strict=True):
        _object(row, {"personaId", "requirementIds"})
        persona_ids.append(_text(row["personaId"]))
        ids = _strings(row["requirementIds"])
        _unique(ids)
        if len(ids) != len(task["requirements"]):
            raise StoreCatalogError("requirement routing count mismatch")
    _unique(persona_ids)
    products, mapping = [], []
    for product, group in zip(value["products"], value["variants"], strict=True):
        colors, routes = [], []
        scenario = product.get("scenario")
        for color in group["colorVariants"]:
            variant = color["variant"]
            colors.append({"colorName": variant["colorName"],
                           "images": [_semantics(image, scenario) for image in color["gallery"]]})
            route = {"variantId": variant["id"], "swatch": variant["swatch"],
                     "images": [_route(image) for image in color["gallery"]]}
            if "image" in variant:
                route["colorPreview"] = copy.deepcopy(variant["image"])
            routes.append(route)
        products.append({"name": product["name"], "category": product["category"],
                         "itemPrice": product["price"], "searchTerms": list(product.get("searchTerms", [])),
                         "colorVariants": colors})
        mapping.append({"productId": product["id"], "colorVariants": routes})
    return ({"schemaVersion": VERSION, "products": products,
             "publicApiFixtureTasks": tasks,
             "taskProvenance": "API fixture tasks; compatibility with current store not established",
             "capabilities": {"stock": "unknown", "sizes": "unknown", "perItemShippingCost": "unsupported",
                              "imageSelection": "declared metadata with external asset/crop bindings",
                              "pixelUnderstanding": "not performed"},
             "shippingTerms": {"scope": "cart-subtotal", "freeWhenSubtotalStrictlyGreaterThan": 50,
                               "otherwiseFlatAmount": 5,
                               "hiddenQuote": "omitted; actual charged delivery cost not established"}},
            {"products": mapping, "publicApiFixtureTasks": copy.deepcopy(list(task_mapping))})


def catalog_text(snapshot, product_index, color_index, image_index):
    """One product/color/gallery candidate's declared text; no external routing."""
    try:
        _object(snapshot, {"schemaVersion", "products", "publicApiFixtureTasks", "taskProvenance", "capabilities", "shippingTerms"})
        if snapshot["schemaVersion"] != VERSION:
            raise StoreCatalogError("unsupported public snapshot version")
        if any(type(i) is not int or i < 0 for i in (product_index, color_index, image_index)):
            raise StoreCatalogError("indices must be nonnegative integers")
        product = snapshot["products"][product_index]
        _object(product, {"name", "category", "itemPrice", "searchTerms", "colorVariants"})
        _text(product["name"]); _text(product["category"]); _number(product["itemPrice"])
        _strings(product["searchTerms"])
        color = product["colorVariants"][color_index]
        _object(color, {"colorName", "images"}); _text(color["colorName"])
        image = color["images"][image_index]
        _object(image, {"label", "alt", "context", "presentation", "tags", "evidenceProvenance"})
        _declared_image(image)
        if image["evidenceProvenance"] != "fixture-declared image text; pixels unexamined":
            raise StoreCatalogError("unsupported image evidence provenance")
    except (IndexError, KeyError, TypeError) as exc:
        raise StoreCatalogError("unknown public candidate index") from exc
    return canonical({"schemaVersion": "actual-store-catalog-text.v1",
                      "product": {key: product[key] for key in ("name", "category", "itemPrice", "searchTerms")},
                      "colorName": color["colorName"], "image": image,
                      "imageEvidence": "declared fixture metadata; no pixel verification"})


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _source_receipts(*, require_worktree):
    result = []
    for source in SOURCES:
        expected = _git("show", f"{COMMIT}:{source}")
        if require_worktree and (ROOT / source).read_bytes() != expected:
            raise StoreCatalogError(f"source differs from pinned commit: {source}")
        result.append({"path": source, "sha256": hashlib.sha256(expected).hexdigest(),
                       "bytes": len(expected), "gitBlob": _git("rev-parse", f"{COMMIT}:{source}").decode().strip()})
    return result


def _tasks():
    data = json.loads(_git("show", f"{COMMIT}:apps/api/fixtures/catalog.json"))
    tasks, mapping = [], []
    for persona in data["personas"]:
        task = persona["task"]
        public = {"query": task["query"], "requirements": [
            {key: r[key] for key in ("label", "kind", "attribute", "value")} for r in task["requirements"]]}
        if "budget" in task:
            public["budget"] = task["budget"]
        tasks.append(validate_task(public))
        mapping.append({"personaId": persona["id"], "requirementIds": [r["id"] for r in task["requirements"]]})
    return tasks, mapping


def capture_runtime():
    before = _source_receipts(require_worktree=True)
    raw = json.loads(subprocess.check_output(["node", "--import", "tsx", "--input-type=module", "-e", _EXPORT_JS], cwd=ROOT))
    if _source_receipts(require_worktree=True) != before:
        raise StoreCatalogError("source changed during runtime capture")
    return validate_export(raw), before


def _asset_receipts(raw):
    images = [im for group in raw["variants"] for color in group["colorVariants"]
              for im in color["gallery"] + ([color["variant"]["image"]] if "image" in color["variant"] else [])]
    assets = []
    for src in sorted({im["src"] for im in images}):
        path = ROOT / "apps/storefront/public" / src.lstrip("/")
        if not path.is_file() or path.is_symlink():
            raise StoreCatalogError(f"missing or symlinked local image asset: {src}")
        data = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        expected_blob = _git("rev-parse", f"{COMMIT}:apps/storefront/public{src}").decode().strip()
        if blob != expected_blob:
            raise StoreCatalogError(f"asset differs from pinned commit: {src}")
        assets.append({"src": src, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "gitBlob": blob})
    return assets


def freeze(directory=DEFAULT_OUT):
    """One bounded CPU capture; refuse existing output directories/changed source."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("preserve existing snapshot; use a new version")
    raw, sources = capture_runtime()
    if digest(raw) != RAW_EXPORT_SHA256:
        raise StoreCatalogError("runtime export differs from independently pinned capture")
    tasks, task_mapping = _tasks()
    inputs, mapping = project_export(raw, tasks, task_mapping)
    assets = _asset_receipts(raw)
    if (digest(inputs), digest(mapping), digest(assets)) != (MODEL_INPUTS_SHA256, MAPPING_SHA256, ASSET_RECEIPTS_SHA256):
        raise StoreCatalogError("observable projection or asset receipts differ from independent pins")
    if len(inputs["products"]) != 16 or sum(len(p["colorVariants"]) for p in inputs["products"]) != 48:
        raise StoreCatalogError("current-store roster changed")
    outputs = {"runtime_export.json": raw, "model_inputs.json": inputs, "external_mapping.json": mapping}
    receipt = {"schemaVersion": "actual-store-import-receipt.v1", "sourceCommit": COMMIT,
               "sources": sources, "importerSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "exportProgramSha256": hashlib.sha256(_EXPORT_JS.encode()).hexdigest(),
               "runtime": {"node": subprocess.check_output(["node", "--version"]).decode().strip(),
                           "tsx": json.loads((ROOT / "node_modules/tsx/package.json").read_text())["version"],
                           "imports": ["config.ts", "variants.ts", "images.ts", "shipping.ts"],
                           "execution": "installed Node+tsx; pure reviewed exports only; no installs"},
               "outputs": {name: {"canonicalSha256": digest(value)} for name, value in outputs.items()},
               "counts": {"products": len(inputs["products"]), "colorVariants": 48,
                          "baseImageRecords": sum(len(v) for v in raw["baseImages"].values()),
                          "variantGalleryRecords": sum(len(c["gallery"]) for g in raw["variants"] for c in g["colorVariants"]),
                          "optionalColorPreviewRecords": sum("image" in c["variant"] for g in raw["variants"] for c in g["colorVariants"]),
                          "uniqueAssetFiles": len(assets), "publicApiFixtureTasks": len(tasks)},
               "assets": assets, "modelCalls": 0, "trainingPerformed": False,
               "excludedPredictiveMetadata": ["IDs", "asset paths", "crop coordinates", "swatches", "CSS color/kind",
                   "synthetic scenario IDs/tags", "persona biography/goal/traits/split/history/outcomes",
                   "illustrative shopper-persona goals and candidate lists"],
               "limitations": ["image declarations are not pixel verification", "stock and sizes unknown",
                   "category/search terms are declared catalog text, not independently verified attributes",
                   "shipping is a cart-subtotal rule; hidden quote is not free-delivery evidence",
                   "API task compatibility not established; no learned quality/ranking claim",
                   "six-jacket historical snapshot is separate from current store"]}
    if _source_receipts(require_worktree=True) != sources:
        raise StoreCatalogError("source changed before snapshot freeze")
    directory.mkdir(parents=True, exist_ok=False)
    outputs["IMPORT_RECEIPT.json"] = receipt
    for name, value in outputs.items():
        with (directory / name).open("x") as handle:
            json.dump(value, handle, sort_keys=True, ensure_ascii=False, indent=2); handle.write("\n")
    return copy.deepcopy(receipt)


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise StoreCatalogError("duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise StoreCatalogError(f"nonfinite JSON constant {value}")
    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=reject)


def load_snapshot(directory=DEFAULT_OUT):
    """Reproduce projection from independently pinned raw export; no runtime call."""
    directory = Path(directory)
    receipt = _read(directory / "IMPORT_RECEIPT.json")
    if receipt["sourceCommit"] != COMMIT or receipt["sources"] != _source_receipts(require_worktree=False):
        raise StoreCatalogError("source provenance receipt mismatch")
    outputs = {name: _read(directory / name) for name in ("runtime_export.json", "model_inputs.json", "external_mapping.json")}
    if RAW_EXPORT_SHA256 is None or digest(outputs["runtime_export.json"]) != RAW_EXPORT_SHA256:
        raise StoreCatalogError("independent runtime export pin mismatch")
    for name, value in outputs.items():
        if digest(value) != receipt["outputs"][name]["canonicalSha256"]:
            raise StoreCatalogError("snapshot output receipt mismatch")
    if (digest(outputs["model_inputs.json"]), digest(outputs["external_mapping.json"]), digest(receipt["assets"])) != (MODEL_INPUTS_SHA256, MAPPING_SHA256, ASSET_RECEIPTS_SHA256):
        raise StoreCatalogError("independent projection/asset receipt pin mismatch")
    if receipt["importerSha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise StoreCatalogError("importer source changed since freeze")
    if receipt["exportProgramSha256"] != hashlib.sha256(_EXPORT_JS.encode()).hexdigest():
        raise StoreCatalogError("runtime export program changed")
    if receipt["counts"] != {"products": 16, "colorVariants": 48, "baseImageRecords": 65,
            "variantGalleryRecords": 195, "optionalColorPreviewRecords": 32,
            "uniqueAssetFiles": 81, "publicApiFixtureTasks": 10}:
        raise StoreCatalogError("snapshot counts receipt mismatch")
    if type(receipt["modelCalls"]) is not int or receipt["modelCalls"] != 0 or receipt["trainingPerformed"] is not False:
        raise StoreCatalogError("unexpected model/training provenance")
    expected_inputs, expected_mapping = project_export(outputs["runtime_export.json"], *_tasks())
    if outputs["model_inputs.json"] != expected_inputs or outputs["external_mapping.json"] != expected_mapping:
        raise StoreCatalogError("snapshot differs from exact ID-separated projection")
    if _asset_receipts(outputs["runtime_export.json"]) != receipt["assets"]:
        raise StoreCatalogError("current local assets differ from pinned source receipt")
    return copy.deepcopy({"inputs": expected_inputs, "externalMapping": expected_mapping,
                          "rawExport": outputs["runtime_export.json"], "receipt": receipt})
