"""Prepare context-only observational choices; optionally exercise CPU fit_inputs.

Never imports a native encoder instance, trains, or replaces a serving artifact.
The existing v5 experiment/driver is immutable and does not consume this packet.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "experiments/ranker/pretrained"
VERSION = "fieldwork.shopping-context-model-input.v1"
SPLITS = ("train", "validation", "test")
UUID = re.compile(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}")
SHA = re.compile(r"[a-f0-9]{64}")
TOKEN = re.compile(r"[A-Za-z0-9._-]{1,64}")
FILES = {
    "catalog": MODEL / "store_function_v5/catalog.json",
    "publicCatalog": MODEL / "store_catalog_v1/model_inputs.json",
    "mapping": MODEL / "store_catalog_v1/external_mapping.json",
    "snapshotImporter": MODEL / "store_catalog_v1.py",
    "snapshotReceipt": MODEL / "store_catalog_v1/IMPORT_RECEIPT.json",
    "snapshotRawExport": MODEL / "store_catalog_v1/runtime_export.json",
    "intakeCatalog": ROOT / "apps/api/internal/shoppingevents/catalog.json",
    "consumer": MODEL / "store_function_adaptation.py",
    "runtime": MODEL / "catalog_fusion/runtime.py",
    "encoderInputs": MODEL / "catalog_fusion/inputs.py",
    "adapter": Path(__file__).resolve(),
    "choiceSchema": ROOT / "apps/api/scripts/shopping-choice.v1.schema.json",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def exact(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"Exact {label} fields required")


def strict_json(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("Duplicate JSON key")
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def model_helpers():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("experiments.ranker.pretrained.store_function_adaptation")


def catalogs():
    # The importer verifies independently pinned raw/projection hashes and exact
    # ordered projection. Merely checking that all IDs exist would permit an
    # external-mapping reorder to silently relabel product indices.
    snapshot = model_helpers().load_snapshot(FILES["mapping"].parent)
    catalog = strict_json(FILES["catalog"].read_bytes())
    public, mapping = snapshot["inputs"], snapshot["externalMapping"]
    intake = strict_json(FILES["intakeCatalog"].read_bytes())
    if catalog != public["products"] or len(catalog) != 16 or len(mapping["products"]) != 16:
        raise ValueError("Frozen model/public catalog alignment changed")
    routing = []
    for i, p in enumerate(mapping["products"]):
        variants = [v["variantId"] for v in p["colorVariants"]]
        if intake.get(p["productId"]) != variants or not variants or len(set(variants)) != len(variants):
            raise ValueError("Intake/model product variant mapping changed")
        routing.append({"productId": p["productId"], "productIndex": i, "variantIds": variants})
    if len({p["productId"] for p in routing}) != 16 or set(intake) != {p["productId"] for p in routing}:
        raise ValueError("Exact sixteen-product routing required")
    return catalog, routing


def validate_choice(row, routing):
    exact(row, ("schema", "split", "provenance", "context", "candidates", "chosen", "choiceScope",
                "signal", "negativeLabels", "realPurchaseObserved", "selectionMechanism"), "choice")
    if row["schema"] != "fieldwork.shopping-choice.v1" or row["split"] not in SPLITS:
        raise ValueError("Versioned choice/session split required")
    p = row["provenance"]
    required = {"sessionId", "batchId", "eventId", "sourceSha256", "catalogEventId", "catalogBatchId",
                "catalogSourceSha256", "classification", "receivedAt", "splitVersion", "derivation"}
    if type(p) is not dict or not required <= set(p) or set(p) - required - {"sourceSchema", "modelVersionSource"}:
        raise ValueError("Versioned allowlisted provenance required")
    for name in ("sessionId", "batchId", "eventId", "catalogEventId", "catalogBatchId"):
        if type(p[name]) is not str or not UUID.fullmatch(p[name]):
            raise ValueError("UUID provenance required")
    for name in ("sourceSha256", "catalogSourceSha256"):
        if type(p[name]) is not str or not SHA.fullmatch(p[name]):
            raise ValueError("Source SHA256 required")
    if p["splitVersion"] != "fieldwork.session-split.v1" or p["derivation"] != "explicit-positive-choice.v1":
        raise ValueError("Known split/choice derivation required")
    if "sourceSchema" in p and p["sourceSchema"] != "fieldwork.shopping-batch.v1":
        raise ValueError("Known source schema required")
    if "modelVersionSource" in p and p["modelVersionSource"] != "client-reported-trusted-build-config":
        raise ValueError("Known client model provenance required")
    if type(p["receivedAt"]) is not str or datetime.fromisoformat(p["receivedAt"].replace("Z", "+00:00")).utcoffset() is None:
        raise ValueError("Timezone-aware receivedAt required")
    exact(p["classification"], ("environment", "traffic"), "classification")
    if p["classification"]["environment"] not in ("production", "development", "test") or p["classification"]["traffic"] not in ("real", "synthetic"):
        raise ValueError("Explicit environment/traffic required")
    if p["classification"]["traffic"] == "real" and p["classification"]["environment"] != "production":
        raise ValueError("Real traffic outside production")
    bucket = hashlib.sha256(("fieldwork.session-split.v1/" + p["sessionId"]).encode()).digest()[0] % 10
    expected_split = "validation" if bucket == 8 else "test" if bucket == 9 else "train"
    if row["split"] != expected_split:
        raise ValueError("Whole-session split mismatch")
    context = row["context"]
    exact(context, ("mode", "order", "shipping", "modelVersion"), "context")
    if context["mode"] not in ("baseline", "local-model") or context["order"] not in ("popular", "budget", "history") or type(context["shipping"]) is not bool:
        raise ValueError("Allowlisted observed UI context required")
    version = context["modelVersion"]
    if version is not None and (type(version) is not str or not TOKEN.fullmatch(version)):
        raise ValueError("Allowlisted model version required")
    routes = {r["productId"]: r for r in routing}
    def item(value, position=False):
        exact(value, ("productId", "variantId", "position") if position else ("productId", "variantId"), "product")
        if type(value["productId"]) is not str or value["productId"] not in routes or value["variantId"] not in routes[value["productId"]]["variantIds"]:
            raise ValueError("Known public product/variant required")
        if position and (type(value["position"]) is not int or not 0 <= value["position"] <= 15):
            raise ValueError("Rendered position 0..15 required")
    if type(row["candidates"]) is not list or not 1 <= len(row["candidates"]) <= 16:
        raise ValueError("Bounded rendered candidates required")
    for c in row["candidates"]:
        item(c, True)
    if len({c["productId"] for c in row["candidates"]}) != len(row["candidates"]) or len({c["position"] for c in row["candidates"]}) != len(row["candidates"]):
        raise ValueError("Unique candidate product IDs/positions required")
    item(row["chosen"])
    rendered = next((c for c in row["candidates"] if c["productId"] == row["chosen"]["productId"]), None)
    if rendered is None:
        raise ValueError("Chosen product absent from rendered candidates")
    scope = "rendered-variant" if rendered["variantId"] == row["chosen"]["variantId"] else "product-with-observed-variant-change"
    if row["signal"] not in ("selection", "cart-add") or row["choiceScope"] != scope or row["signal"] == "cart-add" and scope != "rendered-variant":
        raise ValueError("Choice signal/rendered-variant scope mismatch")
    if row["negativeLabels"] != [] or row["realPurchaseObserved"] is not False or row["selectionMechanism"] != "observational-not-randomized":
        raise ValueError("Positive-only observational demo scope required")


def context_request(context):
    # No selected product, candidate, model version, identity or inferred use enters text.
    return {"query": "Observed storefront presentation context; shopper query was not collected.",
            "preferences": ["Display mode: " + context["mode"], "Catalog ordering: " + context["order"],
                            "Shipping details: " + ("visible" if context["shipping"] else "hidden")],
            "itemBudget": None, "hardRequirements": []}


def prepare(rows, allow_synthetic=False):
    if type(allow_synthetic) is not bool:
        raise ValueError("Explicit boolean synthetic permission required")
    catalog, routing = catalogs()
    base = model_helpers()
    cases, provenance, records = [], [], []
    labels = {split: [] for split in SPLITS}
    routes = {r["productId"]: r for r in routing}
    seen, excluded = set(), 0
    if len(rows) > 10000:
        raise ValueError("At most 10000 choices per frozen packet")
    for row in rows:
        validate_choice(row, routing)
        p = row["provenance"]
        if p["eventId"] in seen:
            raise ValueError("Duplicate choice event ID")
        seen.add(p["eventId"])
        if not allow_synthetic and p["classification"] != {"environment": "production", "traffic": "real"}:
            excluded += 1
            continue
        case_id = "choice-" + p["eventId"]
        request = context_request(row["context"])
        base.validate_request(request)
        cases.append({"caseId": case_id, "split": row["split"], "request": request})
        target = routes[row["chosen"]["productId"]]["productIndex"]
        labels[row["split"]].append({"caseId": case_id, "targetProductIndex": target,
                                     "publicSupport": {k: catalog[target][k] for k in ("name", "category", "searchTerms")}})
        provenance.append({"caseId": case_id, "source": row,
                           "candidateProductIndices": [routes[c["productId"]]["productIndex"] for c in row["candidates"]]})
        text = base.request_text(request)
        records.append({"role": "request", "caseId": case_id, "split": row["split"],
                        "text": text, "textSha256": digest(text.encode())})
    cases.sort(key=lambda c: c["caseId"])
    provenance.sort(key=lambda c: c["caseId"])
    for values in labels.values():
        values.sort(key=lambda c: c["caseId"])
    records.sort(key=lambda c: c["caseId"])
    for i, product in enumerate(catalog):
        text = base.product_text(product)
        records.append({"role": "catalog", "caseId": str(i), "split": "train",
                        "text": text, "textSha256": digest(text.encode())})
    packet = {"catalog.json": catalog, "routing.json": routing, "public-cases.json": cases,
              "provenance.json": provenance, "encoder-records.json": records}
    packet.update({split + "-labels.json": values for split, values in labels.items()})
    manifest = {"schema": VERSION, "inputChoices": len(rows), "acceptedChoices": len(cases),
                "excludedNonProductionReal": excluded, "allowSynthetic": allow_synthetic,
                "splitCounts": {s: sum(c["split"] == s for c in cases) for s in SPLITS},
                "sessionCount": len({r["source"]["provenance"]["sessionId"] for r in provenance}),
                "scope": "observational-context-only-positive-choice",
                "requestConstruction": "fixed public text plus observed mode/order/shipping; no shopper query",
                "queryObserved": False, "semanticIntentSupervision": False, "negativeLabelsInferred": False,
                "realPurchaseObserved": False, "nativeEncoded": False, "trainingExecuted": False,
                "currentV5DriverCompatible": False, "servingModelChanged": False,
                "sourcePins": {k: {"path": str(p.relative_to(ROOT)), "sha256": digest(p.read_bytes())} for k, p in FILES.items()}}
    return packet, manifest


def cpu_consumer_check(packet):
    """Use actual fit_inputs with explicitly synthetic protocol-only CPU vectors.

    This proves the interface and split boundary, not native feature provenance,
    objective suitability, an optimizer step, predictions, or model quality.
    """
    base = model_helpers()
    torch = base.torch
    cases, records = packet["public-cases.json"], packet["encoder-records.json"]
    train_ids = {c["caseId"] for c in cases if c["split"] == "train"}
    if not train_ids:
        return {"status": "skipped-no-training-rows", "trainingRows": 0, "nativeEncoded": False,
                "trainingExecuted": False, "featureSource": "synthetic-protocol-only-cpu"}
    loaded = []
    class ProtocolOnlyFeatureStore:
        def load(self, row):
            if row["role"] == "request" and (row["split"] != "train" or row["caseId"] not in train_ids):
                raise AssertionError("Non-training session accessed by fit_inputs")
            loaded.append((row["role"], row["caseId"]))
            raw = hashlib.sha256(row["text"].encode()).digest()
            # Deliberately fake: deterministic interface fixtures, not Qwen features.
            values = [(raw[i % 32] - 128) / 128 for i in range(4096)]
            return (torch.tensor([values], dtype=torch.float32, device="cpu"),)
    q, c, targets, means, metadata = base.fit_inputs(ProtocolOnlyFeatureStore(), records, cases,
                                                   packet["train-labels.json"])
    if list(q.shape) != [len(train_ids), 4096] or list(c.shape) != [16, 4096] or set(metadata["queryMembers"]) != train_ids:
        raise AssertionError("Actual consumer layout mismatch")
    expected = {r["caseId"]: r["targetProductIndex"] for r in packet["train-labels.json"]}
    if targets.tolist() != [expected[r["caseId"]] for r in cases if r["split"] == "train"]:
        raise AssertionError("Separate target alignment mismatch")
    if q.device.type != "cpu" or c.device.type != "cpu" or any(not torch.isfinite(v).all() for v in (q, c, *means.values())):
        raise AssertionError("Finite CPU fixtures required")
    return {"status": "passed", "consumer": "store_function_adaptation.fit_inputs", "trainingRows": len(train_ids),
            "queryShape": list(q.shape), "catalogShape": list(c.shape), "targetShape": list(targets.shape),
            "validationTestRequestsLoaded": 0, "requestLoads": len([r for r in loaded if r[0] == "request"]),
            "featureSource": "synthetic-protocol-only-cpu", "nativeEncoded": False,
            "trainingExecuted": False, "predictionQualityMeasured": False}


def load_packet(directory):
    """Verify a serialized packet before an independent consumer dry run."""
    manifest = strict_json((directory / "manifest.json").read_bytes())
    if type(manifest.get("allowSynthetic")) is not bool:
        raise ValueError("Explicit boolean synthetic permission required")
    expected_names = {"catalog.json", "routing.json", "public-cases.json", "provenance.json",
                      "encoder-records.json", *(s + "-labels.json" for s in SPLITS)}
    names = set(manifest.get("fileSha256", {}))
    if manifest.get("schema") != VERSION or names not in (expected_names, expected_names | {"cpu-consumer-check.json"}):
        raise ValueError("Versioned exact packet manifest required")
    packet = {}
    for name, sha in manifest["fileSha256"].items():
        raw = (directory / name).read_bytes()
        if digest(raw) != sha:
            raise ValueError("Packet file digest mismatch")
        packet[name] = strict_json(raw)
    expected, regenerated = prepare([p["source"] for p in packet["provenance.json"]], manifest["allowSynthetic"])
    if any(packet[name] != value for name, value in expected.items()) or manifest["sourcePins"] != regenerated["sourcePins"]:
        raise ValueError("Source pins or context/target/provenance derivation changed")
    for name in ("acceptedChoices", "splitCounts", "sessionCount", "scope", "requestConstruction"):
        if manifest.get(name) != regenerated[name]:
            raise ValueError("Packet counts or observational scope changed")
    for flag in ("queryObserved", "semanticIntentSupervision", "negativeLabelsInferred", "realPurchaseObserved",
                 "nativeEncoded", "trainingExecuted", "currentV5DriverCompatible", "servingModelChanged"):
        if manifest.get(flag) is not False:
            raise ValueError("Unsupported model-training claim")
    return packet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-packet", type=Path)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--cpu-consumer-check", action="store_true")
    args = parser.parse_args(argv)
    if args.check_packet:
        if args.input or args.out or args.allow_synthetic or args.cpu_consumer_check:
            raise ValueError("Use --check-packet DIR alone for independent CPU validation")
        result = cpu_consumer_check(load_packet(args.check_packet))
        print(canonical(result))
        return result
    if not args.input or not args.out:
        raise ValueError("Use --input CHOICES_JSONL --out NEW_DIRECTORY")
    raw = args.input.read_bytes()
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("Input exceeds 64 MiB; narrow the export")
    rows = [strict_json(line) for line in raw.splitlines() if line.strip()]
    packet, manifest = prepare(rows, args.allow_synthetic)
    manifest["inputSha256"] = digest(raw)
    if args.cpu_consumer_check:
        packet["cpu-consumer-check.json"] = cpu_consumer_check(packet)
    manifest["fileSha256"] = {name: digest((canonical(value) + "\n").encode()) for name, value in packet.items()}
    # Create exclusively; remove only this invocation's incomplete directory.
    args.out.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        for name, value in {**packet, "manifest.json": manifest}.items():
            path = args.out / name
            with path.open("x", encoding="utf-8") as handle:
                path.chmod(0o600)
                handle.write(canonical(value) + "\n")
    except BaseException:
        shutil.rmtree(args.out)
        raise
    print(canonical(manifest))
    return manifest


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
