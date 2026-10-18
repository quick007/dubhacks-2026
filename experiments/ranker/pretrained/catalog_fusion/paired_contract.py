"""Counterfactual data receipts and product/image pair metrics."""
from __future__ import annotations

import json
from pathlib import Path

from .fit_runner import require_frozen
from .paired_data import VERSION, teacher
from .preflight import file_digest


def verify_freeze(dataset, *, public_only):
    dataset = Path(dataset).resolve()
    receipt = json.loads((dataset / "FREEZE.json").read_text())
    if receipt.get("version") != VERSION or not receipt.get("catalogUnchanged"):
        raise ValueError("Unknown paired data contract or changed catalog")
    if receipt["generatorSha256"] != file_digest(Path(__file__).with_name("paired_data.py")):
        raise ValueError("Frozen teacher/generator source changed")
    for name, expected in receipt["files"].items():
        if public_only and (name.startswith("labels/") or name == "paired-controls.json"):
            continue  # Never open held-out target bytes before checkpoint freeze.
        path = dataset / name
        if path.resolve() != path or file_digest(path) != expected:
            raise ValueError(f"Paired dataset file changed after review/freeze: {name}")
    return receipt


def read_labels(dataset, split, requests, out):
    require_frozen(out)
    receipt = verify_freeze(dataset, public_only=True)
    if split not in ("validation", "test"):
        raise ValueError("Paired held-out evaluator split must be validation/test")
    relative = f"labels/{split}.jsonl"
    path = Path(dataset).resolve() / relative
    if path.resolve() != path or file_digest(path) != receipt["files"][relative]:
        raise ValueError("Held-out targets changed after data freeze")
    labels = {row["recordId"]: row for row in map(json.loads, path.read_text().splitlines())}
    if set(labels) != set(requests):
        raise ValueError("Held-out paired target membership mismatch")
    return labels


def validate_data_for_review(dataset):
    """Explicit data-only audit; no feature extraction or optimization."""
    receipt = verify_freeze(dataset, public_only=False)
    dataset = Path(dataset)
    catalog = {p["id"]: p for p in json.loads((dataset / "catalog.json").read_text())["products"]}
    all_requests, all_labels, split_products = {}, {}, {}
    for split in receipt["splits"]:
        requests = {r["recordId"]: r for r in map(json.loads, (dataset / f"public/{split}.jsonl").read_text().splitlines())}
        labels = {r["recordId"]: r for r in map(json.loads, (dataset / f"labels/{split}.jsonl").read_text().splitlines())}
        if len(requests) != receipt["splits"][split]["requests"] or set(requests) != set(labels):
            raise ValueError("Frozen paired request/label membership mismatch")
        for rid, request in requests.items():
            if teacher(request, catalog) != labels[rid]:
                raise ValueError("Frozen target differs from public-fact teacher")
        all_requests.update(requests)
        all_labels.update(labels)
        split_products[split] = {pid for r in requests.values() for pid in r["candidateProductIds"]}
    controls = json.loads((dataset / "paired-controls.json").read_text())
    for family in controls["families"]:
        requests = [all_requests[rid] for rid in family["recordIds"]]
        reference = requests[0]
        for request in requests:
            for key in ("candidateProductIds", "presentationCompatibility", "history", "asOf"):
                if request[key] != reference[key]:
                    raise ValueError("Family changed a controlled public field")
            for key in ("statedBudget", "statedRequirements", "presentationSurface", "task"):
                if request["context"][key] != reference["context"][key]:
                    raise ValueError("Family changed a controlled context field")
    for pair in controls["pairs"]:
        a, b = [all_labels[rid] for rid in pair["recordIds"]]
        if pair["kind"] == "product" and a["bestProductId"] == b["bestProductId"]:
            raise ValueError("Product pair has no target change")
        if pair["kind"] == "image":
            if a["bestProductId"] != b["bestProductId"]:
                raise ValueError("Style pair changed best product")
            for pid in pair["productIds"]:
                best = [max(label["presentationScores"][pid], key=label["presentationScores"][pid].get) for label in (a, b)]
                if best[0] == best[1]:
                    raise ValueError("Style pair has no target image change")
    split_names = list(split_products)
    if any(split_products[a] & split_products[b] for i, a in enumerate(split_names) for b in split_names[i + 1:]):
        raise ValueError("Generated split product membership overlaps")
    return {"status": "passed-data-only", "requests": len(all_requests), "families": len(controls["families"]),
            "pairs": len(controls["pairs"]), "teacherRecomputedExactly": True,
            "controlsEqualWithinFamilies": True, "crossSplitProductDisjoint": True,
            "freezeSha256": file_digest(dataset / "FREEZE.json")}


def pair_metrics(evaluation, controls, labels, split):
    pairs = [p for p in controls["pairs"] if p["split"] == split]
    result = {}
    for arm, cases in evaluation["cases"].items():
        result[arm] = {}
        for mode in ("raw", "guarded"):
            metrics = {"productPairs": 0, "productPairBothCorrect": 0, "productChoiceChanges": 0,
                       "imagePairs": 0, "imagePairSelectedProductStable": 0,
                       "allCandidateImagePairs": 0, "allCandidateImagePairBothCorrect": 0,
                       "allCandidateImageChoiceChanges": 0, "oracleProductImagePairBothCorrect": 0,
                       "selectedProductImagePairBothCorrect": 0}
            for pair in pairs:
                ids = pair["recordIds"]
                a, b = [cases[rid][mode] for rid in ids]
                if pair["kind"] == "product":
                    metrics["productPairs"] += 1
                    metrics["productChoiceChanges"] += a["selectedProductId"] is not None and b["selectedProductId"] is not None and a["selectedProductId"] != b["selectedProductId"]
                    metrics["productPairBothCorrect"] += all(case["selectedProductId"] == labels[rid]["bestProductId"] for rid, case in zip(ids, (a, b), strict=True))
                    continue
                metrics["imagePairs"] += 1
                stable = a["selectedProductId"] is not None and a["selectedProductId"] == b["selectedProductId"]
                metrics["imagePairSelectedProductStable"] += stable
                correct = {}
                for pid in pair["productIds"]:
                    choices = [case["presentationActions"].get(pid) for case in (a, b)]
                    correct[pid] = all(label["presentationScores"][pid].get(vid, 0) == 3 for label, vid in zip((labels[rid] for rid in ids), choices, strict=True))
                    metrics["allCandidateImagePairs"] += 1
                    metrics["allCandidateImagePairBothCorrect"] += correct[pid]
                    metrics["allCandidateImageChoiceChanges"] += all(v is not None for v in choices) and choices[0] != choices[1]
                metrics["oracleProductImagePairBothCorrect"] += correct[pair["expectedBestProductId"]]
                metrics["selectedProductImagePairBothCorrect"] += stable and correct[a["selectedProductId"]]
            result[arm][mode] = metrics
    return result
