"""Post-freeze targets and labeled replay metrics; no target enters encoders."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch

from .fit_runner import require_frozen
from .preflight import file_digest
from .scorer import assemble_grid, roster
from .training import training_loss


def evaluation_labels(dataset, split, requests, out):
    require_frozen(out)  # Must happen before even opening a held-out label file.
    paths = {"validation": "labels/validation.jsonl", "test": "private/truth.jsonl",
             "diagnostics": "private/diagnostic-truth.jsonl"}
    if split not in paths:
        raise ValueError("Unknown held-out evaluator split")
    path = Path(dataset).resolve() / paths[split]
    if path.resolve() != path:
        raise ValueError("Evaluator label path cannot redirect through symlinks")
    labels = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rid = row["recordId"]
        if rid in requests:
            if rid in labels:
                raise ValueError("Duplicate evaluator target")
            labels[rid] = {key: row[key] for key in ("recordId", "bestProductId", "relevance", "presentationScores", "diagnosticPairId") if key in row}
    if set(labels) != set(requests):
        raise ValueError("Evaluation target membership mismatch")
    return labels, {"path": str(path), "sha256": file_digest(path), "checkpointFreezeVerified": True,
                    "fields": ["recordId", "bestProductId", "relevance", "presentationScores", "diagnosticPairId"]}


def compatible_ids(request, catalog, pid):
    surface = request["context"].get("presentationSurface")
    mapping = request.get("presentationCompatibility", {}).get(pid, {})
    return [v["id"] for v in catalog[pid]["presentationVariants"] if mapping.get(v["id"]) is True
            and (v.get("compatibleSurfaces") is None or surface in v["compatibleSurfaces"])]


def image_stat(pid, chosen, request, catalog, label):
    empty = {"available": 0, "informative": 0, "informativeCorrect": 0, "utilitySum": 0., "regretSum": 0., "incompatible": 0}
    if pid is None:
        return empty
    compatible = compatible_ids(request, catalog, pid)
    targets = label["presentationScores"][pid]
    if not compatible:
        return {**empty, "incompatible": int(chosen is not None)}
    best = max(targets[vid] for vid in compatible)
    informative = len(compatible) > 1 and best > 0 and len({targets[vid] for vid in compatible}) > 1
    utility = targets.get(chosen, 0) if chosen in compatible else 0
    return {"available": 1, "informative": int(informative),
            "informativeCorrect": int(informative and chosen in compatible and targets[chosen] == best),
            "utilitySum": utility, "regretSum": best - utility,
            "incompatible": int(chosen is not None and chosen not in compatible)}


def _sum_images(rows):
    fields = ("available", "informative", "informativeCorrect", "utilitySum", "regretSum", "incompatible")
    totals = {key: sum(row[key] for row in rows) for key in fields}
    totals["informativeAccuracy"] = totals["informativeCorrect"] / totals["informative"] if totals["informative"] else None
    totals["meanUtility"] = totals["utilitySum"] / totals["available"] if totals["available"] else None
    totals["meanRegret"] = totals["regretSum"] / totals["available"] if totals["available"] else None
    return totals


def readout(scores, request, catalog, label, *, guarded):
    actions = roster(request, catalog)
    scores = scores.detach().cpu()
    if scores.shape != (len(actions), 2) or not torch.isfinite(scores).all():
        raise ValueError("Complete finite score roster required")
    values = scores.tolist()
    by_product = {pid: [(vid, row) for (product, vid), row in zip(actions, values, strict=True) if product == pid]
                  for pid in request["candidateProductIds"]}
    means = {pid: sum(row[0] for _, row in rows) / len(rows) for pid, rows in by_product.items()}
    raw_rank = sorted(means, key=lambda pid: (-means[pid], pid))
    if guarded:
        grid = assemble_grid(request, catalog, actions, scores, len(raw_rank))
        rank = [p["productId"] for p in grid["rankedProducts"]]
        images = {p["productId"]: p["variantId"] for p in grid["rankedProducts"]}
        selected = grid["cards"][0]["productId"] if grid["cards"] else None
        eligible = [p["productId"] for p in grid["rankedProducts"] if p["eligible"]]
    else:
        grid = None
        rank, selected, eligible = raw_rank, raw_rank[0], None
        images = {pid: max((row[1], vid) for vid, row in rows)[1] for pid, rows in by_product.items()}
    relevance = label["relevance"]
    dcg = lambda items: sum((2 ** value - 1) / math.log2(index + 2) for index, value in enumerate(items))
    ideal = dcg(sorted((relevance[pid] for pid in raw_rank), reverse=True)[:3])
    margin = means[raw_rank[0]] - means[raw_rank[1]] if len(raw_rank) > 1 else None
    metrics = {"bestProductAccuracy": int(selected == label["bestProductId"]),
               "ndcgAt3": dcg([relevance[pid] for pid in rank[:3]]) / ideal if ideal else 1.,
               "abstain": int(selected is None),
               "oraclePresentation": image_stat(label["bestProductId"], images.get(label["bestProductId"]), request, catalog, label),
               "selectedPresentation": image_stat(selected, images.get(selected), request, catalog, label),
               "allCandidatePresentation": _sum_images([image_stat(pid, images[pid], request, catalog, label) for pid in raw_rank])}
    return {"rankedProductIds": rank, "selectedProductId": selected, "presentationActions": images,
            "eligibleProductIds": eligible, "rawTopMargin": margin,
            "rawTopTie": margin == 0 if margin is not None else False,
            "completeRosterScored": len(actions), "metrics": metrics,
            "gridAssemblyProbe": grid}


def evaluate_split(models, requests, labels, catalog, queries, features):
    if not requests or set(labels) != set(requests):
        raise ValueError("Exact nonempty evaluation membership required")
    sorted_ids = sorted(requests)
    permuted = {rid: sorted_ids[(i + 1) % len(sorted_ids)] for i, rid in enumerate(sorted_ids)}
    cases, summary = {}, {}
    for arm in ("random", "prior", "fusion", "permuted-query"):
        model = models["fusion"] if arm == "permuted-query" else models[arm]
        cases[arm] = {}
        with torch.no_grad():
            for rid, request in requests.items():
                query_rid = permuted[rid] if arm == "permuted-query" else rid
                start = time.perf_counter()
                scores = model(queries[query_rid], features[rid])
                raw = readout(scores, request, catalog, labels[rid], guarded=False)
                guarded = readout(scores, request, catalog, labels[rid], guarded=True)
                cases[arm][rid] = {"querySourceRecordId": query_rid, "variantScores": scores.tolist(),
                                   "raw": raw, "guarded": guarded,
                                   "cpuScoreTwoReadoutsMs": (time.perf_counter() - start) * 1000}
            used_queries = {rid: queries[permuted[rid]] if arm == "permuted-query" else queries[rid] for rid in requests}
            loss = training_loss(model, requests, labels, catalog, used_queries, features).item()
        summary[arm] = {"requests": len(requests), "objectiveLossNoUpdate": loss}
        for mode in ("raw", "guarded"):
            metrics = [case[mode]["metrics"] for case in cases[arm].values()]
            summary[arm][mode] = {key: sum(m[key] for m in metrics) / len(metrics)
                                  for key in ("bestProductAccuracy", "ndcgAt3", "abstain")}
            for key in ("oraclePresentation", "selectedPresentation", "allCandidatePresentation"):
                summary[arm][mode][key] = _sum_images([m[key] for m in metrics])
        summary[arm]["rawTopTies"] = sum(case["raw"]["rawTopTie"] for case in cases[arm].values())
    pairs = {}
    for rid, label in labels.items():
        if "diagnosticPairId" in label:
            pairs.setdefault(label["diagnosticPairId"], []).append(rid)
    pair_results = {arm: [{"pairId": pair, "recordIds": ids,
                          "bothCorrect": all(cases[arm][rid]["guarded"]["selectedProductId"] == labels[rid]["bestProductId"] for rid in ids),
                          "selectionChanges": len({cases[arm][rid]["guarded"]["selectedProductId"] for rid in ids}) > 1}
                         for pair, ids in pairs.items()] for arm in cases}
    return {"summary": summary, "cases": cases, "diagnosticPairs": pair_results,
            "permutation": "cyclic next sorted record within this split; masks/context targets stay original",
            "gridAssemblyProbe": "candidate-count assembly only; not labeled 12/24-card fulfillment evidence"}
