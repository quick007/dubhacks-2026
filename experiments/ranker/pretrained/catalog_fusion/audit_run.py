"""CPU artifact/label-confound audit of a completed genuine fusion run."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import torch

from .fit_runner import require_frozen
from .inputs import canonical
from .preflight import file_digest
from .runner import corpus_rows, public_requests
from .runtime import FeatureStore, write_json
from .scorer import load_checkpoint, roster


def audit(prior, dataset):
    torch.set_num_threads(1)
    prior, dataset = Path(prior), Path(dataset)
    report = json.loads((prior / "report.json").read_text())
    frozen = require_frozen(prior)
    provenance = json.loads((prior / "model-verification.json").read_text())
    rows = corpus_rows(dataset, json.loads((prior / "preflight.json").read_text()))
    store = FeatureStore.open_existing(prior / "features", provenance)
    vectors = {key: store.load(row)[0][0] for key, row in rows.items()}
    requests = public_requests(dataset)
    catalog = {p["id"]: p for p in json.loads((dataset / "catalog.json").read_text())["products"]}
    labels = {r["recordId"]: r for r in map(json.loads, (dataset / "labels/train.jsonl").read_text().splitlines())}
    pools = defaultdict(list)
    for rid, request in requests["train"].items():
        pools[tuple(sorted(request["candidateProductIds"]))].append(rid)
    stable_winners = sum(len({labels[rid]["bestProductId"] for rid in ids}) == 1 for ids in pools.values())
    stable_images, comparable_images = 0, 0
    for pool, ids in pools.items():
        for pid in pool:
            winners = []
            for rid in ids:
                values = labels[rid]["presentationScores"][pid]
                compatible = [vid for vid in values if requests["train"][rid]["presentationCompatibility"][pid].get(vid) is True]
                if len(compatible) < 2 or len({values[v] for v in compatible}) < 2 or max(values[v] for v in compatible) <= 0:
                    break
                best = max(values[v] for v in compatible)
                winners.append(tuple(sorted(v for v in compatible if values[v] == best)))
            if len(winners) == len(ids):
                comparable_images += 1
                stable_images += len(set(winners)) == 1
    models = {arm: load_checkpoint(prior / row["file"], row["sha256"]) for arm, row in frozen["checkpoints"].items()}
    exact, rank_checks = 0, 0
    comparisons, diagnostics = {}, {}
    with torch.no_grad():
        for split, split_requests in requests.items():
            cases = json.loads((prior / f"{split}-evaluation.json").read_text())
            comparisons[split] = {}
            for arm, records in cases["cases"].items():
                model = models["fusion" if arm == "permuted-query" else arm]
                for rid, case in records.items():
                    request = split_requests[rid]
                    actions = roster(request, catalog)
                    x = torch.stack([vectors[("catalog", canonical(list(a)))] for a in actions])
                    q = vectors[("request", canonical(case["querySourceRecordId"]))]
                    actual = model(q, x)
                    if not torch.equal(actual, torch.tensor(case["variantScores"], dtype=torch.float32)):
                        raise AssertionError(f"Saved feature/checkpoint score reload mismatch: {split}/{arm}/{rid}")
                    exact += 1
                    means = {pid: actual[[i for i, a in enumerate(actions) if a[0] == pid], 0].double().mean().item() for pid in request["candidateProductIds"]}
                    ranked = sorted(means, key=lambda pid: (-means[pid], pid))
                    if ranked != case["raw"]["rankedProductIds"]:
                        raise AssertionError("Independent mean-variant ranking mismatch")
                    rank_checks += 1
                summary = cases["summary"][arm]
                comparisons[split][arm] = {
                    "rawProductCorrect": round(summary["raw"]["bestProductAccuracy"] * len(records)),
                    "guardedProductCorrect": round(summary["guarded"]["bestProductAccuracy"] * len(records)), "requests": len(records),
                    "guardedNdcgAt3": summary["guarded"]["ndcgAt3"],
                    "allCandidateImageCorrect": summary["guarded"]["allCandidatePresentation"]["informativeCorrect"],
                    "allCandidateImageInformative": summary["guarded"]["allCandidatePresentation"]["informative"],
                    "oracleImageCorrect": summary["guarded"]["oraclePresentation"]["informativeCorrect"],
                    "oracleImageInformative": summary["guarded"]["oraclePresentation"]["informative"],
                    "selectedImageCorrect": summary["guarded"]["selectedPresentation"]["informativeCorrect"],
                    "selectedImageInformative": summary["guarded"]["selectedPresentation"]["informative"]}
            diagnostics[split] = {arm: {"pairs": len(values), "bothCorrect": sum(v["bothCorrect"] for v in values),
                                      "selectionChanges": sum(v["selectionChanges"] for v in values)}
                                  for arm, values in cases["diagnosticPairs"].items()}
            comparisons[split]["fusionVsPermutedRawRankAgreement"] = sum(
                cases["cases"]["fusion"][rid]["raw"]["rankedProductIds"] == cases["cases"]["permuted-query"][rid]["raw"]["rankedProductIds"] for rid in split_requests)
    timing = {}
    for split in requests:
        observations = json.loads((prior / f"{split}-native-observations.json").read_text())
        for mode in ("cold", "mixed", "warm"):
            selected = [r for r in observations if r["catalogCacheCondition"] == mode]
            if selected:
                timing[f"{split}:{mode}"] = {"requests": len(selected), **{
                    field: {"mean": statistics.mean(r[field] for r in selected), "median": statistics.median(r[field] for r in selected)}
                    for field in ("observedPipelineWallMs", "freshQueryNativeEncodeMs", "catalogResolveConstructHashLoadOrExtractMs")}}
    peak = max(row["memory"]["mlxPeakBytes"] for row in report["phases"].values() if "memory" in row)
    return {"auditSourceSha256": file_digest(__file__), "priorReportSha256": file_digest(prior / "report.json"),
            "execution": report["execution"], "featureEntriesVerified": len(vectors),
            "exactScoreReloadChecks": exact, "independentRawRankChecks": rank_checks,
            "trainingConfounds": {"requests": len(requests["train"]), "distinctCandidatePools": len(pools),
                                  "samePoolSizes": sorted({len(ids) for ids in pools.values()}),
                                  "poolsWithUnchangedBestProduct": stable_winners,
                                  "comparablePoolProductImagePairs": comparable_images,
                                  "comparableImagePairsWithUnchangedBestVariant": stable_images,
                                  "perPoolSkuMembershipDisjoint": sum(map(len, pools)) == len({pid for pool in pools for pid in pool})},
            "quality": comparisons, "diagnosticPairs": diagnostics, "timingMs": timing,
            "peakBytes": peak, "peakGiB": peak / 2 ** 30,
            "activeBytes": report["phases"]["train"]["memory"]["mlxActiveBytes"],
            "declaredGuardBytes": 6 * 2 ** 30, "wallSeconds": report["wallSecondsIncludingHashLoadFitEvaluation"],
            "fits": report["fits"], "nativeForwards": report["nativeForwards"],
            "conclusion": "Genuine fitted/cached pipeline, but no product personalization or held-out image-context benefit established; do not promote.",
            "limitations": ["Prior and fusion memorize constant training pool winners; nonzero interaction gradients alone are insufficient.",
                            "Held-out outcomes have been inspected and are replay evidence.",
                            "Warm/cold groups contain different requests; their timings are not a matched speed ratio.",
                            "This audit does not execute native A/B/A or fresh feature parity; retain the separate probe receipt."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=Path("data/ranker/pretrained/catalog-user-fusion-v1-attempt-1"))
    parser.add_argument("--dataset", type=Path, default=Path("data/ranker/datasets/v2"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.prior, args.dataset)
    write_json(args.out, result)
    print(json.dumps({key: result[key] for key in ("featureEntriesVerified", "exactScoreReloadChecks", "trainingConfounds", "peakGiB", "diagnosticPairs", "conclusion")}, indent=2))


if __name__ == "__main__":
    main()
