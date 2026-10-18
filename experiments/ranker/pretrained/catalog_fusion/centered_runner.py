"""One reviewed CPU-only centering ablation on verified existing native vectors.

--prepare-only freezes training statistics/config without fitting. Actual fitting
requires the reviewed plan checksum; no native backbone is imported or executed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from .cached_inputs import CachedPairedInputs
from .centered import (CENTERED_REPRESENTATION, CENTER_CONFIG, CenteredFusionScorer,
                       center_rows, compute_statistics, load_centered, save_centered,
                       statistics_digest, validate_statistics)
from .evaluate import evaluate_split
from .fit_runner import FIT_CONFIG, require_frozen
from .inputs import canonical
from .paired_contract import pair_metrics, read_labels, verify_freeze
from .preflight import file_digest
from .runner import execution_receipt, source_hashes
from .runtime import write_json
from .training import read_training_inputs, training_loss


def prepare_plan(cache):
    queries, features, items, raw_receipts = cache.load_split("train")
    means, stats = compute_statistics(cache.requests["train"], cache.catalog, queries, items,
                                     provenance={**cache.source_receipts,
                                                 "trainingPublicSha256": file_digest(cache.dataset / "public/train.jsonl"),
                                                 "rawMembershipReceipts": raw_receipts})
    plan = {"representation": CENTERED_REPRESENTATION, "conditioning": CENTER_CONFIG, "fit": FIT_CONFIG,
            "statisticsSha256": statistics_digest(stats), "statistics": stats,
            "sourceHashes": source_hashes(), "inputSources": cache.source_receipts,
            "nativeForwards": 0, "fittingInPreparation": False,
            "controls": ["centered-random", "independently fitted centered-prior", "centered-fusion", "permuted-query"],
            "scope": "one fixed conditioning hypothesis, same rank4/tanh/32786 trainable architecture",
            "evaluationBoundary": "centering from train only; held-out raw vectors and targets loaded after checkpoint freeze"}
    return plan, means, stats, queries, features, items


def write_statistics(path, means, stats):
    validate_statistics(means, stats)
    with Path(path).open("xb") as handle:
        torch.save({"representation": CENTERED_REPRESENTATION + "/statistics-only", "means": means,
                    "statistics": stats, "statisticsSha256": statistics_digest(stats)}, handle)


def _activation_stats(model, queries, items):
    with torch.no_grad():
        q = center_rows(torch.stack([queries[k] for k in sorted(queries)]), model.query_mean)
        c = center_rows(torch.stack([items[k] for k in sorted(items)]), model.catalog_mean)
        result = {}
        for role, values, matrix in (("query", q, model.U), ("catalog", c, model.V)):
            projected = torch.tanh(values @ matrix.T)
            result[role] = {"saturationFractionAbsAtLeast0.99": (projected.abs() >= .99).float().mean().item(),
                            "acrossRowVarianceSum": projected.var(0, correction=0).sum().item(),
                            "distinctProjectedRows": len(torch.unique(projected, dim=0))}
        return result


def fit_centered(arm, out, means, stats, requests, labels, catalog, queries, features, items):
    if arm not in ("fusion", "prior"):
        raise ValueError("Unknown centered fit arm")
    started = time.perf_counter()
    model = CenteredFusionScorer(means, stats, FIT_CONFIG["seed"])
    if arm == "prior":
        with torch.no_grad():
            model.interaction.zero_()
        model.interaction.requires_grad_(False)
        model.U.requires_grad_(False)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    initial = {rid: model(queries[rid], features[rid]).detach().clone() for rid in requests}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=FIT_CONFIG["learningRate"], weight_decay=FIT_CONFIG["weightDecay"])
    trace, first = [], None
    with (Path(out) / f"{arm}-epochs.jsonl").open("x") as handle:
        for epoch in range(FIT_CONFIG["epochs"]):
            if time.perf_counter() - started > FIT_CONFIG["maxSecondsPerArm"]:
                raise TimeoutError("Fixed centered fit budget exceeded")
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, requests, labels, catalog, queries, features)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite centered loss")
            loss.backward()
            gradients = {}
            for name, p in model.named_parameters():
                if p.requires_grad:
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        raise FloatingPointError("Missing/nonfinite centered gradient")
                    gradients[name] = p.grad.norm().item()
            if first is None:
                first = gradients
                required = ("U", "V", "prior", "interaction") if arm == "fusion" else ("V", "prior")
                if any(first[k] <= 0 for k in required):
                    raise FloatingPointError("Centered fit has no genuine projection/interaction gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite centered parameter")
            row = {"epoch": epoch + 1, "lossBeforeStep": loss.item(), "gradientL2": gradients,
                   "activationsAfterStep": _activation_stats(model, queries, items),
                   "elapsedSeconds": time.perf_counter() - started}
            trace.append(row)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
    delta = {name: (p.detach() - before[name]).abs().sum().item() for name, p in model.named_parameters()}
    if any(delta[k] <= 0 for k in required):
        raise FloatingPointError("Centered parameters did not update")
    if arm == "prior" and (delta["U"] != 0 or torch.count_nonzero(model.interaction).item()):
        raise AssertionError("Centered item prior acquired a user interaction")
    sha = save_centered(model, Path(out) / f"{arm}.pt")
    restored = load_centered(Path(out) / f"{arm}.pt", sha, statistics_digest(stats))
    with torch.no_grad():
        final = {rid: model(queries[rid], features[rid]) for rid in requests}
        if any(not torch.equal(final[rid], restored(queries[rid], features[rid])) for rid in requests):
            raise AssertionError("Centered checkpoint/raw-input prediction reload mismatch")
        final_loss = training_loss(restored, requests, labels, catalog, queries, features).item()
    change = sum((final[rid] - initial[rid]).abs().sum().item() for rid in requests)
    if change <= 0 or time.perf_counter() - started > FIT_CONFIG["maxSecondsPerArm"]:
        raise FloatingPointError("Centered fit failed prediction-change or fixed-time check")
    report = {"arm": arm, "representation": CENTERED_REPRESENTATION, "config": FIT_CONFIG,
              "statisticsSha256": statistics_digest(stats), "trained": True,
              "trainableParameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "lossBefore": trace[0]["lossBeforeStep"], "lossAfter": final_loss, "firstGradientL2": first,
              "parameterAbsoluteChange": delta, "predictionAbsoluteChange": change,
              "seconds": time.perf_counter() - started, "sha256": sha,
              "exactTrainingPredictionReload": True, "finalActivations": _activation_stats(restored, queries, items)}
    write_json(Path(out) / f"{arm}-fit.json", report)
    return restored, report


def require_centered_frozen(out, stats_sha):
    frozen = require_frozen(out)
    if frozen.get("representation") != CENTERED_REPRESENTATION or frozen.get("statisticsSha256") != stats_sha:
        raise ValueError("Distinct centered checkpoint/statistics freeze required")
    for checkpoint in frozen["checkpoints"].values():
        load_centered(Path(out) / checkpoint["file"], checkpoint["sha256"], stats_sha)
    return frozen


def freeze_centered(out, reports, hashes, means, stats):
    if set(reports) != {"fusion", "prior"} or any(not r["trained"] or not r["exactTrainingPredictionReload"] for r in reports.values()):
        raise ValueError("Both centered fits must finish/reload before evaluation")
    random = CenteredFusionScorer(means, stats, FIT_CONFIG["seed"])
    random_sha = save_centered(random, Path(out) / "random.pt")
    checkpoints = {arm: {"file": f"{arm}.pt", "sha256": r["sha256"], "trained": True} for arm, r in reports.items()}
    checkpoints["random"] = {"file": "random.pt", "sha256": random_sha, "trained": False}
    write_json(Path(out) / "CHECKPOINTS_FROZEN.json", {"representation": CENTERED_REPRESENTATION,
        "statisticsSha256": statistics_digest(stats), "config": FIT_CONFIG, "conditioning": CENTER_CONFIG,
        "trainHashes": hashes, "checkpoints": checkpoints, "frozenAtUtc": datetime.now(timezone.utc).isoformat(),
        "heldoutLabelsOpenedBeforeFreeze": False})
    require_centered_frozen(out, statistics_digest(stats))
    return random.eval()


def run_reviewed(cache, out, prepared, reviewed_plan_sha):
    started = time.perf_counter()
    plan, means, stats, queries, features, items = prepared
    train, labels, hashes = read_training_inputs(cache.dataset)
    if canonical(train) != canonical(cache.requests["train"]) or hashes["labels/train.jsonl"] != cache.freeze["files"]["labels/train.jsonl"]:
        raise ValueError("Centered training targets changed after paired data freeze")
    config = {"representation": CENTERED_REPRESENTATION, "execution": execution_receipt(),
              "reviewedPlanSha256": reviewed_plan_sha, "plan": plan, "nativeForwards": 0, "generatesCopy": False}
    write_json(Path(out) / "config.json", config)
    models, fits = {}, {}
    for arm in ("fusion", "prior"):
        models[arm], fits[arm] = fit_centered(arm, out, means, stats, train, labels, cache.catalog, queries, features, items)
    models["random"] = freeze_centered(out, fits, hashes, means, stats)
    results = {"train": evaluate_split(models, train, labels, cache.catalog, queries, features)}
    labels_by_split = {"train": labels}
    for split in ("validation", "test"):
        require_centered_frozen(out, statistics_digest(stats))
        if source_hashes() != plan["sourceHashes"]:
            raise ValueError("Centered source snapshot changed during the attempt")
        q, x, _, _ = cache.load_split(split)  # First held-out vector values, after freeze.
        labels_by_split[split] = read_labels(cache.dataset, split, cache.requests[split], out)
        results[split] = evaluate_split(models, cache.requests[split], labels_by_split[split], cache.catalog, q, x)
    verify_freeze(cache.dataset, public_only=False)
    controls = json.loads((cache.dataset / "paired-controls.json").read_text())
    for split, result in results.items():
        result["counterfactualPairs"] = pair_metrics(result, controls, labels_by_split[split], split)
        write_json(Path(out) / f"{split}-evaluation.json", result)
    require_centered_frozen(out, statistics_digest(stats))
    if source_hashes() != plan["sourceHashes"]:
        raise ValueError("Centered sources changed during evaluation")
    # Audit the original training bytes again without changing the frozen means.
    verified_q, _, verified_items, _ = cache.load_split("train")
    _, verified_stats = compute_statistics(cache.requests["train"], cache.catalog, verified_q, verified_items,
                                          provenance=stats["provenance"])
    if statistics_digest(verified_stats) != statistics_digest(stats):
        raise ValueError("Raw training features/statistics changed during the ablation")
    if (file_digest(cache.query_run / "report.json") != plan["inputSources"]["queryReportSha256"]
            or file_digest(cache.catalog_run / "report.json") != plan["inputSources"]["catalogReportSha256"]):
        raise ValueError("Preserved raw native source reports changed")
    report = {"status": "completed", "representation": CENTERED_REPRESENTATION, "execution": config["execution"],
              "conditioning": CENTER_CONFIG, "statisticsSha256": statistics_digest(stats), "statistics": stats,
              "reviewedPlanSha256": reviewed_plan_sha, "fits": fits,
              "quality": {split: r["summary"] for split, r in results.items()},
              "counterfactualPairs": {split: r["counterfactualPairs"] for split, r in results.items()},
              "nativeForwards": 0, "rawFeatureCacheWrites": 0, "postEvaluationTrainingStatisticsReverified": True,
              "wallSeconds": time.perf_counter() - started,
              "generatesCopy": False, "completeViewEvidence": {"aboveFoldThreeCards": "not evaluated", "fullStorefront": "not evaluated"},
              "limitations": ["One conditioning hypothesis on replayed synthetic public-fact supervision; no promotion implied.",
                  "Offline cached-feature CPU timing excludes fresh query encoding/native extraction and full copy generation.",
                  "Old paired/raw outputs are preserved; this is a distinct checkpoint/representation, not old-head equivalence.",
                  "Titles, badges and section grouping are not trained by this scorer; no conversion-lift claim."]}
    write_json(Path(out) / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("experiments/ranker/pretrained/catalog_fusion/paired_v1"))
    parser.add_argument("--queries", type=Path, default=Path("data/ranker/pretrained/catalog-user-fusion-paired-v1-attempt-1"))
    parser.add_argument("--catalog-run", type=Path, default=Path("data/ranker/pretrained/catalog-user-fusion-v1-attempt-1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--reviewed-plan-sha256")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        cache = CachedPairedInputs(args.dataset, args.queries, args.catalog_run)
        prepared = prepare_plan(cache)
        plan, means, stats, *_ = prepared
        if not args.prepare_only:
            if args.plan is None or args.reviewed_plan_sha256 != file_digest(args.plan):
                raise ValueError("Reviewed centering plan checksum required before fitting")
            if json.loads(args.plan.read_text()) != plan:
                raise ValueError("Reviewed centering plan no longer matches inputs/statistics/source")
        write_json(args.out / "plan.json", plan)
        write_statistics(args.out / "statistics.pt", means, stats)
        if args.prepare_only:
            print(json.dumps({"status": "statistics/config-only; no fit", "planSha256": file_digest(args.out / "plan.json"),
                              "queryRows": stats["queryRows"], "catalogRows": stats["catalogRows"],
                              "statisticsSha256": statistics_digest(stats), "rawCenteredNorms": stats["rawCenteredNorms"]}, indent=2))
            return
        run_reviewed(cache, args.out, prepared, args.reviewed_plan_sha256)
    except BaseException as error:
        write_json(args.out / "FAILED.json", {"type": type(error).__name__, "message": str(error), "partialArtifactsPreserved": True})
        raise


if __name__ == "__main__":
    main()
