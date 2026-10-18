"""Reviewed CPU-only seven-atom product distillation; never loads a backbone."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import torch

from .cached_inputs import CachedPairedInputs
from .centered import compute_statistics, load_centered, statistics_digest
from .evaluate import evaluate_split, readout, _sum_images
from .fit_runner import FIT_CONFIG, require_frozen
from .inputs import canonical
from .paired_contract import pair_metrics, verify_freeze
from .paired_data import ATOMS
from .preflight import file_digest
from .runner import execution_receipt, source_hashes
from .runtime import write_json
from .scorer import roster
from .semantic_factors import (POLICY, VERSION, FACTOR_PARAMETERS, SemanticFactors, SemanticScorer,
                               atom_metrics, distillation_loss, load_factors, load_image,
                               public_targets, save_factors)

SCHEDULE = {k: v for k, v in FIT_CONFIG.items() if k != "rank"}
SCHEDULE.update({"atomFactors": 7, "selection": "one reviewed semantic-factor method; no held-out tuning"})


def prepare(cache, image_run):
    """Read only TRAIN vectors/public targets; original image weights stay fixed."""
    image_run = Path(image_run)
    original = require_frozen(image_run)
    image_receipt = original["checkpoints"]["fusion"]
    image = load_centered(image_run / image_receipt["file"], image_receipt["sha256"], original["statisticsSha256"])
    q, x, items, raw = cache.load_split("train")
    means, statistics = compute_statistics(cache.requests["train"], cache.catalog, q, items,
                                          provenance=image.statistics_receipt["provenance"])
    if statistics != image.statistics_receipt:
        raise ValueError("Exact original TRAIN centering statistics required")
    query_members, item_members = sorted(q), sorted(items)
    qt, ct, targets = public_targets(cache.requests["train"], cache.catalog, query_members, item_members)
    metadata = {"policy": POLICY, "statistics": statistics, "statisticsSha256": statistics_digest(statistics),
                "trainingPublicTargets": targets, "queryFrequency": qt.mean(0).tolist(),
                "factorParameters": FACTOR_PARAMETERS,
                "imageCheckpoint": {"file": "centered-image.pt", "sha256": image_receipt["sha256"],
                                    "source": str((image_run / image_receipt["file"]).resolve()),
                                    "sourceFreezeSha256": file_digest(image_run / "CHECKPOINTS_FROZEN.json")}}
    plan = {"version": VERSION, "metadata": metadata, "fit": SCHEDULE,
            "inputSources": cache.source_receipts, "trainingRawReceipts": raw,
            "sourceHashes": source_hashes(), "nativeForwards": 0,
            "rankingLabelFilesOpenedForTraining": [], "fitInPreparation": False,
            "controls": ["independent catalog-factor item prior", "seeded random factors", "permuted query",
                         "nonlearned exact public-tag oracle; diagnostic only"],
            "evaluationBoundary": "held-out vectors, targets and pair controls opened only after all checkpoint freezes"}
    return plan, means, statistics, q, x, items, qt, ct


def fit_arm(arm, out, prepared):
    plan, means, stats, q, _, items, qt, ct = prepared
    model = SemanticFactors(means, stats, qt.mean(0), arm=arm, seed=FIT_CONFIG["seed"])
    qr, cr = torch.stack([q[k] for k in sorted(q)]), torch.stack([items[k] for k in sorted(items)])
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=FIT_CONFIG["learningRate"], weight_decay=FIT_CONFIG["weightDecay"])
    started = time.perf_counter()
    first, first_loss = None, None
    with (Path(out) / f"{arm}-epochs.jsonl").open("x") as handle:
        for epoch in range(FIT_CONFIG["epochs"]):
            if time.perf_counter() - started > FIT_CONFIG["maxSecondsPerArm"]:
                raise TimeoutError("Fixed semantic fit budget exceeded")
            optimizer.zero_grad(set_to_none=True)
            loss, roles = distillation_loss(model, qr, cr, qt, ct)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite semantic BCE")
            loss.backward()
            grads = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
            expected = {n for n, p in model.named_parameters() if p.requires_grad}
            if set(grads) != expected or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
                raise FloatingPointError("Missing/nonfinite semantic gradients")
            if first is None:
                first, first_loss = grads, loss.item()
                if any(value <= 0 for value in first.values()):
                    raise FloatingPointError("Semantic head lacks genuine initial gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite semantic parameters")
            handle.write(json.dumps({"epoch": epoch + 1, "lossBeforeStep": loss.item(), **roles,
                                     "gradientL2": grads, "elapsedSeconds": time.perf_counter() - started}) + "\n")
    change = {n: (p - before[n]).abs().sum().item() for n, p in model.named_parameters()}
    if any(change[n] <= 0 for n in expected):
        raise AssertionError("Semantic parameters did not update")
    if arm == "prior" and (change["query_weight"] or change["query_bias"]):
        raise AssertionError("Item prior acquired query conditioning")
    sha = save_factors(model, Path(out) / f"{arm}.pt", plan["metadata"])
    restored = load_factors(Path(out) / f"{arm}.pt", sha, plan["metadata"])
    with torch.no_grad():
        if any(not torch.equal(a, b) for a, b in zip(model.probabilities(qr, cr), restored.probabilities(qr, cr), strict=True)):
            raise AssertionError("Semantic raw-feature prediction reload mismatch")
        final, roles = distillation_loss(restored, qr, cr, qt, ct)
    report = {"arm": arm, "trained": True, "sha256": sha, "factorParameters": FACTOR_PARAMETERS,
              "trainableParameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "objective": POLICY["objective"] if arm == "fusion" else "independent unique TRAIN catalog BCE only",
              "lossBefore": first_loss, "lossAfter": final.item(), "finalRoles": roles,
              "firstGradientL2": first, "parameterAbsoluteChange": change,
              "seconds": time.perf_counter() - started, "exactTrainingPredictionReload": True}
    if report["seconds"] > FIT_CONFIG["maxSecondsPerArm"]:
        raise TimeoutError("Semantic fixed fit time budget exceeded")
    write_json(Path(out) / f"{arm}-fit.json", report)
    return restored, report


def require_semantic_frozen(out, metadata):
    out = Path(out)
    frozen = json.loads((out / "CHECKPOINTS_FROZEN.json").read_text())
    if (frozen.get("version") != VERSION or frozen.get("metadata") != metadata or frozen.get("config") != SCHEDULE
            or set(frozen.get("checkpoints", {})) != {"fusion", "prior", "random"}):
        raise ValueError("Distinct semantic checkpoint contract required")
    for arm, checkpoint in frozen["checkpoints"].items():
        if checkpoint["file"] != f"{arm}.pt" or file_digest(out / checkpoint["file"]) != checkpoint["sha256"]:
            raise ValueError("Semantic checkpoint changed before evaluation")
    load_image(Path(out) / metadata["imageCheckpoint"]["file"], metadata)
    return {a: load_factors(Path(out) / v["file"], v["sha256"], metadata)
            for a, v in frozen["checkpoints"].items()}


def read_semantic_labels(cache, split, out, metadata):
    """Distinct semantic freeze gate before any train/held-out ranking targets."""
    require_semantic_frozen(out, metadata)
    if split not in ("train", "validation", "test"):
        raise ValueError("Known semantic evaluator split required")
    verify_freeze(cache.dataset, public_only=True)
    relative = f"labels/{split}.jsonl"
    path = cache.dataset / relative
    if path.resolve() != path or file_digest(path) != cache.freeze["files"][relative]:
        raise ValueError("Frozen semantic evaluation labels changed")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {row["recordId"]: row for row in rows}
    if len(rows) != len(result) or set(result) != set(cache.requests[split]):
        raise ValueError("Exact post-freeze evaluation membership required")
    return result


def freeze(out, prepared, factors, fits):
    plan, means, stats, _, _, _, qt, _ = prepared
    metadata = plan["metadata"]
    if set(fits) != {"fusion", "prior"} or not all(f["trained"] and f["exactTrainingPredictionReload"] for f in fits.values()):
        raise ValueError("Both independent factor fits must complete before freeze")
    random = SemanticFactors(means, stats, qt.mean(0), arm="random", seed=FIT_CONFIG["seed"])
    random_sha = save_factors(random, Path(out) / "random.pt", metadata)
    checks = {a: {"file": f"{a}.pt", "sha256": f["sha256"], "trained": True} for a, f in fits.items()}
    checks["random"] = {"file": "random.pt", "sha256": random_sha, "trained": False}
    # Copy bytes without rewriting the old checkpoint or changing its identity.
    with Path(metadata["imageCheckpoint"]["source"]).open("rb") as src, (Path(out) / "centered-image.pt").open("xb") as dst:
        shutil.copyfileobj(src, dst)
    load_image(Path(out) / "centered-image.pt", metadata)
    write_json(Path(out) / "CHECKPOINTS_FROZEN.json", {"version": VERSION, "config": SCHEDULE,
               "metadata": metadata, "checkpoints": checks, "frozenAtUtc": datetime.now(timezone.utc).isoformat(),
               "rankingLabelsOpenedForTraining": False, "heldoutLabelsOpenedBeforeFreeze": False})
    return require_semantic_frozen(out, metadata)


def factor_diagnostics(factors, queries, items, requests, catalog):
    qids, members = sorted(queries), sorted(items)
    qr, cr = torch.stack([queries[k] for k in qids]), torch.stack([items[k] for k in members])
    qt = torch.tensor([[float(a in requests[rid]["context"]["statedPreferences"]) for a in ATOMS] for rid in qids])
    ct = torch.tensor([[float(a in catalog[pid]["tags"]) for a in ATOMS] for pid, _ in members])
    result = {}
    with torch.no_grad():
        for arm in ("fusion", "prior", "random", "permuted-query"):
            model = factors["fusion" if arm == "permuted-query" else arm]
            used = torch.roll(qr, -1, 0) if arm == "permuted-query" else qr
            qp, cp = model.probabilities(used, cr)
            result[arm] = {"query": atom_metrics(qp, qt), "catalog": atom_metrics(cp, ct)}
    return result


def oracle_diagnostic(result, image, requests, catalog, queries, features, labels):
    """Explicit nonlearned teacher ceiling; never a serving scorer/checkpoint."""
    cases = {}
    with torch.no_grad():
        for rid, request in requests.items():
            values = image(queries[rid], features[rid]).clone()
            preferences = set(request["context"]["statedPreferences"])
            values[:, 0] = torch.tensor([len(preferences & set(catalog[pid]["tags"])) for pid, _ in roster(request, catalog)])
            cases[rid] = {mode: readout(values, request, catalog, labels[rid], guarded=mode == "guarded")
                          for mode in ("raw", "guarded")}
    summary = {"requests": len(requests), "nonlearnedDiagnosticOnly": True}
    for mode in ("raw", "guarded"):
        metrics = [case[mode]["metrics"] for case in cases.values()]
        summary[mode] = {k: sum(m[k] for m in metrics) / len(metrics) for k in ("bestProductAccuracy", "ndcgAt3", "abstain")}
        for key in ("oraclePresentation", "selectedPresentation", "allCandidatePresentation"):
            summary[mode][key] = _sum_images([m[key] for m in metrics])
    result["cases"]["public-tag-oracle"] = cases
    result["summary"]["public-tag-oracle"] = summary


def run(cache, out, prepared, reviewed_sha):
    start = time.perf_counter()
    plan, _, _, q, x, items, _, _ = prepared
    execution = execution_receipt()
    write_json(Path(out) / "config.json", {"execution": execution, "reviewedPlanSha256": reviewed_sha,
                                          "plan": plan, "nativeForwards": 0, "generatesCopy": False})
    factors, fits = {}, {}
    for arm in ("fusion", "prior"):
        factors[arm], fits[arm] = fit_arm(arm, out, prepared)
    factors = freeze(out, prepared, factors, fits)
    image = load_image(Path(out) / "centered-image.pt", plan["metadata"])
    models = {a: SemanticScorer(f, image) for a, f in factors.items()}
    results, diagnostics, labels = {}, {}, {}
    for split in ("train", "validation", "test"):
        require_semantic_frozen(out, plan["metadata"])
        if source_hashes() != plan["sourceHashes"]:
            raise ValueError("Semantic source snapshot changed during run")
        if split != "train":
            q, x, items, _ = cache.load_split(split)
        labels[split] = read_semantic_labels(cache, split, out, plan["metadata"])
        results[split] = evaluate_split(models, cache.requests[split], labels[split], cache.catalog, q, x)
        diagnostics[split] = factor_diagnostics(factors, q, items, cache.requests[split], cache.catalog)
        oracle_diagnostic(results[split], image, cache.requests[split], cache.catalog, q, x, labels[split])
    verify_freeze(cache.dataset, public_only=False)
    controls = json.loads((cache.dataset / "paired-controls.json").read_text())
    for split, result in results.items():
        result["counterfactualPairs"] = pair_metrics(result, controls, labels[split], split)
        result["atomDiagnostics"] = diagnostics[split]
        write_json(Path(out) / f"{split}-evaluation.json", result)
    # Recompute preparation from TRAIN again; exact original statistics/source/input contract.
    if prepare(cache, Path(plan["metadata"]["imageCheckpoint"]["source"]).parent)[0] != plan:
        raise ValueError("Semantic TRAIN inputs/means/source changed after evaluation")
    require_semantic_frozen(out, plan["metadata"])
    report = {"status": "completed", "version": VERSION, "execution": execution, "fit": SCHEDULE,
              "reviewedPlanSha256": reviewed_sha, "fits": fits, "metadata": plan["metadata"],
              "quality": {s: v["summary"] for s, v in results.items()},
              "counterfactualPairs": {s: v["counterfactualPairs"] for s, v in results.items()},
              "atomDiagnostics": diagnostics, "nativeForwards": 0, "rawFeatureCacheWrites": 0,
              "imageCheckpointBytesUnchanged": True, "trainingContractReverifiedAfterEvaluation": True,
              "wallSeconds": time.perf_counter() - start, "generatesCopy": False,
              "completeViewEvidence": {"aboveFoldThreeCards": "not evaluated", "fullStorefront": "not evaluated"},
              "limitations": ["Public seven-atom grammar distillation; no latent preference or conversion claim.",
                              "Old catalog and held-out outcomes already inspected/replayed; not untouched test evidence.",
                              "Frozen image behavior inherited from centered-v1; no new image, copy or section learning.",
                              "Cached CPU timing excludes native request extraction and full page generation."]}
    write_json(Path(out) / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("experiments/ranker/pretrained/catalog_fusion/paired_v1"))
    parser.add_argument("--queries", type=Path, default=Path("data/ranker/pretrained/catalog-user-fusion-paired-v1-attempt-1"))
    parser.add_argument("--catalog-run", type=Path, default=Path("data/ranker/pretrained/catalog-user-fusion-v1-attempt-1"))
    parser.add_argument("--image-run", type=Path, default=Path("data/ranker/pretrained/catalog-user-fusion-centered-v1-attempt-1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--reviewed-plan-sha256")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        cache = CachedPairedInputs(args.dataset, args.queries, args.catalog_run)
        prepared = prepare(cache, args.image_run)
        plan = prepared[0]
        if not args.prepare_only:
            if args.plan is None or file_digest(args.plan) != args.reviewed_plan_sha256:
                raise ValueError("Explicit reviewed semantic plan checksum required")
            if json.loads(args.plan.read_text()) != plan:
                raise ValueError("Reviewed semantic inputs/means/source plan changed")
        write_json(args.out / "plan.json", plan)
        if args.prepare_only:
            print(json.dumps({"status": "public TRAIN targets/config only; no fit", "planSha256": file_digest(args.out / "plan.json"),
                              "factorParameters": FACTOR_PARAMETERS, "imageSha256": plan["metadata"]["imageCheckpoint"]["sha256"]}, indent=2))
        else:
            run(cache, args.out, prepared, args.reviewed_plan_sha256)
    except BaseException as error:
        write_json(args.out / "FAILED.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
