"""Real local extraction -> fixed CPU fits -> post-freeze held-out replay.

This command loads the pinned native backbone. Importing it or --help does not.
Run only in the serialized native lane; there is no copy-generation path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import psutil
import torch

from . import REPRESENTATION
from .evaluate import evaluate_split, evaluation_labels
from .fit_runner import FIT_CONFIG, fit_arm, freeze_checkpoints, require_frozen
from .inputs import canonical, catalog_text, request_text
from .preflight import DEFAULT_MODEL, SPLITS, file_digest, inventory, public_corpus
from .runtime import FeatureStore, NativeEncoder, assert_model_files, verify_model, write_json
from .scorer import assemble_grid, roster
from .training import read_training_inputs

LIMITS = {"memoryGiB": 6, "maxTokens": 512, "pilotSeconds": 120,
          "trainingExtractionSeconds": 600, "totalExtractionWorkSeconds": 900}


def execution_receipt():
    root = Path(__file__).resolve().parents[4]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True).stdout.strip()
        status = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain", "--", "experiments/ranker/pretrained/catalog_fusion",
                                 "experiments/ranker/pretrained/backbone.py", "experiments/ranker/pretrained/boundary.py"],
                                cwd=root, check=True, text=True, capture_output=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, ["Git execution identity unavailable; source hashes remain recorded"]
    versions = {}
    for name in ("torch", "mlx", "mlx-lm", "tokenizers", "numpy", "psutil"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"atUtc": datetime.now(timezone.utc).isoformat(), "commit": commit, "relevantSourceStatus": status,
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": versions, "argv": sys.argv,
            "nativeGuard": "fixed 6 GiB active AND observed cumulative MLX peak, checked after load and before/after encode"}


def source_hashes():
    directory = Path(__file__).parent
    files = [*sorted(directory.glob("*.py")), directory.parent / "backbone.py", directory.parent / "boundary.py"]
    return {str(path.resolve()): file_digest(path) for path in files}


def corpus_rows(dataset, preflight):
    rows, _, hashes = public_corpus(dataset)
    if hashes != preflight["inputHashes"]:
        raise ValueError("Public dataset changed after token preflight")
    key = lambda row: (row["role"], canonical(row["member"]))
    expected = {key(row): row for row in preflight["records"]}
    actual = {key(row): row for row in rows}
    if len(expected) != len(preflight["records"]) or len(actual) != len(rows) or set(expected) != set(actual):
        raise ValueError("Catalog/request external membership is not unique and complete")
    return {member: {**actual[member], **expected[member]} for member in actual}


def public_requests(dataset):
    result = {}
    for split in SPLITS:
        path = Path(dataset).resolve() / f"public/{split}.jsonl"
        if path.resolve() != path:
            raise ValueError("Public request file cannot redirect")
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        result[split] = {r["recordId"]: r for r in records}
        if len(result[split]) != len(records):
            raise ValueError("Duplicate request in split")
    return result


def extract_stage(stage, requests, catalog, rows, store, encoder, out, emit, *, run_started,
                  phase_started, limits=LIMITS, model=None):
    """One fresh query per request, content-checked catalog misses/hits, B1 only."""
    queries, features, observations = {}, {}, []
    forwards = 0
    pilot_deadline = phase_started + limits["pilotSeconds"]
    phase_deadline = run_started + limits["trainingExtractionSeconds"] if stage == "train" else float("inf")
    deadline = min(run_started + limits["totalExtractionWorkSeconds"], phase_deadline)

    def guard():
        now = time.perf_counter()
        if now > deadline or (stage in ("train", "validation") and forwards < 2 and now > pilot_deadline):
            raise TimeoutError("Fixed extraction/pilot budget exceeded; no automatic guard increase")

    def encode(row):
        nonlocal forwards
        guard()
        values, timing = encoder.encode(row["text"], row)
        forwards += 1
        guard()
        if stage in ("train", "validation") and forwards <= 2 and time.perf_counter() > pilot_deadline:
            raise TimeoutError("Fixed two-forward pilot budget exceeded")
        if (timing.get("batchRows") != 1 or timing.get("paddedTokensPerRow") != row["tokens"]
                or timing.get("truncatedRows") != 0 or timing.get("backboneForwardCalls") != 1):
            raise ValueError("Encoder did not perform exactly one complete B1 forward")
        return store.save(row, values, timing)

    # Check model load against limits before the first native forward.
    guard()
    for rid, request in requests.items():
        guard()
        start = time.perf_counter()
        query_start = time.perf_counter()
        query_row = {**rows[("request", canonical(rid))], "text": request_text(request, catalog)}
        if store.exists(query_row):
            raise ValueError("Fresh-query extraction cannot reuse an existing request artifact")
        query, query_receipt = encode(query_row)
        query_wall_ms = (time.perf_counter() - query_start) * 1000
        catalog_start = time.perf_counter()
        hits, misses, catalog_native_ms = 0, 0, 0.
        candidate_features = []
        actions = roster(request, catalog)
        for pid, vid in actions:
            product = catalog[pid]
            variant = next(v for v in product["presentationVariants"] if v["id"] == vid)
            row = {**rows[("catalog", canonical([pid, vid]))], "text": catalog_text(product, variant)}
            if store.exists(row):
                values, receipt = store.load(row)
                hits += 1
            else:
                values, receipt = encode(row)
                misses += 1
                catalog_native_ms += receipt["timing"]["encodeTotalMs"]
            candidate_features.append(values[0])
        matrix = torch.stack(candidate_features)
        catalog_ms = (time.perf_counter() - catalog_start) * 1000
        fusion_ms, grid = None, None
        if model is not None:
            head_start = time.perf_counter()
            with torch.no_grad():
                grid = assemble_grid(request, catalog, actions, model(query[0], matrix), len(request["candidateProductIds"]))
            fusion_ms = (time.perf_counter() - head_start) * 1000
        wall_ms = (time.perf_counter() - start) * 1000
        queries[rid], features[rid] = query[0], matrix
        observation = {"stage": stage, "recordId": rid, "catalogCacheCondition": "warm" if not misses else ("cold" if not hits else "mixed"),
                       "catalogHits": hits, "catalogMisses": misses, "freshQueryForwardCalls": 1,
                       "catalogResolveConstructHashLoadOrExtractMs": catalog_ms,
                       "catalogMissNativeEncodeMs": catalog_native_ms,
                       "freshQueryConstructEncodeSaveVerifyMs": query_wall_ms,
                       "freshQueryNativeEncodeMs": query_receipt["timing"]["encodeTotalMs"],
                       "fusionCpuScoreMasksGridMs": fusion_ms,
                       "observedPipelineWallMs": wall_ms,
                       "scope": "fresh query + verified catalog resolution + artifact persistence" + (" + fitted fusion/masks/grid" if model is not None else "; inputs only before fit"),
                       "modelResident": True, "cumulativeStageForwards": forwards,
                       "processRssBytes": psutil.Process().memory_info().rss,
                       "gridAssemblyProbe": grid}
        observations.append(observation)
        emit("query-extracted", **{key: value for key, value in observation.items() if key != "gridAssemblyProbe"})
    write_json(Path(out) / f"{stage}-native-observations.json", observations)
    return queries, features, {"forwards": forwards, "observations": observations, "memory": encoder.memory()}


def run_prepared(out, dataset, preflight, provenance, encoder_factory, *, run_started=None, limits=LIMITS):
    """Orchestration hook; tests inject an explicitly marked CPU stub encoder."""
    out, dataset = Path(out), Path(dataset).resolve()
    if run_started is None:
        run_started = time.perf_counter()
    events = out / "events.jsonl"
    with events.open("x"):
        pass

    def emit(event, **fields):
        row = {"event": event, "atUtc": datetime.now(timezone.utc).isoformat(), **fields}
        with events.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    torch.set_num_threads(1)
    sources = source_hashes()
    rows = corpus_rows(dataset, preflight)
    requests = public_requests(dataset)
    train, labels, train_hashes = read_training_inputs(dataset)
    if canonical(train) != canonical(requests["train"]):
        raise ValueError("Training loader and preflight public membership disagree")
    catalog = {p["id"]: p for p in json.loads((dataset / "catalog.json").read_text())["products"]}
    config = {"representation": REPRESENTATION, "evidenceSource": provenance["evidenceSource"],
              "execution": execution_receipt(),
              "fit": FIT_CONFIG, "limits": limits, "inputHashes": preflight["inputHashes"],
              "trainHashes": train_hashes, "sourceHashes": sources,
              "generatesCopy": False, "freshQueryRequired": True,
              "featureRepresentationDiffersFromJointReadouts": True}
    write_json(out / "config.json", config)
    store = FeatureStore(out / "features", provenance)
    phases = {}
    phase_started = time.perf_counter()
    emit("native-stage-start", stage="train")
    encoder = encoder_factory()
    try:
        if encoder.evidence_source != provenance["evidenceSource"]:
            raise ValueError("Encoder evidence source does not match verified provenance")
        queries, features, phases["train"] = extract_stage("train", train, catalog, rows, store, encoder, out, emit,
                                                          run_started=run_started, phase_started=phase_started, limits=limits)
        phases["train"]["modelLoadSeconds"] = encoder.load_seconds
        phases["train"]["phaseWallSeconds"] = time.perf_counter() - phase_started
    finally:
        encoder.close()
    models, fits = {}, {}
    for arm in ("fusion", "prior"):
        emit("cpu-fit-start", arm=arm)
        models[arm], fits[arm] = fit_arm(arm, out, train, labels, catalog, queries, features)
        emit("cpu-fit-finished", arm=arm, **{key: value for key, value in fits[arm].items() if key in ("lossBefore", "lossAfter", "seconds", "sha256")})
    models["random"] = freeze_checkpoints(out, fits, train_hashes, provenance["evidenceSource"])
    emit("checkpoints-frozen", checkpointHashes={arm: row["sha256"] for arm, row in fits.items()})
    # No held-out target loader is reachable before both fit/checkpoint receipts.
    evaluation = {"train": evaluate_split(models, train, labels, catalog, queries, features)}
    write_json(out / "train-evaluation.json", evaluation["train"])
    if source_hashes() != sources:
        raise ValueError("Runner/encoder sources changed during the attempt")
    require_frozen(out)
    phase_started = time.perf_counter()
    emit("native-stage-start", stage="heldout")
    encoder = encoder_factory()
    targets_receipts = {}
    try:
        if encoder.evidence_source != provenance["evidenceSource"]:
            raise ValueError("Held-out encoder provenance differs")
        for split in ("validation", "test", "diagnostics"):
            q, x, phases[split] = extract_stage(split, requests[split], catalog, rows, store, encoder, out, emit,
                                                run_started=run_started, phase_started=phase_started, limits=limits, model=models["fusion"])
            targets, targets_receipts[split] = evaluation_labels(dataset, split, requests[split], out)
            evaluation[split] = evaluate_split(models, requests[split], targets, catalog, q, x)
            write_json(out / f"{split}-evaluation.json", evaluation[split])
        phases["heldoutLifecycle"] = {"modelLoadSeconds": encoder.load_seconds,
                                      "phaseWallSeconds": time.perf_counter() - phase_started,
                                      "memory": encoder.memory()}
    finally:
        encoder.close()
    require_frozen(out)
    # Final consistency checks include labels/input membership without sending it to F.
    if corpus_rows(dataset, preflight).keys() != rows.keys() or source_hashes() != sources:
        raise ValueError("Input/source snapshot changed during evaluation")
    for relative, sha in train_hashes.items():
        if file_digest(dataset / relative) != sha:
            raise ValueError("Training input changed during evaluation")
    phase_summary = {phase: {key: value for key, value in data.items() if key != "observations"} for phase, data in phases.items()}
    cache_summary = {}
    for phase, data in phases.items():
        for row in data.get("observations", []):
            key = phase + ":" + row["catalogCacheCondition"]
            cache_summary.setdefault(key, []).append(row["observedPipelineWallMs"])
    report = {"status": "completed", "representation": REPRESENTATION,
              "execution": config["execution"],
              "evidenceSource": provenance["evidenceSource"], "generatesCopy": False,
              "config": config, "fits": fits, "phases": phase_summary,
              "nativeForwards": sum(data.get("forwards", 0) for data in phases.values()),
              "evaluationTargets": targets_receipts,
              "quality": {split: result["summary"] for split, result in evaluation.items()},
              "cacheObservations": {key: {"requests": len(values), "meanObservedPipelineMs": sum(values) / len(values)} for key, values in cache_summary.items()},
              "wallSecondsIncludingHashLoadFitEvaluation": time.perf_counter() - run_started,
              "jointReference": {"status": "not reencoded in this runner; keep existing corrected joint evidence separate",
                                 "checkpoint": "data/ranker/pretrained/evidence-plan-v1/heads.pt"},
              "limitations": ["Synthetic fixture, replay held-out outcomes; no human lift or untouched-test claim.",
                              "At most four candidates: assembly probes are not labeled twelve-/twenty-four-card quality.",
                              "No copy generation, no complete storefront claim; image features are text descriptors.",
                              "Catalog cold/mixed/warm traces may be different requests, not a matched speedup experiment.",
                              "Saved vector bytes are checked exactly; matched fresh-vs-cached native extraction parity is not measured.",
                              "Epoch caps are fixed; no fit retries or outcome-driven hyperparameter selection."]}
    write_json(out / "report.json", report)
    emit("completed", nativeForwards=report["nativeForwards"], report=str(out / "report.json"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/ranker/datasets/v2"))
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--out", type=Path, required=True, help="New output directory; existing paths are refused")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "STARTED.json", {"startedAtUtc": datetime.now(timezone.utc).isoformat(),
                                          "dataset": str(args.dataset.resolve()), "model": str(args.model.resolve()),
                                          "fit": FIT_CONFIG, "limits": LIMITS})
    try:
        preflight = inventory(args.dataset, args.model, 512)
        write_json(args.out / "preflight.json", preflight)
        provenance = verify_model(args.model, preflight)
        write_json(args.out / "model-verification.json", provenance)
        def factory():
            assert_model_files(args.model, provenance)
            return NativeEncoder(args.model)
        run_prepared(args.out, args.dataset, preflight, provenance, factory, run_started=started)
    except BaseException as error:
        write_json(args.out / "FAILED.json", {"exception": type(error).__name__, "message": str(error),
                                             "elapsedSeconds": time.perf_counter() - started,
                                             "partialArtifactsPreserved": True})
        raise


if __name__ == "__main__":
    main()
