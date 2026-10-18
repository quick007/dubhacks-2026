"""Reviewed paired data -> 48 fresh query forwards -> fixed fits/replay metrics.

Catalog features are read-only, content/key/checksum verified, with zero catalog
forwards. --preflight-only is CPU-only; the normal command loads native MLX.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import psutil
import torch

from . import REPRESENTATION
from .evaluate import evaluate_split
from .fit_runner import FIT_CONFIG, fit_arm, freeze_checkpoints, require_frozen
from .inputs import canonical, catalog_text, digest, request_text
from .paired_contract import pair_metrics, read_labels, verify_freeze
from .preflight import DEFAULT_MODEL, file_digest
from .runner import LIMITS, execution_receipt, source_hashes
from .runtime import FeatureStore, NativeEncoder, assert_model_files, write_json
from .scorer import assemble_grid, roster
from .training import read_training_inputs


def prepare(dataset, prior, tokenizer):
    """Public preflight + old catalog artifact verification; no held-out targets."""
    dataset, prior = Path(dataset).resolve(), Path(prior)
    frozen = verify_freeze(dataset, public_only=True)
    previous = json.loads((prior / "preflight.json").read_text())
    provenance = json.loads((prior / "model-verification.json").read_text())
    if file_digest(dataset / "catalog.json") != previous["inputHashes"]["catalog.json"]:
        raise ValueError("Catalog changed: cached vectors cannot be reused")
    if json.loads((prior / "report.json").read_text())["status"] != "completed":
        raise ValueError("Preserved native feature run must have completed")
    store = FeatureStore.open_existing(prior / "features", provenance)
    catalog = {p["id"]: p for p in json.loads((dataset / "catalog.json").read_text())["products"]}
    catalog_rows = {}
    for row in previous["records"]:
        if row["role"] != "catalog":
            continue
        pid, vid = row["member"]
        product = catalog[pid]
        variant = next(v for v in product["presentationVariants"] if v["id"] == vid)
        row = {**row, "text": catalog_text(product, variant)}
        store.load(row)
        catalog_rows[pid, vid] = row
    expected_actions = {(pid, v["id"]) for pid, p in catalog.items() for v in p["presentationVariants"]}
    if set(catalog_rows) != expected_actions:
        raise ValueError("Verified old catalog features are incomplete")
    old_ids = {r["member"] for r in previous["records"] if r["role"] == "request"}
    requests, query_rows, lengths = {}, {}, []
    for split in ("train", "validation", "test"):
        path = dataset / f"public/{split}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        requests[split] = {r["recordId"]: r for r in records}
        if len(records) != len(requests[split]) or len(records) != frozen["splits"][split]["requests"]:
            raise ValueError("Paired public membership mismatch")
        for rid, request in requests[split].items():
            if rid in old_ids or rid in query_rows:
                raise ValueError("Paired requests must have fresh unique external IDs")
            text = request_text(request, catalog)
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if not 0 < len(ids) <= 512:
                raise ValueError("Paired query exceeds token cap; refusing truncation")
            query_rows[rid] = {"role": "request", "member": rid, "split": split, "text": text,
                               "textSha256": digest(text), "tokenIds": ids, "tokenSha256": digest(canonical(ids)),
                               "tokens": len(ids), "batchShape": [1, len(ids)]}
            lengths.append(len(ids))
            if any(a not in catalog_rows for a in roster(request, catalog)):
                raise ValueError("Paired request references an unverified catalog variant")
    if len(query_rows) != frozen["nativeQueryForwardsRequired"] or len(query_rows) > 48:
        raise ValueError("Paired experiment is bounded to the frozen 48 queries")
    receipt = {"freezeSha256": file_digest(dataset / "FREEZE.json"), "freeze": frozen,
               "priorReportSha256": file_digest(prior / "report.json"),
               "verifiedCatalogEntries": len(catalog_rows), "nativeCatalogForwardsRequired": 0,
               "nativeQueryForwardsRequired": len(query_rows), "queryTokenRange": [min(lengths), max(lengths)],
               "queryTotalTokens": sum(lengths), "evidenceSource": provenance["evidenceSource"],
               "sourceHashes": source_hashes(), "execution": execution_receipt(),
               "queryRecords": [{k: v for k, v in row.items() if k != "text"} for row in query_rows.values()]}
    return receipt, requests, query_rows, catalog_rows, catalog, store, provenance


def extract_queries(stage, requests, query_rows, catalog_rows, catalog, cached, fresh, encoder, out, *, started, phase_started, model=None):
    queries, features, observations = {}, {}, []
    def guard():
        now = time.perf_counter()
        if now - started > LIMITS["totalExtractionWorkSeconds"] or (stage == "train" and now - started > LIMITS["trainingExtractionSeconds"]):
            raise TimeoutError("Fixed paired extraction work budget exceeded")
        if stage in ("train", "validation") and len(queries) < 2 and now - phase_started > LIMITS["pilotSeconds"]:
            raise TimeoutError("Fixed paired model-load/query pilot budget exceeded")
    for rid, request in requests.items():
        guard()
        start = time.perf_counter()
        row = {**query_rows[rid], "text": request_text(request, catalog)}
        if fresh.exists(row):
            raise ValueError("Fresh paired query cannot reuse a prior request artifact")
        values, timing = encoder.encode(row["text"], row)
        guard()
        if (timing.get("batchRows") != 1 or timing.get("paddedTokensPerRow") != row["tokens"]
                or timing.get("backboneForwardCalls") != 1 or timing.get("truncatedRows") != 0):
            raise ValueError("Paired encoder did not perform exactly one complete B1 query forward")
        query, _ = fresh.save(row, values, timing)
        query_ms = (time.perf_counter() - start) * 1000
        load_start = time.perf_counter()
        actions = roster(request, catalog)
        matrix = torch.stack([cached.load(catalog_rows[action])[0][0] for action in actions])
        load_ms = (time.perf_counter() - load_start) * 1000
        fusion_ms = None
        if model is not None:
            before_head = time.perf_counter()
            with torch.no_grad():
                assemble_grid(request, catalog, actions, model(query[0], matrix), len(request["candidateProductIds"]))
            fusion_ms = (time.perf_counter() - before_head) * 1000
        queries[rid], features[rid] = query[0], matrix
        observation = {"stage": stage, "recordId": rid, "catalogCacheCondition": "warm",
                       "catalogHits": len(actions), "nativeCatalogForwards": 0, "freshQueryForwardCalls": 1,
                       "freshQueryNativeEncodeMs": timing["encodeTotalMs"],
                       "freshQueryConstructEncodeSaveVerifyMs": query_ms, "catalogHashLoadVerifyMs": load_ms,
                       "fusionCpuScoreMasksGridMs": fusion_ms, "observedPipelineWallMs": (time.perf_counter() - start) * 1000,
                       "processRssBytes": psutil.Process().memory_info().rss, "memory": encoder.memory()}
        observations.append(observation)
        print(json.dumps({"event": "paired-query-extracted", **observation}, allow_nan=False), flush=True)
    write_json(Path(out) / f"{stage}-native-observations.json", observations)
    return queries, features, {"queryForwards": len(requests), "catalogForwards": 0, "memory": encoder.memory()}


def run_prepared(out, dataset, prepared, encoder_factory, *, started=None):
    out, dataset = Path(out), Path(dataset).resolve()
    receipt, requests, query_rows, catalog_rows, catalog, cached, provenance = prepared
    started = time.perf_counter() if started is None else started
    torch.set_num_threads(1)
    fresh = FeatureStore(out / "features", provenance)
    train, labels, hashes = read_training_inputs(dataset)
    if canonical(train) != canonical(requests["train"]) or hashes["labels/train.jsonl"] != receipt["freeze"]["files"]["labels/train.jsonl"]:
        raise ValueError("Training inputs changed after data review")
    config = {"representation": REPRESENTATION, "pairedTeacherVersion": receipt["freeze"]["version"],
              "reviewedDataFreezeSha256": receipt["freezeSha256"], "fit": FIT_CONFIG, "limits": LIMITS,
              "execution": receipt["execution"], "sourceHashes": receipt["sourceHashes"], "trainHashes": hashes,
              "evidenceSource": provenance["evidenceSource"], "nativeCatalogForwardsAllowed": 0,
              "nativeQueryForwardsAllowed": 48, "generatesCopy": False}
    write_json(out / "config.json", config)
    phases = {}
    load_start = time.perf_counter()
    encoder = encoder_factory()
    try:
        if encoder.evidence_source != provenance["evidenceSource"]:
            raise ValueError("Paired encoder/provenance mismatch")
        q, x, phases["train"] = extract_queries("train", train, query_rows, catalog_rows, catalog, cached, fresh, encoder, out, started=started, phase_started=load_start)
        phases["train"]["modelLoadSeconds"] = encoder.load_seconds
        phases["train"]["phaseWallSeconds"] = time.perf_counter() - load_start
    finally:
        encoder.close()
    models, fits = {}, {}
    for arm in ("fusion", "prior"):
        models[arm], fits[arm] = fit_arm(arm, out, train, labels, catalog, q, x)
    models["random"] = freeze_checkpoints(out, fits, hashes, provenance["evidenceSource"])
    results = {"train": evaluate_split(models, train, labels, catalog, q, x)}
    all_labels = {"train": labels}
    if source_hashes() != receipt["sourceHashes"]:
        raise ValueError("Paired runner sources changed during extraction/fit")
    require_frozen(out)
    load_start = time.perf_counter()
    encoder = encoder_factory()
    try:
        if encoder.evidence_source != provenance["evidenceSource"]:
            raise ValueError("Paired held-out encoder/provenance mismatch")
        for split in ("validation", "test"):
            q, x, phases[split] = extract_queries(split, requests[split], query_rows, catalog_rows, catalog, cached, fresh, encoder, out, started=started, phase_started=load_start, model=models["fusion"])
            all_labels[split] = read_labels(dataset, split, requests[split], out)
            results[split] = evaluate_split(models, requests[split], all_labels[split], catalog, q, x)
        phases["heldoutLifecycle"] = {"modelLoadSeconds": encoder.load_seconds,
                                      "phaseWallSeconds": time.perf_counter() - load_start, "memory": encoder.memory()}
    finally:
        encoder.close()
    require_frozen(out)
    verify_freeze(dataset, public_only=False)
    controls = json.loads((dataset / "paired-controls.json").read_text())
    for split, result in results.items():
        result["counterfactualPairs"] = pair_metrics(result, controls, all_labels[split], split)
        write_json(out / f"{split}-evaluation.json", result)
    # Recheck immutable catalog vectors after all queries. No projected/personalized
    # scores are reused, and no catalog cache entry is written by this runner.
    for row in catalog_rows.values():
        cached.load(row)
    if source_hashes() != receipt["sourceHashes"]:
        raise ValueError("Paired source snapshot changed during evaluation")
    report = {"status": "completed", "execution": receipt["execution"], "config": config,
              "evidenceSource": provenance["evidenceSource"], "generatesCopy": False,
              "catalogReuse": {"verifiedEntries": len(catalog_rows), "readOnly": True, "nativeForwards": 0,
                               "sourceReportSha256": receipt["priorReportSha256"]},
              "fits": fits, "phases": phases, "nativeQueryForwards": sum(row.get("queryForwards", 0) for row in phases.values()),
              "quality": {split: result["summary"] for split, result in results.items()},
              "counterfactualPairs": {split: result["counterfactualPairs"] for split, result in results.items()},
              "wallSeconds": time.perf_counter() - started,
              "limitations": receipt["freeze"]["limitations"] + ["No fresh copy or full-grid quality claim.",
                  "Public-fact teacher behavior is a controlled learning test, not human utility.",
                  "New request combinations overlap some previously inspected old public contexts; catalog is replayed."]}
    write_json(out / "report.json", report)
    return report


def main():
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("experiments/ranker/pretrained/catalog_fusion/paired_v1"))
    parser.add_argument("--prior", type=Path, default=Path("data/ranker/pretrained/catalog-user-fusion-v1-attempt-1"))
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--reviewed-freeze-sha256", help="Exact FREEZE.json checksum reviewed before the native fit")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "STARTED.json", {"startedAtUtc": datetime.now(timezone.utc).isoformat(),
                                          "preflightOnly": args.preflight_only, "reviewedFreezeSha256": args.reviewed_freeze_sha256})
    try:
        if not args.preflight_only and args.reviewed_freeze_sha256 != file_digest(args.dataset / "FREEZE.json"):
            raise ValueError("Exact reviewed data-freeze checksum required before native extraction/fitting")
        provenance = json.loads((args.prior / "model-verification.json").read_text())
        assert_model_files(args.model, provenance)
        prepared = prepare(args.dataset, args.prior, Tokenizer.from_file(str(args.model / "tokenizer.json")))
        write_json(args.out / "paired-preflight.json", prepared[0])
        if args.preflight_only:
            print(json.dumps({k: prepared[0][k] for k in ("freezeSha256", "verifiedCatalogEntries", "nativeQueryForwardsRequired", "nativeCatalogForwardsRequired", "queryTokenRange", "queryTotalTokens")}, indent=2))
            return
        def factory():
            assert_model_files(args.model, provenance)
            return NativeEncoder(args.model)
        run_prepared(args.out, args.dataset, prepared, factory, started=started)
    except BaseException as error:
        write_json(args.out / "FAILED.json", {"exception": type(error).__name__, "message": str(error),
                                             "elapsedSeconds": time.perf_counter() - started, "partialArtifactsPreserved": True})
        raise


if __name__ == "__main__":
    main()
