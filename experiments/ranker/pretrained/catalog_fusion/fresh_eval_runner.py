"""No-fit fresh-catalog evaluator. --preflight-only is CPU-only; run needs review.

Normal execution performs exactly 72 catalog + 24 fresh-query native B1 forwards.
All trained weights/means stay frozen; private targets are read after the raw
feature/prediction freeze. There is no copy generation or optimizer path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

import psutil
import torch

from ..boundary import guard_reasons
from .centered import load_centered
from .evaluate import readout
from .fresh_eval_data import VERSION, verify_dataset
from .inputs import canonical, catalog_text, digest, request_text
from .paired_contract import pair_metrics
from .preflight import DEFAULT_MODEL, REVISION, file_digest
from .runner import execution_receipt, source_hashes
from .runtime import FeatureStore, NativeEncoder, assert_model_files, verify_model, write_json
from .scorer import assemble_grid, roster
from .semantic_factors import SemanticScorer, load_image
from .semantic_runner import factor_diagnostics, require_semantic_frozen

OFFSETS, CARD_COUNTS = (0, 12, 24), (12, 24)
LIMITS = {"memoryGiB": 6, "maxTokens": 512, "pilotSeconds": 60,
          "nativeExtractionSeconds": 180, "catalogForwards": 72, "queryForwards": 24}
DENOMINATORS = {"requests": 24, "catalogProducts": 36, "catalogVariants": 72,
                "productPairs": 12, "imagePairs": 12, "allCandidateImagePairs": 432,
                "rosterViewsPerQuery": 3, "gridSizes": [12, 24], "gridCasesPerArm": 144}


def public_inputs(dataset):
    dataset = Path(dataset).resolve()
    frozen = verify_dataset(dataset, private=False)
    products = json.loads((dataset / "catalog.json").read_text())["products"]
    catalog = {p["id"]: p for p in products}
    rows = [json.loads(l) for l in (dataset / "public/requests.jsonl").read_text().splitlines() if l.strip()]
    requests = {r["recordId"]: r for r in rows}
    if len(products) != 36 or len(catalog) != 36 or len(rows) != 24 or len(requests) != 24:
        raise ValueError("Fixed fresh catalog/request denominators changed")
    for request in requests.values():
        if set(request["candidateProductIds"]) != set(catalog) or len(roster(request, catalog)) != 72:
            raise ValueError("Every query must score the entire fresh 36-product/72-variant pool")
    return frozen, catalog, requests


def models_from_source(dataset_freeze, semantic_run):
    """Check the exact already-completed source without opening fresh labels."""
    semantic_run = Path(semantic_run)
    expected = dataset_freeze["sourceCheckpoints"]
    if str(semantic_run.resolve()) != expected["semanticRun"] or file_digest(semantic_run / "CHECKPOINTS_FROZEN.json") != expected["freezeSha256"]:
        raise ValueError("Source semantic model changed after fresh data freeze")
    report = json.loads((semantic_run / "report.json").read_text())
    metadata = report["metadata"]
    if report["status"] != "completed" or metadata["statisticsSha256"] != expected["statisticsSha256"]:
        raise ValueError("Completed immutable semantic run required")
    factors = require_semantic_frozen(semantic_run, metadata)
    for arm, checkpoint in expected["checkpoints"].items():
        if file_digest(semantic_run / checkpoint["file"]) != checkpoint["sha256"]:
            raise ValueError("Frozen semantic checkpoint identity mismatch")
    image = load_image(semantic_run / "centered-image.pt", metadata)
    if file_digest(semantic_run / "centered-image.pt") != expected["image"]["sha256"]:
        raise ValueError("Frozen image identity mismatch")
    return metadata, factors, image


def prepare(dataset, semantic_run, prior, model, tokenizer):
    """Deterministic public CPU token packet; no private data or backbone load."""
    dataset, model, prior = Path(dataset).resolve(), Path(model).resolve(), Path(prior)
    frozen, catalog, requests = public_inputs(dataset)
    metadata, _, _ = models_from_source(frozen, semantic_run)
    provenance = json.loads((prior / "model-verification.json").read_text())
    assert_model_files(model, provenance)
    previous = json.loads((prior / "preflight.json").read_text())
    if model.name != REVISION or str(model) != previous["modelPath"]:
        raise ValueError("Pinned local model path required")
    for name, key in (("tokenizer.json", "tokenizerSha256"), ("config.json", "configSha256"), ("chat_template.jinja", "chatTemplateSha256")):
        if file_digest(model / name) != previous[key]:
            raise ValueError("Pinned tokenizer/model config changed")
    texts = [{"role": "catalog", "member": [pid, v["id"]], "text": catalog_text(p, v)}
             for pid, p in catalog.items() for v in p["presentationVariants"]]
    texts += [{"role": "request", "member": rid, "text": request_text(q, catalog)} for rid, q in requests.items()]
    records = []
    for row in texts:
        ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
        if not 0 < len(ids) <= 512:
            raise ValueError("Fresh input exceeds fixed token cap; refusing truncation")
        records.append({k: v for k, v in row.items() if k != "text"} |
                       {"textSha256": digest(row["text"]), "tokenIds": ids, "tokenSha256": digest(canonical(ids)),
                        "tokens": len(ids), "batchShape": [1, len(ids)]})
    if len(records) != 96 or sum(r["role"] == "catalog" for r in records) != 72:
        raise ValueError("Fresh native work must be exactly 72 catalog +24 query forwards")
    packet = {"version": VERSION, "freezeSha256": file_digest(dataset / "FREEZE.json"),
              "sourceHashes": source_hashes(), "sourceSemanticReportSha256": file_digest(Path(semantic_run) / "report.json"),
              "sourceModels": frozen["sourceCheckpoints"], "metadata": metadata,
              "modelPath": str(model), "modelRevision": REVISION, "quantization": previous["quantization"],
              "tokenizerSha256": previous["tokenizerSha256"], "configSha256": previous["configSha256"],
              "chatTemplateSha256": previous["chatTemplateSha256"],
              "priorModelVerificationSha256": file_digest(prior / "model-verification.json"),
              "inputHashes": {k: v for k, v in frozen["files"].items() if not k.startswith("private/")},
              "limits": LIMITS, "fixedDenominators": DENOMINATORS, "nativeForwardsRequired": 96,
              "rawContentVersion": "SHA256 of exact new allowlisted catalog/request text; no old raw vectors reused",
              "tokenRange": {role: [min(r["tokens"] for r in records if r["role"] == role), max(r["tokens"] for r in records if r["role"] == role)] for role in ("catalog", "request")},
              "records": records}
    by_key = {(r["role"], canonical(r["member"])): {**r, "text": t["text"]} for r, t in zip(records, texts, strict=True)}
    if len(by_key) != 96:
        raise ValueError("Duplicate feature membership")
    return packet, catalog, requests, by_key, provenance


def opposite_preferences(requests):
    """Public-only counterpart mapping, independently checked against controls later."""
    result = {}
    for rid, q in requests.items():
        options = [other for other, r in requests.items() if other != rid
                   and r["context"]["task"] == q["context"]["task"]
                   and r["context"]["presentationPreferences"] == q["context"]["presentationPreferences"]
                   and r["context"]["statedBudget"] == q["context"]["statedBudget"]
                   and r["context"]["statedPreferences"] != q["context"]["statedPreferences"]]
        if len(options) != 1:
            raise ValueError("Unique public opposite-preference counterpart required")
        result[rid] = options[0]
    return result


def rotated(request, catalog, matrix, offset):
    order = request["candidateProductIds"]
    changed = {**request, "candidateProductIds": order[offset:] + order[:offset]}
    original = roster(request, catalog);index = {a: i for i, a in enumerate(original)}
    indices = [index[a] for a in roster(changed, catalog)]
    return changed, matrix[indices], indices


def copy_source_models(source, out, metadata, expected):
    if file_digest(Path(source) / "CHECKPOINTS_FROZEN.json") != expected["freezeSha256"]:
        raise ValueError("Reviewed source model freeze changed before copying")
    for name in ("fusion.pt", "prior.pt", "random.pt", "centered-image.pt", "CHECKPOINTS_FROZEN.json"):
        with (Path(source) / name).open("rb") as src, (Path(out) / name).open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if file_digest(Path(source) / name) != file_digest(Path(out) / name):
            raise ValueError("Copied model bytes changed")
    factors = require_semantic_frozen(out, metadata)
    if file_digest(Path(out) / "CHECKPOINTS_FROZEN.json") != expected["freezeSha256"]:
        raise ValueError("Copied checkpoint freeze differs from reviewed identity")
    image = load_image(Path(out) / "centered-image.pt", metadata)
    return factors, image


def extract(out, prepared, encoder, models, *, started):
    """Cold catalog first, then 24 actual fresh queries against verified warm cache."""
    packet, catalog, requests, rows, provenance = prepared
    store = FeatureStore(Path(out) / "features", provenance)
    observations, catalog_receipts, query_receipts = [], {}, {}
    query, features = {}, {}
    forwards = 0
    phase_started = time.perf_counter()
    def encode(row):
        nonlocal forwards
        if forwards >= 96 or time.perf_counter() - phase_started > LIMITS["nativeExtractionSeconds"]:
            raise TimeoutError("Fixed 96-forward/180-second extraction limit")
        if forwards < 2 and time.perf_counter() - phase_started + encoder.load_seconds > LIMITS["pilotSeconds"]:
            raise TimeoutError("Fixed native model-load/two-forward pilot limit")
        if store.exists(row):
            raise ValueError("Fresh native extraction cannot reuse old feature artifacts")
        values, timing = encoder.encode(row["text"], row)
        forwards += 1
        if (timing.get("backboneForwardCalls") != 1 or timing.get("batchRows") != 1
                or timing.get("paddedTokensPerRow") != row["tokens"] or timing.get("truncatedRows") != 0):
            raise ValueError("Expected one exact whole-input B1 forward")
        if time.perf_counter() - phase_started > LIMITS["nativeExtractionSeconds"]:
            raise TimeoutError("Fixed native extraction time exceeded")
        if forwards <= 2 and time.perf_counter() - phase_started + encoder.load_seconds > LIMITS["pilotSeconds"]:
            raise TimeoutError("Fixed native two-forward pilot time exceeded")
        return *store.save(row, values, timing), timing
    with (Path(out) / "native-observations.jsonl").open("x") as log:
        cold_start = time.perf_counter();cold_native = 0.
        for row in rows.values():
            if row["role"] != "catalog":continue
            values, receipt, timing = encode(row)
            cold_native += timing["encodeTotalMs"]
            catalog_receipts[canonical(row["member"])] = {"rawKey": receipt["rawKeys"][0], "artifactSha256": receipt["artifactSha256"], "sha256": receipt["sha256"]}
            log.write(json.dumps({"event": "cold-catalog-forward", "member": row["member"], "nativeForwardCalls": 1,
                                  "encodeTotalMs": timing["encodeTotalMs"], "memory": encoder.memory()}) + "\n");log.flush()
        cold_wall = (time.perf_counter() - cold_start) * 1000
        for rid, request in requests.items():
            start = time.perf_counter()
            row = {**rows["request", canonical(rid)], "text": request_text(request, catalog)}
            values, receipt, timing = encode(row)
            query_ms = (time.perf_counter() - start) * 1000
            query[rid] = values[0]
            query_receipts[rid] = {"rawKey": receipt["rawKeys"][0], "artifactSha256": receipt["artifactSha256"], "sha256": receipt["sha256"]}
            cache_start = time.perf_counter()
            matrix = torch.stack([store.load(rows["catalog", canonical(list(action))])[0][0] for action in roster(request, catalog)])
            cache_ms = (time.perf_counter() - cache_start) * 1000
            head_start = time.perf_counter()
            with torch.no_grad():scores = models["fusion"](query[rid], matrix)
            score_ms = (time.perf_counter() - head_start) * 1000
            grid_start = time.perf_counter()
            for count in CARD_COUNTS:assemble_grid(request, catalog, roster(request, catalog), scores, count)
            grid_ms = (time.perf_counter() - grid_start) * 1000
            observation = {"event": "fresh-query-warm-catalog", "recordId": rid, "catalogCacheCondition": "warm",
                           "catalogHits": 72, "nativeCatalogForwards": 0, "freshQueryForwards": 1,
                           "freshQueryNativeEncodeMs": timing["encodeTotalMs"], "freshQueryConstructEncodeSaveVerifyMs": query_ms,
                           "warmCatalogHashLoadVerifyMs": cache_ms, "full72VariantCpuScoreMs": score_ms,
                           "twoGridCountsMasksMappingMs": grid_ms, "observedFreshQueryWarmCatalogTwoGridsMs": (time.perf_counter() - start) * 1000,
                           "scope": "model resident; actual fresh query + verified72 catalog cache hits + frozen scorer +12/24 grid assembly; no copy",
                           "processRssBytes": psutil.Process().memory_info().rss, "memory": encoder.memory()}
            observations.append(observation);features[rid] = matrix
            log.write(json.dumps(observation) + "\n");log.flush()
            print(json.dumps(observation), flush=True)
    if forwards != 96 or len(query_receipts) != 24 or len(catalog_receipts) != 72:
        raise ValueError("Extraction did not meet the fixed native-forward denominator")
    return store, query, features, {"catalog": catalog_receipts, "query": query_receipts}, {
        "catalogColdNativeEncodeSumMs": cold_native, "catalogColdConstructEncodeSaveVerifyMs": cold_wall,
        "nativeForwards": forwards, "catalogForwards": 72, "queryForwards": 24,
        "freshQueryWarmCatalogObservations": observations, "memoryAfterExtraction": encoder.memory()}


def prediction_freeze(out, prepared, store, queries, features, models, feature_receipts):
    packet, catalog, requests, rows, _ = prepared
    permutation = opposite_preferences(requests)
    predictions = {}
    with torch.no_grad():
        for rid, request in requests.items():
            predictions[rid] = {}
            for arm in ("random", "prior", "fusion", "permuted-query"):
                source = permutation[rid] if arm == "permuted-query" else rid
                model = models["fusion" if arm == "permuted-query" else arm]
                scores = model(queries[source], features[rid])
                for offset in OFFSETS:
                    _, matrix, indices = rotated(request, catalog, features[rid], offset)
                    if not torch.equal(model(queries[source], matrix), scores[indices]):
                        raise AssertionError("Native-feature score mapping changed under roster rotation")
                predictions[rid][arm] = {"querySourceRecordId": source, "variantScores": scores}
    # Reload every feature artifact and copied model, then verify predictions
    # again before sealing; this uses no fresh private labels or atom targets.
    for row in rows.values():store.load(row)
    reloaded = require_semantic_frozen(out, packet["metadata"])
    image = load_image(Path(out) / "centered-image.pt", packet["metadata"])
    with torch.no_grad():
        for rid, arms in predictions.items():
            for arm, case in arms.items():
                scorer = SemanticScorer(reloaded["fusion" if arm == "permuted-query" else arm], image)
                if not torch.equal(scorer(queries[case["querySourceRecordId"]], features[rid]), case["variantScores"]):
                    raise AssertionError("Copied semantic/image prediction reload mismatch")
    path = Path(out) / "predictions.pt"
    with path.open("xb") as handle:torch.save(predictions, handle)
    freeze = {"version": VERSION, "frozenAtUtc": datetime.now(timezone.utc).isoformat(),
              "packetSha256": file_digest(Path(out) / "fresh-preflight.json"), "dataFreezeSha256": packet["freezeSha256"],
              "sourceHashes": packet["sourceHashes"], "modelFreezeSha256": file_digest(Path(out) / "CHECKPOINTS_FROZEN.json"),
              "predictionsSha256": file_digest(path), "features": feature_receipts,
              "privateTargetsOpenedBeforeFreeze": False, "fits": 0, "nativeForwards": 96,
              "exactModelPredictionReload": True, "allThreeRosterScoreMappingsExact": True}
    write_json(Path(out) / "FEATURES_PREDICTIONS_FROZEN.json", freeze)
    return predictions


def require_evaluation_freeze(out, dataset, prepared):
    packet, _, _, rows, provenance = prepared
    freeze = json.loads((Path(out) / "FEATURES_PREDICTIONS_FROZEN.json").read_text())
    if (freeze.get("version") != VERSION or freeze.get("nativeForwards") != 96 or freeze.get("fits") != 0
            or freeze.get("privateTargetsOpenedBeforeFreeze") is not False
            or freeze["dataFreezeSha256"] != file_digest(Path(dataset) / "FREEZE.json")
            or freeze["packetSha256"] != file_digest(Path(out) / "fresh-preflight.json")
            or freeze["predictionsSha256"] != file_digest(Path(out) / "predictions.pt")
            or freeze["modelFreezeSha256"] != file_digest(Path(out) / "CHECKPOINTS_FROZEN.json")
            or freeze["modelFreezeSha256"] != packet["sourceModels"]["freezeSha256"]
            or freeze["sourceHashes"] != source_hashes()):
        raise ValueError("Immutable full extraction/prediction freeze required before fresh labels")
    require_semantic_frozen(out, packet["metadata"])
    store = FeatureStore.open_existing(Path(out) / "features", provenance)
    for row in rows.values():
        _, receipt = store.load(row)
        expected = freeze["features"]["catalog" if row["role"] == "catalog" else "query"][canonical(row["member"]) if row["role"] == "catalog" else row["member"]]
        if {"rawKey": receipt["rawKeys"][0], "artifactSha256": receipt["artifactSha256"], "sha256": receipt["sha256"]} != expected:
            raise ValueError("Feature identity changed after extraction freeze")
    return freeze


def read_private_after_freeze(out, dataset, prepared):
    require_evaluation_freeze(out, dataset, prepared)
    verify_dataset(dataset, private=True)
    labels = {r["recordId"]: r for r in map(json.loads, (Path(dataset) / "private/truth.jsonl").read_text().splitlines())}
    controls = json.loads((Path(dataset) / "private/controls.json").read_text())
    if (set(labels) != set(prepared[2]) or controls["permutedQuerySource"] != opposite_preferences(prepared[2])
            or controls["mandatoryRosterRotationOffsets"] != list(OFFSETS) or controls["requestedCardCounts"] != list(CARD_COUNTS)):
        raise ValueError("Fixed private evaluation/control membership changed")
    return labels, controls


def evaluate(predictions, queries, features, factors, image, catalog, requests, labels, controls):
    cases = {arm: {} for arm in ("random", "prior", "fusion", "permuted-query", "public-tag-oracle")}
    grid_results = {arm: [] for arm in cases}
    def ndcg(ids, relevance, count):
        dcg = lambda values: sum((2 ** v - 1) / math.log2(i + 2) for i, v in enumerate(values))
        ideal = dcg(sorted(relevance.values(), reverse=True)[:count])
        return dcg([relevance[pid] for pid in ids[:count]]) / ideal if ideal else 1.
    for rid, request in requests.items():
        actions = roster(request, catalog)
        for arm in cases:
            if arm == "public-tag-oracle":
                with torch.no_grad():scores = image(queries[rid], features[rid]).clone()
                scores[:, 0] = torch.tensor([len(set(request["context"]["statedPreferences"]) & set(catalog[pid]["tags"])) for pid, _ in actions])
            else:scores = predictions[rid][arm]["variantScores"]
            cases[arm][rid] = {mode: readout(scores, request, catalog, labels[rid], guarded=mode == "guarded") for mode in ("raw", "guarded")}
            baseline = None
            for offset in OFFSETS:
                changed, _, indices = rotated(request, catalog, features[rid], offset)
                for count in CARD_COUNTS:
                    grid = assemble_grid(changed, catalog, roster(changed, catalog), scores[indices], count)
                    ids = [c["productId"] for c in grid["cards"]]
                    identity = [(p["productId"], p["variantId"]) for p in grid["rankedProducts"]]
                    if baseline is None:baseline = identity
                    if identity != baseline:raise AssertionError("ID-mapped full grid changed under roster rotation")
                    mask_errors = sum(bool(guard_reasons(changed, catalog[c["productId"]])) for c in grid["cards"])
                    incompatible = sum(not changed["presentationCompatibility"][c["productId"]].get(c["variantId"], False) for c in grid["cards"])
                    grid_results[arm].append({"recordId": rid, "rosterRotation": offset, "requestedCards": count,
                        "returnedCards": len(ids), "fullCandidateProductsRanked": len(grid["rankedProducts"]),
                        "fullCandidateVariantsScored": len(actions), "shortfall": grid["shortfall"],
                        "maskViolations": mask_errors, "incompatibleSelectedImages": incompatible,
                        "fullGridComponentValid": grid["fullRequestedCardFulfillment"] and not mask_errors and not incompatible and len(ids) == len(set(ids)),
                        "ndcgAtRequestedCount": ndcg(ids, labels[rid]["relevance"], count), "cards": grid["cards"],
                        "selectedImagesTeacherCorrect": sum(labels[rid]["presentationScores"][c["productId"]][c["variantId"]] == 3 for c in grid["cards"])})
    evaluation = {"cases": cases}
    paired_controls = {"pairs": [{**p, "split": "fresh"} for p in controls["pairs"]]}
    pair_summary = pair_metrics(evaluation, paired_controls, labels, "fresh")
    quality = {}
    for arm, arm_cases in cases.items():
        quality[arm] = {"fixedRequests": 24, "gridCases": 144,
                        "validGridComponents": sum(g["fullGridComponentValid"] for g in grid_results[arm]),
                        "meanNdcg12": sum(g["ndcgAtRequestedCount"] for g in grid_results[arm] if g["requestedCards"] == 12) / 72,
                        "meanNdcg24": sum(g["ndcgAtRequestedCount"] for g in grid_results[arm] if g["requestedCards"] == 24) / 72}
        for mode in ("raw", "guarded"):
            quality[arm][mode] = {"productCorrect": sum(c[mode]["metrics"]["bestProductAccuracy"] for c in arm_cases.values()),
                                  "requests": 24, "allCandidateImageCorrect": sum(c[mode]["metrics"]["allCandidatePresentation"]["informativeCorrect"] for c in arm_cases.values()),
                                  "allCandidateImageTotal": 864}
    diagnostics = factor_diagnostics(factors, queries, {a: features[next(iter(requests))][i] for i, a in enumerate(roster(next(iter(requests.values())), catalog))}, requests, catalog)
    # Unlike the historical generic helper's cyclic control, the sealed fresh
    # permutation uses the paired opposing preference; compute its atom block.
    from .semantic_factors import atom_metrics
    from .paired_data import ATOMS
    ids = sorted(requests)
    with torch.no_grad():
        qp, _ = factors["fusion"].probabilities(torch.stack([queries[controls["permutedQuerySource"][rid]] for rid in ids]), features[ids[0]])
    target = torch.tensor([[float(a in requests[rid]["context"]["statedPreferences"]) for a in ATOMS] for rid in ids])
    diagnostics["permuted-query"]["query"] = atom_metrics(qp, target)
    requirement_probes = []
    for probe in controls["requirementGuardProbes"]:
        request = requests[probe["sourceRecordId"]]
        changed = {**request, "context": {**request["context"], "statedRequirements": probe["statedRequirements"]}}
        grid = assemble_grid(changed, catalog, roster(changed, catalog), torch.zeros(72, 2), 24)
        if len(grid["cards"]) != probe["expectedEligible"] or grid["shortfall"] != 24 - probe["expectedEligible"]:
            raise AssertionError("Declared requirement shortfall guard probe failed")
        requirement_probes.append({**probe, "actualShortfall": grid["shortfall"], "passed": True})
    return {"quality": quality, "counterfactualPairs": pair_summary, "atomDiagnostics": diagnostics,
            "cases": cases, "grids": grid_results, "requirementGuardProbes": requirement_probes}


def run_prepared(out, dataset, semantic_run, prepared, encoder_factory, *, started=None):
    """Test hook permits an explicitly marked CPU stub; normal CLI is native only."""
    out, dataset = Path(out), Path(dataset).resolve()
    started = time.perf_counter() if started is None else started
    packet, catalog, requests, _, provenance = prepared
    factors, image = copy_source_models(semantic_run, out, packet["metadata"], packet["sourceModels"])
    models = {a: SemanticScorer(f, image) for a, f in factors.items()}
    write_json(out / "config.json", {"execution": execution_receipt(), "packet": packet, "fits": 0, "generatesCopy": False,
                                     "evidenceSource": provenance["evidenceSource"]})
    load_start = time.perf_counter();encoder = encoder_factory()
    try:
        if encoder.evidence_source != provenance["evidenceSource"]:
            raise ValueError("Encoder/provenance evidence source mismatch")
        store, q, x, receipts, extraction = extract(out, prepared, encoder, models, started=started)
        extraction["modelLoadSeconds"] = encoder.load_seconds
        extraction["modelResidentExtractionPhaseSeconds"] = time.perf_counter() - load_start
    finally:encoder.close()
    if source_hashes() != packet["sourceHashes"]:raise ValueError("Fresh evaluator source changed during extraction")
    predictions = prediction_freeze(out, prepared, store, q, x, models, receipts)
    labels, controls = read_private_after_freeze(out, dataset, prepared)
    result = evaluate(predictions, q, x, factors, image, catalog, requests, labels, controls)
    write_json(out / "evaluation.json", result)
    require_evaluation_freeze(out, dataset, prepared)
    report = {"status": "completed", "version": VERSION, "fixedDenominators": DENOMINATORS,
              "execution": json.loads((out / "config.json").read_text())["execution"],
              "packetSha256": file_digest(out / "fresh-preflight.json"), "freezeSha256": packet["freezeSha256"],
              "sourceModels": packet["sourceModels"], "evidenceSource": provenance["evidenceSource"],
              "fits": 0, "meansUpdated": False, "imageCheckpointBytesUnchanged": True,
              "extraction": extraction, "quality": result["quality"], "counterfactualPairs": result["counterfactualPairs"],
              "atomDiagnostics": result["atomDiagnostics"], "wallSeconds": time.perf_counter() - started,
              "exactModelPredictionReload": True, "allRosterRotationsScoreAndGridInvariant": True,
              "privateTargetsOpenedAfterFeaturePredictionFreeze": True, "generatesCopy": False,
              "completeViewEvidence": {"aboveFoldThreeCards": "not evaluated", "fullStorefront": "not evaluated"},
              "limitations": ["Fresh synthetic catalog/query seven-atom evaluation; no latent human preference/conversion claim.",
                              "Image head is inherited unchanged; no new image/copy/grouping fit.",
                              "Grid component fulfillment is not complete-page quality.",
                              "Warm fresh-query latency includes actual new native request encode; cold catalog/model load and full wall are separately reported."]}
    write_json(out / "report.json", report)
    return report


def failure_report(out, error, started):
    path = Path(out) / "native-observations.jsonl"
    observations = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    result = {"status": "failed", "exception": type(error).__name__, "message": str(error),
              "fixedDenominators": DENOMINATORS, "successfulCompletedEvaluationRequests": 0,
              "successfulFullGridComponents": 0, "unfulfilledGridCasesPerArm": 144,
              "recordedCatalogForwards": sum(r["event"] == "cold-catalog-forward" for r in observations),
              "recordedFreshQueryForwards": sum(r["event"] == "fresh-query-warm-catalog" for r in observations),
              "note": "Recorded completed observations may undercount an interrupted failed forward; no partial denominator is promoted.",
              "wallSeconds": time.perf_counter() - started, "partialArtifactsPreserved": True, "generatesCopy": False}
    write_json(Path(out) / "FAILED.json", result)
    return result


def main():
    from tokenizers import Tokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("experiments/ranker/pretrained/catalog_fusion/fresh_eval_v1"))
    parser.add_argument("--semantic-run", type=Path, default=Path("data/ranker/pretrained/catalog-user-semantic-factors-v1-attempt-1"))
    parser.add_argument("--prior", type=Path, default=Path("data/ranker/pretrained/catalog-user-fusion-v1-attempt-1"))
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--reviewed-packet-sha256")
    parser.add_argument("--reviewed-freeze-sha256")
    args = parser.parse_args();started = time.perf_counter();torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        prepared = prepare(args.dataset, args.semantic_run, args.prior, args.model, Tokenizer.from_file(str(args.model / "tokenizer.json")))
        packet = prepared[0]
        if not args.preflight_only:
            if args.reviewed_freeze_sha256 != packet["freezeSha256"] or args.packet is None or file_digest(args.packet) != args.reviewed_packet_sha256:
                raise ValueError("Reviewed data freeze and exact source/token packet checksums required")
            if json.loads(args.packet.read_text()) != packet:
                raise ValueError("Fresh evaluator reviewed packet no longer matches")
        write_json(args.out / "fresh-preflight.json", packet)
        if args.preflight_only:
            print(json.dumps({"status": "CPU packet only; no native execution", "packetSha256": file_digest(args.out / "fresh-preflight.json"),
                              "tokenRange": packet["tokenRange"], "nativeForwardsRequired": 96}, indent=2));return
        # Full pinned shard hashing is repeated before native load. No downloads.
        verified = verify_model(args.model, packet)
        if verified["weightsSha256"] != prepared[4]["weightsSha256"] or verified["tokenizerSha256"] != prepared[4]["tokenizerSha256"]:
            raise ValueError("Fresh full-model verification differs from original pinned receipt")
        write_json(args.out / "model-verification.json", verified)
        def factory():
            assert_model_files(args.model, verified)
            return NativeEncoder(args.model)
        run_prepared(args.out, args.dataset, args.semantic_run, prepared, factory, started=started)
    except BaseException as error:
        failure_report(args.out, error, started);raise


if __name__ == "__main__":main()
