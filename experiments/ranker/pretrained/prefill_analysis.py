"""Independent CPU/tokenizer-only exact-prefix experiment and saved stage accounting.

Never constructs a model, imports MLX, performs decoding, trains, or changes core files.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def hashed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def lcp(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def exact_append(tokenizer, tokens, prefix_messages, suffix_messages):
    from .fusion import append_generation_suffix
    opaque = object()
    state = {"model": opaque, "consumed": False, "tokens": tokens, "messages": prefix_messages, "cache": []}
    suffix, _ = append_generation_suffix(tokenizer, suffix_messages, state, opaque)
    return {"exactBoundaryPassed": True, "consumedOnce": state["consumed"],
            "prefixTokens": len(tokens), "suffixTokens": len(suffix),
            "fullTokens": len(tokens) + len(suffix), "suffixSha256": hashed(suffix)}


def stage_row(row, arm):
    value = row["arms"][arm]
    timing, generation = value["timing"], value["generation"]
    rank = timing["rankingPrefix"]["prefixTokens"]
    catalog = value["cache"]["prefixTokens"] if value.get("cache") else 0
    hit = bool((value.get("cache") or {}).get("hit"))
    suffix = generation["promptTokens"]
    result = {"caseId": row["caseId"], "source": row["source"], "cards": row["cards"], "arm": arm,
        "catalogHit": hit, "rankerTokens": rank, "separateCatalogTokens": catalog,
        "suffixTokens": suffix, "logicalInputWorkTokens": rank + catalog + suffix,
        "freshPrefillTokens": rank + suffix + (catalog if not hit else 0),
        "reusedCatalogTokens": catalog if hit else 0, "generatedTokens": generation["generatedTokens"],
        "validPage": value["quality"]["allFieldsFulfilled"],
        "stagesMs": {key: timing[key] for key in ("structuredMs", "catalogCachePrepareMs", "promptPreparationMs",
            "prefillAndFirstTokenMs", "autoregressiveAfterFirstTokenMs", "validationMs", "fullCompletionMs")}}
    accounted = sum(timing[k] for k in ("structuredMs", "catalogCachePrepareMs", "promptPreparationMs", "generationMs", "validationMs"))
    result["stagesMs"].update(accountedNonoverlappingMs=accounted,
        residualChecksAndAuditMs=timing["fullCompletionMs"] - accounted,
        nativePromptMsOverlappingFirstToken=timing["nativePromptMs"])
    return result


def run(args):
    from transformers import AutoTokenizer
    from .catalog_cache import catalog_prefix, request_prompt
    from .evidence_plan import prompt_for
    from .fusion import build_shared_prefix
    from .generation_diagnosis import cases_and_saved, manifest_prompt
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    cases = cases_and_saved(args.dataset)
    if len(cases) != 22:
        raise ValueError("Expected all 22 previously fixed cases; no outcome selection")
    records = []
    for case in cases:
        request, catalog, plan = case["request"], case["catalog"], case["saved"]["pagePlan"]
        rank_tokens, _, _, rank_messages = build_shared_prefix(tokenizer, request, catalog, 1536)
        catalog_tokens, catalog_messages, catalog_key = catalog_prefix(tokenizer, request, catalog)
        variants = {"rankBaseline": (rank_tokens, rank_messages, prompt_for(plan, request, catalog, True)),
                    "rankManifest": (rank_tokens, rank_messages, manifest_prompt(plan, request, catalog, False)),
                    "catalogBaseline": (catalog_tokens, catalog_messages, request_prompt(plan, request, catalog)),
                    "catalogManifest": (catalog_tokens, catalog_messages, manifest_prompt(plan, request, catalog, True))}
        append_checks = {name: exact_append(tokenizer, *arguments) for name, arguments in variants.items()}
        profile = copy.deepcopy(request)
        profile["context"].update(task="Explore durable travel bags.", statedPreferences=["durable"])
        profile["history"] = []
        _changed_tokens, _, changed_profile_key = catalog_prefix(tokenizer, profile, catalog)
        changed_catalog = copy.deepcopy(catalog)
        pid = request["candidateProductIds"][0]
        changed_catalog[pid]["inStock"] = not changed_catalog[pid]["inStock"]
        _, _, changed_catalog_key = catalog_prefix(tokenizer, request, changed_catalog)
        common = lcp(rank_tokens, catalog_tokens)
        records.append({"caseId": case["caseId"], "source": case["source"], "cards": case["cards"],
            "rankPrefixSha256": hashed(rank_tokens), "catalogPrefixSha256": hashed(catalog_tokens),
            "rankPrefixTokens": len(rank_tokens), "catalogPrefixTokens": len(catalog_tokens),
            "exactCommonLeadingTokens": common,
            "rankIsCatalogPrefix": common == len(rank_tokens),
            "catalogIsRankPrefix": common == len(catalog_tokens),
            "catalogKey": catalog_key, "profileChangeCatalogKeyUnchanged": catalog_key == changed_profile_key,
            "catalogAvailabilityChangeInvalidates": catalog_key != changed_catalog_key,
            "appendChecks": append_checks})
    if any(r["rankIsCatalogPrefix"] or r["catalogIsRankPrefix"] for r in records):
        raise AssertionError("Unexpected entire-prefix equivalence; inspect before making reuse claims")
    if not all(r["profileChangeCatalogKeyUnchanged"] and r["catalogAvailabilityChangeInvalidates"] for r in records):
        raise AssertionError("Catalog/profile invalidation boundary failed")
    saved = json.loads(args.measurements.read_text())
    stage_rows = [stage_row(row, arm) for row in saved["records"]
        if (row["source"] == "cache" and any(c in row["caseId"] for c in ("cold-catalog", "warm-catalog")))
        or (row["source"] == "evidence" and "page-scale-" in row["caseId"])
        for arm in ("baseline", "manifest")]
    report = {"schemaVersion": 1, "synthetic": True, "recordedAtUtc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Independent tokenizer-only exact-prefix and immutable measured-stage analysis",
        "modelWeightsLoaded": False, "gpuUsed": False, "generationRepeated": False,
        "caseCount": len(records), "appendBoundaryChecks": sum(len(r["appendChecks"]) for r in records),
        "allChecksPassed": True, "lcpTokensMin": min(r["exactCommonLeadingTokens"] for r in records),
        "lcpTokensMax": max(r["exactCommonLeadingTokens"] for r in records),
        "measurementsSource": {"path": str(args.measurements), "sha256": hashlib.sha256(args.measurements.read_bytes()).hexdigest()},
        "tokenizerSource": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.model.iterdir()
            if p.name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")},
        "records": records, "measuredPairedStages": stage_rows,
        "recommendation": "Proposal only for catalog-to-ranker context-layout substitution: not a semantically equivalent cache optimization. Existing evidence path already retains and appends exact user-first ranker state once.",
        "limitations": ["Token tests establish exact boundary identity, not hidden-state/logit/generation equality for a changed prompt.",
            "Qwen3.5 hybrid recurrence/convolution/KV state is conditioned on the complete ordered prefix; independently computed states cannot be spliced or substituted.",
            "Common leading chat-template tokens are not interchangeable completed-prefix states; no splice/trim/deepcopy experiment is performed.",
            "Saved timings are single measured attempts, not new performance measurements or percentile estimates. No p95 subtraction.",
            "nativePromptMs is a subcomponent overlapping prefillAndFirstTokenMs; generationMs already contains first-token and remaining decoding.",
            "Warm catalog reuse saves only its measured prepare work, not the unchanged approximately 1.94-second ranker forward.",
            "Output lengths/quality differ between prompt arms; timing differences cannot be attributed solely to cache reuse.",
            "Synthetic local evidence, no production latency, human conversion, revenue or architecture novelty claim."]}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("caseCount", "appendBoundaryChecks", "allChecksPassed", "lcpTokensMin", "lcpTokensMax")}, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/ranker/datasets/v2"))
    parser.add_argument("--measurements", type=Path, default=Path("data/ranker/pretrained/generation-diagnosis-v1/benchmark-reviewed.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/ranker/pretrained/PREFILL_ANALYSIS.json"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
