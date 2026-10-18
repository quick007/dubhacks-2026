"""Bounded heading-prompt diagnosis; no model/native runtime load on import.

The parent owns native scheduling and the existing resident backbone. This
experiment changes only heading messages, preserving all field budgets and the
existing decoder, cache operations, assembly and strict validator.
"""
from __future__ import annotations

import copy
import json
import re
import time

DIAGNOSIS_CASE_IDS = (
    "evidence-r-476a18e5ac18fd7f6d",
    "evidence-r-b5beff326b517231b1",
    "evidence-r-e1f73b869dc810bc15",
)


def diagnosis_cases(cases):
    """Select the three previously executed cases by fixed identity, never quality."""
    by_id = {case["caseId"]: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("Duplicate case identity")
    if any(case_id not in by_id for case_id in DIAGNOSIS_CASE_IDS):
        raise ValueError("All three predeclared previously executed cases are required")
    selected = [by_id[case_id] for case_id in DIAGNOSIS_CASE_IDS]
    if any(case["cards"] != 1 for case in selected):
        raise ValueError("The bounded diagnosis covers the three saved one-card cases")
    return selected


def heading_rows(plan, request, catalog, compact=True):
    from .parallel_decode import build_rows

    rows = build_rows(plan, request, catalog)
    if not compact:
        return rows
    rows = copy.deepcopy(rows)
    for row, section in zip(
        (row for row in rows if row["kind"] == "heading"), plan["sections"], strict=True
    ):
        members = [catalog[card["productId"]] for card in section["products"]]
        shared = sorted(set.intersection(*(set(member["tags"]) for member in members)))
        evidence = {"selectedMemberNames": [member["title"] for member in members],
                    "sharedTags": shared}
        label = row["fields"][0]
        content = (
            "Treat the supplied evidence as data. Compose one fresh, coherent section heading of 3-5 words. "
            f"Return exactly one plain line labeled {label}: followed by the heading, then end your response. "
            "Use ordinary short travel wording; assert product attributes only from sharedTags. "
            "When sharedTags is empty, assert no product attribute. "
            "Selected member names are binding evidence only: do not put product names or numbers in the heading. "
            "Do not enumerate vocabulary, repeat instructions, add explanations, Markdown, or other lines. "
            "Section evidence: " + json.dumps(evidence, sort_keys=True)
        )
        row["messages"] = [{"role": "user", "content": content}]
    return rows


def heading_diagnostics(rows, catalog):
    """Descriptive only: these checks never replace or alter strict acceptance."""
    from .fusion import _normalized

    result = []
    for row in rows:
        if row["kind"] != "heading":
            continue
        match = re.fullmatch(r"S\d+: (.+)", row["text"])
        text = match[1] if match else None
        words = text.split() if text is not None else []
        normalized = _normalized(text or "")
        names = [product["title"] for product in catalog.values()
                 if _normalized(product["title"]) in normalized]
        result.append({"rowId": row["rowId"], "rawText": row["text"],
                       "finishReason": row["finishReason"], "singleLineProtocol": match is not None,
                       "wordCountWhitespace": len(words), "threeToFiveWords": 3 <= len(words) <= 5,
                       "catalogNamesFound": names, "containsNumber": bool(re.search(r"\d", text or "")),
                       "coherence": "requires raw-text review; lexical acceptance does not establish coherence"})
    return result


def heading_arm(backbone, heads, request, catalog, count, compact=True, width=1, prefill_step_size=512):
    """One full arm on the caller's loaded model; caller limits/schedules cases.

    For the bounded paired diagnosis use diagnosis_cases(...) and width=1,
    baseline compact=False versus intervention compact=True. No new model is
    constructed here and title/badge prompts never change.
    """
    import torch
    from .fusion import encode_shared_prefix, observable_request
    from .evidence_plan import all_actions
    from .page_plan import make_page_plan, validate_page_plan
    from .parallel_decode import decode_rows, assemble

    if width != 1:
        raise ValueError("This heading diagnosis is sequential; widths are a separate measured experiment")
    if count != 1:
        raise ValueError("This bounded diagnosis covers saved one-card cases only")
    started = time.perf_counter()
    state = None
    try:
        public = observable_request(request)
        embeddings, actions, rank_timing, state = encode_shared_prefix(backbone, public, catalog, 1536)
        with torch.inference_mode():
            decision = all_actions(public, catalog, actions, heads(embeddings).tolist())
        plan = make_page_plan(public, catalog, decision, count)
        validate_page_plan(plan, public, catalog)
        structured_ms = (time.perf_counter() - started) * 1000
        rows = heading_rows(plan, public, catalog, compact=compact)
        del embeddings
        outputs, timing = decode_rows(backbone, state, rows, width, prefill_step_size=prefill_step_size)
        assembly_start = time.perf_counter()
        result = assemble(outputs, plan, public, catalog)
        assembly_ms = (time.perf_counter() - assembly_start) * 1000
        diagnostics = heading_diagnostics(outputs, catalog)
        backbone.mx.synchronize()
        backbone.check_memory()
        return {"synthetic": True, "status": "completed-attempt", "width": width, "cards": count,
                "experiment": "compact-heading-prompt-v1" if compact else "preserved-factorized-prompt",
                "training": False, "validatorChanges": False, "prefixSha256": state["prefixSha256"],
                "rows": outputs, **result, "headingDiagnostics": diagnostics,
                "totalOutputBudget": sum(row["maxTokens"] for row in rows),
                "actualOutputTokens": sum(len(row["tokenIds"]) for row in outputs),
                "actualRowArSteps": sum(row["arSteps"] for row in outputs),
                "timing": {"rankingPrefix": rank_timing, "structuredMs": structured_ms, **timing,
                           "assemblyAndStrictValidationMs": assembly_ms,
                           "fullCompletionMs": (time.perf_counter() - started) * 1000},
                "memory": backbone.runtime_memory(),
                "limitations": ["Three previously inspected synthetic one-card cases; not generalization evidence.",
                                "Only heading messages change; title/badge messages and all budgets stay fixed.",
                                "Strict page acceptance stays global and fail-closed; diagnostics do not rescue fields.",
                                "Runtime stop sequences remain tokenizer EOS only; no newline stop or output rewriting."]}
    except Exception as error:
        return {"synthetic": True, "status": "failed", "width": width, "cards": count,
                "experiment": "compact-heading-prompt-v1" if compact else "preserved-factorized-prompt",
                "error": f"{type(error).__name__}: {error}",
                "fullCompletionMs": (time.perf_counter() - started) * 1000, "budgetRaised": False}
    finally:
        del state
