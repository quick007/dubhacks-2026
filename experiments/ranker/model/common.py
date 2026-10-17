"""Observable-only feature boundary, deterministic baseline, constraints and metrics."""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def product_facts(product: dict) -> dict:
    # Explicit whitelist prevents labels, IDs and extra catalog metadata becoming features.
    return {key: product.get(key) for key in
            ("title", "category", "description", "price", "shippingCost", "inStock", "tags")}


def observable_input(request: dict, catalog: dict[str, dict], *, history: bool = True) -> dict:
    context = request.get("context", {})
    result = {
        "context": {key: context.get(key) for key in
                    ("task", "statedBudget", "statedRequirements", "statedPreferences")},
        "candidates": [product_facts(catalog[pid]) for pid in request["candidateProductIds"]],
        "history": [],
    }
    if history:
        for event in request.get("history", []):
            # ISO8601 timestamps from the contract have normalized UTC representation.
            if event.get("at", "~") > request["asOf"] or event.get("productId") not in catalog:
                continue
            result["history"].append({
                "type": event.get("type"), "at": event.get("at"),
                "dwellSeconds": event.get("dwellSeconds"),
                "source": event.get("source"), "confidence": event.get("confidence"),
                "product": product_facts(catalog[event["productId"]]),
            })
    return result


def known_violations(context: dict, product: dict) -> list[str]:
    reasons = []
    if not product.get("inStock", False):
        reasons.append("out_of_stock")
    budget = context.get("statedBudget")
    if budget is not None and product["price"] + product.get("shippingCost", 0) > budget:
        reasons.append("stated_budget")
    tags = {str(tag).lower() for tag in product.get("tags", [])}
    for requirement in context.get("statedRequirements") or []:
        required = str(requirement).lower()
        # Fixture requirements are exact factual tags, not semantic inference.
        if required not in tags:
            reasons.append("requirement:" + required)
    return reasons


def heuristic_scores(request: dict, catalog: dict[str, dict], *, history: bool = True) -> list[float]:
    observed = observable_input(request, catalog, history=history)
    context = observed["context"]
    preference = tokens(" ".join(context.get("statedPreferences") or []))
    task = tokens(context.get("task") or "")
    history_weights: dict[str, float] = {}
    for event in observed["history"]:
        polarity = {"purchase": 2.0, "add_to_cart": 1.3, "view": .25, "click": .4, "return": -1.5}.get(event["type"], .1)
        weight = polarity * float(event.get("confidence") or 0)
        for tag in event["product"].get("tags") or []:
            history_weights[tag] = history_weights.get(tag, 0) + weight
    scores = []
    for product in observed["candidates"]:
        facts = tokens(" ".join(str(product.get(k, "")) for k in ("title", "category", "description")))
        tags = set(product.get("tags") or [])
        score = .6 * len(task & facts) + 1.2 * len(preference & (facts | tags))
        score += sum(history_weights.get(tag, 0) for tag in tags)
        score -= .002 * (product["price"] + product.get("shippingCost", 0))
        scores.append(score)
    return scores


def rank_result(request: dict, catalog: dict[str, dict], scores: list[float], model_id: str,
                *, guarded: bool, latency_ms: float = 0, warnings: list[str] | None = None) -> dict:
    ids = request["candidateProductIds"]
    ordered = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
    suitable = [i for i in ordered if not known_violations(request["context"], catalog[ids[i]])]
    selected = suitable[0] if guarded and suitable else (None if guarded else ordered[0])
    ranking = suitable + [i for i in ordered if i not in suitable] if guarded else ordered
    return {"schemaVersion": 1, "synthetic": True, "recordId": request["recordId"],
            "modelId": model_id, "rankedProducts": [{"productId": ids[i], "score": float(scores[i])} for i in ranking],
            "selectedProductId": ids[selected] if selected is not None else None,
            "presentationAction": None, "timing": {"inferenceMs": latency_ms},
            "warnings": ["Synthetic model and evaluation; no evidence of human conversion lift.",
                         *(["Deterministic known-constraint mask applied; this is not learned behavior."] if guarded else []),
                         *(warnings or [])]}


def ndcg(ranking: list[str], relevance: dict[str, int], k: int = 3) -> float:
    def dcg(values):
        return sum((2 ** value - 1) / math.log2(i + 2) for i, value in enumerate(values))
    denominator = dcg(sorted(relevance.values(), reverse=True)[:k])
    return dcg([relevance.get(pid, 0) for pid in ranking[:k]]) / denominator if denominator else 1.0


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def evaluate_results(requests: list[dict], results: list[dict], truth: dict[str, dict], catalog: dict[str, dict]) -> dict:
    ndcgs, hits, regrets, violations, private_violations, abstains, latencies = [], [], [], [], [], [], []
    for request, result in zip(requests, results, strict=True):
        target = truth[request["recordId"]]
        relevance = target["relevance"]
        selected = result["selectedProductId"]
        ndcgs.append(ndcg([item["productId"] for item in result["rankedProducts"]], relevance))
        best = max(relevance.values(), default=0)
        value = relevance.get(selected, 0)
        hits.append(int(selected is not None and value == best and best > 0))
        regrets.append(best - value)
        violations.append(int(selected is not None and bool(known_violations(request["context"], catalog[selected]))))
        hidden = target.get("hiddenConstraints", {})
        private_bad = False
        if selected is not None:
            product = catalog[selected]
            tags = set(product.get("tags", []))
            private_bad = (product["price"] + product.get("shippingCost", 0) > hidden.get("totalBudget", float("inf"))
                           or not set(hidden.get("requiredTags", [])).issubset(tags)
                           or bool(set(hidden.get("avoidedTags", [])) & tags)
                           or (hidden.get("requiresInStock", False) and not product["inStock"]))
        private_violations.append(int(private_bad))
        abstains.append(int(selected is None))
        latencies.append(result["timing"]["inferenceMs"])
    mean = lambda values: sum(values) / len(values) if values else 0
    return {"synthetic": True, "count": len(results), "ndcgAt3": mean(ndcgs),
            "topBestRelevantRate": mean(hits), "meanRelevanceRegret": mean(regrets),
            "knownConstraintViolationRate": mean(violations), "privateConstraintViolationRate": mean(private_violations),
            "abstentionRate": mean(abstains),
            "warmLatencyMs": {"p50": percentile(latencies, .5), "p95": percentile(latencies, .95)},
            "limitations": ["Synthetic hidden-preference labels are not real-user behavior.",
                            "Missing private preferences cannot be recovered reliably from sparse observations."]}
