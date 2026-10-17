"""Synthetic NONE/SELECT/GENERATE routing from cached shared-prefix readouts.

Import and training are CPU-only. Only benchmark constructs the single MLX model.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import torch
from .boundary import decisions, guard_reasons, input_rows
from .fusion import (encode_shared_prefix, generation_messages, load_trained_heads,
                     native_decode, observable_request, supported_preferences, validate_copy)

ROUTES = ("NONE", "SELECT", "GENERATE")
CONFIG = {"seed": 37, "epochsPerStage": 80, "learningRate": .03, "weightDecay": .001,
          "representation": "shared-prefix-v1", "teacher": "0/1/2+ explicitly supported tags -> NONE/SELECT/GENERATE"}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def grounding(request, product):
    """Exact public preference/requirement -> exact catalog tag; private fields ignored."""
    tags = sorted(set(supported_preferences(request, product)))
    return [{"observablePreference": tag, "productFact": tag,
             "preferenceSource": "context.statedPreferences/statedRequirements",
             "factSource": "catalog.tags"} for tag in tags]


def teacher_route(request, product):
    return ROUTES[min(2, len(grounding(request, product)))]


class RouteHead(torch.nn.Module):
    def __init__(self, width=4096):
        super().__init__()
        torch.manual_seed(CONFIG["seed"])
        self.linear = torch.nn.Linear(width, 3)

    def forward(self, embeddings):
        return self.linear(embeddings.float())


def fit(head, x, targets):
    if not torch.isfinite(x).all():
        raise ValueError("Finite cached readouts required")
    before = head(x).detach().clone()
    initial = torch.nn.functional.cross_entropy(before, targets).item()
    optimizer = torch.optim.AdamW(head.parameters(), lr=CONFIG["learningRate"], weight_decay=CONFIG["weightDecay"])
    gradient = None
    for _ in range(CONFIG["epochsPerStage"]):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(x), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite route loss")
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in head.parameters()):
            raise FloatingPointError("Non-finite route gradient")
        if gradient is None:
            gradient = sum(p.grad.square().sum().item() for p in head.parameters()) ** .5
        optimizer.step()
    after = head(x).detach()
    change = (after - before).abs().mean().item()
    if not gradient or not change or not torch.isfinite(after).all():
        raise FloatingPointError("No finite real routing update")
    return {"epochs": CONFIG["epochsPerStage"], "rows": len(x), "crossEntropyBefore": initial,
            "crossEntropyAfter": torch.nn.functional.cross_entropy(after, targets).item(),
            "firstGradientL2": gradient, "meanAbsolutePredictionChange": change}


def save_head(head, directory, revision):
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / "route.pt"
    torch.save(head.state_dict(), file)
    metadata = {"schemaVersion": 1, "synthetic": True, "width": head.linear.in_features,
                "modelRevision": revision, "representation": CONFIG["representation"],
                "sha256": digest(file), "routes": ROUTES, "config": CONFIG}
    dump(directory / "metadata.json", metadata)
    restored = load_head(directory, revision)
    if any(not torch.equal(value, restored.state_dict()[key]) for key, value in head.state_dict().items()):
        raise AssertionError("Route checkpoint reload differs")
    return {**metadata, "exactWeightReload": True}


def load_head(directory, revision):
    meta = json.loads((directory / "metadata.json").read_text())
    if meta["modelRevision"] != revision or meta["representation"] != CONFIG["representation"] or digest(directory / "route.pt") != meta["sha256"]:
        raise ValueError("Route revision/representation/hash mismatch")
    head = RouteHead(meta["width"])
    head.load_state_dict(torch.load(directory / "route.pt", map_location="cpu", weights_only=True))
    head.eval()
    return head


def train(args):
    torch.set_num_threads(1)
    source = args.heads.parent
    original = json.loads((source / "run-manifest.json").read_text())
    if original["representation"] != "shared-prefix-v1" or original["modelRevision"] != args.revision:
        raise ValueError("Actual shared-prefix cached backbone revision required")
    catalog = {p["id"]: p for p in json.loads((args.dataset / "catalog.json").read_text())["products"]}
    requests = {r["recordId"]: r for r in lines(args.dataset / "public/train.jsonl")}
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    stages = manifest["trainStageRecordIds"]
    if set(stages["initial"]) & set(stages["adaptation"]):
        raise ValueError("Training stages must be disjoint")
    for relative in ("catalog.json", "public/train.jsonl", "manifest.json"):
        if original["inputHashes"][relative] != digest(args.dataset / relative):
            raise ValueError("Cached embeddings belong to different public training inputs")
    cached = torch.load(source / "training-embeddings.pt", map_location="cpu", weights_only=True)
    head = RouteHead(4096)
    report = {"synthetic": True, "config": CONFIG, "modelRevision": args.revision,
              "backboneCacheSha256": digest(source / "training-embeddings.pt"),
              "claims": "Real CPU route-head updates on cached actual pretrained readouts; toy public-policy imitation, not human behavior.", "stages": {}}
    for stage in ("initial", "adaptation"):
        xs, ys = [], []
        for record_id in stages[stage]:
            request = requests[record_id]
            embedding, actions = cached[record_id]
            if embedding.shape != (8, 4096) or actions != input_rows(request, catalog)[1]:
                raise ValueError("Cached action/readout shape mismatch")
            xs.append(embedding)
            ys += [ROUTES.index(teacher_route(request, catalog[a["productId"]])) for a in actions]
        x, y = torch.cat(xs), torch.tensor(ys, dtype=torch.long)
        evidence = fit(head, x, y)
        evidence["teacherClassCounts"] = dict(Counter(ROUTES[i] for i in ys))
        evidence["checkpoint"] = save_head(head, args.out / stage, args.revision)
        reloaded = load_head(args.out / stage, args.revision)
        if not torch.equal(head(x), reloaded(x)):
            raise AssertionError("Reloaded route predictions differ")
        evidence["exactPredictionReload"] = True
        report["stages"][stage] = evidence
    report["recordedAtUtc"] = datetime.now(timezone.utc).isoformat()
    dump(args.out / "training.json", report)
    print(json.dumps(report, indent=2), flush=True)


def approved_labels(catalog):
    """Derived approved dictionary: each label is only an existing catalog attribute."""
    return {"attribute:" + tag: tag.replace("-", " ") for p in catalog.values() for tag in p.get("tags", [])}


def mentions(text, tag):
    normalize = lambda value: " ".join(re.findall(r"[a-z0-9]+", value.lower()))
    return re.search(r"\b" + re.escape(normalize(tag)) + r"\b", normalize(text)) is not None


def label_result(raw_route, request, product, compatible, catalog, generate=None):
    if raw_route not in ROUTES:
        raise ValueError("Invalid route")
    evidence = grounding(request, product) if product else []
    result = {"schemaVersion": 1, "synthetic": True, "rawRoute": raw_route,
              "guardedRoute": raw_route, "servedRoute": raw_route, "selectedLabelId": None,
              "text": None, "grounding": [], "generationAttempted": False, "generationFailure": None}
    if not compatible or not evidence or guard_reasons(request, product):
        result.update(guardedRoute="NONE", servedRoute="NONE", guardReason="no-compatible-supported-observable-attributes")
        return result
    if raw_route == "NONE":
        return result
    tag = evidence[0]["productFact"]
    def select():
        result.update(servedRoute="SELECT", selectedLabelId="attribute:" + tag,
                      text=approved_labels(catalog)["attribute:" + tag], grounding=[evidence[0]])
    if raw_route == "SELECT":
        select()
        return result
    result["generationAttempted"] = True
    if generate is None:
        raise ValueError("GENERATE requires an explicit same-backbone decoder callback")
    try:
        generated = generate()
    except (MemoryError, KeyboardInterrupt):
        raise
    except Exception as error:
        if any(term in str(error).lower() for term in ("out of memory", "memory limit", "memory budget")):
            raise MemoryError(str(error)) from error
        result["generationFailure"] = f"{type(error).__name__}: {error}"
        select()
        return result
    validation = validate_copy(generated["text"], product, catalog, generated.get("finishReason"), [e["productFact"] for e in evidence])
    result["generation"] = {**generated, "validation": validation}
    candidates = [(validation["copy"], validation["copyAccepted"]), (validation["title"], validation["titleAccepted"] and validation["novelTitle"])]
    for field, (text, accepted) in zip(("copy", "title"), candidates):
        supports = [item for item in evidence if text and mentions(text, item["productFact"])]
        if accepted and text and 5 <= len(text) <= 90 and supports:
            result.update(text=text, grounding=supports, generatedField=field)
            return result
    result["generationFailure"] = "No independently accepted short field demonstrates a supported observable attribute"
    select()
    return result


def summary(values):
    ordered = sorted(values)
    def percentile(q):
        position = (len(ordered) - 1) * q
        lo = int(position)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (position - lo)
    return {"n": len(values), "p50": percentile(.5), "p95": percentile(.95)} if values else {"n": 0, "p50": None, "p95": None}


def benchmark(args):
    torch.set_num_threads(1)
    from .backbone import FrozenBackbone
    catalog = {p["id"]: p for p in json.loads((args.dataset / "catalog.json").read_text())["products"]}
    requests = {r["recordId"]: r for name in ("test", "diagnostics") for r in lines(args.dataset / f"public/{name}.jsonl")}
    plan = json.loads((args.baselines / "case-plan.json").read_text())
    previous = json.loads((args.baselines / "benchmark.json").read_text())
    previous_by_id = {r["recordId"]: r for r in previous["records"]}
    route = load_head(args.out / "adaptation", args.revision)
    if len(plan["recordIds"]) != 8 or previous["modelRevision"] != args.revision:
        raise ValueError("Expected the fixed eight-case same-revision benchmark")
    started = time.perf_counter()
    backbone = FrozenBackbone(args.model, args.revision, 1536, 6)
    heads, receipt = load_trained_heads(args.heads, backbone.hidden_size, args.revision)
    load_ms = (time.perf_counter() - started) * 1000
    if previous["headReceipt"]["checkpointSha256"] != receipt["checkpointSha256"]:
        raise ValueError("Prior benchmark uses a different product/presentation checkpoint")
    if route.linear.in_features != backbone.hidden_size or receipt["representation"] != "shared-prefix-v1":
        raise ValueError("Shared-prefix representation required for both heads")
    records = []
    for record_id in plan["recordIds"]:
        start = time.perf_counter()
        request = observable_request(requests[record_id])
        embeddings, actions, timing, state = encode_shared_prefix(backbone, request, catalog, 1536)
        with torch.inference_mode():
            selected = decisions(request, catalog, actions, heads(embeddings).tolist(), "shared-qwen-label-router", True)
        action = selected["presentationAction"]
        product = catalog[selected["selectedProductId"]] if selected["selectedProductId"] else None
        index = actions.index(action) if action else None
        raw = ROUTES[int(route(embeddings[index:index+1]).argmax().item())] if index is not None else "NONE"
        rows = input_rows(request, catalog)[0]
        def generate():
            return native_decode(backbone.model, backbone.tokenizer,
                generation_messages(rows[index], product, catalog, preferred_attributes=supported_preferences(request, product)),
                64, prefix_state=state)
        label = label_result(raw, request, product, action is not None, catalog, generate)
        total_ms = (time.perf_counter() - start) * 1000
        teacher = teacher_route(request, product) if product and action else "NONE"
        prior = previous_by_id[record_id]
        records.append({"recordId": record_id, "teacherRoute": teacher, "label": label,
                        "totalMs": total_ms, "prefixTiming": timing,
                        "priorAlwaysGenerateMs": prior["full"]["timing"]["fullCompletionMs"],
                        "priorSelectionOnlyMs": prior["selectionOnlyMs"]})
        del state, embeddings
        print(json.dumps({"routingCompleted": len(records), "raw": raw, "served": label["servedRoute"], "totalMs": total_ms}), flush=True)
    nonempty = [r for r in records if r["label"]["text"]]
    grounded = lambda row: bool(row["label"]["grounding"]) and all(e["productFact"] == e["observablePreference"] and mentions(row["label"]["text"], e["productFact"]) for e in row["label"]["grounding"])
    metrics = {"caseCount": len(records), "rawRouteAccuracyVsToyPolicy": sum(r["label"]["rawRoute"] == r["teacherRoute"] for r in records) / len(records),
               "guardedRouteAccuracyVsToyPolicy": sum(r["label"]["guardedRoute"] == r["teacherRoute"] for r in records) / len(records),
               "groundedLabelRateAmongServed": sum(grounded(r) for r in nonempty) / len(nonempty) if nonempty else None,
               "observablePreferenceRelevanceRateAmongServed": sum(grounded(r) for r in nonempty) / len(nonempty) if nonempty else None,
               "lexicallyValidatedFactualityRateAmongServed": sum(r["label"]["servedRoute"] == "SELECT" or r["label"].get("generation", {}).get("validation", {}).get(r["label"].get("generatedField", "") + "Accepted", False) for r in nonempty) / len(nonempty) if nonempty else None,
               "generationFailureCount": sum(r["label"]["generationFailure"] is not None for r in records),
               "generationAttemptCount": sum(r["label"]["generationAttempted"] for r in records),
               "generationServedCount": sum(r["label"]["servedRoute"] == "GENERATE" for r in records),
               "generationRate": sum(r["label"]["generationAttempted"] for r in records) / len(records),
               "rawCounts": dict(Counter(r["label"]["rawRoute"] for r in records)),
               "servedCounts": dict(Counter(r["label"]["servedRoute"] for r in records)),
               "totalLatencyMs": summary([r["totalMs"] for r in records]),
               "priorAlwaysGenerateMs": summary([r["priorAlwaysGenerateMs"] for r in records]),
               "priorSelectionOnlyMs": summary([r["priorSelectionOnlyMs"] for r in records])}
    report = {"schemaVersion": 1, "synthetic": True, "recordedAtUtc": datetime.now(timezone.utc).isoformat(),
              "metrics": metrics, "records": records, "modelRevision": args.revision, "productHeadReceipt": receipt,
              "routeCheckpointSha256": digest(args.out / "adaptation/route.pt"), "coldLoadMs": load_ms,
              "baselineSourceSha256": digest(args.baselines / "benchmark.json"),
              "limitations": ["Routing teacher is a toy explicit-public-tag-count policy; accuracy is not human relevance.",
                "Grounding is exact tag evidence plus restrictive lexical validation, not semantic factuality proof.",
                "Known-support NONE guard is deterministic masking, not learned behavior.",
                "Baselines are prior sequential measurements on identical case IDs, not concurrent or paired timing arms; always-generate uses full title/copy rather than this optional sub-label.",
                "Only new routing head trained; language decoder stays frozen and untuned. No architecture novelty claim.",
                "All data is synthetic; no storefront integration, human conversion or revenue evidence."]}
    dump(args.out / "benchmark.json", report)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("train", "benchmark"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--heads", type=Path, default=Path("data/ranker/pretrained/shared-prefix/adaptation"))
    parser.add_argument("--dataset", type=Path, default=Path("data/ranker/datasets/v2"))
    parser.add_argument("--out", type=Path, default=Path("data/ranker/pretrained/routing"))
    parser.add_argument("--baselines", type=Path, default=Path("data/ranker/pretrained/fusion-proof"))
    args = parser.parse_args()
    if args.command == "benchmark" and not args.model:
        parser.error("benchmark requires --model with the previously verified local snapshot")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
