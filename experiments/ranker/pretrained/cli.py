"""Pinned pretrained backbone + bounded, genuinely trained parallel head proof."""
import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "2"
import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path
import torch
from .backbone import FrozenBackbone, ParallelHeads
from .boundary import input_rows, decisions, guard_reasons
from experiments.ranker.model.features import FeatureEncoder
from experiments.ranker.model.network import DecisionNetwork, CONFIG as TINY_CONFIG

CONFIG = {"seed": 728, "initialEpochs": 80, "adaptationEpochs": 80, "learningRate": .03,
          "weightDecay": .001, "maxSeconds": 600, "maxTokens": 1536,
          "trainable": "shared linear suitability and presentation heads only; pretrained backbone frozen"}


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    point = (len(values) - 1) * q
    low = int(point)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (point - low)


def save_heads(heads, path, metadata):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(heads.state_dict(), path / "heads.pt")
    metadata = {**metadata, "sourceHashes": {source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in Path(__file__).parent.glob("*.py")}, "sha256": hashlib.sha256((path / "heads.pt").read_bytes()).hexdigest()}
    dump(path / "metadata.json", metadata)
    return metadata


def load_heads(path, size, expected_revision=None, expected_encoding=None):
    metadata = json.loads((path / "metadata.json").read_text())
    if expected_revision is not None and metadata.get("modelRevision") != expected_revision:
        raise ValueError("Head checkpoint belongs to a different pretrained revision.")
    if expected_encoding is not None:
        representation = "shared-prefix-v1" if expected_encoding == "shared-prefix" else "row-batch-v1"
        if metadata.get("encoding") != expected_encoding or metadata.get("representation") != representation:
            raise ValueError("Head checkpoint belongs to a different hidden-state representation.")
    if metadata.get("sha256") != hashlib.sha256((path / "heads.pt").read_bytes()).hexdigest():
        raise ValueError("Head checkpoint hash mismatch.")
    heads = ParallelHeads(size)
    heads.load_state_dict(torch.load(path / "heads.pt", map_location="cpu", weights_only=True))
    heads.eval()
    return heads


def fit(heads, embeddings, targets, epochs, deadline):
    if not bool(torch.isfinite(embeddings).all()) or not bool(torch.isfinite(targets).all()):
        raise FloatingPointError("Head training inputs must be finite.")
    optimizer = torch.optim.AdamW(heads.parameters(), lr=CONFIG["learningRate"], weight_decay=CONFIG["weightDecay"])
    start = time.perf_counter()
    first = heads(embeddings).detach().clone()
    loss_before = torch.nn.functional.mse_loss(first, targets).item()
    first_gradient = None
    for epoch in range(epochs):
        if time.perf_counter() > deadline:
            raise TimeoutError("Fixed ten-minute proof budget exhausted; no automatic sweep or larger run.")
        optimizer.zero_grad()
        prediction = heads(embeddings)
        loss = torch.nn.functional.mse_loss(prediction, targets)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Head training loss is non-finite.")
        loss.backward()
        if any(not bool(torch.isfinite(parameter.grad).all()) for parameter in heads.parameters()):
            raise FloatingPointError("Head training gradient is non-finite.")
        if first_gradient is None:
            first_gradient = sum(parameter.grad.square().sum().item() for parameter in heads.parameters()) ** .5
            if first_gradient <= 0:
                raise FloatingPointError("No head gradient: refusing to claim a successful training update.")
        optimizer.step()
    final = heads(embeddings).detach()
    if not bool(torch.isfinite(final).all()) or torch.equal(first, final):
        raise FloatingPointError("Head weights produced no finite prediction change.")
    return {"epochs": epochs, "rows": len(embeddings), "seconds": time.perf_counter() - start,
            "mseBefore": loss_before, "mseAfter": torch.nn.functional.mse_loss(final, targets).item(),
            "firstGradientL2": first_gradient, "meanAbsolutePredictionChange": (final - first).abs().mean().item()}


def targets_for(request, actions, labels):
    label = labels[request["recordId"]]
    return torch.tensor([[label["relevance"][action["productId"]] / 3,
                          label["presentationScores"][action["productId"]][action["variantId"]] / 3] for action in actions], dtype=torch.float32)


def extract(backbone, request, catalog):
    if getattr(backbone, "encoding", "batched") == "shared-prefix":
        total = time.perf_counter()
        from .fusion import encode_shared_prefix
        embeddings, actions, timing, state = encode_shared_prefix(backbone, request, catalog, max_prefix_tokens=backbone.max_length)
        timing["textConstructionAndEmbeddingMs"] = (time.perf_counter() - total) * 1000
        # Training does not retain a cache per example. The fusion inference API
        # explicitly keeps this exact state for its selected-product continuation.
        del state
        return embeddings, actions, timing
    total = time.perf_counter()
    rows, actions = input_rows(request, catalog)
    embeddings, timing = backbone.encode(rows)
    timing["textConstructionAndEmbeddingMs"] = (time.perf_counter() - total) * 1000
    return embeddings, actions, timing


def heuristic(request, catalog, actions):
    context = request["context"]
    preferences = " ".join(context.get("statedPreferences") or []).lower()
    presentation = " ".join(context.get("presentationPreferences") or []).lower()
    scores = []
    for action in actions:
        product = catalog[action["productId"]]
        variant = next(variant for variant in product["presentationVariants"] if variant["id"] == action["variantId"])
        relevance = sum(tag.lower() in preferences for tag in product["tags"]) - .002 * (product["price"] + product["shippingCost"])
        style = variant.get("presentationStyle", "")
        # Explicit lexical baseline uses the same current observable preferences,
        # never hidden rubric or labels. It cannot semantically parse paraphrases.
        appeal = sum(word in (variant["title"] + " " + variant.get("imageDescription", "")).lower() for word in presentation.split()) + float(style in presentation)
        scores.append([relevance, appeal])
    return scores


def private_metrics(request, result, truth, catalog):
    label = truth[request["recordId"]]
    pid = result["selectedProductId"]
    relevance = label["relevance"]
    ranking = [entry["productId"] for entry in result["rankedProducts"]]
    import math
    dcg = lambda values: sum((2 ** value - 1) / math.log2(index + 2) for index, value in enumerate(values))
    ideal = dcg(sorted(relevance.values(), reverse=True)[:3])
    ndcg = dcg([relevance[product] for product in ranking[:3]]) / ideal if ideal else 1
    action = result["presentationAction"]
    action_score = label.get("presentationScores", {}).get(pid, {}).get(action["variantId"], 0) if action else 0
    alternatives = label.get("presentationScores", {}).get(pid, {}) if pid else {}
    hidden = label.get("hiddenConstraints", {})
    product = catalog[pid] if pid else None
    private_bad = False
    if product:
        private_bad = (product["price"] + product["shippingCost"] > hidden.get("totalBudget", float("inf"))
                       or not set(hidden.get("requiredTags", [])).issubset(product["tags"])
                       or bool(set(hidden.get("avoidedTags", [])) & set(product["tags"]))
                       or (hidden.get("requiresInStock", False) and not product["inStock"]))
    return {"ndcgAt3": ndcg, "topBestRelevant": float(pid is not None and relevance[pid] == max(relevance.values()) and relevance[pid] > 0),
            "relevanceRegret": max(relevance.values()) - relevance.get(pid, 0), "knownConstraintViolation": float(bool(pid and guard_reasons(request, catalog[pid]))),
            "privateConstraintViolation": float(private_bad), "presentationScore": action_score,
            "presentationRegretWithinSelectedProduct": max(alternatives.values(), default=0) - action_score,
            "presentationBestWithinSelectedProduct": float(bool(action) and action_score == max(alternatives.values()) and action_score > 0),
            "presentationIncompatible": float(bool(action) and not request["presentationCompatibility"][pid][action["variantId"]]),
            "abstain": float(pid is None)}


def smoke(args, backbone, catalog, records, labels):
    request = records[0]
    embeddings, actions, timing = extract(backbone, request, catalog)
    heads = ParallelHeads(backbone.hidden_size)
    before = heads(embeddings).detach()
    report = fit(heads, embeddings, targets_for(request, actions, labels), 1, time.perf_counter() + 30)
    checkpoint = save_heads(heads, args.run / "smoke", {"synthetic": True, "modelRevision": args.revision, "encoding": args.encoding, "representation": "shared-prefix-v1" if args.encoding == "shared-prefix" else "row-batch-v1"})
    reloaded = load_heads(args.run / "smoke", backbone.hidden_size, expected_revision=args.revision, expected_encoding=args.encoding)
    assert torch.equal(heads(embeddings), reloaded(embeddings))
    assert not torch.equal(before, heads(embeddings))
    result = decisions(request, catalog, actions, reloaded(embeddings).detach().tolist(), "pretrained-smoke")
    report = {"synthetic": True, "backboneFrozen": True, "backboneDevice": backbone.device, "deviceMap": backbone.device_map, "availableBeforeBytes": backbone.available_before_bytes, "hiddenSize": backbone.hidden_size,
              "loadMs": backbone.load_ms, "memory": backbone.runtime_memory(), "timing": timing, "update": report, "checkpoint": checkpoint,
              "exactFreshHeadReload": True, "result": result}
    dump(args.run / "smoke.json", report)
    print(json.dumps(report, indent=2), flush=True)


def train_tiny_baseline(args, catalog, records, labels, stages):
    encoder = FeatureEncoder(catalog)
    model = DecisionNetwork(len(encoder.names), TINY_CONFIG["hiddenWidth"], TINY_CONFIG["seed"])
    deadline = time.perf_counter() + TINY_CONFIG["maxTrainingSeconds"]
    training, models = {}, {}
    for stage, epochs in (("initial", TINY_CONFIG["initialEpochs"]), ("adaptation", TINY_CONFIG["adaptationEpochs"])):
        x, y = [], []
        for record_id in stages[stage]:
            request = records[record_id]
            x.extend(encoder.encode(request))
            y.extend(labels[record_id]["relevance"][pid] / 3 for pid in request["candidateProductIds"])
        report = model.fit(x, y, epochs, deadline)
        path = args.run / ("engineered-" + stage)
        report["checkpoint"] = model.save(path, {"synthetic": True, "featureNames": encoder.names, "config": TINY_CONFIG,
            "presentationPolicy": "fixed lexical heuristic, not a trained presentation head"})
        training[stage] = report
        models["engineered-" + stage] = DecisionNetwork.load(path)
    return encoder, models, training


def workflow(args, backbone, catalog, records, labels):
    started = time.perf_counter()
    deadline = started + CONFIG["maxSeconds"]
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    stages = manifest["trainStageRecordIds"]
    record_by_id = {request["recordId"]: request for request in records}
    assert not set(stages["initial"]) & set(stages["adaptation"])
    artifact_receipt = Path("data/ranker/pretrained/qwen3.5-9b-4bit-artifacts.json")
    dump(args.run / "run-manifest.json", {"synthetic": True,
              "artifactVerification": json.loads(artifact_receipt.read_text()) if artifact_receipt.exists() else {"status": "receipt unavailable; expected hashes recorded separately"}, "config": CONFIG, "modelPath": str(args.model), "modelRevision": args.revision, "encoding": args.encoding, "representation": "shared-prefix-v1" if args.encoding == "shared-prefix" else "row-batch-v1",
              "backboneFrozen": True, "backboneDevice": backbone.device, "deviceMap": backbone.device_map, "availableBeforeBytes": backbone.available_before_bytes, "noGeneration": True,
              "tinyBaselineConfig": TINY_CONFIG,
              "expectedQuantizedArtifactHashes": {"model-00001-of-00002.safetensors": "a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd", "model-00002-of-00002.safetensors": "b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d"},
              "sourceHashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob("*.py")},
              "inputHashes": {str(path.relative_to(args.dataset)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.dataset / "catalog.json", args.dataset / "public/train.jsonl", args.dataset / "labels/train.jsonl", args.dataset / "manifest.json")},
              "python": platform.python_version(), "torch": torch.__version__})
    cache = {}
    extraction_times = []
    # Only training records and separate training labels are read before fitting.
    for request in records:
        if time.perf_counter() > deadline:
            raise TimeoutError("Training extraction exceeded ten-minute fixed budget.")
        cache[request["recordId"]] = extract(backbone, request, catalog)
        extraction_times.append(cache[request["recordId"]][2])
        print(json.dumps({"trainingExtracted": len(cache), "total": len(records), "lastMs": extraction_times[-1]["textConstructionAndEmbeddingMs"]}), flush=True)
    torch.save({record: (embedding, actions) for record, (embedding, actions, _) in cache.items()}, args.run / "training-embeddings.pt")
    heads = ParallelHeads(backbone.hidden_size)
    training = {}
    for stage, epochs in (("initial", CONFIG["initialEpochs"]), ("adaptation", CONFIG["adaptationEpochs"])):
        ordered_ids = stages[stage]
        embeddings = torch.cat([cache[record][0] for record in ordered_ids])
        targets = torch.cat([targets_for(record_by_id[record], cache[record][1], labels) for record in ordered_ids])
        training[stage] = fit(heads, embeddings, targets, epochs, deadline)
        training[stage]["checkpoint"] = save_heads(heads, args.run / stage, {"synthetic": True, "config": CONFIG, "modelRevision": args.revision, "encoding": args.encoding, "representation": "shared-prefix-v1" if args.encoding == "shared-prefix" else "row-batch-v1", "backboneFrozen": True})
    tiny_encoder, tiny_models, tiny_training = train_tiny_baseline(args, catalog, record_by_id, labels, stages)
    training["engineeredFeatureBaseline"] = tiny_training
    dump(args.run / "training.json", training)
    # Independent evaluation starts only after the fixed checkpoints are saved.
    test = lines(args.dataset / "public/test.jsonl")
    truth = {item["recordId"]: item for item in lines(args.dataset / "private/truth.jsonl")}
    models = {stage: load_heads(args.run / stage, backbone.hidden_size, expected_revision=args.revision, expected_encoding=args.encoding) for stage in ("initial", "adaptation")}
    models["random-head"] = ParallelHeads(backbone.hidden_size)
    totals = {name: {"raw": [], "guarded": []} for name in ("heuristic", "heuristic-first-variant", *models, *tiny_models)}
    timings, results = [], []
    for request in test:
        if time.perf_counter() > deadline:
            raise TimeoutError("Proof exceeded ten-minute fixed budget during evaluation.")
        start = time.perf_counter()
        embeddings, actions, timing = extract(backbone, request, catalog)
        for name in totals:
            if name.startswith("heuristic"):
                scores = heuristic(request, catalog, actions)
                if name == "heuristic-first-variant":
                    scores = [[score[0], float(index % 2 == 0)] for index, score in enumerate(scores)]
            elif name.startswith("engineered-"):
                values = tiny_models[name].predict(tiny_encoder.encode(request)).tolist()
                product_values = dict(zip(request["candidateProductIds"], values, strict=True))
                scores = [[product_values[action["productId"]], heuristic(request, catalog, actions)[index][1]] for index, action in enumerate(actions)]
            else:
                scores = models[name](embeddings).detach().tolist()
            for guarded in (False, True):
                result = decisions(request, catalog, actions, scores, name, guarded)
                totals[name]["guarded" if guarded else "raw"].append(private_metrics(request, result, truth, catalog))
                results.append({"guarded": guarded, **result})
        timing["uncachedEndToEndAllComparisonsMs"] = (time.perf_counter() - start) * 1000
        # Measure just the adapted heads and mask separately from evaluator work.
        start = time.perf_counter()
        decisions(request, catalog, actions, models["adaptation"](embeddings).detach().tolist(), "adaptation", True)
        timing["headsAndMaskMs"] = (time.perf_counter() - start) * 1000
        timing["uncachedDecisionMs"] = timing["textConstructionAndEmbeddingMs"] + timing["headsAndMaskMs"]
        timings.append(timing)
        print(json.dumps({"testEvaluated": len(timings), "total": len(test), "uncachedDecisionMs": timing["uncachedDecisionMs"]}), flush=True)
    diagnostic_report = {"pairs": [], "notUsedForTrainingOrSelection": True}
    diagnostic_path = args.dataset / "public/diagnostics.jsonl"
    if diagnostic_path.exists():
        diagnostic_truth = {item["recordId"]: item for item in lines(args.dataset / "private/diagnostic-truth.jsonl")}
        pairs = {}
        for request in lines(diagnostic_path):
            if time.perf_counter() > deadline:
                raise TimeoutError("Proof exceeded ten-minute fixed budget during counterfactual diagnostics.")
            embeddings, actions, _ = extract(backbone, request, catalog)
            result = decisions(request, catalog, actions, models["adaptation"](embeddings).detach().tolist(), "adaptation")
            label = diagnostic_truth[request["recordId"]]
            pairs.setdefault(label["diagnosticPairId"], []).append({"recordId": request["recordId"], "selectedProductId": result["selectedProductId"], "expectedProductId": label["bestProductId"]})
        diagnostic_report["pairs"] = [{"pairId": pair, "records": records,
            "selectionChanges": len({record["selectedProductId"] for record in records}) == 2,
            "bothCorrect": all(record["selectedProductId"] == record["expectedProductId"] for record in records)} for pair, records in pairs.items()]
    metrics = {name: {kind: {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]} for kind, rows in kinds.items()} for name, kinds in totals.items()}
    report = {"synthetic": True, "testCount": len(test), "models": metrics,
              "latencyScope": "Uncached text preparation, tokenization, quantized backbone forward, CPU heads and masks; excludes JSON serialization, transport and browser rendering.",
              "latencyMs": {key: {"p50": percentile([timing[key] for timing in timings], .5), "p95": percentile([timing[key] for timing in timings], .95)} for key in ("tokenizeMs", "forwardMs", "headsAndMaskMs", "uncachedDecisionMs")},
              "loadMs": backbone.load_ms, "counterfactualDiagnostics": diagnostic_report, "trainingExtraction": extraction_times, "uncachedTestTimings": timings,
              "backboneForwardCalls": backbone.forward_calls, "trainableHeadParameters": sum(parameter.numel() for parameter in heads.parameters()),
              "peakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
              "runtimeMemory": backbone.runtime_memory(),
              "totalSeconds": time.perf_counter() - started,
              "limitations": ["Tiny synthetic test; no human conversion or revenue evidence.", "Only decision heads trained; pretrained backbone is frozen.", "Training embeddings cached; all reported decision latency includes uncached tokenization and backbone execution.", "Text-only image descriptions, no image pixels.", "No claim that this quantized local backbone meets a page-load latency budget."]}
    dump(args.run / "evaluation.json", report)
    (args.run / "predictions.jsonl").write_text("".join(json.dumps(result) + "\n" for result in results))
    print(json.dumps(report, indent=2), flush=True)


def infer(args, backbone, catalog, records, labels):
    request = json.loads(args.request.read_text()) if args.request else lines(args.dataset / "public/test.jsonl")[0]
    embeddings, actions, timing = extract(backbone, request, catalog)
    heads = load_heads(args.run / "adaptation", backbone.hidden_size, expected_revision=args.revision, expected_encoding=args.encoding)
    result = decisions(request, catalog, actions, heads(embeddings).detach().tolist(), "qwen-frozen-parallel-head-adapted")
    result["timing"] = timing
    dump(args.run / "inference.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["smoke", "workflow", "infer"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/ranker/datasets/v2"))
    parser.add_argument("--run", type=Path, default=Path("data/ranker/pretrained/shared-prefix"))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--encoding", choices=["shared-prefix", "batched"], default="shared-prefix")
    parser.add_argument("--memory-gib", type=float, default=6)
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    catalog = {product["id"]: product for product in json.loads((args.dataset / "catalog.json").read_text())["products"]}
    records = [] if args.command == "infer" else lines(args.dataset / "public/train.jsonl")
    labels = {} if args.command == "infer" else {item["recordId"]: item for item in lines(args.dataset / "labels/train.jsonl")}
    CONFIG["maxTokens"] = args.max_tokens
    CONFIG["encoding"] = args.encoding
    backbone = FrozenBackbone(args.model, args.revision, args.max_tokens, args.memory_gib)
    backbone.encoding = args.encoding
    globals()[args.command](args, backbone, catalog, records, labels)


if __name__ == "__main__":
    main()
