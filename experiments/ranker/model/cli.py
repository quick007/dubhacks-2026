"""Reproducible local proof commands. Run with python -m experiments.ranker.model.cli."""
import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"
import argparse
import hashlib
import json
import platform
import resource
import threading
import time
import urllib.request
from pathlib import Path
from .common import read_jsonl, heuristic_scores, rank_result, evaluate_results, percentile
from .features import FeatureEncoder
from .network import CONFIG, DecisionNetwork
from .service import RankingService, make_server


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_data(dataset):
    products = json.loads((dataset / "catalog.json").read_text())["products"]
    catalog = {product["id"]: product for product in products}
    return catalog, FeatureEncoder(catalog)


def train(args):
    catalog, encoder = load_data(args.dataset)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    stages = manifest["trainStageRecordIds"]
    records = {item["recordId"]: item for item in read_jsonl(args.dataset / "public/train.jsonl")}
    labels = {item["recordId"]: item for item in read_jsonl(args.dataset / "labels/train.jsonl")}
    assert not set(stages["initial"]) & set(stages["adaptation"])
    # Schedule is frozen before any held-out data or private truth is read.
    run_manifest = {"synthetic": True, "architecture": "parallel engineered-feature tanh MLP", "config": CONFIG,
                    "claims": "From-scratch initial supervised fit followed by warm-start adaptation; not LLM post-training or Jev replication.",
                    "featureNames": encoder.names, "stages": {name: len(ids) for name, ids in stages.items()},
                    "inputHashes": {str(path.relative_to(args.dataset)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.dataset / "public/train.jsonl", args.dataset / "labels/train.jsonl", args.dataset / "manifest.json", args.dataset / "catalog.json"]},
                    "runtime": {"python": platform.python_version(), "platform": platform.platform(), "blasThreads": 1}}
    dump(args.run / "run-manifest.json", run_manifest)
    model = DecisionNetwork(len(encoder.names), CONFIG["hiddenWidth"], CONFIG["seed"])
    deadline = time.perf_counter() + CONFIG["maxTrainingSeconds"]
    reports = {}
    for stage, epochs in (("initial", CONFIG["initialEpochs"]), ("adaptation", CONFIG["adaptationEpochs"])):
        x, y = [], []
        for record_id in stages[stage]:
            request = records[record_id]
            x.extend(encoder.encode(request))
            y.extend(labels[record_id]["relevance"][pid] / 3 for pid in request["candidateProductIds"])
        report = model.fit(x, y, epochs, deadline)
        report["checkpoint"] = model.save(args.run / stage, {"stage": stage, "synthetic": True, "config": CONFIG, "featureNames": encoder.names})
        reports[stage] = report
    reports["peakProcessRssBytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    dump(args.run / "training.json", reports)
    print(json.dumps(reports, indent=2))


def evaluate(args):
    catalog, encoder = load_data(args.dataset)
    requests = read_jsonl(args.dataset / "public/test.jsonl")
    truth = {item["recordId"]: item for item in read_jsonl(args.dataset / "private/truth.jsonl")}
    reports = {"synthetic": True, "frozenConfig": CONFIG, "models": {}}
    all_results = []
    for name in ("heuristic", "initial", "adaptation", "adaptation-no-history"):
        load_start = time.perf_counter()
        model = None if name == "heuristic" else DecisionNetwork.load(args.run / ("adaptation" if name.startswith("adaptation") else name))
        load_ms = (time.perf_counter() - load_start) * 1000
        if model:
            model.predict(encoder.encode(requests[0]))  # one documented warm-up, not model selection
        results_by_guard = {}
        for guarded in (False, True):
            results = []
            for request in requests:
                start = time.perf_counter()
                scores = heuristic_scores(request, catalog) if model is None else model.predict(encoder.encode(request, history=name != "adaptation-no-history")).tolist()
                elapsed = (time.perf_counter() - start) * 1000
                results.append(rank_result(request, catalog, scores, name, guarded=guarded, latency_ms=elapsed))
            key = "guarded" if guarded else "raw"
            results_by_guard[key] = evaluate_results(requests, results, truth, catalog)
            all_results.extend({"guarded": guarded, **result} for result in results)
        reports["models"][name] = {"loadMs": load_ms, **results_by_guard}
    dump(args.run / "evaluation.json", reports)
    (args.run / "predictions.jsonl").write_text("".join(json.dumps(item) + "\n" for item in all_results))
    print(json.dumps(reports, indent=2))


def infer(args):
    catalog, encoder = load_data(args.dataset)
    started = time.perf_counter()
    model = DecisionNetwork.load(args.run / "adaptation")
    load_ms = (time.perf_counter() - started) * 1000
    request = json.loads(args.request.read_text()) if args.request else read_jsonl(args.dataset / "public/test.jsonl")[0]
    service = RankingService(catalog, encoder, model, args.deadline_ms)
    result = service.rank(request)
    result["timing"]["checkpointLoadMs"] = load_ms
    service.gate.close()
    dump(args.run / "inference.json", result)
    print(json.dumps(result, indent=2))


def serve(args):
    catalog, encoder = load_data(args.dataset)
    started = time.perf_counter()
    checkpoint = args.run / "adaptation"
    model = DecisionNetwork.load(checkpoint) if (checkpoint / "weights.npz").exists() else None
    service = RankingService(catalog, encoder, model, args.deadline_ms)
    service.warmup(read_jsonl(args.dataset / "public/train.jsonl")[0])
    server = make_server(service, args.port)
    print(json.dumps({"ready": True, "url": f"http://127.0.0.1:{args.port}", "loadMs": (time.perf_counter() - started) * 1000}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.gate.close()


def benchmark(args):
    catalog, encoder = load_data(args.dataset)
    started = time.perf_counter()
    model = DecisionNetwork.load(args.run / "adaptation")
    service = RankingService(catalog, encoder, model, args.deadline_ms)
    load_ms = (time.perf_counter() - started) * 1000
    warmup_start = time.perf_counter()
    service.warmup(read_jsonl(args.dataset / "public/train.jsonl")[0])
    warmup_ms = (time.perf_counter() - warmup_start) * 1000
    server = make_server(service, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    requests = read_jsonl(args.dataset / "public/test.jsonl")
    url = f"http://127.0.0.1:{server.server_port}/rank"
    def call(request, mode):
        payload = json.dumps({"request": request, "mode": mode}).encode()
        start = time.perf_counter()
        with urllib.request.urlopen(urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}), timeout=2) as response:
            result = json.load(response)
        return (time.perf_counter() - start) * 1000, result
    first_ms, _ = call(requests[0], "model")
    baseline, model_times, increments = [], [], []
    fallbacks = 0
    try:
        # Alternate pair order to reduce order/systematic warm-up bias; one fixed pass.
        for index, request in enumerate(requests):
            pair = {}
            for mode in (("default", "model") if index % 2 == 0 else ("model", "default")):
                pair[mode], result = call(request, mode)
                if mode == "model" and any(warning.startswith("Fallback:") for warning in result["warnings"]):
                    fallbacks += 1
            baseline.append(pair["default"])
            model_times.append(pair["model"])
            increments.append(pair["model"] - pair["default"])
    finally:
        server.shutdown()
        server.server_close()
        service.gate.close()
        thread.join()
    summarize = lambda values: {"p50": percentile(values, .5), "p95": percentile(values, .95), "max": max(values)}
    report = {"synthetic": True, "measurement": "Loopback HTTP including JSON, feature preparation, decision, guard and response; not integrated browser page-load timing.", "requestsPerMode": len(requests), "checkpointLoadMs": load_ms, "workerWarmupMs": warmup_ms, "firstHttpModelMs": first_ms, "deadlineMs": args.deadline_ms, "fallbacks": fallbacks, "defaultHttpMs": summarize(baseline), "modelHttpMs": summarize(model_times), "pairedIncrementalMs": summarize(increments)}
    dump(args.run / "http-benchmark.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "infer", "serve", "benchmark"])
    parser.add_argument("--dataset", type=Path, default=Path("data/ranker/datasets/v1"))
    parser.add_argument("--run", type=Path, default=Path("data/ranker/runs/initial"))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--port", type=int, default=4471)
    parser.add_argument("--deadline-ms", type=float, default=5)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
