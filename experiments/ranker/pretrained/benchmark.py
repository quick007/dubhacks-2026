"""Fixed local fusion measurements: repeated fresh processes and all preference controls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summary(values):
    ordered = sorted(values)
    def percentile(q):
        point = (len(ordered) - 1) * q
        lower = int(point)
        return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (point - lower)
    return {"n": len(ordered), "p50": percentile(.5), "p95": percentile(.95), "max": max(ordered)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--heads", type=Path, default=Path("data/ranker/pretrained/shared-prefix/adaptation"))
    parser.add_argument("--dataset", type=Path, default=Path("data/ranker/datasets/v2"))
    parser.add_argument("--output", type=Path, default=Path("data/ranker/pretrained/fusion-proof"))
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    args = parser.parse_args()
    public = read_lines(args.dataset / "public/test.jsonl")
    clear = next(row for row in public if len(row["context"].get("statedPreferences", [])) >= 2)
    sparse = next(row for row in public if not row["context"].get("statedPreferences"))
    # Include ALL six counterfactual requests, including unfavorable cases. Never read private truth.
    cases = [clear, sparse, *read_lines(args.dataset / "public/diagnostics.jsonl")]
    dump(args.output / "case-plan.json", {"synthetic": True, "selection": "First public clear and sparse requests, then every public counterfactual request; no private grading labels used.", "recordIds": [row["recordId"] for row in cases], "frozenBeforeGeneration": True})
    request_file = args.output / "cold-request.json"
    dump(request_file, clear)
    cold = []
    for index in range(3):
        output = args.output / f"cold-{index}.json"
        command = [sys.executable, "-m", "experiments.ranker.pretrained.fusion", "--mode", "full", "--model", str(args.model), "--revision", args.revision, "--heads", str(args.heads), "--request", str(request_file), "--output", str(output), "--max-input-tokens", str(args.max_input_tokens), "--memory-gib", str(args.memory_gib)]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
        elapsed = (time.perf_counter() - started) * 1000
        if completed.returncode:
            raise RuntimeError(completed.stderr[-3000:] or completed.stdout[-3000:])
        result = json.loads(output.read_text())
        cold.append({"processWallMs": elapsed, "timing": result["timing"], "copyStatus": result["copy"]["status"], "generatedTitleFulfilled": result["copy"]["generatedTitleFulfilled"]})
        print(json.dumps({"coldCompleted": index + 1, "processWallMs": elapsed, "copyStatus": result["copy"]["status"]}), flush=True)
    # Heavy libraries and the single resident backbone load only after cold children exit.
    from .backbone import FrozenBackbone
    from .fusion import FusionModel, load_trained_heads, supported_preferences
    started = time.perf_counter()
    backbone = FrozenBackbone(args.model, args.revision, args.max_input_tokens, args.memory_gib)
    heads, receipt = load_trained_heads(args.heads, backbone.hidden_size, args.revision)
    load_ms = (time.perf_counter() - started) * 1000
    model = FusionModel(backbone, heads, receipt)
    catalog = {row["id"]: row for row in json.loads((args.dataset / "catalog.json").read_text())["products"]}
    records = []
    for index, request in enumerate(cases):
        record = {"recordId": request["recordId"], "synthetic": True}
        # Alternate order; both arms run uncached prefix preparation, never head-only timing.
        for arm in (("full", "selection-only") if index % 2 == 0 else ("selection-only", "full")):
            if arm == "full":
                result = model.infer(request, catalog)
                dump(args.output / f"warm-{index}.json", result)
                record["full"] = result
            else:
                started = time.perf_counter()
                events = model.infer_events(request, catalog)
                decision = next(events)["result"]
                events.close()  # Ends before decoder invocation; approved selection baseline only.
                record["selectionOnlyMs"] = (time.perf_counter() - started) * 1000
                record["selectionOnlyProductId"] = decision["selectedProductId"]
        result = record["full"]
        selected = result["structured"]["selectedProductId"]
        record["supportedExplicitPreferences"] = supported_preferences(request, catalog[selected]) if selected else []
        records.append(record)
        print(json.dumps({"warmCompleted": len(records), "total": len(cases), "fullCompletionMs": result["timing"]["fullCompletionMs"], "copyStatus": result["copy"]["status"], "generatedTitleFulfilled": result["copy"]["generatedTitleFulfilled"]}), flush=True)
    report = {"schemaVersion": 1, "recordedAtUtc": datetime.now(timezone.utc).isoformat(), "synthetic": True,
              "modelRevision": args.revision, "headReceipt": receipt, "caseCount": len(records),
              "cold": {"measurement": "Three separate Python processes on the same clear request. OS disk/shader caches uncontrolled; not machine-cold.", "processWallMs": summary([row["processWallMs"] for row in cold]), "samples": cold},
              "residentLoadMs": load_ms,
              "resident": {"measurement": "One resident model, fresh full prefix for every arm/request. Complete response includes autoregressive text and independent field validation; excludes transport/browser.",
                           "fullCompletionMs": summary([row["full"]["timing"]["fullCompletionMs"] for row in records]),
                           "structuredMs": summary([row["full"]["timing"]["uncachedStructuredMs"] for row in records]),
                           "selectionOnlyMs": summary([row["selectionOnlyMs"] for row in records]),
                           "generationMs": summary([row["full"]["timing"]["generationMs"] for row in records]),
                           "generatedTitleCount": sum(row["full"]["copy"]["generatedTitleFulfilled"] for row in records),
                           "bothFieldsAcceptedCount": sum(row["full"]["copy"]["status"] == "generated" for row in records),
                           "partialAcceptanceCount": sum(row["full"]["copy"]["status"] == "generated-partial" for row in records),
                           "fallbackCount": sum(row["full"]["copy"]["status"] == "fallback" for row in records)},
              "records": records, "memory": backbone.runtime_memory(),
              "peakProcessRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
              "limitations": ["Small synthetic workload, not a statistical or human-conversion result.", "Decoder remains frozen/untuned; only structured heads trained.", "Selection-only does not generate new copy; latency is not capability-equivalent.", "The learned NONE/SELECT/GENERATE sub-label router is not implemented or evaluated here.", "Field-level lexical grounding is restrictive and not a semantic factuality proof."]}
    dump(args.output / "benchmark.json", report)
    print(json.dumps({"report": str(args.output / "benchmark.json"), "cold": report["cold"]["processWallMs"], "resident": report["resident"], "memory": report["memory"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
