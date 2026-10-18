"""Root-only six-request fused result benchmark; prepare/tests are CPU-only.

Inspected synthetic replay: focused-v2 product head + original FULL-input image
head + newly completed short-prefix copy adapter, six cards from36 candidates.
All six native compatibility gates precede timed fresh requests. No fits, cached
query inference, text fallback, image fetch, browser or live-store claim.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from .catalog_fusion.inputs import canonical, digest, request_text, tensor_digest
from .catalog_fusion.preflight import DEFAULT_MODEL, REVISION, file_digest

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "catalog_fusion/fresh_eval_v1"
FOCUS_RUN = Path("data/ranker/pretrained/focused-rank-transfer-v1")
IMAGE_RUN = Path("data/ranker/pretrained/catalog-user-semantic-fresh-eval-v1-attempt-1")
COPY_REPORT = Path("data/ranker/pretrained/short-copy-fit-v1/results.json")
VERSION = "fused-six-result.v1"
LIMITS = {"requests": 6, "productPairs": 3, "candidateProducts": 36,
          "candidateVariants": 72, "requestedCards": 6, "width": 6,
          "firstFields": 10, "fullFields": 20, "firstSampledCap": 208,
          "fullSampledCap": 416, "memoryGiB": 7, "contextTokens": 4096,
          "maximumQueryTokens": 512, "maximumSuffixTokens": 256,
          "maximumCommonTokens": 512, "prefillStep": 256}
EXTRA_SOURCES = ("fused_result_benchmark.py", "test_fused_result_benchmark.py",
    "focused_rank_transfer.py", "catalog_fusion/inputs.py", "catalog_fusion/runtime.py",
    "catalog_fusion/fresh_eval_runner.py", "catalog_fusion/fresh_eval_data.py",
    "catalog_fusion/centered.py", "catalog_fusion/scorer.py", "catalog_fusion/semantic_factors.py")


def write_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False); handle.write("\n")


def sources():
    from .short_copy_fit import SOURCES
    return {name: file_digest(ROOT / name) for name in sorted(set(SOURCES) | set(EXTRA_SOURCES))}


def completed_copy(path=COPY_REPORT):
    """Require final completion, real240 updates, reload and frozen-language gates."""
    path = Path(path); report = json.loads(path.read_text())
    if (report.get("status") != "completed" or report.get("phase") != "completed"
            or type(report.get("nativeUpdates")) is not int or report["nativeUpdates"] != 240
            or type(report.get("optimizerUpdateCalls")) is not int or report["optimizerUpdateCalls"] != 240
            or len(report.get("extraction", [])) != 240 or len(report.get("steps", [])) != 240
            or report.get("languageParameterTensorsHashed") != 927):
        raise ValueError("A fully completed240-update short-copy run is required")
    controls = report.get("controls", {})
    if any(controls.get(k) is not True for k in ("frozenLanguageHashesUnchanged",
            "learnedOriginalArchivedNativeParityExact", "sourceFilesUnchanged", "datasetManifestUnchanged",
            "all240NonzeroBGatesPassed", "fitStartedFromExactWarmStart")):
        raise ValueError("Completed frozen-language/source/learned parity controls required")
    if (any(r.get("status") != "completed" or r.get("updateEvaluated") is not True for r in report["steps"])
            or sorted(r.get("step", -1) for r in report["steps"]) != list(range(240))
            or sorted(r.get("featureIndex", -1) for r in report["steps"]) != list(range(240))):
        raise ValueError("All240 evaluated unique updates and complete feature permutation required")
    from .short_copy_fit import CONFIG, SOURCES
    if report.get("config") != CONFIG or report.get("preflight", {}).get("sources") != {n: file_digest(ROOT / n) for n in SOURCES}:
        raise ValueError("Copy producer configuration or exact frozen source bytes differ")
    if controls.get("new-C", {}).get("status") != "exact-controls-passed":
        raise ValueError("Completed learned-copy native matched-width controls required")
    checkpoint = report.get("checkpoint", {})
    if (checkpoint.get("exact") is not True or checkpoint.get("parametersBeforeReload") != checkpoint.get("parametersAfterReload")
            or set(checkpoint.get("parametersBeforeReload", {})) != {"lora_a", "lora_b"}):
        raise ValueError("Exact final A/B checkpoint reload receipt required")
    model = path.parent / "adapter.safetensors"
    if file_digest(model) != checkpoint.get("sha256"):
        raise ValueError("Final copy checkpoint checksum mismatch")
    return {"report": str(path.resolve()), "reportSha256": file_digest(path),
            "checkpoint": str(model.resolve()), "checkpointSha256": checkpoint["sha256"],
            "nativeUpdates": 240, "languageTensors": 927, "selection": "fixed final step"}


def select_six(requests, controls):
    """Predetermined first3 product pairs, identities only; labels are unused."""
    pairs = [p for p in controls["pairs"] if p["kind"] == "product"][:3]
    if len(pairs) != 3 or any(len(p["recordIds"]) != 2 for p in pairs):
        raise ValueError("Exactly first3 inspected product pairs required")
    ids = [rid for pair in pairs for rid in pair["recordIds"]]
    if len(set(ids)) != 6 or any(rid not in requests for rid in ids):
        raise ValueError("Six distinct existing public requests required")
    return ids, [{"pairIndex": i, "recordIds": list(p["recordIds"])} for i, p in enumerate(pairs)]


def combine_plan(request, catalog, product_order, product_scores, image_scores):
    """Combine two genuine trained scores for every actual72 action rows."""
    from .catalog_fusion.scorer import roster
    from .evidence_plan import all_actions
    from .page_plan import make_page_plan, validate_page_plan
    if len(product_order) != 36 or len(set(product_order)) != 36 or set(product_order) != set(catalog):
        raise ValueError("Complete36-product focused score mapping required")
    products, images = torch.as_tensor(product_scores), torch.as_tensor(image_scores)
    actions = roster(request, catalog)
    if products.shape != (36,) or images.shape != (72,) or len(actions) != 72 or not torch.isfinite(products).all() or not torch.isfinite(images).all():
        raise ValueError("Finite complete36-product/72-image scores required")
    mapping = dict(zip(product_order, products.tolist(), strict=True))
    scores = [[mapping[pid], float(images[i])] for i, (pid, _) in enumerate(actions)]
    action_dicts = [{"productId": p, "variantId": v} for p, v in actions]
    decision = all_actions(request, catalog, action_dicts, scores)
    decision["modelId"] = "focused-v2-product+frozen-full-centered-image"
    plan = make_page_plan(request, catalog, decision, 6)
    validate_page_plan(plan, request, catalog)
    cards = [c for s in plan["sections"] for c in s["products"]]
    if plan["shortfall"] or len(cards) != 6 or len({c["productId"] for c in cards}) != 6:
        raise ValueError("Six eligible distinct cards required; no smaller fallback")
    return decision, plan, {"actions": action_dicts, "scores": scores,
                            "productOrder": list(product_order), "productScores": products.tolist()}


def layout_case(request, catalog, plan):
    from .short_copy_data import layout
    from .short_copy_fit import FIRST_FIELDS, SECOND_FIELDS
    raw, common, rows, evidence = layout(plan, request, catalog)
    if (tuple(r["fields"][0] for r in rows) != FIRST_FIELDS + SECOND_FIELDS
            or sum(r["maxTokens"] for r in rows[:10]) != 208 or sum(r["maxTokens"] for r in rows) != 416):
        raise ValueError("Unchanged complete20 field roster/caps required")
    for row in rows:
        if "targetValue" in row:
            raise ValueError("Teacher targets cannot enter runtime rows")
        row["key"] = row["rowId"]
    return {"caseId": request["recordId"], "request": request, "catalog": catalog,
            "pendingPagePlan": plan}, raw, common, rows, evidence


def roster(ids):
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ValueError("Fixed six-request denominator required")
    return [{"recordId": rid, "status": "not-attempted", "requestedCards": 6} for rid in ids]


def summary(arms):
    if len(arms) != 6 or len({a["recordId"] for a in arms}) != 6:
        raise ValueError("Never shrink six-request denominator")
    done = [a for a in arms if a["status"] == "completed-attempt"]
    first = [a for a in done if a.get("firstSection", {}).get("validation", {}).get("valid") is True]
    full = [a for a in done if a.get("fullValidation", {}).get("valid") is True]
    result = {"fixedRequests": 6, "completed": len(done), "failed": sum(a["status"] == "failed" for a in arms),
        "unattempted": sum(a["status"] == "not-attempted" for a in arms),
        "inProgress": sum(a["status"] == "running" for a in arms),
        "validThreeCardResults": len(first), "validSixCardResults": len(full),
        "validThreeRateFixedDenominator": len(first) / 6, "validSixRateFixedDenominator": len(full) / 6,
        "allSixFirstThreeValid": len(first) == 6, "allSixFullSixValid": len(full) == 6,
        "attemptFirstReadyMs": [a["firstReceipt"]["validatedAndSerializedAtMs"] for a in done],
        "validFirstReadyMs": [a["firstReceipt"]["validatedAndSerializedAtMs"] for a in first],
        "attemptFullCompletionMs": [a["fullReceipt"]["validatedAndSerializedAtMs"] for a in done],
        "validFullCompletionMs": [a["fullReceipt"]["validatedAndSerializedAtMs"] for a in full],
        "tailLatencyClaim": False, "browserPaintMeasured": False}
    for key in ("validFirstReadyMs", "validFullCompletionMs"):
        result[key.replace("Ms", "MedianMs")] = statistics.median(result[key]) if result[key] else None
    return result


def _rows(packet, role, text_for):
    result = {}
    for row in packet["records"]:
        if row["role"] == role:
            text = text_for(row["member"])
            if digest(text) != row["textSha256"]:
                raise ValueError("Existing catalog feature text representation changed")
            key = canonical(row["member"])
            if key in result: raise ValueError("Duplicate feature membership")
            result[key] = {**row, "text": text}
    return result


def components(dataset=DATA, focus_run=FOCUS_RUN, image_run=IMAGE_RUN):
    """CPU-only immutable head/provenance/store setup; no cached query access."""
    from .focused_rank_transfer import heads
    from .catalog_fusion.fresh_eval_runner import public_inputs, models_from_source
    from .catalog_fusion.generalization_v2 import catalog_text_v2
    from .catalog_fusion.inputs import catalog_text
    from .catalog_fusion.runtime import FeatureStore
    frozen, catalog, requests = public_inputs(dataset)
    fp = json.loads((focus_run / "packet.json").read_text())
    ip = json.loads((image_run / "fresh-preflight.json").read_text())
    fr = json.loads((focus_run / "report.json").read_text())
    ir = json.loads((image_run / "report.json").read_text())
    if fr["status"] != "completed" or fr["nativeForwards"] != 60 or ir["status"] != "completed" or ir["fits"] != 0:
        raise ValueError("Completed immutable focused/full feature sources required")
    for packet, report, key in ((fp, fr, "dataFreezeSha256"), (ip, ir, "freezeSha256")):
        if packet[key] != file_digest(dataset / "FREEZE.json"):
            raise ValueError("Feature source belongs to another catalog")
    head_receipt, factors, _ = heads(Path(fp["sourceHeads"]["sourceDirectory"]))
    if head_receipt != fp["sourceHeads"]:
        raise ValueError("Focused-v2 factor checkpoint changed")
    _, _, image = models_from_source(frozen, Path(frozen["sourceCheckpoints"]["semanticRun"]))
    if file_digest(image_run / "centered-image.pt") != frozen["sourceCheckpoints"]["image"]["sha256"]:
        raise ValueError("Original FULL-image checkpoint changed")
    provenance = [json.loads((run / "model-verification.json").read_text()) for run in (focus_run, image_run)]
    for key in ("weightsSha256", "tokenizerSha256", "templateSha256", "quantization", "configSha256", "evidenceSource"):
        if provenance[0][key] != provenance[1][key]:
            raise ValueError("Focused/image feature backbone provenance differs")
    focus_rows = _rows(fp, "catalog", lambda pid: catalog_text_v2(catalog[pid], "focused"))
    full_rows = _rows(ip, "catalog", lambda member: catalog_text(catalog[member[0]],
        next(v for v in catalog[member[0]]["presentationVariants"] if v["id"] == member[1])))
    if len(focus_rows) != 36 or len(full_rows) != 72:
        raise ValueError("Require36 focused +72 original FULL catalog rows")
    return {"catalog": catalog, "requests": requests, "focusHead": factors["learned"], "imageHead": image,
        "focusStore": FeatureStore.open_existing(focus_run / "features", provenance[0]),
        "fullStore": FeatureStore.open_existing(image_run / "features", provenance[1]),
        "focusRows": focus_rows, "fullRows": full_rows, "provenance": provenance[0],
        "focusPacket": fp, "imagePacket": ip, "focusRun": focus_run, "imageRun": image_run,
        "focusFreeze": json.loads((focus_run / "PREDICTIONS_FROZEN.json").read_text()),
        "imageFreeze": json.loads((image_run / "FEATURES_PREDICTIONS_FROZEN.json").read_text()),
        "historicalCatalogPreparation": {"focused36Ms": fr["coldCatalogEncodeSaveVerifyMs"],
            "full72Ms": ir["extraction"]["catalogColdConstructEncodeSaveVerifyMs"],
            "scope": "historical native catalog extraction; separately disclosed, not newly free compute"}}


def warm_catalogs(component, request):
    """Every request reauthenticates108 real catalog entries; no query cache read."""
    from .catalog_fusion.scorer import roster as action_roster
    order = request["candidateProductIds"]; actions = action_roster(request, component["catalog"])
    focus, full = [], []
    for role, members, rows_key, store_key, run_key, destination in (
            ("focused", order, "focusRows", "focusStore", "focusRun", focus),
            ("full", [list(a) for a in actions], "fullRows", "fullStore", "imageRun", full)):
        for member in members:
            row = component[rows_key][canonical(member)]; store = component[store_key]
            values, receipt = store.load(row)
            data, metadata = store.paths("catalog", member)
            if role == "focused":
                for path in (data, metadata):
                    relative = str(path.relative_to(component[run_key]))
                    if file_digest(path) != component["focusFreeze"]["files"][relative]:
                        raise ValueError("Focused catalog artifact differs from frozen extraction")
            else:
                expected = component["imageFreeze"]["features"]["catalog"][canonical(member)]
                if (receipt["rawKeys"][0], receipt["artifactSha256"], receipt["sha256"]) != (
                        expected["rawKey"], expected["artifactSha256"], expected["sha256"]):
                    raise ValueError("FULL-image catalog feature differs from original frozen extraction")
            destination.append(values[0])
    return torch.stack(focus), torch.stack(full)


def prepare_packet(tokenizer, *, dataset=DATA, focus_run=FOCUS_RUN, image_run=IMAGE_RUN, copy_report=COPY_REPORT):
    from .fused_prefix import focused_query_text
    copy_receipt = completed_copy(copy_report)
    component = components(dataset, focus_run, image_run)
    controls = json.loads((dataset / "private/controls.json").read_text())
    ids, pairs = select_six(component["requests"], controls)
    paths = [dataset / "FREEZE.json", dataset / "catalog.json", dataset / "public/requests.jsonl",
             dataset / "private/controls.json", focus_run / "packet.json", focus_run / "report.json",
             focus_run / "PREDICTIONS_FROZEN.json", focus_run / "model-verification.json",
             image_run / "fresh-preflight.json", image_run / "report.json", image_run / "FEATURES_PREDICTIONS_FROZEN.json",
             image_run / "model-verification.json", image_run / "centered-image.pt"]
    tokens = []
    for rid in ids:
        request = component["requests"][rid]
        row = {"recordId": rid}
        for role, text in (("focused", focused_query_text(request, component["catalog"])),
                           ("fullImage", request_text(request, component["catalog"]))):
            values = tokenizer.encode(text, add_special_tokens=False)
            values = list(values.ids) if hasattr(values, "ids") else list(values)
            if not 0 < len(values) <= 512: raise ValueError("Untruncated B1 query <=512 required")
            row[role] = {"textSha256": digest(text), "tokenIds": values, "tokens": len(values)}
        tokens.append(row)
        # CPU source/cache authenticity only; these vectors never choose runtime products here.
        warm_catalogs(component, request)
    from .semantic_copy_feature_fit import model_input_receipt
    return {"version": VERSION, "sourceHashes": sources(), "limits": LIMITS,
        "dataset": str(dataset.resolve()), "focusRun": str(focus_run.resolve()), "imageRun": str(image_run.resolve()),
        "copy": copy_receipt, "requestIds": ids, "pairs": pairs, "queryTokens": tokens,
        "inputFiles": {str(p.resolve()): file_digest(p) for p in paths}, "modelFiles": model_input_receipt(),
        "historicalCatalogPreparation": component["historicalCatalogPreparation"],
        "scope": "resident same9B,108 verified warm catalog entries, actual fresh two-query representations, trained decisions, genuine first3/full6 copy",
        "selection": "first3 product pairs of previously inspected synthetic fresh_eval_v1 replay; no unseen claim",
        "controlPhase": "all6 exact feature+adapter/cache gates before timed requests; never reused query/control state",
        "limitations": ["Six requested cards from36 synthetic products; not a16/24-card or live-store result",
            "Old centered image scorer remains imperfect; image metadata is text, synthetic image keys are not fetched/rendered",
            "Known budget/stock/shipping/compatibility guards and section grouping remain deterministic",
            "First3 serialized model payload is not browser paint; actual body fields retain unchanged validator",
            "Cold model/catalog/control costs disclosed separately; no tail-latency or throughput claim",
            "Current16-product storefront snapshot and historical6-jacket import are separate domains"]}


def feature_gate(baseline, cached, state):
    if any(v.shape != (1, 4096) or v.dtype != torch.float32 or not torch.isfinite(v).all() for v in (baseline, cached)):
        raise ValueError("Native normalized B1 FP32 query vectors required")
    return {"featureMode": "frozen-base", "featureTokenSha256": state["featureTokenSha256"],
        "cacheVsNoCacheFeaturesExact": torch.equal(baseline, cached),
        "baselineSha256": tensor_digest(baseline), "cachedSha256": tensor_digest(cached),
        "maxAbsDelta": float((baseline - cached).abs().max()),
        "policy": "exact; any numerical mismatch stops unchanged-head fused claim"}


def _query_tokens(tokenizer, text, expected):
    actual = tokenizer.encode(text, add_special_tokens=False)
    if actual != expected["tokenIds"] or digest(text) != expected["textSha256"]:
        raise ValueError("Fresh native query differs from frozen raw text/token representation")


class NativeBackend:
    """Created only by root run after completed-copy and reviewed-packet checks."""
    def __init__(self, packet, out):
        import mlx.core as mx
        import numpy as np
        from mlx_lm.generate import generation_stream
        from mlx_lm.tuner.lora import LoRALinear
        from .backbone import FrozenBackbone
        from .fused_prefix import RawPrefixTokenizer
        from .semantic_copy_feature_fit import verify_model_files
        from .semantic_copy_training import parameter_hashes
        self.packet, self.out, self.mx, self.np = packet, out, mx, np
        self.component = components(Path(packet["dataset"]), Path(packet["focusRun"]), Path(packet["imageRun"]))
        self.stream = generation_stream; self.context = mx.stream(generation_stream); self.context.__enter__()
        self.started = time.perf_counter()
        self.model_verified = verify_model_files(packet["modelFiles"])
        class Guarded(FrozenBackbone):
            def check_memory(self):
                super().check_memory()
                if max(self.mx.get_peak_memory(), self.mx.get_active_memory()) > self.memory_budget:
                    raise MemoryError("Observed active/peak exceeds unchanged7GiB guard")
        self.bb = Guarded(Path(DEFAULT_MODEL), REVISION, 512, LIMITS["memoryGiB"])
        self.bb.tokenizer = RawPrefixTokenizer(self.bb.tokenizer)
        self.model = self.bb.model; self.base_hashes = parameter_hashes(self.model.language_model)
        if len(self.base_hashes) != 927: raise ValueError("Pinned927 frozen language tensors required")
        self.adapter = LoRALinear.from_base(self.model.layers[-1].mlp.down_proj, r=4, scale=20, dropout=0)
        self.model.layers[-1].mlp.down_proj = self.adapter
        weights = mx.load(packet["copy"]["checkpoint"])
        if (set(weights) != {"lora_a", "lora_b"} or tuple(weights["lora_a"].shape) != (12288, 4)
                or tuple(weights["lora_b"].shape) != (4, 4096)):
            raise ValueError("Only final-down rank4 A/B copy adapter supported")
        if not all(bool(mx.all(mx.isfinite(v)).item()) for v in weights.values()):
            raise FloatingPointError("Nonfinite learned copy weights")
        self.adapter.load_weights(list(weights.items()), strict=False)
        self.model.freeze(); self.model.eval(); mx.eval(self.adapter.parameters()); mx.synchronize(generation_stream)
        if not bool(mx.any(self.adapter.lora_b != 0).item()): raise ValueError("Learned nonzeroB required")
        self.adapter_sha = packet["copy"]["checkpointSha256"]
        self.adapter_tensors = self._adapter_receipts()
        self.query_tokens = {r["recordId"]: r for r in packet["queryTokens"]}
        self.cold = {"modelResidentSetupMs": (time.perf_counter() - self.started) * 1000,
                     "modelVerification": self.model_verified, "residentLoadMs": self.bb.load_ms,
                     "historicalCatalogPreparation": packet["historicalCatalogPreparation"],
                     "nativeCatalogForwardsThisRun": 0}

    def _adapter_receipts(self):
        return {name: hashlib.sha256(self.np.array(getattr(self.adapter, name)).tobytes()).hexdigest()
                for name in ("lora_a", "lora_b")}

    def _cheap_integrity(self):
        if sources() != self.packet["sourceHashes"] or self._adapter_receipts() != self.adapter_tensors:
            raise ValueError("Source or actual active adapter bytes changed")
        if file_digest(self.packet["copy"]["checkpoint"]) != self.adapter_sha:
            raise ValueError("Completed learned checkpoint file changed")

    def selection(self, rid):
        """Fresh native query features; no old query artifact is read here."""
        from .focused_rank_transfer import score
        from .semantic_copy_training import disabled
        from .fused_prefix import focused_query_text, prepare_fused_prefix
        request = self.component["requests"][rid]; catalog = self.component["catalog"]
        start = time.perf_counter(); self._cheap_integrity(); integrity_ms = (time.perf_counter() - start) * 1000
        before = time.perf_counter(); focused_matrix, full_matrix = warm_catalogs(self.component, request)
        warm_ms = (time.perf_counter() - before) * 1000
        expected = self.query_tokens[rid]
        with disabled(self.model, self.adapter):
            image_text = request_text(request, catalog)
            _query_tokens(self.bb.tokenizer, image_text, expected["fullImage"])
            full_query, image_timing = self.bb.encode([image_text])
            raw_text = focused_query_text(request, catalog)
            _query_tokens(self.bb.tokenizer, raw_text, expected["focused"])
            raw_query, raw_timing, raw = prepare_fused_prefix(self.bb, request, catalog, feature_mode="frozen-base")
        before = time.perf_counter()
        with torch.no_grad():
            product_scores, _, _ = score(self.component["focusHead"], raw_query[0], focused_matrix, request["candidateProductIds"])
            image_scores = self.component["imageHead"](full_query[0], full_matrix)[:, 1]
        heads_ms = (time.perf_counter() - before) * 1000
        before = time.perf_counter()
        decision, plan, scored = combine_plan(request, catalog, request["candidateProductIds"], product_scores, image_scores)
        case, root, common, rows, evidence = layout_case(request, catalog, plan)
        if root != raw_text: raise AssertionError("Copy root and focused factor text differ")
        planning_ms = (time.perf_counter() - before) * 1000
        return {"case": case, "raw": raw, "rawQuery": raw_query, "fullQuery": full_query,
            "common": common, "rows": rows, "evidence": evidence, "decision": decision, "scored": scored,
            "timing": {"sourceAdapterIntegrityMs": integrity_ms, "verifiedWarmCatalog108Ms": warm_ms,
                "freshFullImageQuery": image_timing, "freshRawFocusedQuery": raw_timing,
                "trainedCpuHeadsMs": heads_ms, "guardsPlanLayoutMs": planning_ms}}

    def _common(self, selected):
        from .fused_prefix import append_common_message
        from .semantic_copy_training import disabled
        with disabled(self.model, self.adapter):
            return append_common_message(self.bb, selected["raw"], selected["common"], current_mode="frozen-base")

    def _cache_proof(self, selected, expected_hash, directory, receipt):
        """Actual raw-B1/common256/field-B1 off/on cache + forced3 proof."""
        from .fused_prefix import prepare_fused_prefix, append_common_message, token_hash
        from .native_batch_diagnosis import state_hash
        from .semantic_copy_training import disabled
        states, logits, source_hashes, prompt_hashes = [], [], [], []
        mx, model, field = self.mx, self.model, selected["rows"][0]
        forced = [v for v in self.bb.tokenizer.encode(field["messages"][0]["content"], add_special_tokens=False)
                  if v not in self.bb.tokenizer.eos_token_ids][:3]
        if len(forced) != 3: raise ValueError("Three actual instruction tokens required for diagnostic continuation")
        for enabled in (False, True):
            with disabled(model, self.adapter) if not enabled else nullcontext():
                mode = self.adapter_sha if enabled else "frozen-base"
                features, _, raw = prepare_fused_prefix(self.bb, selected["case"]["request"], selected["case"]["catalog"], feature_mode=mode)
                del features
                state, _ = append_common_message(self.bb, raw, selected["common"], current_mode=mode)
                source_hashes.append(state_hash(state["cache"]))
                receipt["cacheProof"] = {"sourceStateHashes": list(source_hashes), "forcedTokens": forced,
                    "boundary": "raw wholeB1/common chunks256/field wholeB1", "teacherAnswerUsed": False}
                write_progress(directory / "gate.json", receipt)
                if source_hashes[-1] != expected_hash:
                    raise AssertionError("Actual-segmentation base/learned cache differs from prepared base copy source")
                prompt = self.bb.tokenizer.apply_chat_template(state["messages"] + field["messages"],
                    tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
                if prompt[:len(state["tokens"])] != state["tokens"]: raise ValueError("Proof prompt changed native cache boundary")
                suffix = prompt[len(state["tokens"]):]
                if not 1 <= len(suffix) <= 256: raise ValueError("Complete proof suffix exceeds256")
                hidden = model.model(mx.array([suffix]), cache=state["cache"])
                value = model.language_model.lm_head(hidden[:, -1:])
                mx.eval(value, [c.state for c in state["cache"]]); mx.synchronize(self.stream); self.bb.check_memory()
                if not bool(mx.all(mx.isfinite(value)).item()): raise FloatingPointError("Nonfinite proof logits")
            states.append(state); logits.append(value); prompt_hashes.append(token_hash(prompt))
        hashes = [state_hash(s["cache"]) for s in states]
        effect = float(mx.max(mx.abs(logits[0].astype(mx.float32) - logits[1].astype(mx.float32))).item())
        receipt["cacheProof"].update(promptStateHashes=hashes, promptHashes=prompt_hashes, maxAbsLogitEffect=effect)
        write_progress(directory / "gate.json", receipt)
        if hashes[0] != hashes[1] or prompt_hashes[0] != prompt_hashes[1] or not effect > 0:
            raise AssertionError("Nonzero final-copy adapter changed native cache or had no logit effect")
        continuations = []
        for state in states:
            values = []
            for token in forced:
                value = model(mx.array([[token]]), cache=state["cache"])[:, -1]
                mx.eval(value, [c.state for c in state["cache"]]); mx.synchronize(self.stream); self.bb.check_memory()
                if not bool(mx.all(mx.isfinite(value)).item()): raise FloatingPointError("Nonfinite forced continuation")
                values.append(value)
            continuations.append((values, state_hash(state["cache"])))
        exact = continuations[0][1] == continuations[1][1] and all(bool(mx.array_equal(a, b).item())
            for a, b in zip(continuations[0][0], continuations[1][0], strict=True))
        gate = {"sourceMode": "frozen-base", "copyMode": self.adapter_sha,
            "prefixSha256": selected["copyState"]["prefixSha256"], "adapterNonzero": effect > 0,
            "completeCacheStatesExact": source_hashes[0] == source_hashes[1] == expected_hash and hashes[0] == hashes[1],
            "forcedContinuationLogitsExact": exact}
        receipt["cacheProof"].update(compatibilityGate=gate, continuationStateHashes=[v[1] for v in continuations])
        write_progress(directory / "gate.json", receipt)
        if not exact: raise AssertionError("New-prefix base-cache/learned-copy forced continuation differs")
        return gate

    def control(self, rid, slot):
        from .fused_prefix import assert_feature_compatibility, focused_query_text
        from .native_batch_diagnosis import state_hash
        from .semantic_copy_training import disabled
        directory = self.out / ("control-" + str(slot["index"])); directory.mkdir()
        slot["artifactDirectory"] = str(directory)
        begin = time.perf_counter(); selected = self.selection(rid)
        with disabled(self.model, self.adapter):
            baseline, baseline_timing = self.bb.encode([focused_query_text(selected["case"]["request"], selected["case"]["catalog"])])
        gate = feature_gate(baseline, selected["rawQuery"], selected["raw"])
        receipt = {"status": "running", "recordId": rid, "featureGate": gate,
            "baselineTiming": baseline_timing, "selectionTiming": selected["timing"]}
        torch.save({"baselineCacheNone": baseline, "cachedRaw": selected["rawQuery"]}, directory / "raw-query-vectors.pt")
        write_progress(directory / "gate.json", receipt)
        if gate["cacheVsNoCacheFeaturesExact"] is not True:
            raise AssertionError("Strict cached-vs-cacheNone raw feature gate failed; unchanged v2 head cannot be reused")
        state, common_timing = self._common(selected); selected["copyState"] = state
        source_hash = state_hash(state["cache"])
        cache_gate = self._cache_proof(selected, source_hash, directory, receipt)
        assert_feature_compatibility(state, "frozen-base", self.adapter_sha, feature_gate=gate, copy_cache_gate=cache_gate)
        # Check every actual serving suffix before any timed request starts.
        inventory = []
        for row in selected["rows"]:
            full = self.bb.tokenizer.apply_chat_template(state["messages"] + row["messages"],
                tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
            if full[:len(state["tokens"])] != state["tokens"] or not 1 <= len(full) - len(state["tokens"]) <= 256 or len(full) + row["maxTokens"] > 4096:
                raise ValueError("Actual selected-field token boundary/cap mismatch")
            inventory.append({"rowId": row["rowId"], "suffixTokens": len(full) - len(state["tokens"]),
                              "maxTokens": row["maxTokens"], "promptSha256": digest(canonical(full))})
        receipt.update(status="exact-controls-passed", cacheAdapterExact=True,
            rawFeatureEqual=True, cacheGate=cache_gate, commonTiming=common_timing,
            rawFeatureSha256=tensor_digest(selected["rawQuery"]), fullImageFeatureSha256=tensor_digest(selected["fullQuery"]),
            rawCacheSha256=state_hash(selected["raw"]["cache"]), copyCacheSha256=source_hash,
            planSha256=digest(canonical(selected["case"]["pendingPagePlan"])), copyPrefixSha256=state["prefixSha256"],
            commonSha256=digest(canonical(selected["common"])), inventory=inventory,
            controlWallMs=(time.perf_counter() - begin) * 1000, memory=self.bb.runtime_memory())
        write_progress(directory / "gate.json", receipt)
        del selected, state, baseline
        gc.collect(); self.mx.clear_cache(); self.mx.synchronize(self.stream); self.bb.check_memory()
        return receipt

    def request(self, rid, gate, arm):
        from .short_copy_fit import decode_complete_rows, assemble_section, encoded_payload
        from .fused_prefix import assert_feature_compatibility
        from .native_batch_diagnosis import state_hash
        begin = time.perf_counter(); arm["stage"] = "warm-catalog/fresh-queries/heads/plan"
        selected = self.selection(rid)
        if (tensor_digest(selected["rawQuery"]) != gate["rawFeatureSha256"] or tensor_digest(selected["fullQuery"]) != gate["fullImageFeatureSha256"]
                or digest(canonical(selected["case"]["pendingPagePlan"])) != gate["planSha256"]
                or digest(canonical(selected["common"])) != gate["commonSha256"]):
            raise AssertionError("Fresh timed features/actual trained plan differ from proven startup conditions")
        metadata_ready = (time.perf_counter() - begin) * 1000
        state, common_timing = self._common(selected)
        if state["prefixSha256"] != gate["copyPrefixSha256"]: raise AssertionError("Timed exact copy prefix differs")
        assert_feature_compatibility(state, "frozen-base", self.adapter_sha,
            feature_gate=gate["featureGate"], copy_cache_gate=gate["cacheGate"])
        verification_start = time.perf_counter()
        before_raw = state_hash(selected["raw"]["cache"]); before_copy = state_hash(state["cache"])
        if before_raw != gate["rawCacheSha256"] or before_copy != gate["copyCacheSha256"]:
            raise AssertionError("Fresh complete native source states differ from exact startup controls")
        before_verify_ms = (time.perf_counter() - verification_start) * 1000
        arm.update(stage="independent-field-decode", actualDecision=selected["decision"],
            actualPagePlan=selected["case"]["pendingPagePlan"], trainedScores=selected["scored"], publicEvidence=selected["evidence"])
        admitted = (time.perf_counter() - begin) * 1000
        def first_ready(rows):
            first = assemble_section(selected["case"], rows, 0); encoded = encoded_payload(first)
            arm["firstSection"] = first
            arm["firstReceipt"] = {"payloadSha256": first["payloadSha256"], "payloadBytes": len(encoded),
                "validatedAndSerializedAtMs": (time.perf_counter() - begin) * 1000,
                "metadataReadyAtMs": metadata_ready, "copyAdmittedAtMs": admitted,
                "scope": "actual fresh queries+verified108cataloghits+heads+plan+common+first10decode+validation+serialization",
                "browserPaintMeasured": False}
        try:
            rows, decode_timing = decode_complete_rows(self.bb, state, selected["rows"], 6, first_ready)
            arm["rawRows"] = rows; arm["stage"] = "full-six-validation-serialization"
            second = assemble_section(selected["case"], rows, 1); encoded_payload(second)
            payload = {"schemaVersion": VERSION + "/six-card-payload",
                       "sections": [arm["firstSection"]["payload"], second["payload"]]}
            encoded = canonical(payload).encode()
            valid = arm["firstSection"]["validation"]["valid"] and second["validation"]["valid"]
            verification_start = time.perf_counter()
            after_raw = state_hash(selected["raw"]["cache"]); after_copy = state_hash(state["cache"])
            after_verify_ms = (time.perf_counter() - verification_start) * 1000
            if (after_raw, after_copy) != (before_raw, before_copy):
                raise AssertionError("Actual native decode mutated pristine complete raw/common source")
            arm.update(status="completed-attempt", stage="completed", secondSection=second, fullPayload=payload,
                fullValidation={"valid": valid, "requiredFields": 20, "requiredCards": 6,
                                "sections": [arm["firstSection"]["validation"], second["validation"]]},
                fullReceipt={"payloadSha256": hashlib.sha256(encoded).hexdigest(), "payloadBytes": len(encoded),
                    "validatedAndSerializedAtMs": (time.perf_counter() - begin) * 1000},
                timing={**selected["timing"], "commonPreparation": common_timing, "copy": decode_timing,
                        "completeSourceHashBeforeDecodeMs": before_verify_ms, "completeSourceHashAfterDecodeMs": after_verify_ms},
                completeSourceStates={"rawBefore": before_raw, "rawAfter": after_raw,
                                      "commonBefore": before_copy, "commonAfter": after_copy},
                actualOutputTokens=sum(len(r["tokenIds"]) for r in rows), actualArSteps=sum(r["arSteps"] for r in rows),
                featureGateReusedAsCertificateOnly=True, freshQueryNativeForwards=2, nativeCatalogForwards=0,
                sampledCapFirst=208, sampledCapFull=416, memory=self.bb.runtime_memory())
            if state["consumed"] or selected["raw"]["consumed"]: raise AssertionError("Immutable source marked consumed")
            self.bb.check_memory()
        except Exception as exc:
            arm.update(rawRows=getattr(exc, "partial_rows", arm.get("rawRows", [])),
                       failedAtMs=(time.perf_counter() - begin) * 1000)
            raise
        finally:
            del selected, state
            gc.collect(); self.mx.clear_cache(); self.mx.synchronize(self.stream)

    def close(self):
        from .semantic_copy_training import parameter_hashes
        try:
            if getattr(self, "failed", False):
                return  # Failure already blocks promotion; do no further native tensor work.
            final = {k.replace(".31.mlp.down_proj.linear.", ".31.mlp.down_proj."): v
                     for k, v in parameter_hashes(self.model.language_model).items()
                     if not k.endswith((".lora_a", ".lora_b"))}
            if final != self.base_hashes or self._adapter_receipts() != self.adapter_tensors:
                raise AssertionError("Frozen language or copy adapter bytes changed")
            self.cold["finalIntegrity"] = {"frozenLanguageUnchanged": True, "languageTensors": len(final),
                "frozenLanguageHashesSha256": digest(canonical(final)), "adapterTensorsUnchanged": True,
                "adapterTensorHashes": self.adapter_tensors}
        finally:
            self.mx.synchronize(self.stream); self.context.__exit__(None, None, None)


def write_progress(path, value):
    """Only this new attempt's mutable progress file; old attempts are refused."""
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def evaluate_fixed(arms, labels, pairs):
    """Post-freeze synthetic quality, with strict first3/full6 qualification."""
    if len(arms) != 6 or len(pairs) != 3:
        raise ValueError("Fixed six-request/three-pair evaluator required")
    observations = {}; correct = first_images = full_images = qualified_first = qualified_full = 0
    for arm in arms:
        if arm["status"] != "completed-attempt": continue
        truth = labels[arm["recordId"]]
        cards = [c for s in arm["actualPagePlan"]["sections"] for c in s["products"]]
        if len(cards) != 6 or len({c["productId"] for c in cards}) != 6:
            raise ValueError("A completed result must contain all six distinct cards")
        product_correct = cards[0]["productId"] == truth["bestProductId"]
        images = [truth["presentationScores"][c["productId"]][c["imageCaptionId"]] == 3 for c in cards]
        first_valid = arm["firstSection"]["validation"]["valid"]
        full_valid = arm["fullValidation"]["valid"]
        correct += product_correct; first_images += sum(images[:3]); full_images += sum(images)
        qualified_first += bool(product_correct and all(images[:3]) and first_valid)
        qualified_full += bool(product_correct and all(images) and full_valid)
        observations[arm["recordId"]] = {"topProductId": cards[0]["productId"], "uniqueGoldProductCorrect": product_correct,
            "selectedImageTeacherCorrect": images, "selectedFirst3ImageTeacherCorrect": sum(images[:3]),
            "selectedFull6ImageTeacherCorrect": sum(images), "qualifiedFirst3": bool(product_correct and all(images[:3]) and first_valid),
            "qualifiedFull6": bool(product_correct and all(images) and full_valid)}
    paired = []
    for pair in pairs:
        selected = [observations.get(rid) for rid in pair["recordIds"]]
        complete = all(v is not None for v in selected)
        paired.append({"recordIds": pair["recordIds"], "bothAttempted": complete,
            "productChoiceChanged": bool(complete and selected[0]["topProductId"] != selected[1]["topProductId"]),
            "bothCorrect": bool(complete and all(v["uniqueGoldProductCorrect"] for v in selected))})
    return {"fixedRequests": 6, "fixedProductPairs": 3, "topProductGoldCorrect": correct,
        "selectedFirst3ImagesCorrect": first_images, "fixedFirst3ImageSlots": 18,
        "selectedFull6ImagesCorrect": full_images, "fixedFull6ImageSlots": 36,
        "teacherQualifiedFirst3Results": qualified_first, "teacherQualifiedFull6Results": qualified_full,
        "allSixTeacherQualifiedFirst3": qualified_first == 6, "allSixTeacherQualifiedFull6": qualified_full == 6,
        "pairs": paired, "cases": observations,
        "qualification": "complete unchanged copy validation AND correct unique top product AND all selected synthetic image gold choices",
        "evaluationScope": "inspected synthetic replay; image head remains imperfect; not a real-user metric"}


def run(packet, out, *, maximum_requests=6, backend_factory=NativeBackend):
    """All six controls gate even a bounded one-request timing attempt."""
    if type(maximum_requests) is not int or maximum_requests not in (1, 6):
        raise ValueError("One bounded request or all fixed six, never a width/search sweep")
    out = Path(out)
    if out.exists(): raise FileExistsError("Preserve previous fused result attempt")
    out.mkdir(parents=True)
    arms = roster(packet["requestIds"])
    controls = [{"index": i, "recordId": rid, "status": "not-attempted"} for i, rid in enumerate(packet["requestIds"])]
    report = {"version": VERSION, "status": "started", "phase": "resident-setup", "limits": LIMITS,
        "fixedRequests": 6, "maximumTimedRequestsThisAttempt": maximum_requests, "arms": arms,
        "controls": controls, "trainingPerformed": False, "limitations": packet.get("limitations", []),
        "packetSha256": digest(canonical(packet))}
    started = time.perf_counter(); backend = None
    def save():
        report.update(wallMs=(time.perf_counter() - started) * 1000, summary=summary(arms))
        write_progress(out / "results.json", report)
    save(); write_new(out / "packet.json", packet)
    try:
        backend = backend_factory(packet, out); report["coldSetup"] = backend.cold
        report["phase"] = "all-six-startup-controls"; save()
        control_begin = time.perf_counter()
        for slot in controls:
            slot["status"] = "running"; save()
            try:
                gate = backend.control(slot["recordId"], slot)
                if gate.get("status") != "exact-controls-passed" or gate.get("rawFeatureEqual") is not True or gate.get("cacheAdapterExact") is not True:
                    raise AssertionError("Every exact startup gate must pass before any timed request")
                slot.update(status="passed", receipt=gate); save()
            except Exception:
                slot["status"] = "failed"
                if "artifactDirectory" in slot:
                    path = Path(slot["artifactDirectory"]) / "gate.json"
                    if path.exists():
                        gate = json.loads(path.read_text()); gate["status"] = "failed-strict-control"; write_progress(path, gate)
                save(); raise
        report["startupControlWallMs"] = (time.perf_counter() - control_begin) * 1000
        report["phase"] = "timed-fresh-requests"; save()
        for index, arm in enumerate(arms[:maximum_requests]):
            arm["status"] = "running"; save()
            try:
                backend.request(arm["recordId"], controls[index]["receipt"], arm)
                if arm["status"] != "completed-attempt": raise AssertionError("Request ended without complete attempt receipt")
            except Exception as exc:
                arm.update(status="failed", errorType=type(exc).__name__, error=str(exc),
                    rawRows=getattr(exc, "partial_rows", arm.get("rawRows", [])))
                save(); raise
            save()
        report.update(status="completed-six-attempts" if maximum_requests == 6 else "partial-one-request-attempt", phase="completed")
        write_new(out / "decisions-and-copy.json", arms)
        write_new(out / "TIMED_RESULTS_FROZEN.json", {"packetSha256": report["packetSha256"],
            "resultsSha256": file_digest(out / "decisions-and-copy.json"), "fixedRequests": 6})
        if "dataset" in packet:
            labels = {r["recordId"]: r for r in map(json.loads,
                (Path(packet["dataset"]) / "private/truth.jsonl").read_text().splitlines()) if r}
            report["quality"] = evaluate_fixed(arms, labels, packet["pairs"])
            report["allSixFusedResultsQualified"] = report["quality"]["allSixTeacherQualifiedFull6"]
            report["privateEvaluationAfterTimedResultsFreeze"] = True
    except Exception as exc:
        if backend is not None: backend.failed = True
        if "control_begin" in locals() and "startupControlWallMs" not in report:
            report["startupControlWallMs"] = (time.perf_counter() - control_begin) * 1000
        report.update(status="blocked-strict-controls" if report["phase"] != "timed-fresh-requests" else "partial-failed-benchmark",
                      blocker={"type": type(exc).__name__, "message": str(exc)})
        save(); raise
    finally:
        try:
            if backend is not None: backend.close()
        except Exception as exc:
            report.update(status="failed-final-integrity", allSixFusedResultsQualified=False,
                          blocker={"type": type(exc).__name__, "message": str(exc)})
            save(); raise
        finally: save()
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("prepare", "run")); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--packet", type=Path); parser.add_argument("--reviewed-packet-sha256")
    parser.add_argument("--copy-report", type=Path, default=COPY_REPORT)
    parser.add_argument("--maximum-requests", type=int, choices=(1, 6), default=6)
    parser.add_argument("--native-owner", choices=("root",)); parser.add_argument("--confirm-serialized", action="store_true")
    args = parser.parse_args(); torch.set_num_threads(1)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, local_files_only=True, trust_remote_code=False)
    packet = prepare_packet(tokenizer, copy_report=args.copy_report)
    if args.action == "prepare":
        if args.out.exists(): raise FileExistsError("Preserve previous preflight packet")
        write_new(args.out, packet); print(canonical({"packetSha256": file_digest(args.out), "fixedRequests": 6})); return
    if (args.native_owner != "root" or not args.confirm_serialized or args.packet is None
            or file_digest(args.packet) != args.reviewed_packet_sha256 or json.loads(args.packet.read_text()) != packet):
        raise ValueError("Exact reviewed packet and serialized root ownership required")
    from .catalog_fusion.generalization_v2 import native_owner_lock
    with native_owner_lock(Path("data/ranker/pretrained/.native-generalization-v2.lock"), args.native_owner):
        run(packet, args.out, maximum_requests=args.maximum_requests)


if __name__ == "__main__": main()
