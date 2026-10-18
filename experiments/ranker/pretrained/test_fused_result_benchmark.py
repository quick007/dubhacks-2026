"""CPU-only integration contracts and failure accounting; no model runtime."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from . import fused_result_benchmark as fused


class FakeBackend:
    instances = []
    fail_control = None
    fail_request = None
    fail_close = False
    def __init__(self, packet, out):
        self.packet = packet; self.out = out; self.controls = []; self.requests = []; self.closed = False
        self.cold = {"modelResidentSetupMs": 1}; self.instances.append(self)
    def control(self, rid, slot):
        self.controls.append(rid)
        if rid == self.fail_control: raise AssertionError("strict feature/cache mismatch")
        return {"status": "exact-controls-passed", "rawFeatureEqual": True, "cacheAdapterExact": True}
    def request(self, rid, gate, arm):
        if len(self.controls) != 6: raise AssertionError("timing began before all six gates")
        self.requests.append(rid)
        if rid == self.fail_request:
            error = RuntimeError("native partial failure"); error.partial_rows = [{"text": "retained raw"}]; raise error
        valid = rid != "r5"
        arm.update(status="completed-attempt", firstSection={"validation": {"valid": valid}},
                   fullValidation={"valid": valid}, firstReceipt={"validatedAndSerializedAtMs": 100},
                   fullReceipt={"validatedAndSerializedAtMs": 150})
    def close(self):
        self.closed = True
        if self.fail_close: raise AssertionError("final immutable weights mismatch")


def packet(): return {"requestIds": ["r" + str(i) for i in range(6)], "limits": fused.LIMITS}


class FusedResultTests(unittest.TestCase):
    def setUp(self):
        FakeBackend.instances = []; FakeBackend.fail_control = FakeBackend.fail_request = None; FakeBackend.fail_close = False

    def test_complete_six_runs_after_all_gates_fixed_denominator_and_valid_latency(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "new"
            report = fused.run(packet(), out, backend_factory=FakeBackend)
            self.assertEqual(report["status"], "completed-six-attempts")
            self.assertEqual(report["summary"]["fixedRequests"], 6)
            self.assertEqual(report["summary"]["completed"], 6)
            self.assertEqual(report["summary"]["validSixCardResults"], 5)
            self.assertEqual(report["summary"]["validSixRateFixedDenominator"], 5 / 6)
            self.assertFalse(report["summary"]["allSixFullSixValid"])
            self.assertEqual(len(report["summary"]["validFullCompletionMs"]), 5)
            self.assertTrue(FakeBackend.instances[0].closed)
            self.assertTrue((out / "TIMED_RESULTS_FROZEN.json").is_file())

    def test_one_request_preserves_five_unattempted_and_all_six_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            result = fused.run(packet(), Path(temp) / "new", maximum_requests=1, backend_factory=FakeBackend)
            self.assertEqual(result["status"], "partial-one-request-attempt")
            self.assertEqual(result["summary"]["unattempted"], 5)
            self.assertEqual(result["summary"]["validSixRateFixedDenominator"], 1 / 6)
            self.assertEqual(len(FakeBackend.instances[0].controls), 6)
            self.assertEqual(len(FakeBackend.instances[0].requests), 1)

    def test_strict_control_failure_stops_every_timed_case(self):
        FakeBackend.fail_control = "r2"
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "new"
            with self.assertRaisesRegex(AssertionError, "strict feature"):
                fused.run(packet(), out, backend_factory=FakeBackend)
            result = json.loads((out / "results.json").read_text())
            self.assertEqual(result["status"], "blocked-strict-controls")
            self.assertEqual(result["summary"]["unattempted"], 6)
            self.assertEqual([s["status"] for s in result["controls"]], ["passed", "passed", "failed", "not-attempted", "not-attempted", "not-attempted"])
            self.assertEqual(FakeBackend.instances[0].requests, [])
            self.assertTrue(FakeBackend.instances[0].closed)

    def test_timed_failure_retains_partial_raw_and_fixed_unattempted_cases(self):
        FakeBackend.fail_request = "r1"
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "new"
            with self.assertRaisesRegex(RuntimeError, "partial failure"):
                fused.run(packet(), out, backend_factory=FakeBackend)
            result = json.loads((out / "results.json").read_text())
            self.assertEqual(result["summary"]["completed"], 1)
            self.assertEqual(result["summary"]["failed"], 1)
            self.assertEqual(result["summary"]["unattempted"], 4)
            self.assertEqual(result["arms"][1]["rawRows"][0]["text"], "retained raw")

    def test_final_integrity_failure_is_never_completed_success(self):
        FakeBackend.fail_close = True
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "new"
            with self.assertRaisesRegex(AssertionError, "immutable"):
                fused.run(packet(), out, backend_factory=FakeBackend)
            self.assertEqual(json.loads((out / "results.json").read_text())["status"], "failed-final-integrity")

    def test_no_overwrite_and_denominator_argument_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError): fused.run(packet(), temp, backend_factory=FakeBackend)
        for maximum in (True, 0, 2, 7):
            with self.assertRaises(ValueError): fused.run(packet(), "unused", maximum_requests=maximum, backend_factory=FakeBackend)
        with self.assertRaises(ValueError): fused.roster(["same"] * 6)
        with self.assertRaises(ValueError): fused.summary([])

    def test_predetermined_inspected_pair_selection_ignores_gold_fields(self):
        requests = {str(i): {} for i in range(8)}
        controls = {"pairs": [{"kind": "image", "recordIds": ["6", "7"]}] + [
            {"kind": "product", "recordIds": [str(i), str(i + 1)], "expectedBestProductIds": ["hidden", "labels"]}
            for i in (0, 2, 4, 6)]}
        ids, pairs = fused.select_six(requests, controls)
        self.assertEqual(ids, [str(i) for i in range(6)])
        self.assertEqual(len(pairs), 3)
        self.assertNotIn("expectedBestProductIds", str(pairs))
        controls["pairs"][2]["recordIds"][0] = "0"
        with self.assertRaises(ValueError): fused.select_six(requests, controls)

    def actual_public_case(self):
        from .boundary import guard_reasons
        from .catalog_fusion.scorer import roster
        catalog = {p["id"]: p for p in json.loads((fused.DATA / "catalog.json").read_text())["products"]}
        request = json.loads((fused.DATA / "public/requests.jsonl").read_text().splitlines()[0])
        order = request["candidateProductIds"]
        scores = torch.arange(36, dtype=torch.float32)
        for i, pid in enumerate(order):
            if guard_reasons(request, catalog[pid]): scores[i] = 1000
        images = torch.arange(len(roster(request, catalog)), dtype=torch.float32)
        return request, catalog, order, scores, images

    def test_actual36_pool_72_trained_score_bindings_guards_complete6_and20jobs(self):
        from .boundary import guard_reasons
        request, catalog, order, scores, images = self.actual_public_case()
        _, plan, record = fused.combine_plan(request, catalog, order, scores, images)
        cards = [c for section in plan["sections"] for c in section["products"]]
        self.assertEqual(len(cards), 6); self.assertEqual(len({c["productId"] for c in cards}), 6)
        self.assertTrue(all(not guard_reasons(request, catalog[c["productId"]]) for c in cards))
        self.assertEqual(len(record["actions"]), 72)
        for action, pair in zip(record["actions"], record["scores"], strict=True):
            self.assertEqual(pair[0], scores[order.index(action["productId"])] .item())
        _, _, common, rows, evidence = fused.layout_case(request, catalog, plan)
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(r["maxTokens"] for r in rows[:10]), 208)
        self.assertEqual(sum(r["maxTokens"] for r in rows), 416)
        self.assertEqual(len(evidence["cards"]), 6)
        self.assertTrue(all("targetValue" not in r and r["key"] == r["rowId"] for r in rows))
        self.assertEqual(common["role"], "user")

    def test_no_partial_grid_or_nonfinite_score_fallback(self):
        request, catalog, order, scores, images = self.actual_public_case()
        bad = copy.deepcopy(catalog)
        for product in bad.values(): product["inStock"] = False
        with self.assertRaisesRegex(ValueError, "Six eligible"):
            fused.combine_plan(request, bad, order, scores, images)
        scores[0] = float("nan")
        with self.assertRaises(ValueError): fused.combine_plan(request, catalog, order, scores, images)
        with self.assertRaises(ValueError): fused.combine_plan(request, catalog, order, torch.zeros(36), images[:-1])

    def test_exact_feature_gate_preserves_numerical_mismatch(self):
        values = torch.zeros(1, 4096); values[0, 0] = 1
        state = {"featureTokenSha256": "actual-token-hash"}
        self.assertTrue(fused.feature_gate(values, values.clone(), state)["cacheVsNoCacheFeaturesExact"])
        changed = values.clone(); changed[0, 1] = 1e-8
        result = fused.feature_gate(values, changed, state)
        self.assertFalse(result["cacheVsNoCacheFeaturesExact"])
        self.assertGreater(result["maxAbsDelta"], 0)
        self.assertNotEqual(result["baselineSha256"], result["cachedSha256"])
        self.assertEqual(values[0, 1].item(), 0)

    def copy_receipt(self, root):
        from .short_copy_fit import CONFIG, SOURCES
        model = root / "adapter.safetensors"; model.write_bytes(b"test checkpoint bytes, not a native model")
        parameters = {"lora_a": {"sha256": "a"}, "lora_b": {"sha256": "b"}}
        value = {"status": "completed", "phase": "completed", "nativeUpdates": 240, "optimizerUpdateCalls": 240,
            "extraction": [None] * 240, "steps": [{"step": i, "featureIndex": i, "status": "completed", "updateEvaluated": True} for i in range(240)], "languageParameterTensorsHashed": 927,
            "config": CONFIG, "preflight": {"sources": {n: fused.file_digest(fused.ROOT / n) for n in SOURCES}},
            "controls": {**{k: True for k in ("frozenLanguageHashesUnchanged", "learnedOriginalArchivedNativeParityExact",
                "sourceFilesUnchanged", "datasetManifestUnchanged", "all240NonzeroBGatesPassed", "fitStartedFromExactWarmStart")}, "new-C": {"status": "exact-controls-passed"}},
            "checkpoint": {"exact": True, "sha256": fused.file_digest(model),
                "parametersBeforeReload": parameters, "parametersAfterReload": parameters}}
        report = root / "results.json"; report.write_text(json.dumps(value)); return report, value

    def test_completed_final_checkpoint_required_no_intermediate_adapter(self):
        with tempfile.TemporaryDirectory() as temp:
            path, report = self.copy_receipt(Path(temp))
            self.assertEqual(fused.completed_copy(path)["nativeUpdates"], 240)
            for key, changed in (("status", "started"), ("nativeUpdates", 239), ("optimizerUpdateCalls", True)):
                bad = {**report, key: changed}; path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): fused.completed_copy(path)
            path.write_text(json.dumps(report)); (path.parent / "adapter.safetensors").write_bytes(b"changed")
            with self.assertRaises(ValueError): fused.completed_copy(path)

    def test_strict_teacher_qualification_separate_from_copy_validity(self):
        arms = []; labels = {}
        for i in range(6):
            rid = f"r{i}"; cards = [{"productId": f"p{i}-{j}", "imageCaptionId": f"v{j}"} for j in range(6)]
            arms.append({"recordId": rid, "status": "completed-attempt", "actualPagePlan": {"sections": [{"products": cards[:3]}, {"products": cards[3:]}]},
                "firstSection": {"validation": {"valid": True}}, "fullValidation": {"valid": True}})
            labels[rid] = {"bestProductId": cards[0]["productId"],
                "presentationScores": {c["productId"]: {c["imageCaptionId"]: 3} for c in cards}}
        pairs = [{"recordIds": [f"r{i}", f"r{i + 1}"]} for i in (0, 2, 4)]
        labels["r0"]["presentationScores"]["p0-5"]["v5"] = 0
        result = fused.evaluate_fixed(arms, labels, pairs)
        self.assertEqual(result["teacherQualifiedFirst3Results"], 6)
        self.assertEqual(result["teacherQualifiedFull6Results"], 5)
        self.assertFalse(result["allSixTeacherQualifiedFull6"])
        self.assertEqual(result["fixedFull6ImageSlots"], 36)
        self.assertTrue(all(p["productChoiceChanged"] and p["bothCorrect"] for p in result["pairs"]))


if __name__ == "__main__": unittest.main()
