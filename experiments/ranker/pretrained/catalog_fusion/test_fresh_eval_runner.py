"""CPU stub-only fresh evaluator integration; never native quality evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from .centered import CenteredFusionScorer, save_centered, statistics_digest
from .fresh_eval_data import freeze_dataset
from .fresh_eval_runner import (DENOMINATORS, extract, failure_report, models_from_source,
                                opposite_preferences, public_inputs, read_private_after_freeze,
                                run_prepared)
from .inputs import canonical, catalog_text, request_text
from .preflight import file_digest
from .runner import source_hashes
from .semantic_factors import POLICY, SemanticFactors, SemanticScorer, save_factors
from .semantic_runner import SCHEDULE
from .test_runner import CpuEncoderStub, provenance, token_record
from .test_semantic_factors import toy


def prepared_fixture(root):
    _, _, _, _, _, means, stats, qt, _, receipt = toy()
    source = root / "source";source.mkdir()
    image = CenteredFusionScorer(means, stats)
    image_sha = save_centered(image, source / "centered-image.pt")
    metadata = {"policy": POLICY, "statistics": stats, "statisticsSha256": statistics_digest(stats),
                "trainingPublicTargets": receipt, "queryFrequency": qt.mean(0).tolist(), "factorParameters": 57358,
                "imageCheckpoint": {"file": "centered-image.pt", "sha256": image_sha, "source": str(source / "centered-image.pt")}}
    checks = {}
    for arm in ("fusion", "prior", "random"):
        model = SemanticFactors(means, stats, qt.mean(0), arm=arm)
        sha = save_factors(model, source / f"{arm}.pt", metadata)
        checks[arm] = {"file": f"{arm}.pt", "sha256": sha, "trained": False}
    # Valid-format but explicitly untrained CPU fixtures, never genuine results.
    (source / "CHECKPOINTS_FROZEN.json").write_text(json.dumps({"version": POLICY["version"], "config": SCHEDULE,
        "metadata": metadata, "checkpoints": checks, "syntheticUntrainedCpuTestOnly": True}))
    (source / "report.json").write_text(json.dumps({"status": "completed", "metadata": metadata, "scope": "CPU stub fixture only"}))
    old = root / "old.json";old.write_text('{"products": []}')
    dataset = root / "dataset";frozen = freeze_dataset(dataset, source, old)
    _, catalog, requests = public_inputs(dataset)
    texts = [{"role": "catalog", "member": [pid, v["id"]], "text": catalog_text(p, v)}
             for pid, p in catalog.items() for v in p["presentationVariants"]]
    texts += [{"role": "request", "member": rid, "text": request_text(q, catalog)} for rid, q in requests.items()]
    rows = {(r["role"], canonical(r["member"])): {**token_record(r), "text": r["text"]} for r in texts}
    packet = {"metadata": metadata, "freezeSha256": file_digest(dataset / "FREEZE.json"),
              "sourceHashes": source_hashes(), "sourceModels": frozen["sourceCheckpoints"], "scope": "CPU stub only"}
    return dataset, source, (packet, catalog, requests, rows, provenance())


class FreshEvaluationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_full_cpu_stub_96_forward_pipeline_freezes_before_private_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);dataset, source, prepared = prepared_fixture(root)
            out = root / "out";out.mkdir()
            (out / "fresh-preflight.json").write_text(json.dumps(prepared[0]))
            encoder = CpuEncoderStub();reads = []
            original = Path.open
            def guard(path, *args, **kwargs):
                if path.resolve().is_relative_to((dataset / "private").resolve()):
                    self.assertTrue((out / "FEATURES_PREDICTIONS_FROZEN.json").exists())
                    self.assertEqual(len(list((out / "features/catalog").glob("*.pt"))), 72)
                    self.assertEqual(len(list((out / "features/request").glob("*.pt"))), 24)
                    reads.append(path.name)
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", guard), \
                 patch("experiments.ranker.pretrained.catalog_fusion.fresh_eval_runner.NativeEncoder", side_effect=AssertionError("Native disabled in CPU test")), \
                 patch("torch.optim.AdamW", side_effect=AssertionError("No optimizer allowed")), \
                 contextlib.redirect_stdout(io.StringIO()):
                report = run_prepared(out, dataset, source, prepared, lambda: encoder)
            self.assertEqual(report["evidenceSource"], "cpu-test-stub")
            self.assertEqual(len(encoder.calls), 96);self.assertTrue(encoder.closed)
            self.assertEqual(sum(role == "catalog" for _, role in encoder.calls), 72)
            self.assertEqual(sum(role == "request" for _, role in encoder.calls), 24)
            self.assertEqual(report["fixedDenominators"], DENOMINATORS)
            self.assertEqual(report["fits"], 0);self.assertFalse(report["meansUpdated"])
            self.assertTrue(report["exactModelPredictionReload"])
            self.assertEqual(set(reads), {"truth.jsonl", "controls.json"})
            self.assertEqual(report["quality"]["fusion"]["gridCases"], 144)
            self.assertEqual(report["quality"]["fusion"]["validGridComponents"], 144)
            for obs in report["extraction"]["freshQueryWarmCatalogObservations"]:
                self.assertEqual(obs["catalogHits"], 72);self.assertEqual(obs["freshQueryForwards"], 1)
                self.assertIn("freshQueryNativeEncodeMs", obs);self.assertIn("warmCatalogHashLoadVerifyMs", obs)
            self.assertEqual((out / "centered-image.pt").read_bytes(), (source / "centered-image.pt").read_bytes())

    def test_private_targets_unavailable_without_completed_feature_prediction_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);dataset, _, prepared = prepared_fixture(root)
            out = root / "out";out.mkdir()
            original = Path.open
            def guard(path, *args, **kwargs):
                if path.resolve().is_relative_to((dataset / "private").resolve()):self.fail("Early private target read")
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", guard):
                with self.assertRaises(FileNotFoundError):read_private_after_freeze(out, dataset, prepared)

    def test_public_counterpart_swap_is_bijective_opposite_preferences_same_style(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, prepared = prepared_fixture(Path(temp));requests = prepared[2]
            mapping = opposite_preferences(requests)
            self.assertEqual(set(mapping.values()), set(requests))
            for rid, other in mapping.items():
                self.assertEqual(mapping[other], rid)
                self.assertNotEqual(requests[rid]["context"]["statedPreferences"], requests[other]["context"]["statedPreferences"])
                self.assertEqual(requests[rid]["context"]["presentationPreferences"], requests[other]["context"]["presentationPreferences"])

    def test_malformed_batched_forward_fails_without_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);_, _, prepared = prepared_fixture(root)
            out = root / "out";out.mkdir()
            class Bad(CpuEncoderStub):
                def encode(self, text, expected):
                    value, timing = super().encode(text, expected);timing["batchRows"] = 2
                    return value, timing
            with self.assertRaisesRegex(ValueError, "whole-input B1"):
                extract(out, prepared, Bad(), {}, started=time.perf_counter())
            self.assertFalse((out / "FEATURES_PREDICTIONS_FROZEN.json").exists())

    def test_failure_keeps_fixed_denominators_and_partial_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "native-observations.jsonl").write_text(json.dumps({"event": "cold-catalog-forward"}) + "\n")
            result = failure_report(out, ValueError("CPU test failure"), time.perf_counter())
            self.assertEqual(result["fixedDenominators"]["requests"], 24)
            self.assertEqual(result["unfulfilledGridCasesPerArm"], 144)
            self.assertEqual(result["successfulFullGridComponents"], 0)
            self.assertEqual(result["recordedCatalogForwards"], 1)

    def test_changed_frozen_semantic_model_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset, source, _ = prepared_fixture(Path(temp))
            frozen = json.loads((dataset / "FREEZE.json").read_text())
            models_from_source(frozen, source)
            with (source / "fusion.pt").open("ab") as handle:handle.write(b"changed")
            with self.assertRaises(ValueError):models_from_source(frozen, source)


if __name__ == "__main__":unittest.main()
