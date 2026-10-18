"""Credential-free protocol tests; all example observations are synthetic fixtures."""
import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("shopping_model_input", Path(__file__).with_name("shopping_model_input.py"))
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def fixture(split="train", event=1):
    for number in range(1000):
        session = f"00000000-0000-4000-8000-{number:012x}"
        bucket = hashlib.sha256(("fieldwork.session-split.v1/" + session).encode()).digest()[0] % 10
        if ("validation" if bucket == 8 else "test" if bucket == 9 else "train") == split:
            break
    return {"schema": "fieldwork.shopping-choice.v1", "split": split,
            "provenance": {"sessionId": session, "batchId": "10000000-0000-4000-8000-000000000001",
                           "eventId": f"20000000-0000-4000-8000-{event:012x}",
                           "sourceSha256": "a" * 64, "catalogEventId": "30000000-0000-4000-8000-000000000001",
                           "catalogBatchId": "10000000-0000-4000-8000-000000000001", "catalogSourceSha256": "b" * 64,
                           "classification": {"environment": "test", "traffic": "synthetic"},
                           "receivedAt": "2026-10-18T00:00:00Z", "splitVersion": "fieldwork.session-split.v1",
                           "derivation": "explicit-positive-choice.v1"},
            "context": {"mode": "baseline", "order": "popular", "shipping": False, "modelVersion": "fixture-v1"},
            "candidates": [{"productId": "mug", "variantId": "mug-sage", "position": 0},
                           {"productId": "bottle", "variantId": "bottle-graphite", "position": 1}],
            "chosen": {"productId": "mug", "variantId": "mug-cobalt"},
            "choiceScope": "product-with-observed-variant-change", "signal": "selection",
            "negativeLabels": [], "realPurchaseObserved": False, "selectionMechanism": "observational-not-randomized"}


class ModelInputTests(unittest.TestCase):
    def test_default_excludes_synthetic_and_empty_real_packet_has_no_fake_training(self):
        packet, manifest = bridge.prepare([fixture()])
        self.assertEqual(packet["public-cases.json"], [])
        self.assertEqual(manifest["excludedNonProductionReal"], 1)
        self.assertFalse(manifest["queryObserved"])
        self.assertEqual(bridge.cpu_consumer_check(packet)["status"], "skipped-no-training-rows")

    def test_context_does_not_contain_target_candidate_or_model_identity(self):
        one, _ = bridge.prepare([fixture()], True)
        changed = fixture()
        changed["chosen"] = {"productId": "bottle", "variantId": "bottle-graphite"}
        changed["choiceScope"] = "rendered-variant"
        changed["context"]["modelVersion"] = "a-different-model"
        two, _ = bridge.prepare([changed], True)
        self.assertEqual(one["public-cases.json"], two["public-cases.json"])
        self.assertEqual(one["encoder-records.json"], two["encoder-records.json"])
        self.assertEqual(one["train-labels.json"][0]["targetProductIndex"], 0)
        self.assertEqual(two["train-labels.json"][0]["targetProductIndex"], 4)
        self.assertEqual(one["provenance.json"][0]["source"]["chosen"]["variantId"], "mug-cobalt")

    def test_actual_cpu_consumer_excludes_validation_and_test_sessions(self):
        rows = [fixture("train", 1), fixture("validation", 2), fixture("test", 3)]
        packet, manifest = bridge.prepare(rows, True)
        self.assertEqual(manifest["splitCounts"], {"train": 1, "validation": 1, "test": 1})
        base = bridge.model_helpers()
        with patch.object(base, "NativeEncoder", side_effect=AssertionError("Native encoding forbidden")):
            report = bridge.cpu_consumer_check(packet)
        self.assertEqual(report["queryShape"], [1, 4096])
        self.assertEqual(report["catalogShape"], [16, 4096])
        self.assertEqual(report["requestLoads"], 1)
        self.assertEqual(report["validationTestRequestsLoaded"], 0)
        self.assertFalse(report["trainingExecuted"])

    def test_all_sixteen_public_ids_map_to_exact_separate_targets(self):
        _, routing = bridge.catalogs()
        rows = []
        for route in routing:
            row = fixture(event=route["productIndex"] + 1)
            chosen = {"productId": route["productId"], "variantId": route["variantIds"][0]}
            row.update(chosen=chosen, candidates=[{**chosen, "position": 0}], choiceScope="rendered-variant")
            rows.append(row)
        packet, _ = bridge.prepare(rows, True)
        self.assertEqual([r["targetProductIndex"] for r in packet["train-labels.json"]], list(range(16)))

    def test_malformed_evidence_split_unknown_ids_negatives_and_duplicates_rejected(self):
        corruptions = [lambda r: r.update(split="test"),
                       lambda r: r["chosen"].update(variantId="unknown"),
                       lambda r: r.update(negativeLabels=["bottle"]),
                       lambda r: r.update(realPurchaseObserved=True),
                       lambda r: r.update(signal="cart-add"),
                       lambda r: r["context"].update(query="secret text"),
                       lambda r: r["candidates"].append(copy.deepcopy(r["candidates"][0])),
                       lambda r: r["provenance"].update(sourceSha256="missing")]
        for corrupt in corruptions:
            row = fixture()
            corrupt(row)
            with self.assertRaises(ValueError):
                bridge.prepare([row], True)
        with self.assertRaises(ValueError):
            bridge.prepare([fixture(), fixture()], True)
        with self.assertRaises(ValueError):
            bridge.strict_json('{"a":1,"a":2}')
        with self.assertRaises(ValueError):
            bridge.strict_json('{"a":NaN}')

    def test_only_actual_boolean_permits_synthetic(self):
        for value in ("false", "true", 0, 1, None, [], {}):
            with self.assertRaisesRegex(ValueError, "boolean"):
                bridge.prepare([fixture()], value)

    def test_reordered_or_corrupted_catalog_snapshot_cannot_relabel_targets(self):
        for filename in ("external_mapping.json", "model_inputs.json", "runtime_export.json"):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                for name in ("external_mapping.json", "model_inputs.json", "runtime_export.json", "IMPORT_RECEIPT.json"):
                    shutil.copyfile(bridge.FILES["mapping"].parent / name, directory / name)
                data = json.loads((directory / filename).read_text())
                data["products"][0], data["products"][1] = data["products"][1], data["products"][0]
                (directory / filename).write_text(json.dumps(data))
                # Even changing the per-file receipt cannot bypass independent pins.
                receipt = json.loads((directory / "IMPORT_RECEIPT.json").read_text())
                receipt["outputs"][filename]["canonicalSha256"] = bridge.digest(bridge.canonical(data).encode())
                (directory / "IMPORT_RECEIPT.json").write_text(json.dumps(receipt))
                with patch.dict(bridge.FILES, {"mapping": directory / "external_mapping.json"}):
                    with self.assertRaises(ValueError):
                        bridge.catalogs()

    def test_serialized_packet_consumer_readback_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "choices.jsonl"
            source.write_text(json.dumps(fixture()) + "\n")
            out = root / "packet"
            with redirect_stdout(io.StringIO()):
                bridge.main(["--input", str(source), "--out", str(out), "--allow-synthetic", "--cpu-consumer-check"])
            loaded = bridge.load_packet(out)
            self.assertEqual(bridge.cpu_consumer_check(loaded)["status"], "passed")
            self.assertEqual((out / "manifest.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual(out.stat().st_mode & 0o777, 0o700)
            with self.assertRaises(FileExistsError):
                bridge.main(["--input", str(source), "--out", str(out)])
            manifest = json.loads((out / "manifest.json").read_text())
            manifest["allowSynthetic"] = "false"
            (out / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "boolean"):
                bridge.load_packet(out)
            manifest["allowSynthetic"] = True
            (out / "manifest.json").write_text(json.dumps(manifest))
            (out / "train-labels.json").write_text("[]\n")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                bridge.load_packet(out)


if __name__ == "__main__":
    unittest.main()
