"""CPU stubs exercise orchestration; none of these are native quality evidence."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import torch

from . import DIMENSION
from .evaluate import evaluation_labels, evaluate_split
from .fit_runner import FIT_CONFIG, prior_control, require_frozen
from .inputs import canonical, digest
from .preflight import REVISION, SPLITS, file_digest, public_corpus
from .runner import corpus_rows, extract_stage, run_prepared
from .runtime import FeatureStore, NativeEncoder, assert_model_files, guarded_native_memory, verify_model
from .scorer import FusionScorer
from .test_catalog_fusion import fixture, normalized


def provenance():
    return {"evidenceSource": "cpu-test-stub", "modelRevision": REVISION,
            "weightsSha256": "a" * 64, "tokenizerSha256": "b" * 64,
            "templateSha256": "c" * 64, "quantization": {"bits": 4}}


def token_record(row):
    # Deterministic test tokens only, never substituted in the real CLI.
    ids = list(hashlib.sha256(row["text"].encode()).digest())
    return {key: value for key, value in row.items() if key != "text"} | {
        "textSha256": digest(row["text"]), "tokenIds": ids, "tokenSha256": digest(canonical(ids)),
        "tokens": len(ids), "batchShape": [1, len(ids)]}


class CpuEncoderStub:
    evidence_source = "cpu-test-stub"
    load_seconds = 0.

    def __init__(self):
        self.calls = []
        self.closed = False

    def encode(self, text, expected):
        self.calls.append((text, expected["role"]))
        seed = int(digest(text)[:8], 16)
        return normalized(1, seed), {"batchRows": 1, "paddedTokensPerRow": expected["tokens"],
                                    "truncatedRows": 0, "backboneForwardCalls": 1, "encodeTotalMs": .01}

    def memory(self):
        return {"scope": "cpu-test-stub; no native allocation"}

    def close(self):
        self.closed = True


def synthetic_dataset(root):
    request, catalog = fixture(2)
    pids = list(catalog)
    for i, product in enumerate(catalog.values()):
        product["title"] = ("Pine 18L Daypack", "River 24L Daypack")[i]
        product["description"] = "Distinct public test facts " + str(i)
    (root / "public").mkdir()
    (root / "labels").mkdir()
    (root / "private").mkdir()
    (root / "catalog.json").write_text(json.dumps({"products": list(catalog.values())}))
    all_labels = []
    for split in SPLITS:
        records, targets = [], []
        for i in range(2):
            row = copy.deepcopy(request)
            row["recordId"] = f"{split}-{i}"
            row["context"]["task"] = f"Public task for option {i}"
            row["context"]["presentationPreferences"] = ["detail" if i else "in-use"]
            records.append(row)
            target = {"recordId": row["recordId"], "bestProductId": pids[i],
                      "relevance": {pid: 3 if j == i else 1 for j, pid in enumerate(pids)},
                      "presentationScores": {pid: {v["id"]: 3 if j == i else 1 for j, v in enumerate(p["presentationVariants"])} for pid, p in catalog.items()},
                      "privateTruth": "must never enter a feature", "hiddenConstraints": {"avoidedTags": ["secret"]}}
            if split == "diagnostics":
                target["diagnosticPairId"] = "test-pair"
            targets.append(target)
        (root / "public" / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        if split in ("train", "validation"):
            (root / "labels" / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in targets))
        if split == "diagnostics":
            (root / "private/diagnostic-truth.jsonl").write_text("".join(json.dumps(r) + "\n" for r in targets))
        else:
            all_labels.extend(targets)
    (root / "private/truth.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_labels))
    rows, counts, hashes = public_corpus(root)
    return {"records": [token_record(row) for row in rows], "inputHashes": hashes, "splits": counts}


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complete_pipeline_freezes_before_private_reads_and_marks_cpu_stub(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            preflight = synthetic_dataset(dataset)
            out = root / "output"
            out.mkdir()
            encoders, private_reads = [], []
            original = Path.read_text

            def checked_read(path, *args, **kwargs):
                if path.parent.resolve() == (dataset / "private").resolve():
                    self.assertTrue((out / "CHECKPOINTS_FROZEN.json").exists())
                    require_frozen(out)
                    private_reads.append(path.name)
                return original(path, *args, **kwargs)

            def factory():
                if encoders:
                    require_frozen(out)
                encoder = CpuEncoderStub()
                encoders.append(encoder)
                return encoder

            with patch.object(Path, "read_text", checked_read), contextlib.redirect_stdout(io.StringIO()):
                report = run_prepared(out, dataset, preflight, provenance(), factory)
            self.assertEqual(report["evidenceSource"], "cpu-test-stub")
            self.assertFalse(report["generatesCopy"])
            self.assertEqual(report["nativeForwards"], 12)  # four variants + eight distinct fresh requests
            self.assertEqual(report["phases"]["train"]["forwards"], 6)
            self.assertEqual(private_reads, ["truth.jsonl", "diagnostic-truth.jsonl"])
            self.assertTrue(all(e.closed for e in encoders))
            self.assertEqual(len(encoders), 2)
            self.assertEqual(sum(len(e.calls) for e in encoders), 12)
            self.assertEqual(report["fits"]["fusion"]["trainableParameters"], 32786)
            self.assertEqual(report["fits"]["prior"]["trainableParameters"], 16394)
            self.assertEqual(report["fits"]["prior"]["parameterAbsoluteChange"]["U"], 0)
            self.assertGreater(report["fits"]["fusion"]["parameterAbsoluteChange"]["interaction"], 0)
            for arm in ("fusion", "prior"):
                self.assertTrue(report["fits"][arm]["exactTrainingPredictionReload"])
                self.assertEqual(len((out / f"{arm}-epochs.jsonl").read_text().splitlines()), FIT_CONFIG["epochs"])
            for split in ("validation", "test", "diagnostics"):
                observations = report["phases"][split]
                self.assertEqual(observations["forwards"], 2)
                quality = report["quality"][split]
                self.assertEqual(set(quality), {"random", "prior", "fusion", "permuted-query"})
                self.assertEqual(quality["fusion"]["guarded"]["allCandidatePresentation"]["incompatible"], 0)
            query_texts = [text for encoder in encoders for text, role in encoder.calls if role == "request"]
            self.assertTrue(all("privateTruth" not in text and "secret" not in text for text in query_texts))
            self.assertEqual(len(list((out / "features/request").glob("*.pt"))), 8)
            checkpoint = out / "fusion.pt"
            checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "Frozen checkpoint changed"):
                evaluation_labels(dataset, "test", {"test-0": {}}, out)

    def test_feature_store_rejects_stale_inputs_corruption_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeatureStore(Path(directory) / "features", provenance())
            row = {"role": "catalog", "member": ["external-product", "external-image"], "text": "Public selected facts"}
            row |= token_record(row)
            x = normalized(1)
            loaded, _ = store.save(row, x, {"encodeTotalMs": 1.})
            self.assertTrue(torch.equal(x, loaded))
            with self.assertRaises(FileExistsError):
                store.save(row, x, {})
            changed = {**row, "text": "Changed price/public facts"}
            changed |= token_record(changed)
            with self.assertRaisesRegex(ValueError, "Stale"):
                store.load(changed)
            data, _ = store.paths(row["role"], row["member"])
            data.write_bytes(data.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                store.load(row)

    def test_evaluator_cannot_open_targets_before_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError) as error:
                evaluation_labels(root / "missing-dataset", "test", {"r": {}}, root)
            self.assertIn("CHECKPOINTS_FROZEN", str(error.exception))

    def test_real_adapter_refuses_tokenizer_mismatch_before_forward(self):
        adapter = object.__new__(NativeEncoder)
        calls = []
        adapter.backbone = types.SimpleNamespace(tokenizer=types.SimpleNamespace(encode=lambda *a, **k: [1, 9]),
                                                 encode=lambda rows: calls.append(rows), check_memory=lambda: None,
                                                 runtime_memory=lambda: {"mlxActiveBytes": 1, "mlxPeakBytes": 1, "processBudgetBytes": 6 * 2 ** 30})
        with self.assertRaisesRegex(ValueError, "tokenizer IDs"):
            adapter.encode("facts", {"tokenIds": [1, 2], "tokenSha256": digest(canonical([1, 2]))})
        self.assertEqual(calls, [])

    def test_extraction_budget_stops_before_native_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = FeatureStore(root / "features", provenance())
            encoder = CpuEncoderStub()
            with self.assertRaisesRegex(TimeoutError, "budget"):
                extract_stage("train", {}, {}, {}, store, encoder, root, lambda *a, **k: None,
                              run_started=time.perf_counter() - 1000, phase_started=time.perf_counter())
            self.assertEqual(encoder.calls, [])

    def test_native_guard_rejects_peak_over_budget_even_with_low_active_memory(self):
        budget = 6 * 2 ** 30
        backbone = types.SimpleNamespace(check_memory=lambda: None,
                                         runtime_memory=lambda: {"mlxActiveBytes": budget // 2, "mlxPeakBytes": budget + 1,
                                                                 "processBudgetBytes": budget})
        with self.assertRaisesRegex(MemoryError, "observed peak"):
            guarded_native_memory(backbone)
        backbone.runtime_memory = lambda: {"mlxActiveBytes": budget + 1, "mlxPeakBytes": budget + 1, "processBudgetBytes": budget}
        with self.assertRaises(MemoryError):
            guarded_native_memory(backbone)

    def test_pinned_shard_hashes_and_reload_file_identity_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / REVISION
            model.mkdir()
            for name in ("weights.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "chat_template.jinja", "config.json"):
                (model / name).write_bytes(("cpu fixture " + name).encode())
            preflight = {"modelPath": str(model.resolve()), "tokenizerSha256": file_digest(model / "tokenizer.json"),
                         "chatTemplateSha256": file_digest(model / "chat_template.jinja"),
                         "configSha256": file_digest(model / "config.json"), "quantization": {"bits": 4}}
            with patch("experiments.ranker.pretrained.catalog_fusion.runtime.SHARDS", {"weights.safetensors": file_digest(model / "weights.safetensors")}):
                receipt = verify_model(model, preflight)
                assert_model_files(model, receipt)
                (model / "weights.safetensors").write_bytes(b"changed fixture")
                with self.assertRaisesRegex(ValueError, "changed before load"):
                    assert_model_files(model, receipt)
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    verify_model(model, preflight)

    def test_prior_control_is_query_invariant_and_presentation_masks_apply(self):
        request, catalog = fixture(2)
        prior = prior_control()
        q, x = normalized(1)[0], normalized(4, seed=11)
        self.assertTrue(torch.equal(prior(q, x), prior(-q, x)))
        labels = {"r": {"bestProductId": request["candidateProductIds"][0],
                         "relevance": {pid: 3 - i for i, pid in enumerate(catalog)},
                         "presentationScores": {pid: {v["id"]: i + 1 for i, v in enumerate(p["presentationVariants"])} for pid, p in catalog.items()}}}
        for pid, product in catalog.items():
            request["presentationCompatibility"][pid][product["presentationVariants"][1]["id"]] = False
        models = {"prior": prior, "fusion": FusionScorer(), "random": FusionScorer()}
        result = evaluate_split(models, {"r": request}, labels, catalog, {"r": q}, {"r": x})
        for arm in result["cases"]:
            images = result["cases"][arm]["r"]["guarded"]["presentationActions"]
            for pid in catalog:
                self.assertEqual(images[pid], catalog[pid]["presentationVariants"][0]["id"])
            self.assertEqual(result["summary"][arm]["guarded"]["allCandidatePresentation"]["informative"], 0)


if __name__ == "__main__":
    unittest.main()
