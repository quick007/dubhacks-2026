"""Synthetic CPU tests of paired supervision, read-only reuse and fit ordering."""
import contextlib
import copy
import io
import itertools
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from .inputs import digest
from .paired_contract import validate_data_for_review, verify_freeze
from .paired_data import ATOMS, STYLES, freeze_dataset, teacher
from .paired_runner import prepare, run_prepared
from .preflight import public_corpus
from .runtime import FeatureStore, write_json
from .test_catalog_fusion import fixture, normalized
from .test_runner import CpuEncoderStub, provenance, token_record


def source_dataset(root):
    (root / "public").mkdir(parents=True)
    request, _ = fixture()
    products = []
    for split in ("train", "validation", "test"):
        ids = []
        for i, tags in enumerate(itertools.product(("durable", "laptop-sleeve", "lightweight", "water-resistant"), ("compact", "roomy", "organizer"))):
            pid = f"old-{split}-product-{i}"
            ids.append(pid)
            products.append({"id": pid, "title": f"Public {split} pack {i}", "category": "backpacks", "description": "Public synthetic facts",
                             "price": 30 + i, "shippingCost": 5, "inStock": True, "tags": list(tags),
                             "presentationVariants": [{"id": f"{pid}-{style}", "title": "Public pack image", "imageDescription": style,
                                                       "presentationStyle": style, "facts": list(tags), "compatibleSurfaces": ["detail-panel"]} for style in STYLES]})
        record = copy.deepcopy(request)
        record["recordId"] = f"old-{split}-request"
        record["candidateProductIds"] = ids
        (root / "public" / f"{split}.jsonl").write_text(json.dumps(record) + "\n")
    (root / "public/diagnostics.jsonl").write_text("")
    (root / "catalog.json").write_text(json.dumps({"products": products}))
    # Data generation may not depend on any old teacher/private labels.
    (root / "labels").mkdir()
    (root / "labels/train.jsonl").write_text("unparseable old label data")


class TokenizerStub:
    def encode(self, text, add_special_tokens=False):
        return types.SimpleNamespace(ids=list(bytes.fromhex(digest(text))))


class PairedTests(unittest.TestCase):
    def test_data_freeze_has_controlled_unique_counterfactuals_and_all_atom_positives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dataset(root / "source")
            receipt = freeze_dataset(root / "source", root / "paired")
            audit = validate_data_for_review(root / "paired")
            self.assertEqual(audit["requests"], 48)
            self.assertEqual(audit["families"], 12)
            self.assertEqual(audit["pairs"], 48)
            self.assertEqual(receipt["positiveTrainingAtomQueryCounts"], {atom: 4 for atom in ATOMS})
            self.assertEqual(receipt["nativeCatalogForwardsRequired"], 0)
            self.assertEqual((root / "source/catalog.json").read_bytes(), (root / "paired/catalog.json").read_bytes())

    def test_public_only_receipt_never_opens_heldout_target_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dataset(root / "source")
            freeze_dataset(root / "source", root / "paired")
            original = Path.open

            def guarded_open(path, *args, **kwargs):
                if path.name in ("validation.jsonl", "test.jsonl") and path.parent.name == "labels":
                    self.fail("Pre-fit public receipt read held-out targets")
                if path.name == "paired-controls.json":
                    self.fail("Pre-fit public receipt read target-bearing pair controls")
                return original(path, *args, **kwargs)

            with patch.object(Path, "open", guarded_open):
                verify_freeze(root / "paired", public_only=True)

    def test_teacher_refuses_factual_ties_instead_of_using_opaque_ids(self):
        request, catalog = fixture(2)
        request["context"]["presentationSurface"] = "detail-panel"
        for product in catalog.values():
            for variant, style in zip(product["presentationVariants"], STYLES, strict=True):
                variant["compatibleSurfaces"] = ["detail-panel"]
                variant["presentationStyle"] = style
        request["context"]["presentationPreferences"] = [STYLES[0]]
        with self.assertRaisesRegex(ValueError, "unique factual winner"):
            teacher(request, catalog)

    def test_full_cpu_stub_pipeline_reuses_catalog_and_reads_targets_only_after_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dataset(root / "source")
            freeze_dataset(root / "source", root / "paired")
            prior = root / "prior"
            prior.mkdir()
            rows, _, hashes = public_corpus(root / "source")
            previous = {"records": [token_record(row) for row in rows], "inputHashes": hashes}
            write_json(prior / "preflight.json", previous)
            write_json(prior / "model-verification.json", provenance())
            write_json(prior / "report.json", {"status": "completed", "scope": "CPU test stub, not native evidence"})
            old_store = FeatureStore(prior / "features", provenance())
            for row in rows:
                if row["role"] == "catalog":
                    full = {**row, **token_record(row)}
                    old_store.save(full, normalized(1, int(digest(row["text"])[:8], 16)), {"encodeTotalMs": .01})
            prepared = prepare(root / "paired", prior, TokenizerStub())
            out = root / "fit"
            out.mkdir()
            encoders = []
            original = Path.open

            def checked_open(path, *args, **kwargs):
                if path.parent.resolve() == (root / "paired/labels").resolve() and path.name in ("validation.jsonl", "test.jsonl"):
                    self.assertTrue((out / "CHECKPOINTS_FROZEN.json").exists())
                return original(path, *args, **kwargs)

            def factory():
                if encoders:
                    self.assertTrue((out / "CHECKPOINTS_FROZEN.json").exists())
                value = CpuEncoderStub()
                encoders.append(value)
                return value

            with patch.object(Path, "open", checked_open), contextlib.redirect_stdout(io.StringIO()):
                report = run_prepared(out, root / "paired", prepared, factory)
            self.assertEqual(report["nativeQueryForwards"], 48)
            self.assertEqual(report["catalogReuse"]["nativeForwards"], 0)
            self.assertTrue(all(role == "request" for encoder in encoders for _, role in encoder.calls))
            self.assertEqual([len(e.calls) for e in encoders], [28, 20])
            self.assertTrue(all(e.closed for e in encoders))
            self.assertEqual(list((out / "features/catalog").glob("*.pt")), [])
            for split, count in (("train", 14), ("validation", 4), ("test", 6)):
                metrics = report["counterfactualPairs"][split]["prior"]["guarded"]
                self.assertEqual(metrics["productPairs"], count)
                self.assertEqual(metrics["productPairBothCorrect"], 0)
                self.assertEqual(metrics["productChoiceChanges"], 0)
                self.assertEqual(metrics["allCandidateImagePairBothCorrect"], 0)
                self.assertEqual(metrics["allCandidateImagePairs"], count * 4)


if __name__ == "__main__":
    unittest.main()
