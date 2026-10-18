"""Fast CPU contracts only; no genuine native extraction or project fitting."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import torch

from .generalization_v2 import (ATOMS, MODES, SPLITS, Factors, catalog_text_v2, evaluate,
    freeze_data, holdout_catalog_sets, load_head, native_owner_lock, query_text_v2,
    read_labels, read_public, run, save_head, signatures, teacher, train_inputs)
from .inputs import canonical
from .runner import source_hashes
from .test_catalog_fusion import normalized
from .test_runner import CpuEncoderStub, provenance, token_record


class FakeStore:
    """Synthetic normalized vectors, never asserted to be native features."""
    def __init__(self, train_only=False):self.calls = [];self.train_only = train_only
    def load(self, row):
        if self.train_only and row["split"] != "train":raise AssertionError("Held-out vector read by training")
        self.calls.append((row["mode"], row["split"], row["role"], row["id"]))
        return normalized(1, int(row["textSha256"][:8], 16)), {"scope": "CPU synthetic fixture"}


def fixture(root):
    dataset = root / "data";frozen = freeze_data(dataset)
    _, catalog, requests, metadata = read_public(dataset)
    rows = []
    for mode in MODES:
        for split in SPLITS:
            for pid in metadata["productSplits"][split]:
                text = catalog_text_v2(catalog[pid], mode)
                r = {"role": "catalog", "member": [mode, pid], "mode": mode, "split": split, "id": pid, "text": text}
                rows.append({**token_record(r), "text": text})
            for rid, request in requests[split].items():
                text = query_text_v2(request, catalog, mode)
                r = {"role": "request", "member": [mode, rid], "mode": mode, "split": split, "id": rid, "text": text}
                rows.append({**token_record(r), "text": text})
    packet = {"sourceHashes": source_hashes(), "dataFreezeSha256": "fixture", "scope": "CPU stub only"}
    return dataset, (packet, frozen, catalog, requests, metadata, rows, provenance())


class GeneralizationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_balanced_all_atom_cardinalities_and_partitioned_combinations(self):
        spec = signatures()
        for split in SPLITS:
            for cardinality in (1, 2, 3):
                sets = spec[split][cardinality]
                self.assertEqual(len(sets), 7)
                for atom in ATOMS:self.assertEqual(sum(atom in p for p in sets), cardinality)
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertEqual(set(spec[a][1]), set(spec[b][1]))
            for k in (2, 3):self.assertFalse(set(spec[a][k]) & set(spec[b][k]))
        for split in ("validation", "test"):
            values = holdout_catalog_sets(spec[split], split)
            self.assertEqual([sum(len(t) == k for t in values) for k in (1, 2, 3)], [2, 2, 2])
            self.assertEqual({a for tags in values for a in tags}, set(ATOMS))

    def test_identity_full_text_disjoint_and_focused_singleton_overlap_disclosed(self):
        with tempfile.TemporaryDirectory() as temp:
            _, prepared = fixture(Path(temp));_, frozen, catalog, requests, metadata, rows, _ = prepared
            self.assertEqual(len(rows), 192)
            for role in ("catalog", "request"):
                self.assertEqual(set(frozen["encoderTextOverlap"]["full"][role].values()), {0})
            self.assertEqual(set(frozen["encoderTextOverlap"]["focused"]["request"].values()), {7})
            self.assertEqual(frozen["encoderTextOverlap"]["focused"]["catalog"]["train/validation"], 2)
            self.assertEqual(frozen["encoderTextOverlap"]["focused"]["catalog"]["train/test"], 2)
            self.assertEqual(frozen["encoderTextOverlap"]["focused"]["catalog"]["validation/test"], 0)
            self.assertEqual(len({p["title"] for p in catalog.values()}), 33)
            for row in rows:
                self.assertNotIn(row["id"], row["text"])
                if row["mode"] == "focused":
                    self.assertNotIn("bestProduct", row["text"]);self.assertNotIn("onehot", row["text"])
            self.assertEqual([len(metadata["productSplits"][s]) for s in SPLITS], [21, 6, 6])

    def test_nuisance_mutations_leave_focused_facts_unchanged_but_not_full_text(self):
        with tempfile.TemporaryDirectory() as temp:
            _, prepared = fixture(Path(temp));_, _, catalog, requests, _, _, _ = prepared
            product = next(iter(catalog.values()));changed = {**product, "title": "A different name", "description": "Other prose", "price": 99}
            self.assertEqual(catalog_text_v2(product, "focused"), catalog_text_v2(changed, "focused"))
            self.assertNotEqual(catalog_text_v2(product, "full"), catalog_text_v2(changed, "full"))
            q = next(iter(requests["train"].values()));changedq = {**q, "recordId": "opaque", "context": {**q["context"], "task": "Other wording"}}
            self.assertEqual(query_text_v2(q, catalog, "focused"), query_text_v2(changedq, catalog, "focused"))
            self.assertNotEqual(query_text_v2(q, catalog, "full"), query_text_v2(changedq, catalog, "full"))

    def test_training_reads_only_train_vectors_and_means_reload_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);_, prepared = fixture(root);_, _, cat, req, meta, rows, _ = prepared
            store = FakeStore(train_only=True);q, c, qt, ct, means, contract = train_inputs("focused", cat, req, meta, store, rows)
            self.assertEqual(len(store.calls), 42);self.assertTrue(all(call[1] == "train" for call in store.calls))
            self.assertTrue(torch.equal(means["query"], q.double().mean(0).float()))
            self.assertTrue(torch.equal(means["catalog"], c.double().mean(0).float()))
            self.assertEqual(qt.sum(0).tolist(), [6.] * 7);self.assertEqual(ct.sum(0).tolist(), [6.] * 7)
            model = Factors(means, qt.mean(0));ql, cl = model.logits(q, c)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(ql, qt) + torch.nn.functional.binary_cross_entropy_with_logits(cl, ct)
            loss.backward();self.assertGreater(model.query_weight.grad.norm().item(), 0);self.assertGreater(model.catalog_weight.grad.norm().item(), 0)
            # One synthetic gradient perturbation tests reload of changed weights;
            # this is not a fit on actual native/project features.
            with torch.no_grad():
                for parameter in model.parameters():parameter -= .01 * parameter.grad
            path = root / "head.pt";sha = save_head(model, path, contract);restored = load_head(path, sha, contract)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.probabilities(q, c), restored.probabilities(q, c), strict=True)))
            self.assertEqual(sum(p.numel() for p in model.parameters()), 57358)
            with self.assertRaises(FileExistsError):save_head(model, path, contract)
            wrong = dict(contract);wrong["mode"] = "full"
            with self.assertRaises(ValueError):load_head(path, sha, wrong)

    def test_prior_query_independence_and_tie_aware_public_oracle(self):
        with tempfile.TemporaryDirectory() as temp:
            _, prepared = fixture(Path(temp));_, _, cat, req, meta, rows, _ = prepared;store = FakeStore()
            q, c, qt, _, means, _ = train_inputs("full", cat, req, meta, store, rows)
            models = {a: Factors(means, qt.mean(0), a) for a in ("fusion", "prior", "random")}
            self.assertTrue(torch.equal(models["prior"].probabilities(q[0], c)[0], models["prior"].probabilities(q[-1], c)[0]))
            labels = {rid: teacher(r, cat) for rid, r in req["validation"].items()}
            results = evaluate("full", "validation", models, cat, req["validation"], meta, store, rows, labels, [])
            self.assertEqual(results["summary"]["public-oracle"]["bestSetCorrect"], 21)
            self.assertEqual(results["summary"]["public-oracle"]["meanNdcg3"], 1)
            for case in results["cases"]["fusion"].values():self.assertEqual(len(case["scores"]), 6);self.assertEqual(len(case["catalogProbabilities"]), 6)

    def test_private_labels_fail_closed_before_freeze_and_owner_lock_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);dataset, prepared = fixture(root);out = root / "out";out.mkdir()
            original = Path.open
            def guard(path, *args, **kwargs):
                if path.resolve().is_relative_to((dataset / "private").resolve()):self.fail("Early private target read")
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", guard):
                with self.assertRaises(FileNotFoundError):read_labels(dataset, out, prepared[0], prepared[3])
            with self.assertRaises(ValueError):
                with native_owner_lock(root / "lock", "worker"):pass
            with native_owner_lock(root / "lock", "root"):
                with self.assertRaises(RuntimeError):
                    with native_owner_lock(root / "lock", "root"):pass

    def test_cpu_stub_extraction_192_rows_stops_before_any_fitting_or_private_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);dataset, prepared = fixture(root);out = root / "out";out.mkdir()
            (out / "packet.json").write_text(canonical(prepared[0]));encoder = CpuEncoderStub()
            original = Path.open
            def guard(path, *args, **kwargs):
                if path.resolve().is_relative_to((dataset / "private").resolve()):self.fail("Private labels before heads freeze")
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", guard), \
                 patch("experiments.ranker.pretrained.catalog_fusion.generalization_v2.fit", side_effect=RuntimeError("CPU test stops before fitting")), \
                 patch("torch.optim.AdamW", side_effect=AssertionError("Optimizer forbidden in stub extraction test")), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "stops before fitting"):
                    run(out, dataset, prepared, lambda: encoder, started=time.perf_counter())
            self.assertEqual(len(encoder.calls), 192);self.assertTrue(encoder.closed)
            self.assertEqual(sum(role == "catalog" for _, role in encoder.calls), 66)
            self.assertEqual(sum(role == "request" for _, role in encoder.calls), 126)
            self.assertTrue((out / "RAW_FEATURES_FROZEN.json").exists())
            self.assertFalse((out / "HEADS_FROZEN.json").exists())


if __name__ == "__main__":unittest.main()
