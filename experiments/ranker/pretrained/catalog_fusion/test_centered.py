"""Synthetic CPU centering contracts; no genuine project fit is run here."""
import copy
from pathlib import Path
import tempfile
import types
import unittest

import torch

from .centered import (CENTER_CONFIG, CenteredFusionScorer, center_rows, compute_statistics,
                       load_centered, save_centered, statistics_digest)
from .centered_runner import fit_centered, freeze_centered, prepare_plan, require_centered_frozen
from .scorer import FusionScorer, load_checkpoint, roster, save_checkpoint
from .test_catalog_fusion import fixture, normalized


def synthetic_training():
    request, catalog = fixture(2)
    requests = {f"train-{i}": {**copy.deepcopy(request), "recordId": f"train-{i}"} for i in range(2)}
    q = normalized(2, 91)
    queries = {rid: q[i] for i, rid in enumerate(requests)}
    x = normalized(4, 73)
    actions = roster(request, catalog)
    items = {a: x[i] for i, a in enumerate(actions)}
    features = {rid: torch.stack([items[a] for a in roster(r, catalog)]) for rid, r in requests.items()}
    labels = {rid: {"bestProductId": request["candidateProductIds"][i],
                    "presentationScores": {pid: {v["id"]: 3 if j == i else 1 for j, v in enumerate(p["presentationVariants"])} for pid, p in catalog.items()}}
              for i, rid in enumerate(requests)}
    return requests, catalog, queries, items, features, labels


class CenteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_statistics_use_only_exact_train_members_and_unique_items(self):
        requests, catalog, queries, items, _, _ = synthetic_training()
        means, stats = compute_statistics(requests, catalog, queries, items, provenance={"scope": "CPU test"})
        self.assertTrue(torch.equal(means["query"], torch.stack([queries[k] for k in sorted(queries)]).double().mean(0).float()))
        self.assertTrue(torch.equal(means["catalog"], torch.stack([items[k] for k in sorted(items)]).double().mean(0).float()))
        self.assertEqual(stats["queryRows"], 2)
        self.assertEqual(stats["catalogRows"], 4)  # Not eight repeated per-query rows.
        with self.assertRaisesRegex(ValueError, "TRAIN-only"):
            compute_statistics(requests, catalog, {**queries, "heldout": normalized(1)[0]}, items, provenance={})
        with self.assertRaisesRegex(ValueError, "TRAIN-only"):
            compute_statistics(requests, catalog, queries, {**items, ("heldout", "variant"): normalized(1)[0]}, provenance={})

    def test_prepare_reads_only_train_vector_values(self):
        requests, catalog, queries, items, features, _ = synthetic_training()
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "public").mkdir()
            (root / "public/train.jsonl").write_text("public CPU training input\n")
            def load_split(split):
                calls.append(split)
                self.assertEqual(split, "train")
                return queries, features, items, {"queries": {}, "catalog": {}}
            cache = types.SimpleNamespace(dataset=root, requests={"train": requests, "validation": {"never-read": {}}},
                                          catalog=catalog, source_receipts={"scope": "CPU test"}, load_split=load_split)
            plan, _, stats, *_ = prepare_plan(cache)
            self.assertEqual(calls, ["train"])
            self.assertEqual(plan["statisticsSha256"], statistics_digest(stats))
            self.assertFalse(plan["fittingInPreparation"])

    def test_fixed_epsilon_is_finite_for_zero_residual_and_is_not_learned(self):
        zero = torch.zeros(4096)
        self.assertTrue(torch.equal(center_rows(zero, zero), zero))
        tiny = zero.clone()
        tiny[0] = 1e-10
        self.assertAlmostEqual(center_rows(tiny, zero)[0].item(), .01, places=6)
        self.assertEqual(CENTER_CONFIG["epsilon"], 1e-8)

    def test_distinct_checkpoint_exact_reload_and_frozen_means_after_heldout_prediction(self):
        requests, catalog, queries, items, features, _ = synthetic_training()
        means, stats = compute_statistics(requests, catalog, queries, items, provenance={"scope": "CPU test"})
        model = CenteredFusionScorer(means, stats)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 32786)
        self.assertEqual({name for name, _ in model.named_buffers()}, {"query_mean", "catalog_mean"})
        q_mean, c_mean = model.query_mean.clone(), model.catalog_mean.clone()
        rid = next(iter(requests))
        output = model(queries[rid], features[rid])
        (output[0, 0] - output[2, 0] + output[0, 1] - output[1, 1]).backward()
        self.assertGreater(model.U.grad.abs().sum().item(), 0)
        self.assertIsNone(model.query_mean.grad)
        with torch.no_grad():
            for p in model.parameters():
                p -= .1 * p.grad
        model(normalized(1, 555)[0], normalized(4, 999))
        self.assertTrue(torch.equal(model.query_mean, q_mean))
        self.assertTrue(torch.equal(model.catalog_mean, c_mean))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "centered.pt"
            sha = save_centered(model, path)
            reloaded = load_centered(path, sha, statistics_digest(stats))
            self.assertTrue(torch.equal(model(queries[rid], features[rid]), reloaded(queries[rid], features[rid])))
            with self.assertRaises(FileExistsError):
                save_centered(model, path)
            with self.assertRaisesRegex(ValueError, "statistics identity"):
                load_centered(path, sha, "f" * 64)
            with self.assertRaises(ValueError):
                load_checkpoint(path, sha)
            old = Path(directory) / "old.pt"
            old_sha = save_checkpoint(FusionScorer(), old)
            with self.assertRaisesRegex(ValueError, "old heads"):
                load_centered(old, old_sha, statistics_digest(stats))

    def test_fixed_cpu_stub_fits_freeze_statistics_and_reload_both_controls(self):
        requests, catalog, queries, items, features, labels = synthetic_training()
        means, stats = compute_statistics(requests, catalog, queries, items, provenance={"scope": "CPU test"})
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            reports = {}
            for arm in ("fusion", "prior"):
                _, reports[arm] = fit_centered(arm, out, means, stats, requests, labels, catalog, queries, features, items)
                self.assertTrue(reports[arm]["exactTrainingPredictionReload"])
            random = freeze_centered(out, reports, {}, means, stats)
            frozen = require_centered_frozen(out, statistics_digest(stats))
            self.assertEqual(reports["prior"]["trainableParameters"], 16394)
            self.assertEqual(reports["prior"]["parameterAbsoluteChange"]["U"], 0)
            self.assertEqual(frozen["statisticsSha256"], statistics_digest(stats))
            self.assertTrue(torch.equal(random.query_mean, means["query"]))


if __name__ == "__main__":
    unittest.main()
