"""Fast synthetic CPU contracts only; no genuine project fitting/native calls."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from .centered import CenteredFusionScorer, compute_statistics, save_centered, statistics_digest
from .fit_runner import FIT_CONFIG
from .paired_data import ATOMS
from .preflight import file_digest
from .scorer import load_checkpoint, roster
from .semantic_factors import (FACTOR_PARAMETERS, POLICY, SemanticFactors, SemanticScorer,
                               atom_metrics, distillation_loss, load_factors, public_targets, save_factors)
from .semantic_runner import freeze, prepare, require_semantic_frozen, run
from .test_catalog_fusion import fixture, normalized


def toy():
    request, catalog = fixture(7)
    pids = list(catalog)
    for i, pid in enumerate(pids):
        catalog[pid]["tags"] = [ATOMS[i]]
    requests = {f"train-{i}": {**copy.deepcopy(request), "recordId": f"train-{i}"} for i in range(7)}
    for i, req in enumerate(requests.values()):
        req["context"]["statedPreferences"] = [ATOMS[i]]
    q = {rid: row for rid, row in zip(requests, normalized(7, 911), strict=True)}
    members = roster(request, catalog)
    items = {a: row for a, row in zip(members, normalized(14, 713), strict=True)}
    x = {rid: torch.stack([items[a] for a in roster(r, catalog)]) for rid, r in requests.items()}
    means, stats = compute_statistics(requests, catalog, q, items, provenance={"test": "synthetic TRAIN only"})
    qt, ct, receipt = public_targets(requests, catalog, sorted(q), sorted(items))
    return requests, catalog, q, items, x, means, stats, qt, ct, receipt


def fake_image_source(directory, means, stats):
    directory.mkdir()
    model = CenteredFusionScorer(means, stats)
    checkpoints = {}
    for arm in ("fusion", "prior", "random"):
        path = directory / f"{arm}.pt"
        checkpoints[arm] = {"file": path.name, "sha256": save_centered(model, path), "trained": False}
    (directory / "CHECKPOINTS_FROZEN.json").write_text(json.dumps({"config": FIT_CONFIG, "checkpoints": checkpoints,
                         "statisticsSha256": statistics_digest(stats), "syntheticUntrainedTestOnly": True}))


class SemanticFactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_public_targets_exact_train_members_unique_catalog_and_atom_coverage(self):
        req, cat, q, items, _, _, _, qt, ct, receipt = toy()
        self.assertEqual(qt.shape, (7, 7));self.assertEqual(ct.shape, (14, 7))
        self.assertEqual(receipt["queryAtomPositiveCounts"], [1.] * 7)
        self.assertEqual(receipt["catalogAtomPositiveCounts"], [2.] * 7)
        self.assertFalse(receipt["usesRankingLabels"])
        # Extra held-out rows and repeated variants cannot enter TRAIN targets.
        with self.assertRaises(ValueError):
            public_targets(req, cat, sorted(q) + ["heldout"], sorted(items))
        with self.assertRaises(ValueError):
            public_targets(req, cat, sorted(q), sorted(items) + [sorted(items)[0]])
        req["train-0"]["context"]["statedPreferences"] = ["opaque-hidden-target"]
        with self.assertRaises(ValueError):
            public_targets(req, cat, sorted(q), sorted(items))

    def test_prepare_never_loads_heldout_features_or_opens_any_label_file(self):
        req, cat, q, items, x, means, stats, *_ = toy()
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "image"
            fake_image_source(source, means, stats)
            calls = []
            def load(split):
                calls.append(split)
                self.assertEqual(split, "train")
                return q, x, items, {}
            cache = types.SimpleNamespace(requests={"train": req, "validation": {"heldout": {}}}, catalog=cat,
                                          source_receipts={}, load_split=load)
            original = Path.read_text
            def guard(path, *args, **kwargs):
                if "labels" in path.parts or "paired-controls.json" in path.parts:
                    raise AssertionError("Pre-freeze target/control access")
                return original(path, *args, **kwargs)
            with patch.object(Path, "read_text", guard):
                prepared = prepare(cache, source)
            self.assertEqual(calls, ["train"])
            self.assertEqual(prepared[0]["rankingLabelFilesOpenedForTraining"], [])
            self.assertEqual(prepared[0]["metadata"]["statistics"], stats)
            self.assertEqual(prepared[0]["metadata"]["factorParameters"], 57358)

    def test_equal_role_bce_genuine_gradients_and_query_independent_prior(self):
        _, _, q, items, _, means, stats, qt, ct, _ = toy()
        qr, cr = torch.stack(list(q.values())), torch.stack([items[k] for k in sorted(items)])
        for arm in ("fusion", "prior"):
            model = SemanticFactors(means, stats, qt.mean(0), arm=arm)
            loss, roles = distillation_loss(model, qr, cr, qt, ct)
            self.assertAlmostEqual(loss.item(), roles["catalogBce"] + (roles["queryBce"] if arm == "fusion" else 0), places=6)
            loss.backward()
            self.assertGreater(model.catalog_weight.grad.norm().item(), 0)
            if arm == "fusion":
                self.assertGreater(model.query_weight.grad.norm().item(), 0)
            else:
                self.assertIsNone(model.query_weight.grad)
                self.assertTrue(torch.equal(model.product_scores(qr[0], cr), model.product_scores(qr[-1], cr)))
            self.assertIsNone(model.query_mean.grad)
            self.assertEqual(sum(p.numel() for p in model.parameters()), FACTOR_PARAMETERS)

    def test_frozen_image_exact_identity_and_factor_checkpoint_reload_fail_closed(self):
        req, cat, q, items, x, means, stats, qt, _, _ = toy()
        with tempfile.TemporaryDirectory() as temp:
            source, out = Path(temp) / "image", Path(temp) / "new"
            fake_image_source(source, means, stats);out.mkdir()
            cache = types.SimpleNamespace(requests={"train": req}, catalog=cat, source_receipts={},
                                          load_split=lambda _: (q, x, items, {}))
            prepared = prepare(cache, source);metadata = prepared[0]["metadata"]
            reports = {}
            for arm in ("fusion", "prior"):
                model = SemanticFactors(means, stats, qt.mean(0), arm=arm)
                sha = save_factors(model, out / f"{arm}.pt", metadata)
                restored = load_factors(out / f"{arm}.pt", sha, metadata)
                self.assertTrue(torch.equal(model.product_scores(q["train-0"], x["train-0"]), restored.product_scores(q["train-0"], x["train-0"])))
                with self.assertRaises(FileExistsError):save_factors(model, out / f"{arm}.pt", metadata)
                with self.assertRaises(ValueError):load_checkpoint(out / f"{arm}.pt", sha)
                wrong = copy.deepcopy(metadata);wrong["policy"]["vocabulary"] = list(reversed(ATOMS))
                with self.assertRaises(ValueError):load_factors(out / f"{arm}.pt", sha, wrong)
                reports[arm] = {"sha256": sha, "trained": False, "exactTrainingPredictionReload": True}
            # An incomplete fit receipt must fail; seal a synthetic manifest only
            # to exercise byte identity/reload, never as a genuine quality run.
            with self.assertRaises(ValueError):freeze(out, prepared, {}, reports)
            for report in reports.values():report["trained"] = True
            models = freeze(out, prepared, {}, reports)
            self.assertEqual((out / "centered-image.pt").read_bytes(), (source / "fusion.pt").read_bytes())
            self.assertEqual(set(models), {"fusion", "prior", "random"})
            (out / "centered-image.pt").write_bytes(b"corrupt image")
            with self.assertRaises(ValueError):require_semantic_frozen(out, metadata)

    def test_missing_freeze_stops_before_heldout_labels_or_features(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):require_semantic_frozen(temp, {})
            cache = types.SimpleNamespace(load_split=lambda _: self.fail("Feature read before freeze"))
            prepared = ({"metadata": {}, "fit": FIT_CONFIG}, None, None, {}, {}, {}, None, None)
            with patch("experiments.ranker.pretrained.catalog_fusion.semantic_runner.fit_arm", return_value=(None, {})), \
                 patch("experiments.ranker.pretrained.catalog_fusion.semantic_runner.freeze", side_effect=ValueError("freeze denied")), \
                 patch("experiments.ranker.pretrained.catalog_fusion.semantic_runner.read_semantic_labels", side_effect=AssertionError("early labels")):
                with self.assertRaisesRegex(ValueError, "freeze denied"):
                    run(cache, temp, prepared, "reviewed-test-only")

    def test_image_not_updated_and_public_ids_not_forward_inputs(self):
        _, _, q, _, x, means, stats, qt, _, _ = toy()
        factors = SemanticFactors(means, stats, qt.mean(0))
        image = CenteredFusionScorer(means, stats)
        scorer = SemanticScorer(factors, image)
        before = {k: v.clone() for k, v in image.state_dict().items()}
        result = scorer(q["train-0"], x["train-0"])
        self.assertTrue(torch.equal(result[:, 1], image(q["train-0"], x["train-0"])[:, 1]))
        self.assertTrue(((result[:, 0] >= 0) & (result[:, 0] <= 7)).all())
        result[:, 0].sum().backward()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in image.parameters()))
        self.assertTrue(all(torch.equal(image.state_dict()[k], v) for k, v in before.items()))

    def test_atom_precision_recall_counts_use_fixed_threshold(self):
        prediction = torch.tensor([[.9, .6, .1, .1, .1, .1, .1], [.1] * 7])
        target = torch.tensor([[1., 0, 0, 0, 0, 0, 0], [0., 0, 1, 0, 0, 0, 0]])
        metrics = atom_metrics(prediction, target)
        self.assertEqual(metrics["micro"], {"tp": 1, "fp": 1, "fn": 1, "precision": .5, "recall": .5})
        self.assertEqual(metrics["threshold"], .5)


if __name__ == "__main__":
    unittest.main()
