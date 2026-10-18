"""Synthetic CPU contracts only. These are not quality/latency measurements."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from . import DIMENSION, PARAMETERS, REPRESENTATION
from .inputs import (EncoderSpec, canonical, catalog_text, check_features, digest,
                     projected_key, raw_key, request_text, tensor_digest)
from .scorer import (FusionScorer, assemble_grid, load_checkpoint, roster,
                     save_checkpoint, score_grid)
from .training import read_training_inputs, training_loss


def fixture(count=3):
    catalog = {}
    for i in range(count):
        pid = f"opaque-product-{i:02}"
        catalog[pid] = {"id": pid, "title": "Travel bag", "category": "bags", "description": "Compact bag",
                        "price": 30, "shippingCost": 5, "inStock": True, "tags": ["compact"],
                        "presentationVariants": [{"id": f"opaque-image-{i:02}-{j}", "title": "Travel bag",
                            "imageDescription": "Detail view" if j else "In use", "facts": ["compact"],
                            "presentationStyle": "detail" if j else "in-use", "compatibleSurfaces": ["card"]} for j in range(2)]}
    request = {"recordId": "opaque-record", "userId": "opaque-user", "asOf": "2026-10-01T00:00:00Z",
               "candidateProductIds": list(catalog), "history": [],
               "context": {"task": "Find a compact bag", "statedBudget": 40, "statedRequirements": ["compact"],
                           "statedPreferences": ["compact"], "presentationSurface": "card", "presentationPreferences": ["detail"]},
               "presentationCompatibility": {pid: {v["id"]: True for v in p["presentationVariants"]} for pid, p in catalog.items()}}
    return request, catalog


def normalized(count, seed=8):
    x = torch.randn(count, DIMENSION, generator=torch.Generator().manual_seed(seed))
    return x / x.norm(dim=1, keepdim=True)


class CatalogFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.request, self.catalog = fixture()
        self.product = next(iter(self.catalog.values()))
        self.variant = self.product["presentationVariants"][0]
        self.spec = EncoderSpec("pinned-revision", "a" * 64, "b" * 64, "c" * 64, (1, 512), 128, "4bit")

    def test_canonical_allowlists_and_private_identity_position_separation(self):
        original = catalog_text(self.product, self.variant)
        p, v = copy.deepcopy(self.product), copy.deepcopy(self.variant)
        p.update(id="changed", position=999, labels="secret", personaFamily="secret")
        v.update(id="changed", imageKey="secret", position=100, label="secret")
        self.assertEqual(original, catalog_text(p, v))
        p["tags"] = ["compact", "compact"]
        self.assertEqual(original, catalog_text(p, v))
        request = copy.deepcopy(self.request)
        request.update(userId="secret", recordId="secret", labels="secret", family="secret", privateTruth="secret")
        request["context"]["hiddenPreferences"] = "secret"
        request["candidateProductIds"].reverse()
        request["presentationCompatibility"] = {}
        self.assertEqual(request_text(self.request, self.catalog), request_text(request, self.catalog))
        self.assertNotIn("opaque", original + request_text(request, self.catalog))
        p["tags"] = [{"private": "secret"}]
        with self.assertRaises(ValueError):
            catalog_text(p, v)

    def test_latest_history_cutoff_unknowns_and_fresh_request(self):
        request = copy.deepcopy(self.request)
        pid = request["candidateProductIds"][0]
        request["history"] = [{"at": "2026-09-01T00:00:00Z", "productId": pid, "type": "view", "confidence": .35,
                               "description": "private description ignored", "labels": "secret"}]
        a = request_text(request, self.catalog)
        self.assertIn("0.35", a)
        self.assertNotIn("private", a)
        request["history"] += [{"at": "2027-01-01T00:00:00Z", "productId": pid, "type": "future"},
                                {"at": "2026-09-30T00:00:00Z", "productId": "unknown", "type": "unknown"}]
        self.assertEqual(a, request_text(request, self.catalog))
        request["context"]["statedPreferences"] = ["durable"]
        self.assertNotEqual(a, request_text(request, self.catalog))

    def test_cache_invalidation_and_immutable_a_b_a_catalog(self):
        text = catalog_text(self.product, self.variant)
        key = lambda text=text, spec=self.spec, version="1", token="d" * 64: raw_key(text, token, version, spec, role="catalog")
        original = key()
        for field, value in (("model_revision", "new"), ("weights_sha256", "e" * 64), ("tokenizer_sha256", "e" * 64),
                             ("template_sha256", "e" * 64), ("batch_shape", (2, 512)), ("chunk_tokens", 64), ("quantization", "8bit")):
            self.assertNotEqual(original, key(spec=replace(self.spec, **{field: value})))
        self.assertNotEqual(original, key(version="2"))
        self.assertNotEqual(original, key(token="e" * 64))
        for field, value in (("price", 31), ("shippingCost", 7), ("inStock", False)):
            product = {**self.product, field: value}
            self.assertNotEqual(original, key(text=catalog_text(product, self.variant)))
        self.assertNotEqual(original, key(text=catalog_text(self.product, {**self.variant, "imageDescription": "New view"})))
        self.assertNotEqual(projected_key(original, "a" * 64), projected_key(original, "b" * 64))
        entries = {pid: tuple(catalog_text(p, v) for v in p["presentationVariants"]) for pid, p in self.catalog.items()}
        before = canonical(entries)
        a = request_text(self.request, self.catalog)
        changed = copy.deepcopy(self.request)
        changed["context"]["task"] = "Different current task"
        self.assertNotEqual(a, request_text(changed, self.catalog))
        self.assertEqual(a, request_text(self.request, self.catalog))
        self.assertEqual(before, canonical(entries))

    def test_feature_receipts_reject_stale_corrupt_unormalized_and_wrong_members(self):
        x = normalized(2)
        keys = {"external-a": "a" * 64, "external-b": "b" * 64}
        receipt = {"representation": REPRESENTATION, "role": "catalog", "members": list(keys), "rawKeys": list(keys.values()),
                   "sha256": tensor_digest(x), "truncated": False, "pooling": "last-valid-token"}
        self.assertIs(check_features(x, receipt, keys, role="catalog"), x)
        for updates in ({"rawKeys": ["c" * 64, "b" * 64]}, {"members": list(reversed(keys))}, {"role": "request"},
                        {"representation": "shared-prefix-v1"}, {"truncated": True}, {"sha256": "f" * 64}):
            with self.assertRaises(ValueError):
                check_features(x, {**receipt, **updates}, keys, role="catalog")
        for changed in (x[:, :4], x.double(), x * 2, x * float("nan")):
            with self.assertRaises(ValueError):
                check_features(changed, receipt, keys, role="catalog")
        with self.assertRaises(ValueError):
            check_features(x, {**receipt, "role": "request"}, keys, role="request")
        with self.assertRaises(ValueError):
            raw_key("x", "d" * 64, "1", replace(self.spec, pooling="mean"), role="catalog")

    def test_true_interaction_gradients_updates_and_exact_checkpoint_reload(self):
        model = FusionScorer()
        self.assertEqual(sum(p.numel() for p in model.parameters()), PARAMETERS)
        self.assertTrue(all(torch.count_nonzero(p) for p in model.parameters()))
        q, x = normalized(1)[0], normalized(6, seed=9)
        scores = model(q, x)
        other = model(-q, x)
        # Within-roster differences change: a query-only offset cannot pass this.
        self.assertFalse(torch.equal(scores[0] - scores[1], other[0] - other[1]))
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        (scores[0, 0] - scores[1, 0] + scores[2, 1] - scores[3, 1]).backward()
        for name, p in model.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            if name != "bias":
                self.assertGreater(p.grad.abs().sum().item(), 0, name)
        with torch.no_grad():
            for p in model.parameters():
                p -= .1 * p.grad
        for name in ("U", "V", "prior", "interaction"):
            self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-untrained.pt"
            sha = save_checkpoint(model, path)
            reloaded = load_checkpoint(path, sha)
            self.assertTrue(torch.equal(model(q, x), reloaded(q, x)))
            with self.assertRaises(ValueError):
                load_checkpoint(path, "f" * 64)

    def test_checkpoint_refuses_overwrite_and_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.pt"
            existing = b"preexisting user checkpoint bytes"
            path.write_bytes(existing)
            with self.assertRaises(FileExistsError):
                save_checkpoint(FusionScorer(), path)
            self.assertEqual(path.read_bytes(), existing)

    def test_complete_12_24_grid_mapping_and_masks(self):
        for count in (12, 24):
            request, catalog = fixture(count + 2)
            actions = roster(request, catalog)
            scores = torch.tensor([[float(i // 2), float(i % 2)] for i in range(len(actions))])
            blocked, no_image = request["candidateProductIds"][-2:]
            catalog[blocked]["inStock"] = False
            request["presentationCompatibility"][no_image] = {}
            result = assemble_grid(request, catalog, actions, scores, count)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["rankedProducts"]), count + 2)
            self.assertEqual(len(result["cards"]), count)
            self.assertEqual(len({c["productId"] for c in result["cards"]}), count)
            for card in result["cards"]:
                self.assertEqual(card["variantId"], catalog[card["productId"]]["presentationVariants"][1]["id"])
            pid = result["cards"][0]["productId"]
            catalog[pid]["price"] = 100
            shortfall = assemble_grid(request, catalog, actions, scores, count)
            self.assertEqual(shortfall["shortfall"], 1)
            self.assertFalse(shortfall["fullRequestedCardFulfillment"])
            with self.assertRaises(ValueError):
                assemble_grid(request, catalog, actions[:-1], scores[:-1], count)
        # Exercise the complete CPU scoring entry point, no model/backbone load.
        self.assertEqual(len(score_grid(FusionScorer(), normalized(1)[0], normalized(6), self.request, self.catalog, 3)["cards"]), 3)

    def test_reordered_roster_preserves_ids_and_surface_and_requirement_masks(self):
        actions = roster(self.request, self.catalog)
        scores = torch.tensor([[1., float(i % 2)] for i in range(len(actions))])
        first = assemble_grid(self.request, self.catalog, actions, scores, 3)
        request = copy.deepcopy(self.request)
        request["candidateProductIds"].reverse()
        shuffled = roster(request, self.catalog)
        by_action = dict(zip(actions, scores))
        second = assemble_grid(request, self.catalog, shuffled, torch.stack([by_action[a] for a in shuffled]), 3)
        self.assertEqual(first["cards"], second["cards"])
        # Reorder image slots too: join by exact external action, never slot.
        catalog = copy.deepcopy(self.catalog)
        for product in catalog.values():
            product["presentationVariants"].reverse()
        reordered = roster(request, catalog)
        self.assertEqual(first["cards"], assemble_grid(request, catalog, reordered,
                         torch.stack([by_action[a] for a in reordered]), 3)["cards"])
        request["context"]["presentationSurface"] = "unsupported"
        self.assertEqual(assemble_grid(request, self.catalog, shuffled, scores, 3)["shortfall"], 3)
        request["context"]["presentationSurface"] = "card"
        request["context"]["statedRequirements"] = ["missing"]
        self.assertEqual(assemble_grid(request, self.catalog, shuffled, scores, 3)["shortfall"], 3)

    def test_train_only_loader_and_synthetic_objective(self):
        label = {"recordId": self.request["recordId"], "bestProductId": self.request["candidateProductIds"][0],
                 "presentationScores": {pid: {v["id"]: i for i, v in enumerate(p["presentationVariants"])} for pid, p in self.catalog.items()}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind, row in (("public", self.request), ("labels", label)):
                (root / kind).mkdir()
                (root / kind / "train.jsonl").write_text(json.dumps(row) + "\n")
                (root / kind / "test.jsonl").write_text("invalid held-out data; must never open")
            requests, labels, hashes = read_training_inputs(root)
            self.assertEqual(set(hashes), {"public/train.jsonl", "labels/train.jsonl"})
            model = FusionScorer()
            rid = self.request["recordId"]
            loss = training_loss(model, requests, labels, self.catalog, {rid: normalized(1)[0]}, {rid: normalized(6)})
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(model.U.grad.abs().sum().item(), 0)
            (root / "public/train.jsonl").unlink()
            (root / "public/train.jsonl").symlink_to(root / "public/test.jsonl")
            with self.assertRaises(ValueError):
                read_training_inputs(root)


if __name__ == "__main__":
    unittest.main()
