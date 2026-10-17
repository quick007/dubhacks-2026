"""Small offline tests; no model downloads or pretrained weights are needed."""
import copy
import tempfile
import time
import unittest
from pathlib import Path
import torch
from experiments.ranker.pretrained.boundary import input_rows, decisions
from experiments.ranker.pretrained.backbone import ParallelHeads
from experiments.ranker.pretrained.cli import fit, save_heads, load_heads


class PretrainedBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {"audit-p": {"id": "audit-p", "title": "Everyday pack", "category": "bags", "description": "A factual bag", "price": 30, "shippingCost": 5, "inStock": True, "tags": ["compact"], "presentationVariants": [{"id": "audit-v1", "title": "Compact pack", "imageDescription": "The bag in use", "facts": ["compact"], "presentationStyle": "in-use-view"}, {"id": "audit-v2", "title": "Everyday pack", "imageDescription": "Close view of the bag", "facts": ["compact"], "presentationStyle": "detail-view"}]}}
        self.request = {"recordId": "audit-r", "userId": "audit-u", "asOf": "2026-10-01T00:00:00Z", "context": {"task": "A compact travel bag", "statedBudget": 40, "statedRequirements": ["compact"], "statedPreferences": [], "presentationSurface": "compact-card", "presentationPreferences": ["detail-view"]}, "history": [], "candidateProductIds": ["audit-p"], "presentationCompatibility": {"audit-p": {"audit-v1": False, "audit-v2": True}}}

    def test_private_identity_and_future_fields_never_reach_text(self):
        rows, _ = input_rows(self.request, self.catalog)
        changed = copy.deepcopy(self.request)
        changed.update({"recordId": "other", "hiddenPreferences": "private-secret", "futureActions": "future-secret", "personaFamilyId": "family-secret"})
        changed["history"] = [{"at": "2027-01-01T00:00:00Z", "productId": "audit-p", "description": "future-secret"}]
        self.assertEqual(rows, input_rows(changed, self.catalog)[0])
        self.assertNotIn("audit-", str(rows))

    def test_future_events_cannot_evict_observed_history(self):
        self.request["history"] = [{"at": "2026-09-10T00:00:00Z", "productId": "audit-p", "type": "save", "confidence": .9}]
        original = input_rows(self.request, self.catalog)
        changed = copy.deepcopy(self.request)
        changed["history"] += [{"at": "2027-01-01T00:00:00Z", "productId": "audit-p", "type": "view"}] * 5
        self.assertEqual(original, input_rows(changed, self.catalog))
        changed["history"].reverse()
        self.assertEqual(original, input_rows(changed, self.catalog))

    def test_variant_permutation_is_equivariant_and_guarded(self):
        rows, actions = input_rows(self.request, self.catalog)
        scores = [[.8, 100], [.8, .2]]
        result = decisions(self.request, self.catalog, actions, scores, "test")
        self.assertEqual(result["presentationAction"]["variantId"], "audit-v2")
        self.assertEqual(result, decisions(self.request, self.catalog, actions[::-1], scores[::-1], "test"))
        self.request["context"]["statedBudget"] = 30
        self.assertIsNone(decisions(self.request, self.catalog, actions, scores, "test")["selectedProductId"])

    def test_real_head_updates_and_exact_checkpoint_reload(self):
        embeddings = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 24
        targets = torch.zeros((6, 2))
        head = ParallelHeads(4)
        before = head(embeddings).detach().clone()
        report = fit(head, embeddings, targets, 2, time.perf_counter() + 3)
        self.assertGreater(report["firstGradientL2"], 0)
        self.assertFalse(torch.equal(before, head(embeddings)))
        permutation = torch.tensor([4, 0, 2, 5, 1, 3])
        self.assertTrue(torch.equal(head(embeddings)[permutation], head(embeddings[permutation])))
        with tempfile.TemporaryDirectory() as directory:
            save_heads(head, Path(directory), {"synthetic": True})
            restored = load_heads(Path(directory), 4)
            self.assertTrue(torch.equal(restored(embeddings), head(embeddings)))


if __name__ == "__main__":
    unittest.main()
