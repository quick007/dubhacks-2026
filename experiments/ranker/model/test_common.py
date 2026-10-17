import copy
import unittest
from experiments.ranker.model.common import observable_input, heuristic_scores, known_violations, rank_result, ndcg, percentile


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {"p1": {"id": "p1", "title": "Steel bottle", "category": "bottle", "description": "Reusable bottle", "price": 20, "shippingCost": 4, "inStock": True, "tags": ["steel", "lightweight"], "private": "secret"}, "p2": {"id": "p2", "title": "Glass bottle", "category": "bottle", "description": "Reusable bottle", "price": 30, "shippingCost": 4, "inStock": False, "tags": ["glass"]}}
        self.request = {"schemaVersion": 1, "synthetic": True, "recordId": "id", "userId": "user", "personaFamilyId": "private", "asOf": "2026-01-01T00:00:00Z", "context": {"task": "bottle", "statedBudget": 25, "statedRequirements": ["steel"], "statedPreferences": []}, "history": [], "candidateProductIds": ["p1", "p2"], "hiddenPreferences": {"steel": 1}, "futureOutcome": "p1"}

    def test_whitelist_and_identity_invariance(self):
        original = observable_input(self.request, self.catalog)
        changed = copy.deepcopy(self.request)
        for key in ("recordId", "userId", "personaFamilyId", "hiddenPreferences", "futureOutcome"):
            changed[key] = "different"
        self.assertEqual(original, observable_input(changed, self.catalog))
        self.assertNotIn("secret", str(original))
        self.assertNotIn("p1", str(original))

    def test_future_history_is_removed(self):
        self.request["history"] = [{"at": "2026-01-02T00:00:00Z", "productId": "p1", "type": "purchase", "confidence": 1}]
        self.assertEqual(observable_input(self.request, self.catalog)["history"], [])

    def test_constraint_mask_includes_shipping_and_abstains(self):
        scores = [0, 10]
        raw = rank_result(self.request, self.catalog, scores, "test", guarded=False)
        guarded = rank_result(self.request, self.catalog, scores, "test", guarded=True)
        self.assertEqual(raw["selectedProductId"], "p2")
        self.assertEqual(guarded["selectedProductId"], "p1")
        self.request["context"]["statedBudget"] = 20
        self.assertIsNone(rank_result(self.request, self.catalog, scores, "test", guarded=True)["selectedProductId"])

    def test_metrics_known_values(self):
        self.assertEqual(ndcg(["a", "b"], {"a": 3, "b": 1}), 1)
        self.assertLess(ndcg(["b", "a"], {"a": 3, "b": 1}), 1)
        self.assertEqual(percentile([1, 2, 3], .5), 2)


if __name__ == "__main__":
    unittest.main()
