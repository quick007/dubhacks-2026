"""CPU/stub tests only: none of these load MLX or a pretrained checkpoint."""
import tempfile
from pathlib import Path
import unittest
import torch
from .routing import RouteHead, fit, grounding, label_result, load_head, save_head, teacher_route


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.product = {"id": "p", "title": "Compact durable backpack", "tags": ["compact", "durable"],
                        "price": 20, "shippingCost": 2, "inStock": True,
                        "presentationVariants": [{"id": "v", "title": "Compact backpack"}]}
        self.catalog = {"p": self.product}
        self.request = {"context": {"statedPreferences": [], "statedRequirements": [], "statedBudget": 30}}

    def test_teacher_none_select_generate_and_private_unknown(self):
        self.assertEqual(teacher_route(self.request, self.product), "NONE")
        self.request["hiddenPreferences"] = ["compact", "durable"]
        self.assertEqual(teacher_route(self.request, self.product), "NONE")
        self.request["context"]["statedPreferences"] = ["compact"]
        self.assertEqual(teacher_route(self.request, self.product), "SELECT")
        self.request["context"]["statedRequirements"] = ["durable"]
        self.assertEqual(teacher_route(self.request, self.product), "GENERATE")

    def test_no_generation_on_none_select_or_missing_support(self):
        def forbidden():
            self.fail("Decoder must not run")
        self.request["context"]["statedPreferences"] = ["compact"]
        self.assertIsNone(label_result("NONE", self.request, self.product, True, self.catalog, forbidden)["text"])
        selected = label_result("SELECT", self.request, self.product, True, self.catalog, forbidden)
        self.assertEqual(selected["selectedLabelId"], "attribute:compact")
        self.assertEqual(selected["grounding"][0]["productFact"], "compact")
        self.request["context"]["statedPreferences"] = []
        self.assertEqual(label_result("GENERATE", self.request, self.product, True, self.catalog, forbidden)["servedRoute"], "NONE")

    def test_compatible_supported_evidence_only(self):
        self.request["context"]["statedPreferences"] = ["compact", "waterproof"]
        self.assertEqual(len(grounding(self.request, self.product)), 1)
        self.assertEqual(label_result("SELECT", self.request, self.product, False, self.catalog)["servedRoute"], "NONE")
        self.product["inStock"] = False
        self.assertEqual(label_result("GENERATE", self.request, self.product, True, self.catalog)["servedRoute"], "NONE")

    def test_generated_short_supported_title_and_rejected_copy(self):
        self.request["context"]["statedPreferences"] = ["compact", "durable"]
        result = label_result("GENERATE", self.request, self.product, True, self.catalog,
            lambda: {"text": "Title: Your compact durable travel companion\nCopy: Free shipping tomorrow guaranteed.", "finishReason": "stop"})
        self.assertTrue(result["generationAttempted"])
        self.assertEqual(result["servedRoute"], "GENERATE")
        self.assertIn("compact", result["text"])
        self.assertNotIn("shipping", result["text"])
        failed = label_result("GENERATE", self.request, self.product, True, self.catalog,
            lambda: {"text": "For you", "finishReason": "stop"})
        self.assertEqual(failed["servedRoute"], "SELECT")
        self.assertTrue(failed["generationFailure"])

    def test_decode_error_falls_back_once(self):
        self.request["context"]["statedPreferences"] = ["compact", "durable"]
        calls = []
        def broken():
            calls.append(1)
            raise ValueError("Exact prefix boundary unavailable")
        result = label_result("GENERATE", self.request, self.product, True, self.catalog, broken)
        self.assertEqual(calls, [1])
        self.assertEqual(result["servedRoute"], "SELECT")
        self.assertIn("Exact prefix", result["generationFailure"])

    def test_real_cpu_update_exact_reload_and_hash(self):
        torch.set_num_threads(1)
        head = RouteHead(4)
        x = torch.eye(4)
        evidence = fit(head, x, torch.tensor([0, 1, 2, 0]))
        self.assertGreater(evidence["firstGradientL2"], 0)
        self.assertGreater(evidence["meanAbsolutePredictionChange"], 0)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            receipt = save_head(head, directory, "test-revision")
            self.assertTrue(receipt["exactWeightReload"])
            self.assertTrue(torch.equal(head(x), load_head(directory, "test-revision")(x)))
            with self.assertRaises(ValueError):
                load_head(directory, "different-revision")
            with (directory / "route.pt").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(ValueError):
                load_head(directory, "test-revision")


if __name__ == "__main__":
    unittest.main()
