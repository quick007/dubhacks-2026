"""Synthetic CPU unit tests only; no native work or actual TRAIN fit."""
import tempfile
from pathlib import Path
import unittest
import torch
from . import store_query_intent_audit_v3 as audit
from . import store_query_intent_gate_v3 as gate


class Tests(unittest.TestCase):
    def test_independent_seed_exact_gate_initialization(self):
        expected = gate.QueryIntentHead().linear.state_dict()
        actual = audit.independent_initial_head().state_dict()
        self.assertEqual(set(expected), set(actual))
        self.assertTrue(all(torch.equal(actual[k], expected[k]) for k in actual))

    def test_independent_bce_and_gradient_match(self):
        a = torch.tensor([.1, -.4, 1., .3, -.8], requires_grad=True)
        b = a.detach().clone().requires_grad_()
        labels = torch.tensor([0., 0., 1., 1., 1.])
        first = audit.independent_loss(a, labels)
        second = gate.class_balanced_bce(b, labels)
        self.assertTrue(torch.equal(first, second))
        first.backward(); second.backward()
        self.assertTrue(torch.equal(a.grad, b.grad))
        with self.assertRaises(ValueError):
            audit.independent_loss(torch.zeros(3), torch.zeros(3))

    def test_update_proof_fails_unchanged_and_finds_weight_bias_delta(self):
        initial = {'weight': torch.zeros(1, 4), 'bias': torch.zeros(1)}
        same = audit.state_delta(initial, initial)
        self.assertTrue(all(not x['changed'] for x in same.values()))
        changed = {k: v + .01 for k, v in initial.items()}
        result = audit.state_delta(initial, changed)
        self.assertEqual(result['weight']['changedElements'], 4)
        self.assertEqual(result['bias']['changedElements'], 1)
        with self.assertRaises(ValueError):
            audit.state_delta(initial, {'weight': torch.zeros(1, 4)})
        changed['bias'] = torch.tensor([float('nan')])
        with self.assertRaises(ValueError):
            audit.state_delta(initial, changed)

    def test_saved_prediction_strict_identity_denominator_and_probability(self):
        rows = [{'caseId': 'a', 'labelSpecific': 0}, {'caseId': 'b', 'labelSpecific': 1}]
        saved = {'rows': [dict(rows[0], pSpecific=.25), dict(rows[1], pSpecific=.75)]}
        self.assertTrue(audit.verify_saved_predictions(rows, [.25, .75], saved)['exactSavedProbabilityParity'])
        with self.assertRaises(ValueError):
            audit.verify_saved_predictions(rows, [.25], saved)
        with self.assertRaises(ValueError):
            audit.verify_saved_predictions(rows, [.25, .76], saved)
        with self.assertRaises(ValueError):
            audit.verify_saved_predictions(rows[::-1], [.25, .75], saved)

    def test_failure_receipt_retains_identity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); out = directory / 'audit'
            with self.assertRaises(FileNotFoundError):
                audit.audit(directory / 'absent', directory / 'no-fit', out, 'frozen-plan')
            import json
            receipt = json.loads((out / 'audit.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['sourceManifestSha256'], 'frozen-plan')
            self.assertFalse(receipt['privateQualityLabelsRead'])
            with self.assertRaises(FileExistsError):
                audit.audit(directory, directory, out, 'frozen-plan')

    def test_no_implicit_run_and_fixed_replay_steps(self):
        self.assertEqual(audit.REPLAY_STEPS, 250)
        self.assertEqual(audit.EXPECTED_ROWS, 1272)
        self.assertEqual(audit.EXPECTED_FRESH, 1176)


if __name__ == '__main__':
    unittest.main()
