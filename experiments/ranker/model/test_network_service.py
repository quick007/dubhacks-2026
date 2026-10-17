import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import tempfile
import threading
import time
import unittest
from pathlib import Path
import numpy as np
from experiments.ranker.model.network import DecisionNetwork
from experiments.ranker.model.service import DeadlineGate


class TrainingTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_difference(self):
        model = DecisionNetwork(2, 3)
        x = np.array([[.1, .2], [.3, .4]], dtype=np.float32)
        y = np.array([[.2], [.8]], dtype=np.float32)
        _, gradients = model.gradients(x, y)
        epsilon = .001
        original = float(model.weights["b1"][0])
        losses = []
        for direction in (1, -1):
            model.weights["b1"][0] = original + direction * epsilon
            losses.append(float(np.mean((model.predict(x).reshape(-1, 1) - y) ** 2)))
        model.weights["b1"][0] = original
        self.assertAlmostEqual((losses[0] - losses[1]) / (2 * epsilon), float(gradients["b1"][0]), places=3)

    def test_weight_update_and_checkpoint_reload(self):
        model = DecisionNetwork(2, 3)
        x = [[0, 1], [1, 0]]
        before = model.predict(x).copy()
        report = model.fit(x, [0, 1], 20, time.perf_counter() + 2)
        self.assertGreater(report["meanAbsolutePredictionChange"], 0)
        self.assertFalse(np.allclose(before, model.predict(x)))
        with tempfile.TemporaryDirectory() as directory:
            model.save(Path(directory), {"synthetic": True})
            self.assertTrue(np.array_equal(model.predict(x), DecisionNetwork.load(Path(directory)).predict(x)))


class DeadlineTests(unittest.TestCase):
    def test_slow_predictor_times_out_and_busy_requests_never_queue(self):
        release = threading.Event()
        calls = []
        def slow(request):
            calls.append(request)
            release.wait(1)
            return [10]
        gate = DeadlineGate(slow, deadline_ms=2)
        try:
            first, reason = gate.predict("first", lambda: [1])
            self.assertEqual((first, reason), ([1], "deadline"))
            second, reason = gate.predict("second", lambda: [2])
            self.assertEqual((second, reason), ([2], "busy"))
            self.assertEqual(calls, ["first"])
        finally:
            release.set()
            gate.close()

    def test_missing_predictor_falls_back_immediately(self):
        gate = DeadlineGate(None)
        self.assertEqual(gate.predict({}, lambda: [1]), ([1], "model_missing"))
        gate.close()


if __name__ == "__main__":
    unittest.main()
