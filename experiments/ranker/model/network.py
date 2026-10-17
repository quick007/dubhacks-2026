"""Tiny parallel decision network: all candidate scores in one NumPy matrix pass.

From-scratch supervised fit, then warm-start adaptation. Not a language model,
pretrained foundation model, or implementation of the Jev/TypeSafe architecture.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np

CONFIG = {"seed": 1729, "hiddenWidth": 32, "initialEpochs": 160, "adaptationEpochs": 160,
          "learningRate": .06, "l2": .001, "maxTrainingSeconds": 60, "dtype": "float32"}


class DecisionNetwork:
    def __init__(self, width: int, hidden: int = 32, seed: int = 1729):
        rng = np.random.default_rng(seed)
        self.weights = {"w1": (rng.normal(size=(width, hidden)) / np.sqrt(width)).astype(np.float32),
                        "b1": np.zeros(hidden, dtype=np.float32),
                        "w2": (rng.normal(size=(hidden, 1)) / np.sqrt(hidden)).astype(np.float32),
                        "b2": np.zeros(1, dtype=np.float32)}

    def predict(self, features):
        x = np.asarray(features, dtype=np.float32)
        h = np.tanh(x @ self.weights["w1"] + self.weights["b1"])
        return (h @ self.weights["w2"] + self.weights["b2"]).reshape(-1)

    def gradients(self, x, y):
        w = self.weights
        h = np.tanh(x @ w["w1"] + w["b1"])
        prediction = h @ w["w2"] + w["b2"]
        error = prediction - y
        dy = 2 * error / len(x)
        dh = (dy @ w["w2"].T) * (1 - h * h)
        gradients = {"w1": x.T @ dh + CONFIG["l2"] * w["w1"], "b1": dh.sum(axis=0),
                     "w2": h.T @ dy + CONFIG["l2"] * w["w2"], "b2": dy.sum(axis=0)}
        return float(np.mean(error * error)), gradients

    def fit(self, x, y, epochs, deadline):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        before = self.predict(x)
        loss_start = float(np.mean((before - y.reshape(-1)) ** 2))
        first_gradient_norm = None
        start = time.perf_counter()
        for epoch in range(epochs):
            if time.perf_counter() > deadline:
                raise TimeoutError("Fixed 60-second total training cap exceeded; no sweep or automatic retry.")
            _, gradients = self.gradients(x, y)
            if first_gradient_norm is None:
                first_gradient_norm = float(np.sqrt(sum(np.sum(value ** 2) for value in gradients.values())))
            for key, gradient in gradients.items():
                self.weights[key] -= CONFIG["learningRate"] * gradient
        after = self.predict(x)
        return {"epochs": epochs, "examples": len(x), "seconds": time.perf_counter() - start,
                "mseBefore": loss_start, "mseAfter": float(np.mean((after - y.reshape(-1)) ** 2)),
                "firstGradientL2": first_gradient_norm, "meanAbsolutePredictionChange": float(np.mean(abs(after - before)))}

    def save(self, path: Path, metadata: dict):
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "weights.npz", **self.weights)
        digest = hashlib.sha256((path / "weights.npz").read_bytes()).hexdigest()
        metadata = {**metadata, "sha256": digest, "parameterCount": sum(value.size for value in self.weights.values())}
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata

    @classmethod
    def load(cls, path: Path):
        weights = np.load(path / "weights.npz", allow_pickle=False)
        model = cls(weights["w1"].shape[0], weights["w1"].shape[1])
        model.weights = {key: weights[key] for key in weights.files}
        return model
