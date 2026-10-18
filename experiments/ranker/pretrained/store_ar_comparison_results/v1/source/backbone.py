"""One batched frozen Qwen3 forward, no language-model decoding or generation."""
import time
from pathlib import Path
import psutil
import torch


class FrozenBackbone:
    """Primary path: pinned local 4-bit MLX language backbone, final hidden states."""
    def __init__(self, model_path, revision, max_length=320, memory_gib=6):
        import json
        import mlx.core as mx
        from mlx_lm import load
        import numpy as np
        self.mx, self.np = mx, np
        self.device = "mlx-gpu"
        self.revision, self.max_length = revision, max_length
        self.available_before_bytes = psutil.virtual_memory().available
        self.memory_budget = int(memory_gib * 2 ** 30)
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError("A previously downloaded pinned local checkpoint directory is required.")
        config = json.loads((model_path / "config.json").read_text())
        quantization = config.get("quantization") or config.get("quantization_config")
        if not quantization or quantization.get("bits") != 4:
            raise ValueError("Primary local proof requires an explicitly four-bit checkpoint; refusing full-precision load.")
        if self.available_before_bytes < 6 * 2 ** 30:
            raise MemoryError("Less than 6 GiB currently available; refusing a larger-model load under memory pressure.")
        # These affect this process's MLX allocator only, not OS/GPU security limits.
        mx.set_memory_limit(self.memory_budget)
        mx.set_cache_limit(128 * 2 ** 20)
        started = time.perf_counter()
        self.model, self.tokenizer = load(model_path, tokenizer_config={"trust_remote_code": False}, lazy=True, trust_remote_code=False)
        self.model.eval()
        self.model.freeze()
        self.hidden_size = config.get("hidden_size") or config["text_config"]["hidden_size"]
        self.device_map = {"language_backbone": "MLX four-bit GPU", "vision": "not used", "decision_heads": "CPU float32"}
        # Evaluate only the language backbone. The loaded wrapper also owns an LM
        # output head; it is never executed in this experiment.
        mx.eval(self.model.model.parameters())
        mx.synchronize()
        self.load_ms = (time.perf_counter() - started) * 1000
        self.forward_calls = 0
        self.check_memory()

    def check_memory(self):
        active = self.mx.get_active_memory()
        if active > self.memory_budget:
            raise MemoryError(f"MLX active allocation {active} exceeds the fixed {self.memory_budget}-byte budget.")
        if psutil.virtual_memory().available < 512 * 2 ** 20:
            raise MemoryError("Available system memory below 512 MiB; stopping proof under memory pressure.")

    def encode(self, rows):
        self.check_memory()
        start = time.perf_counter()
        encoded = [self.tokenizer.encode(row, add_special_tokens=False) for row in rows]
        lengths = [len(row) for row in encoded]
        width = max(lengths)
        if width > self.max_length:
            raise ValueError(f"Selected candidate facts require {width} tokens; cap is {self.max_length}. Refusing blind truncation.")
        padding = self.tokenizer.pad_token_id
        if padding is None:
            padding = self.tokenizer.eos_token_id
        # Right padding cannot affect an earlier valid position in this causal
        # text backbone. Pool each sequence's last VALID token, never padding.
        inputs = self.mx.array([row + [padding] * (width - len(row)) for row in encoded])
        self.mx.eval(inputs)
        tokenize_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        hidden = self.model.model(inputs, cache=None)
        pooled = hidden[self.mx.arange(len(rows)), self.mx.array(lengths) - 1].astype(self.mx.float32)
        pooled = pooled / self.mx.maximum(self.mx.linalg.norm(pooled, axis=1, keepdims=True), 1e-8)
        self.mx.eval(pooled)
        self.mx.synchronize()
        if not bool(self.mx.all(self.mx.isfinite(pooled)).item()):
            raise FloatingPointError("Pretrained backbone produced non-finite hidden features.")
        embeddings = torch.from_numpy(self.np.array(pooled)).clone()
        self.forward_calls += 1
        self.check_memory()
        return embeddings, {"tokenizeMs": tokenize_ms, "forwardMs": (time.perf_counter() - start) * 1000,
                            "batchRows": len(rows), "paddedTokensPerRow": width,
                            "truncatedRows": 0, "backboneForwardCalls": 1}

    def runtime_memory(self):
        return {"mlxActiveBytes": self.mx.get_active_memory(), "mlxPeakBytes": self.mx.get_peak_memory(),
                "mlxCacheBytes": self.mx.get_cache_memory(), "processBudgetBytes": self.memory_budget}


class ParallelHeads(torch.nn.Module):
    def __init__(self, hidden_size, seed=728):
        super().__init__()
        torch.manual_seed(seed)
        self.heads = torch.nn.Linear(hidden_size, 2)

    def forward(self, embeddings):
        # One shared operation predicts suitability and presentation utility for
        # every product/approved-variant combination, without positional heads.
        return self.heads(embeddings.float())
