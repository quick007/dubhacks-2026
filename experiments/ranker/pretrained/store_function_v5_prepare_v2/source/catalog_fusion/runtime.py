"""Verified local native adapter and immutable feature files. No load on import."""
from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from . import REPRESENTATION
from .inputs import EncoderSpec, canonical, check_features, digest, raw_key, tensor_digest
from .preflight import REVISION, file_digest

SHARDS = {
    "model-00001-of-00002.safetensors": "a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd",
    "model-00002-of-00002.safetensors": "b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d",
}


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def verify_model(model, preflight):
    """Fresh local hashes, without importing MLX or downloading anything."""
    start = time.perf_counter()
    model = Path(model).resolve()
    if str(model) != preflight["modelPath"] or model.name != REVISION:
        raise ValueError("Pinned model path/revision mismatch")
    for name, expected in SHARDS.items():
        hasher = hashlib.sha256()
        with (model / name).open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                hasher.update(block)
        if hasher.hexdigest() != expected:
            raise ValueError(f"Pinned local weight checksum mismatch: {name}")
    tokenizer_files = {name: file_digest(model / name) for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json")}
    if tokenizer_files["tokenizer.json"] != preflight["tokenizerSha256"]:
        raise ValueError("Tokenizer changed after preflight")
    template = file_digest(model / "chat_template.jinja")
    if template != preflight["chatTemplateSha256"]:
        raise ValueError("Template changed after preflight")
    config_sha = file_digest(model / "config.json")
    if config_sha != preflight["configSha256"]:
        raise ValueError("Model configuration changed after preflight")
    names = [*SHARDS, *tokenizer_files, "chat_template.jinja", "config.json"]
    stats = {name: {"bytes": (model / name).stat().st_size, "mtimeNs": (model / name).stat().st_mtime_ns,
                    "inode": (model / name).stat().st_ino} for name in names}
    return {"evidenceSource": "native-frozen-qwen", "modelRevision": REVISION,
            "weightsSha256": digest(canonical({"shards": SHARDS, "configSha256": config_sha})), "shards": SHARDS,
            "tokenizerSha256": digest(canonical(tokenizer_files)), "tokenizerFiles": tokenizer_files,
            "templateSha256": template, "quantization": preflight["quantization"],
            "configSha256": config_sha, "verifiedFileStats": stats,
            "weightVerificationSeconds": time.perf_counter() - start}


def assert_model_files(model, provenance):
    """Check the verified local snapshot has not changed before each model load."""
    model = Path(model).resolve()
    for name, expected in provenance["verifiedFileStats"].items():
        stat = (model / name).stat()
        if {"bytes": stat.st_size, "mtimeNs": stat.st_mtime_ns, "inode": stat.st_ino} != expected:
            raise ValueError(f"Verified model/tokenizer file changed before load: {name}")


def guarded_native_memory(backbone):
    backbone.check_memory()
    memory = backbone.runtime_memory()
    budget = 6 * 2 ** 30
    if memory["processBudgetBytes"] != budget:
        raise MemoryError("Native process budget differs from the fixed 6 GiB guard")
    if memory["mlxActiveBytes"] > budget or memory["mlxPeakBytes"] > budget:
        raise MemoryError(f"MLX active/observed peak exceeds the fixed {budget}-byte guard: {memory}")
    return memory


class NativeEncoder:
    """B=1, cache=None, whole-sequence native forwards; created only by the CLI."""
    evidence_source = "native-frozen-qwen"

    def __init__(self, model, *, memory_gib=6, max_tokens=512):
        if memory_gib != 6 or max_tokens != 512:
            raise ValueError("This attempt fixes the 6 GiB guard and 512-token cap")
        from ..backbone import FrozenBackbone

        start = time.perf_counter()
        self.backbone = FrozenBackbone(model, REVISION, max_length=max_tokens, memory_gib=memory_gib)
        guarded_native_memory(self.backbone)
        self.load_seconds = time.perf_counter() - start

    def encode(self, text, expected):
        guarded_native_memory(self.backbone)
        start = time.perf_counter()
        actual = self.backbone.tokenizer.encode(text, add_special_tokens=False)
        if actual != expected["tokenIds"] or digest(canonical(actual)) != expected["tokenSha256"]:
            raise ValueError("Native tokenizer IDs differ from CPU preflight")
        check_ms = (time.perf_counter() - start) * 1000
        values, timing = self.backbone.encode([text])
        if timing["batchRows"] != 1 or timing["paddedTokensPerRow"] != expected["tokens"] or timing["truncatedRows"] != 0:
            raise ValueError("Native extraction shape/truncation mismatch")
        return values, {**timing, "nativeTokenIdCheckMs": check_ms,
                        "encodeTotalMs": (time.perf_counter() - start) * 1000,
                        "memory": guarded_native_memory(self.backbone)}

    def memory(self):
        return guarded_native_memory(self.backbone)

    def close(self):
        if self.backbone is not None:
            mx = self.backbone.mx
            self.backbone = None
            gc.collect()
            mx.clear_cache()
            mx.synchronize()


class FeatureStore:
    """Immutable per-member vectors; warm hits still verify current keys/bytes."""
    def __init__(self, root, provenance):
        self.root, self.provenance = Path(root), provenance
        self.readonly = False
        self.root.mkdir(exist_ok=False)
        for role in ("catalog", "request"):
            (self.root / role).mkdir()

    @classmethod
    def open_existing(cls, root, provenance):
        """Read-only verified reuse; every hit still checks content and bytes."""
        store = cls.__new__(cls)
        store.root, store.provenance, store.readonly = Path(root), provenance, True
        if not all((store.root / role).is_dir() for role in ('catalog','request')):
            raise ValueError('Existing feature store is incomplete')
        return store

    def paths(self, role, member):
        base = self.root / role / digest(canonical(member))
        return base.with_suffix(".pt"), base.with_suffix(".json")

    def expected(self, row):
        text = row["text"]
        if digest(text) != row["textSha256"] or digest(canonical(row["tokenIds"])) != row["tokenSha256"]:
            raise ValueError("Public text/tokens changed after preflight")
        if row["tokens"] != len(row["tokenIds"]) or row["batchShape"] != [1, row["tokens"]]:
            raise ValueError("Expected exact B1 whole-input shape")
        if not 0 < row["tokens"] <= 512:
            raise ValueError("Empty/over-limit feature input; refusing truncation")
        provenance = self.provenance
        spec = EncoderSpec(provenance["modelRevision"], provenance["weightsSha256"], provenance["tokenizerSha256"],
                           provenance["templateSha256"], (1, row["tokens"]), 512, canonical(provenance["quantization"]))
        key = raw_key(text, row["tokenSha256"], digest(text), spec, role=row["role"])
        # ID is only external receipt membership, not raw feature key material.
        member = canonical(row["member"])
        return {member: key}, spec

    def exists(self, row):
        data, receipt = self.paths(row["role"], row["member"])
        if data.exists() != receipt.exists():
            raise ValueError("Incomplete immutable feature entry")
        return data.exists()

    def save(self, row, values, timing):
        if self.readonly:raise ValueError('Read-only feature store cannot save')
        keys, spec = self.expected(row)
        values = values.detach().clone()
        receipt = {"representation": REPRESENTATION, "role": row["role"], "members": list(keys),
                   "rawKeys": list(keys.values()), "sha256": tensor_digest(values), "truncated": False,
                   "pooling": "last-valid-token", "encoderSpec": asdict(spec),
                   "textSha256": row["textSha256"], "tokenSha256": row["tokenSha256"],
                   "contentVersion": digest(row["text"]), "timing": timing,
                   "evidenceSource": self.provenance["evidenceSource"]}
        check_features(values, receipt, keys, role=row["role"])
        data, path = self.paths(row["role"], row["member"])
        with data.open("xb") as handle:
            torch.save(values, handle)
        receipt["artifactSha256"] = file_digest(data)
        write_json(path, receipt)
        # Confirm serialized bytes and metadata immediately, not just RAM values.
        return self.load(row)

    def load(self, row):
        keys, spec = self.expected(row)
        data, path = self.paths(row["role"], row["member"])
        receipt = json.loads(path.read_text())
        if (receipt.get("textSha256") != row["textSha256"] or receipt.get("tokenSha256") != row["tokenSha256"]
                or receipt.get("contentVersion") != digest(row["text"])
                or canonical(receipt.get("encoderSpec")) != canonical(asdict(spec))
                or receipt.get("evidenceSource") != self.provenance["evidenceSource"]):
            raise ValueError("Stale/incompatible extraction receipt")
        if receipt.get("artifactSha256") != file_digest(data):
            raise ValueError("Serialized feature checksum mismatch")
        values = torch.load(data, map_location="cpu", weights_only=True)
        check_features(values, receipt, keys, role=row["role"])
        return values, receipt
