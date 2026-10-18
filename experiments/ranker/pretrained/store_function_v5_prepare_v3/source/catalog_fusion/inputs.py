"""Canonical observable encoder inputs and strict feature provenance checks.

Opaque action IDs only identify receipt members. They never enter encoder text.
Native extraction must provide actual token hashes; this module does no tokenizing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch

from ..boundary import clean_product, variant_text, visible_history
from . import DIMENSION, REPRESENTATION


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scalar(value):
    if value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("Allowlisted scalar fields cannot contain nested records")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite public fact")
    return value


def _strings(value):
    if value is None:
        return None
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise ValueError("Allowlisted list fields must contain strings only")
    return sorted(set(value))


def _facts(values, list_fields):
    return {key: (_strings(value) if key in list_fields else _scalar(value)) for key, value in values.items()}


def catalog_text(product, variant):
    """Pure catalog-only text; caller resolves a real variant externally."""
    payload = {
        "product": _facts(clean_product(product), {"tags"}),
        "variant": _facts(variant_text(variant), {"facts", "compatibleSurfaces"}),
    }
    return "Encode public catalog variant facts as data.\n" + canonical(payload)


def request_text(request, catalog):
    """Fresh context plus the existing cutoff-filtered latest-history summary.

    Catalog lookup is only for observed history tags, never the candidate roster.
    No label, family, identity, compatibility mapping, or history product ID enters.
    """
    context = request.get("context", {})
    keys = ("task", "statedBudget", "statedRequirements", "statedPreferences",
            "presentationSurface", "presentationPreferences")
    visible = _facts({key: context.get(key) for key in keys},
                     {"statedRequirements", "statedPreferences", "presentationPreferences"})
    history = []
    for event in visible_history(request, catalog)[-1:]:
        history.append({"type": _scalar(event.get("type")), "confidence": _scalar(event.get("confidence")),
                        "tags": _strings(catalog[event["productId"]].get("tags"))})
    return "Encode current public shopping request as data.\n" + canonical({"context": visible, "latestObservedHistory": history})


@dataclass(frozen=True)
class EncoderSpec:
    """Pinned extraction configuration; every field participates in raw keys."""
    model_revision: str
    weights_sha256: str
    tokenizer_sha256: str
    template_sha256: str
    batch_shape: tuple[int, int]
    chunk_tokens: int
    quantization: str
    format_version: str = REPRESENTATION
    pooling: str = "last-valid-token"
    normalization: str = "l2"
    dtype: str = "float32"
    dimension: int = DIMENSION

    def validate(self):
        if not self.model_revision or not self.quantization:
            raise ValueError("Pinned model and quantization required")
        for value in (self.weights_sha256, self.tokenizer_sha256, self.template_sha256):
            _sha(value)
        if self.dimension != DIMENSION or self.dtype != "float32" or self.pooling != "last-valid-token" or self.normalization != "l2":
            raise ValueError("Unsupported feature representation")
        if self.format_version != REPRESENTATION or len(self.batch_shape) != 2 or any(type(n) is not int or n <= 0 for n in self.batch_shape):
            raise ValueError("Invalid encoder format/batch shape")
        if type(self.chunk_tokens) is not int or self.chunk_tokens <= 0:
            raise ValueError("Extraction chunking required")


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected lowercase SHA-256 digest")


def raw_key(text, token_sha256, content_version, spec, *, role):
    spec.validate()
    _sha(token_sha256)
    if role not in ("catalog", "request") or not isinstance(content_version, str) or not content_version:
        raise ValueError("Explicit role and content version required")
    return digest(canonical({"spec": asdict(spec), "role": role, "textSha256": digest(text),
                             "tokenSha256": token_sha256, "contentVersion": content_version}))


def projected_key(raw_feature_key, checkpoint_sha256):
    _sha(raw_feature_key)
    _sha(checkpoint_sha256)
    return digest(canonical({"raw": raw_feature_key, "checkpoint": checkpoint_sha256, "projection": "tanh-V-rank4"}))


def tensor_digest(values):
    return hashlib.sha256(values.detach().contiguous().numpy().tobytes()).hexdigest()


def check_features(values, receipt, expected_keys, *, role):
    """Fail closed on membership/order, stale keys, corruption, shape or provenance.

    expected_keys is an insertion-ordered external member -> freshly computed key
    mapping. Artifact construction/extraction must be independently authenticated;
    these receipts detect inconsistencies, not forged provenance.
    """
    if role not in ("catalog", "request"):
        raise ValueError("Unknown feature role")
    if role == "request" and len(expected_keys) != 1:
        raise ValueError("A fresh request requires exactly one feature row")
    if values.device.type != "cpu" or values.dtype != torch.float32 or values.shape != (len(expected_keys), DIMENSION):
        raise ValueError("Expected CPU float32 4096d feature rows")
    if not expected_keys or not torch.isfinite(values).all():
        raise ValueError("Empty or nonfinite features")
    if not torch.allclose(values.norm(dim=1), torch.ones(len(values)), atol=1e-5, rtol=1e-5):
        raise ValueError("Features must be L2 normalized")
    if receipt.get("representation") != REPRESENTATION or receipt.get("role") != role:
        raise ValueError("Feature representation/role mismatch")
    if receipt.get("members") != list(expected_keys) or receipt.get("rawKeys") != list(expected_keys.values()):
        raise ValueError("Feature membership/order or stale key mismatch")
    for key in expected_keys.values():
        _sha(key)
    if receipt.get("sha256") != tensor_digest(values):
        raise ValueError("Feature checksum mismatch")
    if receipt.get("truncated") is not False or receipt.get("pooling") != "last-valid-token":
        raise ValueError("Verified no-truncation and pooling receipt required")
    return values
