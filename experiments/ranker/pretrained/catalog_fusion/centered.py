"""One declared role-centering hypothesis; means are frozen training-only buffers."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from . import DIMENSION, PARAMETERS, REPRESENTATION
from .inputs import canonical, digest, tensor_digest
from .scorer import FusionScorer, roster

CENTERED_REPRESENTATION = "catalog-user-fusion-role-centered-v1"
CENTER_CONFIG = {"version": CENTERED_REPRESENTATION, "epsilon": 1e-8,
                 "queryMeanMembership": "exact paired TRAIN query rows, each once",
                 "catalogMeanMembership": "unique paired TRAIN candidate variant members, each once",
                 "accumulation": "float64 mean cast to float32", "inferenceDtype": "float32",
                 "transform": "(x - trainRoleMean) / max(l2(x - trainRoleMean), epsilon)",
                 "architecture": "unchanged rank4 tanh fusion, 32786 trainable parameters",
                 "selection": "one fixed conditioning ablation; no learning-rate search or held-out tuning"}


def _raw_rows(values):
    if values.device.type != "cpu" or values.dtype != torch.float32 or values.ndim != 2 or values.shape[1] != DIMENSION:
        raise ValueError("Raw CPU float32 4096d rows required")
    if not len(values) or not torch.isfinite(values).all() or not torch.allclose(values.norm(dim=1), torch.ones(len(values)), atol=1e-5, rtol=1e-5):
        raise ValueError("Expected finite normalized raw native feature rows")


def compute_statistics(train_requests, catalog, queries, unique_items, *, provenance):
    """No label input; reject extra/missing members, including held-out rows."""
    expected = {a for request in train_requests.values() for a in roster(request, catalog)}
    if not train_requests or set(queries) != set(train_requests) or set(unique_items) != expected:
        raise ValueError("Statistics require exact TRAIN-only query/unique-catalog membership")
    query_members, item_members = sorted(train_requests), sorted(expected)
    q = torch.stack([queries[rid] for rid in query_members])
    c = torch.stack([unique_items[action] for action in item_members])
    _raw_rows(q)
    _raw_rows(c)
    means = {"query": q.double().mean(0).float(), "catalog": c.double().mean(0).float()}
    receipt = {"config": CENTER_CONFIG, "queryMembers": query_members,
               "catalogMembers": [list(a) for a in item_members], "queryRows": len(q), "catalogRows": len(c),
               "queryInputsSha256": tensor_digest(q), "catalogInputsSha256": tensor_digest(c),
               "queryMeanSha256": tensor_digest(means["query"]), "catalogMeanSha256": tensor_digest(means["catalog"]),
               "provenance": provenance, "usesLabels": False,
               "rawCenteredNorms": {role: {"minimum": (values - means[role]).norm(dim=1).min().item(),
                                           "mean": (values - means[role]).norm(dim=1).mean().item(),
                                           "epsilonClampedRows": int(((values - means[role]).norm(dim=1) <= CENTER_CONFIG["epsilon"]).sum())}
                                    for role, values in (("query", q), ("catalog", c))}}
    return means, receipt


def statistics_digest(receipt):
    return digest(canonical(receipt))


def validate_statistics(means, receipt):
    if receipt.get("config") != CENTER_CONFIG or receipt.get("usesLabels") is not False:
        raise ValueError("Wrong frozen centering policy")
    for role in ("query", "catalog"):
        value = means[role]
        if value.device.type != "cpu" or value.dtype != torch.float32 or value.shape != (DIMENSION,) or not torch.isfinite(value).all():
            raise ValueError("Malformed role mean")
        if tensor_digest(value) != receipt[role + "MeanSha256"]:
            raise ValueError("Frozen role mean checksum mismatch")
    if receipt["queryRows"] != len(receipt["queryMembers"]) or receipt["catalogRows"] != len(receipt["catalogMembers"]):
        raise ValueError("Frozen mean membership/count mismatch")


def center_rows(values, mean):
    centered = values - mean
    return centered / centered.norm(dim=-1, keepdim=True).clamp_min(CENTER_CONFIG["epsilon"])


class CenteredFusionScorer(FusionScorer):
    """Consumes raw native vectors; normalizes residuals inside the checkpoint."""
    def __init__(self, means, receipt, seed=728):
        validate_statistics(means, receipt)
        super().__init__(seed)
        self.register_buffer("query_mean", means["query"].detach().clone())
        self.register_buffer("catalog_mean", means["catalog"].detach().clone())
        self.statistics_receipt = receipt
        assert sum(p.numel() for p in self.parameters()) == PARAMETERS

    def forward(self, query, catalog):
        if query.shape != (DIMENSION,):
            raise ValueError("Exactly one raw query vector required")
        _raw_rows(query[None])
        _raw_rows(catalog)
        return super().forward(center_rows(query, self.query_mean), center_rows(catalog, self.catalog_mean))


def save_centered(model, path):
    means = {"query": model.query_mean, "catalog": model.catalog_mean}
    validate_statistics(means, model.statistics_receipt)
    artifact = {"representation": CENTERED_REPRESENTATION, "rawFeaturesRepresentation": REPRESENTATION,
                "parameters": PARAMETERS, "statistics": model.statistics_receipt,
                "statisticsSha256": statistics_digest(model.statistics_receipt), "state": model.state_dict()}
    path = Path(path)
    with path.open("xb") as handle:
        torch.save(artifact, handle)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_centered(path, expected_sha256, expected_statistics_sha256):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Centered checkpoint checksum mismatch")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if (artifact.get("representation") != CENTERED_REPRESENTATION or artifact.get("rawFeaturesRepresentation") != REPRESENTATION
            or artifact.get("parameters") != PARAMETERS):
        raise ValueError("Wrong centered checkpoint format; old heads are incompatible")
    receipt = artifact["statistics"]
    if artifact.get("statisticsSha256") != expected_statistics_sha256 or statistics_digest(receipt) != expected_statistics_sha256:
        raise ValueError("Centered training statistics identity mismatch")
    means = {"query": artifact["state"]["query_mean"], "catalog": artifact["state"]["catalog_mean"]}
    model = CenteredFusionScorer(means, receipt)
    model.load_state_dict(artifact["state"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite centered checkpoint")
    return model.eval()
