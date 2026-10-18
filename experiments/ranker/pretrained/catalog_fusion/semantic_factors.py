"""Public-atom distillation from cached native vectors; no native execution."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from .centered import center_rows, load_centered, statistics_digest, validate_statistics
from .inputs import canonical, digest, tensor_digest
from .paired_data import ATOMS
from .scorer import roster

VERSION = "catalog-user-seven-atom-sigmoid-v1"
FACTOR_PARAMETERS = 2 * (4096 * 7 + 7)
POLICY = {"version": VERSION, "vocabulary": list(ATOMS), "activation": "sigmoid",
          "objective": "mean TRAIN query BCE + mean unique TRAIN catalog-variant BCE; equal role weights",
          "productReadout": "dot of predicted query/catalog atom probabilities; mean over product variants",
          "prior": "independent catalog BCE fit; query factor fixed to empirical TRAIN preference frequencies",
          "image": "byte-identical frozen centered-v1 fusion checkpoint; no image updates",
          "targets": "public TRAIN statedPreferences and product tags only; no relevance/image labels"}


def public_targets(requests, catalog, query_members, item_members):
    """Exact caller-owned TRAIN membership; IDs select rows, never model inputs."""
    expected = {a for request in requests.values() for a in roster(request, catalog)}
    if list(query_members) != sorted(requests) or list(item_members) != sorted(expected):
        raise ValueError("Exact sorted TRAIN query and unique catalog membership required")
    q, c = [], []
    for rid in query_members:
        preferences = requests[rid]["context"]["statedPreferences"]
        if not preferences or not set(preferences).issubset(ATOMS):
            raise ValueError("Public preferences must use declared atom vocabulary")
        q.append([float(atom in preferences) for atom in ATOMS])
    for pid, _ in item_members:
        tags = catalog[pid]["tags"]
        if not set(tags).issubset(ATOMS):
            raise ValueError("Public tags outside declared atom vocabulary")
        c.append([float(atom in tags) for atom in ATOMS])
    q, c = torch.tensor(q), torch.tensor(c)
    if not len(q) or not len(c) or (q.sum(0) == 0).any() or (c.sum(0) == 0).any():
        raise ValueError("Every atom needs positive TRAIN coverage in both roles")
    receipt = {"queryMembers": list(query_members), "catalogMembers": [list(a) for a in item_members],
               "queryTargetSha256": tensor_digest(q), "catalogTargetSha256": tensor_digest(c),
               "queryAtomPositiveCounts": q.sum(0).tolist(), "catalogAtomPositiveCounts": c.sum(0).tolist(),
               "usesRankingLabels": False, "catalogWeighting": "each unique variant once"}
    return q, c, receipt


class SemanticFactors(nn.Module):
    def __init__(self, means, statistics, query_frequency, *, arm="fusion", seed=728):
        super().__init__()
        validate_statistics(means, statistics)
        if arm not in ("fusion", "prior", "random"):
            raise ValueError("Unknown semantic factor arm")
        if query_frequency.shape != (7,) or not torch.isfinite(query_frequency).all() or not ((query_frequency > 0) & (query_frequency <= 1)).all():
            raise ValueError("Positive empirical TRAIN atom frequencies required")
        self.arm, self.statistics = arm, statistics
        self.register_buffer("query_mean", means["query"].clone())
        self.register_buffer("catalog_mean", means["catalog"].clone())
        self.register_buffer("query_frequency", query_frequency.clone())
        # Use a local generator, preserving unrelated experiment RNG state.
        generator = torch.Generator().manual_seed(seed)
        self.query_weight = nn.Parameter(torch.randn(7, 4096, generator=generator) * .05)
        self.catalog_weight = nn.Parameter(torch.randn(7, 4096, generator=generator) * .05)
        self.query_bias = nn.Parameter(torch.zeros(7))
        self.catalog_bias = nn.Parameter(torch.zeros(7))
        if arm == "prior":
            self.query_weight.requires_grad_(False)
            self.query_bias.requires_grad_(False)
        assert sum(p.numel() for p in self.parameters()) == FACTOR_PARAMETERS

    def logits(self, query, catalog):
        q = center_rows(query, self.query_mean) @ self.query_weight.T + self.query_bias
        c = center_rows(catalog, self.catalog_mean) @ self.catalog_weight.T + self.catalog_bias
        return q, c

    def probabilities(self, query, catalog):
        q, c = self.logits(query, catalog)
        qp = self.query_frequency.expand_as(q) if self.arm == "prior" else torch.sigmoid(q)
        return qp, torch.sigmoid(c)

    def product_scores(self, query, catalog):
        qp, cp = self.probabilities(query, catalog)
        return cp @ qp


def distillation_loss(model, query_rows, item_rows, query_targets, item_targets):
    q, c = model.logits(query_rows, item_rows)
    catalog_loss = nn.functional.binary_cross_entropy_with_logits(c, item_targets)
    query_loss = nn.functional.binary_cross_entropy_with_logits(q, query_targets)
    loss = catalog_loss if model.arm == "prior" else query_loss + catalog_loss
    return loss, {"queryBce": query_loss.item(), "catalogBce": catalog_loss.item()}


class SemanticScorer(nn.Module):
    """Inference consumes only native vectors; the image head stays frozen."""
    def __init__(self, factors, image):
        super().__init__()
        self.factors, self.image = factors, image.eval()
        for p in self.image.parameters():
            p.requires_grad_(False)

    def forward(self, query, catalog):
        # Frozen image scorer also validates finite normalized raw input shape.
        with torch.no_grad():
            image = self.image(query, catalog)[:, 1]
        return torch.stack((self.factors.product_scores(query, catalog), image), dim=1)


def save_factors(model, path, metadata):
    if metadata["policy"] != POLICY or metadata["statisticsSha256"] != statistics_digest(model.statistics):
        raise ValueError("Wrong factor policy/statistics")
    artifact = {"version": VERSION, "factorParameters": FACTOR_PARAMETERS, "arm": model.arm,
                "metadata": metadata, "metadataSha256": digest(canonical(metadata)),
                "state": model.state_dict()}
    with Path(path).open("xb") as handle:
        torch.save(artifact, handle)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_factors(path, expected_sha, metadata):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_sha:
        raise ValueError("Semantic checkpoint checksum mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("version") != VERSION or saved.get("factorParameters") != FACTOR_PARAMETERS
            or saved.get("metadata") != metadata or saved.get("metadataSha256") != digest(canonical(metadata))):
        raise ValueError("Wrong semantic checkpoint format or training contract")
    state = saved["state"]
    if not torch.equal(state["query_frequency"], torch.tensor(metadata["queryFrequency"])):
        raise ValueError("Checkpoint query frequency differs from TRAIN contract")
    means = {"query": state["query_mean"], "catalog": state["catalog_mean"]}
    model = SemanticFactors(means, metadata["statistics"], state["query_frequency"], arm=saved["arm"])
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite semantic parameters")
    return model.eval()


def load_image(path, metadata):
    return load_centered(path, metadata["imageCheckpoint"]["sha256"], metadata["statisticsSha256"])


def atom_metrics(probabilities, targets):
    predictions, positive = probabilities >= .5, targets.bool()
    tp = (predictions & positive).sum(0)
    fp = (predictions & ~positive).sum(0)
    fn = (~predictions & positive).sum(0)
    def counts(t, f, n):
        t, f, n = int(t), int(f), int(n)
        return {"tp": t, "fp": f, "fn": n, "precision": t / (t + f) if t + f else None,
                "recall": t / (t + n) if t + n else None}
    return {"threshold": .5, "rows": len(targets), "micro": counts(tp.sum(), fp.sum(), fn.sum()),
            "perAtom": {a: counts(tp[i], fp[i], fn[i]) for i, a in enumerate(ATOMS)},
            "predictedProbabilityMatrixRank": int(torch.linalg.matrix_rank(probabilities.double()))}
