"""New CPU fusion architecture and complete-roster action assembly, never MLX."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from ..boundary import guard_reasons
from . import DIMENSION, PARAMETERS, RANK, REPRESENTATION


class FusionScorer(nn.Module):
    def __init__(self, seed=728):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.U = nn.Parameter(torch.randn(RANK, DIMENSION, generator=generator) * .05)
        self.V = nn.Parameter(torch.randn(RANK, DIMENSION, generator=generator) * .05)
        self.prior = nn.Parameter(torch.randn(2, RANK, generator=generator) * .1)
        self.interaction = nn.Parameter(torch.randn(2, RANK, generator=generator) * .1)
        self.bias = nn.Parameter(torch.randn(2, generator=generator) * .01)
        assert sum(p.numel() for p in self.parameters()) == PARAMETERS

    def forward(self, query, catalog):
        if query.shape != (DIMENSION,) or catalog.ndim != 2 or catalog.shape[1] != DIMENSION:
            raise ValueError("Expected one 4096d query and all catalog rows")
        if any(x.device.type != "cpu" or x.dtype != torch.float32 for x in (query, catalog, *self.parameters())):
            raise ValueError("CPU float32 preparation scorer only")
        if not torch.isfinite(query).all() or not torch.isfinite(catalog).all():
            raise ValueError("Nonfinite features")
        user = torch.tanh(query @ self.U.T)
        item = torch.tanh(catalog @ self.V.T)
        return self.bias + item @ self.prior.T + (user * item) @ self.interaction.T


def save_checkpoint(model, path):
    """Create a new artifact without overwriting any existing checkpoint."""
    path = Path(path)
    with path.open("xb") as handle:
        torch.save({"representation": REPRESENTATION, "parameters": PARAMETERS,
                    "state": model.state_dict()}, handle)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path, expected_sha256):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint checksum mismatch")
    artifact = torch.load(path, weights_only=True, map_location="cpu")
    if artifact.get("representation") != REPRESENTATION or artifact.get("parameters") != PARAMETERS:
        raise ValueError("Checkpoint is not catalog-user-fusion-v1")
    model = FusionScorer()
    model.load_state_dict(artifact["state"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite checkpoint")
    return model.eval()


def roster(request, catalog):
    pids = request["candidateProductIds"]
    if not pids or len(pids) != len(set(pids)):
        raise ValueError("Candidate roster must contain unique products")
    actions = []
    for pid in pids:
        variants = catalog[pid]["presentationVariants"]
        vids = [v["id"] for v in variants]
        if not vids or len(vids) != len(set(vids)):
            raise ValueError("Each candidate requires unique presentation variants")
        actions.extend((pid, vid) for vid in vids)
    return actions


def assemble_grid(request, catalog, actions, scores, cards):
    """Current snapshot masks and every selected product's compatible image.

    Every candidate remains in rankedProducts. Shortfall is explicit and partial
    cards never count as complete. Caller must validate features against the same
    snapshot before scoring, including on a catalog cache hit.
    """
    if type(cards) is not int or cards <= 0:
        raise ValueError("Positive requested card count required")
    if list(actions) != roster(request, catalog):
        raise ValueError("Scores must cover the exact complete current roster")
    scores = torch.as_tensor(scores)
    if scores.shape != (len(actions), 2) or not torch.isfinite(scores).all():
        raise ValueError("Finite product/presentation scores required for every variant")
    grouped = {}
    for (pid, vid), row in zip(actions, scores.tolist(), strict=True):
        grouped.setdefault(pid, []).append((vid, row))
    products = []
    for pid, rows in grouped.items():
        product = catalog[pid]
        reasons = guard_reasons(request, product)
        compatibility = request.get("presentationCompatibility", {}).get(pid, {})
        surface = request["context"].get("presentationSurface")
        variants = {v["id"]: v for v in product["presentationVariants"]}
        compatible = [(score[1], vid) for vid, score in rows if compatibility.get(vid) is True
                      and (variants[vid].get("compatibleSurfaces") is None
                           or surface in variants[vid]["compatibleSurfaces"])]
        if not compatible:
            reasons.append("no_compatible_presentation")
        products.append({"productId": pid, "score": sum(row[0] for _, row in rows) / len(rows),
                         "eligible": not reasons, "maskReasons": reasons,
                         "variantId": max(compatible)[1] if compatible else None})
    products.sort(key=lambda p: (not p["eligible"], -p["score"], p["productId"]))
    selected = [p for p in products if p["eligible"]][:cards]
    shortfall = cards - len(selected)
    return {"representation": REPRESENTATION, "status": "shortfall" if shortfall else "complete",
            "requestedCards": cards, "shortfall": shortfall, "fullRequestedCardFulfillment": not shortfall,
            "rankedProducts": products,
            "cards": [{"productId": p["productId"], "variantId": p["variantId"]} for p in selected]}


def score_grid(model, query, features, request, catalog, cards):
    """One CPU tensor scoring operation for the complete current variant roster."""
    actions = roster(request, catalog)
    if features.shape[0] != len(actions):
        raise ValueError("Feature count does not cover complete roster")
    return assemble_grid(request, catalog, actions, model(query, features), cards)
