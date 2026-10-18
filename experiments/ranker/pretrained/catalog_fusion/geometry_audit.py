"""Unlabeled CPU geometry/saturation audit; never fits or loads native models."""
import argparse
import json
from pathlib import Path

import torch

from .cached_inputs import CachedPairedInputs
from .fit_runner import require_frozen
from .preflight import file_digest
from .runtime import write_json
from .scorer import load_checkpoint


def geometry(values, projection):
    values = values.detach().double()
    linear = values @ projection.detach().double().T
    nonlinear = torch.tanh(linear)
    # Also retain actual float32 forward geometry; float64 descriptive recompute
    # can reveal variation already rounded out by float32 tanh saturation.
    actual = torch.tanh(values.float() @ projection.detach().float().T)
    mean = values.mean(0)
    centered = (values - mean).norm(dim=1)
    cosine = values @ values.T / (values.norm(dim=1)[:, None] * values.norm(dim=1)[None, :])
    off = cosine[~torch.eye(len(values), dtype=torch.bool)]
    return {"rows": len(values), "meanVectorNorm": mean.norm().item(),
            "centeredNorm": {"minimum": centered.min().item(), "mean": centered.mean().item(), "maximum": centered.max().item()},
            "pairCosine": {"minimum": off.min().item(), "mean": off.mean().item()} if len(values) > 1 else None,
            "linearPerDimension": {"mean": linear.mean(0).tolist(), "std": linear.std(0, correction=0).tolist(),
                                   "minimum": linear.amin(0).tolist(), "maximum": linear.amax(0).tolist()},
            "tanhPerDimension": {"mean": nonlinear.mean(0).tolist(), "std": nonlinear.std(0, correction=0).tolist()},
            "saturationFractionAbsAtLeast0.99": (actual.abs() >= .99).float().mean().item(),
            "saturationFractionAbsAtLeast0.999": (actual.abs() >= .999).float().mean().item(),
            "actualFloat32TanhExactlyAbsOneFraction": (actual.abs() == 1).float().mean().item(),
            "actualFloat32TanhAcrossRowVarianceSum": actual.var(0, correction=0).sum().item(),
            "actualFloat32TanhDistinctRows": len(torch.unique(actual, dim=0)),
            "meanTanhDerivativeFloat64": (1 - nonlinear.square()).mean().item()}


def audit(dataset, queries, catalog_run):
    torch.set_num_threads(1)
    cache = CachedPairedInputs(dataset, queries, catalog_run)
    frozen = require_frozen(queries)
    models = {arm: load_checkpoint(Path(queries) / row["file"], row["sha256"]) for arm, row in frozen["checkpoints"].items()}
    splits = {}
    for split in ("train", "validation", "test"):
        q, _, items, _ = cache.load_split(split)
        q = torch.stack([q[rid] for rid in sorted(q)])
        c = torch.stack([items[key] for key in sorted(items)])
        splits[split] = {arm: {"query": geometry(q, model.U), "catalog": geometry(c, model.V)} for arm, model in models.items()}
    return {"status": "unlabeled-geometry-audit-only", "auditSourceSha256": file_digest(__file__),
            "sources": cache.source_receipts, "splits": splits, "nativeForwards": 0, "fitting": False,
            "interpretation": "Describe observed projection/tanh geometry; this does not identify a causal fix or establish learned utility.",
            "nextHypothesis": "One fixed train-only role-centered L2-normalized feature ablation; same rank4/tanh/32786 trainable architecture and optimizer schedule."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("experiments/ranker/pretrained/catalog_fusion/paired_v1"))
    p.add_argument("--queries", type=Path, default=Path("data/ranker/pretrained/catalog-user-fusion-paired-v1-attempt-1"))
    p.add_argument("--catalog-run", type=Path, default=Path("data/ranker/pretrained/catalog-user-fusion-v1-attempt-1"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    r = audit(a.dataset, a.queries, a.catalog_run)
    write_json(a.out, r)
    print(json.dumps({split: {role: {k: v for k, v in rows["fusion"][role].items() if k not in ("linearPerDimension", "tanhPerDimension")}
                                     for role in ("query", "catalog")} for split, rows in r["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
