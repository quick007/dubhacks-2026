"""One fixed genuine-feature CPU fit; evaluator inputs are never opened here."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from . import PARAMETERS
from .preflight import file_digest
from .runtime import write_json
from .scorer import FusionScorer, load_checkpoint, save_checkpoint
from .training import training_loss

FIT_CONFIG = {"seed": 728, "rank": 4, "dimension": 4096, "epochs": 80,
              "learningRate": .03, "weightDecay": .001, "cpuThreads": 1, "maxSecondsPerArm": 60,
              "selection": "fixed attempt; no validation/test outcome selection"}


def prior_control():
    model = FusionScorer(FIT_CONFIG["seed"])
    with torch.no_grad():
        model.interaction.zero_()
    model.interaction.requires_grad_(False)
    model.U.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 16394
    return model


def predictions(model, requests, queries, features):
    with torch.no_grad():
        return {rid: model(queries[rid], features[rid]).detach().clone() for rid in requests}


def fit_arm(arm, out, requests, labels, catalog, queries, features):
    if arm not in ("fusion", "prior"):
        raise ValueError("Unknown fixed fit arm")
    started = time.perf_counter()
    torch.set_num_threads(FIT_CONFIG["cpuThreads"])
    out = Path(out)
    model = prior_control() if arm == "prior" else FusionScorer(FIT_CONFIG["seed"])
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    initial = predictions(model, requests, queries, features)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=FIT_CONFIG["learningRate"], weight_decay=FIT_CONFIG["weightDecay"])
    first_gradients = None
    trace = []
    with (out / f"{arm}-epochs.jsonl").open("x") as handle:
        for epoch in range(FIT_CONFIG["epochs"]):
            if time.perf_counter() - started > FIT_CONFIG["maxSecondsPerArm"]:
                raise TimeoutError("Fixed CPU fit budget exceeded")
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, requests, labels, catalog, queries, features)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            gradients = {}
            for name, p in model.named_parameters():
                if p.requires_grad:
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        raise FloatingPointError(f"Missing/nonfinite trainable gradient: {name}")
                    gradients[name] = p.grad.norm().item()
            if first_gradients is None:
                first_gradients = gradients
                required = ("U", "V", "interaction", "prior") if arm == "fusion" else ("V", "prior")
                if any(gradients[name] <= 0 for name in required):
                    raise FloatingPointError("No genuine projection/head gradients")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite fitted parameter")
            row = {"epoch": epoch + 1, "lossBeforeStep": loss.item(), "gradientL2": gradients,
                   "elapsedSeconds": time.perf_counter() - started}
            trace.append(row)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
    if time.perf_counter() - started > FIT_CONFIG["maxSecondsPerArm"]:
        raise TimeoutError("Fixed CPU fit budget exceeded")
    deltas = {name: (p.detach() - before[name]).abs().sum().item() for name, p in model.named_parameters()}
    required = ("U", "V", "interaction", "prior") if arm == "fusion" else ("V", "prior")
    if any(deltas[name] <= 0 for name in required):
        raise FloatingPointError("Projection/interaction parameters did not change")
    if arm == "prior" and (deltas["U"] != 0 or torch.count_nonzero(model.interaction).item() != 0):
        raise AssertionError("Prior control learned a user interaction")
    final = predictions(model, requests, queries, features)
    prediction_change = sum((final[rid] - initial[rid]).abs().sum().item() for rid in requests)
    if prediction_change <= 0:
        raise FloatingPointError("Fitted predictions did not change")
    checkpoint = out / f"{arm}.pt"
    sha = save_checkpoint(model, checkpoint)
    restored = load_checkpoint(checkpoint, sha)
    reloaded = predictions(restored, requests, queries, features)
    if any(not torch.equal(final[rid], reloaded[rid]) for rid in requests):
        raise AssertionError("Checkpoint reload predictions differ")
    with torch.no_grad():
        final_loss = training_loss(restored, requests, labels, catalog, queries, features).item()
    if time.perf_counter() - started > FIT_CONFIG["maxSecondsPerArm"]:
        raise TimeoutError("Fixed CPU fit budget exceeded during checkpoint verification")
    report = {"arm": arm, "config": FIT_CONFIG, "trained": True, "architectureParameters": PARAMETERS,
              "trainableParameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "requests": len(requests), "lossBefore": trace[0]["lossBeforeStep"], "lossAfter": final_loss,
              "firstGradientL2": first_gradients, "parameterAbsoluteChange": deltas,
              "predictionAbsoluteChange": prediction_change, "seconds": time.perf_counter() - started,
              "checkpoint": checkpoint.name, "sha256": sha, "exactTrainingPredictionReload": True}
    write_json(out / f"{arm}-fit.json", report)
    write_json(out / f"{arm}-training-predictions.json", {
        rid: {"initial": initial[rid].tolist(), "fitted": final[rid].tolist()} for rid in requests})
    return restored, report


def freeze_checkpoints(out, fit_reports, train_hashes, evidence_source):
    out = Path(out)
    if set(fit_reports) != {"fusion", "prior"} or any(not r["trained"] or not r["exactTrainingPredictionReload"] for r in fit_reports.values()):
        raise ValueError("Both independently fitted/reloaded checkpoints required")
    random = FusionScorer(FIT_CONFIG["seed"])
    random_sha = save_checkpoint(random, out / "random.pt")
    checkpoints = {arm: {"file": f"{arm}.pt", "sha256": report["sha256"], "trained": True}
                   for arm, report in fit_reports.items()}
    checkpoints["random"] = {"file": "random.pt", "sha256": random_sha, "trained": False}
    receipt = {"frozenAtUtc": datetime.now(timezone.utc).isoformat(), "config": FIT_CONFIG,
               "evidenceSource": evidence_source, "trainHashes": train_hashes, "checkpoints": checkpoints,
               "heldoutLabelsOpenedBeforeFreeze": False}
    write_json(out / "CHECKPOINTS_FROZEN.json", receipt)
    require_frozen(out)
    return random.eval()


def require_frozen(out):
    out = Path(out)
    receipt = json.loads((out / "CHECKPOINTS_FROZEN.json").read_text())
    if receipt.get("config") != FIT_CONFIG or set(receipt.get("checkpoints", {})) != {"fusion", "prior", "random"}:
        raise ValueError("Fixed checkpoint freeze receipt required before evaluation")
    for arm, checkpoint in receipt["checkpoints"].items():
        if checkpoint["file"] != f"{arm}.pt" or file_digest(out / checkpoint["file"]) != checkpoint["sha256"]:
            raise ValueError("Frozen checkpoint changed before evaluation")
    return receipt
