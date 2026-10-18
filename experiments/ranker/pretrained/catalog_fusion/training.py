"""Train-only data boundary and objective; deliberately no optimizer/fit command."""
import hashlib
import json
from pathlib import Path

import torch

from .scorer import roster


def read_training_inputs(dataset):
    """Only open public/train.jsonl and labels/train.jsonl; reject redirection."""
    root = Path(dataset).resolve()
    records = []
    hashes = {}
    for relative in ("public/train.jsonl", "labels/train.jsonl"):
        path = root / relative
        if path.resolve() != path:
            raise ValueError("Training split paths cannot redirect through symlinks")
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        by_id = {row["recordId"]: row for row in rows}
        if not by_id or len(by_id) != len(rows):
            raise ValueError("Training records must be nonempty and unique")
        records.append(by_id)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    if set(records[0]) != set(records[1]):
        raise ValueError("Training public/label membership mismatch")
    return *records, hashes


def training_loss(model, requests, labels, catalog, queries, features):
    """Same best-product CE / informative compatible-presentation CE definitions.

    This function trusts its explicitly train-only caller. No held-out data access
    or automatic fit occurs here. Validate native feature receipts before use.
    """
    if not requests or any(set(mapping) != set(requests) for mapping in (labels, queries, features)):
        raise ValueError("Exact training request/label/feature membership required")
    product_losses, presentation_losses = [], []
    for rid, request in requests.items():
        actions = roster(request, catalog)
        values = model(queries[rid], features[rid])
        if len(values) != len(actions):
            raise ValueError("Complete training feature roster required")
        pids = request["candidateProductIds"]
        logits = torch.stack([values[[i for i, a in enumerate(actions) if a[0] == pid], 0].mean() for pid in pids])
        product_losses.append(torch.nn.functional.cross_entropy(logits[None], torch.tensor([pids.index(labels[rid]["bestProductId"])])))
        for pid in pids:
            indices = [i for i, (product, vid) in enumerate(actions) if product == pid
                       and request["presentationCompatibility"].get(pid, {}).get(vid, False)]
            targets = [labels[rid]["presentationScores"][pid][actions[i][1]] for i in indices]
            if len(targets) > 1 and max(targets) > 0 and len(set(targets)) > 1:
                presentation_losses.append(torch.nn.functional.cross_entropy(values[indices, 1][None], torch.tensor([targets.index(max(targets))])))
    return torch.stack(product_losses).mean() + (torch.stack(presentation_losses).mean() if presentation_losses else 0)
