"""CPU-only corpus/token inventory. Never imports MLX or loads model weights.

Example: python -m experiments.ranker.pretrained.catalog_fusion.preflight
Tokenization uses the already installed local tokenizer, with no downloads.
No labels, private truth, training optimizer or native forward is accessed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from . import DIMENSION, REPRESENTATION
from .inputs import canonical, catalog_text, digest, request_text

REVISION = "8b2b98c00a6b4d291155e4890773ca8f769aee53"
DEFAULT_MODEL = ".cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/" + REVISION
SPLITS = ("train", "validation", "test", "diagnostics")


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_corpus(dataset):
    """Read public content only; output separate roles and external membership."""
    dataset = Path(dataset).resolve()
    products = json.loads((dataset / "catalog.json").read_text())["products"]
    catalog = {p["id"]: p for p in products}
    if len(catalog) != len(products):
        raise ValueError("Duplicate public catalog IDs")
    rows = []
    for product in products:
        seen = set()
        for variant in product["presentationVariants"]:
            if variant["id"] in seen:
                raise ValueError("Duplicate presentation IDs")
            seen.add(variant["id"])
            rows.append({"role": "catalog", "member": [product["id"], variant["id"]],
                         "text": catalog_text(product, variant)})
    split_counts = {}
    hashes = {"catalog.json": file_digest(dataset / "catalog.json")}
    for split in SPLITS:
        relative = f"public/{split}.jsonl"
        path = dataset / relative
        if path.resolve() != path:
            raise ValueError("Public split paths cannot redirect through symlinks")
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len({r["recordId"] for r in records}) != len(records):
            raise ValueError("Duplicate public request IDs")
        candidates = {pid for r in records for pid in r["candidateProductIds"]}
        if not candidates.issubset(catalog):
            raise ValueError("Unknown public candidate")
        split_counts[split] = {"requests": len(records), "uniqueCandidateProducts": len(candidates),
                               "maxCandidates": max((len(r["candidateProductIds"]) for r in records), default=0)}
        rows.extend({"role": "request", "split": split, "member": r["recordId"], "text": request_text(r, catalog)} for r in records)
        hashes[relative] = file_digest(path)
    return rows, split_counts, hashes


def inventory(dataset, model, max_tokens=512):
    from tokenizers import Tokenizer

    started = time.perf_counter()
    model = Path(model).resolve()
    config = json.loads((model / "config.json").read_text())
    quantization = config.get("quantization") or config.get("quantization_config")
    if not quantization or quantization.get("bits") != 4:
        raise ValueError("Pinned four-bit local model configuration required")
    dimension = config.get("hidden_size") or config["text_config"]["hidden_size"]
    if dimension != DIMENSION or model.name != REVISION:
        raise ValueError("Wrong pinned model revision or dimension")
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    rows, splits, hashes = public_corpus(dataset)
    records = []
    for row in rows:
        ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
        if not ids or len(ids) > max_tokens:
            raise ValueError(f"{row['role']} input has {len(ids)} tokens; limit {max_tokens}. Refusing truncation.")
        records.append({key: value for key, value in row.items() if key != "text"} |
                       {"textSha256": digest(row["text"]), "tokenSha256": digest(canonical(ids)), "tokens": len(ids),
                        "batchShape": [1, len(ids)], "wholeSequenceForwardTokens": len(ids)})
    return {"representation": REPRESENTATION, "cpuOnly": True, "nativeForwards": 0,
            "modelPath": str(model), "modelRevision": REVISION, "quantization": quantization,
            "inputHashes": hashes, "tokenizerSha256": file_digest(model / "tokenizer.json"),
            "chatTemplateSha256": file_digest(model / "chat_template.jinja"),
            "promptFormat": "raw catalog_text/request_text strings; no chat wrapper or special-token insertion",
            "inputSourceSha256": file_digest(Path(__file__).with_name("inputs.py")),
            "splits": splits, "roles": dict(Counter(r["role"] for r in records)),
            "tokenRange": {role: [min(r["tokens"] for r in records if r["role"] == role),
                                  max(r["tokens"] for r in records if r["role"] == role)] for role in ("catalog", "request")},
            "totalTokens": sum(r["tokens"] for r in records),
            "rawFloat32FeatureBytes": len(records) * DIMENSION * 4,
            "cpuInventorySeconds": time.perf_counter() - started,
            "weightVerification": "not performed; local config/tokenizer only, no model weights loaded",
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/ranker/datasets/v2"))
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    report = inventory(args.dataset, args.model, args.max_tokens)
    if args.summary:
        report.pop("records")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
