"""Read-only verified native inputs for offline CPU ablations; no model loads."""
import json
from pathlib import Path

import torch

from .inputs import catalog_text, request_text
from .paired_contract import verify_freeze
from .preflight import file_digest
from .runtime import FeatureStore
from .scorer import roster


class CachedPairedInputs:
    def __init__(self, dataset, queries, catalog_run):
        self.dataset, self.query_run, self.catalog_run = Path(dataset).resolve(), Path(queries), Path(catalog_run)
        self.freeze = verify_freeze(self.dataset, public_only=True)
        query_report = json.loads((self.query_run / "report.json").read_text())
        if query_report["status"] != "completed" or query_report["config"]["reviewedDataFreezeSha256"] != file_digest(self.dataset / "FREEZE.json"):
            raise ValueError("Completed raw-query run must bind this exact data freeze")
        prepared = json.loads((self.query_run / "paired-preflight.json").read_text())
        old = json.loads((self.catalog_run / "preflight.json").read_text())
        if prepared["priorReportSha256"] != file_digest(self.catalog_run / "report.json"):
            raise ValueError("Catalog source receipt changed")
        if file_digest(self.dataset / "catalog.json") != old["inputHashes"]["catalog.json"]:
            raise ValueError("Cached catalog content changed")
        self.provenance = json.loads((self.catalog_run / "model-verification.json").read_text())
        self.qstore = FeatureStore.open_existing(self.query_run / "features", self.provenance)
        self.cstore = FeatureStore.open_existing(self.catalog_run / "features", self.provenance)
        self.catalog = {p["id"]: p for p in json.loads((self.dataset / "catalog.json").read_text())["products"]}
        self.requests = {split: {r["recordId"]: r for r in map(json.loads, (self.dataset / f"public/{split}.jsonl").read_text().splitlines())}
                         for split in ("train", "validation", "test")}
        self.catalog_rows = {}
        for row in old["records"]:
            if row["role"] == "catalog":
                pid, vid = row["member"]
                product = self.catalog[pid]
                variant = next(v for v in product["presentationVariants"] if v["id"] == vid)
                self.catalog_rows[pid, vid] = {**row, "text": catalog_text(product, variant)}
        self.query_rows = {}
        for row in prepared["queryRecords"]:
            rid, split = row["member"], row["split"]
            if rid in self.query_rows or rid not in self.requests[split]:
                raise ValueError("Raw native request membership mismatch")
            self.query_rows[rid] = {**row, "text": request_text(self.requests[split][rid], self.catalog)}
        if set(self.query_rows) != {rid for records in self.requests.values() for rid in records}:
            raise ValueError("Raw native request artifact membership incomplete")
        self.source_receipts = {"dataFreezeSha256": file_digest(self.dataset / "FREEZE.json"),
                                "queryReportSha256": file_digest(self.query_run / "report.json"),
                                "catalogReportSha256": file_digest(self.catalog_run / "report.json"),
                                "evidenceSource": self.provenance["evidenceSource"]}

    def load_split(self, split):
        """Only this split's vectors are read; unique variants are not repeated."""
        requests = self.requests[split]
        query, items, receipts = {}, {}, {"queries": {}, "catalog": {}}
        for rid in sorted(requests):
            value, receipt = self.qstore.load(self.query_rows[rid])
            query[rid] = value[0]
            receipts["queries"][rid] = {"rawKey": receipt["rawKeys"][0], "sha256": receipt["sha256"]}
        required = sorted({action for request in requests.values() for action in roster(request, self.catalog)})
        for action in required:
            value, receipt = self.cstore.load(self.catalog_rows[action])
            items[action] = value[0]
            receipts["catalog"]["/".join(action)] = {"rawKey": receipt["rawKeys"][0], "sha256": receipt["sha256"]}
        features = {rid: torch.stack([items[a] for a in roster(request, self.catalog)]) for rid, request in requests.items()}
        return query, features, items, receipts
