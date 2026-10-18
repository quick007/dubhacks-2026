"""Data-only freeze checks; no feature extraction, model calls or fitting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ..boundary import guard_reasons
from .fresh_eval_data import eval_teacher, freeze_dataset, make_catalog, verify_dataset
from .inputs import catalog_text, request_text
from .preflight import file_digest
from .scorer import assemble_grid, roster


def fixture_files(root):
    source = root / "model";source.mkdir()
    checks = {}
    for arm in ("fusion", "prior", "random"):
        path = source / f"{arm}.pt";path.write_bytes(("synthetic test bytes " + arm).encode())
        checks[arm] = {"file": path.name, "sha256": file_digest(path), "trained": arm != "random"}
    image = source / "centered-image.pt";image.write_bytes(b"synthetic image test bytes")
    (source / "CHECKPOINTS_FROZEN.json").write_text(json.dumps({"checkpoints": checks,
        "metadata": {"imageCheckpoint": {"file": image.name, "sha256": file_digest(image)}, "statisticsSha256": "a" * 64}}))
    old = root / "old.json";old.write_text('{"products": []}')
    return source, old


class FreshEvaluationDataTests(unittest.TestCase):
    def frozen(self, root):
        source, old = fixture_files(root)
        out = root / "eval"
        freeze = freeze_dataset(out, source, old)
        catalog = {p["id"]: p for p in json.loads((out / "catalog.json").read_text())["products"]}
        requests = {r["recordId"]: r for r in map(json.loads, (out / "public/requests.jsonl").read_text().splitlines())}
        labels = {r["recordId"]: r for r in map(json.loads, (out / "private/truth.jsonl").read_text().splitlines())}
        controls = json.loads((out / "private/controls.json").read_text())
        return out, freeze, catalog, requests, labels, controls

    def test_new_catalog_signatures_and_opaque_identity_separation(self):
        products = make_catalog()["products"]
        self.assertEqual(len(products), 36)
        self.assertEqual(sum(len(p["tags"]) == 2 for p in products), 21)
        self.assertEqual(sum(len(p["tags"]) == 3 for p in products), 15)
        self.assertEqual(len({tuple(p["tags"]) for p in products}), 36)
        for product in products:
            self.assertEqual(len(product["presentationVariants"]), 2)
            for variant in product["presentationVariants"]:
                text = catalog_text(product, variant)
                self.assertNotIn(product["id"], text);self.assertNotIn(variant["id"], text)
                self.assertNotIn(variant["imageKey"], text)

    def test_full_candidate_teacher_unique_and_both_grid_sizes_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            _, freeze, cat, requests, labels, _ = self.frozen(Path(temp))
            self.assertEqual(freeze["eligibleCounts"], [27, 33])
            self.assertEqual(len(requests), 24)
            for rid, request in requests.items():
                self.assertEqual(eval_teacher(request, cat), labels[rid])
                actions = roster(request, cat)
                self.assertEqual(len(actions), 72)
                # Raw overlap can favor blocked candidates; assembly must mask
                # stock, delivered-price and compatibility using current data.
                scores = torch.tensor([[len(set(request["context"]["statedPreferences"]) & set(cat[p]["tags"])),
                                        labels[rid]["presentationScores"][p][v]] for p, v in actions], dtype=torch.float32)
                grids = [assemble_grid(request, cat, actions, scores, k) for k in (12, 24)]
                self.assertEqual(grids[0]["cards"], grids[1]["cards"][:12])
                for grid, count in zip(grids, (12, 24), strict=True):
                    self.assertEqual(len(grid["rankedProducts"]), 36)
                    self.assertEqual(grid["status"], "complete");self.assertEqual(len(grid["cards"]), count)
                    self.assertEqual(grid["cards"][0]["productId"], labels[rid]["bestProductId"])
                    self.assertTrue(all(not guard_reasons(request, cat[c["productId"]]) for c in grid["cards"]))

    def test_factorial_pairs_control_masks_and_changed_choices(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, reqs, labels, controls = self.frozen(Path(temp))
            for family in controls["families"]:
                first = reqs[family["recordIds"][0]]
                for rid in family["recordIds"]:
                    q = reqs[rid]
                    for key in ("candidateProductIds", "presentationCompatibility", "history", "asOf"):
                        self.assertEqual(q[key], first[key])
                    for key in ("statedBudget", "statedRequirements", "task", "presentationSurface"):
                        self.assertEqual(q["context"][key], first["context"][key])
            for pair in controls["pairs"]:
                a, b = [labels[rid] for rid in pair["recordIds"]]
                if pair["kind"] == "product":
                    self.assertNotEqual(a["bestProductId"], b["bestProductId"])
                else:
                    self.assertEqual(a["bestProductId"], b["bestProductId"])
                    for pid in pair["productIds"]:
                        self.assertNotEqual(max(a["presentationScores"][pid], key=a["presentationScores"][pid].get),
                                            max(b["presentationScores"][pid], key=b["presentationScores"][pid].get))
            self.assertEqual(set(controls["permutedQuerySource"]), set(reqs))
            self.assertTrue(all(a != b for a, b in controls["permutedQuerySource"].items()))

    def test_public_verification_never_reads_private_targets_and_roster_not_encoded(self):
        with tempfile.TemporaryDirectory() as temp:
            out, _, cat, requests, _, _ = self.frozen(Path(temp))
            original = Path.open
            def guard(path, *args, **kwargs):
                if path.is_relative_to(out.resolve()) and "private" in path.relative_to(out.resolve()).parts:
                    raise AssertionError("Private target read during public verification")
                return original(path, *args, **kwargs)
            with patch.object(Path, "open", guard):verify_dataset(out, private=False)
            for q in requests.values():
                text = request_text(q, cat)
                self.assertNotIn(q["recordId"], text)
                self.assertFalse(any(pid in text for pid in q["candidateProductIds"]))
                self.assertNotIn("bestProductId", text)
                changed = copy.deepcopy(q);changed["recordId"] = "different-id";changed["candidateProductIds"].reverse()
                self.assertEqual(text, request_text(changed, cat))

    def test_position_balance_and_requirement_shortfall_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            _, freeze, _, _, _, controls = self.frozen(Path(temp))
            self.assertEqual(len(freeze["aggregateBestPositionCountsThreeViews"]), 36)
            self.assertEqual(set(freeze["aggregateBestPositionCountsThreeViews"].values()), {2})
            self.assertEqual(controls["mandatoryRosterRotationOffsets"], [0, 12, 24])
            self.assertTrue(all(p["expectedEligible"] < 24 for p in controls["requirementGuardProbes"]))
            self.assertEqual(freeze["totalNativeForwards"], 96)

    def test_frozen_mutation_and_output_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            out, _, _, _, _, _ = self.frozen(Path(temp))
            verify_dataset(out, private=True)
            with self.assertRaises(FileExistsError):freeze_dataset(out, Path(temp) / "model", Path(temp) / "old.json")
            with (out / "public/requests.jsonl").open("a") as h:h.write("\n")
            with self.assertRaises(ValueError):verify_dataset(out)


if __name__ == "__main__":unittest.main()
