"""Freeze public-fact-only counterfactual supervision, without features or fitting."""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

from ..boundary import guard_reasons
from .inputs import canonical, digest
from .preflight import file_digest
from .runtime import write_json

VERSION = "catalog-public-counterfactual-v1"
STYLES = ("detail-view", "in-use-view")
ATOMS = ("compact", "durable", "laptop-sleeve", "lightweight", "organizer", "roomy", "water-resistant")
SCHEDULE = {
    "train": [((ATOMS[i],), (ATOMS[(i + 1) % len(ATOMS)],)) for i in range(len(ATOMS))],
    "validation": [(("durable", "roomy"), ("laptop-sleeve", "organizer")),
                   (("lightweight", "compact"), ("water-resistant", "roomy"))],
    "test": [(("durable", "compact"), ("laptop-sleeve", "roomy")),
             (("lightweight", "roomy"), ("water-resistant", "organizer")),
             (("durable", "organizer"), ("water-resistant", "compact"))],
}


def teacher(request, catalog):
    """Equal positive tag overlap + exact compatible style, with no opaque tie."""
    context = request["context"]
    preferences = set(context["statedPreferences"])
    if not preferences or not preferences.issubset(ATOMS) or len(context["presentationPreferences"]) != 1:
        raise ValueError("Declared public tag and one style preferences required")
    style = context["presentationPreferences"][0]
    relevance, images = {}, {}
    for pid in request["candidateProductIds"]:
        product = catalog[pid]
        if guard_reasons(request, product):
            raise ValueError("Counterfactual supervision requires every candidate eligible")
        relevance[pid] = len(preferences & set(product["tags"]))
        images[pid] = {}
        compatible_styles = []
        for variant in product["presentationVariants"]:
            compatible = request["presentationCompatibility"][pid].get(variant["id"]) is True
            compatible &= variant.get("compatibleSurfaces") is None or context["presentationSurface"] in variant["compatibleSurfaces"]
            if compatible:
                compatible_styles.append(variant["presentationStyle"])
            images[pid][variant["id"]] = (3 if variant["presentationStyle"] == style else 1) if compatible else 0
        if Counter(compatible_styles) != Counter(STYLES):
            raise ValueError("Both styles must be genuinely compatible once per candidate")
    best = max(relevance.values())
    winners = [pid for pid, value in relevance.items() if value == best]
    if best <= 0 or len(winners) != 1:
        raise ValueError("Teacher needs a unique factual winner; never resolve ties by ID")
    return {"recordId": request["recordId"], "bestProductId": winners[0],
            "relevance": relevance, "presentationScores": images}


def select_pool(catalog, allowed, preferences, family, seen):
    eligible = [pid for pid in allowed if catalog[pid]["inStock"] and
                Counter(v["presentationStyle"] for v in catalog[pid]["presentationVariants"]
                        if "detail-panel" in v.get("compatibleSurfaces", [])) == Counter(STYLES)]
    # Selection randomness is frozen external mapping, never an encoded feature.
    eligible.sort(key=lambda pid: digest(canonical([VERSION, family, pid])))
    left, right = [set(p) for p in preferences]
    candidates_a = [pid for pid in eligible if left.issubset(catalog[pid]["tags"]) and not right.issubset(catalog[pid]["tags"])]
    candidates_b = [pid for pid in eligible if right.issubset(catalog[pid]["tags"]) and not left.issubset(catalog[pid]["tags"])]
    for a, b in itertools.product(candidates_a, candidates_b):
        if a == b:
            continue
        distractors = [pid for pid in eligible if pid not in (a, b) and
                       not left.issubset(catalog[pid]["tags"]) and not right.issubset(catalog[pid]["tags"])]
        for extra in itertools.combinations(distractors, 2):
            pool = [a, b, *extra]
            membership = tuple(sorted(pool))
            if membership not in seen:
                seen.add(membership)
                # Rotate slots deterministically; the four factorial queries keep
                # this exact order and all masks. No position predicts an arm.
                rotate = family % 4
                return pool[rotate:] + pool[:rotate]
    raise ValueError(f"No eligible unique public-fact pool for {preferences}")


def freeze_dataset(source, out):
    source, out = Path(source).resolve(), Path(out)
    catalog_bytes = (source / "catalog.json").read_bytes()
    products = json.loads(catalog_bytes)["products"]
    catalog = {p["id"]: p for p in products}
    old_records = {}
    allowed, old_signatures, old_ids = {}, set(), set()
    for split in SCHEDULE:
        records = [json.loads(line) for line in (source / f"public/{split}.jsonl").read_text().splitlines() if line.strip()]
        old_records[split] = records
        allowed[split] = {pid for r in records for pid in r["candidateProductIds"]}
        for row in records:
            old_ids.add(row["recordId"])
            old_signatures.add((tuple(sorted(row["context"].get("statedPreferences") or [])),
                                tuple(sorted(row["context"].get("presentationPreferences") or []))))
    if any(allowed[a] & allowed[b] for a, b in itertools.combinations(SCHEDULE, 2)):
        raise ValueError("Source train/validation/test product pools overlap")
    out.mkdir(parents=True, exist_ok=False)
    (out / "public").mkdir()
    (out / "labels").mkdir()
    with (out / "catalog.json").open("xb") as handle:
        handle.write(catalog_bytes)  # Exact source bytes preserve all catalog keys.
    pairs, families, controls, signatures = [], [], {}, {}
    positive_atoms = Counter()
    old_overlap = Counter()
    for split, schedule in SCHEDULE.items():
        records, labels, split_families, seen = [], [], [], set()
        signatures[split] = set()
        for index, preferences in enumerate(schedule):
            family = f"cf1-family-{split}-{digest(canonical([VERSION, split, preferences]))[:12]}"
            pool = select_pool(catalog, allowed[split], preferences, index, seen)
            budget = max(catalog[pid]["price"] + catalog[pid].get("shippingCost", 0) for pid in pool) + 10
            ids, expected = {}, {}
            for choice, preference in enumerate(preferences):
                for style in STYLES:
                    rid = "cf1-request-" + digest(canonical([family, sorted(preference), style]))[:18]
                    if rid in old_ids:
                        raise ValueError("New record collides with old public record")
                    context = {"task": "Choose a travel backpack using my stated positive attributes; show the requested image style.",
                               "statedBudget": budget, "statedRequirements": [], "statedPreferences": sorted(preference),
                               "presentationSurface": "detail-panel", "presentationPreferences": [style]}
                    request = {"recordId": rid, "asOf": "2026-10-18T00:00:00Z", "candidateProductIds": pool[:],
                               "context": context, "history": [],
                               "presentationCompatibility": {pid: {v["id"]: True for v in catalog[pid]["presentationVariants"]} for pid in pool}}
                    label = teacher(request, catalog)
                    ids[choice, style], expected[choice, style] = rid, label
                    records.append(request)
                    labels.append(label)
                    signature = (tuple(sorted(preference)), (style,))
                    signatures[split].add(signature)
                    old_overlap[split] += signature in old_signatures
                    if split == "train":
                        positive_atoms.update(preference)
            for style in STYLES:
                a, b = expected[0, style], expected[1, style]
                if a["bestProductId"] == b["bestProductId"]:
                    raise ValueError("Preference intervention failed to change unique best product")
                pairs.append({"familyId": family, "split": split, "kind": "product", "recordIds": [ids[0, style], ids[1, style]],
                              "expectedBestProductIds": [a["bestProductId"], b["bestProductId"]]})
            for choice in (0, 1):
                a, b = (expected[choice, style] for style in STYLES)
                if a["bestProductId"] != b["bestProductId"]:
                    raise ValueError("Style-only intervention changed product teacher")
                pairs.append({"familyId": family, "split": split, "kind": "image", "recordIds": [ids[choice, style] for style in STYLES],
                              "expectedBestProductId": a["bestProductId"], "productIds": pool[:]})
            split_families.append(family)
            families.append({"familyId": family, "split": split, "candidateProductIds": pool,
                             "positivePreferenceSets": [sorted(p) for p in preferences],
                             "recordIds": [ids[choice, style] for choice in (0, 1) for style in STYLES]})
        for kind, rows in (("public", records), ("labels", labels)):
            with (out / kind / f"{split}.jsonl").open("x") as handle:
                handle.writelines(canonical(row) + "\n" for row in rows)
        controls[split] = {"requests": len(records), "pairFamilies": len(split_families),
                           "productPairs": 2 * len(split_families), "imagePairs": 2 * len(split_families),
                           "candidateProductsUsed": len({pid for row in records for pid in row["candidateProductIds"]}),
                           "allowedOldSplitProducts": len(allowed[split]),
                           "bestPositionCounts": dict(Counter(row["candidateProductIds"].index(label["bestProductId"]) for row, label in zip(records, labels, strict=True)))}
    if set(positive_atoms) != set(ATOMS) or any(value <= 0 for value in positive_atoms.values()):
        raise ValueError("All seven attribute atoms need positive training coverage")
    if any(signatures[a] & signatures[b] for a, b in itertools.combinations(SCHEDULE, 2)):
        raise ValueError("New split preference/style combinations overlap")
    write_json(out / "paired-controls.json", {"version": VERSION, "families": families, "pairs": pairs})
    relative = ["catalog.json", "paired-controls.json", *(f"{kind}/{split}.jsonl" for kind in ("public", "labels") for split in SCHEDULE)]
    receipt = {"version": VERSION, "status": "frozen-data-only; no feature extraction or fitting", "teacher": {
        "product": "unique maximum unweighted overlap of public stated positive tags and public product tags; all candidates eligible",
        "presentation": "3 for requested exact compatible presentationStyle, 1 for other compatible style; both styles valid",
        "usesPrivateTruth": False, "usesOldRelevanceOrImageLabels": False, "idTieBreaking": False},
        "sourceCatalogSha256": file_digest(source / "catalog.json"), "catalogUnchanged": file_digest(out / "catalog.json") == file_digest(source / "catalog.json"),
        "sourcePublicHashes": {split: file_digest(source / f"public/{split}.jsonl") for split in SCHEDULE},
        "generatorSha256": file_digest(__file__), "files": {name: file_digest(out / name) for name in relative},
        "splits": controls, "positiveTrainingAtomQueryCounts": dict(positive_atoms),
        "preferenceStyleSignatures": {split: [[list(p), list(s)] for p, s in sorted(values)] for split, values in signatures.items()},
        "newQuerySignatureOccurrencesAlsoInOldPublic": dict(old_overlap),
        "crossSplitProductMembershipDisjoint": True, "crossSplitFamilyMembershipDisjoint": True,
        "crossSplitPreferenceStyleSignaturesDisjoint": True,
        "nativeQueryForwardsRequired": sum(row["requests"] for row in controls.values()), "nativeCatalogForwardsRequired": 0,
        "reviewRequiredBeforeFit": True,
        "limitations": ["Public-fact teacher task, not latent preference or human utility ground truth.",
                        "Uses old train/validation/test catalog partitions; old test outcomes already inspected/replayed.",
                        "Fresh request/family holdouts are not an untouched catalog test or a sealed independent evaluation.",
                        "All old catalog bytes must pass provenance/key/checksum verification before vector reuse."]}
    write_json(out / "FREEZE.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/ranker/datasets/v2"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_dataset(args.source, args.out)
    print(json.dumps({key: result[key] for key in ("splits", "positiveTrainingAtomQueryCounts", "nativeQueryForwardsRequired", "newQuerySignatureOccurrencesAlsoInOldPublic", "catalogUnchanged")}, indent=2))


if __name__ == "__main__":
    main()
