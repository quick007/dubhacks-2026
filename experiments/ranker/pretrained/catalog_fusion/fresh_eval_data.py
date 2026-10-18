"""Freeze an evaluation-only new catalog before features or model predictions."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

from ..boundary import guard_reasons
from .inputs import canonical, catalog_text, digest, request_text
from .paired_data import ATOMS, STYLES
from .preflight import file_digest
from .runtime import write_json

VERSION = "fresh-seven-atom-grid-evaluation-v1"
NAMES = ("Estuary", "Cinder", "Rill", "Plover", "Marsh", "Kestrel", "Sora", "Fen", "Tern",
         "Bracken", "Islet", "Osprey", "Shale", "Tamarack", "Hollow", "Sedge", "Drift", "Cove",
         "Lichen", "Moraine", "Gannet", "Flint", "Aster", "Dune", "Sorrel", "Cairn", "Grove",
         "Wren", "Reed", "Fjord", "Rowan", "Vale", "Spruce", "Briar", "Inlet", "Moss")
WORDING = (
    "Help me compare travel backpacks for the three positive attributes I listed. Keep the delivered cost within my budget and use the image style I requested.",
    "I am browsing a new set of packs. Rank them by how many of my three listed attributes they have, and show the requested kind of photograph.",
    "For this trip, all three stated attributes matter equally. Choose from the available backpacks within my total price limit, with the view I selected.",
    "Please sort these travel packs using the three features in my preferences. Include shipping in the price limit and use my preferred picture style.",
    "I want to compare backpacks against my three equally important attributes. Respect availability and the delivered-price limit, and show the selected view.",
    "Build a backpack comparison from my three positive feature preferences. Keep the total purchase cost affordable and display the requested image style.")


def make_catalog():
    pairs = list(itertools.combinations(ATOMS, 2))
    triples = sorted(itertools.combinations(ATOMS, 3), key=lambda t: digest(canonical([VERSION, t])))[:15]
    signatures = pairs + triples
    products = []
    for i, tags in enumerate(signatures):
        pid = "fresh-product-" + digest(canonical([VERSION, "product", i]))[:18]
        capacity = f"{17 + i}.3L"
        title = f"{NAMES[i]} {capacity} Travel Backpack"
        price, shipping = ((120, 12) if i < 3 else (145, 9)) if i < 6 else (82 + (i % 6) * 5, 5)
        product = {"id": pid, "title": title, "category": "backpacks", "price": price,
                   "shippingCost": shipping, "inStock": i not in (6, 7, 8), "tags": list(tags),
                   "description": f"{title}. This new catalog model has the public features: {', '.join(tags)}. Capacity {capacity}. Color {('slate','ochre','pine','clay','plum','sand')[i % 6]}.",
                   "presentationVariants": []}
        for style in STYLES:
            vid = "fresh-view-" + digest(canonical([VERSION, i, style]))[:18]
            description = ("A close view of the backpack construction and listed features." if style == "detail-view"
                           else "The backpack worn during ordinary travel, with its shape and scale visible.")
            product["presentationVariants"].append({"id": vid, "title": title,
                "imageDescription": f"{description} {title}.", "facts": list(tags),
                "presentationStyle": style, "compatibleSurfaces": ["detail-panel", "product-grid"],
                "imageKey": f"synthetic/fresh-v1/{i}/{style}"})
        products.append(product)
    assert len(products) == 36 and len({tuple(p["tags"]) for p in products}) == 36
    return {"schemaVersion": 1, "synthetic": True, "products": products}


def eval_teacher(request, catalog):
    """Public-fact ranking target; unique eligible triple beats pair distractors."""
    prefs = set(request["context"]["statedPreferences"])
    relevance, images, eligible = {}, {}, []
    for pid in request["candidateProductIds"]:
        product = catalog[pid]
        valid = not guard_reasons(request, product)
        if valid:
            eligible.append(pid)
        relevance[pid] = len(prefs & set(product["tags"])) if valid else 0
        images[pid] = {v["id"]: 3 if v["presentationStyle"] in request["context"]["presentationPreferences"] else 1
                       for v in product["presentationVariants"]}
    best = max(relevance.values())
    winners = [pid for pid, value in relevance.items() if value == best]
    if len(prefs) != 3 or best != 3 or len(winners) != 1 or len(eligible) < 24:
        raise ValueError("Fresh evaluation requires a unique eligible triple winner and at least 24 eligible products")
    return {"recordId": request["recordId"], "bestProductId": winners[0], "relevance": relevance,
            "presentationScores": images, "eligibleProductIds": sorted(eligible),
            "maskReasons": {pid: guard_reasons(request, catalog[pid]) for pid in request["candidateProductIds"]}}


def freeze_dataset(out, semantic_run, old_catalog):
    out, semantic_run = Path(out), Path(semantic_run)
    old_products = json.loads(Path(old_catalog).read_text())["products"]
    payload = make_catalog()
    catalog = {p["id"]: p for p in payload["products"]}
    if set(catalog) & {p["id"] for p in old_products} or {p["title"] for p in catalog.values()} & {p["title"] for p in old_products}:
        raise ValueError("New catalog identity/title collision")
    old_texts = {catalog_text(p, v) for p in old_products for v in p["presentationVariants"]}
    if any(catalog_text(p, v) in old_texts for p in catalog.values() for v in p["presentationVariants"]):
        raise ValueError("Reused old catalog encoder text")
    source_freeze = json.loads((semantic_run / "CHECKPOINTS_FROZEN.json").read_text())
    source = {"semanticRun": str(semantic_run.resolve()), "freezeSha256": file_digest(semantic_run / "CHECKPOINTS_FROZEN.json"),
              "checkpoints": source_freeze["checkpoints"], "image": source_freeze["metadata"]["imageCheckpoint"],
              "statisticsSha256": source_freeze["metadata"]["statisticsSha256"]}
    for checkpoint in source["checkpoints"].values():
        if file_digest(semantic_run / checkpoint["file"]) != checkpoint["sha256"]:
            raise ValueError("Frozen semantic checkpoint changed")
    if file_digest(semantic_run / "centered-image.pt") != source["image"]["sha256"]:
        raise ValueError("Frozen image checkpoint changed")
    requests, labels, families, pairs = [], [], [], []
    triples = [p for p in catalog.values() if len(p["tags"]) == 3]
    for family in range(6):
        target_products = triples[2 * family:2 * family + 2]
        target_ids = [p["id"] for p in target_products]
        order = sorted((pid for pid in catalog if pid not in target_ids), key=lambda p: digest(canonical([VERSION, family, p])))
        # Three mandatory rotated views move these twelve balanced base target
        # positions through all 36 slots without changing any encoder input.
        for j, pid in enumerate(target_ids):
            order.insert(2 * family + j, pid)
        ids = {}
        budget = 125 if family % 2 == 0 else 185
        for choice, product in enumerate(target_products):
            for style in STYLES:
                rid = "fresh-request-" + digest(canonical([VERSION, family, choice, style]))[:18]
                request = {"recordId": rid, "asOf": "2026-10-18T01:00:00Z", "candidateProductIds": order[:],
                           "context": {"task": WORDING[family], "statedBudget": budget,
                                       "statedRequirements": [], "statedPreferences": product["tags"][:],
                                       "presentationSurface": "detail-panel", "presentationPreferences": [style]}, "history": [],
                           "presentationCompatibility": {pid: {v["id"]: True for v in p["presentationVariants"]} for pid, p in catalog.items()}}
                label = eval_teacher(request, catalog)
                if label["bestProductId"] != product["id"]:
                    raise ValueError("Construction did not preserve unique teacher winner")
                requests.append(request);labels.append(label);ids[choice, style] = rid
        family_id = f"fresh-family-{family}"
        families.append({"familyId": family_id, "recordIds": list(ids.values()), "candidateProductIds": order,
                         "preferenceSets": [p["tags"] for p in target_products]})
        for style in STYLES:
            pairs.append({"familyId": family_id, "kind": "product", "recordIds": [ids[0, style], ids[1, style]],
                          "expectedBestProductIds": target_ids})
        for choice in (0, 1):
            pairs.append({"familyId": family_id, "kind": "image", "recordIds": [ids[choice, s] for s in STYLES],
                          "expectedBestProductId": target_ids[choice], "productIds": order[:]})
    by_id = {q["recordId"]: q for q in requests}
    # These public request transforms test the guard using deterministic score
    # fixtures only. They are not scored with stale native request embeddings.
    requirement_probes = [{"sourceRecordId": requests[0]["recordId"], "statedRequirements": [atom],
                           "requestedCards": 24, "expectedEligible": sum(not guard_reasons({**requests[0], "context": {**requests[0]["context"], "statedRequirements": [atom]}}, p) for p in catalog.values()),
                           "scope": "CPU guard/shortfall probe only; no learned/native quality claim"} for atom in ATOMS]
    permutation = {rid: other for pair in pairs if pair["kind"] == "product" for rid, other in (pair["recordIds"], list(reversed(pair["recordIds"])))}
    controls = {"families": families, "pairs": pairs, "permutedQuerySource": permutation,
                "requestedCardCounts": [12, 24], "mandatoryRosterRotationOffsets": [0, 12, 24],
                "rotationPolicy": "rotate complete candidate roster and feature rows together; encoder texts/IDs map unchanged; aggregate all three mandatory views",
                "requirementGuardProbes": requirement_probes,
                "arms": ["frozen semantic", "frozen independently trained item prior", "frozen seeded random", "paired-opposite-preference query", "nonlearned public-tag oracle"],
                "negativeControls": ["image-style intervention leaves product teacher unchanged", "12-card grid equals prefix of 24-card grid", "three roster rotations preserve ID-mapped scores/actions", "known stock/budget/shipping/requirements guards"]}
    positions = Counter()
    for label in labels:
        base = by_id[label["recordId"]]["candidateProductIds"].index(label["bestProductId"])
        for offset in controls["mandatoryRosterRotationOffsets"]:
            positions[(base - offset) % 36] += 1
    if len(positions) != 36 or len(set(positions.values())) != 1:
        raise ValueError("Mandatory readout views must balance teacher best positions across all 36 slots")
    out.mkdir(parents=True, exist_ok=False)
    (out / "public").mkdir();(out / "private").mkdir()
    write_json(out / "catalog.json", payload)
    for name, rows in (("public/requests.jsonl", requests), ("private/truth.jsonl", labels)):
        with (out / name).open("x") as handle:
            for row in rows:handle.write(canonical(row) + "\n")
    write_json(out / "private/controls.json", controls)
    freeze = {"version": VERSION, "createdAtUtc": datetime.now(timezone.utc).isoformat(),
              "purpose": "evaluation-only; no training split or model fitting", "status": "frozen before feature extraction/predictions; pending root data review",
              "sourceCheckpoints": source, "generatorSha256": file_digest(__file__),
              "oldCatalogSha256": file_digest(old_catalog), "oldCatalogIdentityAndEncoderTextsDisjoint": True,
              "newSkuCount": 36, "uniqueTwoTagSignatures": 21, "uniqueThreeTagSignatures": 15,
              "newCatalogVariantForwards": 72, "freshQueryForwards": 24, "totalNativeForwards": 96,
              "requests": 24, "families": 6, "productPairs": 12, "imagePairs": 12,
              "allCandidatePoolSize": 36, "allCandidateVariants": 72, "requestedCardCounts": [12, 24],
              "eligibleCounts": sorted({len(label["eligibleProductIds"]) for label in labels}),
              "aggregateBestPositionCountsThreeViews": {str(k): positions[k] for k in sorted(positions)},
              "primaryBestPositionCounts": dict(Counter(by_id[l["recordId"]]["candidateProductIds"].index(l["bestProductId"]) for l in labels)),
              "requestTextSha256": {q["recordId"]: digest(request_text(q, catalog)) for q in requests},
              "queryVocabulary": list(ATOMS), "queryPreferenceCardinality": 3,
              "featureBoundary": "only catalog_text/request_text canonical allowlists; no private file, target, family/record/product/variant ID or candidate slot in encoder text",
              "files": {name: file_digest(out / name) for name in ("catalog.json", "public/requests.jsonl", "private/truth.jsonl", "private/controls.json")},
              "limitations": ["New synthetic SKUs/wording/compositions, seven-atom public teacher only; no real-user or conversion evidence.",
                              "Sealed from training and model-output adjustment; generator and root reviewer can see public facts/teacher labels.",
                              "Head/grid component only; no complete page, learned sections/copy or above-fold/full-storefront result.",
                              "Nonempty requirements cannot fulfill24 on this catalog; separate explicit CPU guard shortfall probes.",
                              "No features, fitting, model prediction, native timing or quality result produced by this freeze."]}
    write_json(out / "FREEZE.json", freeze)
    return freeze


def verify_dataset(path, *, private=False):
    path = Path(path).resolve()
    freeze = json.loads((path / "FREEZE.json").read_text())
    if freeze["version"] != VERSION or freeze["generatorSha256"] != file_digest(__file__):
        raise ValueError("Fresh evaluation generator/contract changed")
    for name, expected in freeze["files"].items():
        if name.startswith("private/") and not private:continue
        if (path / name).resolve() != (path / name).absolute() or file_digest(path / name) != expected:
            raise ValueError("Fresh evaluation bytes changed after freeze: " + name)
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--semantic-run", type=Path, default=Path("data/ranker/pretrained/catalog-user-semantic-factors-v1-attempt-1"))
    parser.add_argument("--old-catalog", type=Path, default=Path("data/ranker/datasets/v2/catalog.json"))
    args = parser.parse_args()
    result = freeze_dataset(args.out, args.semantic_run, args.old_catalog)
    print(json.dumps({k: result[k] for k in ("status", "newSkuCount", "requests", "eligibleCounts", "totalNativeForwards")}, indent=2))


if __name__ == "__main__":main()
