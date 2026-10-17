"""Generate fictional observable requests and separate private evaluation truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

VERSION = "synthetic-ranker-v1.0.0"
SPLITS = ("train", "validation", "test")
PRODUCT_ROWS = {
    "backpacks": [
        ("Commuter 18L", 48, 4, ["lightweight", "laptop-sleeve", "compact"]),
        ("Trail 24L", 69, 0, ["water-resistant", "roomy", "durable"]),
        ("Foldaway 16L", 25, 6, ["lightweight", "compact", "packable"]),
        ("Transit 28L", 89, 0, ["laptop-sleeve", "roomy", "water-resistant"]),
        ("Canvas 20L", 39, 5, ["durable", "simple", "roomy"]),
        ("Weather 22L", 78, 4, ["water-resistant", "lightweight", "laptop-sleeve"]),
    ],
    "headphones": [
        ("Studio Wired", 45, 4, ["over-ear", "wired", "detailed-sound"]),
        ("Quiet Wireless", 99, 0, ["over-ear", "noise-cancelling", "wireless"]),
        ("Pocket Buds", 35, 5, ["in-ear", "wireless", "compact"]),
        ("Travel Fold", 65, 0, ["over-ear", "wireless", "compact"]),
        ("Focus USB", 55, 3, ["over-ear", "usb-c", "microphone"]),
        ("Active Buds", 75, 0, ["in-ear", "water-resistant", "wireless"]),
    ],
    "desk-lamps": [
        ("Focus Adjustable", 32, 5, ["adjustable", "warm-light", "compact"]),
        ("Read Wide", 58, 0, ["wide-light", "dimmable", "warm-light"]),
        ("Clip USB", 22, 4, ["usb-c", "compact", "clip-mount"]),
        ("Work Bright", 49, 0, ["wide-light", "adjustable", "dimmable"]),
        ("Portable Glow", 42, 5, ["rechargeable", "warm-light", "compact"]),
        ("Studio Arm", 79, 0, ["adjustable", "usb-c", "wide-light"]),
    ],
}
TASKS = {
    "backpacks": "Choose a backpack for everyday travel.",
    "headphones": "Choose headphones for everyday listening.",
    "desk-lamps": "Choose a lamp for a work desk.",
}


def catalog() -> dict:
    products = []
    for category, rows in PRODUCT_ROWS.items():
        for i, (title, price, shipping, tags) in enumerate(rows):
            product_id = f"{category}-{i + 1}"
            products.append({
                "id": product_id, "title": title, "category": category,
                "description": "; ".join(tags) + ".",
                "price": price, "shippingCost": shipping, "inStock": i != 5,
                "tags": tags,
                "presentationVariants": [
                    {"id": f"{product_id}-default", "title": title,
                     "imageKey": f"fixture/{product_id}/front",
                     "facts": tags},
                    {"id": f"{product_id}-detail", "title": f"{title} — {tags[0]}",
                     "imageKey": f"fixture/{product_id}/detail",
                     "facts": tags},
                ],
            })
    return {"schemaVersion": 1, "synthetic": True, "products": products}


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def truth_for(products: list[dict], preferred: dict[str, float], budget: float,
              required: list[str], avoided: list[str]) -> tuple[dict, str | None]:
    utilities = {}
    for p in products:
        feasible = (p["inStock"] and p["price"] + p["shippingCost"] <= budget
                    and set(required).issubset(p["tags"])
                    and not set(avoided).intersection(p["tags"]))
        utilities[p["id"]] = (sum(preferred.get(tag, 0) for tag in p["tags"])
                              - 0.006 * (p["price"] + p["shippingCost"])) if feasible else None
    valid = sorted((pid for pid, value in utilities.items() if value is not None),
                   key=lambda pid: (-utilities[pid], pid))
    best = valid[0] if valid else None
    relevance = {p["id"]: 0 for p in products}
    for rank, pid in enumerate(valid):
        relevance[pid] = 3 if rank == 0 else 2 if rank <= 2 else 1
    return relevance, best


def generate(seed: int = 2026) -> dict:
    rng = random.Random(seed)
    product_catalog = catalog()
    families = [f"family-{i:03d}" for i in range(60)]
    rng.shuffle(families)
    assignment = {family: "train" if i < 40 else "validation" if i < 50 else "test"
                  for i, family in enumerate(families)}
    public = {split: [] for split in SPLITS}
    labels = {split: [] for split in SPLITS[:2]}
    truths = []
    family_metadata = []
    for family_index in range(60):
        family_id = f"family-{family_index:03d}"
        split = assignment[family_id]
        category = list(PRODUCT_ROWS)[family_index % 3]
        products = [p for p in product_catalog["products"] if p["category"] == category]
        candidates = [p["id"] for p in products]
        # IDs are opaque linkage, not encoded category/split/family features.
        def opaque(prefix: str, suffix: str) -> str:
            return prefix + hashlib.sha256(f"{seed}:{family_id}:{suffix}".encode()).hexdigest()[:16]
        family_metadata.append({"personaFamilyId": family_id, "split": split})
        common_asof = datetime(2026, 9, 1, 12, tzinfo=timezone.utc) + timedelta(days=family_index % 20)
        preferred_product = products[family_index % 5]
        shared_tag = preferred_product["tags"][0]
        for record_index in range(6):
            record_id = opaque("r-", str(record_index))
            user_id = opaque("u-", str(record_index))
            asof = common_asof + timedelta(days=max(0, record_index - 1))
            mode = ["uncertain-twin-a", "uncertain-twin-b", "sparse", "noisy", "incomplete", "established"][record_index]
            # First two cases intentionally have identical model-visible input but divergent truth.
            target = products[0] if record_index == 0 else products[1] if record_index == 1 else preferred_product
            preferred = {tag: (3.0 if i == 0 else 1.5) for i, tag in enumerate(target["tags"])}
            preferred[target["tags"][0]] += 1.0
            budget = 120 if record_index < 2 else rng.choice([40, 60, 80, 110])
            if family_index % 13 == 0 and record_index == 4:
                budget = 15  # None suitable: tests abstention instead of forced spend.
            hidden_required = [] if record_index < 2 else ([shared_tag] if family_index % 3 == 0 else [])
            stated_required = hidden_required[:] if record_index in (3, 5) else []
            stated_budget = budget if record_index in (2, 3, 5) else None
            stated_prefs = [shared_tag] if record_index in (2, 5) else []
            context = {"task": TASKS[category], "statedBudget": stated_budget,
                       "statedRequirements": stated_required, "statedPreferences": stated_prefs}
            history = []
            if record_index >= 2:
                count = {2: 1, 3: 3, 4: 2, 5: 3}[record_index]
                for event_index in range(count):
                    noisy = record_index == 3 and event_index < 2
                    item = rng.choice(products[:5]) if noisy else target
                    event = {"at": iso(asof - timedelta(days=1 + event_index * 7, hours=2)),
                             "type": "view" if noisy else ["view", "save", "purchase"][event_index % 3],
                             "productId": item["id"],
                             "dwellSeconds": None if event_index == 0 else rng.choice([8, 35, 80]),
                             "source": "synthetic-shared-session" if noisy else "synthetic-first-party-event",
                             "confidence": 0.35 if noisy else 0.85}
                    history.append(event)
                history.sort(key=lambda event: event["at"])
            missing = []
            if stated_budget is None:
                missing.append("context.statedBudget")
            if not history:
                missing.append("history")
            if not stated_prefs:
                missing.append("context.statedPreferences")
            if not stated_required:
                missing.append("context.statedRequirements")
            if record_index == 4:
                # A missing property is distinct from an explicit null value.
                history[-1].pop("dwellSeconds")
                missing.append(f"history.{len(history)-1}.dwellSeconds")
            request = {"schemaVersion": 1, "synthetic": True, "recordId": record_id,
                       "userId": user_id, "asOf": iso(asof), "context": context,
                       "history": history, "candidateProductIds": candidates[:],
                       "missingFields": missing,
                       "provenance": {"source": "fictional-generator", "generatorVersion": VERSION,
                                      "observedThrough": iso(asof)}}
            relevance, best = truth_for(products, preferred, budget, hidden_required, [])
            truth = {"recordId": record_id, "userId": user_id, "personaFamilyId": family_id,
                     "split": split, "synthetic": True,
                     "hiddenPreferences": {"tagWeights": preferred, "pricePenalty": 0.006},
                     "hiddenConstraints": {"totalBudget": budget, "requiredTags": hidden_required,
                                           "avoidedTags": [], "requiresInStock": True},
                     "relevance": relevance, "bestProductId": best, "observationMode": mode,
                     "counterfactualTwinGroup": family_id if record_index < 2 else None,
                     "labelProvenance": "synthetic-hidden-utility-v1; no observed human outcome",
                     "futureOutcome": {"at": iso(asof + timedelta(hours=1)), "intendedChoice": best}}
            public[split].append(request)
            truths.append(truth)
            if split in labels:
                labels[split].append({"recordId": record_id, "relevance": relevance, "bestProductId": best})
    return {"catalog": product_catalog, "public": public, "labels": labels,
            "truths": truths, "families": family_metadata, "seed": seed}


def write_dataset(output: Path, seed: int = 2026) -> dict:
    data = generate(seed)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    def write(relative: str, value: object, jsonl: bool = False) -> None:
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = ("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in value)
                   if jsonl else json.dumps(value, indent=2, sort_keys=True) + "\n")
        path.write_text(content)
        files[relative] = hashlib.sha256(content.encode()).hexdigest()
    write("catalog.json", data["catalog"])
    for split in SPLITS:
        write(f"public/{split}.jsonl", data["public"][split], True)
    for split in SPLITS[:2]:
        write(f"labels/{split}.jsonl", data["labels"][split], True)
    write("private/truth.jsonl", data["truths"], True)
    train_families = [f["personaFamilyId"] for f in data["families"] if f["split"] == "train"]
    initial_families = set(train_families[:20])
    train_stage_ids = {"initial": [], "adaptation": []}
    for truth in data["truths"]:
        if truth["split"] == "train":
            stage = "initial" if truth["personaFamilyId"] in initial_families else "adaptation"
            train_stage_ids[stage].append(truth["recordId"])
    manifest = {"schemaVersion": 1, "synthetic": True, "seed": seed,
                "trainStageRecordIds": train_stage_ids,
                "trainStagePolicy": "First 20 train families by family ID initial; remaining 20 adaptation. Audit metadata only.",
                "generatorVersion": VERSION, "familyAssignments": data["families"],
                "counts": {"families": 60, "products": 18,
                           "records": {s: len(data["public"][s]) for s in SPLITS},
                           "newUsers": 120, "uncertainTwinPairs": 60},
                "sha256": dict(files), "publicFileHashes": {k: v for k, v in files.items() if k.startswith("public/")},
                "cutoffPolicy": "History at <= request.asOf; future actions exist in private truth only.",
                "splitPolicy": "Shuffle families with seed, assign 40/10/10 before records. No user crosses a family.",
                "provenance": "Entirely fictional synthetic fixtures; no people, PII, protected traits or real outcomes.",
                "limitations": ["Utility labels reflect generator assumptions, not human preferences.",
                                "Observationally identical twins have different hidden optimums.",
                                "Catalog, language, event noise and missingness are deliberately small.",
                                "No synthetic metric establishes real conversion or revenue lift."]}
    write("manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/ranker/datasets/v1"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps({"output": str(args.output), "counts": write_dataset(args.output, args.seed)["counts"]}, indent=2))


if __name__ == "__main__":
    main()
