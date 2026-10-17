"""Generate bounded fictional observable records and separate latent grading truth."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random

VERSION = "synthetic-ranker-v2.0.0"
SPLITS = ("train", "validation", "test")
COUNTS = {"train": 24, "validation": 6, "test": 12}
FOCUS = ("lightweight", "durable", "water-resistant", "laptop-sleeve")
USE = ("compact", "roomy", "organizer")
STYLES = ("detail-view", "in-use-view")
WORDS = ("Cedar", "Alder", "Pine", "River", "Cove", "Summit", "Vale", "Harbor", "Meadow", "Ridge", "Grove", "Field")
FOCUS_TEXT = {"lightweight": "carrying as little extra weight as possible", "durable": "a bag built for repeated everyday use", "water-resistant": "some protection from light rain", "laptop-sleeve": "a dedicated sleeve for my laptop"}
USE_TEXT = {"compact": "something compact for short trips", "roomy": "extra room for a full day away", "organizer": "separate places for small essentials"}
STYLE_TEXT = {"detail-view": "a close-up that shows the bag's actual pockets and fabric", "in-use-view": "a picture showing how the bag looks being carried"}


def opaque(seed: int, key: str, prefix: str) -> str:
    return prefix + hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:18]


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def stable(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def private_labels(products, weights, budget, required, preferred_style, compatibility):
    utilities = {}
    for p in products:
        suitable = p["inStock"] and p["price"] + p["shippingCost"] <= budget and set(required) <= set(p["tags"])
        utilities[p["id"]] = sum(weights.get(t, 0) for t in p["tags"]) - .004 * (p["price"] + p["shippingCost"]) if suitable else None
    ordered = sorted((pid for pid in utilities if utilities[pid] is not None), key=lambda pid: (-utilities[pid], pid))
    relevance = {p["id"]: 0 for p in products}
    for rank, pid in enumerate(ordered):
        relevance[pid] = max(1, 3 - rank)
    best = ordered[0] if ordered else None
    presentation_scores = {}
    for p in products:
        presentation_scores[p["id"]] = {
            v["id"]: (3 if v["presentationStyle"] == preferred_style else 1)
            if compatibility[p["id"]][v["id"]] and relevance[p["id"]] else 0
            for v in p["presentationVariants"]
        }
    action = None
    if best:
        feasible = [v for v in presentation_scores[best] if compatibility[best][v]]
        if feasible:
            variant_id = max(feasible, key=lambda vid: presentation_scores[best][vid])
            action = {"productId": best, "variantId": variant_id}
    return relevance, best, presentation_scores, action


def generate(seed: int = 2026) -> dict:
    rng = random.Random(seed)
    combinations = list(itertools.product(FOCUS, USE, STYLES))
    rng.shuffle(combinations)
    # Assign semantic combinations and families before constructing any example or SKU.
    plan = []
    offset = 0
    for split in SPLITS:
        for local in range(COUNTS[split] // 2):
            combo = combinations[offset]
            family = opaque(seed, f"family:{offset}", "f-")
            plan.append((split, local, offset, family, combo))
            offset += 1
    products = []
    public = {s: [] for s in SPLITS}
    labels = {s: [] for s in SPLITS[:2]}
    truths, family_assignments, product_assignments = [], [], []
    stage_ids = {"initial": [], "adaptation": []}
    position_counts = {s: Counter() for s in SPLITS}
    action_position_counts = {s: Counter() for s in SPLITS}
    for split, local, index, family, combo in plan:
        focus, usage, preferred_style = combo
        family_assignments.append({"personaFamilyId": family, "split": split, "preferenceCombination": list(combo)})
        family_products = []
        other_focus = [t for t in FOCUS if t != focus]
        rng.shuffle(other_focus)
        other_use = [t for t in USE if t != usage]
        rng.shuffle(other_use)
        tag_sets = [[focus, usage], [focus, other_use[0]], [other_focus[0], usage], [other_focus[1], other_use[1]]]
        for slot, tags in enumerate(tag_sets):
            pid = opaque(seed, f"product:{family}:{slot}", "p-")
            capacity = rng.choice([14, 17, 20, 23, 26])
            price = round(rng.randint(28, 92) + rng.randint(0, 99) / 100, 2)
            title = f"{rng.choice(WORDS)} {capacity}L Daypack"
            description = f"{capacity}-litre daypack; {', '.join(tags)}. Polyester shell; zip closure."
            variants = []
            for style in STYLES:
                vid = opaque(seed, f"variant:{pid}:{style}", "v-")
                closeup = style == "detail-view"
                variants.append({
                    "id": vid, "title": f"{title} — {tags[0]}" if closeup else title,
                    "imageKey": f"fictional-descriptor/{vid}", "facts": [f"{capacity}L", *tags],
                    "imageDescription": (f"Close view of the {title}, showing its zip closure and polyester fabric."
                                         if closeup else f"The {title} shown being carried during a daytime walk."),
                    "presentationStyle": style,
                    "compatibleSurfaces": ["compact-card", "detail-panel"] if closeup else ["detail-panel"],
                })
            rng.shuffle(variants)
            product = {"id": pid, "title": title, "category": "backpacks", "description": description,
                       "price": price, "shippingCost": rng.choice([0, 4, 6]), "inStock": not (slot == 3 and index % 5 == 0),
                       "tags": tags, "presentationVariants": variants}
            family_products.append(product)
            products.append(product)
            product_assignments.append({"productId": pid, "split": split, "personaFamilyId": family})
        for repeat in range(2):
            rid = opaque(seed, f"record:{family}:{repeat}", "r-")
            uid = opaque(seed, f"user:{family}:{repeat}", "u-")
            asof = datetime(2026, 9, 3, 12, tzinfo=timezone.utc) + timedelta(days=rng.randint(0, 20))
            # One clearer and one incomplete observation of each latent preference family.
            sparse = repeat == 1
            surface = "compact-card" if (index + repeat) % 4 == 0 else "detail-panel"
            max_total = max(p["price"] + p["shippingCost"] for p in family_products)
            budget = round(max_total + 5 if index % 3 == 0 else family_products[0]["price"] + family_products[0]["shippingCost"] + 3, 2)
            required = [] if index % 4 else [focus]
            stated_budget = budget if not sparse else None
            stated_requirements = required[:] if not sparse else []
            stated_preferences = [focus, usage] if not sparse else ([usage] if index % 2 else [])
            presentation_preferences = [preferred_style] if not sparse and index % 3 != 0 else []
            task = f"I need a backpack for everyday travel. I care about {FOCUS_TEXT[focus]} and {USE_TEXT[usage]}."
            if sparse:
                task = "I'm looking for a backpack for everyday travel. I have not decided which details matter most."
                if stated_preferences:
                    task += f" I'd like {USE_TEXT[usage]}."
            if presentation_preferences:
                task += f" Before choosing, I would like {STYLE_TEXT[preferred_style]}."
            context = {"task": task, "statedBudget": stated_budget,
                       "statedRequirements": stated_requirements, "statedPreferences": stated_preferences,
                       "presentationSurface": surface, "presentationPreferences": presentation_preferences}
            history = []
            history_count = 0 if sparse and index % 3 == 0 else 1 if sparse else 2
            for event_index in range(history_count):
                noisy = sparse or event_index == 1 and index % 3 == 0
                item = rng.choice(family_products) if noisy else family_products[0]
                event = {"at": iso(asof - timedelta(days=2 + event_index * 5)),
                         "type": "view" if noisy else "save", "productId": item["id"],
                         "dwellSeconds": None if noisy else 40 + rng.randint(0, 30),
                         "source": "synthetic-shared-session" if noisy else "synthetic-first-party-event",
                         "confidence": .35 if noisy else .85,
                         "description": ("A short product view; intent is unknown." if noisy
                                         else f"Saved a {', '.join(item['tags'])} backpack after inspecting its specifications.")}
                if not noisy and index % 2 == 0:
                    event["presentationStyle"] = preferred_style
                    event["description"] += f" Opened {STYLE_TEXT[preferred_style]}."
                if sparse and index % 2:
                    event.pop("dwellSeconds")
                history.append(event)
            history.sort(key=lambda e: e["at"])
            compatibility = {p["id"]: {v["id"]: surface in v["compatibleSurfaces"] for v in p["presentationVariants"]}
                             for p in family_products}
            weights = {focus: 4.0, usage: 2.0}
            relevance, best, scores, action = private_labels(family_products, weights, budget, required, preferred_style, compatibility)
            ordered_products = family_products[:]
            rng.shuffle(ordered_products)
            # Stratified random permutations remove a privileged correct answer position.
            target_position = (2 * local + repeat) % 4
            best_index = next(i for i, p in enumerate(ordered_products) if p["id"] == best)
            ordered_products[best_index], ordered_products[target_position] = ordered_products[target_position], ordered_products[best_index]
            missing = ["context.statedBudget"] if stated_budget is None else []
            if not history:
                missing.append("history")
            if not stated_requirements:
                missing.append("context.statedRequirements")
            if not presentation_preferences:
                missing.append("context.presentationPreferences")
            for event_index, event in enumerate(history):
                if event.get("dwellSeconds") is None:
                    missing.append(f"history.{event_index}.dwellSeconds")
            request = {"schemaVersion": 2, "synthetic": True, "recordId": rid, "userId": uid,
                       "asOf": iso(asof), "context": context, "history": history,
                       "candidateProductIds": [p["id"] for p in ordered_products],
                       "presentationCompatibility": compatibility, "missingFields": missing,
                       "provenance": {"source": "fictional-generator", "generatorVersion": VERSION}}
            label = {"recordId": rid, "relevance": relevance, "bestProductId": best,
                     "presentationScores": scores, "bestPresentationAction": action}
            truth = {**label, "synthetic": True, "userId": uid, "personaFamilyId": family,
                     "split": split, "preferenceCombination": list(combo),
                     "hiddenPreferences": {"tagWeights": weights, "pricePenalty": .004},
                     "hiddenConstraints": {"totalBudget": budget, "requiredTags": required,
                                           "avoidedTags": [], "requiresInStock": True},
                     "hiddenPresentationPreferences": {"preferredStyle": preferred_style,
                                                       "rubric": "compatible preferred descriptor style=3; other compatible=1; incompatible/unsuitable=0"},
                     "futureOutcome": {"at": iso(asof + timedelta(hours=1)), "intendedChoice": best},
                     "observationMode": "sparse-noisy" if sparse else "clearer"}
            public[split].append(request)
            truths.append(truth)
            if split in labels:
                labels[split].append(label)
            if split == "train":
                stage_ids["initial" if local < 6 else "adaptation"].append(rid)
            position_counts[split][target_position] += 1
            if action:
                chosen_product = next(p for p in family_products if p["id"] == action["productId"])
                action_position_counts[split][next(i for i, v in enumerate(chosen_product["presentationVariants"]) if v["id"] == action["variantId"])] += 1
    diagnostics, diagnostic_truths, diagnostic_pairs = [], [], []
    product_by_id = {p["id"]: p for p in products}
    # Separate paired interventions: same products/history, different explicit preference.
    # They are excluded from training, selection, and the final held-out metric population.
    for pair_index, base in enumerate(public["test"][::4]):
        candidates = [product_by_id[pid] for pid in base["candidateProductIds"]]
        usages = set(USE)
        left = next(p for p in candidates if p["inStock"] and sum(t in usages for t in p["tags"]) == 1)
        right = next(p for p in candidates if p["tags"][0] != left["tags"][0] and p["inStock"])
        pair_ids = []
        for side, preferred_product in enumerate((left, right)):
            row = copy.deepcopy(base)
            rid = opaque(seed, f"diagnostic-record:{pair_index}:{side}", "r-")
            row.update(recordId=rid, userId=opaque(seed, f"diagnostic-user:{pair_index}:{side}", "u-"), history=[])
            preference = preferred_product["tags"][0]
            usage_preference = preferred_product["tags"][1]
            row["context"] = {"task": f"I want a backpack, especially {FOCUS_TEXT[preference]} and {USE_TEXT[usage_preference]}.",
                              "statedBudget": 150, "statedRequirements": [],
                              "statedPreferences": [preference, usage_preference],
                              "presentationSurface": "detail-panel", "presentationPreferences": ["detail-view"]}
            row["presentationCompatibility"] = {p["id"]: {v["id"]: True for v in p["presentationVariants"]} for p in candidates}
            row["missingFields"] = ["history", "context.statedRequirements"]
            relevance, best, scores, action = private_labels(candidates, {preference: 4, usage_preference: 2}, 150, [], "detail-view", row["presentationCompatibility"])
            diagnostic_truths.append({"recordId": rid, "userId": row["userId"], "synthetic": True,
                                      "diagnosticPairId": f"pair-{pair_index}", "relevance": relevance,
                                      "bestProductId": best, "presentationScores": scores,
                                      "bestPresentationAction": action,
                                      "hiddenPreferences": {"tagWeights": {preference: 4, usage_preference: 2}, "pricePenalty": .004},
                                      "hiddenConstraints": {"totalBudget": 150, "requiredTags": [], "avoidedTags": [], "requiresInStock": True}})
            diagnostics.append(row)
            pair_ids.append(rid)
        diagnostic_pairs.append({"diagnosticPairId": f"pair-{pair_index}", "recordIds": pair_ids,
                                 "intervention": "Only explicit product preferences/task text change; candidate facts, history, cutoff, and presentation surface stay fixed."})
    # Prevent row order carrying family/category/target-position sequence into training.
    for records in public.values():
        rng.shuffle(records)
    for records in labels.values():
        rng.shuffle(records)
    rng.shuffle(products)
    rng.shuffle(truths)
    return {"catalog": {"schemaVersion": 2, "synthetic": True, "products": products},
            "public": public, "labels": labels, "truths": truths,
            "familyAssignments": family_assignments, "productAssignments": product_assignments,
            "trainStageRecordIds": stage_ids, "diagnostics": diagnostics,
            "diagnosticTruths": diagnostic_truths, "diagnosticPairs": diagnostic_pairs,
            "positionCounts": {s: {str(i): position_counts[s][i] for i in range(4)} for s in SPLITS},
            "actionPositionCounts": {s: {str(i): action_position_counts[s][i] for i in range(2)} for s in SPLITS}}


def write_dataset(output: Path, seed: int = 2026) -> dict:
    data = generate(seed)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    def write(relative, value, jsonl=False):
        content = "".join(stable(row) + "\n" for row in value) if jsonl else json.dumps(value, sort_keys=True, indent=2) + "\n"
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        hashes[relative] = hashlib.sha256(content.encode()).hexdigest()
    write("catalog.json", data["catalog"])
    for split in SPLITS:
        write(f"public/{split}.jsonl", data["public"][split], True)
    for split in SPLITS[:2]:
        write(f"labels/{split}.jsonl", data["labels"][split], True)
    write("private/truth.jsonl", data["truths"], True)
    write("public/diagnostics.jsonl", data["diagnostics"], True)
    write("private/diagnostic-truth.jsonl", data["diagnosticTruths"], True)
    manifest = {"schemaVersion": 2, "synthetic": True, "generatorVersion": VERSION, "seed": seed,
                "counts": {"records": COUNTS, "families": {s: COUNTS[s] // 2 for s in SPLITS},
                           "users": 42, "products": 84, "maxCandidates": 4, "variantsPerProduct": 2,
                           "diagnosticRecords": 6, "diagnosticPairs": 3},
                "familyAssignments": data["familyAssignments"], "productAssignments": data["productAssignments"],
                "userAssignments": [{"userId": t["userId"], "personaFamilyId": t["personaFamilyId"], "split": t["split"]} for t in data["truths"]],
                "trainStageRecordIds": data["trainStageRecordIds"],
                "diagnosticPairs": data["diagnosticPairs"],
                "positionDiagnostics": {"bestProductPositionCounts": data["positionCounts"],
                                        "bestVariantPositionCounts": data["actionPositionCounts"]},
                "sha256": dict(hashes),
                "publicFileHashes": {k: v for k, v in hashes.items() if k.startswith("public/")},
                "splitPolicy": "Family and exact (focus, use, presentation style) combination assigned before generating users/products; all SKU IDs and complete product fact sets disjoint by split.",
                "orderPolicy": "Opaque IDs, shuffled product/variant/catalog/record order; best-product positions stratified within each split; variant order seeded random and diagnosed.",
                "cutoffPolicy": "Only history <= asOf is public; hypothetical future actions remain private.",
                "provenance": "Fictional synthetic text fixture; no real people, sensitive traits, images, clicks, conversion or training outcomes.",
                "limitations": ["42 cases are a workflow proof, not a representative benchmark.",
                                "Held-out SKU identities/fact sets and exact preference combinations do not imply unseen attributes.",
                                "Presentation supervision grades text descriptors against an invented latent preference rubric, never actual image pixels.",
                                "Catalog recommendations are synthetic; this cannot establish real human conversion or revenue lift."]}
    write("manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/ranker/datasets/v2"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    manifest = write_dataset(args.output, args.seed)
    print(json.dumps({"output": str(args.output), "counts": manifest["counts"], "positionDiagnostics": manifest["positionDiagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
