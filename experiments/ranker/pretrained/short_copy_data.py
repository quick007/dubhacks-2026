"""CPU-only short-root STB+C supervision; no model load, fit or native decode.

12 new public TRAIN requests on old TRAIN-only catalogs, 20 fields each.
Old inspected dev/test cases remain evaluation-only; no old file is rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import itertools
import json
from pathlib import Path

from .above_fold import CAPS, REQUIRED
from .above_fold_copy.layout import assemble_payload
from .boundary import clean_product, visible_history
from .branch_local_copy import RULES, _card_facts
from .catalog_fusion.generalization_v2 import query_text_v2
from .catalog_fusion.inputs import canonical, digest
from .catalog_fusion.preflight import file_digest
from .catalog_fusion.runtime import write_json
from .fusion import observable_request
from .semantic_copy_pairs.generator import TAGS, plan_for

ROOT = Path(__file__).resolve().parent
VERSION = "short-raw-root-stbc-v1"
DATA = ROOT / "short_copy_v1"
GLOBAL_RULES = RULES.replace("Public evidence: ",
    "Copy: one NEW factual sentence,10-180 characters, independently mention a supportedObservedTag when present; "
    "otherwise use only selected facts and ordinary travel framing. Do not copy descriptions/category/image text. "
    "Every field stands alone; no field relies on another generated value. Public request: ")
LIMITS = {"rootTokens": 512, "commonAppendTokens": 512, "contextTokens": 4096,
          "teacherSuffixTokens": 256, "firstThreeFields": 10, "fullSixFields": 20,
          "firstThreeOutputCap": 208, "fullSixOutputCap": 416, "trainingFields": 240}


def records(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def expected_fields(cards=6):
    if cards not in (3, 6):raise ValueError("Only complete first3 or full6 card views")
    result = []
    for section in range(cards // 3):
        result.append("S" + str(section + 1))
        for index in range(section * 3 + 1, section * 3 + 4):result.extend(letter + str(index) for letter in ("T", "B", "C"))
    return result


def public_context(request, catalog):
    public = observable_request(request)
    history = [{**{key: e.get(key) for key in ("at", "type", "confidence", "description", "presentationStyle")},
                "productFacts": clean_product(catalog[e["productId"]])} for e in visible_history(public, catalog)]
    return {"asOf": public["asOf"], "request": public["context"], "observedHistory": history}


def layout(plan, request, catalog):
    from .page_plan import validate_page_plan
    if plan["requestedCardCount"] != 6 or plan["shortfall"] != 0 or len(plan["sections"]) != 2 or any(len(s["products"]) != 3 for s in plan["sections"]):raise ValueError("Six complete ordered cards/two sections required")
    validate_page_plan(plan, request, catalog)
    cards = [_card_facts(i, c, catalog) for i, c in enumerate((c for s in plan["sections"] for c in s["products"]), 1)]
    root = query_text_v2(request, catalog, "focused")
    common = {"role": "user", "content": GLOBAL_RULES + canonical(public_context(request, catalog))}
    sections = []
    for i, section in enumerate(plan["sections"], 1):
        universal = sorted(set.intersection(*(set(catalog[c["productId"]]["tags"]) for c in section["products"])))
        sections.append({"universalTags": universal, "members": [{"identity": cards[j]["identity"], "tags": cards[j]["productFacts"]["tags"]} for j in range((i - 1) * 3, i * 3)]})
    rows = []
    for field in expected_fields():
        prefix, index = field[0], int(field[1:]);kind = {"S": "heading", "T": "title", "B": "badge", "C": "copy"}[prefix]
        facts = {"section": sections[index - 1]} if prefix == "S" else {"card": cards[index - 1]}
        content = f"Write only one fresh {kind} value using these bound public facts and shared {kind} rules; no label; stop. " + canonical(facts)
        rows.append({"rowId": f"{kind}-{index}", "fields": [field], "field": field, "kind": kind,
                     "maxTokens": CAPS[prefix], "messages": [{"role": "user", "content": content}]})
    if len(rows) != 20 or sum(r["maxTokens"] for r in rows[:10]) != 208 or sum(r["maxTokens"] for r in rows) != 416:raise AssertionError("Complete visible/full cap changed")
    return root, common, rows, {"cards": cards, "sections": sections, "publicContext": public_context(request, catalog)}


def target_values(plan, catalog, evidence):
    targets = {}
    for i, section in enumerate(evidence["sections"], 1):
        tags = section["universalTags"]
        targets[f"S{i}"] = tags[0].capitalize() + " travel choices" if tags else "Everyday travel choices"
    for card in evidence["cards"]:
        i = card["index"];support = card["supportedObservedTags"];tag = support[0] if support else card["productFacts"]["tags"][0]
        targets[f"T{i}"] = card["identity"] + " for " + (support[0] if support else "everyday") + " travel"
        targets[f"B{i}"] = support[0] + " choice" if support else "NONE"
        targets[f"C{i}"] = "A " + tag + " daypack for everyday travel."
    return targets


def build_layout(plan, request, catalog):
    """Serving adapter: raw focused root is prepared separately by fused_prefix."""
    _, common, rows, evidence = layout(plan, request, catalog)
    return common, rows, evidence


def validate_rows(rows, plan, request, catalog, *, require_native_receipts=False):
    """Reuse unchanged strict3-card validator twice; never repair raw values."""
    expected = expected_fields();mapping = {}
    for row in rows:
        labels = row.get("fields", [])
        if len(labels) != 1 or labels[0] not in expected or labels[0] in mapping:raise ValueError("One unique expected scheduler field per row")
        mapping[labels[0]] = row
    first_fields = expected_fields(3)
    first = assemble_payload([copy.deepcopy(mapping[f]) for f in first_fields], plan, request, catalog,
                             require_native_receipts=require_native_receipts) if set(first_fields) <= set(mapping) else None
    full = set(mapping) == set(expected)
    second = None
    if full:
        second_plan = copy.deepcopy(plan);second_plan["sections"] = [second_plan["sections"][1], second_plan["sections"][0]]
        second_plan["sectionOrder"] = [s["sectionId"] for s in second_plan["sections"]]
        local = []
        for f in first_fields:
            original = "S2" if f[0] == "S" else f[0] + str(int(f[1:]) + 3)
            row = copy.deepcopy(mapping[original]);row["fields"] = [f]
            # Validation metadata maps a section view; text/tokens remain exact.
            row["rowId"] = row["kind"] + "-" + f[1:]
            local.append(row)
        second = assemble_payload(local, second_plan, request, catalog, require_native_receipts=require_native_receipts)
    full_valid = bool(full and first and first["validation"]["valid"] and second and second["validation"]["valid"])
    return {"firstThreeValid": bool(first and first["validation"]["valid"]), "fullSixValid": full_valid,
            "firstThree": first, "secondThree": second, "all20RowsPresent": full,
            "fullSixPayload": {"schemaVersion": VERSION + "/six-card-payload", "sections": [first["payload"], second["payload"]]} if full_valid else None,
            "rawRowsUnmodified": copy.deepcopy(rows), "nativeReceiptsRequired": require_native_receipts,
            "scope": "copy-stage locked decisions; no ranking/image/browser result"}


def choose_training_preferences(sources):
    counts = Counter();result = []
    for source in sources:
        choices = {tuple(sorted(pair)) for p in source["catalog"].values() for pair in itertools.combinations(p["tags"], 2)}
        pair = min(choices, key=lambda t: (-sum(counts[a] == 0 for a in t), sum(counts[a] for a in t), digest(canonical([source["caseId"], t]))))
        result.append(pair);counts.update(pair)
    if set(counts) != set(TAGS):raise ValueError("Twelve TRAIN-only requests do not cover all18 support atoms")
    return result


def build():
    sources = records(ROOT / "semantic_copy_data/cases.train.jsonl")
    if len(sources) != 12:raise ValueError("Expected12 original TRAIN-only six-product catalogs")
    preferences = choose_training_preferences(sources);groups = {"train": [], "dev": [], "test": []};field_groups = {s: [] for s in groups};coverage = {}
    for split in groups:
        donors = sources if split == "train" else records(ROOT / f"semantic_copy_pairs/cases.{split}.jsonl")
        for i, source in enumerate(donors):
            catalog = copy.deepcopy(source["catalog"]);request = observable_request(copy.deepcopy(source["request"]))
            request["recordId"] = "short-copy-" + digest(canonical([VERSION, split, source["caseId"]]))[:18]
            if split == "train":
                request["context"]["statedPreferences"] = list(preferences[i])
                request["context"]["task"] = "Choose daypacks for everyday travel with " + " and ".join(preferences[i]) + " details."
            plan = plan_for(request, catalog, source["pendingPagePlan"])
            raw, common, rows, evidence = layout(plan, request, catalog);targets = target_values(plan, catalog, evidence)
            teacher = [{**row, "text": targets[row["field"]], "finishReason": "stop"} for row in rows]
            audited = validate_rows(teacher, plan, request, catalog)
            if not audited["firstThreeValid"] or not audited["fullSixValid"]:raise ValueError("Teacher fails unchanged strict STBC validator: " + source["caseId"])
            case = {"datasetVersion": VERSION, "caseId": request["recordId"], "split": split,
                    "sourceCaseId": source["caseId"], "sourceSplit": "train" if split == "train" else split,
                    "synthetic": True, "request": request, "catalog": catalog, "pendingPagePlan": plan,
                    "rawPrefixText": raw, "commonCopyMessage": common, "publicEvidence": evidence,
                    "decisionScope": "locked fixture selections; no ranking/image quality supervision",
                    "evaluationCasesPreviouslyInspected": split != "train"}
            groups[split].append(case)
            for row in rows:
                field = {**row, "datasetVersion": VERSION, "caseId": case["caseId"], "split": split,
                         "targetValue": targets[row["field"]], "teacherOnlySyntheticStop": True}
                content = canonical([raw, common, row["messages"]])
                if case["caseId"] in content or source.get("personaFamilyId", "__no_persona__") in content or "targetValue" in content:raise ValueError("Private/opaque/teacher metadata leaked into prompt")
                if field["targetValue"] != "NONE" and field["targetValue"].lower() in content.lower():raise ValueError("Exact answer leaked into prompt")
                field_groups[split].append(field)
        atoms = Counter(a for c in groups[split] for card in c["publicEvidence"]["cards"] for a in card["supportedObservedTags"])
        coverage[split] = {"cases": len(groups[split]), "fields": len(field_groups[split]),
            "firstThreeFieldsPerCase": 10, "fullSixFieldsPerCase": 20, "bodyTargets": len(groups[split]) * 6,
            "positiveObservedBodyCountsByAtom": dict(sorted(atoms.items())),
            "requiredNoneBadges": sum(f["kind"] == "badge" and f["targetValue"] == "NONE" for f in field_groups[split]),
            "universalHeadings": sum(bool(s["universalTags"]) for c in groups[split] for s in c["publicEvidence"]["sections"]),
            "neutralHeadings": sum(not s["universalTags"] for c in groups[split] for s in c["publicEvidence"]["sections"]),
            "allTeacherFirstThreeAndFullSixPassUnchangedValidators": True}
    if len(field_groups["train"]) != 240 or set(coverage["train"]["positiveObservedBodyCountsByAtom"]) != set(TAGS):raise ValueError("Fixed240 TRAIN fields/all18support atom coverage required")
    return groups, field_groups, coverage


def freeze(out):
    groups, fields, coverage = build();out = Path(out);out.mkdir(parents=True, exist_ok=False)
    for split in groups:
        for name, values in (("cases", groups[split]), ("fields", fields[split])):
            with (out / f"{name}.{split}.jsonl").open("x") as h:
                for value in values:h.write(canonical(value) + "\n")
    deps = ("above_fold.py", "above_fold_copy/layout.py", "branch_local_copy.py", "scheduler_identity_experiment.py",
            "fusion.py", "page_plan.py", "evidence_plan.py", "semantic_copy_tokens.py", "semantic_copy_gradient_features.py",
            "catalog_fusion/generalization_v2.py")
    sources = ("semantic_copy_data/cases.train.jsonl", "semantic_copy_pairs/cases.dev.jsonl", "semantic_copy_pairs/cases.test.jsonl")
    manifest = {"datasetVersion": VERSION, "frozenBeforeNative": True, "sourceSha256": file_digest(__file__),
                "dependencySha256": {n: file_digest(ROOT / n) for n in deps}, "parentSha256": {n: file_digest(ROOT / n) for n in sources},
                "fileSha256": {p.name: file_digest(p) for p in sorted(out.glob("*.jsonl"))},
                "coverage": coverage, "limits": LIMITS, "nativeExecution": False, "trainingExecuted": False,
                "prefixFormat": "exact untemplated generalization_v2.query_text_v2(...,focused); one shared common native user turn then one local native field user turn",
                "fieldOrder": expected_fields(), "firstThreeFieldOrder": expected_fields(3),
                "limitations": ["12 new public training requests on old TRAIN-only catalogs; old dev/test inspected and evaluator-only.",
                                "Synthetic teacher strings are supervision, never runtime fallback/template output.",
                                "All18 copy facts covered; seven-atom ranking head is a separate model/domain and is not validated by these locked copy cases.",
                                "Validation reuses unchanged strict3-card STBC contract on each true section with metadata-only local index projection.",
                                "Full6 copy completion is distinct from first3 payload; no browser/render/conversion/learned grouping claim.",
                                "Every native field must still pass exact nonzeroA/B gradient parity/source-cache preservation before240updates."]}
    write_json(out / "manifest.json", manifest);return manifest


def load_dataset(data):
    data = Path(data);manifest = json.loads((data / "manifest.json").read_text())
    if manifest["datasetVersion"] != VERSION or manifest["sourceSha256"] != file_digest(__file__):raise ValueError("Frozen short-copy source changed")
    for name, sha in manifest["dependencySha256"].items():
        if file_digest(ROOT / name) != sha:raise ValueError("Strict validator/layout dependency changed: " + name)
    for name, sha in manifest["parentSha256"].items():
        if file_digest(ROOT / name) != sha:raise ValueError("Retained parent changed")
    for name, sha in manifest["fileSha256"].items():
        if file_digest(data / name) != sha:raise ValueError("Frozen short-copy payload changed")
    cases = {s: records(data / f"cases.{s}.jsonl") for s in ("train", "dev", "test")}
    fields = {s: records(data / f"fields.{s}.jsonl") for s in cases}
    return manifest, cases, fields


def preflight(tokenizer, data):
    """The supplied tokenizer must be the raw-prefix proxy, no native calls."""
    from .semantic_copy_gradient_features import build_native_contract
    manifest, cases, fields = load_dataset(data);inventory = {};total = 0
    for split in cases:
        inventory[split] = []
        for case in cases[split]:
            root_message = {"role": "raw_prefix", "content": case["rawPrefixText"]}
            root_tokens = tokenizer.encode(case["rawPrefixText"], add_special_tokens=False)
            state_messages = [root_message, case["commonCopyMessage"]]
            common = tokenizer.apply_chat_template(state_messages, tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=False)
            if common[:len(root_tokens)] != root_tokens or not 1 <= len(root_tokens) <= 512 or not 1 <= len(common) - len(root_tokens) <= 512:raise ValueError("Short raw/common prefix boundary/cap failed")
            state = {"messages": state_messages, "tokens": common, "consumed": False}
            rows = [f for f in fields[split] if f["caseId"] == case["caseId"]]
            if [f["field"] for f in rows] != expected_fields():raise ValueError("Full20 scheduler roster/order changed")
            for field in rows:
                contract = build_native_contract(tokenizer, state, field)
                inventory[split].append({"caseId": case["caseId"], "rowId": field["rowId"], "field": field["field"],
                    "maxTokens": field["maxTokens"], "rawRootTokens": len(root_tokens), "commonAppendTokens": len(common) - len(root_tokens),
                    "sharedTokens": len(common), "suffixTokens": len(contract["suffixTokenIds"]), "targetTokens": len(contract["targetTokenIds"]),
                    "fullTeacherTokens": contract["fullTokens"], "promptSha256": contract["promptSha256"],
                    "modelInputSha256": contract["modelInputSha256"], "sourceTokenSha256": contract["sourceTokenSha256"],
                    "projectedHostFeatureBytes": len(contract["suffixTokenIds"]) * 20480 * 4})
            if split == "train":total += len(rows)
    if total != 240:raise ValueError("Exactly240 TRAIN fields required")
    return {"status": "CPU token contract passed; no native/fit", "datasetManifestSha256": file_digest(Path(data) / "manifest.json"),
            "limits": LIMITS, "inventory": inventory, "trainFields": 240,
            "trainProjectedFloat32HostFeatureBytes": sum(r["projectedHostFeatureBytes"] for r in inventory["train"]),
            "suffixRange": [min(r["suffixTokens"] for rows in inventory.values() for r in rows), max(r["suffixTokens"] for rows in inventory.values() for r in rows)],
            "sourceSha256": file_digest(__file__), "dependencySha256": manifest["dependencySha256"]}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("action", choices=("freeze", "check"));p.add_argument("--data", type=Path, default=DATA);p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.action == "freeze":result = freeze(args.out or args.data)
    else:result = load_dataset(args.data)[0]
    print(json.dumps({"datasetVersion": result["datasetVersion"], "coverage": result["coverage"], "limits": result["limits"]}, indent=2))


if __name__ == "__main__":main()
