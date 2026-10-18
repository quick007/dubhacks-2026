"""CPU replay of frozen teachers and saved native output under current identity policy."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from ..fusion import TITLE_IDENTITY_POLICY, identity_errors
from ..parallel_decode import assemble
from ..scheduler_identity_experiment import assemble_values
from .generator import HERE, SPLITS, audit, build


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = HERE.parent
    paths = [source / name for name in ("fusion.py", "evidence_plan.py",
             "scheduler_identity_experiment.py", "parallel_decode.py",
             "SCHEDULER_IDENTITY_CASES.json", "SCHEDULER_IDENTITY_RESULTS.json")]
    hashes = {p.name: sha(p) for p in paths}
    cases, fields = build()
    teacher = audit(cases, fields)
    teacher_identity = {}
    for split, _ in SPLITS:
        count = 0
        for case in cases[split]:
            cards = [c for s in case["pendingPagePlan"]["sections"] for c in s["products"]]
            for target in case["targets"]:
                if target["kind"] != "title":
                    continue
                product = case["catalog"][cards[target["cardIndex"] - 1]["productId"]]
                errors = identity_errors(target["targetValue"], product, case["catalog"])
                assert not errors, (case["caseId"], target["field"], errors)
                count += 1
        teacher_identity[split] = count
    packet = json.loads((source / "SCHEDULER_IDENTITY_CASES.json").read_text())
    native = json.loads((source / "SCHEDULER_IDENTITY_RESULTS.json").read_text())
    by_id = {c["caseId"]: c for c in packet["diagnosticCases"] + packet["freshSyntheticHoldouts"]}
    records = []
    for record in native["records"]:
        case = by_id[record["caseId"]]
        for name, arm in record["arms"].items():
            if arm["status"] != "completed-attempt":
                records.append({"caseId": record["caseId"], "arm": name, "status": arm["status"]})
                continue
            fn = assemble_values if name == "schedulerIdentity" else assemble
            replay = fn(copy.deepcopy(arm["rows"]), arm["pagePlan"], case["request"], case["catalog"])
            cards = [c for s in arm["pagePlan"]["sections"] for c in s["products"]]
            after = [c for s in replay["pagePlan"]["sections"] for c in s["products"]]
            titles = []
            for row in arm["rows"]:
                if row["kind"] != "title":
                    continue
                index = int(row["fields"][0][1:]) - 1
                raw = row["text"]
                body = raw if name == "schedulerIdentity" else raw.removeprefix(row["fields"][0] + ": ")
                product = case["catalog"][cards[index]["productId"]]
                errors = identity_errors(body, product, case["catalog"])
                titles.append({"field": row["fields"][0], "rawSavedText": raw,
                    "selectedIdentity": product["title"], "identityErrors": errors,
                    "previouslyAccepted": cards[index]["title"]["source"] == "generated",
                    "nowAccepted": after[index]["title"]["source"] == "generated"})
            records.append({"caseId": record["caseId"], "freshHoldout": record["freshHoldout"],
                "arm": name, "status": "saved-native-output-cpu-replay",
                "oldQuality": arm["quality"], "newQuality": replay["quality"], "titles": titles})
    assert hashes == {p.name: sha(p) for p in paths}, "Source changed during CPU audit; rerun"
    output = {"schemaVersion": 1, "identityPolicy": TITLE_IDENTITY_POLICY,
        "nativeExecution": False, "trainingExecuted": False,
        "teacherTargetsAreNotModelOutputs": True, "sourceSha256": hashes,
        "teacherAudit": teacher, "teacherTitlesPassingIdentity": teacher_identity,
        "savedOutputReplay": records,
        "limits": ["Teacher page acceptance is a grammar audit, not a generation pass rate.",
            "Saved native outputs were not regenerated, rescued, retried or relabeled.",
            "All saved pages were already rejected; identity can reduce accepted fields without changing page acceptance."]}
    (HERE / "identity-policy-audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"teacherTitlesPassingIdentity": teacher_identity,
        "savedReplay": [{"caseId": r["caseId"], "arm": r["arm"],
            "oldTitles": r["oldQuality"]["newTitleCount"], "newTitles": r["newQuality"]["newTitleCount"],
            "newIdentityRejections": [t for t in r["titles"] if t["identityErrors"]]}
            for r in records if "oldQuality" in r]}, indent=2))


if __name__ == "__main__":
    main()
