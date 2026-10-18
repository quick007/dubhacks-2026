"""CPU-only public-input import from a pinned teammate commit; no model imports.

The source scenario's scripted paths, optimal paths and purchase outcomes are
never copied. Source bytes are hashed for provenance but only products and each
persona.task enter the input snapshot. IDs stay in a separate external map.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = "65b5896b28d4901b95a691fd68436e79ad264e61"
BRANCH = "origin/feat/design-dash"
PRODUCT_SOURCE = "apps/api/fixtures/scenarios/jacket.json"
TASK_SOURCE = "apps/api/fixtures/catalog.json"
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(path):
    data = subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)
    blob = subprocess.check_output(["git", "rev-parse", f"{COMMIT}:{path}"], cwd=ROOT).decode().strip()
    return json.loads(data), {"path": path, "gitBlob": blob, "bytes": len(data), "sha256": sha(data)}


def check_product(task, product):
    """Exact public facts only. Preference failures do not exclude a product."""
    checked = []
    for requirement in task["requirements"]:
        values = product["attributes"].get(requirement["attribute"], [])
        checked.append({"attribute": requirement["attribute"], "value": requirement["value"],
                        "kind": requirement["kind"], "observedValues": values,
                        "met": requirement["value"] in values})
    budget_met = "budget" not in task or product["price"] <= task["budget"]
    eligible = budget_met and all(r["met"] for r in checked if r["kind"] == "hard")
    return {"eligible": eligible, "budgetMet": budget_met, "requirements": checked}


def prepare():
    names = ("model_inputs.json", "external_mapping.json", "guard_receipt.json", "IMPORT_RECEIPT.json")
    if any((OUT / name).exists() for name in names):
        raise FileExistsError("Preserve frozen import; use a new version directory for any regeneration")
    scenario, product_receipt = source(PRODUCT_SOURCE)
    catalog, task_receipt = source(TASK_SOURCE)
    products, tasks, product_ids, persona_ids = [], [], [], []
    for product in scenario["products"]:
        assert set(product) == {"id", "name", "price", "attributes"}
        assert all(isinstance(v, list) and all(isinstance(x, str) for x in v)
                   for v in product["attributes"].values())
        product_ids.append(product["id"])
        products.append({k: product[k] for k in ("name", "price", "attributes")})
    for persona in catalog["personas"]:
        task = persona["task"]
        assert set(task).issubset({"query", "budget", "requirements"})
        assert all(set(r) == {"id", "label", "kind", "attribute", "value"}
                   and r["kind"] in {"hard", "preference"} for r in task["requirements"])
        # Requirement labels are public display text; their IDs are external routing only.
        tasks.append({**{k: task[k] for k in ("query", "budget") if k in task},
                      "requirements": [{k: r[k] for k in ("label", "kind", "attribute", "value")}
                                       for r in task["requirements"]]})
        persona_ids.append({"personaId": persona["id"],
                           "requirementIds": [r["id"] for r in task["requirements"]]})
    assert len(products) == 6 and len(tasks) == 10
    assert len(set(product_ids)) == 6 and len({p["personaId"] for p in persona_ids}) == 10
    inputs = {"schemaVersion": "jacket-public-inputs.v1", "products": products, "tasks": tasks}
    mapping = {"productIdsByInputIndex": product_ids, "tasksByInputIndex": persona_ids}
    guards = []
    for ti, task in enumerate(tasks):
        checks = [check_product(task, p) for p in products]
        eligible = [i for i, check in enumerate(checks) if check["eligible"]]
        guards.append({"taskIndex": ti, "productChecks": checks, "eligibleProductIndices": eligible,
                       "eligibleCount": len(eligible), "requestedCardCountForShortfallAudit": 3,
                       "shortfallToThree": max(0, 3 - len(eligible)),
                       "rankingPerformed": False, "modelCalls": 0})
    outputs = {"model_inputs.json": inputs, "external_mapping.json": mapping,
               "guard_receipt.json": {"schemaVersion": "jacket-exact-guards.v1",
                   "policy": "all hard exact attribute memberships and item price <= stated budget; preferences soft",
                   "shippingAndStock": "unknown; no delivered-price or in-stock eligibility claim",
                   "imageSelectionCapability": "unsupported: source contains no image assets or variants",
                   "tasks": guards}}
    receipt = {"schemaVersion": "jacket-public-import.v1", "sourceBranch": BRANCH,
               "sourceCommit": COMMIT, "sources": [product_receipt, task_receipt],
               "preparerSha256": sha(Path(__file__).read_bytes()),
               "modelInputSha256": sha(canonical(inputs)),
               "excluded": ["bio", "traits", "goal", "name (persona)", "split", "history",
                            "scripted paths", "purchases", "optimalSteps", "run outcomes"],
               "idsAreModelFeatures": False, "nativeCalls": 0, "fitPerformed": False,
               "outputs": {name: {"canonicalSha256": sha(canonical(value))}
                           for name, value in outputs.items()}}
    outputs["IMPORT_RECEIPT.json"] = receipt
    for name, value in outputs.items():
        with (OUT / name).open("x") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
    return {"tasks": len(tasks), "products": len(products),
            "eligibleCounts": [row["eligibleCount"] for row in guards],
            "inputSha256": receipt["modelInputSha256"]}


if __name__ == "__main__":
    print(json.dumps(prepare(), sort_keys=True))
