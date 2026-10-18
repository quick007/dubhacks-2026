"""Versioned observable jacket boundary, independent of historical backpack code.

No model/runtime imports. External IDs are routing metadata, never encoder
inputs. The frozen loader trusts pinned canonical digests, not hashes supplied
by a possibly modified receipt. Input strings and attribute arrays are preserved
exactly; guards do not parse free text or infer unsupported facts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

INPUT_VERSION = "jacket-public-inputs.v1"
CATALOG_TEXT_VERSION = "jacket-catalog-text.v1"
TASK_TEXT_VERSION = "jacket-task-text.v1"
GUARD_VERSION = "jacket-typed-guards.v1"
DEFAULT_SNAPSHOT = Path(__file__).resolve().parent / "jacket_alignment_v1"
_PINNED = {
    "model_inputs.json": "dfe8123ad96da8a86a664961bcf4a706a142feacf8b52c390dfc0742b97cb1e1",
    "external_mapping.json": "0963bdaf5e552709f9f5c035936d5c1c2512406d50be2baa1694c9727ff37a2a",
    "guard_receipt.json": "6028718911f18af66dc4da12f97316c88ddbbda4c92bc6b8dcfc607fa1e3754c",
    "IMPORT_RECEIPT.json": "9ccce4821ae008ae6e4aa9b63715bd4e32b4ec9a6afec57a0c91cd3fa719cbec",
}


class JacketInputError(ValueError):
    """Malformed, unsupported or unverified public inputs."""


def _object(value, required, optional=(), *, where):
    if type(value) is not dict or any(type(k) is not str for k in value):
        raise JacketInputError(f"{where} must be an object with string keys")
    if not set(required) <= value.keys() or not value.keys() <= set(required) | set(optional):
        raise JacketInputError(f"{where} has missing or unsupported fields")


def _text(value, where):
    if type(value) is not str or not value.strip():
        raise JacketInputError(f"{where} must be a nonempty string")
    return value


def _money(value, where):
    if type(value) not in (int, float):
        raise JacketInputError(f"{where} must be a finite nonnegative number, excluding bool")
    try:
        valid = math.isfinite(value) and value >= 0
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise JacketInputError(f"{where} must be finite and nonnegative")
    return value


def _list(value, where):
    if type(value) is not list:
        raise JacketInputError(f"{where} must be a list")
    return value


def validate_product(product):
    """Return a fresh exact public product; reject IDs and all extra metadata."""
    _object(product, {"name", "price", "attributes"}, where="product")
    _text(product["name"], "product.name")
    _money(product["price"], "product.price")
    attributes = product["attributes"]
    if type(attributes) is not dict:
        raise JacketInputError("product.attributes must be a string-to-list object")
    for attribute, values in attributes.items():
        _text(attribute, "attribute name")
        _list(values, f"attribute {attribute}")
        for value in values:
            _text(value, f"attribute {attribute} value")
        if len(set(values)) != len(values):
            raise JacketInputError(f"attribute {attribute} contains duplicate values")
    return copy.deepcopy(product)


def validate_task(task):
    """Return public query/budget/typed requirements, with no persona projection."""
    _object(task, {"query", "requirements"}, {"budget"}, where="task")
    _text(task["query"], "task.query")
    if "budget" in task:
        _money(task["budget"], "task.budget")
    for requirement in _list(task["requirements"], "task.requirements"):
        _object(requirement, {"label", "kind", "attribute", "value"}, where="requirement")
        for key in ("label", "attribute", "value"):
            _text(requirement[key], f"requirement.{key}")
        if type(requirement["kind"]) is not str or requirement["kind"] not in {"hard", "preference"}:
            raise JacketInputError("requirement.kind must be hard or preference")
    return copy.deepcopy(task)


def validate_mapping(products, tasks, mapping):
    """Check positional cardinality and per-namespace ID uniqueness; copy on return.

    Requirement IDs are unique within their task, not across unrelated tasks.
    This validates generic external mapping structure; only load_snapshot also
    proves correspondence to the pinned teammate source.
    """
    for product in _list(products, "products"):
        validate_product(product)
    for task in _list(tasks, "tasks"):
        validate_task(task)
    _object(mapping, {"productIdsByInputIndex", "tasksByInputIndex"}, where="external mapping")
    product_ids = _list(mapping["productIdsByInputIndex"], "product IDs")
    task_ids = _list(mapping["tasksByInputIndex"], "task mapping")
    if len(product_ids) != len(products) or len(task_ids) != len(tasks):
        raise JacketInputError("external mapping cardinality differs from public inputs")
    for identifier in product_ids:
        _text(identifier, "product ID")
    if len(set(product_ids)) != len(product_ids):
        raise JacketInputError("duplicate product IDs")
    persona_ids = []
    for task, entry in zip(tasks, task_ids, strict=True):
        _object(entry, {"personaId", "requirementIds"}, where="task mapping entry")
        persona_ids.append(_text(entry["personaId"], "persona ID"))
        requirement_ids = _list(entry["requirementIds"], "requirement IDs")
        if len(requirement_ids) != len(task["requirements"]):
            raise JacketInputError("requirement ID count differs from public requirements")
        for identifier in requirement_ids:
            _text(identifier, "requirement ID")
        if len(set(requirement_ids)) != len(requirement_ids):
            raise JacketInputError("duplicate requirement IDs within task")
    if len(set(persona_ids)) != len(persona_ids):
        raise JacketInputError("duplicate persona IDs")
    return copy.deepcopy(mapping)


def capabilities():
    """Explicit missing source evidence, never synthetic defaults."""
    return {"shipping": "unknown", "stock": "unknown", "imageSelection": "unsupported"}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def catalog_text(product):
    """Task-independent, identity-preserving product text; accepts no task or IDs."""
    return canonical({"schemaVersion": CATALOG_TEXT_VERSION, "product": validate_product(product),
                      "capabilities": capabilities(), "priceScope": "item-price"})


def task_text(task):
    return canonical({"schemaVersion": TASK_TEXT_VERSION, "task": validate_task(task),
                      "budgetScope": "inclusive-item-price-ceiling"})


def guard_product(task, product):
    task, product = validate_task(task), validate_product(product)
    checked = []
    for index, requirement in enumerate(task["requirements"]):
        values = product["attributes"].get(requirement["attribute"], [])
        checked.append({"requirementIndex": index, **requirement,
                        "observedValues": list(values), "met": requirement["value"] in values})
    budget_met = "budget" not in task or product["price"] <= task["budget"]
    return {"schemaVersion": GUARD_VERSION,
            "eligible": budget_met and all(r["met"] for r in checked if r["kind"] == "hard"),
            "budgetMet": budget_met, "budgetSpecified": "budget" in task,
            "budgetScope": "inclusive-item-price-ceiling", "requirements": checked,
            "unsupported": capabilities(), "learned": False}


def eligibility(task, products):
    """Guard all supplied rows in source order; performs no learned ranking."""
    task = validate_task(task)
    checks = [guard_product(task, product) for product in _list(products, "products")]
    eligible = [index for index, check in enumerate(checks) if check["eligible"]]
    return {"schemaVersion": GUARD_VERSION, "productChecks": checks,
            "eligibleProductIndices": eligible, "eligibleCount": len(eligible),
            "rankingPerformed": False, "unsupported": capabilities()}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise JacketInputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_verified(path, expected):
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               JacketInputError(f"nonfinite JSON constant: {value}")))
        actual = hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise JacketInputError(f"invalid snapshot artifact {path.name}: {exc}") from exc
    if actual != expected:
        raise JacketInputError(f"pinned canonical digest mismatch: {path.name}")
    return value


def load_snapshot(directory=DEFAULT_SNAPSHOT):
    """Load only the pinned v1 import, checking receipt and all input/guard hashes.

    Canonical-content integrity tolerates formatting-only changes. No Git, model,
    tokenizer or source importer is executed. A new import needs a new loader
    version/pin, not an untrusted receipt claiming its own expected hashes.
    """
    directory = Path(directory)
    loaded = {name: _read_verified(directory / name, expected) for name, expected in _PINNED.items()}
    inputs = loaded["model_inputs.json"]
    _object(inputs, {"schemaVersion", "products", "tasks"}, where="snapshot")
    if inputs["schemaVersion"] != INPUT_VERSION:
        raise JacketInputError("unsupported snapshot version")
    products = [validate_product(p) for p in _list(inputs["products"], "snapshot products")]
    tasks = [validate_task(t) for t in _list(inputs["tasks"], "snapshot tasks")]
    mapping = validate_mapping(products, tasks, loaded["external_mapping.json"])
    receipt = loaded["IMPORT_RECEIPT.json"]
    for name in ("model_inputs.json", "external_mapping.json", "guard_receipt.json"):
        if receipt["outputs"][name]["canonicalSha256"] != _PINNED[name]:
            raise JacketInputError("receipt does not match independently pinned content")
    if receipt["modelInputSha256"] != _PINNED["model_inputs.json"]:
        raise JacketInputError("model input receipt mismatch")
    guard_receipt = loaded["guard_receipt.json"]
    if len(guard_receipt["tasks"]) != len(tasks):
        raise JacketInputError("guard receipt cardinality mismatch")
    for task, row in zip(tasks, guard_receipt["tasks"], strict=True):
        actual = eligibility(task, products)
        if any(actual[key] != row[key] for key in ("eligibleProductIndices", "eligibleCount")):
            raise JacketInputError("guard receipt does not reproduce public eligibility")
    return copy.deepcopy({"schemaVersion": INPUT_VERSION, "products": products, "tasks": tasks,
                          "externalMapping": mapping, "receipt": receipt,
                          "guardReceipt": guard_receipt})
