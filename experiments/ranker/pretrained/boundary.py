"""Observable-only text construction and stable external action mapping."""
import json
import math


def clean_product(product):
    return {key: product.get(key) for key in ("title", "category", "description", "price", "shippingCost", "inStock", "tags")}


def variant_text(variant):
    # Deliberately text-only; imageKey/IDs are not predictive fields.
    return {key: variant.get(key) for key in ("title", "imageDescription", "facts", "presentationStyle", "compatibleSurfaces")}


def input_rows(request, catalog):
    context = request.get("context", {})
    visible = {key: context.get(key) for key in ("task", "statedBudget", "statedRequirements", "statedPreferences", "presentationSurface", "presentationPreferences")}
    history = []
    for event in request.get("history", [])[-4:]:
        if event.get("at", "~") <= request["asOf"] and event.get("productId") in catalog:
            history.append({"type": event.get("type"), "confidence": event.get("confidence"),
                            "description": event.get("description"), "presentationStyle": event.get("presentationStyle"),
                            "product": clean_product(catalog[event["productId"]])})
    rows, actions = [], []
    history_summary = [{"type": event["type"], "confidence": event["confidence"], "tags": event["product"]["tags"]} for event in history[-1:]]
    for pid in request["candidateProductIds"]:
        product = catalog[pid]
        for variant in product["presentationVariants"]:
            # Compact field selection is explicit and fixed before evaluation.
            # Preserve all selected candidate facts; tokenizer never truncates.
            text = (f"Request: {visible['task']}\n"
                    f"Budget including shipping: {visible['statedBudget']}; requirements: {visible['statedRequirements']}; preferences: {visible['statedPreferences']}.\n"
                    f"Surface: {visible['presentationSurface']}; presentation preference: {visible['presentationPreferences']}.\n"
                    f"Latest observed history: {history_summary}.\n"
                    f"Product: {product['title']}; category {product['category']}; price {product['price']}; shipping {product['shippingCost']}; stock {product['inStock']}; tags {product['tags']}.\n"
                    f"Approved title: {variant['title']}. Text-only image description: {variant.get('imageDescription')}. Facts: {variant.get('facts')}. Style: {variant.get('presentationStyle')}.")
            rows.append(text)
            actions.append({"productId": pid, "variantId": variant["id"]})
    return rows, actions


def guard_reasons(request, product):
    context = request["context"]
    reasons = []
    if not product["inStock"]:
        reasons.append("out_of_stock")
    budget = context.get("statedBudget")
    if budget is not None and product["price"] + product.get("shippingCost", 0) > budget:
        reasons.append("stated_budget")
    if not set(context.get("statedRequirements") or []).issubset(product.get("tags", [])):
        reasons.append("stated_requirements")
    return reasons


def decisions(request, catalog, actions, scores, model_id, guarded=True):
    if any(not all(math.isfinite(float(value)) for value in score) for score in scores):
        raise FloatingPointError("Decision scores must be finite.")
    product_scores = {}
    variant_scores = {}
    for action, score in zip(actions, scores, strict=True):
        pid, variant = action["productId"], action["variantId"]
        product_scores.setdefault(pid, []).append(float(score[0]))
        variant_scores[(pid, variant)] = float(score[1])
    # Product suitability is averaged across factual variants; presentation utility
    # selects an approved action only after choosing a suitable product.
    ranked = sorted(product_scores, key=lambda pid: (-sum(product_scores[pid]) / len(product_scores[pid]), pid))
    eligible = [pid for pid in ranked if not guard_reasons(request, catalog[pid])]
    if guarded:
        ranked = eligible + [pid for pid in ranked if pid not in eligible]
    selected = (eligible[0] if eligible else None) if guarded else ranked[0]
    compatible = [key for key in variant_scores if key[0] == selected and (not guarded or request.get("presentationCompatibility", {}).get(key[0], {}).get(key[1], False))]
    chosen_variant = max(compatible, key=lambda key: (variant_scores[key], key[1])) if compatible else None
    return {"schemaVersion": 1, "synthetic": True, "recordId": request["recordId"], "modelId": model_id,
            "rankedProducts": [{"productId": pid, "score": sum(product_scores[pid]) / len(product_scores[pid])} for pid in ranked],
            "selectedProductId": selected,
            "presentationAction": {"productId": chosen_variant[0], "variantId": chosen_variant[1]} if chosen_variant else None,
            "timing": {}, "warnings": ["Synthetic trained decision heads on a frozen pretrained backbone; no human lift evidence.",
                                        "Image descriptions are text, not visual understanding.",
                                        *(["Deterministic known-constraint guard applied, not learned safety."] if guarded else [])]}
