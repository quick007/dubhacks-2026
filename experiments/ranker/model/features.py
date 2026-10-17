"""Fixed feature fusion for a parallel decision scorer; no autoregressive decoding.

Catalog tag vocabulary is fixed from catalog facts, never from hidden truth or labels.
Context and history are fused with each candidate's factual attributes. This is an
explicit small tabular/text-tag representation, not a pretrained semantic encoder.
"""
from .common import observable_input, known_violations


class FeatureEncoder:
    def __init__(self, catalog: dict[str, dict]):
        self.catalog = catalog
        self.tags = sorted({tag for product in catalog.values() for tag in product.get("tags", [])})
        self.categories = sorted({product["category"] for product in catalog.values()})
        self.names = ["price", "shipping", "stock", "budget_present", "budget", "budget_margin", "known_violation", "history_count", "preference_count", "requirement_count"]
        self.names += ["category:" + category for category in self.categories]
        self.names += [kind + ":" + tag for kind in ("candidate", "preference_match", "requirement_match", "history_match") for tag in self.tags]

    def encode(self, request: dict, *, history: bool = True) -> list[list[float]]:
        observed = observable_input(request, self.catalog, history=history)
        context = observed["context"]
        preferences = set(context.get("statedPreferences") or [])
        requirements = set(context.get("statedRequirements") or [])
        affinity = {tag: 0.0 for tag in self.tags}
        for event in observed["history"]:
            polarity = {"purchase": 1.0, "save": .7, "add_to_cart": .7, "view": .15, "click": .2, "return": -.8}.get(event["type"], .05)
            weight = polarity * float(event.get("confidence") or 0)
            for tag in event["product"].get("tags") or []:
                affinity[tag] += weight
        rows = []
        budget = context.get("statedBudget")
        for product in observed["candidates"]:
            tags = set(product.get("tags") or [])
            total = product["price"] + product.get("shippingCost", 0)
            row = [product["price"] / 150, product.get("shippingCost", 0) / 20, float(product["inStock"]), float(budget is not None), (budget or 0) / 150, (budget - total) / 150 if budget is not None else 0, float(bool(known_violations(context, product))), len(observed["history"]) / 10, len(preferences) / 5, len(requirements) / 5]
            row += [float(product["category"] == category) for category in self.categories]
            row += [float(tag in tags) for tag in self.tags]
            row += [float(tag in tags and tag in preferences) for tag in self.tags]
            row += [float(tag in tags and tag in requirements) for tag in self.tags]
            row += [affinity[tag] if tag in tags else 0 for tag in self.tags]
            rows.append(row)
        return rows
