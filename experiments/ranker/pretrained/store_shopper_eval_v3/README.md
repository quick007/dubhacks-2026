# Held-out synthetic shopper packet v3

Ten paired scenarios, twenty assigned sessions, and unchanged 24-action/120-second budgets. Eight scenarios permit a purchase; two require abstention. This packet is scenario-held-out and catalog-known. It does not establish unseen-product generalization or human conversion lift.

Public queries are separate from private goals and color preferences. `evaluator-gold.json` and `catalog-binding.json` are for independent scoring only. They must never enter shopper prompts, recommender inputs, or model training. `preregistration.json` freezes the design and analysis intent; `overlap-check.json` records exact text checks against the previous ten queries and the current 96-query ranking training roster and independent 32-query ranking evaluation.

Positive eligibility uses the visible product identity/function and item price; private color is a soft preference. No size, stock, waterproof performance or other unavailable fact is required for a positive case. The earlier waterproof visibility failure remains in its original report.

Preparation executes no browser or native model. Both arms receive the same query, needs, catalog, prices, budgets and viewport, with five AB and five BA orders shuffled once under the declared seed. Actual sessions and results belong in a separate versioned output directory. Every assigned session remains in conversion/revenue denominators.

The source receipt must be finalized only after the model owner declares the current storefront snapshot ready. Then run `python -m experiments.ranker.pretrained.store_shopper_eval_v3.validation --verify-live` and use the unchanged TypeScript packet loader. No outcome-conditioned edits are allowed after execution starts.
