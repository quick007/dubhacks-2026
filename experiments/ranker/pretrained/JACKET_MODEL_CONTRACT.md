# Typed jacket model contract

The pinned six-jacket/ten-task input now has a working model-side path through
`jacket_inputs.py`, `jacket_plan.py` and `jacket_contract_runner.py`. Run:

```sh
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.jacket_contract_runner --out /tmp/jacket-contract-new-output
```

The output directory must not already exist. The runner verifies the independently
pinned public snapshot and constructs all ten plans and independent copy jobs.
`jacket_alignment_results/v1/report.json` preserves the actual first run, input
texts, per-task plans, prompts, source hashes and CPU preparation timings. All ten
contracts validate: 15 suitable cards returned out of 30 requested, 55 generation
jobs prepared, every shortfall disclosed. This is not native inference,
generated-copy quality, learned ranking or page latency.

The catalog encoder retains exact name, item price and typed multi-value
attributes. The query encoder retains only public query, optional budget and
hard/preference requirements. External IDs remain separate routing metadata;
extra persona/outcome fields are rejected. Catalog text is independent of the
query. No current backpack checkpoint is certified for these jacket inputs.

Hard requirements use exact attribute membership and an inclusive item-price
budget. Preferences remain soft. Waterproof is not substituted with
water-resistant; size options are not stock guarantees. Shipping and stock are
unknown, image selection unsupported. No unsuitable fillers or fictional image
variants are introduced. Plans can return zero cards with an explicit shortfall;
empty plans create no section or generation jobs.

Caller scores cover the entire source roster, contain finite numbers and cannot
override hard guards. Ties use stable source indices. Caller scores do not prove
learned provenance; grouping is a single deterministic section. Full canonical
plan recomputation rejects altered facts, bindings, counts, images or claims.

Each title/badge/body prompt has its own complete bound facts; headings have the
section attribute intersection as universal support. Each job has isolated
evidence. Jobs need a native runner and jacket-specific output validation before
copy/page success can be claimed. Tokens remain autoregressive within each field.

Legacy backpack source/checkpoint/result hashes are preserved. New jacket callers
must use this versioned path instead of passing nested facts into the old flat
`boundary.clean_product` or old `catalog_fusion.inputs.catalog_text`. No live
storefront/API/shopper inference integration was added. After the normal merge
of teammate main `d8af475`, the actual storefront has 16 products, color variants
and gallery assets. That catalog is separate from this six-jacket import, which
remains pinned to design branch commit
`65b5896b28d4901b95a691fd68436e79ad264e61`. New actual-store catalog work must
use its own source receipt and cannot inherit jacket-domain quality claims.

Validation: 21 targeted CPU tests and an independent corruption audit cover
integrity/types, external ID separation, all ten shortfalls, hard-mask precedence,
empty results, complete score binding, forged provenance and isolated job facts.
The separate seven-atom generalization experiment is a backpack ranking component;
it establishes no jacket, image, generated-copy or complete-page quality.
