# Actual-store paired shopper smoke packet v1

Ten hand-authored synthetic development scenarios, two arms each: **20 planned
runs**, one observation per arm. This is an initial shopper-agent smoke test,
not human conversion or causal personalization evidence. No browser/model run
was performed by this packet's preparation. Execution receipts live separately;
missing, failed and timed-out runs remain in the fixed denominator.

| Pair | Public need | Evaluator outcome class |
|---|---|---|
| 01 | Notebook for quick notes | Satisfiable |
| 02 | Drinking-water bottle under $40 | Satisfiable |
| 03 | Bedside table light up to $70 | Satisfiable |
| 04 | Grocery carry item up to $50 | Satisfiable |
| 05 | Zipped cable pouch up to $30 | Satisfiable |
| 06 | Cobalt mug up to $24 | Satisfiable; actual color choice matters |
| 07 | Carry bag up to $90, excluding backpacks | Satisfiable; multiple bag forms accepted |
| 08 | Explicitly waterproof jacket up to $130 | Satisfiable; one qualifying product is legitimate |
| 09 | Exact Trail mug strictly under $24 | Known empty; no cheaper unrelated substitution |
| 10 | Waterproof jacket with verified stock up to $130 | Unknown required fact; abstain without inventing stock |

## Three information boundaries

- `public-observations.json`: only the query and explicitly empty typed extras
  go to the recommender. Browser arms submit the same public query. Private
  needs must not be appended to it.
- `shopper-instructions.json`: private goal, preferences and hard constraints,
  with typed comparisons and natural descriptions. The shopper gets no target
  product/variant IDs, correct selectors, gold answers, existing Run object or
  the other arm's decision. Explicit product names in a shopper's own public
  exact-name request are legitimate observations, not injected route IDs.
- `evaluator-gold.json` and `catalog-binding.json`: independent evaluator only.
  Gold uses declared actual names/categories/search terms/prices/colors plus
  disclosed ordinary semantic function annotations. These annotations never
  become inference keyword aliases. Unknown stock remains absent, not false.
  The existing 11 illustrative image-asset personas are not evaluation truth.

`validation.shopper_case()` merges only public, private and plan rows into the
new `shopper-session.v1` contract. `shopper_assignment()` emits only shopper
needs, public context, common instructions and the selected entry URL. Neither
projection reads evaluator gold or catalog files; tests enforce this boundary.
The framework owner in `apps/runner/src/shopper-contract.ts` owns actual session
and evidence validation; this directory does not duplicate orchestration.

## Pairing, budgets and honest isolation

`pair-plan.json` freezes five generic→local and five local→generic pairs. Each
pair has one seed, identical private needs/public query, default UI settings,
viewport 1280×960 and **24 actions / 120 seconds per arm**. Pair02 is generic
first, seed 2026101802. Wall time starts at navigation dispatch and includes
native queue/waits. Model startup is reported separately. Fresh shopper agents
must decide independently; do not reuse another arm's memory or choice.

Default entry URLs are `http://127.0.0.1:4312/` and the same URL with
`?model=local`. The named local origin `http://shop.dubhacks.localhost:1356/`
is the declared alternative. No remote URL or other setting difference is
allowed. Submit the public query through visible search after opening the arm.
Only a fake order with quantity one may be confirmed; justified abstention is
the intended outcome for the two no-provable-eligible cases.

Use a fresh React state and observe an empty initial cart. A fresh Chrome tab
normally shares origin storage: record `kind=fresh-tab`, `sharedStorage=true`
and `storageCleared=false`. A genuinely separate storage context may claim
isolation only with actual provider evidence. Record browser/tab/context,
initial-cart evidence, missing screenshots/traces, failures and fallback honestly.
No fixed first-button purchase script qualifies as an independent shopper.

## Catalog binding and CPU checks

Preparation executed only installed pure TypeScript catalog exports and verified
the current raw export exactly against the pinned actual-store source: 16
products and 48 declared color variants. Original public-input/export/mapping
hashes, selected current fact-source hashes and the independent typed evaluator
catalog digest are in `catalog-binding.json`. Generated UI rendering can change
separately; runtime facts must continue to match before execution. Price means
item price; stock, sizes and per-item delivered cost are not invented.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m experiments.ranker.pretrained.store_shopper_eval_v1.validation --verify-live
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest experiments.ranker.pretrained.store_shopper_eval_v1.test_packet
```

Tests cover hidden-label exclusion, projections that do not read gold, exact
paired conditions, budget equality versus strict exclusion, meaningful backpack
negation, actual Cobalt variant binding and known-empty versus unknown-stock
abstention. No legacy frontend, API, runner, model checkpoint or old packet is
edited by this preparation. Legacy runner ownership remains: Go owns orchestration
and metadata; the old Playwright adapter is scripted and is not this shopper
agent. Root/framework owners supply fresh headed Chrome agents and sidecars.
