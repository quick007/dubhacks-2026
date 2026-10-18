# Store transparency audit v1

Existing grounding and budget safeguards have useful CPU coverage. Explicit
transparency supervision was missing, so a new **untrained public-only packet
with eight TRAIN and four evaluation cases** now supplements those checks.
No model fit, GPU, service/frontend edit or Git operation was performed.

## Existing evidence and gaps

- `store_function_v3/public-cases.json` has 32 TRAIN and 16 evaluation requests;
  all 48 have null budgets. `store_function_v3/guard-probes.json` separately
  covers `inclusive-24-mug`, `below-mug-price`, `rain-budget-shortfall`, unknown
  color/stock/size and exact declared category/color evidence. These are
  deterministic guards, not examples of learned budget reasoning.
- `test_store_function_adaptation.py::test_guard_probes_precede_high_scores_and_disclose_shortfalls`
  checks eligibility against deliberately unfavorable scores. The demo tests
  `test_budget_strictness_and_exact_identity`,
  `test_typed_budget_is_inclusive_and_overrides_query_budget` and
  `test_conservative_ceilings_and_delivered_budget` add exclusive `under`,
  inclusive typed ceilings, contradictory ceilings and unsupported delivered
  totals. `test_unknown_stock_size_fail_closed_and_colors_bind_actual_variant`
  covers unknown hard facts without inventing availability.
- `store_copy_v2/fields.train.jsonl` contains 320 synthetic field targets,
  including 96 source-bound body sentences. It trains selected name/color/item
  price wording, not thermal retention, delivery promises or provenance
  explanations. `test_store_copy_data_v2.py` covers exact source facts,
  unsupported claims, numeric formatting and fact swaps. Passing its finite
  grammar is not comprehensive English factuality.
- `test_store_demo_engine.py::test_collection_price_scope_is_withheld_without_rewriting`
  now guards the independently observed misleading `collection of` wording.
  Other demo tests cover unsupported use/shipping, cross-product/color claims,
  absent native receipts and preserving null fields. The previous v1 receipts
  and qualitative findings remain unchanged.
- Backend `store_demo_engine.py` exposes `COPY_DISCLOSURE`, `NEUTRAL_COPY_REQUEST`
  and event provenance: shopper query affects ranking only; fresh copy uses
  selected public facts plus a neutral request; guards/exact-name rules and
  gallery bindings are deterministic. No hidden persona is used.
- Visible `apps/storefront/src/local-model.ts::localModelStatus` currently
  abbreviates this to learned product order, existing gallery, timings and
  fresh/withheld field counts. It does not surface the neutral-copy conditioning
  or deterministic budget/exact-name distinction. Full truth is in receipts
  and `STORE_DEMO_SERVICE.md`; this is a disclosure gap, left for the integration
  owner rather than an unsolicited UI edit here. `local-model.test.ts` already
  checks exact catalog price/identity/color, null fields, shortfalls and native
  copy disappearing after a variant change. `ProductCard.tsx` retains catalog
  `product.price` and falls back to `product.name`; those are not native text.

## Small new packet

`store_transparency_v1/public-cases.json` contains only standalone public task
inputs; case/split metadata, explanations and support labels remain external in
`train-labels.json` and `evaluation-labels.json`. It is not connected to the
current ranker, copy model or service. The manifest marks training, tokenization
and native execution false. It is hand-authored synthetic material, not sealed
or independent human evaluation. Public operation facts describe pipeline roles;
no private persona traits, goals, outcomes or hidden preference truth appear.

| Cases | Meaningful distinction |
|---|---|
| TRAIN 00–02 | $24 mug fits typed inclusive $24; fails exclusive `under $24`; delivered-total ceiling is unsupported. |
| TRAIN 03–05 | Name/color/$24 sentence is grounded; invented 12-hour heat retention is unsupported; correct lamp amount with unsupported collection/bundle scope is withheld. |
| TRAIN 06–07 | Truthful learned-rank/fact-conditioned-copy/deterministic-gallery disclosure versus falsely attributing guards, images and catalog prices to learned generation. |
| EVAL 00 | Exact mug exceeds inclusive $23.99; do not substitute another item for the ineligible named product. |
| EVAL 01 | Public mug facts do not establish free shipping. Unsupported means insufficient evidence, not proven physically false. |
| EVAL 02 | Claim that shopper query conditions generated wording is false for this agreed neutral-copy path. |
| EVAL 03 | Original name, selected color and $24 can remain visible catalog metadata while all native card text is null. Do not call it accepted generated copy. |

## CPU verification

Added `test_store_transparency_v1.py`: seven tests bind labels to the pinned
catalog, run budget cases under adversarial scores, check supported/unsupported
claims against the practical screen, validate disclosure truth-table labels,
exercise the real rank-input versus neutral-copy-input boundary, reject private
request fields and distinguish zero accepted native fields from retained
metadata. These tests do not purport to evaluate arbitrary-prose inference or
trained-model transparency behavior.

```text
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -B -m unittest \
  experiments.ranker.pretrained.test_store_transparency_v1 \
  experiments.ranker.pretrained.test_store_demo_engine.DemoContractTests
23 tests passed (7 new, 16 existing).
```

The system Python lacks Torch; the existing project environment was used for
these CPU imports. No model was constructed or native execution launched.
Remaining gaps are actual future training/evaluation on this packet, independent
human assessment of disclosure clarity, and any agreed UI expansion. Guard
passing, operational truth labels and visible receipt provenance must continue
to be distinguished from learned reasoning or conversion evidence.
