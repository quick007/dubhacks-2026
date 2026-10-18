# Fixed first-section payload workload

`ABOVE_FOLD_WORKLOAD.json` freezes one logical first section and its three
ordered complete cards. This is a separate benchmark endpoint on the path to
the user's full personalized grid. It does not replace a 12/24-card objective
with three cards, change the UI, or establish a browser above-the-fold paint
time. The full page keeps its own unchanged validation and completion metric.

## Inspected contracts and required scope

The present storefront renders a hero before a three-card collection. Its
section includes an h2 and introductory sentence; each card renders CSS
`ProductArt`, original product name, price, a marketing sentence stored as
`product.category`, and Add to cart. Thus the visible body line cannot disappear
from a complete-view benchmark. Actual viewport fit beneath the hero varies by
layout; this workload deliberately names a fixed logical three-card view.

The shared TypeScript/Go HTTP schema currently exposes only headline,
showShipping and recommendationOrder interventions. The recommendation adapter
is deterministic. Its model adapter returns mock text. The separate experimental
`PagePlanV1` describes product/image choices, titles, badges and section headings,
but has no per-card body-copy field. The fixture storefront does not consume it.
These limitations are recorded with hashes of the inspected sources in the
workload; none of those UI, API, model or harness files was edited.

The prospective complete payload requires ten genuinely generated values:

| Values | Meaning | Per-row sampled cap |
| --- | --- | ---: |
| S1 | First section heading | 16 |
| T1, T2, T3 | Fresh visible product titles | 24 |
| B1, B2, B3 | Supported badge or actual NONE | 8 |
| C1, C2, C3 | Fresh factual visible card body copy | 32 |

The summed scoped cap is 208, including the C rows. This is a prospective
workload budget, not a measured cap-fit/performance result. Existing S/T/B caps
stay unchanged. Current fourteen-row native page runs omit C rows, so they are
**incomplete for this workload**, even when their seven first-section S/T/B
values pass. No new decoder, prompt variant, template answer or native work was
implemented here. A future bounded native producer must add genuine C generation
and check actual target/context/cap fit first.

Each card payload must also bind the exact ordered product ID, compatible image
variant ID and nonempty image key, original catalog name, actual price and
shipping cost, and catalog category metadata. Original name is always present as
a distinct display field; `NONE` becomes an absent badge (`null`) only when the
model actually returns that exact sentinel. The default existing static section
intro `A few of our favorites.` and CTA `Add to cart` remain required visible
metadata. They are not generated answers. Personalized section-intro copy would
require an extra generated row in a new workload version; the helper does not
silently omit a visible personalized field.

## Validation boundary

`above_fold.validate_view(rows, plan, request, catalog, payload)` creates a new
first-section validation projection from the original plan without mutating it.
It requires unambiguous section/product order and validates all three selected
products and images against the full original request/catalog. It filters only
the ten required rows. Missing, duplicate, multiline, labeled, unfinished or
wrong-cap required values reject the view. No raw output is repaired. A rejected
or unfinished below-fold S2/T4/B4/etc. cannot invalidate an independently
completed first-section payload. The original full-page validator is unchanged
and still rejects the original full page when any of its fields fail.

The seven S/T/B rows go through the existing scheduler assembly and policy.
Strict raw-row checks apply even if an older assembler retains the first line
of a malformed row. C rows use the existing fusion copy factuality, character
length, stop and observable-support checks, and cannot exactly copy catalog
description/category/image-description text. The complete payload must contain
the actual accepted values and exact selected metadata; validating only token
arrival or a partially assembled object is insufficient. The helper returns
`fullPageValidated=false` and retains the original requested full-page count.

The current generated-title factuality, identity, novelty and observed-support
rules remain mandatory. A generic title does not become acceptable merely
because originalName is now a separate visible field. The user's desired
subtitle-bound title freedom needs a distinct prospective title-policy version,
separate receipts, and a new comparison; it must never inflate old reports.

Original names are checked against the catalog. Explicit complete catalog names
in the public task must occur in the visible three cards; a caller may also
provide public-query-resolved `exact_product_ids`, which must be visible. No
partial-name, nickname, ambiguous-capacity or semantic query resolver was built.
Those cases require an explicit future resolution contract. This check is a
constraint boundary, not evidence that ranking or query interpretation improved.

## Timing and availability

`above_fold.payload_timing` uses **all ten** required row completion timestamps.
`completedMs` is relative to the native decoder's actual wall-start. Supply a
measured `planning_prefix_ms` from the copy-stage measurement origin to that
same decoder start; do not add overlapping phase timers. The token-completion
lower bound is:

```text
planning_prefix_ms + max(S1,T1,B1,T2,B2,T3,B3,C1,C2,C3 completedMs)
```

The helper reports an actual `copyStagePayloadReadyMs` only when the whole view
validates and a receipt for its exact serialized payload hash records
`validatedAndSerializedAtMs` at or after that lower bound and
`metadataReadyAtMs`. Missing/invalid views have no accepted ready-time sample.
First-token timings never enter this calculation. Payload validation and
serialization must actually have happened; callers must not fabricate the
receipt from a sum of phase estimates.

This distinction matters now: `parallel_decode.decode_rows` stamps EOS arrival
per row, but calls tokenizer.decode only after every row finishes. Earlier
required-field EOS timestamps do not prove an early actual payload existed.
Without incremental scoped assembly, the real payload receipt may occur after
all below-fold rows. The helper retains only a lower bound when no actual
completion receipt exists, rather than claiming unimplemented early delivery.

The present prototype is copy-stage only. Ranking, user encoding and planning
before that copy stage are excluded and `fullRequestViewReadyMs` remains null.
A future end-to-end benchmark must start before user encoding/ranking, carry the
same completion/metadata receipts and retain full-page time separately. No
request-latency or usable-page speedup is established by these CPU helpers.

Selected image identity/key is validated, but browser image load/decode, CSS
art rendering and paint are unmeasured. Model image variants are text-described
fixture choices, not pixel inputs, and the existing UI's CSS art is not wired to
those keys. A real visual-view metric needs actual UI asset/render readiness
receipts; this package cannot establish them. No UI/browser/harness work was
performed.

## Verification and product limits

Eight CPU tests pass, including required body copy, strict unchanged title
identity despite visible original name, exact-name constraints, price/image
bindings, malformed values, below-fold failure with unchanged full-page failure,
all-row timing and exact payload receipts. The fixtures contain synthetic
teacher text and times solely for unit tests; they are not model or UI results.

```sh
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_above_fold -v
```

The broader product target still includes learned ranking/section/image
personalization and genuine fresh copy from the same LLM fusion architecture.
Grouping in current experiments remains deterministic. Conversion is the
eventual primary product outcome; current synthetic correctness, selection
labels and completion times are proxies and establish no human conversion or
revenue lift. No production integration, new external service, native execution
or GPU run was introduced.
