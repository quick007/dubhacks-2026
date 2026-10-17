# Page-plan audit

CPU-only audit of the page-plan v1 contract, observable evidence projection and generated-field acceptance. No models/GPU were loaded; no implementation files, UI, harness, commits or pushes were changed by this worker. The only audit files are `test_page_plan_audit.py` and this report.

```sh
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_page_plan_audit -v
```

The initial audit reproduced **eight correctness gaps**. The tests assert the intended rejection or metric behavior, so they fail until implementation fixes land. The parent owns those fixes and posthoc revalidation of saved original outputs; generation should not be rerun merely to replace inconvenient results.

| Finding                                                                                   | Minimal target/fix                                                                                                                                                                                                      |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A completed plan accepts `source: generated` with `text: null`.                           | `page_plan.validate_page_plan`: enforce provenance/text relations; generated means a nonempty accepted string, pending/rejected/none mean null.                                                                         |
| Arbitrary novel factual wording may be labeled `approved-fallback`.                       | Restrict fallback title to exact catalog/approved title; reject fallback provenance on unsupported field types.                                                                                                         |
| `recordId` need not match the observable request.                                         | Require exact plan/request record identity.                                                                                                                                                                             |
| `not compact` is treated as a positive compact preference and enables a badge.            | `fusion.supported_preferences`: conservatively recognize explicit positive tags; do not infer positive intent from negated substring matches.                                                                           |
| A future save encoded with a timezone offset becomes historical under lexical comparison. | `boundary.visible_history`: parse aware ISO timestamps, compare/sort in UTC; reject malformed/naive times. Test: `2026-09-30T23:30:00-02:00` is after the midnight-Oct-1-Z cutoff.                                      |
| A valid catalog fact ID may claim observed alignment without observable support.          | Bind `approvedFactIds` to explicit supported preferences or the attributes of referenced valid observed-save reasons. The repro substitutes durable evidence for a compact-only preference on a product with both tags. |
| An accepted generic title with no support reports `titleObservedAlignmentRate: 1.0`.      | `evidence_plan.validate_generated`: compute alignment over supported cards only and return null when the support denominator is zero.                                                                                   |
| Embedded NUL passes lexical validation and yields `allFieldsFulfilled: true`.             | Reject control characters globally before accepting any generated fields.                                                                                                                                               |

The adversarial controls verify that unknown/future-Z/weak-view/private history cannot justify badges; invented reason IDs fail; unsupported section attributes fail unless true for every section product; duplicates require explicit across-section allowance and remain forbidden within a section; truncated, markup and duplicate-line responses do not fulfill the whole page.

## Contract fixture

The existing six-card fixture passes `validate_page_plan` while pending and correctly fails `require_complete=True`. Its actual `pagePlan` JSON literal also passes the declared `PagePlanV1` TypeScript interface under a temporary, strict `tsc --noEmit` check with no repository changes. Catalog/image IDs, section/product ordering, six unique cards and nullable field shapes are valid. TypeScript's current `GeneratedFieldV1` does not express provenance/text relationships; stronger semantic checks remain necessary even for type-valid input.

## Saved experiment interpretation

The already completed original `evidence-plan-v1/benchmark.json` was inspected, not modified. These are original-validator results and must be distinguished from any stricter posthoc audit:

| Cards | No-plan full completion | Plan full completion | Fully fulfilled scale page |
| ----- | ----------------------: | -------------------: | -------------------------- |
| 1     |               3575.4 ms |            3568.6 ms | Neither arm                |
| 3     |               4675.1 ms |            4925.4 ms | Neither arm                |
| 6     |               7768.9 ms |            7724.3 ms | Neither arm                |

Across all twelve fixed cases the original validator marked zero no-plan pages and one plan page fully fulfilled. Rejected fields remain failures even where generation completed. Both explicit no-history/no-preference control arms produced the correct NONE badge, but neither fulfilled the complete page. The original no-plan control's 1.0 observed-alignment metric is affected by the zero-support denominator gap above.

Selection diversity is identical across arms by construction: the evidence plan conditions wording after the same trained selection/order. For the 1/3/6-card pool, unique-product rate is 1.0, attribute coverage is 2/4/6, and mean pairwise tag Jaccard redundancy is 0/0.2222/0.2222. Explicit-preference coverage is 0.5/1.0/1.0. These measure selected-product attributes, not generated-title linguistic diversity, and do not show that an evidence plan improves selection. One measurement per page size is neither p95 evidence nor a production-latency guarantee. Synthetic results do not establish human conversion lift.

## Resolution verification

The parent applied implementation fixes for all eight reproduced gaps. A fresh CPU-only run of this audit file then passed **13 of 13 tests**. The parent additionally reports that the complete 55-check inherited/discovered regression suite passes. This worker independently observed the focused 13-test pass; it did not run GPU/native/model tests or alter the parent's implementation.

The original `benchmark.json` remains preserved. The parent is writing stricter saved-output revalidation separately as `benchmark-revalidated.json`; original counts above must not be relabeled as posthoc results. No actual generation was rerun for this audit. The root also reports that the corrective fit regressed top-product accuracy (58.3% versus 66.7%) and counterfactual performance (zero versus one of three pairs); an evidence-plan wording change does not establish a ranking improvement.
