# Page-plan v1 consumer interface

[page-plan.v1.ts](page-plan.v1.ts) is an additive TypeScript interface for model experiment consumers. Existing fusion responses are unchanged. [page-plan.v1.fixture.json](page-plan.v1.fixture.json) contains six actual synthetic catalog IDs, matching image-caption IDs, an observable request, and a pending six-card response. The fixture is a contract example, not model output; pending fields are never claimed as generated.

The versioned response references section IDs/types and explicit section order, product IDs and product order per section, image-caption IDs, approved catalog fact IDs, and observed history reason IDs. The model experiment uses a fixed maximum-three-card section grouping; this is an exposed structural policy, not a trained layout head. Trained product scores supply selection/order, and the same frozen backbone's decoder must generate new titles, section headings and supported optional badges in the measured proof. Generated fields record `generated`, `rejected`, `none`, `approved-fallback`, or `pending` provenance.

`page_plan.validate_page_plan` enforces catalog membership, stock, stated budget/requirements, image compatibility, complete unique ordering, known evidence references, factual card and section wording, card counts and explicit shortfalls. Duplicates are forbidden by default; only an explicit `duplicatePolicy: "explicit-allow"` permits repetition across sections. Within-section product references remain unique. Section headings may assert only attributes shared by every product in the section. This restrictive lexical validator is not semantic factuality proof.

History reasons reference only timestamp-filtered observed high-confidence saves and exact shared catalog attributes. They assert that an action occurred, never that it reveals private intent. Weak views, future events, unknown products and private persona fields cannot justify a badge. A no-history request may still justify a badge through explicit current preferences; with neither history evidence nor explicit supported preferences, `NONE` is correct.

```sh
.tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_page_plan -v
```

Scope stays under model experiments. No dashboard, storefront or shopper harness consumes this contract yet. Full page-plan latency and diversity/preference diagnostics belong to the measured experiment, including every heading/card generation and rejection. The prior roughly three-second single-product result remains below the near-traditional-page latency target. No synthetic result establishes human conversion lift or novelty.

Adversarial validation also enforces matching request identity, nonempty generated text, null text for pending/rejected/none, exact approved fallback titles, positive exact attribute support (negated prose is unknown), timezone-aware history cutoffs, and fact references bound to explicit preferences or referenced observed actions. See [PAGE_PLAN_AUDIT.md](PAGE_PLAN_AUDIT.md).
