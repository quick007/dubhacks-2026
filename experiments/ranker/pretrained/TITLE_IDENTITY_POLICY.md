# Proposed generated-title identity contract

CPU-only design audit; no validator edits, model execution, or generated-output
reevaluation. The proposed helper is `identity_errors(title, selected, catalog)`.
It supplements the existing factual-claim, novelty, preference-support and output
format checks. Passing identity alone never means that a title is accepted.

## Public catalog observations

Inspected only `data/ranker/datasets/v2/catalog.json`. There are 84 products, all
with category `backpacks`. Public identity fields are `title`, `category` and
`tags`; there is no separate brand field. Titles combine a name, capacity and
`Daypack`. Capacities are 14L, 17L, 20L, 23L and 26L; 18L is appropriate only for a
synthetic policy test. Tags are compact, durable, laptop-sleeve, lightweight,
organizer, roomy and water-resistant.

There are 47 distinct complete titles; 26 complete-title groups are duplicated.
For example, `Pine 26L Daypack` appears with laptop-sleeve/compact and separately
with lightweight/organizer. Even capacity plus both tags is not universally
unique: 25 such signatures identify one product, while 23 signature groups
collide. Neither a capacity nor a name should be reported as a unique SKU.

## Conservative acceptance rule

Accept identity if either condition holds:

1. The title contains a distinctive selected-name anchor. Derive it from the
   selected public title after removing capacity literals, recognized product
   type/category terms, selected tag terms and `GENERIC_WORDS`. Match complete
   normalized tokens, never substring overlap. `Pine` is a name anchor; `compact`,
   `travel`, `pack`, `daypack`, `choice`, `18`, and the capacity unit `l` are not.
   This preserves a recognizable selected product name without requiring the
   entire prewritten title or claiming SKU uniqueness. Several products may
   legitimately share a name anchor.
2. The title contains an explicit selected product type plus a factual
   discriminator, and the conjunction of **the facts actually mentioned**
   identifies exactly the selected product in the supplied public catalog.
   Discriminators are exact capacity literals from the selected title and/or
   complete selected tag phrases. Product types come from the selected title or
   a deliberately bounded category alias table; `choice`, `companion`, `option`,
   `style`, `travel`, `everyday` and `selected` are framing, not product types.

For the second condition, construct the mentioned signature, then find every
catalog product satisfying that exact conjunction: type, mentioned capacity if
any, and every mentioned tag. The selected product must match and there must be
exactly one match. Do not require unmentioned selected tags to match: otherwise
an ambiguous output could appear unique through facts absent from its text.
Require an explicit product type and at least one discriminator even in a
single-item catalog; roster size cannot make empty/generic wording informative.

Use the full current public `catalog` argument for uniqueness. Do not silently
reduce it to selected cards, eligible products or the current candidate order.
This keeps an identical title's identity policy stable when ranking, stock or
page cardinality changes. If a future contract intentionally scopes uniqueness
to a public request pool, declare and test that separate policy explicitly.

Reject a title that introduces a distinctive different product's name anchor,
even if a valid selected signature is also present. Unknown words, wrong sizes,
unselected tags and forbidden claims still fail the existing lexical policy.
Do not treat price/shipping numerals, descriptions, presentation titles or
arbitrary request wording as identity facts. No opaque IDs enter this decision.

## Matching details and integration

Reuse Unicode NFKC and case normalization. Name and product-type matching require
whole tokens. A compound tag matches as its complete normalized phrase: a
`water-resistant` tag can match the normalized phrase `water resistant`; seeing
only `water` or `resistant` is not enough. Require a complete selected capacity
literal such as `18L`, with boundaries; `18` alone cannot match it. Keep the
existing numeric literal rule unchanged: do not introduce `18 L`, `18-litre` or
`18.0L` paraphrases unless the selected public title and current lexical rules
already permit their exact form.

The current `_words` tokenizer splits `18L` into `18` and `l`. Do not let the
stray `l` become a distinctive name anchor. Extract/remove numeric-unit literals
before deriving anchors, or use a dedicated alphanumeric identity tokenizer.

The existing `validate_copy` anchor check accepts any selected title/tag word,
so `compact` alone can satisfy it. Replace that check with the shared helper.
`evidence_plan.validate_generated` currently applies factuality/novelty/support
without a distinct identity condition; call the same helper for each title
there. Apply identity to titles only. Do not impose it on badges, optional copy
or universal section headings, whose contracts are different.

Suggested errors distinguish `missing_selected_product_identity`,
`ambiguous_public_fact_identity` and `other_product_identity`. Preserve the
actual collision membership in CPU debug/test evidence if useful, while keeping
opaque IDs out of encoder text and generated customer copy.

## Focused synthetic regression cases

Use selected `Pine 18L Daypack`, tags compact/laptop-sleeve. The other products
initially are `River 18L Daypack` with durable/roomy and `Alder 24L Daypack` with
compact/lightweight. All have category backpacks. These are policy fixtures,
not new project labels or training records.

| Generated title | Identity result | Reason |
| --- | --- | --- |
| Meet the compact everyday travel choice | Reject | No selected name or explicit product type/unique signature |
| Compact 18L daypack for travel | Accept | Explicit type + exact capacity + compact uniquely match selected |
| Everyday 18L daypack for travel | Reject | Capacity/type also match River; generic framing does not disambiguate |
| Compact daypack for everyday travel | Reject | Compact/type also match Alder |
| Pine daypack for everyday travel | Accept | Distinctive selected name anchor; whole approved title is not required |
| River daypack for everyday travel | Reject | Different public name anchor |
| River compact 18L daypack for travel | Reject | Foreign name remains invalid despite an otherwise selected signature |
| Compact 24L daypack for travel | Reject | Wrong selected capacity, also rejected by current numeric policy |
| Compact 18 daypack for everyday travel | Reject | A bare numeral is not exact selected capacity identity |
| Pineapple compact daypack for travel | Reject | Pine is not a whole-token match; compact/type still collide |
| Meet your everyday travel companion | Reject | Framing alone carries no selected identity |

Add a fourth product, `Field 18L Daypack`, with compact/organizer. Then
`Compact 18L daypack for travel` must become ambiguous and fail. A title that
also mentions the selected full laptop-sleeve phrase can pass only if no other
product matches the resulting mentioned conjunction. Giving the colliding
product the same complete tag set must make that factual signature fail too.

Additional boundaries: changing IDs, insertion order, candidate order, variant
order, price or stock cannot change identity outcomes; omitted unrelated
private fields cannot matter. A standalone `l` cannot pass as a name. A
single-product generic fixture can accept `Compact pack for your travel` only
when it actually mentions the selected product type and a supported tag, while
`Meet the compact everyday travel choice` must still fail. Run shared-helper
cases through both title-validation paths to prevent policy drift.

An identical approved catalog title still fails the separate novelty check.
Supported preference requirements and forbidden/unsupported lexical claims also
remain mandatory. This policy audit establishes no model-output pass rate.
