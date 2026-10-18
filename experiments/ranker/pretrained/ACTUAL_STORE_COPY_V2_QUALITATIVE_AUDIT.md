# Actual-store copy v2: exhaustive qualitative review

The new adapter's strict failures substantially overstate harmful factual copy
failures. I read **all 160 generated fields across all eight new-store arms**,
against their bound facts and the structured payload. I found no direct wrong
product identity, selected color or item price. Most rejected fields express
correct facts in wording outside the pilot's finite grammar. There are still
three unusable generated fields and one plausible but insufficiently sourced
component detail, so this is not a clean end-to-end success.

This is an **offline LLM reviewer judgment**, analogous to a contextual human
read, not independent human review, a new formal validator or conversion evidence.
No output was repaired, benchmark rate changed, threshold relaxed, native model
run or existing file modified. The companion JSON records every raw quote,
bound fact, judgment, strict rejection and exact source JSON pointer.

## Fixed results and separate judgments

| Endpoint / field | Unchanged strict contract | Offline usable with existing bound metadata |
|---|---:|---:|
| Complete first-three text payloads | 2 / 8 | 6 / 8 |
| Complete full-six text payloads | 1 / 8 | 6 / 8 |
| All generated fields | 132 / 160 | 156 / 160 |
| Headings | 16 / 16 | 16 / 16 |
| Titles | 43 / 48 | 47 / 48 |
| Badges | 48 / 48 | 48 / 48 |
| Bodies | 25 / 48 | 45 / 48 |

The frozen and failed-daypack controls retain their recorded strict 0/8
first-three and 0/8 full-six results. Their texts were not included in this
qualitative review, so there is no qualitative semantic-quality comparison
against those controls.

The 160 reviewed fields divide into 132 usable fields inside the grammar,
24 usable fields outside the grammar, three unusable text fields and one
unsupported-specific-detail field set aside for verification. None of the last
four is called a proven false product claim. All 160 native rows ended at stop,
so the incomplete title below is early semantic stopping, not a sampled-cap
truncation. All 48 structured card bindings agree with the expected product,
variant, image action, original name, selected color and item price.

“Operationally usable” here means every required generated field is usable
as shown alongside the original bound metadata, without repair. It is a
text-payload judgment. It does not establish a live storefront, browser paint,
correct ranking, learned image/color selection, preference sensitivity or
constraint compliance. These are locked balanced copy fixtures with empty
preferences, null budgets and no hard requirements.

## Substantive problems

| Source case / field | Exact output | Judgment |
|---|---|---|
| eval-05 T2 | `In` | Unusable title: incomplete and missing product/color. The original-name metadata does not make the displayed generated title usable. |
| eval-00 C3; eval-05 C5 | `Everyday round sunglasses with selected Black selected` | Unusable body text under this conservative review: redundant, unfinished construction. Black and product identity are correct; missing price alone is not the reason for exclusion. |
| eval-05 C1 | `In Navy, Everyday over-ear headphones with cushioned earcups is listed at $89.00.` | Name, Navy and $89 are correct. The bound category says `Cushioned over-ear headphones`, but does not explicitly locate cushioning on the earcups. This is a plausible component inference, conservatively set aside for verification, not a demonstrated falsehood. |

These fields exclude eval-00 and eval-05 from the offline usable first-three
and full-six counts. The other six cases contain usable generated text in all
required fields. The review deliberately does not repair the incomplete title,
delete the repeated word, or infer the precise cushioning detail.

No wrong color, wrong dollar amount, stock/size/shipping assertion or unsupported
performance benefit was found in the eight new-store arms. That observation
applies to these inspected outputs; it is not a guarantee about future decoding.

## Useful wording rejected by the pilot grammar

The original gate is deliberately narrower than ordinary factual English.
Examples below retain the correct item identity, selected color and item price:

| Source case / field | Exact output | Why the user-facing distinction matters |
|---|---|---|
| eval-00 C2 | `Steel lunch box with selected Rust band has an item price of $32.00.` | Correct facts; `with selected` is not one of the accepted pilot relation forms. |
| eval-01 C2 | `This Woven everyday throw is selected in Rust for your collection at $58.00.` | Correct facts; ordinary collection framing adds no product capability or hidden preference claim. |
| eval-01 C6; eval-04 C4; eval-07 C2 | `Everyday canvas sneakers shown in Charcoal at $96.00.` | Readable product-caption style; missing auxiliary `is` is a grammar/style difference, not a wrong fact. |
| eval-03 C4 | `Your selected Weekender duffel is shown in Charcoal at $84.00.` | Correct facts with harmless `Your selected` framing. |
| eval-05 C4 | `This Steel lunch box with bamboo lid is selected in Rust band for $32.00.` | `Steel box with bamboo lid` is the bound declared category. Bamboo lid is source-supported, although the pilot excludes all category assertions. This is not an invented material claim; physical verification remains outside the dataset. |
| eval-07 T6 | `In Sage, this Zip essentials pouch is listed at $26.00.` | Correct priced title sentence; punctuation and price in a title violate the pilot style, not factuality. |

Word order and style are not entirely cost-free. For example, eval-05 T4 says
`Steel lunch box with selected Rust band selected`. It is repetitive and needs
polish, but still communicates the correct selected product/color. It is judged
usable as a terse title. The sunglasses body above is excluded as an unfinished,
repetitious body construction. That boundary is a disclosed qualitative judgment,
not a formal property or new runtime acceptance rule.

## Price omission and identity context

Four bodies omit price. Two are the unusable sunglasses constructions above.
The other two are:

- eval-01 C5: `Market canvas tote shown in Navy`
- eval-07 C1: `Market canvas totes with Navy selected`

Both are readable color/product captions when shown next to the existing exact
original name, $42 item price, Navy metadata and bound image. The second uses
ordinary generic plural inflection; it does not identify another SKU. They
fail the strict independent-name/color/price sentence contract, but price
omission alone is not a harmful or user-essential omission in this payload.
The first duplicates its generated title, so it adds little editorial value.
Neither is counted as an independently complete factual price sentence.

Most accepted bodies also largely restate name/color/price. All 48 badges are
legitimate selected-color phrases. The 16 headings contain only five distinct
generic phrases, with repeated section headings. These are modest improvements
in source-grounded adapted copy; they do not demonstrate rich personal tone,
new merchandising insight or inference from sparse observed profiles.

## Sources, timing and audit boundary

Primary source:
`data/ranker/pretrained/actual-store-copy-fit-v2-attempt-1/results.json`.
The exhaustive JSON preserves its SHA-256 and each original `/arms/N/rawRows/M`
pointer, and binds `store_copy_v2/cases.eval.jsonl` plus the frozen manifest.
New-store arms are at indices 2, 5, 8, 11, 14, 17, 20 and 23. For example,
the incomplete title is `/arms/17/rawRows/4`, and the correct lunch-box phrase
rejected in eval-00 is `/arms/2/rawRows/6`.

Recorded attempt medians are 5,668.83 ms to serialized first-three and 6,835.72 ms
to serialized full-six copy payloads. Those are the existing measured attempt
times, not new accepted-result latency samples under the offline judgment.
They exclude ranking and browser rendering. The original strict-valid latency
arrays and all quality denominators remain unchanged.

The practical next quality questions are whether training continuation removes
the early `In` stop and redundant fragments, whether more precise category
claims stay faithful, and whether actual source-fact swaps change native output.
Simply fitting the exact finite grammar more closely can raise strict scores
without an equivalent gain in useful copy. Keep that distinction visible while
continuing the fixed, predeclared training schedule and preserving all failures.
