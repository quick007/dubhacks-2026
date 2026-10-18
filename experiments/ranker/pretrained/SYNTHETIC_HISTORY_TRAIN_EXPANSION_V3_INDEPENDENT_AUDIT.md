# Public history TRAIN v3: independent semantic audit

All 128 serialized TRAIN requests (64 pairs) were reviewed against the frozen public sixteen-product catalog. **Product relevance labels pass; exact variant preferences need correction.** This is a qualified clearance for a consumer using only `relevanceGrades`, not approval of the entire annotation packet or evidence of model quality.

The review read each actual background and color statement, independently recorded primary and secondary products, and compared the resulting full grade vectors with the separate labels. It checked public history, prices, exact colors, ownership, replacement intent and actual pair changes directly. It did not invoke the author's build/validation functions or read any DEV, private or future evaluation files. No fit, native model, GPU, browser or Git operation occurred.

## Product labels and public grounding

All 2,048 product-grade cells match the independently adjudicated functional priorities: 128 primaries, 144 stated secondaries and 1,776 contextual negatives. All 272 positive bindings satisfy their item budget and explicit hard color requirements. All 638 history events match canonical item name, category, variant/color and ordered dates. Purchases are excluded except in the sixteen explicitly requested replacement counterfactuals. All negative-reason categories have public support; zero functional, ownership, hard-color or price-positive label errors were found. Thirty-eight requests include a positive item exactly at the stated budget.

All sixteen history-only pairs change precisely one event from view to purchase, preserving the other request fields. The priority, replacement and color pairs preserve history. All pair change annotations match actual serialization. The ordinary uses are supported by public names/categories: drinking, handwritten notes, daily/travel carrying, waterproof hooded shell, cotton layers/throw/towel, table lighting, lunch container, over-ear listening, canvas shoes, sunglasses and small-item organizing. No stock, size, capacity, insulation, UV rating or noise-cancellation facts were invented.

The request has only the allowed query, budget, hard requirements, persona background/colors and observed-history fields. Labels and rationales remain separate. Real product/variant IDs in history are public observations, not hidden targets. The inspected preflight formatter removes event/profile/session IDs, product/variant IDs and absolute timestamps. The actual choice prompt source was also inspected: `store_qwen_lexical_choice_inputs_v2.py:72` delegates its system message to `store_qwen_lexical_choice_pilot_v1.py:38`. It renders the full catalog and public observations, drops opaque event/product/variant IDs, and includes public dates. This is distinct from the compact preflight formatter.

## Thirty-seven unsupported singleton preferences

Of 272 positive product bindings, 235 select the first expressed color actually offered for that product. In **37 bindings across 37 requests**, none of the expressed soft colors is offered. The label nevertheless marks the first catalog variant as uniquely `preferred`. Those variants are real and eligible; their asserted preference is unsupported.

Examples from the actual serialized rows:

| Profile | Public preference | Product and labeled preference | Judgment |
| --- | --- | --- | --- |
| `007-base` | “I enjoy Sand most and would also consider Graphite.” | organizer → Ochre | Neither expressed color is offered; arbitrary singleton. |
| `011-base` | “I prefer Navy when it is offered; I do not need a different shade.” | lamp → Ivory | Navy is unavailable; language gives especially little support for a different-color preference. |
| `022-base` | “I keep returning to Olive; Olive band is also a color I like.” | bottle → Graphite | Exact color names must stay distinct; neither is a bottle color. |
| `062-counterfactual` | “My first color choice is Blue stripe. I also like Sand, in that order.” | bottle → Graphite | Other needs remain valid; the bottle's color preference is unspecified. |

Full profile IDs have prefix `functional-v3-profile-`. The JSON's `variantPreferenceCorrections` lists every affected profile/product, public wording, asserted singleton and all eligible alternatives. Nine distinct catalog-default variants are affected. This also means the advertised coverage of all 48 preferred variants is an annotation coverage count, not proof that every preferred binding has public preference support.

Keep frozen v3 unchanged. In a new version, use explicit no-match abstention or clearly recorded indifference among eligible variants. Do not train or score the arbitrary singleton as shopper-preference truth. Preserve all acceptable variants, product grades and public bytes; do not infer a preferred shade from an unrelated observed color without an explicit policy. A product-only loss consuming `relevanceGrades` is unaffected by this variant-label issue.

## Limits to preserve before fitting

Twelve of sixteen color/constraint pairs also lower the budget. Both positive products remain eligible, so their labels are supported, but these are not isolated color-only experiments. Grade 2 ties sensible second and third needs in history-base rows; it does not establish their exact relative ordering. Every request has an eligible primary and secondary, so these examples do not establish empty-set abstention or an over-budget tradeoff.

The actual choice system instruction explicitly permits replacement or repeat purchase when requested, resolving the apparent replacement conflict in the shorter preflight prefix. The packet’s 205–294 token preflight is a compact observation encoding, not the actual full-catalog choice prompt; this audit adds no full-prompt tokenization result. Some one-color soft prose says another shade is unnecessary; when that shade is unavailable, indifference or abstention must be explicit in annotation rather than fabricated as a preference.

This is authored synthetic TRAIN data, with manually judged semantics, not sealed evaluation, observed shopping behavior, unseen-catalog generalization or human conversion evidence.

## Frozen evidence

All sixteen files pinned by the original manifest still match their original hashes. The audit wrote only these two new report files; frozen v3 and its archives were not edited.

- Manifest: `ac15b0242899c75ee5b80f33d057c7e5c32ef096ec4892eaf9a41f7011aca8d7`
- Public: `391c08b9cc80619d0c9d3723289d776f1aa70d02a86dae73d3c370f57fcc46f4`
- Labels: `c09cff4f4580687100ac0f8ea900f54334eefe9c14e411c08b478833295488a4`
- Catalog: `bdd7e2f5f5bc41c147830b5ecd03b550794a237ad633d4e4233688ec223b8450`
- [Complete machine-readable audit](SYNTHETIC_HISTORY_TRAIN_EXPANSION_V3_INDEPENDENT_AUDIT.json), SHA-256 `09752fc7053394867fbf95c7157dcef645bad586b58130b34046d272bc95797d`.
