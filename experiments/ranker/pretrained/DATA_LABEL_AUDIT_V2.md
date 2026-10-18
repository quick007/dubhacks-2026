# Data and label audit before changing model capacity

The evidence supports correcting and testing the training/evaluation design before attributing failures to 9B capacity. Both frozen datasets' ranking labels reproduce their declared public-fact teachers exactly: 48 old paired requests and 63 generalization-v2 requests. No contradictory or incorrectly computed target was found within those narrow objectives. That does **not** make their coverage, nuisance distribution or real-store relevance sufficient.

This is a read-only CPU data/label/source review. No new fit, model/native call, source-Python change, old-result change, UI change or task generation was performed. The only new artifact from this review is this report. Existing input/planning/import review evidence is in `JACKET_ALIGNMENT_IMPLEMENTATION_AUDIT.md`.

## What changed, and what the observations establish

| Corpus | TRAIN query attribute counts | TRAIN catalog attribute counts | Held-out design |
| --- | --- | --- | --- |
| `paired_v1` | 28 requests, all singleton preferences | 21 used products, all exactly two tags; 42 unique variant embeddings | 8 validation and 12 test requests, all two-attribute preferences; four candidates per request; inherited old catalog partitions |
| `generalization_v2` | 21 requests: seven each with 1/2/3 positive attributes | 21 products: seven each with 1/2/3 tags | 21 requests per split: seven each with 1/2/3 positives; six candidate products per held-out split, two per tag cardinality |

Old training did cover every atom positively. Its weakness was that query-role training never saw compound positives and the entire 84-product source catalog had exactly two tags. The 28 old training queries also contain repeated preference semantics paired with two requested image styles; they are not 28 independent attribute combinations. Catalog rows duplicate each product's semantic target across its two genuine text presentation variants.

V2 changes the learning problem to product-only seven-atom decoding, supplies compound positives in both roles, balances TRAIN atoms, uses tie-aware ranking labels and independently randomized identity/price/capacity assignments, and compares full inputs against a focused public-fact representation. It removes the presentation task rather than proving that image selection is solved. Its equal-role query/catalog BCE avoids weighting frequently repeated candidate appearances as extra catalog supervision.

Important counterevidence prevents a simplistic “missing compounds made the old model fail” story: the prior **semantic-factor** audit on `paired_v1` already recorded 28/28 TRAIN, 8/8 validation and 12/12 test product choices with the same frozen 9B features. That is distinct from the earlier rank-four paired scorer, and those catalogs/tests were already inspected. Singleton TRAIN coverage was an evaluation weakness, not proof that the old public-atom task was unlearnable.

The completed v2 audit records focused validation/test 21/21 each, including 14/14 new pair/triple combinations per split, while full text gives 8/21 and 7/21. The same frozen backbone can therefore represent the narrow structured target sufficiently under the focused input on these splits. This does not prove arbitrary-language personalization, the current-store ontology, fresh copy quality, hidden-profile learning, or that every remaining failure is exclusively a data issue. Comparisons with old runs change dataset, representation and product/variant task together; only v2's full/focused comparison holds its corrected supervision and schedule fixed.

## Positive and zero-label coverage

Cells below are **positive / zero target counts**, not positive/explicit-negative preference examples. Query zero means the atom was not listed as a positive preference. Catalog zero means the atom is absent from the closed seven-tag annotation. Neither means an explicit dislike or an unknown capability has been modeled. Unknown and false are not separate target states in these factor tasks.

Old catalog columns count unique used products. The actual old TRAIN catalog target tensor has 42 variant rows, so its per-atom positive and zero counts are exactly twice the shown 21-product column.

| Atom | Old query TRAIN / 28 | Old catalog TRAIN / 21 | Old query validation / 8 | Old catalog validation / 5 | Old query test / 12 | Old catalog test / 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| compact | 4 / 24 | 7 / 14 | 2 / 6 | 1 / 4 | 4 / 8 | 3 / 7 |
| durable | 4 / 24 | 7 / 14 | 2 / 6 | 1 / 4 | 4 / 8 | 2 / 8 |
| laptop-sleeve | 4 / 24 | 6 / 15 | 2 / 6 | 2 / 3 | 2 / 10 | 2 / 8 |
| lightweight | 4 / 24 | 4 / 17 | 2 / 6 | 1 / 4 | 2 / 10 | 3 / 7 |
| organizer | 4 / 24 | 7 / 14 | 2 / 6 | 1 / 4 | 4 / 8 | 3 / 7 |
| roomy | 4 / 24 | 7 / 14 | 4 / 4 | 3 / 2 | 4 / 8 | 4 / 6 |
| water-resistant | 4 / 24 | 4 / 17 | 2 / 6 | 1 / 4 | 4 / 8 | 3 / 7 |

In v2 every query split and the TRAIN catalog have 21 rows with **6 positive / 15 zero for every atom**. Within query cardinalities 1/2/3, each atom appears respectively 1/2/3 times among seven requests. Held-out catalog atom coverage is positive but uneven:

| Atom | V2 catalog validation / 6 | V2 catalog test / 6 |
| --- | ---: | ---: |
| compact | 4 / 2 | 3 / 3 |
| durable | 1 / 5 | 2 / 4 |
| laptop-sleeve | 1 / 5 | 1 / 5 |
| lightweight | 3 / 3 | 1 / 5 |
| organizer | 1 / 5 | 3 / 3 |
| roomy | 1 / 5 | 1 / 5 |
| water-resistant | 1 / 5 | 1 / 5 |

V2 corrects missing **training cardinalities**, not all semantic coverage. It never trains/evaluates zero preferences, four-or-more preferences, explicit negative preferences, contradictory positives/negatives, uncertainty, priorities/weights, typed jacket sizes or a three-state known-true/known-false/unknown ontology.

## Nuisance assignment is not empirical decorrelation

Old product descriptions explicitly repeat tag facts, capacity, polyester shell and zip closure. Their tag words are redundant public evidence, rather than hidden label leakage. Name/capacity, price, family-level budget and variant wording are additional cues. Old TRAIN query task wording is one fixed sentence; its seven budget values are derived from candidate-pool maximum delivered prices and remain constant within counterfactual families. That supports controlled within-family preference/style interventions, but does not counterbalance arbitrary query prose or budget semantics.

V2 uses globally unique product names from one shuffled pool, no explicit TRAIN/validation/test name prefix, shuffled capacities and prices, and three TRAIN product-description and query-wording families versus distinct held-out wording families. This removes obvious split-coded identity words and tag repetition from descriptions, but one seed and one assignment do not establish nuisance independence in the realized small dataset.

Descriptive Pearson correlations, computed from unique products and shown without any statistical-significance claim:

| V2 split | Products | Item-price vs tag cardinality | Capacity vs tag cardinality |
| --- | ---: | ---: | ---: |
| TRAIN | 21 | -0.116 | 0.029 |
| validation | 6 | -0.359 | -0.359 |
| test | 6 | **0.837** | 0.000 |

Some held-out atom/price correlations are also large simply because one of six products contains an atom: absolute correlations reach 0.655. The old used TRAIN catalog's largest absolute item-price/atom correlation is 0.383 (`organizer`); old validation reaches 0.795 with only five products. These figures are shortcut-risk diagnostics, not proof that a trained scorer used price or that the labels are wrong. All old products have two tags, so price/cardinality correlation is undefined there.

V2 query-wording assignment is deterministic `i % 3`, not crossed factorially with semantics. Its three TRAIN query templates have 9/6/6 rows, with `roomy` positive in **2/9, 0/6 and 4/6**, respectively, versus 6/21 overall. The three TRAIN catalog-description templates each have seven rows, with `roomy` positive in 1/7, 3/7 and 2/7. Thus globally balanced atoms still coexist with template-label correlations. Unique names can also be memorized across 21 training products even though they cannot identify unseen held-out names. Every v2 TRAIN choice fitting perfectly is not a nuisance-control result.

Next data should deliberately cross the same semantic request with several independently assigned wording families and apply controlled name/description/price/order interventions that leave the intended label unchanged. Price interventions must respect the task: once budget or price is an actual preference/guard, changing it can legitimately change the label and is no longer a nuisance-invariance control. Report realized contingency tables/correlations and all predeclared variants; do not choose a favorable random seed after seeing model outcomes.

## Contradiction, unknown and observable-context limits

Both datasets have empty history and empty hard-requirement lists in every evaluated public request. All used candidates are in stock and pass the deterministic guard; old budgets are chosen above pool costs, and v2 uses budget 160 with every base candidate eligible. The old presentation teacher requires both styles to be compatible; there are no incompatible-only, no-image or missing-asset tasks in that paired set. V2 removes images entirely. CPU guard/corruption tests elsewhere do not turn these omissions into tested native-language robustness.

Neither dataset tests “not compact,” “prefer roomy but avoid bulky,” preference conflicts, absent versus explicitly false product attributes, unknown stock/size, stale observations, future-history cutoff, ambiguous exact-name resolution, or recovery from unsupported requests. Their teacher rewards only overlap of explicitly supplied positive atoms. Missing tags cannot safely be transferred into negative labels for the real store: absent structured size/stock means unknown, not known unavailable.

Focused v2 query text contains only sorted `statedPreferences`. It deliberately drops task prose, history, budget, requirements and presentation context from the predictive vector; known guards remain outside the factor scorer. That is valid for the declared all-eligible positive-atom objective, not a replacement for a general public context/profile/history encoder. No hidden persona, latent preference, cross-session profile or causal user-learning claim is validated. Focused product text likewise contains only sorted public tags and does not learn price, description, image or availability reasoning.

Routing record/product IDs, split/family metadata and private ranking targets are excluded from encoder text. Public descriptive facts are inputs by design, and public atom-presence targets are supervised derivatives of those facts; this is semantic decoding, not independent discovery of private user truth. The old import and current-store projection explicitly exclude persona biographies, goals, traits, scripted outcomes, candidate labels and split membership. Public API task query/typed requirements are retained as statements, without importing their persona's hidden scenario goal.

## Leakage, prior inspection and effective evaluation size

Old paired requests and product memberships are disjoint across splits; exact query/catalog encoder texts also have zero cross-split overlap. Nevertheless old catalog partitions and outcomes had already been inspected before the paired exercise. The freeze explicitly discloses old-public preference/style signature occurrences in the new request records: 0 TRAIN, 5 validation, 7 test occurrences. New IDs/families do not make those old catalog tests untouched or sealed.

V2 pair and triple signatures are partitioned across splits and full encoder texts have zero cross-split overlap. Focused singleton semantics/text repeat deliberately: seven identical request texts between every split pair; two identical catalog texts for TRAIN/validation and TRAIN/test, none for validation/test. The completed run's separate audit verifies all 25 cross-split identical-text comparisons have bit-exact hidden vectors; 192 saved artifacts contain 174 unique raw keys. Singleton success is therefore nuisance invariance, not unseen semantic transfer. Pair/triple results are new-composition evidence inside the same seven-atom grammar, not all-combination or current-store generalization.

Training means, frequency/targets and updates are TRAIN-only, with private ranking labels opened after every head freeze; integrity controls do not show a fitting-time label leak. Public v2 inputs and signature design were reviewed before native launch, and outcomes are now inspected. Future tuning on this packet must call it an inspected replay. The 94 validation and 115 test disjoint-best-set pair metrics reuse the same 21 requests and six products; they are correlated comparisons rather than 94/115 independent trials. One frozen synthetic dataset and one seed do not support confidence in an arbitrary new domain.

## Current-store archive authenticity versus active freshness

The permanent sixteen-product snapshot was independently loaded and verified after freeze: 16 products, 48 colors, 65 base image records, 195 bound gallery candidates, 32 optional previews and 81 actual assets. Loader/source/projection/asset checks and 195 ID-free candidate encodings pass; four snapshot byte hashes remain unchanged. Import receipt byte SHA-256 is `9e87c057b71b2e4b6ca6200484a310631fe30de95bcd2f0549943e325567f7da`. Canonical model input hash is `205572ed535b6f40cd56ede51208c348ad05998ad4d205f193a8b5bfe3e79fb7`.

This authenticates the archive at pinned teammate commit `e172f6057aa76c58089f5e9b86b214a1543052a6`. `load_snapshot` compares source receipts to that Git commit with `require_worktree=False`; it can correctly load the old archive after a worktree product-name/price/configuration change if image bytes stay unchanged. It must **not** certify active-serving freshness. Preserve frozen v1 unchanged. A separate versioned model bridge should require current input-producing source bytes to match the selected archive before using its candidates, check again after preparation, and fail stale rather than silently serving old facts. Keep that bridge/representation/source/checkpoint provenance in cache keys. Current-store source freshness and archive authenticity are different checks.

The root subsequently implemented that separate `store_model_bridge.py` without altering the frozen importer. Independent review and all three bridge CPU tests pass. `assert_current_sources` requires the complete pinned source inventory and current byte hashes before and after all candidate text construction; temporary changed/missing-source and partial-inventory controls fail closed. All 195 recorded member/text/action bindings in `store_model_results/v1/report.json` reproduce the archive exactly, including product/variant/image/crop routes. Bridge source SHA-256 is `91124b1793bf9960761547f0feb4c1e8609593bd70f4f7abf3c1b97c40c38eaf`; recorded report hash is `b3208e14a9bf2ff0b9e555256b9d409520144dfba9e228971b7039b6eb9490e1` and remains unchanged. The receipt records zero native/optimizer calls, no selected actions and no learned ranking/image claim. Its 625.0757080 ms measured CPU preparation includes cold archive/source/asset verification and 195 text encodings; it is not a request-inference or warm-cache latency claim. Active freshness is established at this preparation boundary, not guaranteed indefinitely after files change.

Store category/search text and image alt/context/presentation are source declarations. A family/campus/commute image setting is not a user persona or product capability. Stock and sizes remain unknown. Shipping is cart-scoped: visible quote is $5 at subtotal 50 and free only above 50; hidden quote omission/displayed subtotal is not proof of free delivered cost. The ten historical API jacket tasks are not automatically compatible with the heterogeneous store. The seven-atom backpack labels cannot be transplanted into sixteen-product/current-image actions by renaming IDs.

## Next genuinely new, frozen actual-store task set

Prepare a new versioned **data-only** contract before another fit. The following is a proposal, not a generated/frozen dataset or a claim that unseen tasks already exist:

Under the user's later speed-first priority, the larger package below is a future generalization option, not a prerequisite for the next bounded latency experiment. First use a small prospectively fixed useful-quality packet and fresh C supervision; do not keep expanding synthetic evaluation or chase perfect synthetic accuracy before measuring a safe faster producer.

1. Freeze the current source/asset projection and versioned public task ontology. Keep all sixteen product candidates and all bound color/gallery actions; do not shrink the roster to make accuracy or timing easier. Separate known product identity/price/color, declared presentation metadata, and unknown physical/stock/size facts. Labels need evidence references and true/false/unknown distinctions where applicable.
2. Use explicit public requests with no biography, hidden persona, purchase script or outcome fields. Prospectively allocate independent request families and wording families to TRAIN/dev/test, with new combinations for test. If catalog products remain the same sixteen, call this **new-request generalization on a known catalog**. An unseen-product/category claim needs a separately declared product/category training holdout; the entire catalog still remains eligible for the final full-roster ranking test.
3. A practical first bounded package is 24 TRAIN, 8 dev and 16 fixed test requests, balanced by semantic family rather than counting paraphrases as independent users. Include positive retrieval/preference changes, exact-name constraints, exact color/presentation choices, inclusive item-budget boundaries and honest ties/abstentions. Preserve all tied valid products/images instead of constructing arbitrary ID winners. Weight/count paired paraphrases within families explicitly.
4. Within each split, counterbalance multiple literal query wordings, category/identity phrasing and candidate order against the same public semantics. Independently vary irrelevant descriptions/names only in a separately labeled synthetic nuisance arm, rather than falsifying the actual-store source. Hold image declarations and crop routes fixed for a pure wording test. Include budget/price interventions as causal task changes with recomputed labels, not invariance tests.
5. Reserve explicit missing-size/stock, unsupported material/warmth, absent image evidence, negative preference, conflicting requirement and ambiguous-name tasks. Define the permissible abstention/clarification output before inference. Do not label unknown size or unseen stock as false simply to force a winner. Cart-shipping tests require supplied cart subtotal and visibility; a per-product shipping guarantee has no current evidence.
6. Freeze all public inputs, private/evaluator labels, family memberships, evidence rules, source hashes and fixed denominators **before any model output is inspected**. Human-check factual/ambiguous labels independently; label any scripted teacher as synthetic, rather than human validated utility. Keep a fresh test generation seed/wording pool beyond every inspected packet. Once evaluated, retire its “unseen” status for subsequent tuning.
7. Reuse the same frozen 9B model for the first controlled comparison, report TRAIN and unseen-family diagnostics separately, and retain oracle/prior/permuted/no-request controls. Audit labels/coverage/nuisance tables and raw errors before choosing another model size. Any generated titles/badges/body/headings require their own genuine-output factual/protocol validator and complete-payload timing; factor ranking accuracy does not certify copy.

The proposed counts are a bounded starting design, not a sample-size guarantee. Public-fact relevance and presentation matching are proxies; conversion, subjective usefulness and persistent user learning require separate valid evidence. No model-capacity comparison is necessary to establish the data gaps above, and this audit recommends no larger-model migration on the current evidence.

## Reproducible evidence and scope

- Old freeze `1e7238f090c21c83dc2715f5b402dc8b9c84cfe9b30e334d4efdae11fdbbed54`: `paired_contract.validate_data_for_review` recomputed all 48 labels and 48 product/image controls from public data exactly; family controls and cross-split product disjointness pass.
- V2 freeze `0795ac1495802010422e9358793d904f49f9e4f9311fdab3158175d6c4508140`: all 63 private labels were read **after the completed fit/evaluation** for this explicitly requested data audit and independently recomputed from public requests/catalog exactly. This is later analysis, not a pre-freeze label access claim.
- Source `generalization_v2.py` SHA-256 `a4802adbac9eaff3dddc5ef2ec1859c981392f881b1516f90141f7f192dc604c`; packet `fc09197c54017c2b950f04939c7225a4ea6f766153ca852f1afff525e0dfc332`.
- Historical semantic-factor results are preserved in `CATALOG_FUSION_SEMANTIC_ATTEMPT_1_AUDIT.md/.json`; v2 results in `CATALOG_FUSION_GENERALIZATION_V2_ATTEMPT_1_AUDIT.md/.json`. This review reads their recorded metrics and does not rerun or rewrite their model evaluations.

Per-atom counts, cardinalities, template contingencies and descriptive correlations above were calculated directly from frozen **public** JSON, using unique used catalog products rather than repeated request appearances. No training or native feature recomputation occurred.
