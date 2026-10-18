# Independent jacket boundary and planning implementation audit

The current `jacket_inputs.py` and `jacket_plan.py` implement the declared CPU-only public-input, hard-guard and variable-card selection contracts. No unresolved blocking issue was found in those contracts after the evidence-alias correction described below. This is an input/planning review, not a learned ranking, generated-copy, image-selection, native-cache or end-to-end performance result.

The six-jacket snapshot is a historical fixture alignment set. The newly merged actual storefront has sixteen heterogeneous products; its separately versioned import review appears below. Neither contract is silently substituted for the other.

The reviewer edited this new report only. No implementation, legacy validator, UI/API/harness, frozen import, GPU worker or external communication was changed or launched. No mutation was denied or retried.

## Reviewed source receipts

| File | SHA-256 |
| --- | --- |
| `jacket_inputs.py` | `f71d6964dc41868bae86d47d3ed9d1b918de16ac851f77cd097701907c09260b` |
| `test_jacket_inputs.py` | `1e3ec9f3f43268f335c03b42a25df5b102f6290f7c86af41ceacb67266f90dd3` |
| `jacket_plan.py` | `58dade8ccd689972cd8fff901f90538922f1e30b817859ae9e8c88d63358aef7` |
| `test_jacket_plan.py` | `42cb98333fed9b3398f51e4ee1016a4c921afd4fe0698bea668b8adde6bbaf81` |

All 21 tests pass with the local Python environment, using no model/runtime imports. The reviewer additionally ran eight independent malformed-plan probes and a sibling-evidence isolation probe. Source hashes above identify the code reviewed; later implementation changes need review against their new hashes.

## Boundary and factual eligibility

- Products retain exact `name`, finite nonnegative item `price` and a typed string-to-array `attributes` map. All size/color/waterproof values remain available. `size: [S,L]` fails `M`; missing fields, lowercase `l` versus `L`, `waterproof: no`, and `water-resistant` versus `waterproof` do not satisfy an exact hard `waterproof: yes` or size requirement. There is no synonym, case or free-text coercion.
- Hard requirements are conjoined independently of scores. Preferences are observable model inputs and are checked in receipts, but a preference miss does not remove a product. An explicit budget is an inclusive item-price ceiling. Query wording alone is not parsed into extra hard guards; a query saying “under $0” without typed budget does not create a budget filter. This is consistent with the prospectively declared typed contract, not a general natural-language constraint resolver.
- Boolean, negative, nonfinite, enormous-overflowing and malformed money values fail closed. Attribute scalars, tuple values, boolean values, empty strings and duplicate values are rejected. Empty attribute arrays represent no listed value; they do not invent availability.
- Input objects reject extra product/task/requirement keys, including IDs, persona fields, hidden outcomes, split labels, images, shipping and stock. They are not silently projected into model text. Catalog text takes no task or external map, so it is task-independent by API construction. No equivalence with the earlier joint user/catalog features is claimed. Attribute-map key order is canonicalized; array order is preserved exactly and is not claimed to be an invariant.
- External product/persona/requirement IDs remain in a separately validated positional map. Changing valid external IDs leaves catalog/task encoder text unchanged. Generic map validation proves shape, cardinality and per-namespace uniqueness; only the independently pinned snapshot loader proves correspondence to the teammate source.
- The snapshot loader pins canonical digests for the input, external map, guard receipt and import receipt independently in source. Coordinated changes to input and self-reported receipt are rejected. Duplicate JSON keys, nonfinite JSON constants, missing artifacts, map reorder, false source commit and altered guard counts are covered. Formatting-only JSON changes preserve canonical content and remain allowed.

The actual ten alignment tasks reproduce eligible counts `[1,2,2,1,2,2,1,1,2,1]`. No source task has three eligible products. These are inspected fixture tasks, including their inherited “held-out” names; they are not sealed generalization evidence.

## Selection, shortfall and provenance

Scores must be a complete finite numeric list in original product-source order. Boolean scores and mismatched/partial vectors are rejected. The eligible mask is applied before score ordering; a huge score on ineligible Ridgeline cannot overcome the hiker task's item budget. Ties use source index ascending. Supplied scores are explicitly `caller_supplied_scores`, with `learnedProvenanceEstablished: false`; their presence does not certify a model produced them.

The plan carries requested, eligible, returned and shortfall counts. It returns actual zero, one or two cards without filler or duplicates. Zero eligible products produce no sections, no cards and no copy jobs; requested four gives shortfall four. Empty catalog produces a full shortfall. Requesting one card from two eligible products gives one returned card and no shortfall while retaining eligible count two. The public request count itself is prospectively bounded to 1–64; zero is an actual result count, not an accepted request count.

Each card binds a unique selected source index to exact public name, price and attributes, with `imageAction: null`. Image selection stays unsupported, stock/shipping unknown, budget item-price-only and generated copy unexecuted. The single-section grouping is explicitly deterministic and `learned: false`. No image default, variant, stock guarantee, shipping price or shipping promise is fabricated.

`validate_plan` rebuilds the entire expected plan from public facts and supplied score vector, then compares canonical serialized values. It rejects unknown keys, type changes and inflated provenance, rather than trusting returned guard flags. Independent probes rejected all of:

1. Negative, boolean and out-of-range ranked product indices.
2. `selection.source: learned_model` without a model receipt.
3. Forged `eligible: true` for a hard-failing product.
4. Claimed `stock: in_stock` without evidence.
5. Missing shortfall.
6. A same-length selection containing the same card twice.

The checked-in tests additionally reject corrupted card indices (`true` and `99`), changed prices, appended duplicate cards, false learned-selection/grouping flags, false shortfall and an invented image action. Invalid serialized plan member types and score overflow are included in the root's hardened tests.

## Independent copy jobs and resolved alias finding

For a nonempty plan, jobs are one heading plus independent title/badge/body jobs for each returned product. Each title/badge/body receives exactly its bound product facts and the public task, with no generated sibling dependency. The heading receives all selected products and an exact intersection of their typed attribute values. Hiker's two products have universal `size: [L]` and `waterproof: [yes]`; their different colors are excluded. A shared size option is not asserted to be stock.

An initial implementation shared returned jobs' task, support and capability objects. Mutating one job's evidence could alter sibling evidence while its already-rendered prompt and digest stayed unchanged. This was reported to the root; the root now deep-copies the entire evidence object in each `add()` call. The reviewer verified that mutating one job's task query, capability stock value, nested support attributes and product attributes leaves its sibling byte-for-byte unchanged, keeps the sibling digest valid and leaves original inputs untouched. The corrected code is represented by the hashes above.

The raw task and facts are enclosed as public data with instructions against unsupported stock/shipping/material/warmth/image completion. This is a prompt boundary, not an enforcement result. No jacket-specific raw output validator, EOS receipts, generated text, copy acceptance, learned product ranking or native speed is established by these jobs. In particular “useful badge” and any policy for exact NONE still need a separately declared jacket output predicate before model evaluation. The previous backpack validators and pass rates cannot certify jacket attributes, negation, variable-card output or these new caps.

## Separate generalization v2 pre-native review

The subsequently completed `catalog_fusion/generalization_v2.py` is a synthetic seven-attribute backpack role-decoding experiment, separate from both the six-jacket alignment fixtures and the current storefront catalog. No unresolved pre-native blocking issue was found in its final reviewed source. Its seven CPU tests pass independently; these use explicit synthetic feature stubs and one synthetic gradient/reload check, not a genuine native extraction or project fit. The reviewer did not open the frozen private ranking-label files or execute a native/model run.

| Reviewed v2 artifact | SHA-256 |
| --- | --- |
| `generalization_v2.py` | `a4802adbac9eaff3dddc5ef2ec1859c981392f881b1516f90141f7f192dc604c` |
| `test_generalization_v2.py` | `e6baec63738749f879144c6904bfa5f8fa289e19ab2e119892213eb77448d658` |
| `generalization_v2/FREEZE.json` | `0795ac1495802010422e9358793d904f49f9e4f9311fdab3158175d6c4508140` |
| `generalization_v2/prepare/packet.json` | `fc09197c54017c2b950f04939c7225a4ea6f766153ca852f1afff525e0dfc332` |

Independent public-data verification reproduces 33 distinct catalog products and 63 requests, encoded under full/focused representations for exactly 192 proposed B1 forwards. All fixed token records have shape `[1, tokens]`, exact token-array length and no truncation: full text 60–75 tokens, focused 15–25. This verifies CPU preparation, not actual native completion. Reviewed checksum arguments, pinned local model/tokenizer checks, fixed source roster, bounded native time/memory, explicit root ownership and nonblocking owner lock precede the proposed native action. The reviewer does not authorize or launch that action.

Every split has seven requests at each positive-preference cardinality 1/2/3; each of the seven atoms appears respectively 1/2/3 times within those groups. TRAIN has seven catalog products at each attribute cardinality and six positives per atom. Validation/test have two products at each cardinality and cover all atoms, but their atom frequencies are uneven: validation `[4,1,1,3,1,1,1]`, test `[3,2,1,1,3,1,1]` in the declared atom order. This is coverage rather than balanced held-out catalogs. These cardinalities count requested attribute atoms, not returned cards; all base candidates pass deterministic guards. There is no zero/one/two-card shortfall, negative-preference, complete-grid or jacket-size evidence in v2.

Pair and triple signatures are disjoint across all three splits. Singleton semantics deliberately repeat: focused query text has seven identical singleton inputs shared between each split pair, and focused catalog text has two shared singleton inputs for TRAIN/validation and TRAIN/test, zero for validation/test. Full catalog and request encoder-text overlap is zero, with globally unique names and separate wording families. Thus the data can support a prospectively frozen combination/nuisance experiment, but not a claim that all held-out semantics or all feature inputs are unseen. Known public atom strings are intentionally the factual supervision target; private ranking IDs/outcomes are not encoder inputs. The tie-aware overlap teacher is a synthetic objective, not shopper intent, production ranking utility or conversion ground truth.

The review identified three issues that the owning agent resolved before this freeze: explicit split-coded product-name prefixes were replaced by one globally shuffled name pool; native elapsed time is captured before CPU fitting/evaluation and reported separately; and raw-feature sealing now hashes artifact bytes without deserializing validation/test numeric vectors into the fitting phase. Native extraction necessarily validates each vector when saving it, which is distinct from training access. `train_inputs` loads only the 21 TRAIN request plus 21 TRAIN catalog vectors per representation and computes means/targets from those rows. All heads are frozen and serialized before private ranking-label loading; training statistics are rechecked unchanged afterward. Frozen raw feature hashes, checkpoint hashes, metadata, exact prediction reload, query-independent prior, permuted-query and public-oracle controls remain in the reviewed code.

The full/focused comparison removes public nuisance wording while sharing its objective/schedule. It also repairs coverage relative to older runs, so changes from those older runs cannot isolate a representation effect alone. No image, copy, learned grouping, 12/24-card grid, cold/warm complete request latency, buyer behavior or production generalization follows from this pre-native review. The root remains the sole native owner. Current-store importing is a separate reviewed contract below.

## Actual sixteen-product storefront import

The permanent `store_catalog_v1` snapshot and final importer pass independent read-only verification. All fourteen importer CPU tests pass in 4.167 seconds, using temporary write-once snapshots and mocked corruption; they do not mutate storefront assets or run a model. The reviewer then loaded the permanent snapshot without executing Node/TS again, rechecked its pinned source/projection/receipt and all 81 current image-file bytes/Git blobs, and exercised all 195 ID-free predictive gallery encodings. The four permanent snapshot file byte hashes were unchanged afterward.

| Artifact or canonical content | SHA-256 |
| --- | --- |
| `store_catalog_v1.py` source | `7dca3e1bcedc40c21f0ac877a6635cf7715206b92039a8ba4702fce71c282a8b` |
| `test_store_catalog_v1.py` source | `505bab7236c2140e0bfeedbe7605f756ddccf67d917f2ce144aa7f575cc326e4` |
| `IMPORT_RECEIPT.json` file bytes | `9e87c057b71b2e4b6ca6200484a310631fe30de95bcd2f0549943e325567f7da` |
| Canonical runtime export | `fa440e919e1f36890a1c97d8e05ea01bd4ce46811eaed57c0dc61db7b049a0d8` |
| Canonical model inputs | `205572ed535b6f40cd56ede51208c348ad05998ad4d205f193a8b5bfe3e79fb7` |
| Canonical external mapping | `4c5caa231e70945ac65ae0804db7c9ea37a1c05387262909f6a930f660224fee` |
| Canonical asset receipts | `a4e7f5289b362d565cffd3d41e6e8a6d7af35135334c41e13178fc70b6519b90` |

The import is pinned to teammate commit `e172f6057aa76c58089f5e9b86b214a1543052a6`. Its independently captured runtime export uses installed Node+tsx only for existing pure configuration, variant, image and shipping modules; the importer does not invent a TypeScript parser or install dependencies. Model input contains exact names, item prices, category/discovery text, 48 declared colors and 195 gallery metadata records for sixteen products. Separate routing retains 65 base image records, all 195 per-color gallery bindings and 32 optional color previews across 81 unique actual asset files. Optional previews are retained in external mapping, not counted as additional predictive gallery candidates.

The image identity is the selected product/color/gallery action plus exact asset path, position and optional tile geometry. One path alone is insufficient: the mug tabletop colors share the same source file but use tile indices 0/1/2. The importer preserves the default color's exact original gallery and binds each alternate color to the runtime `getVariantImages` gallery, including reused tabletop assets and non-tabletop color sheets. Tests reject wrong owners, invalid crop bounds, raw crop/source tampering and coordinated forged projection receipts. A frozen raw pin is the authority for exact source bindings; generic shape checks alone do not authenticate a fabricated gallery.

Predictive text excludes product/variant/image/persona/requirement IDs, asset routes, crop geometry, swatches, CSS kind/color markers and synthetic scenario IDs. Changing routing identities leaves projected model inputs identical. Declared image labels, alt text, context, presentation and descriptive tags remain explicitly marked **fixture-declared; pixels unexamined**. Source category and search terms are marketing/discovery declarations; they are not automatically verified functional capabilities. An image of a commuter, family or campus does not prove the shopper's identity or intent, and its setting does not establish material, warmth, durability, stock, size availability or any other unlisted property. This import performs no pixel interpretation, learned ranking, image-choice quality evaluation or factual body-copy generation.

Shipping remains a cart-subtotal rule. With `showShipping: true`, subtotal exactly 50 produces a $5 quote and 50.01 produces a zero-cost quote; the threshold is strictly greater than 50. With visibility false, the runtime returns no quote and its displayed total equals subtotal. That omission is preserved as hidden/unknown actual delivery-cost evidence, not converted into a per-item free-shipping promise. Source constants and visible/hidden boundary probes are frozen separately from public model text; no product-specific delivery, tax or stock guarantee is inferred.

The ten API fixture tasks are allowlisted to public query, optional budget and typed requirements. External persona/requirement IDs remain separate. Their compatibility with this sixteen-product store is expressly unestablished: the catalog does not contain the historical jackets' structured size lists. No old six-jacket eligibility count or v2 backpack accuracy is reported as current-store quality. Illustrative storefront shopper biographies/goals and candidate lists are excluded from predictive input, not treated as observed behavior.

The review initially found that snapshot loading trusted receipt asset/count/provenance claims without rechecking current image bytes. The owning agent resolved this before permanent freeze by independently pinning raw/model/mapping/asset-receipt digests, reproducing the exact projection, checking importer/export-program and pinned source hashes, requiring exact counts and zero model calls/false training, and revalidating every current asset against its pinned Git blob/hash. Fourteen tests include forged matching output hashes, inflated model/training/count claims, duplicate JSON/nonfinite data, route/crop corruption and mocked asset-byte changes without touching real UI files. No unresolved blocking issue remains in the reviewed import. Its honest capability boundaries and exact routing do not by themselves establish a learned storefront model or production generalization.

## Verified recorded CLI receipt

After execution-service recovery, the reviewer independently read `jacket_contract_runner.py` and `jacket_alignment_results/v1/report.json`. The recorded report SHA-256 is `4b4c5411e2f6056ee5d145efc4bc55dafcd044f10e417bf182318f83021eaa47`; the runner source SHA-256 is `ff4238d0f85ab8d38e0dd25030a20f7ed7743ede70cdb62e9e6f662e6c5c999e`. All three recorded source hashes match current input/plan/runner bytes. The pinned snapshot and external mapping match exactly.

For all ten distinct task indices, the reviewer reconstructed the complete plan and every copy job from the pinned public facts; stored plans, guards, task text, catalog text, job evidence, evidence digests and rendered prompts agree exactly. Recomputed summary is 10/10 contract-valid tasks, 15 returned of 30 requested cards, shortfalls `[2,1,1,2,1,1,2,2,1,2]`, and 55 independent unexecuted copy jobs. The original report hash is unchanged after this read-only verification.

The recorded median selection/validation/job-preparation time is 0.6383335567 ms, with 8.6015829584 ms CPU run wall excluding import and report I/O. This is a recorded deterministic CPU contract-preparation measurement, not inference, generated-copy validation, complete-view latency, a production throughput estimate or a ranking-quality result. The receipt correctly records zero native forwards, optimizer updates and generated fields, and false learned ranking/grouping and complete-page quality/latency claims. The reviewer did not rerun the CLI or rewrite its receipt.

Model fit, learned scoring, full-grid decisions, new image capability and useful generated-copy quality remain subsequent work, rather than completed by this contract implementation. The transport outage was infrastructure failure rather than an approval denial; the pending report write completed once, and no duplicate or alternative mutation was attempted.
