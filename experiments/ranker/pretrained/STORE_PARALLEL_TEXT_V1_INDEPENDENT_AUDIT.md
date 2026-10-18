# Parallel-text v1: independent data audit

All **48 public product/color records match the catalog**, and all **612 offline teachers** state supported names, selected colors and prices. There are no contradictory targets for the same source/kind/style input. The packet provides a reasonable first bounded native experiment. It also has a concrete positional supervision gap that must be considered before blaming model capacity.

This audit used CPU reads and lexical/source assertions. It did not extract native features, fit a model, change frozen data or assess a new native outcome. The [JSON evidence](STORE_PARALLEL_TEXT_V1_INDEPENDENT_AUDIT.json) records source hashes, counts, every affected evaluation field and useful-versus-gate counterexamples. Audited packet manifest SHA-256: `6a4197799ff625bc500e3e7fc524e583c61b3ae561f65fd0eeca886a1e234235`.

## Global word coverage misses positional coverage

The [data split](store_parallel_text_data_v1.py) ensures every evaluation word has positive TRAIN supervision somewhere. The [head](store_parallel_text_head_v1.py) predicts each slot through a separate set of output parameters. **20 evaluation `(slot, word)` pairs have zero positive TRAIN examples**. Conditioning on field kind and style yields **35 unseen `(kind, style, slot, word)` pairs**. This affects **22/210 evaluation fields: 15 titles, six bodies and one badge**.

| Heldout source binding | Unsupported positive price positions in TRAIN |
| --- | --- |
| Daypack / Slate blue | `$68.00` at plain-copy slot5 and browse-copy slot7 |
| Steel bottle / Steel blue | `$36.00` at plain-copy slot5 and browse-copy slot7 |
| Table light / Slate blue | `$64.00` at plain-copy slot5 and browse-copy slot7 |

Slots are zero-based. Multiword heldout colors shift the price and sometimes product-name words into positions where that word's output weights have received only negative training gradients. This is an output-position compositionality problem, despite a complete global vocabulary. It does not make failure inevitable, but a failure here is not a clean capacity test. The JSON lists all 20 pairs and 22 affected fields.

## Balance, scope and source semantics

| Item | TRAIN | Evaluation |
| --- | --- | --- |
| Distinct product/color bindings | 34 | 14 |
| Title / badge / body rows, each | 102 | 42 |
| Section rows | 96 | 84 |
| Total label rows | 402 | 210 |
| Rows per explicit style | 134 | 70 |
| EOS share of supervised token loss terms | 19.5% | 20.3% |

The maximum teacher is nine lexical/punctuation tokens plus EOS, within 14 slots. No teacher approaches truncation. Slots10–13 receive no target supervision because all targets have already ended; decoding should continue to rely on the first learned EOS, not arbitrary tail predictions. There is no extreme EOS imbalance apparent from these counts.

Sections use only six generic phrases, each repeated16 times in TRAIN and14 in evaluation. Thus headings comprise **40% of evaluation label rows**, although they are only2/20 fields in a six-card view. Report field-kind counts and complete-view quality separately; an easy heading score can conceal card failures. The 42 evaluation views are correlated circular recombinations of 14 heldout bindings, not 42 independent shoppers or unseen products. All16 product identities are TRAIN-seen; sunglasses and towel retain every color in TRAIN.

Each of the16 dollar-price tokens has six or nine positive TRAIN body examples and remains tied to the same known product identity. This packet does not establish novel-price decoding or a response to a changed public price. The engineered fixture sums name/color word vectors and omits price as an independent factor; its price predictions can follow product identity. Neither fixture quality nor the declared 4096-wide Qwen representation proves native identity/color/price separability. Inspect native within-product color changes and source-fact readout before interpreting a weak fit as insufficient capacity.

Three style settings have coherent teachers across fields; no source/style input demands conflicting text. Full public facts contain category slogans, but teachers do not convert them into size or performance claims. Input text contains only name, category, item price and selected color. Feature hashes identify public facts; teacher strings, split labels and private personas do not enter that text. Public kind/style/section controls are explicit. Serving rows should retain the existing strict input projection rather than pass outer training row IDs or targets.

The separate32-query frontier covers all16 known primary product functions, but is curated development with three exact v3 TRAIN wording overlaps. It can test integrated breadth and honest six-card completion; it adds no sealed, budget, negative preference or multi-need evidence. See the [frontier packet audit](STORE_SPEED_QUALITY_FRONTIER_V1_AUDIT.md).

## Usefulness is broader than the teacher grammar

The user accepts fresh concise captions with canonical names as subtitles. Under that context, “Graphite at $36.00.” is factual and useful, yet v1 rejects it for lacking one of three prescribed relations. “Graphite steel bottle” is a reasonable adapted title but fails v1's full canonical-name requirement. Conversely, “Graphite” is an accepted factual badge despite differing from every badge teacher. Exact teacher match is therefore a separate mechanical metric.

The gate also accepts “Shown in Graphite at $36.00. at at at.” and “Graphite selected selected selected.” These preserve facts but are repetitive and unsuitable. Preserve the frozen gate results and independently read both accepted and rejected raw predictions. Repetition and incoherence are usability defects; alternative safe phrasing is not a factual failure. This audit found no false teacher facts.

## Bounded next step after preserving v1

Preserve the first v1 native fit and its fixed evaluation. For a separate v2, either:

- Apply a predeclared small repertoire of factual prefixes/reorderings to **every TRAIN binding**, varying word positions without adding unsupported facts. Derive augmentation from TRAIN/public facts alone, not selected evaluation teacher words or failed heldout outputs. Freeze the new packet before feature/fit work and maintain the heldout binding boundary.
- Share a lexical classifier across positions, conditioned on position embeddings plus source/kind/style. All slots can still execute in one feedforward batch, while a word receives transferable supervision instead of a distinct negative-only classifier at an unseen position.

Neither change guarantees factual generation. Keep source-factual/useful quality, exact teacher match, EOS mechanics and old grammar acceptance separate. Head-only speed excludes fresh query ranking, verification, serialization and browser readiness; no low-hundreds end-to-end result is established by this audit.
