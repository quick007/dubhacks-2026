# Pointer v3: coverage and copy-decision diagnostic

The frozen role/position pointer remains **102/126 useful heldout card fields**, with 24 failures and no promotion. All 24 failures involve the three heldout *Slate blue / Steel blue* bindings. This post hoc investigation reads saved predictions and labels only; it changes no model, source, training packet or holdout. These 14 combinations are now development evidence, so another experiment on them cannot claim fresh generalization.

[Machine-readable evidence](STORE_PARALLEL_TEXT_POINTER_V3_COVERAGE_DIAGNOSTIC.json) includes all 24 failed strings, per-slot attention/gates, source facts, TRAIN support counts and SHA-256 hashes of seven unchanged input artifacts. No training, evaluation rerun, GPU/browser work or Git operation was performed.

## Distribution and missing combinations

| Name words | Color words | TRAIN bindings | Heldout bindings |
| --- | --- | ---: | ---: |
| 2 | 1 | 6 | 2 |
| 2 | 2 | **0** | 1 |
| 3 | 1 | 21 | 8 |
| 3 | 2 | 7 | 3 |

TRAIN has 27 single-word colors and seven two-word colors. Its seven multiword bindings cover Ocean blue, Navy band, Rust band, Slate blue, Sage stripe, Blue stripe and Clay stripe. Only **two bindings** supervise lowercase `blue` as the second color word: throw/Ocean blue and crewneck/Slate blue. Every TRAIN binding has all three styles and all three card kinds; style coverage itself is complete. Source atoms total 4–6 per binding, below the eight-atom cap, and all source words are in the frozen vocabulary.

All 90 heldout fields for ten single-word-color bindings pass. The four two-word-color bindings produce 12/36 passes:

| Heldout binding | TRAIN colors for that product | Passed fields |
| --- | --- | ---: |
| Everyday daypack / Slate blue / $68 | Sand, Forest | 0/9 |
| Everyday steel bottle / Steel blue / $36 | Graphite, Cream | 0/9 |
| Portable table light / Slate blue / $64 | Ivory, Terracotta | 3/9 |
| Steel lunch box / Olive band / $32 | Navy band, Rust band | 9/9 |

The daypack additionally occupies the absent two-word-name/two-word-color shape. All products and all 16 price literals already appear in TRAIN. All three failing products have only one-word colors in TRAIN; the successful lunch box has two other two-word colors. This is evidence of confounded length/product support, not proof of a particular cause.

## What the 24 failed slots do

At the reference position for the second color word, **20/24 failures already place over 99% attention on the correct `blue` source atom**. Their copy gate is below 0.5, and the selected vocabulary word is a connective, punctuation or EOS. Examples are `Slate selected`, `Shown in Steel at $36.00.` and a title ending after `Steel`. The other four failures attend more strongly to the first color word than to `blue`; they occur in the daypack plain/compact titles and daypack/bottle browse titles.

Thus the stored evidence identifies both a copying decision problem and a smaller ordering problem. Attention is descriptive evidence; this is not a causal ablation. Truncated `Slate` or `Steel` is incomplete or ambiguous rather than automatically a false color. Reordered or repeated words and malformed price relations still reduce utility.

Only **two** failed reference slots lack global TRAIN positives for `(kind, style, output slot, source role, within-role position)`: the daypack's second-color word in plain/compact titles. Each of the other 22 has seven positive rows for that role/position/slot. Global zero-support positions alone cannot explain the 24 failures.

All nine failed body fields lack the **same price literal** at the longer-color shifted slot: `$68/$36/$64` at slot 5 in plain, 7 in browse and 4 in compact, using zero-based indices. Each position does have seven other-price TRAIN examples. The price is available as a source atom; this is another product/length correlation, not evidence that copying it is impossible. Titles/badges/bodies fail 7/8/9 fields respectively; 84 passing generic section rows remain outside the 126-card-field denominator.

## Proposed next experiment: balanced source-copy supervision

Keep the original 34/14 split and its bytes unchanged. Add a separately named synthetic auxiliary corpus built only from TRAIN word/phrase/price pools. Give it disjoint fixture names, balance two/three-word names against one/two-word colors, cross TRAIN-seen prices independently of those lengths, and cover all three styles. Reject collisions with heldout merchant fact tuples. Clearly label these as synthetic facts, not additional merchant variants. Offline source-factual teachers may use varied clean word orders; no teacher wording or formatting rule enters inference.

Synthetic facts have no matching cached Qwen vector. Reusing a merchant vector would violate the feature contract. Either use an explicitly declared zero-Qwen auxiliary branch with Qwen-feature dropout on original TRAIN rows, or request new frozen features in a later root-owned native window. Any new mode and its mask must be explicit and audited.

Two bounded arms would distinguish data from loss design: (A) the same pointer with balanced auxiliary rows and feature mixing; (B) identical data, steps and parameter budget plus TRAIN-derived source-alignment/copy-gate supervision. Duplicate source words should marginalize valid source positions. EOS remains a learned vocabulary decision, and runtime generation retains no hard field formatter, template or repair. Predeclare seeds/steps and freeze both heads before decoding evaluation labels.

Before fitting, audit role/position/output-slot support, exact price shifts, length balance and source/teacher separation. Report both arms with fixed 126-field denominators, kinds, whole bindings, exact color/price and raw-language failures. Reserve new independent combinations for any later generalization claim. No augmentation or model change is implemented by this diagnostic.
