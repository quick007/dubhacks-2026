# Public-fact pointer v3: independent audit

The pointer decoder substantially improves heldout product/color composition while leaving important failures. On the unchanged 34-TRAIN/14-heldout split, the role/position arm has **102/126 clearly factual, useful card fields**; the lexical-key arm has **101/126** under independent short-caption review. These are known-catalog combinations, not unseen products, prices or shoppers. Neither research head was promoted or timed as a fresh-query storefront.

The primary evidence is `data/ranker/pretrained/store-parallel-text-pointer-v3-attempt-2/`. The [JSON audit](STORE_PARALLEL_TEXT_POINTER_V3_AUDIT.json) records all 420 heldout field rows, source facts, raw strings, gate decisions, judgments, hashes and 42 correlated views per arm. All 1,680 view-field projections match the stored slot predictions. Both arms also recover all 402 source-grounded TRAIN teachers. No GPU, fit or live-serving change was performed by this auditor.

| Candidate | Frozen gate: heldout card fields | Clearly useful card fields | Frozen first3 / full6 views |
| --- | --- | --- | --- |
| Original v1 pooled-feature head |0/126 | Original audit:0/126 |0/42;0/42 |
| Strongest prior CPU v2 shared-slot arm |8/126 | Not newly rejudged here |0/42;0/42 |
| v3 typed role/position pointer |102/126 |102/126 |15/42;6/42 |
| v3 typed role/position/word pointer |98/126 |101/126 |6/42;0/42 |

The 84 generic section rows per arm are excluded from the 126-card-field denominator. They all pass, but must not inflate the harder title/badge/body result. The 42 views reuse 14 heldout bindings; they are not 42 independent customers or catalogs. Both fixed 800-step arms remain reported, without choosing one checkpoint from heldout performance.

## Quality gains and remaining errors

The role/position arm passes35/42 titles,34/42 badges and33/42 bodies. All accepted fields are readable and source-correct in this review. Its24 rejected card fields concentrate on the daypack's Slate blue, bottle's Steel blue and lamp's Slate blue. Typical failures are incomplete palette labels (“Shown in Slate at $68.00.”), reordered words (“Everyday daypack blue in Slate”), repetition and lost price slots (“Color: Steel. Item::.”). A truncated *Slate* or *Steel* is not automatically a false fact; it is an ambiguous/incomplete variant expression. These fields are conservatively outside the clearly useful count without repairing them.

The lexical-key arm passes35 titles,37 badges and26 bodies under the frozen screen. One accepted nonteacher badge, “Steel blue,” is a useful two-word factual badge. Three further raw outputs are useful despite rejection:

- “Steel blue at $36.00.” states the exact selected color and item price in a concise caption; the gate demands one of its prescribed relation phrases.
- “Shown in blue at $36.00..” has a truthful coarse color description of the declared Steel blue variant and the correct price. The extra period is cosmetic.
- “Everyday steel bottle, blue” retains identity and a truthful coarse descriptor, with the precise variant retained in canonical metadata.

These give 101/126 clearly useful fields, with 9/42 clearly useful first3 views and 0/42 full6 views under this **post hoc offline judgment**. Frozen gate counts remain 98/126 and 6/42 first3. No result or timestamp was regraded. Other failures are substantive: “Color: $24.00. Item: $24.00.” puts a price in the color role; “Olive band option at Olive.” puts a color word in the price slot; duplicate amounts and repeated words reduce utility. Missing a repeated price alone need not be essential when canonical price metadata is visible, but these malformed relations still fail as useful raw copy.

The improvement is not evidence that v1 merely needed more parameters. V3 adds an explicit public source-word channel; its role and lexical arms have277,516 and 280,940 parameters respectively. This changes the information available to the decoder, alongside the attention architecture. Better source binding is demonstrated for these combinations; the experiment does not isolate a single cause.

## Input and generation boundary

Source review found no runtime teacher lookup, field-format program or output repair. Every card input contains public name words, selected-color words and a canonical dollar-price atom, plus typed roles, within-fact positions and word counts. They accompany the existing verified cached 4,096-dimensional Qwen features and public kind/style/section conditions. Source serialization is not an output target. All 48 public bindings need only 4–6 atoms, below the 8-atom cap, and every atom exists in the original 107-word vocabulary. Unknown source words and stale fact/text/key bindings fail rather than truncate or synthesize replacements.

Both heads predict 14 slots through a shared learned vocabulary/pointer mixture. The model learns source attention and the copy gate; it receives no previous target words. EOS comes from the learned vocabulary, not a predetermined output length. Detokenization only joins whitespace and punctuation. Sections have no product-token source; a mathematical source-presence mask forces their copy mass to zero without a field-specific wording rule. Before fitting, I found a saturated-gate edge case that produced zero probability mass for empty sources. The owner fixed it with this mask, and its regression passes.

I reran all **eight CPU tests**, including teacher construction disabled during generation, target/private-field rejection, stale price/color bindings, source role/count correctness, actual loss gradients, EOS exclusion from source atoms, empty-source normalization and freeze-before-label-read control. The original data, teachers, vocabulary and 34/14 partition remain unchanged. All product identities and price tokens were already TRAIN-seen; source words improve copying within this bounded catalog.

## Provenance correction and controlled repeat

Attempt 1 predeclared both schedules and kept heldout targets out of training inputs, but evaluated arm 1 before fitting arm 2. Its broad `heldoutLabelTextDecodedBeforeFreeze:false` flag could therefore misdescribe the second arm. The original files remain unchanged, with an external `PROVENANCE_CLARIFICATION.json`.

Attempt 2 trains and freezes **both** checkpoints, writes `ALL_HEADS_FROZEN.json`, then decodes any heldout target text. Its checkpoint byte hashes exactly match attempt1: role/position `3890eb69d5a72c0bce3530380a5730e0412b7347b438819783009b5752f44425`; lexical-key `a57cbea7d1da830a2d0758738db58d9e9e651e6daea392bd5454f7e8f615a999`. Thus the stricter control reproduces the quality results without model selection or changed mathematics. No fresh-query, browser readiness or conversion measurement was added; the fully supervised all-catalog demo remains a separate lane.
