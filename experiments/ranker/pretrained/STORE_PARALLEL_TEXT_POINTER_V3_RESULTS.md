# Parallel public-fact pointer research

The learned pointer/vocabulary decoder improved source-correct heldout card
copy while retaining the original 34 TRAIN / 14 heldout product-color split.
The role/position arm passed 102/126 card fields, compared with 0/126 for the
original Qwen-feature-only head and 8/126 for the previous best shared-word
classifier. It remains a research candidate; the live catalog-trained demo is
unchanged.

| Fixed 800-step CPU arm | Titles | Badges | Bodies | Complete first-three views | Complete six-card views |
|---|---:|---:|---:|---:|---:|
| Typed role + within-fact position pointer | 35/42 | 34/42 | 33/42 | 15/42 | 6/42 |
| Same pointer + lexical word key embeddings | 35/42 | 37/42 | 26/42 | 6/42 | 0/42 |

These 42 views are correlated rotations/styles of 14 heldout bindings in a
known 16-product catalog, not independent shoppers or unseen product catalogs.
Generic headings are excluded from the 126-card-field headline. Both arms
also passed all 84 heldout headings and reproduced all 402 TRAIN fields.
Field acceptance uses the existing bounded factual screen; literal checks and
raw predictions remain separate for independent language review.

The role/position arm preserved complete identity in 41/42 titles, selected
color in 35/42 titles, 34/42 badges and 33/42 bodies, and the exact item price in
37/42 bodies. The lexical-key arm preserved identity in 41/42 titles, selected
color in 35/42 titles, 37/42 badges and 30/42 bodies, and exact price in 33/42
bodies. Wrong colors, wrong prices, missing facts and malformed phrases are
retained without repair. Neither arm is promoted.

## Inputs and learned generation

The decoder receives the same cached, frozen Qwen vectors plus the public
name/color/item-price words for its selected product. Those words have typed
fact roles, positions within each fact and word counts derived only from the
public facts. All 48 source lists fit within six atoms, below the enforced
eight-atom cap, and contain no out-of-vocabulary words. There are no private
profiles, teacher output strings, teacher lengths, output format programs or
hidden product answers in inference.

For every output slot, neural layers predict a mixture of a shared vocabulary
distribution and learned attention over the public source words. All 14 word
and EOS slots are predicted in one pass. Source words are copied only when
learned attention and mixture weights select them. There is no field-to-format
branch, complete-string lookup, autoregressive loop, output repair or template
fallback. EOS remains a learned vocabulary choice. Empty sources have zero
effective copy mass, including when the copy gate numerically saturates.

No new labels or augmentation were introduced. Both 800-step configurations,
seed 7211 and width 64 were declared before evaluation. Each fit used one CPU
thread and took about 6.2 seconds. No native forward, GPU job, backbone update,
live service change or end-to-end latency measurement occurred in this lane.

## Preservation and provenance

The final run is
`data/ranker/pretrained/store-parallel-text-pointer-v3-attempt-2/`. Its
`ALL_HEADS_FROZEN.json` records both checkpoint hashes before any heldout target
text is decoded. Heldout labels are absent from each training packet. Raw
slot IDs, attention weights, copy gates, source words, per-kind literal metrics,
rejected text and all six-card views are saved for both arms.

The earlier attempt is preserved separately. It evaluated the first fixed arm
before fitting the second fixed arm, although both schedules were predeclared
and both training packets excluded heldout labels. Its
`PROVENANCE_CLARIFICATION.json` corrects an ambiguous no-prior-label-read flag
without replacing original files. The two-phase control rerun produced
**identical checkpoint bytes and predictions** for both arms.

Checkpoint SHA-256:

- Role/position: `3890eb69d5a72c0bce3530380a5730e0412b7347b438819783009b5752f44425`
- With lexical keys: `a57cbea7d1da830a2d0758738db58d9e9e651e6daea392bd5454f7e8f615a999`

Eight CPU tests passed. They cover public fact/key binding, stale color/price
and injected-target rejection, teacher-blocked inference, finite normalized
empty-source distributions under saturated gates, real pointer/vocabulary
gradients, the unchanged split, and freezing both heads before opening heldout
target text. Independent output review remains a separate prerequisite for
any promotion decision.
