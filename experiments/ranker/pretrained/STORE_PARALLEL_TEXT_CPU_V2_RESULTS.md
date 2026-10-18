# Parallel text CPU followups

The preserved native v1 experiment generated all six cards and 20 fields at a
median 197.05 ms from fresh query encoding through canonical JSON. Its usable
title/body proxy passed 10/32 first-three views and 1/32 full-six views. All 126
heldout card fields failed; the 84 accepted heldout fields were generic headings.
Those results remain unchanged. They justify investigating the representation
and training distribution before drawing a capacity conclusion.

## Explicit catalog-trained demo

After the v1 result was frozen, a separate CPU adaptation supervised all 48
public product-color bindings in all three declared styles. Every title, badge
and short body is now TRAIN data. This is a known-catalog demo candidate, with
zero heldout bindings and no unseen-generalization claim. It retains the same
Qwen feature backbone and 718,810-parameter parallel word output head. Runtime
still predicts every word/EOS slot from features and public conditions; it does
not load target prose, perform a complete-string lookup or repair rejected text.

The fixed 160-step checkpoint reproduced 526/528 supervised fields correctly.
Two Slate-blue bodies failed: one repeated “Item” and omitted punctuation; the
other repeated its price and omitted “at.” Both are retained in
`data/ranker/pretrained/store-parallel-text-train-all-demo-v1-attempt-1/` with
its original source snapshot and checkpoint.

Those two unresolved TRAIN fields and final TRAIN loss 0.10426 justified a
separate predeclared 320-step fit. It reproduced all 528 supervised fields:
96 section headings, 144 titles, 144 badges and 144 bodies. It also produced
accepted complete 20-field output for each of the 32 previously selected card
sets in all three styles. These were CPU replays of known public cards, not new
Qwen-query or browser timing measurements. Final loss was 0.01147; fitting took
1.756 seconds on one CPU thread. Safe checkpoint reload preserved all slot IDs.

The 320-step checkpoint is
`data/ranker/pretrained/store-parallel-text-train-all-demo-v1-attempt-2/head.pt`,
SHA-256 `074169e31bf5f6b99cc5bc35c9b2e64d82396d5225b3a67cc2cf5f613fcc4889`.
Its `serving-manifest.json` and `public-runtime-features.json` export checksums
and the 48 public fact inputs, with no teacher labels. Independent source/caption
review and a fresh-query native benchmark are separate checks; training replay
alone does not establish useful generalization or live readiness.

## Heldout research candidates

The original 34 TRAIN / 14 heldout split remained fixed. Descriptive geometry
showed mean pairwise cosine 0.97615 and mostly small within-product color
differences. Normalization therefore used only the 34 unique TRAIN feature
vectors; no heldout vector fitted its mean or scale. A coverage audit also
identified 20 heldout position/word classes without positive TRAIN examples in
v1's separate position classifiers. A shared vocabulary classifier removes
that parameter-level restriction, but does not guarantee compositional color
decoding.

Three arms were declared before evaluation, saved at their fixed final step,
and retained regardless of result:

| Candidate | Heldout title pass | Badge pass | Body pass | Exact body price |
|---|---:|---:|---:|---:|
| Independent slots, TRAIN normalization, 160 steps | 0/42 | 1/42 | 0/42 | 16/42 |
| Shared word classifier, raw features, 800 steps | 3/42 | 3/42 | 2/42 | 32/42 |
| Shared word classifier, TRAIN normalization, 800 steps | 0/42 | 0/42 | 0/42 | 6/42 |

These failures are preserved under
`data/ranker/pretrained/store-parallel-text-cpu-v2-attempt-1/`, including raw
predictions, references kept offline, per-kind literal checks, normalization
provenance, checkpoints and all 42 correlated six-card views. TRAIN-only
normalization hurt this setup. Slot sharing improved a few literal outputs but
did not solve heldout selected-color transfer. None of these research arms is
promoted. The evaluation remains 14 heldout combinations of known products,
not unseen catalogs or 42 independent customers.

## Verification and runtime boundary

Six CPU tests check TRAIN-only normalization, shared output shape, positional
coverage, literal metrics, explicit all-48 supervision and teacher-free loading.
Eight additional benchmark tests cover the 32-request/8-guard fixed workload,
service refusal, native ownership, deadlines/failure receipts, source integrity,
and reading relevance labels only after prediction hashes are frozen. No GPU
job was launched by this lane and no live UI/service was changed.
