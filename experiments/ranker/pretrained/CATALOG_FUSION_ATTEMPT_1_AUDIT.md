# First separated-feature fusion: audit and next supervision

The native run completed and trained genuine new checkpoints, but it does not
establish personalized product ranking or reliable held-out use of image
preferences. Do not promote this scorer. Source run:
`data/ranker/pretrained/catalog-user-fusion-v1-attempt-1`, execution commit
`e9fb4afc11e489220c747f5f7d03f9d10754a3b6`.

The independent CPU audit verified all 216 immutable feature entries against
current canonical inputs/keys/provenance/checksums, then reproduced all 192 saved
score matrices exactly from features and checkpoints. All 192 independent
mean-variant raw rankings also agree. This verifies the artifacts and execution
path; it does not make the labels sufficient. Detailed evidence is in
`CATALOG_FUSION_ATTEMPT_1_AUDIT.json`.

| Replay split | Random product raw / guarded | Item prior | Full fusion | Permuted query |
| --- | --- | --- | --- | --- |
| Train, 24 requests | 2 / 3 | 24 / 24 | 24 / 24 | 24 / 24 |
| Validation, 6 requests | 4 / 5 | 0 / 1 | 0 / 1 | 0 / 1 |
| Test, 12 requests | 6 / 6 | 2 / 5 | 2 / 5 | 2 / 5 |
| Diagnostic, 6 requests | 2 / 2 | 1 / 1 | 1 / 1 | 1 / 1 |

Numbers are correct-request counts, not percentages. Full fusion and its
permuted-query control have identical complete raw product rankings on all 48
requests. On the three diagnostic pairs whose target products change, every arm
has zero both-correct pairs and zero product-selection changes.

The training fixture has 12 disjoint four-product pools, each shown twice. All
12 pools keep the same best-product label across their two observable requests.
All 19 pool/product pairs that have informative compatible image labels in both
requests also keep the same best variant. A static item prior therefore has a
direct memorization path. Nonzero U/V/interaction gradients and a lower training
loss demonstrate actual optimization, not the desired dependence on current
preferences.

| Informative compatible images, all candidates | Prior | Fusion | Permuted query |
| --- | --- | --- | --- |
| Train | 54/54 | 54/54 | 39/54 |
| Validation replay | 8/16 | 8/16 | 15/16 |
| Test replay | 4/23 | 12/23 | 13/23 |
| Diagnostic replay | 16/22 | 22/22 | 22/22 |

Image denominators cover every candidate with two compatible, differently
scored variants. On the test oracle-product set, fusion is 5/8, the prior 2/8
and permuted queries 5/8. On model-selected products these are 3/6, 1/6 and 4/6.
The selected-product denominator changes with model choices; it cannot replace
the all-candidate or oracle-product figures. The correct current query does not
outperform permutation on the inspected held-out image sets.

The full run took 62.623 seconds, including verification, two model loads,
216 native forwards, both CPU fits and evaluation. Model loads were 2.582 and
1.551 seconds; full/prior fits were 0.442/0.442 seconds. Observed cumulative MLX
peak was 4,809,720,344 bytes (4.4794 GiB), below the fixed 6 GiB active/peak guard.

| Phase | Catalog cold mean pipeline ms | Catalog warm mean pipeline ms | Warm fresh-query native ms |
| --- | --- | --- | --- |
| Train, inputs before fit | 2326.62, n=12 | 202.63, n=12 | 193.90 |
| Validation, fitted decisions | 2363.51, n=3 | 191.77, n=3 | 183.57 |
| Test, fitted decisions | 2374.94, n=6 | 209.35, n=6 | 198.64 |
| Diagnostics, fitted decisions | — | 178.93, n=6 | 173.44 |

Warm catalog construction/hash/load costs averaged 5.82/3.99/5.19/2.43 ms in
those phases. Cold and warm groups contain different requests, so these are
observed trace groups, not a matched speed factor. Artifact persistence/checks
are included. No generated copy, full storefront, or labeled 12-/24-card quality
was measured. Native A/B/A/fresh-feature parity belongs to the separate probe.

The next frozen data version is
`catalog_fusion/paired_v1/FREEZE.json`, SHA-256
`1e7238f090c21c83dc2715f5b402dc8b9c84cfe9b30e334d4efdae11fdbbed54`.
It uses unchanged old catalog bytes/vectors, but new 2×2 query families: two
observable product preferences crossed with detail/in-use presentation style.
Each family keeps the exact pool/order, budget, requirements, history, surface
and compatibility masks fixed. Every target is a unique public tag-overlap
winner; both image styles are valid for every candidate. No old relevance,
private persona or latent preference labels are reused.

There are 28 training, 8 validation and 12 test queries. Each of all seven tags,
including water-resistant, has four positive training queries. Training uses
singleton preference sets; validation/test use mutually disjoint two-tag sets.
Products remain within old train48/validation12/test24 partitions, with new
disjoint pair families. Five validation and seven test query preference/style
signatures already occur in old public contexts; these are fresh request
holdouts on replayed catalogs, not an untouched/sealed catalog test.

A deterministic item prior can get at most half the individual product labels
in a balanced family and zero product both-correct or image both-correct pairs.
The next fixed 80-epoch fit must be compared with that separately fitted prior,
random weights and query permutation, reporting both-correct and choice-change
counts alongside ordinary raw/guarded metrics. The new runner needs exactly
48 fresh query forwards and zero catalog forwards. Data/teacher/pool selection
was reviewed by the parent before any genuine next fit; this CPU lane performs
no native extraction or project-data fitting.
