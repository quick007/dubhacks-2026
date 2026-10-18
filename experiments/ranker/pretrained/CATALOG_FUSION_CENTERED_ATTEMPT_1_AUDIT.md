# Independent centered-fusion audit

The fixed centered run completed at commit `98d71f6848f633ad7ac5ce019340519ef48c89ec`. Independent read-only CPU replay passed all **192 score tensors and 384 raw/guarded readouts**, all counterfactual pair summaries, checkpoint checksums and preserved native-report checksums. No fitting or native execution occurred in this audit.

Means exactly match a separate float64 accumulation cast to float32 over **28 train queries and 42 unique train variants**. Every checkpoint contains those same frozen means. All 20 held-out query members and 15 used held-out products are excluded. Statistics SHA-256: `fb00e986642aac467a60286d4ca1a6ceebaef9ac048587341be0810ce621bc53`. Full source, checkpoint and membership receipts are in the adjacent JSON.

| Split | Fusion product | Prior product | Shuffled product | Fusion product pairs both correct | Fusion image choices | Fusion image pairs both correct |
|---|---:|---:|---:|---:|---:|---:|
| Train |28/28|14/28|12/28|14/14|112/112|56/56|
| Validation |2/8|4/8|2/8|0/4|32/32|16/16|
| Test |6/12|2/12|5/12|0/6|48/48|24/24|

Image learning is conditional on the requested style on this synthetic teacher: prior image accuracy is half and neither prior image pair changes correctly; shuffled image accuracy is 16/32 validation and 32/48 test. Product choices change on 4/4 validation and 4/6 test interventions, but none of those held-out pairs is fully correct. This does not establish compositional product transfer.

Training query activations are 99.107% saturated at absolute 0.99, with variance sum 3.97655 and 28 distinct exact rows. They differ across queries; this is different from the earlier constant saturated projection. The shared product interaction has rank at most four. The used training catalog's seven-atom tag matrix has rank six, although that does not prove the ranking task needs rank six.

A broad disjoint two-tag training expansion is constrained by the catalog. Excluding the ten held-out public two-tag combinations leaves eleven combinations, but only compact+laptop-sleeve (five eligible old-train products) and lightweight+organizer (six) have full matches. The existing singleton training still covers all seven atoms.

Propose one separate **seven-atom additive semantic product head**: two biased linear 4096→7 decoders over the existing train-centered query and catalog vectors (57,358 trainable parameters), supervised on public train preference/tag multi-hot targets plus product cross-entropy with fixed equal loss weights. Average variant atom predictions per product, then dot with the query atom prediction; use no tanh. Preserve the existing image checkpoint frozen. Use seed 728, 80 epochs and the same fixed optimizer settings; freeze a new format/config before evaluating. Fit independent item-prior, seeded-random and query-permutation controls; report exact pair correctness, flips and atom errors. This method jointly changes the factorization and adds public-atom supervision, so a gain would not isolate either component. It requires no new native features. No implementation or fit is authorized by this proposal.

This is public factual teacher distillation, not hidden-preference evidence. The public-tag-overlap oracle is a sanity check. Existing held-out outcomes are already inspected/replayed. No copy, section grouping, complete above-fold/full storefront or conversion-lift claim follows. The 1.8469-second cached CPU run excludes native encoding and page generation.
