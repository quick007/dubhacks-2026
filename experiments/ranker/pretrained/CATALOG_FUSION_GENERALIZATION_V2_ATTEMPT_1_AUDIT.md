# Independent generalization-v2 audit

The native run at commit `5ca10cbfaae1e6fd848c9bf622c7a1dbbfc36f41` completed the fixed192 B1 encodes and four80-step CPU fits. Independent read-only replay verifies192 raw artifacts/receipts,384 sealed files, six checkpoint reloads, all630 saved case score vectors and their role probabilities,63 teacher labels,60 atom diagnostic blocks and420 independently counted per-atom confusion tables. Fourfit traces contain exactly80 consecutive steps; genuine gradients, parameter changes, final BCE and TRAIN-only means/targets recompute exactly. No fitting/native execution occurred in this audit. Hashes and full evidence are in the adjacent JSON.

| Representation | Validation best-set | Test best-set | Test prior | Test shuffled | Test random | Validation/test nDCG3 | Fully correct disjoint-best-set pairs |
|---|---:|---:|---:|---:|---:|---:|
| Full text |8/21|7/21|10/21|5/21|2/21|0.6552 /0.6214|4/94 and9/115|
| Focused facts |21/21|21/21|10/21|11/21|4/21|1.0000 /1.0000|94/94 and115/115|

Both representations fit21/21 training choices. Focused held-out singleton/pair/triple groups each pass7/7; its fourteen unseen pair/triple query compositions per split therefore pass14/14. Full-text cardinality groups pass4/7,2/7,2/7 on validation and4/7,1/7,2/7 on test. Public-oracle choices/nDCG pass throughout. Query permutation/prior/random controls remain worse. These paired comparisons reuse the same21queries per split;94/115pair counts are correlated, not independent trials.

The focused input is actual public preference/tag strings sent through the frozen9B model, not one-hot teacher predictions. Native4096d vectors and subsequent learned seven-factor probabilities are persisted. TRAIN has21queries/21uniqueproducts and each atom has6positives in each role. Held-out catalogs each have sixproducts/two per cardinality and all seven atoms covered, but their atom frequencies are unequal. Identities and full encoder texts are disjoint; pair/triple signatures are partitioned.

Focused singleton inputs intentionally overlap: seven request texts per split pair, two catalog texts train/validation and two train/test. All25cross-split matching-text comparisons have identical raw keys and **bit-exact hidden vectors**. The192artifacts contain174distinct raw keys. Thus singleton results show nuisance invariance, not unseen singleton semantic transfer. New pair/triple combinations are separate evidence within this seven-atom grammar. Full/focused methods share the same corrected data and fixed schedule; comparisons with old experiments jointly change representation and supervision coverage.

Native test query-encode medians are180.804ms full and124.990ms focused. Full wall including startup is37.041s; native lifecycle including model load33.669s, load3.009s, CPU fit/freeze0.125s and evaluation/audit0.281s. Maximum observed native peak is4.384GiB below the fixed6GiB guard. Query values measure actual native encoding only, not end-to-end request/copy/render latency.

This supports a bounded synthetic backpack atom-decoding/input-alignment correction. It does not validate the teammate's jacket catalog, latent preferences, arbitrary language, image selection,12/24-card grids, learned grouping/copy, complete storefront or conversion lift. No image/dummy variants were introduced. All base price/stock/requirement masks remain deterministic and base cases are eligible. The packet is now inspected and must not be called untouched in later tuning. Seven new CPU contract tests had passed before launch.
