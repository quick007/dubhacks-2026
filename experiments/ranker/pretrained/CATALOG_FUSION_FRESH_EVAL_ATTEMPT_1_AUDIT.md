# Independent fresh-catalog audit

Audit passed at native execution commit `0322f983ea5f02889b10fb37f54243b952f198b5`. All96 raw feature artifacts and exact B1 whole-input receipts verify; all96 saved learned/control score tensors reload exactly. All288 learned rotation-score recomputations,240 raw/guarded readouts,720 full grid assemblies,24 teacher labels,8 atom diagnostic blocks and56 independent per-atom confusion checks match. No rotation failure occurred. Full source/model/data hashes and receipts are in the adjacent JSON. No native execution or fitting occurred in this audit.

| Arm | Product correct | Product pairs both correct | Preference flips | nDCG12 | nDCG24 |
|---|---:|---:|---:|---:|---:|
|fusion|2/24|0/12|1/12|0.4727|0.6031|
|prior|2/24|0/12|0/12|0.4090|0.5710|
|permuted-query|2/24|0/12|1/12|0.3959|0.5653|
|random|0/24|0/12|0/12|0.3602|0.5489|
|public-tag-oracle|24/24|12/12|12/12|1.0000|1.0000|

The inspected-split product success does not transfer strongly to the36 new SKUs and new three-tag requests: full accuracy matches prior and opposing-query controls, with no fully correct product pair. nDCG improves modestly over those controls. The public teacher oracle passes all products/pairs and has nDCG1, so the labels and grid task remain feasible.

All five arms fulfill144/144 grid components with correct masks, requested12/24 counts, all36 ranked products/all72 scored variants and compatible per-selected-product images. These are assembly properties, not ranking correctness or complete-page quality. All three roster views preserve exact ID-mapped learned scores and actions.

Inherited image choice is639/864 (73.96%), with207/432 both-correct image interventions (47.92%). The full atom decoder query precision/recall is53.62%/51.39%; catalog precision/recall is46.54%/42.53%. Every copied factor/image checkpoint is byte-identical to the frozen source; original TRAIN means remain unchanged and exclude all fresh rows.

Median genuine fresh-query native encode is190.17ms; warm72-row catalog verify/load is20.67ms; head score is1.15ms; two grid mappings take0.23ms. The observed model-resident fresh-query/warm-cache/head/two-grid median is210.82ms. Cold catalog work is21.018s, model load2.855s, full run wall33.012s. Maximum observed native peak is4.523GiB under the fixed6GiB guard. These timings include no fresh copy generation or browser/page rendering.

The feature/prediction freeze gate precedes private evaluation in source/tests/receipts; no syscall trace exists. The generator/root reviewer could inspect labels, but no fresh labels entered fitting or encoding. This new packet is now inspected and cannot be reused as an untouched test. Results concern a synthetic seven-atom backpack rank/grid component. They do not validate the teammate's six-jacket/ten-task mock deck, storefront mapping, complete page, learned grouping/copy or conversion lift.
