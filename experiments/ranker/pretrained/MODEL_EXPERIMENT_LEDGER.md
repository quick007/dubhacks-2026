# Model experiment ledger

Run names distinguish optimizer updates from numerical probes and inference.
The 9B quantized backbone weights stay frozen throughout these experiments.
Synthetic public-fact correctness is a proxy; none of these results establishes
human conversion lift or a production storefront integration.

| Run | Actual optimizer work | Checkpoint and outcome |
| --- | --- | --- |
| `semantic-copy-direct-v1` | Completed 336 updates, 65,536 final-layer `mlp.down_proj` LoRA A/B parameters | Saved `data/ranker/pretrained/semantic-copy-direct-v1/adapter.safetensors`; exact direct-loss reload. Dev titles 6→15/18; test titles 6→6/18; zero valid pages in each three-case split and model arm. Not promoted. |
| `gradient-feature-probe-v1` | Zero updates; one nonzero-adapter numerical control row | Direct native, auxiliary native and reconstructed-feature loss/A/B gradients exact. No trained checkpoint. |
| `catalog-user-fusion-paired-v1-attempt-1` | Completed 80 CPU steps for full head and 80 for independent prior | Raw-feature head had zero preference-driven product flips, including training. Saved full/prior/random checkpoints; not promoted. |
| `catalog-user-fusion-centered-v1-attempt-1` | Completed 80 CPU steps per full/prior fit | Train-only role means restore conditional image-style choices: validation 32/32, test 48/48. Product pairs remain 0/4 and 0/6 both correct. Not a complete-view result. |
| `catalog-user-semantic-factors-v1-attempt-1` | Completed 80 CPU steps per full/prior fit; full product model has 57,358 trainable parameters | Learned seven-atom sigmoid factors pass validation 8/8 and test 12/12 product choices, all paired interventions correct. Shuffled-query product controls fall to 1/8 and 4/12. Image checkpoint is byte-identical to centered-v1. Inspected synthetic replay, pending fresh transfer evaluation. |
| `semantic-copy-feature-fit-v1` | Completed 504 exact extraction gates, then 504 real globally shuffled optimizer updates, 65,536 final-down LoRA parameters | Saved and exactly reloaded `data/ranker/pretrained/semantic-copy-feature-fit-v1/adapter.safetensors`. Dev pages 0→2/3; test pages 0→3/3; trained titles 18/18 in each split. Peak MLX 6,360,122,248 bytes. |
| `semantic-copy-learned-probe-v1` | Zero updates; one exact learned-parameter control row | Native loss/A/B gradients match both recaptured and original archived features exactly; full source cache and frozen language tensors unchanged. |

The earlier 140/336 copy-update progress belongs to the now-completed direct
fit. A later 183/504 or similar zero-update report belongs to the separate
feature-extraction stage of the new fit, which subsequently completed all 504
updates. They are different runs; both checkpoints and reports are preserved.

The new fit's complete report contains all twelve frozen/learned evaluation
arms and all 504 per-row gate receipts. All 504 stored feature archives passed
independent checksum, shape, dtype and finite-value verification. The learned
checkpoint SHA-256 is `0b0aeda377a9c9ad7eb37b529c139dac84bc5acd4bb559176d40245865ffe1b2`.
All 927 frozen language tensor hashes are unchanged. The reported fit/extraction/
evaluation wall time is 779.472 seconds and excludes earlier startup token/file
verification; whole-process startup wall time was not separately captured.
Five valid learned pages have median 11.342 seconds and range 11.099–11.725.
Frozen controls have no valid-page latency sample. These times include the full
rank context encoding but use locked fixture decisions; they are copy-stage
measurements, not newly measured ranking or complete visible-card endpoints.
The one learned dev rejection is `Structured travel choices` where the selected
section's universal attribute intersection is empty. Validation was not weakened.

The direct fit's all-attempt completion medians were dev 9.290→8.832 seconds
and test 9.648→9.468 seconds (frozen→learned). With zero valid pages, there is
no valid-page latency sample and no demonstrated usable-page speedup.
An actual held-out raw title changed from `Herilis 24L Daypack — compact` to
`Herilis 24L Daypack for structured travel`; `structured` was unsupported.

Current S/T/B copy experiments contain fourteen fields for six cards, capped at
224 sampled tokens. They omit visible per-card body copy. The distinct fixed
first-section workload requires ten fields including C1/C2/C3 (cap 208), exact
original-name/price/image metadata and a measured validated/serialized payload
receipt. First-token times and last-token lower bounds are not payload readiness.
Section grouping remains deterministic. No UI, API or shopper harness was changed.

Raw reports, checkpoint hashes, exact replay audits, token counts, wall time and
memory receipts are retained in each named result directory and tracked report
copies. Failed experiments and historical validation accounting remain intact;
new values-only attempts use strict field provenance and exact raw `NONE`.


## Completed transfer and complete-card diagnostics

The first-section ten-field endpoint completed all 36 frozen/learned arms at
native widths 1/3/6. Independently rebuilt, exact-forked and repeat cache controls
pass at every matched width in both modes, including raw token IDs and complete
log-probability hashes. Across-width text equality is not the quality criterion.
No arbitrary hybrid-cache trim/splice was used. Strict complete-view acceptance
is **0/36**. All 108 body rows stop naturally within 32 tokens (maximum 23), so
this failure is not body truncation. Unsupported cross-product attributes and
lexical failures dominate; original raw evidence and conservative policy remain.
Accepted bodies are 3/54 frozen and 6/54 learned. For example, generated rolltop
was unsupported for structured-only Neridor, and compressible/reflective were
unsupported for Tavulan. Ordinary lexical rejections and invented mechanisms
remain distinguished in the qualitative audit.

Attempt medians at widths 1/3/6 are frozen 12.113/9.823/8.804 seconds and learned
10.943/9.742/7.890 seconds. There are no accepted-payload latency or speedup
samples. Wall time is 426.624 seconds and cumulative MLX peak 6,756,602,444 bytes
under the fixed seven-GiB guard. See above_fold_copy/BENCHMARK_AUDIT.md,
BENCHMARK_QUALITY_NOTES.md and results/v1/ for exact field/token/payload/control
receipts. This adapter had no body-copy supervision and changed prompt layout.

The separately frozen fresh-catalog evaluation used 36 new backpack SKUs,
72 variants, 24 new three-tag requests and 96 actual B1 native encodes. There
was no fitting. Learned product correctness is **2/24**, matching prior and
opposing-query controls; product pairs are 0/12 both correct and only 1/12
preference flips. The public-fact oracle is 24/24. Perfect inspected-split replay
therefore did not establish transfer. All 144 grid components per arm pass
assembly guards, which is distinct from ranking quality or complete pages.
Inherited image choices are 639/864 with 207/432 both-correct paired actions.
The independent audit verified 96 artifacts, 288 score rotations, 240 readouts
and 720 assembled grids. See CATALOG_FUSION_FRESH_EVAL_ATTEMPT_1_AUDIT.md and
catalog_fusion/results/fresh-eval-v1-attempt-1/ for preserved raw evidence.

Actual model-resident fresh-query encoding plus 72-row verified warm catalog,
scoring and both 12/24 grids has median 210.816 ms. Cold catalog preparation is
21.018 seconds, model load 2.855 seconds, complete native run wall 33.012 seconds,
and peak MLX 4,856,720,708 bytes under six GiB. These exclude generated copy and
browser rendering; they establish neither page quality nor whole-page latency.

## Actual storefront alignment and next bounded experiment

Read-only teammate inspection pinned origin/feat/design-dash at
65b5896b28d4901b95a691fd68436e79ad264e61. Its mock presentation has six jackets
and ten public typed tasks. The separate actual storefront still shows fixed
mug/pack/notebook products and has no learned inference hook. Main incorporates
the teammate VISION.md merge; no design branch was merged and no UI/API/shopper
harness was changed. The model-only jacket_alignment_v1 snapshot preserves
multi-valued size/color/waterproof attributes and excludes persona metadata,
scripted outcomes and source IDs from predictive text.

Every public jacket task has only one or two eligible products. Images, stock
and shipping are absent. Current fixed-three-card/image-required backpack
contracts cannot honestly represent those tasks. The next bounded experiment
should freeze an attribute-preserving jacket input and variable-card,
image-abstaining output contract with explicit shortfalls, then compare real
learned copy/ranking on separately frozen tasks and catalogs. Keep item-price
budget semantics separate from delivered cost and waterproof distinct from
water-resistant. Do not fit these ten inspected tasks and call their inherited
held-out names a fresh evaluation. Grouping remains deterministic. No checkpoint
from the failed transfer or complete-card arms is promoted.


## Corrected-input generalization v2, completed

The explicitly approved restart adds versioned typed jacket inputs and honest
variable-card plans without rewriting legacy encoder/validator/checkpoint bytes.
All21 targeted CPU tests pass; all10 pinned public tasks reproduce15 returned
cards of30 requested,15 disclosed shortfalls and55 independent unexecuted copy
jobs. Image selection abstains, stock/shipping remain unknown, and hard masks
precede caller scores. See JACKET_MODEL_CONTRACT.md and the independently checked
jacket_alignment_results/v1/report.json. This is a model-side contract, not native
jacket ranking/copy or a live serving integration.

A real corrected-data v2 comparison completed at execution5ca10cb:192B1 native
encodes, four fixed80-step factor fits (320 actual steps) and two random heads.
The frozen9B weights were unchanged; TRAIN-only role means use21queries and21
products per representation. Private ranking targets are opened only after all
six head checkpoints freeze. Exactly matched supervision/schedule compares full
public text against a new compact native encoding of only declared positive
preferences/product attributes. Both remain real native hidden features, not
one-hot oracle features. No image variants were invented.

Focused validation and test each achieve21/21 best-set choices, nDCG3=1 and
94/94,115/115 both-correct disjoint-best-set pairs. Full text achieves8/21 and
7/21; its test prior gets10/21. Focused test prior10/21, random4/21 and
opposing-query11/21 distinguish the learned conditional result. All14 novel
pair/triple queries per held-out split pass; singleton meanings/texts necessarily
overlap and some focused catalog rows match TRAIN singleton text exactly. This
is nuisance invariance plus combination transfer on synthetic public attributes,
not evidence of unseen singleton meanings, natural preference learning or human
conversion. Pair metrics reuse21queries and are correlated, not115independent
observations. Both modes share newly balanced data, so gains over older runs
cannot isolate the coverage correction from the representation correction.

Actual test native query-encode medians are180.804ms full and124.990ms focused;
full context60–75tokens, focused15–25. These are encoding measurements from one
fixed-order run, not end-to-end rank/grid/copy/page speedups or latency tails.
Model load3.009s, native lifecycle33.669s, CPU fit/freezes.125s, evaluation.281s,
wall including startup37.041s, peak MLX4,706,802,514bytes under6GiB. No cache or
memory limit was hit. Full raw tokens/counts/timings/probabilities/scores,
checkpoints/fit traces and immutablefeature/checkpoint receipts are retained in
catalog_fusion/results/generalization-v2-attempt-1 and the original data run.

The independent audit verifies192native artifacts,384sealed feature files,
six checkpoint reloads,320steps,630score vectors,52,920probability scalars and
420atom confusion tables. See CATALOG_FUSION_GENERALIZATION_V2_ATTEMPT_1_AUDIT.md.
No v2 checkpoint is promoted to jacket, image, generated-copy or full-page use.
The next transfer experiment must use current storefront facts and a new public
task/label freeze, then validate fresh copy and complete-request latency.

Teammate main advanced during this restart. A normal non-force merge at e172f60
preserves their16-product storefront/color/gallery work and the model fix. That
catalog is separate from the pinned six-jacket scenario; versioned actual-store
input mapping is being recorded separately. MEMORY_MODEL_ASSESSMENT.md records
real24GiBhardware/headroom/swap and larger-model planning. The6/7GiBguards are
experiment limits, not user hardware restrictions. No apps were closed and no
weights were downloaded: current corrected-data quality improved on the same9B.


The actual-store import is now completed and independently checked:16products,
48color variants,195gallery records,32separate previews and81hashed local assets.
`store_catalog_v1.py` preserves all external crop/action bindings while excluding
routing IDs and persona briefs from model text. The additional `store_model_bridge`
checks current bytes of15selected input-producing sources before/after preparing
all195rows; historical authenticity alone does not establish active freshness.
The actual CPU receipt is store_model_results/v1/report.json; no native action,
rank, image choice or new copy is claimed. Three freshness/action tests supplement
14import tests. See ACTUAL_STORE_MODEL_INPUTS.md for capabilities and public task
compatibility limits. The old jacket snapshot remains unchanged.

The independent data review finds old labels teacher-consistent, but the old
TRAINquerydistribution was28singletons while validation/test usedonlypairs;
all84oldcatalogproducts hadexactlytwo tags. V2balances bothTRAINroles at21rows,
seven each cardinality1/2/3, eachatom6positives/15negatives. Seeded nuisance
assignment is not empirical decorrelation: its six-product testcatalog has
price-versus-tag-cardinality Pearsonr=.837. This narrow result needs new catalog
replications and explicit nuisance counterfactuals before broad claims. Focused
v2also omits free task wording/history from its positive-preference vector; it
is not a replacement for observed-profile/history reasoning. Unknown/negative/
contradictorypreferences and hiddenpersona inference were not trained/evaluated.
Hard public constraints remain outside learned scores. No larger model was
substituted to disguise these data/coverage limitations.
