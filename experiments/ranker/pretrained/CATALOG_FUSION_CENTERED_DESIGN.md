# One fixed role-centered conditioning ablation

Preparation only: no project-data centered fit has run in this lane. The paired
native attempt is preserved at
`data/ranker/pretrained/catalog-user-fusion-paired-v1-attempt-1`, execution commit
`fd8fcf86bcdc01b5b87d77de92494a4c381a6c11`. It completed 48 fresh query forwards
and zero catalog forwards in 14.271 seconds, then independently fitted genuine
raw-feature full/prior checkpoints. Fusion and prior each got 14/28 training
products correct, with zero both-correct product pairs and zero product-choice
changes out of 14 controlled pairs. Fusion also got zero both-correct image
pairs and zero image-choice changes out of 56 all-candidate image pairs.

The unlabeled CPU geometry audit is
`CATALOG_FUSION_PAIRED_GEOMETRY_AUDIT.json`. It reads only verified raw feature
vectors and checkpoint parameters, not targets. Every learned full-fusion query
projection is exactly `[1, -1, 1, 1]` after float32 tanh, on all 28 training,
8 validation and 12 test requests. Each split has one distinct projected query
row and zero across-query variance. Training pre-tanh means are approximately
`[12.297, -13.336, 12.378, 11.563]`, with standard deviations
`[0.0157, 0.0244, 0.0432, 0.0137]`; all dimensions are saturated. Float64 tanh's
mean derivative is about 1.3e-10. This gives a direct numerical explanation for
the observed query-independent decisions, despite nonzero earlier updates.

The item path is not saturated: all 42 distinct training variant members remain
distinct after projection, none reaches absolute tanh .99, and summed
across-item variance is .6644 (validation .7414, test .4037). The training raw
query mean has norm .999226; mean query residual norm is .038879. Item mean norm
is .991439 and mean residual norm .129444. The backbone features occupy a narrow
cone. These observations motivate one conditioning hypothesis; they do not prove
that centering will restore useful learning or justify promotion.

Freeze this exact transform before fitting:

```text
μquery   = float32(mean_float64(exactly the 28 paired TRAIN query vectors))
μcatalog = float32(mean_float64(the 42 unique paired TRAIN candidate variants))
q′ = (q − μquery)   / max(l2(q − μquery),   1e-8)
c′ = (c − μcatalog) / max(l2(c − μcatalog), 1e-8)
```

The catalog mean uses the 21 products actually appearing among paired training
candidates, with each variant counted once. It does not use per-request repeated
rows, the other old training products, validation/test products, relevance labels
or image targets. Zero residuals deterministically map to zero under the fixed
epsilon and are counted in preparation receipts. No epsilon/optimizer sweep or
outcome-driven retry is proposed.

Keep the same seed 728, rank-four tanh scorer, 32,786 trainable full-fusion
parameters, objective, AdamW learning rate .03, weight decay .001, 80 epochs, one
CPU thread and 60-second per-arm cap. Independently fit the centered prior with
16,394 trainable parameters and zero/frozen user interaction. Retain a centered
random baseline and cyclic query permutation. Log each epoch's loss, gradients,
query/item tanh saturation, projected variance and distinct-row count. Keep
failed attempts; do not adapt the schedule to inspected outcomes.

The distinct representation is `catalog-user-fusion-role-centered-v1`.
`CenteredFusionScorer` stores the two 4096d means as frozen buffers (32,768 bytes),
with epsilon, exact training membership, raw-input hashes, mean hashes and source
receipts in its checkpoint. There remain exactly 32,786 trainable parameters;
the 8,192 statistics floats are not parameters. The new loader refuses old raw
heads, and the old loader refuses centered checkpoints. Reload predictions must
match exactly on raw inputs, including the internal centering transform.

`centered_runner --prepare-only` reads only training vector values, computes
statistics and writes a reviewable plan/statistics artifact. Fitting requires
that exact reviewed plan checksum and identical input/statistics/source hashes.
Both full/prior checkpoints plus the random baseline are saved, reloaded and
frozen before any held-out vector values or labels are loaded. All cached input
stores are read-only; the runner performs zero native forwards and writes no
raw feature-cache entries. Training vector statistics and original source-report
checksums are verified again after evaluation without updating the means.

Report raw/guarded product accuracy, image denominators, product both-correct and
choice-change pairs, image both-correct and style-change pairs, and the random,
prior and permutation controls on train/validation/test. Keep all paired/raw
reports separate. Validation/test are replayed synthetic public-fact request
holdouts on previously inspected catalogs, not untouched or sealed tests.

This is a decision-conditioning experiment. CPU cached-feature timings exclude
fresh native query encoding and copy generation. The user's above-fold complete
three-card view and the complete full storefront are separate output contracts;
neither is evaluated by this scorer-only ablation. It does not train fresh copy
or section grouping and supports no conversion-lift claim.
