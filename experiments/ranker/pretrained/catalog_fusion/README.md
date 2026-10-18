# Catalog fusion experiment

The paired raw-feature attempt's query-path saturation is documented in
[CATALOG_FUSION_CENTERED_DESIGN.md](../CATALOG_FUSION_CENTERED_DESIGN.md) and
`CATALOG_FUSION_PAIRED_GEOMETRY_AUDIT.json`. A single CPU-only conditioning
ablation is prepared in `centered_runner.py`: means from exactly the 28 training
queries and 42 unique training candidate variants, followed by fixed-epsilon
centering/L2 normalization and the unchanged rank-four tanh scorer. Frozen means
live in a distinct checkpoint format; no raw native cache entries are changed.

`centered_v1_prepare/plan.json` is a statistics/config-only review artifact. A
genuine fit requires its checksum plus unchanged input/statistics/source hashes:

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.catalog_fusion.centered_runner --prepare-only --out NEW_PREPARATION_DIRECTORY
```

After source/plan review, run once using `--plan PATH --reviewed-plan-sha256 SHA
--out NEW_RUN_DIRECTORY`. It uses existing verified raw vectors and zero native
forwards, with the same fixed 80-epoch full/prior/random/permutation comparison.
Held-out vector values/targets are loaded after checkpoint freeze. The prepared
ablation has not been fitted in this lane. Three-card above-fold complete views
and full storefront completion remain separate, unmeasured generation contracts.

The first genuine native run is independently audited in
[CATALOG_FUSION_ATTEMPT_1_AUDIT.md](../CATALOG_FUSION_ATTEMPT_1_AUDIT.md).
It verifies execution/artifacts but does not establish learned personalization;
the scorer is not promoted. The next data-only freeze is
[`paired_v1/FREEZE.json`](paired_v1/FREEZE.json), with 28/8/12 new queries on
unchanged old train/validation/test catalog partitions. Every family crosses two
observable product preferences with both compatible image styles, forcing
product/image target changes under identical masks. All seven tag atoms have
positive training coverage. Old catalog/test outcomes are replay evidence;
five validation and seven test preference/style signatures overlap old public
contexts. These fresh request holdouts are not a sealed or untouched catalog test.

`paired_runner` verifies all old catalog feature keys/checksums read-only, then
performs **only 48 fresh query forwards**, the same fixed full/prior fits, and
post-freeze held-out/pair evaluation. A CPU preflight is available with
`--preflight-only`. The actual serialized native command binds the reviewed data:

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.catalog_fusion.paired_runner --out data/ranker/pretrained/catalog-user-fusion-paired-v1-attempt-1 --reviewed-freeze-sha256 1e7238f090c21c83dc2715f5b402dc8b9c84cfe9b30e334d4efdae11fdbbed54
```

The data was frozen/reviewed before a genuine next fit. CPU stub tests are
orchestration evidence only. The new runner reports ordinary raw/guarded quality,
product both-correct/choice-change pairs and all-candidate/oracle/selected image
both-correct/style-change pairs. A prior cannot satisfy both opposite labels in
any controlled pair. No catalog native forward, response reuse, copy generation,
hyperparameter sweep or fresh human-utility claim is part of this attempt.

This package implements the proposed `catalog-user-fusion-v1` input contract,
32,786-parameter rank-four scorer, train-only objective boundary, checkpoint
checksums, complete-roster image mapping and an executable extraction/fit runner.
The runner has not been executed natively by this preparation lane. There are no
new native features, trained project checkpoints, quality results or speed claims
from the CPU tests.
Existing joint heads and their results remain a separate representation.
Actual project training, backbone/model outputs, native feature extraction and
fresh-request encoding latencies are unmeasured by this package. Checkpoint
saving uses exclusive creation and refuses to overwrite an existing artifact.

Run the synthetic CPU contracts from the repository root:

```sh
.tools/pretrained-venv/bin/python -m unittest discover -s experiments/ranker/pretrained/catalog_fusion -t . -v
```

The CPU-only `preflight` module inventories the real public corpus with the
already local tokenizer, without loading weights or labels. See
[FIT_EXTRACTION_PLAN.md](FIT_EXTRACTION_PLAN.md) for exact available paths,
observed token bounds, artifact sizes and a bounded native-extraction/real-fit
handoff. Native extraction and actual project fitting remain unexecuted here.

Run the actual experiment only in the serialized native lane, with a **new**
output directory:

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.catalog_fusion.runner --out data/ranker/pretrained/catalog-user-fusion-v1-attempt-1
```

This verifies the local pinned weights, checks exact CPU/native tokenizer IDs,
extracts B=1 whole-input features, fits full fusion and a separate item-prior
control using training targets only, saves/reloads/freezes both checkpoints,
then extracts/evaluates held-out replay requests. The existing 6 GiB guard and
512-token cap are fixed; active **and observed cumulative MLX peak** are checked
after load and before/after every encode. Limits and the 80-epoch schedule are
in `runner.LIMITS` and `fit_runner.FIT_CONFIG`. Existing output directories and
checkpoints are refused. Failures retain partial artifacts plus `FAILED.json`;
there is no retry loop, download, worker or automatic memory increase.

`report.json` contains full/random/prior/permuted-query quality with raw and
guarded decisions, oracle-/selected-/all-candidate presentation denominators,
fit gradients/updates/reload receipts and actual per-query catalog hit/miss,
fresh-query and CPU score/mask/grid timings. Per-feature `.pt`/JSON files retain
exact keys, serialized/vector checksums, provenance and native timing/memory.
`CHECKPOINTS_FROZEN.json` must verify before held-out target files are opened.
Execution commit, source status/hashes, interpreter and runtime package versions
are retained. The native model is unloaded during fitting and reloaded for
held-out extraction; those load costs are reported separately. No copy is
generated. Candidate-count grid probes on the four-candidate fixture do not
establish 12-/24-card quality. A/B/A and matched fresh-native extraction parity
remain unmeasured; saved artifact bytes are checked exactly.

`inputs.catalog_text` uses the existing `clean_product`/`variant_text` allowlists.
`request_text` encodes fresh public context and the same cutoff-filtered latest
history type/confidence/tags summary as the compact representation. It does not
encode candidate products, opaque identifiers, compatibility tables or labels.
Structured values inside scalar/string-list fields fail closed. Opaque IDs
remain external roster/receipt keys. This is schema separation, not detection of
an identifier deliberately inserted inside an otherwise public prose field.

`EncoderSpec` and `raw_key` bind model/weights, quantization, tokenizer/template,
canonical text and exact token hashes, role, format, pooling, normalization,
dtype, dimensions, controlled extraction batch shape/chunking, and catalog
content version. A projected key also binds the checkpoint checksum. Raw
features remain independent of a fusion update. IDs/order do not enter a raw
key. Native extraction must supply actual token hashes and a content version;
there is no placeholder tokenizer or extraction path here.

Before scoring, callers must build expected keys from the **current consistent
catalog snapshot and fresh request**, then call `check_features` for each role.
It validates exact external membership/order, keys, saved float32 bytes, finite
normalized 4096-dimensional rows, pooling and explicit no-truncation receipt.
Catalog rows must be arranged in `scorer.roster` order. Request input is one row;
pass its `[0]` vector to the scorer. Receipts check consistency, not maliciously
forged extraction claims. `score_grid` is a low-level numerical API and does not
perform this native provenance step on the caller's behalf.

`FusionScorer` performs all variant scores in one CPU operation with shared
nonzero deterministic projections. Product scores average all variant rows;
`assemble_grid` reuses current stock/budget/shipping/requirements guards and
requires explicit compatible image masks for every selected product. Intrinsic
surface restrictions also apply. Every candidate stays in the returned ranking;
an insufficient eligible pool returns `status=shortfall` and cannot count as a
complete grid. Ties use external IDs. It does not group sections or generate
copy; the existing deterministic grouping/copy pipeline must consume verified
complete cards in a future integration.

`read_training_inputs` opens only the two literal training JSONL files, rejects
symlink redirection and duplicate/mismatched memberships, and returns input
hashes. `training_loss` matches the existing mean-variant best-product CE and
informative compatible-presentation CE. It is a low-level function: its caller
must use the train-only loader and verified features. There is deliberately no
automatic data discovery; the explicit runner owns fixed fitting and the
separate post-freeze evaluator boundary.

Next steps require a separately authorized native run with the same pinned
quantized backbone/tokenizer, explicit bounded memory guard, exact prompt/token
and extraction receipts, last-valid-token pooling, and no silent truncation.
Bind immutable artifacts to snapshot/version keys; reject stale/missing entries
rather than silently dropping cards. Extract training features afresh, freeze
one fit schedule before outcomes, then train a **new** fusion checkpoint and
verify gradients, parameter changes, checkpoint hashes and exact reload scores.
The synthetic gradient update in tests is not a fitted project result.

Before promotion, use the design's random/prior-only/trained/shuffled-query
controls and genuinely new sealed family-/SKU-disjoint full-grid labels.
Previously inspected evaluation examples are replay evidence. Measure native
extraction, fresh query encoding, full-grid mapping, and fresh copy generation
separately and end to end. Existing semantic copy failures are not solved by
this architecture scaffold. Stop promotion on quality regression, stale
invalidation, missing compatible images, or incomplete grids.
