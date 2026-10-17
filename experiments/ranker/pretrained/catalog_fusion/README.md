# Catalog fusion CPU preparation

This package implements the proposed `catalog-user-fusion-v1` input contract,
32,786-parameter rank-four scorer, train-only objective boundary, checkpoint
checksums, and complete-roster image mapping. It is preparation only. There are no
native features, trained project checkpoints, quality results, or speed claims.
Existing joint heads and their results remain a separate representation.
Actual project training, backbone/model outputs, native feature extraction and
fresh-request encoding latencies are unmeasured by this package. Checkpoint
saving uses exclusive creation and refuses to overwrite an existing artifact.

Run the synthetic CPU contracts from the repository root:

```sh
.tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.catalog_fusion.test_catalog_fusion -v
```

The CPU-only `preflight` module inventories the real public corpus with the
already local tokenizer, without loading weights or labels. See
[FIT_EXTRACTION_PLAN.md](FIT_EXTRACTION_PLAN.md) for exact available paths,
observed token bounds, artifact sizes and a bounded native-extraction/real-fit
handoff. Native extraction and actual project fitting remain unexecuted here.

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
project fitting command or automatic data discovery.

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
