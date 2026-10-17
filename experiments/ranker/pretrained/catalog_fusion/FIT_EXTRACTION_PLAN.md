# Bounded native extraction and genuine fusion fit handoff

This is a CPU preparation result. Native extraction and project training have
not run. The existing cache cannot supply this representation. The next main
lane must extract real frozen features before a fit is described as trained.

## Verified local inputs

All relative paths below are relative to
`/Users/lukas/Documents/Projects/dubhacks-2026`.

| Input | Exact path / observed contents |
| --- | --- |
| Interpreter | `.tools/pretrained-venv/bin/python` |
| Catalog | `data/ranker/datasets/v2/catalog.json`: 84 products, 168 variants |
| Optimization requests | `data/ranker/datasets/v2/public/train.jsonl`: 24 requests, 48 distinct candidate products |
| Optimization labels | `data/ranker/datasets/v2/labels/train.jsonl`; only this split may enter optimization |
| Validation replay | `data/ranker/datasets/v2/public/validation.jsonl` and `labels/validation.jsonl`: 6 requests, 12 candidate products |
| Test replay | `data/ranker/datasets/v2/public/test.jsonl`: 12 requests, 24 candidate products; evaluator-only labels are in `data/ranker/datasets/v2/private/truth.jsonl` |
| Counterfactual replay | `data/ranker/datasets/v2/public/diagnostics.jsonl`: 6 requests, 12 candidate products already in the catalog; evaluator-only labels are in `data/ranker/datasets/v2/private/diagnostic-truth.jsonl` |
| Existing joint feature cache | `data/ranker/pretrained/shared-prefix/training-embeddings.pt`, 3,162,883 bytes, `shared-prefix-v1`; reference only, never reinterpreted as independent catalog vectors |
| Existing corrected joint head | `data/ranker/pretrained/evidence-plan-v1/heads.pt`; retain its `training.json` receipt and checkpoint checksum |
| Earlier joint artifact manifest | `data/ranker/pretrained/shared-prefix/run-manifest.json`; existing pinned-artifact verification is recorded here |
| Local backbone | `.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53` |

The local backbone directory exists. Shard sizes are 5,349,771,222 and
600,449,850 bytes (5,950,221,072 total stored bytes, including wrapper components).
This directory/config check is not a fresh weight hash verification or a
resident-memory measurement. The pinned receipt contains the two expected SHA
values. Hash the local shards once before native extraction, retain that cost,
and stop if either disagrees. No download is needed.

There is no `catalog-user-fusion-v1` feature artifact in the current pretrained
data tree. The old decoder-state `catalog-cache-v1` safetensors are not frozen
embedding features and must not be loaded into this scorer.

## Reproducible CPU preflight

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.catalog_fusion.preflight --summary
```

`preflight.py` loads the local `tokenizer.json` with the CPU `tokenizers` library,
constructs only allowlisted catalog/request strings, and reads no label/private
files. It never imports MLX or loads model weights. Without `--summary`, stdout
also contains each external member, text/token SHA-256 and token/batch shape;
retain that inventory in a new run directory during the actual extraction.

Observed on the current public inputs: 168 catalog strings of 127–153 tokens,
48 fresh request strings of 66–131 tokens, 27,819 tokens total. The token cap of
512 passes without truncation. CPU inventory construction/tokenization/hashing
took 0.211 seconds in this one observation; this is not native feature latency.
The full 216 float32 feature vectors occupy 3,538,944 bytes (3.375 MiB), excluding
Python/serialization/provenance metadata. Each vector is 16,384 bytes.

The frozen prompt format for this first attempt is the exact raw
`catalog_text`/`request_text` string, with `add_special_tokens=False` and no chat
wrapper. Do not quietly substitute the old joint prompt or a generation prompt.
The extraction adapter must compare its actual native tokenizer IDs exactly to
the preflight IDs before each forward. A mismatch stops extraction; it must not
be hidden by merely accepting the same token count.

## Native stage: explicit bounded adapter, still to implement/run

Use one already pinned `FrozenBackbone` from `pretrained/backbone.py` in one
foreground process, `max_length=512`, `memory_gib=6`. This reuses its frozen/eval
model, exact last-valid-token pooling, float32 L2 normalization, no-truncation
guard, and system-memory checks. No LM decode, vision, training, second worker,
decoder-state feature cache, or automatic memory increase is needed.

Use one row per forward, no padding across items, no persistent decoder state,
no native chunking: each complete sequence is at most 512 tokens. Bind actual
`batch_shape=(1, sequence_tokens)` and `chunk_tokens=512` (whole-sequence cap) in
each `EncoderSpec`. This fixes extraction configuration per item independently
of catalog/candidate order. Both fresh controls and cache rebuilds must use the
same shape. Do not claim `FrozenBackbone.encode` implements chunked prefill; it
currently executes a whole-sequence forward.

Retain revision, actual verified weight-manifest digest, hashes of every
tokenizer input actually used (`tokenizer.json`, `tokenizer_config.json`, and
`vocab.json` if used), pinned chat-template hash, raw prompt/input source hash,
library versions, role, exact text/tokens, content version, raw key, vector SHA,
timings and memory per row. A canonical hash of the verified shard-name/SHA map
can be `weights_sha256`; document its construction. A canonical hash of the
tokenizer-file/SHA map can be `tokenizer_sha256`. Derive per-variant content
version from its canonical public product/variant facts, excluding IDs. Request
versions bind the fresh public request text, not identity or labels.

1. Reserve a new run directory with exclusive creation, for example
   `data/ranker/pretrained/catalog-user-fusion-v1-attempt-1/`. Freeze configuration
   and source/input hashes before outcomes. Preserve partial/failed receipts.
2. Hash/tokenize before model load; run a deterministic first catalog row and
   first training request as the pilot. Stop on any dimension, finite-value,
   normalization, token-ID, memory or 120-second pilot-budget failure. The budget
   is an operational cap, not a predicted duration.
3. Finish **only training** catalog members (96 variants) and 24 request features
   first: 120 unique forwards including pilot rows, 1,966,080 raw feature bytes
   (1.875 MiB). Catalog membership comes from public training candidates, never
   private family assignments. If the pilot is cached, do not duplicate its
   forward in this count. Stop this stage at 600 seconds including its model
   load/hash/extraction accounting; retain uncompleted membership explicitly.
4. Save immutable vectors and receipts; verify saved bytes with `check_features`.
   Refuse incomplete, stale, nonfinite, malformed, wrong-dimensional, unknown-
   representation or wrongly mapped entries. Assemble request-specific catalog
   row views by exact `(productId, variantId)` keys in `roster` order, not slots.
5. Unload the native model before the CPU fit. It is not needed for optimization.
   Fit/freeze both CPU checkpoints before reading held-out outcome labels.
6. A later bounded resident extraction adds the remaining 72 catalog variants
   and 24 queries (6 validation, 12 test, 6 diagnostic): 96 additional unique
   forwards. Keep its startup cost separate if the model was unloaded. Do not
   represent all 216 forwards as a single warm-resident extraction when they
   were split across sessions. A 900-second total extraction work budget can be
   used for this first attempt; stop and report incomplete features rather than
   increasing the guard or relabeling partial extraction as success.

There is no empirical native duration for these new inputs. B=1 and ≤153-token
catalog rows reduce the operation's shape relative to previous long joint
prefixes, but that is not a wall-time estimate. MLX active/peak/cache bytes,
system available memory, RSS and total wall time must be recorded in the native
run. Existing `FrozenBackbone` refuses load with <6 GiB available and stops with
<512 MiB available; keep those guards. If 6 GiB active allocation is insufficient,
stop this attempt and report it rather than modifying another lane's limits.

## CPU fit: one fixed attempt on genuine features

Use `read_training_inputs` for the two literal training files. Verify their
hashes against the frozen run config and exact feature membership before
optimization. This loader opens neither validation/test labels nor manifest
private family records. `training_loss` is already the proposed mean-variant
best-product CE plus compatible informative within-product presentation CE.

Freeze this initial hypothesis before any training outcomes: rank 4, dimension
4096, seed 728, AdamW learning rate .03, weight decay .001, 80 full-batch epochs,
one CPU thread, at most 60 seconds **per fit arm**. This deliberately matches the
old corrective-fit schedule as a first comparison; no schedule adequacy claim
is implied. Do not tune these values on validation/test. Save each epoch's loss,
gradient norm and elapsed time, first finite nonzero gradient, parameter delta,
initial/final predictions, train labels/input hashes and checkpoints even if the
fit fails or quality is poor. Stop on nonfinite loss/gradients, zero interaction
updates, timing cap, or receipt mismatch.

Fit two new checkpoints from the same deterministic initialization:

- Full fusion, all 32,786 parameters trainable using the existing objective.
- Item-prior-only control: set/freeze interaction weights to zero and freeze U;
  train V/prior/bias only (16,394 trainable parameters). This is a separately
  trained control, not post-hoc ablation of the full fitted checkpoint.

Also retain the unchanged random initialized fusion scores. Full-fusion weight,
gradient and Adam moment tensors total about 524,576 bytes before framework,
activation and optimizer overhead. Cached training features are under 2 MiB;
loading repeated per-request catalog views would add about 3 MiB. These are
tensor-byte calculations, not total-process peak measurements or a fit-duration
claim. Record real process RSS and CPU fit wall time.

Save checkpoints with exclusive creation; reload using the checksum and require
`torch.equal` predictions on every training request. Do not call a random
checkpoint, synthetic test update, or reused joint checkpoint a fitted fusion.

## Readout comparison after checkpoint freeze

Report rows for train, validation replay, test replay and diagnostic replay,
keeping validation/test outcome files inaccessible during optimization. The
existing `labels/` directory provides only train and validation labels. The
repository's `pretrained/cli.py` already implements the evaluator boundary:
save fixed checkpoints first, then open `private/truth.jsonl` for test and
`private/diagnostic-truth.jsonl` for diagnostic replay. Follow that same ordering
in a separate evaluator process and select only records belonging to the public
evaluation split. Use `bestProductId`, `relevance`, `presentationScores` and
diagnostic pair IDs as evaluator targets; keep latent/private fields out of
encoder construction, cache keys and optimization. Hash the evaluator input
files in the evaluation receipt. Do not invent labels or treat prediction
confidence as accuracy.

For each labeled split compare: random separated fusion; independently trained
item prior; trained fusion; same trained fusion with deterministic cyclic query
permutation; existing corrected joint scorer on its actual joint readouts.
Apply current deterministic guards consistently, and report raw/unmasked ranks
as well as guarded decisions. Existing joint train readouts can be reused as
the **joint reference**; held-out joint readouts need their own correctly shaped
native extraction or already preserved genuine baseline predictions. They
cannot be reconstructed from separated embeddings.

Metrics: best-product accuracy, NDCG@3, all-candidate ranks/margins/ties, per-
product compatible image choices, informative-image denominators, image quality
on oracle-product and model-selected-product sets separately, and abstentions.
Use the same train-only objective to report train/validation losses when labels
are available; do not optimize on those replay losses. Stronger diagnostic
preference changes must affect appropriate choices when labels require it.

Compare cached vectors to their immutable saved bytes, then repeat a bounded
fresh control at matched native shape. Exact file equality and numerical fresh-
extraction sensitivity are separate measurements. Measure changed-query/A-B-A
and one changed-product invalidation before claiming reusable catalog behavior.

The fixture has at most four candidates per request. It cannot establish
12-/24-card ranking/image quality. The existing twelve-/twenty-four-card CPU
mapping tests establish assembly contracts only. A sealed new family/SKU-disjoint
full-grid labeled set and fresh-copy evaluation remain promotion prerequisites.
Keep old joint metrics and known copy failures visible; no trained-vector result
alone establishes a complete successful storefront.
