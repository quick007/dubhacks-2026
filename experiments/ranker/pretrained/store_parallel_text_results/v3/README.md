# Public-fact pointer v3 archive

This archive preserves both CPU research attempts, including every raw field, attention receipt, failed output and exact run source. Lossless compression keeps the roughly 31.7 MiB of run text within a small archive. Local checkpoints are excluded; their hashes remain in each inventory. The original run directories remain byte unchanged.

| Origin | Preserved result | Control |
| --- | --- | --- |
| [Attempt1](store-parallel-text-pointer-v3-attempt-1/results.json) | Role-position: 102/126 heldout card fields;15/42 first3 and6/42 full6. Lexical-pointer: 98/126;6/42 first3 and0/42 full6. | Arm1 evaluation preceded arm2 fit. Read the preserved [clarification](store-parallel-text-pointer-v3-attempt-1/PROVENANCE_CLARIFICATION.json) alongside the original per-arm freeze receipts. |
| [Attempt2](store-parallel-text-pointer-v3-attempt-2/results.json) | Same predictions and checkpoint bytes; both arms 402/402 TRAIN exact. All failures remain available. | [Both checkpoints frozen](store-parallel-text-pointer-v3-attempt-2/ALL_HEADS_FROZEN.json) before any heldout target text was decoded. |

These are the original 34 TRAIN and 14 heldout product-color bindings from the known 16-product catalog. The 42 heldout views recombine those 14 bindings. Product identities, prices and the finite 107-word vocabulary are TRAIN-seen. Public name/color/price words enter a learned pointer/vocabulary decoder alongside cached Qwen features. This changes the input representation from earlier pooled-feature heads. Neither attempt supplies fresh-query latency, new-catalog evidence or a live promotion.

The [independent audit](../../STORE_PARALLEL_TEXT_POINTER_V3_AUDIT.md) separates source-factual short captions from exact teacher wording and narrow gates. It finds three additional useful lexical-arm fields beyond the frozen gates, while incomplete color names, repetitive bodies and typed-role mistakes remain failures. These offline judgments do not alter the archived counts.

## Layout and recovery

Each origin directory contains byte-identical readable `results.json`, an `inventory.json`, `source/store_parallel_text_pointer_v3.py` matching that run's pinned source hash, readable chronology controls, and `artifacts.tar.xz`. Each original JSON/source file appears unchanged at `run/<relative-path>` inside the archive, including per-arm `HEAD_FROZEN.json`. Attempt2 also contains both `TRAIN-fit.json` files. Attempt1's earlier source remains distinct from the corrected two-phase source in attempt2.

`inputs.tar.gz` and its inventory preserve the frozen packet, separately stored TRAIN/eval labels and public catalog snapshot for interpretation. These are archival evidence, not serving exports. Frozen shared dependency sources remain in the [v1 archive](../v1/README.md); its manifest hash is bound here.

```sh
tar -xOf experiments/ranker/pretrained/store_parallel_text_results/v3/\
store-parallel-text-pointer-v3-attempt-2/artifacts.tar.xz \
run/typed-role-position-pointer800/fields.json
```

Every compressed member was reopened and verified against its original SHA-256. All 37 observed original files, including four excluded checkpoints, were rechecked unchanged. The [manifest](manifest.json) binds the archives and inventories. Later runs belong under separate origin IDs; preserve these bytes and their provenance caveats.
