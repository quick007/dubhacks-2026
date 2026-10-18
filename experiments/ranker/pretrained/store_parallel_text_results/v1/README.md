# Parallel-text results archive

This archive preserves six research, curated-demo and fresh-benchmark origins, including failures. It contains about **2.0 MiB** of lossless compressed receipts, readable result summaries, exact historical source snapshots and SHA-256 inventories, representing about16.8 MiB of original run text. All source directories remain intact. No checkpoint weights or feature arrays were copied.

| Origin | Preserved result | Interpretation |
| --- | --- | --- |
| [Native v1](store-parallel-text-v1-attempt-1/results.json) | first3 10/32; full6 1/32; attempted full6 median197.05 ms | Only generic sections pass among210 heldout label rows; all126 heldout card fields fail. One complete202.09 ms view uses TRAIN-seen bindings. |
| [All-catalog TRAIN160](store-parallel-text-train-all-demo-v1-attempt-1/results.json) | 526/528 supervised fields pass | Two repetitive bodies remain preserved; all48 bindings declared TRAIN. CPU replay only. |
| [All-catalog TRAIN320](store-parallel-text-train-all-demo-v1-attempt-2/results.json) | 528/528 supervised fields;96/96 old-card/style views pass | Fully supervised catalog demo. CPU replay supplies no fresh-query or heldout evidence. |
| [CPU research follow-ups](store-parallel-text-cpu-v2-attempt-1/results.json) | Three separate normalization/shared-slot arms; each0/42 complete heldout views | Raw fields and failed generalization preserved; no candidate promoted. |
| [Fresh benchmark1](store-parallel-text-train-all-fresh-v1-attempt-1/results.json) | Overall **failed-benchmark**;32 valid main requests,8 style sets and5 completed guards | Later contradictory-color guard raises an uncaught safe-refusal exception. Main payloads remain available; failure status remains unchanged. |
| [Fresh benchmark2](store-parallel-text-train-all-fresh-v1-attempt-2/results.json) |32/32 first3/full6; median193.43/193.50 ms;8 guard probes pass | Separate successful attempt with expected contradictory-color rejection recorded. All-catalog TRAIN scope remains explicit. |

Fresh benchmark2 has30/32 offline primary-function top1 matches and31/32 targets present in six. The notebook query misses entirely; the blanket query ranks a towel first with the throw third. Copy remains neutral selected-product facts plus explicit style, rather than shopper-intent prose. Budget and unknown-fact probes exercise deterministic guards, not learned constraint reasoning. The32 queries are curated development, with disclosed prior training-wording overlap. Backend timing excludes startup, offline feature extraction/fitting, network, React and images; there is no browser-paint or conversion measurement here.

Independent quality evidence remains at its original repository paths:

- [Research data audit](../../STORE_PARALLEL_TEXT_V1_INDEPENDENT_AUDIT.md): positional supervision gaps and useful wording versus narrow grammar.
- [Native v1 audit](../../STORE_PARALLEL_TEXT_V1_NATIVE_AUDIT.md): all640 frontier fields and210 heldout label rows, with failures preserved.
- [TRAIN-ALL320 audit](../../STORE_PARALLEL_TEXT_TRAIN_ALL_DEMO_V1_AUDIT.md):528 outputs,1920 replay fields, CPU reproduction with teacher construction disabled and fact-change invalidation.
- [Successful fresh-run audit](../../STORE_PARALLEL_TEXT_TRAIN_ALL_FRESH_V1_AUDIT.md):640 main fields,480 style fields, eight guard receipts and separate ranking misses.

## Layout and recovery

Each origin-ID directory contains:

- `results.json`: byte-identical readable source result, including its status.
- `inventory.json`: original relative paths, byte lengths and SHA-256 values, excluded binary-file hashes, archive checksum and exact historical source provenance.
- `artifacts.tar.gz`: every original JSON/text/source receipt, at `run/<original-relative-path>` inside the archive. Original contents and formatting are unchanged; only tar metadata is normalized.
- `source/`: readable source files matching that run's pinned hashes. The failed first fresh attempt uses its preserved old benchmark source; the second uses the distinct corrected source.

`inputs.tar.gz` and `inputs-inventory.json` preserve the parallel-text packet, frontier packet and public catalog snapshot needed to interpret source bindings and offline labels. These are archival evidence, not serving inputs. The label boundary remains unchanged.

To inspect a compressed receipt without modifying the archive:

```sh
tar -xOf experiments/ranker/pretrained/store_parallel_text_results/v1/\
store-parallel-text-v1-attempt-1/artifacts.tar.gz run/request-011.json
```

Every compressed member was reopened and verified against its original byte hash. All320 observed original run/source/input files, including locally retained excluded binaries, were rechecked unchanged after copying. Historical source copies match pinned run hashes. The [top-level manifest](manifest.json) binds each origin archive and inventory. Keep later attempts under separate origin IDs or archive versions; preserve these bytes and failure statuses.
