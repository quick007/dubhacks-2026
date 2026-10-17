# Exact catalog-first input-state cache

This bounded experiment caches complete native decoder input state at an exact stable catalog boundary, then appends observable user context and the request's grounded page plan. It does not cache completed responses or precompute generated pages. The trained user-first ranker remains unchanged and is recomputed on every request; its cache is discarded before the decoder cache is materialized. This extra prefill is included in full latency. No differently ordered hidden readout is passed to old trained heads.

Installed MLX-LM 0.32.2 native serialization includes `ArraysCache` recurrent/convolution/padding/length state and KV tensors/offsets. Immutable safetensors snapshots are loaded as complete native objects per request. Exact tokenizer prefix identity is verified before single-use append. No independently cached user state, arbitrary trimming, cache splicing or speculative decoding is used. A tiny native roundtrip test compares fresh full execution to restored catalog-plus-suffix execution for A→B→A.

Catalog keys bind model revision, runtime, exact token prefix and relevant catalog facts/IDs. Request keys additionally bind the observable profile, generated-plan prompt, trained-head checkpoint and decoding configuration. Changed profiles reuse only catalog input state and regenerate. Changed catalog facts invalidate the catalog key. Snapshots are bounded to two; weights/cache snapshots stay ignored on local disk.

```sh
.tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_catalog_cache -v
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.catalog_cache --model /absolute/pinned/local/snapshot
```

The fixed probe includes 1-card cold input, warm catalog, repeated unchanged A, changed preferences B, return to A, and changed product availability; 3/6-card probes compare cold/warm catalog input. Every attempt regenerates all requested headings/titles/badges with identical per-size budgets. Timing includes ranker input, cache SHA verification/materialization/disk I/O/build/save, forced MLX evaluation, fresh decoder prefill, generation and validation. One sample per condition is not p95 evidence. The compact catalog-first decoder prompt differs from the earlier evidence-plan comparison; only within-probe cold/warm comparisons isolate input caching.

Actual 9B measurements completed; [CATALOG_CACHE_RESULTS.json](CATALOG_CACHE_RESULTS.json) includes all ten conditions, failures, stage timings, hashes and checks. Cache hits occurred on 6/10 attempts. Catalog state snapshots were about 85.1 MB, bounded to two. Warm one-card decoder token coverage was 72.0%, but coverage across all input work was only 34.1% because the trained ranker still prefills 1446 tokens every call.

| Cards | Cold catalog full attempt | Warm catalog full attempt | Cold catalog prepare | Warm prepare |
| ----- | ------------------------: | ------------------------: | -------------------: | -----------: |
| 1     |                   6518 ms |                   5070 ms |              1350 ms |        66 ms |
| 3     |                   9157 ms |                   7963 ms |              1315 ms |        50 ms |
| 6     |                  11289 ms |                   9803 ms |              1370 ms |        37 ms |

Each row instruments actual stage durations; no percentile subtraction is used. Repeated unchanged one-card A took 5020 ms, changed-preference B 5053 ms, return A 5089 ms, and changed catalog availability 6414 ms with a cache miss. A→B→A preserved A's request key and raw greedy generation; B changed the full request key while safely reusing only catalog state. Changed catalog facts changed the prefix key and the unavailable item was excluded. Native tiny-model restored/fresh full-state suffix parity also passed. These establish state/key correctness under the tested controls, not successful personalized wording.

**All ten generated page attempts failed validation; zero accepted new titles and zero fulfilled pages.** The compact decoder prompt produced extra/missing fields and truncation. In particular its broad S1/S2/card-count instructions are a protocol-quality confound; failed output is retained and no architecture conclusion follows. None of the cold/warm timings represents a successful fast page. This prototype duplicates stable-input work across the unchanged trained head path and separately cached decoder; it is slower than the preserved user-first baseline despite reducing decoder-input prefill. The cache probe is a correctness/measurement increment, not a serving recommendation.

Peak MLX allocation was 6,344,673,924 bytes under the 6,442,450,944-byte cap (about 98 MB headroom). No extra GPU cache clones or parallel field generation were attempted. Additional memory would allow testing larger concurrently materialized cache/batch states, but would not establish quality or remove the measured fresh-decoding cost. No paid compute or external machine was used or requested.

After measurement, request-key hardening additionally bound the actual loaded-head receipt and full structural/image plan, and the CLI was restricted to the verified local model snapshot. Four cache checks cover actual model/catalog/profile/head/image invalidation plus complete native A/B/A state parity. The saved measured source revision and post-measurement key-only changes are explicitly distinguished in the result file; no favorable generations replaced failures. The prior roughly three-second single-card baseline remains below the near-traditional-page latency target. Failed pages and stale-personalization checks must be reported alongside cache hit/token coverage and full completion latency; an empty or rejected response is not a speed success.
