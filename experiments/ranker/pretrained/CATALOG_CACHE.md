# Exact catalog-first input-state cache

This bounded experiment caches complete native decoder input state at an exact stable catalog boundary, then appends observable user context and the request's grounded page plan. It does not cache completed responses or precompute generated pages. The trained user-first ranker remains unchanged and is recomputed on every request; its cache is discarded before the decoder cache is materialized. This extra prefill is included in full latency. No differently ordered hidden readout is passed to old trained heads.

Installed MLX-LM 0.32.2 native serialization includes `ArraysCache` recurrent/convolution/padding/length state and KV tensors/offsets. Immutable safetensors snapshots are loaded as complete native objects per request. Exact tokenizer prefix identity is verified before single-use append. No independently cached user state, arbitrary trimming, cache splicing or speculative decoding is used. A tiny native roundtrip test compares fresh full execution to restored catalog-plus-suffix execution for A→B→A.

Catalog keys bind model revision, runtime, exact token prefix and relevant catalog facts/IDs. Request keys additionally bind the observable profile, generated-plan prompt, trained-head checkpoint and decoding configuration. Changed profiles reuse only catalog input state and regenerate. Changed catalog facts invalidate the catalog key. Snapshots are bounded to two; weights/cache snapshots stay ignored on local disk.

```sh
.tools/pretrained-venv/bin/python -m unittest experiments.ranker.pretrained.test_catalog_cache -v
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.catalog_cache --model /absolute/pinned/local/snapshot
```

The fixed probe includes 1-card cold input, warm catalog, repeated unchanged A, changed preferences B, return to A, and changed product availability; 3/6-card probes compare cold/warm catalog input. Every attempt regenerates all requested headings/titles/badges with identical per-size budgets. Timing includes ranker input, cache SHA verification/materialization/disk I/O/build/save, forced MLX evaluation, fresh decoder prefill, generation and validation. One sample per condition is not p95 evidence. The compact catalog-first decoder prompt differs from the earlier evidence-plan comparison; only within-probe cold/warm comparisons isolate input caching.

Actual 9B measurements are pending at this implementation increment. The prior roughly three-second single-card baseline remains below the near-traditional-page latency target. Failed pages and stale-personalization checks must be reported alongside cache hit/token coverage and full completion latency; an empty or rejected response is not a speed success.
