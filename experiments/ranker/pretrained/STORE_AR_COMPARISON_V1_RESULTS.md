# Matched native conventional versus hybrid comparison

The three-query native comparison completed on the M5 with six real cards and
20 fresh copy fields per request. Median full attempted completion was **21.199
seconds** for conventional autoregressive generation and **5.937 seconds** for
the hybrid. The ratio is **3.57×** for attempted completion. Neither system
passed the frozen complete title/body readiness proxy, so this is not a measured
useful-page speedup.

| Warm resident endpoint | Conventional A | Hybrid B |
|---|---:|---:|
| Timed requests | 3 | 3 |
| Median first-three attempted completion | 11.350 s | 5.555 s |
| Median full-six attempted completion | 21.199 s | 5.937 s |
| Frozen useful first3 / full6 passes | 0/3; 0/3 | 0/3; 0/3 |
| Independently judged relevant first product | 3/3 | 2/3 |
| Sampled tokens per request, including EOS | 419 / 411 / 425 | 173 / 171 / 178 |

The independent [exhaustive audit](STORE_AR_COMPARISON_V1_INDEPENDENT_AUDIT.md)
reviewed all 120 raw fields and 36 card bindings. Conventional A's 18 body prices
are numerically correct, but all are withheld because the frozen gates reject
currency written as “36 dollars.” Its raw complete titles and bodies would be
usable in all three requests under the offline judgment; this does not change
the observed serialized payload or its null useful-readiness timestamps. Two A
headings misclassify unrelated products, and two badges add unverified subjective
claims. Hybrid B withholds three ambiguous collection/item-price bodies and three
unfinished bodies. Its notebook request selects an organizer and omits the
notebook from the first six. No accepted hybrid field had a wrong name, color or
price in this audit. These inspected development queries do not establish broad
personalization or unseen generalization.

## Workload and timing

Both systems used the same warm resident frozen Qwen3.5-9B four-bit backbone,
full public 16-product catalog, and public water, bedside-light and notebook
queries. A is one ordinary autoregressive JSON stream making its own product
choices and copy. B uses the trained actual-store v3 ranking head and adapted
copy-v2 generation, batching 20 independent autoregressive fields at width10.
The fixed order is AB, BA, AB. B copy sees selected facts with a neutral request;
A receives the shopper query, catalog and a neutral-copy instruction. No private
shopper truth enters either path.

The caller passes the same entry `perf_counter` boundary to both systems.
First3 means serialized complete first-section attempt; full6 means serialized
complete payload attempt. These are native service-path measurements, excluding
HTTP transport, React rendering, physical paint and image loading. Both arms have
fresh request caches; allocator caches are cleared between arms. Startup,
controls and one complete warmup per system are excluded and recorded separately.
There is no cached-response or generated-template substitution.

This compares two complete systems. Training, prompts, selected products, token
counts and execution differ; the ratio cannot isolate the effect of batching.
There are only three narrow queries, without budgets or hard constraints. Existing
budget and exact-name tests remain separate evidence. The 124 ms synthetic rank
replay and roughly 200 ms actual-store ranking remain ranking-only measurements.

## Execution and preservation

All six timed attempts completed without execution failures. Engine construction
took 9.456 seconds, excluding process imports and lock wait. Exact frozen-base
and adapted width10 cache fork/rebuild/repeat controls passed; their 13.653 and
14.160 seconds are excluded. Unscored complete warmups took 21.064 and 5.585
seconds. The entire execution including setup, controls and warmup took149.162
seconds. Peak observed MLX allocation was 6,592,642,354 bytes under the configured
7 GiB guard. Shutdown confirms all 927 frozen tensor hashes and the adapter file
unchanged.

The checked-in [native archive](store_ar_comparison_results/v1/results.json)
contains raw outputs, per-request receipts, controls, preflight, source snapshots,
launch log and lifecycle receipts. Its [manifest](store_ar_comparison_results/v1/archive-manifest.json)
hashes all 41 preserved files. Original local data and checkpoints remain intact;
no weights or feature arrays are committed. Driver/deadline/failure handling had
24 CPU tests passing before this native run. No deployment occurred.

After the exclusive run, the local service was restored at
`http://127.0.0.1:8766`, output `data/ranker/pretrained/store-demo-v5-post-comparison-1`.
Its health reported ready, zero queued requests and 927 verified source tensors;
the storefront at `http://127.0.0.1:4312/?model=local` returned HTTP200. The separate
shopper task received the timing release for its real Sol screenshot smoke. It
owns shopper orchestration and results. Future currency-parser changes must be
versioned prospectively; these results and gate decisions remain frozen.
