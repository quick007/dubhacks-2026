# Native copy component profile

The bounded profile completed on the same frozen Qwen3.5-9B backbone, current
ranking head and copy adapter. All three paired requests preserve exact product
bindings, every sampled token and EOS. Ordinary warm resident full-six attempts
have a median **5.580 seconds**; the synchronized observer takes **5.946 seconds**.
This is a diagnostic, not an optimized model or a new useful-page speedup.

| Instrumented component | Median |
|---|---:|
| Fresh 328-token common root |494.4ms|
| Suffix prefill |1969.8ms|
| Initial and subsequent autoregressive steps |2517.9ms|
| Cache merge, extend, split and filter operations |562.5ms|
| Per-field root cloning/materialization |3.4ms|
| Separately measured phase pre-boundary synchronization waits |5.1ms|

These medians describe separate statistics, not an additive trace. Nested
inclusive parent times must not be added to child times. The native phase table
uses exclusive durations; pre-boundary waits are recorded separately. The cache
aggregate excludes the separately measured initial per-field clones. Raw events,
per-case values and definitions are in the [phase summary](store_demo_performance_results/v1/phase-summary.json)
and [native receipts](store_demo_performance_results/v1/results.json).

The observer explicitly synchronizes phases and materializes lazy cache operations.
It changes overlap and token lookahead. Paired full-attempt overhead is273.6,
379.2 and365.4ms, with a median365.4ms (6.55% of the ordinary median). The5.1ms
pre-boundary waits are not the entire observer overhead. This is synchronized
native host-phase attribution, not an unperturbed GPU kernel profile.

## Workload and controls

The same three inspected water, light and notebook development queries used
baseline/profile, profile/baseline, baseline/profile order. One complete baseline
warmup is excluded. Each request retains six real cards,20 freshly generated
fields and the416-token cap. No templates, cached text or shortened roster enter
these runs. Startup, imports, model verification and archive writes are separate
from serialized request readiness; images, HTTP transport and React are excluded.

| Native serialized endpoint | Ordinary baseline median | Profile median |
|---|---:|---:|
| Rank |180.3ms|171.3ms|
| First-three attempted completion |5122.5ms|5479.8ms|
| Full-six attempted completion |5580.1ms|5945.5ms|

These are three requests per arm, with no failed native attempts. Each profiled
full payload accepts18/20 fields under the unchanged practical gate. Completion
can still contain withheld fields; no complete useful-copy latency is established.
The earlier valid-dollar scoring issue and notebook ranking miss remain separate
quality evidence. Differences from earlier runs are not a claimed optimization:
source, model and generation settings did not change, and system conditions vary.

Initial AR steps are included, rather than lost inside generation-batch construction:
water38steps =32outerrounds+6transitions; light40=32+8; notebook36=28+8.
Returned sampled-token totals173/171/178 sum across independent rows, so they are
not the number of batched model forwards. The installed generator performs
lookahead as well. No per-token hardware-throughput claim follows from dividing
these totals. All event sums and full raw token/binding signatures were independently
recomputed in a read-only review; timing arithmetic and initial-step coverage pass.

Shutdown verifies all927 frozen source tensors and the adapter unchanged. Peak
MLX allocation was6,622,274,410bytes under the7GiB guard. Driver/observer sources
are unchanged from preflight. The original data remains under
`data/ranker/pretrained/store-demo-performance-v1-attempt-2`; the checked-in
[archive manifest](store_demo_performance_results/v1/archive-manifest.json) hashes34
files, including source snapshots, launch records and original lifecycle receipts.
The first launcher attempt failed before preflight or model load because its
Python symlink resolved outside the virtual environment. Its error log and launch
receipt are preserved separately; the corrected launch used the venv path directly.

## Implication and current service

Per-field cloning and explicit outer sync are small costs here. Suffix prefill
and repeated9B AR steps are the largest measured targets. Common-root reuse can
remove work represented by roughly0.49seconds, subject to independent cache parity;
it cannot remove the remaining seconds by itself. Width20 and compact fresh text
remain bounded next comparisons. Generated-earlier copy caching and the parallel
text head are explicitly different queued conditions with no new native latency
result. The low-hundreds-of-milliseconds target remains unachieved.

Major architecture work remains queued per the user's timing instruction.
The unchanged width10 service was restored as PID4452, output
`data/ranker/pretrained/store-demo-v6-post-profile-1`; `/health` reports ready,
zero queued requests and927 verified source tensors. Storefront styling and
serving code are unchanged. No paid compute, download, install or deployment occurred.
