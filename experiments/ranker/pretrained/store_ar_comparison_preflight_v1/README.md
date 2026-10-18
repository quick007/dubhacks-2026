# Executable comparison readiness

CPU-only preparation completed. **No native comparison was launched.** The
parent-reported pending approval remains unresolved. The driver does not stop
the live service or resolve that approval; it refuses native execution while
anything is listening on `127.0.0.1:8766`, and also acquires the existing
exclusive native-owner lock before model initialization.

Driver preflight SHA-256:
`a6e75a33057ec706c1dfc0217040ec23a7ec9580d460df526099051a2a590fdb`

The fixed roster is water, bedside light and notebook, in that order, each
with all 16 public catalog products, six real selected cards and 20 fresh copy
fields. The arm order is conventional/hybrid, hybrid/conventional,
conventional/hybrid. Complete conventional prompts are 1334, 1335 and 1334
tokens. No existing copy gate or workload was reduced.

The driver constructs one verified `ResidentDemoEngine` and uses its same
resident Qwen model for both systems. Conventional decoding explicitly disables
the installed copy adapter. Hybrid retains the trained v3 ranking head and
copy v2 adapter. Before timed requests, same-width fork/rebuild/repeat native
cache controls run for frozen and adapted modes, and each complete system
warms up once. Those costs are recorded separately and excluded from the three
timed paired cases. Both systems use fresh request caches, with allocator
cleanup before each arm. The same caller `perf_counter` entry is passed to
both paths through the backwards-compatible optional `request_started`
parameter in the existing conventional helper.

Each arm preserves first-three/full-six attempted readiness, the same
title-and-body usable-copy proxy, all-field acceptance, actual sampled tokens,
raw outputs and failure duration. A failed usefulness proxy has no usable
latency. Outer call times include subsequent archive I/O and are labelled
separately from validated serialized payload readiness. Full frozen-tensor
and checkpoint shutdown verification runs **before releasing native ownership**.
The driver never automatically restarts the service afterward.

Seven driver tests use a fake engine and pass, covering explicit-run gating,
service refusal, one engine, fixed arm order, warmups, failure preservation,
shutdown inside the lock and honest null usable latency. The existing 14
baseline CPU tests also pass.

After the parent resolves the pending approval and explicitly stops the demo
service gracefully, native execution is available with:

```sh
.tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.store_ar_comparison_run_v1 \
  --run --out data/ranker/pretrained/store-ar-comparison-v1-attempt-1 \
  --manifest experiments/ranker/pretrained/store_demo_integration_v3/manifest.json
```

The output path must be unused. Omitting `--run` performs only a new local CPU
preflight and cannot load the model or run GPU work.

The baseline helper's timer parameter changed its source hash, so the current
baseline preparation is preserved separately in
`store_ar_baseline_preflight_v3/preflight.json`, SHA-256
`fdaf4b33bb84f4a1c9693dab03ecf87ebd45831059c5a36adbbcc94ee8a3a40c`.
It retains exactly the original five v3 receipt paths from preflight v2.
Subsequent independent shopper queries are preserved separately and were not
silently added to that fixed smoke-test roster. The historical 0/5 title/body
proxy result applies only to those five receipts, not every future query.
