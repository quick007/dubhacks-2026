# Explicit 8 GiB candidate preparation

CPU-only preparation completed; **16 cache/driver tests pass**. This task did
not launch GPU work. Earlier preflight v1 and the partial width-20 failure
remain preserved.

Preflight SHA-256:
`4772679729d59a554b11d053ac7b550256fad40bcc76dbdaa434083b36c7c619`

`store_fast_decode_run_v1` still defaults to `--memory-gib 7`. The new explicit
`--memory-gib 8` option never activates automatically after a failure. It
constructs the unchanged `ResidentDemoEngine` at its verified default 7 GiB,
then checks real headroom before changing the caller-owned backbone budget
and MLX allocator limit to exactly **8,589,934,592 bytes**. Both compared arms
use the same requested budget.

For explicit 8 GiB, the driver requires at least 10 GiB available before model
load: the requested 8 GiB plus a 2 GiB system reserve. After default model
initialization, it requires available bytes of at least
`max(0, 8 GiB − current active native bytes) + 2 GiB`. It records system total,
available bytes, percent used and swap usage before loading and before budget
configuration. An insufficient snapshot fails without changing the allocator.
The parent will inspect actual macOS memory pressure before launching.

The report preserves the initial default-7 startup separately, then records
the explicit runtime configuration, previous allocator limit, real active and
peak allocations, and the 8 GiB budget in subsequent request receipts. The
immutable prompt cache reports the actual caller budget and never changes it.
The live engine source and live service configuration are untouched.

All prior workload and proof requirements remain: six real products, twenty
fresh native fields, 416 summed token cap, same-width controls, width-20 cache
miss/hit and A–B–A token equality, independent cross-width quality assessment,
exclusive ownership and frozen-927/adapter shutdown checks. No generated text
is cached. This is not evidence of a new speedup or a completed 8 GiB run.

After the other native worker finishes, the root owner can use an unused path:

```sh
.tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.store_fast_decode_run_v1 \
  --run --memory-gib 8 --out data/ranker/pretrained/store-fast-decode-8g-v1-attempt-1 \
  --manifest experiments/ranker/pretrained/store_demo_integration_v3/manifest.json
```

The driver refuses an active local service and does not stop or replace one.
No runtime budget may exceed the explicit requested value. A new memory
failure is archived rather than retried with another silent increase.
