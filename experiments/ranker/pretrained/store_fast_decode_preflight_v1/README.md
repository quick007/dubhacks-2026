# Fresh-copy speed candidate

Prepared CPU-only; no candidate GPU execution has occurred. Eleven cache and
driver tests pass. The live engine, frontend, installed MLX-LM code, checkpoints
and earlier profile modules are unchanged.

Preflight SHA-256:
`2231f2e345a072a5e3d5d2eef7d6bb32dacc65c822e9a50cef821b23862c4dbc`

The candidate combines width 20 with exact immutable common-prompt state reuse,
compared against width 10 with a fresh common root. It preserves six products,
all 20 independent native fields and the original 416-token summed cap. It
caches no generated text. Every title, badge, body and heading is freshly
sampled from a new independent field fork.

The cache key binds the verified model file/revision receipt, exact resident
model/tokenizer objects, adapter checkpoint, actual installed native LoRA A/B
tensor bytes, tokenizer receipt, complete common message and exact root token
IDs. A consumed or mutated source fails without repair. The cache holds only
one entry and checks the caller's existing active/peak memory guard; it never
raises that guard. Other public requests can share the state only because the
common copy prompt is exactly neutral and independent of their ranking query.
All selected public product facts still enter the fresh per-field suffixes.

The standalone driver requires explicit `--run`, unused output and exclusive
native ownership. It refuses a listening service and never stops it itself.
It uses one resident engine and fixed water/light/notebook AB/BA/AB ordering.
It performs same-width fork/rebuild/repeat controls for widths 10 and 20 outside
timing. It checks width-20 miss/hit exact native token parity and a width-20
A–B–A return after other shopper requests. Width-10 and width-20 tokens may
differ; their quality, complete field counts, EOS, actual tokens and readiness
are reported separately. Learned ranking/action bindings must stay identical.

Initial native memory policy is explicitly 7 GiB, with a real available-memory
check before model load. A control or request exceeding the guard is preserved
as a failure; there is no automatic budget increase. Shutdown checks the 927
frozen tensors and adapter checkpoint before releasing the lock.

The authorized root owner can launch after its other GPU work ends and the
service is stopped gracefully:

```sh
.tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.store_fast_decode_run_v1 \
  --run --out data/ranker/pretrained/store-fast-decode-v1-attempt-1 \
  --manifest experiments/ranker/pretrained/store_demo_integration_v3/manifest.json
```

This is a combined initial candidate. It cannot independently attribute a
speedup to width versus root reuse. Previously measured root construction was
about 0.49 seconds; no new speedup or quality result is claimed from CPU tests.
