# Current AR comparison preflight

CPU-only preparation completed against the final audited driver. No native comparison was run, no service was stopped and no model or GPU work was started. Earlier comparison preflights v1–v3 remain preserved; v2 and v3 record intermediate stages of the bounded correctness fixes below.

Current preflight SHA-256: `686d09de3f961901e0f50a88f18acd131fbe7f265c315a31116ddaebaced77a3`.

Driver SHA-256: `332112e55b14543b642ea5a926ce1e05b37c62ffe311554df79ba548db803574`.

Driver tests SHA-256: `a08ce0d9145b171676f7d76a451ea86db2149e95ae8b1841be76f576affd950a`.

The source inventory now includes the reused deadline implementation in `store_copy_fit_v2.py`. Every recorded runtime source hash matches current bytes. Complete conventional prompts remain 1334, 1335 and 1334 tokens for the same water, bedside-light and notebook requests, with all sixteen catalog products, six cards and twenty copy fields. The conventional baseline helper and its existing baseline preflight v3 are unchanged; its source inventory does not include this comparison driver.

The audit fixed three concrete driver issues:

- The conventional helper checked its time budget only after a generated response. The driver now wraps the entire conventional call with the existing 120-second main-thread alarm guard, covering prefill and `next()` before the first response.
- The hybrid engine's decode-loop budget starts after ranking/root preparation. The standalone driver now applies the same whole-call alarm guard to its hybrid arm. This does not change the live service or its thread behavior.
- Failed comparison/shutdown statuses previously printed a failed result but exited zero. Explicit native `--run` failures now exit one; successful comparisons and CPU-only preflights retain normal successful exits.

The tests inject SIGALRM before the conventional first response and during hybrid preparation, verifying failure receipts, restoration of the previous signal handler, cleared alarms, engine shutdown and native-lock release. Separate mocked CLI tests check failed comparison, failed shutdown and failed shutdown-integrity statuses without executing native work. All **24 baseline/parser/driver CPU tests pass**:

```sh
.tools/pretrained-venv/bin/python -B -m unittest \
  experiments.ranker.pretrained.test_store_autoregressive_baseline \
  experiments.ranker.pretrained.test_store_ar_comparison_run_v1
```

The conventional frozen-base path disables the installed copy adapter before creating a fresh request cache and streaming generation, then restores it. The hybrid arm retains its learned rank head/adapted independent copy. Existing same-width fork/rebuild/repeat controls, frozen927/checkpoint shutdown checks and native-owner locking remain in place. CPU tests and this preflight do not replace those future native proofs.

Both arms retain the same caller request-start timer and validated/serialized first3/full6 endpoints. Setup, controls, warmup and later archive I/O remain separate. Accepted title-and-body counts are explicitly a copy proxy; they do not prove product relevance, human usefulness or browser readiness. No speedup claim is supported until both measured results and quality evidence exist. The main-thread signal guard is tested at Python signal-delivery boundaries; actual Metal/C execution interruption has not been measured.

Reproduce CPU-only preparation into a new unused directory:

```sh
.tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.store_ar_comparison_run_v1 \
  --out NEW_UNUSED_PREFLIGHT_DIRECTORY \
  --manifest experiments/ranker/pretrained/store_demo_integration_v3/manifest.json
```

Native execution still requires the root's explicit orchestration, resolution of any reported approval blocker, a gracefully stopped demo service and the exclusive shared worker lock. This audit does not authorize stopping the service or claim that the native benchmark completed.
