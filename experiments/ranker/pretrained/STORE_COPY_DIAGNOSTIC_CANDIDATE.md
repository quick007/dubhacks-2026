# Actual-store six-card copy diagnostic candidate

Prepared during the authorized local model continuation. **CPU tested; no actual-store native copy or ranking execution has occurred.** Migration paused this lane at the active short-copy experiment boundary. This is candidate implementation, not quality or latency evidence.

Files: `store_copy_diagnostic.py` and `test_store_copy_diagnostic.py`. Ten CPU tests pass with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest experiments.ranker.pretrained.test_store_copy_diagnostic -v
```

The candidate consumes the completed actual-store `store_function_adaptation.py` report and the frozen public-case file. It selects exactly `store-eval-00` through `store-eval-05`, the first three predetermined function preference pairs. Product choices come from `results.evaluation.learned.cases[caseId].topSix.productIndices`, with all 16 `scores` retained. Targets never choose the copy case or products. Six distinct products are required; shortfall or unsupported hard-requirement cases remain not-attemptable in the fixed six-case denominator.

`prepare` checks the completed learned ranking report, exact current store input SHA, public-case SHA and learned-head checkpoint SHA. It requires the completed 240-update copy report, exact checkpoint reload, original archived native gradient replay, unchanged 927 base-language tensor proof and source/data checks. `run` rechecks sources/checkpoints and recomputes the complete bound decisions from the ranking receipt. Preparation cannot run successfully until those native prerequisite reports exist.

The native copy path uses a fresh ordinary native chat root through the unchanged `branch_local_copy.prepare_root`. It never shares the incompatible focused-seven ranking prefix or claims feature/cache equivalence. The full public request goes into the common message; each field receives only its own bound facts. This differs from the short-fit raw-prefix training distribution, so transfer quality remains unproven.

Each request has 20 independent S/T/B/C fields, width 6, continuous refill, the original per-field caps and 416 sampled-token total cap. The first ten fields complete a real three-card section. A callback validates and UTF-8 serializes that payload while below-fold fields can continue. Full six-card completion has its own validation/serialization timestamp. Neither endpoint measures browser rendering. Startup controls compare reused complete caches, independently rebuilt complete caches and repeats at the same width 6, hashing every full native probability vector and token/EOS response with original caps. Any mismatch stops the native diagnostic and preserves evidence.

Only name/category/item price/selected color support generated facts. The validator is a new prospective **conservative lexical gate**, not comprehensive semantic proof. It checks selected identity, new wording relative to literal source fields, exact field membership, plain text, unsupported vocabulary/numbers, strict sampled cap/EOS receipts, and unknown stock/size/shipping. Badges require generated `NONE`; it never injects missing values. It does not establish preference-sensitive copy quality or paraphrase originality. Historical validators and outputs remain unchanged.

Images/colors are deterministically the first declared color and first declared image, with exact external action bindings. This is untrained metadata selection; no pixels are read. The standalone CLI measures locked-decision copy only. It does not yet run fresh ranking inside each timed request. The callable `run_case(bb, bound, request_started=None)` supports a later integrated owner: passing an earlier `time.perf_counter()` start includes the caller's fresh rank phase in callback timestamps, but the caller must actually execute and document that phase.

After migration, first inspect active workers and complete/audit the short-copy job, then run the actual-store rank experiment with its frozen packet. Only once both prerequisite reports show completion:

```sh
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.store_copy_diagnostic prepare \
  --rank-report data/ranker/pretrained/store-function-adaptation-v1-attempt-1/report.json \
  --public-cases experiments/ranker/pretrained/store_function_v1/public-cases.json \
  --copy-report data/ranker/pretrained/short-copy-fit-v1/results.json \
  --out experiments/ranker/pretrained/store_copy_prepare_v1
```

Root alone may subsequently run the bounded native diagnostic, with no other owned GPU worker:

```sh
PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.store_copy_diagnostic run \
  --packet experiments/ranker/pretrained/store_copy_prepare_v1/packet.json \
  --out data/ranker/pretrained/store-copy-diagnostic-v1-attempt-1
```

Outstanding native checks include actual token lengths, same-shape cache/distribution controls, adapter transfer quality, real output examples, accepted fields/pages, memory and actual payload endpoint timings. No tokenizer preflight or native call was executed in this lane. A failed suffix/context guard must preserve the failed attempt; changing caps or prompts requires a new preparation/version and explicit documentation.
