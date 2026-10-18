# Actual-store multi-need v4 runner preparation

CPU preparation is complete. The combined runner and dataset suite passed **22 tests in 16.764 seconds**. No native model execution or GPU work was performed. `preflight.json` contains **104 B1 whole-text rows**, with 45–120 tokens per row and no truncation: 16 catalog plus 64 TRAIN requests, followed by 24 evaluation requests. Its source hashes include the new runner and tests; the frozen v4 data builder/tests remain unchanged.

The two 4096-to-16 affine heads retain 131,104 trainable parameters. Query probabilities use 16 independent sigmoids; catalog probabilities use a 16-class softmax. The fixed 80-update learned fit averages flat desired-function BCE and catalog CE. Its targets use TRAIN `desiredProductIndices` before eligibility; a budget-blocked desired product remains semantically desired. A separate query-independent 16-logit prior gets exactly 80 BCE updates. Seed 728, learning rate 0.03 and weight decay 0.001 match v3. Flat zero labels mean unrequested in the synthetic case, not universal irrelevance; no negative reweighting or tuning was added.

The semantic compatibility threshold is fixed at **0.5 before evaluation**. Guards apply before selection; the runner returns up to six passing products and discloses shortfalls, with no padding. It also preserves all 16 unthresholded scores/ranks, the raw top six and the guarded unthresholded top six. Semantic strict, exact-name policy and unthresholded guarded ranking metrics are reported separately. The exact-name path recognizes an entire current-query identity request (original name or anchored find/show forms); stale searches cannot redirect it, and hard guards still apply. Incidental, negative and multiple-name mentions do not route. This is a deterministic policy, never counted as learned semantic quality.

Learned, prior, random and permuted-query predictions are frozen before evaluation labels are parsed. The fixed counterfactual shuffle uses TRAIN offset 17 and evaluation offset 1, chosen without evaluation labels; it does not guarantee disjoint semantic targets. The complete CPU mock verified 80 TRAIN features before checkpoint freeze, then 24 fresh endpoints. Each endpoint spans fresh query encode, immutable query serialization/verification, 16 verified warm catalog reads, learned scoring, both decision policies and serialization. The audit does not convert artificial-vector timings into model evidence. Generated copy, image decisions, browser rendering and complete useful above-fold/full-page readiness remain excluded.

The CPU run-path tests use explicitly artificial unit vectors, with the native encoder and lock mocked. They verify all 104 inputs are used once, checkpoint reloads, 80+80 updates, all control prediction membership, payload hash replay and durable partial progress. Instrumented reads confirm `evaluation-labels.json` is parsed once, after `PREDICTIONS_FROZEN.json`. Failure receipts cover preparation, verification, postprocessing and cleanup; a timeout after the first returned forward reports one completed forward and zero serialized features. Existing output directories are preserved. The shared native-owner lock and CLI remain root-only. The deadline is cooperative.

Root should review and commit the source, tests and this frozen packet before launch. A native run belongs after the current copy workload completes; no automatic promotion is implemented. The prepared packet may be used with:

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.store_function_adaptation_v4 run \
  --packet experiments/ranker/pretrained/store_function_v4_prepare_v1/preflight.json \
  --out data/ranker/pretrained/actual-store-function-v4-attempt-1 \
  --owner root
```

This command has **not** been executed. New output paths are mandatory. All identities and functions remain TRAIN-seen, v3 remains inspected development replay, and v4 evaluation is author-visible synthetic wording rather than a sealed benchmark. Actual quality, control tallies, abstention and ranking-only latency still require the native run and independent output audit.
