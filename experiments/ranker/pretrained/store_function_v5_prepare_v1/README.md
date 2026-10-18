# Known-store v5 ranking pilot

This CPU-prepared pilot keeps the exact live v3 request serialization and 16-class softmax `FunctionHead`. It adds 64 public-function utterances (four per product) to the original 32 training requests. The 32 evaluation requests were authored separately before native extraction, with two requests for each public product function. Every product identity is training-seen; this measures synthetic request paraphrases, not new-catalog or human generalization.

The native plan reuses 16 catalog and 32 original TRAIN feature vectors after exact text, token, model provenance, tensor, and serialized-artifact checks. It then makes 64 new TRAIN and 32 evaluation forwards: **96 new native forwards, 144 total feature vectors**. Evaluation features and labels do not enter fitting. The schedule is one fixed 80-step TRAIN fit with no evaluation-based stopping. All inputs fit 64–103 tokens without truncation.

Controls are the frozen v3 checkpoint, balanced class prior, random head, and a fixed 17-row query permutation. Fresh endpoint timing includes query encoding, immutable feature serialization, 16 verified catalog reads, head scoring, hard guards and real-action serialization; copy, network, browser and images are excluded. The permutation's actual same-target rate is reported after predictions freeze.

Prospective review goals are at least 28/32 top choices, 32/32 targets in six results, at least one of two top choices for every function, and 16/16 supported hard-guard controls. These are goals, not measured results. The `including delivery` wording is a separate known parser-gap diagnostic; it must not be represented as delivered-budget support.

After the owner gracefully stops the local model service, the parent may run:

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.store_function_adaptation_v5 run \
  --integration-manifest experiments/ranker/pretrained/store_demo_integration_v6/manifest.json \
  --packet experiments/ranker/pretrained/store_function_v5_prepare_v1/preflight.json \
  --out data/ranker/pretrained/actual-store-function-v5-attempt-1 --run
```

The driver refuses an active service, takes the existing native-owner lock, uses a 180-second main-thread deadline, and preserves fixed evaluation denominators and failures. It verifies all 927 frozen tensors before controlled shutdown. Successful weights keep the v3 architecture and loader format, but the existing `StoreRankSession` report guard intentionally still requires the original 64-forward report. A **new explicit serving adapter** must validate `serving-head.json`; do not falsify the v5 forward count or edit the frozen live loader.

Preflight SHA-256: `be5e5eb4b2f79e0827f90583178866d1d08ad06cb116990e190cd5301efef56f`.
Dataset freeze SHA-256: `a7a0e64f094f7574671c91a70af33525068b81f9e5cb0a184660ef38eafa8e75`.

CPU validation: nine tests pass, including actual cached-feature preflight, split and balance checks, safe checkpoint reload on engineered features, guard controls, and an injected fake-native failure proving cleanup and the 32-case unattempted evaluation roster. No native inference or fitting has been executed by this preparation task.
