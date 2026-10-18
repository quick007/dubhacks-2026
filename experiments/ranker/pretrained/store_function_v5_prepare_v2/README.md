# V5 preparation v2: explicit responsive-image binding

This packet preserves the original v5 dataset and fit: 96 TRAIN requests (32 verified reused queries plus 64 new balanced utterances), 32 separately authored evaluation requests, 48 reused catalog/TRAIN vectors, 96 new native forwards, 144 total vectors, and one fixed 80-step TRAIN-only schedule. The v1 preflight and exact prior driver/tests/helpers are preserved under `store_function_v5_prepare_v1`; no native v1 run occurred.

Preparation v2 explicitly selects `ImageDeliveryCatalogBinding` and the reviewed v8 manifest. That verifier proves the runtime 16-product export/model input is unchanged, original image assets are unchanged, and only the exact reviewed renderer/image-delivery source hashes are allowed. The original strict `DemoCatalogBinding` remains the default and rejects this image-delivery manifest. The new packet pins 13 Python helpers and both image-verification scripts. Nothing changes cached feature provenance or introduces learned image selection.

After gracefully closing the existing native service, the parent may launch:

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.store_function_adaptation_v5 run \
  --catalog-binding image-delivery \
  --integration-manifest experiments/ranker/pretrained/store_demo_integration_v8/manifest.json \
  --packet experiments/ranker/pretrained/store_function_v5_prepare_v2/preflight.json \
  --out data/ranker/pretrained/actual-store-function-v5-attempt-1 --run
```

Use the recorded relative manifest argument: the driver compares the full prepared packet before native admission. The existing service is never stopped automatically. Runtime source or manifest changes require a new preserved packet; there is no old-manifest fallback.

Preflight SHA-256: `eb6420492f63f141fd61c92c40e75c66b0f6dab241458cd9b96747bdaa09904f`.
Driver SHA-256: `59a76f16d421c1d98d62a594f60e3453fcc13502ca6689a93b29872ec53fa1f0`.
Dataset freeze remains `a7a0e64f094f7574671c91a70af33525068b81f9e5cb0a184660ef38eafa8e75`.

CPU validation: ten tests passed as a suite, then the added original-strict-default rejection test passed separately. These cover current image-aware preflight and real reused feature receipts, the unchanged data/split/guard/fit checks, explicit factory forwarding, source-pin failure detection, and fake-native failure cleanup. No GPU inference, service change, or Git action was performed by this preparation.
