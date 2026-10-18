# V5 preparation v3: reviewed upstream API fixture exception

This third packet binds the unchanged v5 ranking pilot to the reviewed v9 image-delivery manifest after upstream merge `ea33cf7`. The unchanged preparation algorithm remains version 2. The native driver remains SHA `59a76f16d421c1d98d62a594f60e3453fcc13502ca6689a93b29872ec53fa1f0`; it still requires explicit image binding, an unused output, stopped local service and the native-owner lock.

The image binding helper is now `actual-store-render-image-delivery.v2`. It permits only the exact reviewed API mock fixture hash as an auxiliary source exception, alongside exact reviewed Card/Photo/Gallery renderer hashes. Runtime product export, model input facts, variant mapping, and original assets remain unchanged. Current API fixture persona goals/labels are never model inputs or training labels; historical snapshot tasks remain provenance only. The original strict catalog rules and default binder are unchanged.

Compared with preparation v2, all **144 encoded records and token IDs, CONFIG, dataset freeze/files, feature provenance, 96 reused feature-file pins, token range, and original v3 report/packet hashes are exactly equal**. The only changed Python source pin is the image binding helper. This remains 96 TRAIN queries (32 reused +64 new), independently authored 32 evaluation queries, 48 reused vectors, **96 new native forwards /144 total feature vectors**, and one fixed 80-step TRAIN-only fit.

All 12 CPU tests pass. The additional test forbids parsing current `apps/api/fixtures/catalog.json` during preparation and requires exact v2→v3 encoded-input equality. V1/v2 preflights and exact source/test archives remain preserved. No native v5 fitting has occurred in this preparation task.

After graceful service shutdown, the parent may run:

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.store_function_adaptation_v5 run \
  --catalog-binding image-delivery \
  --integration-manifest experiments/ranker/pretrained/store_demo_integration_v9/manifest.json \
  --packet experiments/ranker/pretrained/store_function_v5_prepare_v3/preflight.json \
  --out data/ranker/pretrained/actual-store-function-v5-attempt-1 --run
```

Preflight SHA-256: `047cdbc3dea0121513ef35d73111af030a724aee2f8eff8ea8664aebc8962015`.
V9 manifest SHA-256: `471f7c341986af21368972876463daf0056a913868b168dd9c65ada4af2f1bd3`.
Dataset freeze remains `a7a0e64f094f7574671c91a70af33525068b81f9e5cb0a184660ef38eafa8e75`.

No promotion or new quality result is claimed. The old runtime's 64-forward report guard remains untouched; successful v5 results require a new explicit serving adapter. The existing `including delivery` wording gap stays a separate documented parser diagnostic, not a learned ranking result.
