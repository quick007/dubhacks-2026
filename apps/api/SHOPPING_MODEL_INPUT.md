# Telemetry to model input: CPU consumer bridge

[Shopping telemetry](SHOPPING_TELEMETRY.md) now has a concrete preparation bridge from `fieldwork.shopping-choice.v1` to the public request/catalog layout used by `store_function_adaptation.fit_inputs`. It emits a separate, versioned `fieldwork.shopping-context-model-input.v1` packet. It does not modify the fixed v5 experiment, encode with Qwen, optimize weights, or deploy a head.

The collection contract excludes shopper query text. These examples therefore describe **observed storefront context and positive actions**, with `queryObserved: false` and `semanticIntentSupervision: false`. Request text is constructed from fixed public wording plus observed display mode, ordering and shipping visibility. It contains no chosen product, candidate positions, session identity, model version, inferred intent, budget or private persona. Target product indices live in separate label files. Context does not establish a shopper preference, and these observational choices do not establish causal lift or real purchases.

## Prepare and independently consume a packet

Run from the repository root using the existing local Python runtime with CPU PyTorch:

```sh
.tools/pretrained-venv/bin/python apps/api/scripts/shopping_model_input.py \
  --input data/shopping-choices-v1.jsonl \
  --out data/shopping-context-v1 --cpu-consumer-check

.tools/pretrained-venv/bin/python apps/api/scripts/shopping_model_input.py \
  --check-packet data/shopping-context-v1

.tools/pretrained-venv/bin/python -m unittest discover \
  -s apps/api/scripts -p test_shopping_model_input.py
```

Use a fresh output directory for each snapshot. Files are private (`600`) in a private directory (`700`). Preparation and the consumer reject synthetic/development/test choices by default; add the explicit `--allow-synthetic` preparation flag only for protocol testing. The serialized flag must be an actual JSON boolean. The original cloud export and choice builder have their own independent synthetic gates.

The packet includes:

| File | Contents |
| --- | --- |
| `catalog.json` | Frozen sixteen-product public facts, in model index order. |
| `routing.json` | External public product/variant IDs mapped to product indices. IDs stay outside encoder text. |
| `public-cases.json` | Constructed context-only requests, case IDs and original whole-session train/validation/test splits. |
| `train-labels.json`, `validation-labels.json`, `test-labels.json` | Separate observed chosen-product targets and public catalog support. |
| `encoder-records.json` | Actual `request_text` / `product_text` helper outputs and SHA-256 digests; no native features. |
| `provenance.json` | Original choice records, candidates/rendered variants, chosen variants, action, source hashes, session/batch/event IDs, environment and client-reported model provenance. |
| `manifest.json` | Counts, input/file/source hashes, exact scope and unperformed-training flags. |
| `cpu-consumer-check.json` | Optional layout check result; always marked synthetic protocol-only CPU features. |

The adapter calls the actual verified `store_catalog_v1.load_snapshot()`: its independent raw-export/projection pins and exact ordered projection bind public IDs to catalog indices. Swapping mapping rows cannot silently relabel a product. It verifies the v5 catalog matches the verified snapshot and the intake catalog has exactly the same product/variant identities. A later catalog change requires a deliberately versioned import and bridge update.

Independent `--check-packet` validates file hashes, source pins, counts and reconstruction of requests/targets from preserved provenance. It then calls the actual `fit_inputs` helper with deterministic, explicitly synthetic 4,096-dimensional **CPU protocol fixtures**. The fake feature store fails if the helper accesses a validation/test request. These vectors are not Qwen outputs, are not saved as native features, and measure no prediction quality. An empty train split reports `skipped-no-training-rows`; it is not a successful fit.

## Verified scope and remaining training work

The real browser → own backend → Tiger → export path was previously verified with labeled synthetic traffic; its immutable exported four choices from two sessions were prepared here. Default preparation accepted zero. Explicit synthetic preparation accepted four train choices, preserved both session identities and their source/model provenance, and independently exercised `fit_inputs` with query shape `[4,4096]`, catalog shape `[16,4096]` and targets `[4]`. Separate fixtures cover train/validation/test isolation, all sixteen ID mappings, target-independent request text, corrupted/reordered catalog rejection, strict synthetic permission and serialized tamper rejection. See [shopping-model-input.verification.json](shopping-model-input.verification.json).

This closes the file/schema/consumer-layout gap. It does not make the current fixed 96/32 v5 driver consume telemetry. A new native training job still needs a frozen real-data cohort, native feature provenance, a suitable positive-choice objective with candidate/exposure handling, validation and session weighting, evaluation, and explicit serving promotion. In particular, `fit_inputs` is a tensor/target preparation helper: this check does not establish that the existing all-sixteen-class loss is appropriate for observational choices. The adapter preserves candidate denominators and invents no nonclick/variant negatives. Two actions on the same product are preserved evidence, not independent preference measurements. No real customer rows or model improvement have been established.

## Production enablement

The normal local app remains off. For an approved public deployment, configure its own backend with the existing private `TIGER_DATABASE_URL`, `SHOPPING_TELEMETRY_ENABLED=1`, `SHOPPING_TELEMETRY_ENVIRONMENT=production`, and `SHOPPING_TELEMETRY_ORIGINS` equal to the exact HTTPS storefront origin(s). Leave `SHOPPING_TELEMETRY_ALLOW_SYNTHETIC` unset. The existing team database already has migration 002; startup does not migrate.

Build the storefront with `VITE_SHOP_TELEMETRY=production`, `VITE_SHOP_TELEMETRY_HOSTS` equal to its exact public hostname(s), and `VITE_SHOP_TELEMETRY_API_ORIGIN` pointing to that backend over HTTPS. Keep `VITE_SHOP_TELEMETRY_CONSENT=required`, or use the separately approved notice/opt-out policy. Set `VITE_SHOP_MODEL_VERSION` only to the actual approved artifact reference. No database secret belongs in browser build settings. Both client and server deployment policies must match; opting out, DNT or GPC prevents collection.

Export with fixed received-at cutoffs using `cmd/shopping-export` and no synthetic flag, run `shopping-training.mjs` with no synthetic flag, then this adapter with no synthetic flag. Archive manifests together. A `202` is local durability; verify asynchronous cloud acknowledgement before treating a batch as available to a Tiger export. Production hosting/configuration, real-data collection, native training and deployment remain separate steps; this work creates none of them.
