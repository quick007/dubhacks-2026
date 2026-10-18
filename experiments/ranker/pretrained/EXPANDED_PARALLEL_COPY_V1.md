# Expanded public catalog parallel copy

`store_expanded_parallel_copy_v1.py` is a separate dynamic-count packet, CPU fitter and serving loader. Existing fixed48 assets and validators are unchanged. The final preparation command requires 35 products and 105 actual product/color bindings; the partial30 catalog cannot launch preparation.

```sh
python -B -m experiments.ranker.pretrained.store_expanded_parallel_copy_v1 prepare \
  --catalog /Users/lukas/Documents/Projects/dubhacks-2026/packages/contracts/data/expanded-store-catalog-v1.json \
  --out experiments/ranker/pretrained/expanded_parallel_copy_prepare_v1
```

The packet declares every public binding TRAIN: three card-field kinds in three styles, plus both generic section positions in three styles pooled from triples of distinct products. Vocabulary derives from this offline supervision. Source features contain only exact `fact_text` strings and hashes. IDs, user histories, hidden personas and target strings are absent from feature text. All supervision is synthetic factual catalog copy; no heldout quality claim applies.

Root owns the actual native extraction. Use `make_expanded_feature_bundle(records, values, provenance, catalog_sha=FINAL_SHA)`, with CPU float32, finite unit-L2 features `[105,4096]` in frozen record order. Provenance must be the following exact fields plus `nativeReceiptFile` (absolute) and `nativeReceiptSha256`:

```json
{
  "evidenceSource": "native-frozen-qwen",
  "modelRevision": "8b2b98c00a6b4d291155e4890773ca8f769aee53",
  "copyAdapterEnabled": false,
  "rankAdapterEnabled": false,
  "nativeForwards": 105,
  "truncatedRows": 0,
  "catalogSha256": "FINAL_CATALOG_SHA256",
  "frozenTensorCount": 927,
  "frozen927HashesUnchanged": true,
  "shutdownVerified": true
}
```

The durable final native receipt must include these exact fields as `copyFeatureProvenance`. Those integrity flags are populated only after actual extraction and teardown, never prepared as a success receipt in advance. The helper performs no MLX/native work. Optional catalog vectors remain a separate native extraction: `catalogVectors`, `catalogTensorSha256`, `catalogProductIds`, `catalogTextSha256`, `catalogNativeForwards`, with the same public catalog SHA. The expanded ranking owner independently binds their exact product text.

```sh
python -B -m experiments.ranker.pretrained.store_expanded_parallel_copy_v1 fit-cpu \
  --packet experiments/ranker/pretrained/expanded_parallel_copy_prepare_v1 \
  --bundle /absolute/path/to/actual-native-features.pt \
  --out /absolute/unused/expanded-copy-fit-attempt-1
```

The head retains the existing 4096→128→14 lexical slot architecture, learned EOS, AdamW `.004`, weight decay `.0001`, seed7211 and one CPU thread. The schedule is at most2000 actual updates. At fixed100-update checkpoints it may stop only when all supervised fields exactly match and pass the factual/EOS gate. Finite changed weights, optimizer count, loss/checkpoint snapshots, immutable input/source pins and exact reload slot parity are recorded. A failed completeness gate preserves its checkpoint and receipts but cannot load for serving.

Serving API:

```python
head, feature_map, records, metadata = load_expanded_parallel_assets(run, expected_manifest_sha)
```

It opens only the selected head, pure public fact export, actual feature bundle, pinned native receipt and catalog. It never opens `packet.TRAIN.json` or invokes the teacher. Metadata includes `assetSha256`, `featureBundleFile`, `featuresSha256`, `checkpointSha256`, `nativeProvenance`, counts and explicit supervised scope. Existing `generate`, `parallel_rows` and `parallel_payload` compute fresh field slots from the selected facts; short groups abstain from untrained section means rather than padding.

Seven CPU tests cover dynamic source bindings, overflow/duplicates, partial-catalog refusal, native-receipt/order tampering, unchanged old48 rejection, actual one-update fixture failure preservation, changed weights and serving teacher/label isolation. Fixture receipts are temporary synthetic protocol tests; they establish no real native or model-quality result.
