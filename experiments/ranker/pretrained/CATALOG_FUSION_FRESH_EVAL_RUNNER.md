# Fresh no-fit evaluator handoff

The new `catalog_fusion/fresh_eval_runner.py` is ready for root source/data review. No native execution or fitting was performed in this lane. Six new CPU-stub integration tests pass; all49 catalog-fusion tests pass in4.718 seconds. Stub tests are explicitly labeled and are not native quality or latency evidence.

The frozen data checksum is `39e3f0073a2a32da6b94dfd18d943ee76eeef9ccb0af181ceebd2c9ed37d5da9`. The deterministic CPU token/source packet is `catalog_fusion/fresh_eval_v1_prepare/fresh-preflight.json`, SHA-256 **`4e81f7bde927b7fc0237773d9cbefe4d66eb6dadd661b75cbf8b976be28d857f`**. Every packet source hash still matches. There are96 exact B1 whole-input records,13,966 tokens total: catalog151–178 tokens and requests83–92. No input needs truncation. Raw float32 features require1,572,864 bytes; request-specific repeated candidate matrices add about27MiB. These are storage estimates, not a measured native memory result.

The normal command requires both reviewed checksums, re-creates the same public token packet and refuses source/input/token/model changes. It repeats full pinned shard/tokenizer/config verification before the native model loads. The local model path/revision is pinned; there is no downloading. Native execution retains the fixed6GiB active-and-observed-peak guard,60-second load/two-forward pilot and180-second extraction budget, with no automatic increases.

Before extraction, copy and verify the already completed full/prior/random semantic checkpoints, frozen image checkpoint and original model-freeze manifest byte for byte. Statistics remain `fb00e986642aac467a60286d4ca1a6ceebaef9ac048587341be0810ce621bc53`; image bytes remain `e292b8e1ae0498774f7e1eee8f2a765e42be0743df57dcc45d6afe480682c582`. No optimizer or means update is called. New catalog/request content receives new text-bound raw keys and immutable FeatureStore receipts.

Extract72 cold catalog variants once, then24 genuinely fresh query vectors against the verified warm72-row catalog cache. Each fresh-query observation records native encoding, construct/encode/save/verify, warm catalog hashing/loading, all72-row scorer timing and12/24-card mapping separately. It also records the observed resident-model fresh-query/cache/two-grid pipeline wall time. Model load, cold catalog work, complete run wall time, RSS and native active/peak memory are separate. Cached CPU replay is never reported as fresh-request latency. This does not measure fresh copy generation or complete-page latency.

Before reading any private fresh file, score full/prior/random/paired-opposite-preference controls and verify exact ID-mapped predictions under all three roster rotations. Reload all96 feature artifacts and copied model checkpoints; compare predictions exactly; seal `predictions.pt` and `FEATURES_PREDICTIONS_FROZEN.json`. Only then read frozen private labels/controls and public atom targets for diagnostics. Recheck source/model/feature/prediction hashes after evaluation. Private teacher files never enter encoding or inference inputs. The paired query source is independently derived from public tasks/styles/preferences and compared to the private control mapping after sealing.

Evaluate five arms including the separately labeled nonlearned public-tag oracle. All24 queries score all72 variants/all36 products. Three mandatory roster views ×12/24 cards produce144 guarded grid cases per arm. Report full grid component fulfillment/duplicates/masks/compatibility separately from best-product, nDCG12/24, image correctness, pair both-correct/flips and atom precision/recall. A component-valid grid does not imply semantically correct ranking or a complete generated page. Nonempty-requirement shortfall probes use deterministic CPU score fixtures only.

Failures preserve partial artifacts and use fixed24-request/144-grid-per-arm denominators; no partial success is promoted to a smaller denominator. Recorded successful native observations may undercount a forward interrupted before its observation was written, and the failure receipt says so.

After reviewing and committing source, root can launch the serialized native job **after copy-fit and learned-probe jobs finish**:

```sh
.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.catalog_fusion.fresh_eval_runner \
  --packet experiments/ranker/pretrained/catalog_fusion/fresh_eval_v1_prepare/fresh-preflight.json \
  --reviewed-packet-sha256 4e81f7bde927b7fc0237773d9cbefe4d66eb6dadd661b75cbf8b976be28d857f \
  --reviewed-freeze-sha256 39e3f0073a2a32da6b94dfd18d943ee76eeef9ccb0af181ceebd2c9ed37d5da9 \
  --out data/ranker/pretrained/catalog-user-semantic-fresh-eval-v1-attempt-1
```

Committing does not change hashed source bytes. If source needs a fix, preserve the reviewed packet and prepare a new packet in a new directory. Existing fresh data, model checkpoints, prior reports and `FULL_GRID_WORKLOAD.json` remain unchanged. “Sealed” refers to exclusion from fitting and freezing before new model outputs; the generator/reviewer can inspect the data. This remains a seven-atom synthetic fact task, with inherited image weights and no learned copy/sections or human conversion claim.
