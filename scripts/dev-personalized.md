Prepared one-command personalized store entry, pending the final reviewed model/manifest choice. This work does not promote the current experimental persona checkpoint. It starts no processes unless explicitly invoked with `--run`.

The launcher now has two explicit accepted-profile versions. The original context
model contract below remains supported. The adapter branch described at the end
selects the new LoRA decision adapter and learned query gate. Neither branch
starts without its actual accepted profile; this preparation writes no profile
or acceptance receipt.

Existing root package command (unchanged by this launcher update):

```json
"dev:personalized": "node scripts/dev-personalized.mjs --run"
```

The morning workflow is `pnpm dev:personalized`, then open `http://127.0.0.1:4313`. The ordinary shop opens with the persona selector and personalization enabled through trusted `VITE_PERSONALIZED_STORE=true`. Checkout uses the existing same-origin proxy. The launcher prints one URL and the session directory; no browser is opened automatically. Ctrl+C stops only this session's children, allowing the model's shutdown integrity checks to finish.

Running `node scripts/dev-personalized.mjs` without `--run` performs checks only. It creates no session or application services. The only optional path flag is `--asset-root` for an already populated local repository; no model or dependency is downloaded or installed. The default is the current repository if its existing venv is present, otherwise the sibling `dubhacks-2026` asset repository. The lexical venv Python path is retained. Native code starts from this worktree; the model owner's wrapper then changes asset CWD before native imports, preserving the one shared main `native-owner.lock`.

Preflight refuses busy ports8766,8767 or4313, and refuses an occupied shared native lock. Nothing is killed or taken over. It prefers the existing checkout on4331 after a read-only receipt-route probe, with verified normal API4311 as a fallback. An experiment-only or unrelated4311 receiver does not block a free or valid checkout4331 and is never stopped. If neither existing service offers checkout and4331 is free, the existing standalone Go checkout runs there with offline dependency settings and a unique receipt directory. An unknown listener occupying4331 produces a targeted error when no verified4311 fallback exists. Port4200 is unrelated and is not reserved. The launcher does not start the normal database application, load an environment file, read credentials, call a paid API or inherit API secrets. Existing storefront dotenv files cause a clear preflight failure without their contents being read.

Every invocation reserves a new `data/personalized-dev/<timestamp>-<uuid>/` lifetime with private logs, a public-path/checkpoint launch receipt, ready metadata, native output, optional checkout receipts and a launcher shutdown receipt. Previous outputs are never overwritten. A launcher exit alone is not a model-integrity claim: inspect the native shutdown receipt. Forced child cleanup is recorded as an error.

The model contract is the owner's `experiments.ranker.pretrained.store_persona_engine_v1` on8767, with the existing verified v5 rank export and parallel copy assets. The root must publish an explicitly accepted public serving profile at `experiments/ranker/pretrained/store_persona_serving_profile_v1.json` after the new native-context model passes its actual quality gate. Until then, the default command fails clearly without starting a model. The previous weak experimental12629e...head is explicitly rejected, even if accidentally marked accepted. There is no fallback to that checkpoint.

The root-owned profile schema is:

```json
{
  "version": "personalized-local-profile.v1",
  "status": "accepted",
  "contextKind": "actual-reviewed-model-kind",
  "personaHead": {
    "root": "worktree",
    "file": "data/ranker/pretrained/REVIEWED_RUN/head.pt",
    "sha256": "EXACT_REVIEWED_SHA256"
  },
  "personaFeatureRoot": {
    "root": "assets",
    "directory": "data/ranker/pretrained/actual-store-function-v3-attempt-1"
  }
}
```

This example is documentation only, not an accepted artifact. Paths are relative to either the worktree or existing asset repository. The profile selects exact reviewed artifacts, not private goals or evaluation labels. The launcher records its SHA256. The fixed integration manifest is `experiments/ranker/pretrained/store_demo_integration_personalized_v1/manifest.json`; its exact bytes are also recorded. No older or latest-looking alternative is selected automatically. The model wrapper remains responsible for its original full source, model and checkpoint guards.

`personaFeatureRoot` has one specific meaning: the original16-product v3 catalog feature run consumed by `base.load_catalog_features`. It must resolve to the existing asset repository's `data/ranker/pretrained/actual-store-function-v3-attempt-1`, containing its original report, packet, raw-feature freeze and catalog tensor/receipt files. It is not the new432-row context extraction directory. The launcher passes this exact original catalog path to `--persona-feature-root` without rewriting any inputs.

The new head's `public-demo-features.pt` is a separate artifact adjacent to `head.pt`. Preflight requires both files at that location; the runtime resolves and verifies the adjacent cache against the unchanged checkpoint's recorded checksum. Copy that pair together when moving reviewed model artifacts to main. Neither the launcher nor the profile changes saved checkpoint bytes or redirects the catalog-feature argument to this demo cache.

Portability: this entry uses the existing macOS/POSIX Python/MLX runtime and shared flock, the existing local Go toolchain when checkout must be started, and already installed workspace dependencies. It is not a cross-platform installer. After final evaluation, the parent copies the reviewed `head.pt` and adjacent `public-demo-features.pt` pair into main's ignored data directory and writes `personaHead.root: "assets"`. The original catalog feature run stays at its existing main asset path. That permits the worktree to merge or move without retaining worktree-only weight paths. The initial native working directory remains the code repository; the wrapper changes to the explicit asset root before resolving legacy assets or acquiring the global native lock. The fixed code/profile/manifest files must travel together. No paths are rewritten inside model-owned files by this launcher.

CPU validation, with all port probes, process spawning and signals replaced by fixtures:

```sh
node --test scripts/dev-personalized.test.mjs
```

No frontend, root package command, README, model-owned source, running service or browser was changed or started by this preparation.

## Adapter and query gate profile

The same profile filename accepts version `personalized-local-adapter-profile.v2`
with `status: "accepted"` and `kind: "lexical-choice-lora-query-gate.v2"`.
Its parser and complete admission checks are in
[`personalized-adapter-profile.mjs`](./personalized-adapter-profile.mjs).
The root task publishes this profile only after actual offline and live
acceptance. The launcher never changes acceptance state or falls back to an
older head.

The profile contains these explicit artifact references, each shaped
`{root: "assets" | "worktree", file: "data/ranker/pretrained/...", sha256}`
(research JSON under `experiments/ranker/pretrained/` is also allowed):

- `servingManifest`, `queryIntentHead`, `colorHead`, `colorCache`, `rankExport`,
  `copyManifest` and `integrationManifest`.
- `catalogFeatureRoot: {root: "assets", directory:
  "data/ranker/pretrained/actual-store-function-v3-attempt-1"}`.
- `queryIntentSourceManifestSha256`: the reviewed v3 TRAIN manifest
  `25fdb0c976dc3b6e01e826e7f30ea63fc65e3fbb85f56f24f4c83e8538703232`.
- `sourceSha256`: current code-root-relative module paths mapped to exact
  checksums, including the launcher/helper and required adapter runtime modules.
- `evidence`: references named `offlineAudit`, `admissionBeforeGold`, `queryMechanicalAudit`,
  `queryCombinationReplay`, `httpValidation`, `browserValidation`, `shutdown`
  and `acceptance`.

Before spawn, preflight checks the selected adapter safetensors, sanitized
inference certificate, pure public20 feature index and all20 co-located NPZ
checksums. The NPZ layout stays relative to the index, with exact prompt-token
SHA keys; there is no original-worktree fallback. Native startup still verifies
the actual NPZ tensors and model math. Cached scoring must be `plain`; dynamic
capture must be `gradient-public-XHY-last16`. Plain dynamic inference is refused.
The existing source-bound public48 neural copy assets and original v5 model
remain checked. The old product head supplies learned color scores only.

Admission requires the independent offline pass, actual query TRAIN replay and
combined replay, all320 valid HTTP requests, the browser probe's13 protocol
requests/ten loaded switches/two functional arms, and actual927/copy/v5/rank/
cache/source shutdown integrity. Partial or failed evidence cannot start a
process. The final root-owned `persona-adapter-live-acceptance.v1` receipt must
have `status: "accepted"`, `overallPromotionPass: true`, `browserPassed: true`,
`freshCopyFactsPassed: true`, `shutdownIntegrityPassed: true` and
`sourceByteEquivalenceVerified: true`. It binds `adapterSha256`,
`servingManifestSha256`, `gateCheckpointSha256`, `integrationManifestSha256`,
all seven profile artifact checksums in `artifactSha256`, all seven measured
evidence checksums in `evidenceSha256`, and the full current relative
`sourceSha256` inventory. Historical audit paths remain provenance; they are not
runtime fallback locations. After transfer to main, root must verify identical
artifact bytes and regenerate the current source/integration inventories before
publishing this final receipt.

Specific-query preservation additionally follows the root's frozen
`PERSONA_SPECIFIC_QUERY_REGRESSION_POLICY_V1.json`, SHA
`b1436edfd3f9641c9431cb41aab9fb04b99c3a5f7606f3d456d2c6b87bfe4734`.
All320 actual HTTP requests must complete with protocol/facts passing and all320
top choices or abstentions agreeing with v5 under the same effective guards.
There are no exclusions, retries or replacements. The launcher checks the
actual per-case errors, facts, identities, unique request IDs and agreement
flags as well as both320-count summaries and the exact co-located public
`plan.json`. The final receipt must bind
`specificQueryRegressionPassed: true`, `specificQueryRegressionPolicySha256`,
`specificQueryRegressionPlanSha256` and `v5CheckpointSha256`. This is preservation
of the existing query behavior; v5 choices are not treated as human truth or
evidence of new relevance quality.

The adapter branch invokes `store_persona_adapter_engine_v2` with explicit
`--serving-manifest/--serving-manifest-sha256`, `--catalog-feature-root`,
`--color-head/--color-cache`,
`--query-intent-head/--query-intent-head-sha256`,
`--query-intent-source-manifest-sha256`, `--candidate-run`,
`--integration-manifest`, `--rank-head-export/--rank-head-export-sha256`,
`--asset-root` and unique `--output`. The engine fixes8767/width20/7GiB;
legacy `--persona-head` and `--port` are absent. Ready health must identify
the accepted adapter, serving manifest, learned gate and color head before
checkout/storefront children start.

Exact pre-adapter launcher/source/test/doc bytes are preserved in
`experiments/ranker/pretrained/persona_launcher_baseline_preserved_v1/`.
CPU protocol fixtures (no trained-quality or native evidence):

```sh
node --test scripts/dev-personalized.test.mjs scripts/personalized-adapter-profile.test.mjs
```
