Prepared one-command personalized store entry, pending the final reviewed model/manifest choice. This work does not promote the current experimental persona checkpoint. It starts no processes unless explicitly invoked with `--run`.

Proposed root package command, for the parent to add after integration:

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
