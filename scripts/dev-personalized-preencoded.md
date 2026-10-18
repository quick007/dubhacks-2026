# Preencoded public-profile development demo

This separate entry prepares the V3 measurement path. It preserves `dev-personalized-development.mjs` as the V2 comparison entry and leaves the accepted-profile selector unchanged. The model quality gate remains failed and `qualityGatePassed`, `modelPromotion`, and `acceptedProfile` remain false. Changing the normal package command is a separate root decision after actual warm-path measurements and functional QA pass.

After the new scripts and pinned V3 runtime/engine/analyzer sources are merged, use the existing copied MAIN model bundle and the byte-exact warm receipt:

```sh
node scripts/dev-personalized-preencoded.mjs --run \
  --receipt data/ranker/pretrained/persona-preencoded-development-admission-v3-attempt-1/receipt.json \
  --receipt-sha256 af33c1ff447835bcc63e0132b97dbe3b334936beb2136aea9741369e7f66a2cf
```

The receipt may also be supplied by its explicit preserved absolute path. To use the relative command above, copy that receipt byte-for-byte from the historical worktree to the same MAIN data path. No additional model weights are needed beyond the existing28-file development serving bundle. All runtime model arguments map to MAIN assets; historical worktree files remain evidence only. The reviewed source bytes must match the current launch code. Missing MAIN files cause failure, with no historical-runtime fallback.

Omit `--run` for preflight. Both modes first reject occupied8766/8767/4313 or a held global native lock; neither stops an existing service. Ports, checkout reuse/standalone checkout, safe offline environment, unique session logs, owned child lifetime and graceful shutdown are reused from the existing launcher. `--asset-root ABS_MAIN` and `--session ABS_UNIQUE_UNDER_CURRENT_CODE_ROOT` remain available. No endpoint/module/query-cache controls can be overridden through this CLI.

The wrapper verifies the exact receipt using `store_persona_warm_http_probe_v1.selection`, which checks the extra V3 source pins and the actual startup's ten public Browse context texts, tensor/key hashes, native-call counts and warm-cache receipt hash. It then reuses the V2 development plan's full checkpoint/co-located public-cache/source-equivalence checks. The only native argument changes select `store_persona_adapter_engine_v3` and force:

```text
--force-fresh-query --preencode-public-profiles --color-cache-entries 32
```

Before READY, V3 encodes the ten authorized public `Browse the store` color/context features. It creates no ranking or copy warm responses. Every real request performs one fresh pristine-base B1 query encoding, learned route projection and fresh neural copy generation. Rank-X/H/Y and color-context feature keys include the actual query, so a distinct query can still require both fresh rank-prefix capture and fresh context encoding. This is not query-independent persona encoding. One-time initialization/profile encoding, per-request model readiness and browser readiness must be reported separately.

The warm receipt admits a technical demonstration of the existing failed-quality model; it is not acceptance or evidence of learned adaptation advantage. Actual browser/factual probe receipts establish what worked for this explicit development path.

CPU checks:

```sh
node --test scripts/dev-personalized-preencoded.test.mjs \
  scripts/dev-personalized-development.test.mjs \
  scripts/dev-personalized.test.mjs scripts/personalized-adapter-profile.test.mjs
```

The eight new tests exercise warm-only admission, reviewed selector arguments, exact V3 argument replacement, false promotion, MAIN path preservation, source drift, busy-port fail-fast, inherited lock/checkout behavior, owned-runner delegation and CLI override rejection. All44 combined tests pass. No native, socket, browser or service was launched by these tests.
