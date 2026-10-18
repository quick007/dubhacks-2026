# Shopper evaluation framework assessment

Read-only assessment of the existing runner/API. No browser agent benchmark was executed, no service was started, and no legacy runner, API, contract or UI source was modified. CPU tests used temporary test data and existing workspace caches.

## What actually runs

- **Real browser smoke:** `apps/runner/src/playwright.ts:33` launches bundled headless Chromium. Lines 47–63 open the local shop, click the first Add to cart button, open the cart and confirm a fake order. It does not read the persona's goal/task, search, choose by preferences or enable `?model=local`. It records a real screenshot, Playwright trace and result file. There is **no existing approved Chrome-session provider API** here: no computer-use agent, attached user Chrome session, CDP connection or shopper model loop.
- **Scripted dashboard experiment:** `apps/api/internal/runner/runner.go:52` replays authored persona/variant paths and draws SVG page stand-ins. `apps/api/fixtures/catalog.json` has ten jacket personas and two variants; `fixtures/scenarios/jacket.json` supplies six jacket products and predetermined purchases/abandonments. These are separate from the actual sixteen-product storefront. `apps/storefront/fixtures/shopper-personas.json` contains eleven actual-store briefs, not a frozen ten-case paired browser evaluation.
- **Real orchestration:** the Go API snapshots runs, persists SQLite metadata/events/artifacts and dispatches a TypeScript child process (`apps/api/app.go:191`, `internal/runner/runner.go:127`). Current queue code has **eight workers, two browser slots and capacity twenty**, with a 90-second child timeout (`queue.go:19`, `queue.go:141`). The README's concurrency-one statement is stale. Restarted pending/running jobs become failed (`store.go:43`).
- **Limited scoring:** `internal/evaluator/evaluator.go:17` checks scenario facts only when structured page events exist. Current browser events lack those fields and purchased-product identity, so fake-checkout visibility alone becomes score one. Soft preferences are reported but do not gate success. `app.go:217` excludes browser runs from `/api/comparison`; its results are completed mock runs only. `internal/model.Mock` and deterministic recommendation helpers do not provide a shopper agent.

## Installed prerequisites and boundaries

Node v26.9.0, the workspace `tsx` loader, local Go toolchain and the Playwright Chromium executable are present. The runner uses a local browser and needs no external browser provider or API key. This inspection did not launch Chromium or verify a live web/model service.

Local app routing is `http://shop.dubhacks.localhost:1356` through Portless (`packages/contracts/src/local.ts`, `portless.json`, `scripts/dev.mjs`). The existing runner permits requests only to that shop origin (`playwright.ts:35`), so it would block the model endpoint **`http://127.0.0.1:8766/personalize`** (`apps/storefront/src/local-model.ts:54`). A scoped local-model origin exception is required; broad external access is unnecessary. The model service already permits the shop's CORS origin (`store_demo_service.py:24`).

The merchant model service is separate from the Go queue. It owns one resident native worker, four queued jobs and the shared native-owner lock (`store_demo_service.py:30`). Starting it is GPU work reserved for the root's serialized orchestration. Its request schema admits public query, optional item budget and hard requirements; it does not accept hidden persona truth (`store_demo_engine.py:43`). The current frontend sends only `{query}` (`use-local-model.ts:61`), so no sparse merchant profile currently flows through this hook.

Existing browser readiness evidence is reusable: `local-model-performance.ts:12` exposes `data-local-model-readiness` and User Timing measures for rank, firstThree and complete. They measure hook-effect start through React commit plus two animation frames, including network/service queue, while explicitly excluding click-to-effect, image readiness and physical paint. The legacy runner does not collect them.

## Smallest compatible bridge

1. **Freeze ten actual-store cases and twenty paired runs.** Keep shopper-only needs/gold outcomes separate from the merchant-visible query/profile. Pair the same case, seed and action budget across generic and `?model=local`; use fresh browser state and record failure/unattempted outcomes in the fixed denominator. Do not repurpose the jacket paths as actual-store evidence.
2. **Connect the authorized Chrome computer-use agent.** Drive search, reading, product/variant selection, cart and fake checkout from browser observations and the shopper task. Preserve observed actions, abandonment and errors. The existing first-button script is a smoke test and cannot stand in for this agent. No approved Chrome-session provider interface exists in the legacy runner; the new bridge must explicitly supply that authorized connection rather than inventing credentials or claiming the headless runner already supports it.
3. **Add explicit arm routing and the narrow model-origin allowance.** Current `VariantConfig`/`RunRequest` have no model-mode field (`apps/api/internal/contracts/types.go:15`, `:82`), and browser URL generation carries only headline/shipping/order. A compatible sidecar or coordinated versioned contract must identify generic versus local mode and preserve the same public query. Do not send private/gold data into model requests.
4. **Reuse persistence with real observations.** TypeScript `RunnerResult` currently has completion/reasons/notes only; Go additionally supports purchasedProductId. Structured Page/click/search fields exist in Go events but not the TypeScript event interface. Go strictly rejects unknown child-output fields (`runner.go:148`), so coordinate protocol additions or put expanded evidence in the existing `result.json` artifact. Existing allowed artifact filenames are `trace.zip`, `screenshot.png`, `result.json`; preserve their provenance and do not label a CUA action record as a Playwright trace. Events are imported after child exit, not streamed live. Screenshot-linked events also need artifact registration first: `queue.go:96` validates an existing artifact, while `runner.go:152` currently imports events before artifacts.
5. **Score independently against actual-store facts.** Record observed purchased product/variant and cart state; evaluate known hard constraints, item-price budget, preference fit, abandonment, unsafe copy and retrieval separately. Unknown size/stock/shipping facts remain unknown. Use real trace/DOM/readiness evidence for timing. The current binary checkout flag and six-jacket evaluator cannot establish those outcomes; browser comparisons must remain separate from legacy mock aggregates.

These are integration gaps, not a need for a second framework. A first bridge can preserve batch persistence without promising live dashboard streaming. Physical paint, image decode and human conversion remain separate claims.

## Owner coordination

- `copy_fit_runner` owns the **new** `apps/runner/src/shopper-contract.ts` and its tests; coordinate runner/provider/result compatibility there without overwriting legacy runner files.
- `copy_audit` owns the ten actual-store cases, public/private/gold separation and paired-run plan. Candidate product IDs and gold outcomes are scoring inputs, not merchant inference features.
- Root owns the compatibility contract, authorized Chrome/CUA session, local-origin bridge, service lifecycle, final integration and paired execution. Root's separate three-query browser/render/cart/readiness smoke checks are not twenty completed paired shopper runs.
- This lane supplied read-only code assessment and CPU checks only; no competing framework, browser run, GPU job, Git operation or new credentials.

## Verification performed

- From `apps/runner`: `node --import tsx --test src/protocol.test.ts` — **3/3 passed**. These validate protocol handling, not browser behavior.
- From `apps/api`: `GOTOOLCHAIN=local GOPROXY=off node ../../scripts/go.mjs test ./...` — **all Go package tests passed**, including API persistence/queue, scripted runner, evaluator, optimizer and scenario tests. External dependency fetching was disabled.
- Confirmed installed Chromium executable via `chromium.executablePath()` and file existence, without launch. Confirmed local Go and `tsx` loader files.
- Did not run `pnpm test:smoke`: it starts local services/browsers, performs checkout and includes an HMR test that temporarily writes UI files (`tests/local-demo.spec.ts:115`). Existing browser smoke coverage is `tests/local-demo.spec.ts:77` and `:85`; neither is a preference-aware shopper evaluation.
