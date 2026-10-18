# Storefront Lab

A local DubHacks demo for adapting a storefront to a shopper's public request.
The React shop has 16 fixture products, color/gallery controls and a fake
checkout. An optional local Qwen path learns product ordering and generates
fresh titles, badges, headings and short copy from selected catalog facts.

The dashboard also runs synthetic comparisons through a Go API. Its mock
scores, the scripted checkout smoke test and independent shopper-agent evidence
are separate results. All purchases are fake; no human conversion lift has been
established.

## Run the default demo

Use Node 24.11+ in the 24.x line or Node 26+, pnpm 12.5.1, and Go 1.24+ with
CGO and a C compiler. The Go helper prefers `.tools/go/bin/go` when present,
then system Go; `STOREFRONT_GO_BINARY` overrides it. SQLite needs no database
server. The default demo requires no model weights or API key.

```sh
pnpm install
pnpm seed
pnpm dev
```

| App | Address |
|---|---|
| React dashboard | <http://dubhacks.localhost:1356> |
| Go API | <http://api.dubhacks.localhost:1356> |
| React storefront | <http://shop.dubhacks.localhost:1356> |

`pnpm dev` owns a loopback-only Portless proxy on port 1356 and the three app
processes. Ctrl-C stops them. It requires no sudo, certificate installation or
hosts-file change. Seed before starting the API; its data lock prevents
concurrent mutation. Local metadata and artifacts live in ignored `data/`.

Optional experiment-result persistence uses `TIGER_DATABASE_URL` in the ignored
root `.env`; `pnpm --filter @storefront/api dev` loads it. See the
[results adapter guide](apps/api/EXPERIMENT_RESULTS.md) for setup and queries.
A `202` response confirms a durable local receipt; cloud acknowledgement is
separate. Hosted export of real shopper evidence requires approval for that
specific payload and destination.

The [shopping telemetry guide](apps/api/SHOPPING_TELEMETRY.md) covers separately
configured session observations, sanitized replay and versioned training exports.
Normal development collection is off. Synthetic test sessions stay excluded from
the default real-session export; replay is separate from training examples. The
[model-input bridge](apps/api/SHOPPING_MODEL_INPUT.md) validates a separate
context-only learner packet. Its consumer check uses explicitly synthetic CPU
protocol vectors; no shopper query was captured and no live model was retrained.

In the dashboard, open **All sessions** and choose **Run a round** to queue
the deterministic mock comparisons. The separate legacy Playwright adapter
exercises a scripted fake checkout and requires
`pnpm exec playwright install chromium`. Its first-item purchase script does
not represent an independent shopper choosing a suitable product.

## Enable the local model

The verified native environment uses Apple silicon, Python 3.14.7, MLX 0.32.4
and MLX-LM 0.32.2. One frozen Qwen3.5-9B four-bit backbone supplies fresh query
features to a trained ranking head. The fast catalog demo predicts fresh short
text with a trained parallel word head over cached public Qwen fact features.
The adapted autoregressive copy decoder remains a separate backend. The resident
service handles requests serially on `127.0.0.1:8766`.

This optional path requires the existing local environment, pinned Qwen weights
and completed serving checkpoints. They are not supplied by a fresh Git clone:

| Local prerequisite | Location |
|---|---|
| Python environment | `.tools/pretrained-venv/` |
| Pinned model snapshot | `.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53/` |
| Actual-store ranking run | `data/ranker/pretrained/actual-store-function-v3-attempt-1/` |
| Additional ranking candidate | `data/ranker/pretrained/actual-store-function-v5-attempt-1/serving-head.json` |
| Copy adapter run | `data/ranker/pretrained/actual-store-copy-fit-v2-attempt-1/` |
| Catalog-trained parallel head | `data/ranker/pretrained/store-parallel-text-train-all-demo-v1-attempt-2/` |
| Verified public fact features | `data/ranker/pretrained/store-parallel-text-v1-attempt-1/features.pt` |

With those artifacts available, start one resident worker from the repository
root. Choose a new output directory for each service lifetime:

```sh
.tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.store_candidate_image_service_v1 \
  --candidate-run data/ranker/pretrained/store-parallel-text-train-all-demo-v1-attempt-2 \
  --integration-manifest experiments/ranker/pretrained/store_demo_integration_v11/manifest.json \
  --rank-head-export data/ranker/pretrained/actual-store-function-v5-attempt-1/serving-head.json \
  --rank-head-export-sha256 17a0a15a2da6e2fc90e651e1f5bb2c237d61241795ec0cca548a4f3ce319d703 \
  --output data/ranker/pretrained/store-candidate-new-session
```

Wait for `GET http://127.0.0.1:8766/health` to report `ready: true`, then open
<http://shop.dubhacks.localhost:1356/?model=local>. For a standalone shop,
`pnpm --filter @storefront/storefront dev --host 127.0.0.1 --port 4312` makes
the standalone address explicit; open
<http://127.0.0.1:4312/?model=local>. Ordinary browsing remains available without
the model parameter. See the [service guide](experiments/ranker/pretrained/STORE_DEMO_SERVICE.md)
for artifact checks, constraints and request/receipt formats.

The shopper query affects ranking. Copy receives selected public facts and a
neutral browsing request. Item budgets, exact-name rules and supported hard
constraints use deterministic guards. Images use the existing storefront
gallery; image selection and hidden-persona inference are not learned here.
Stock and sizes remain unknown. Unsafe or unfinished generated fields are
withheld, while catalog names and prices remain available.

The parallel head is a fully supervised catalog demo: all 48 product/color
bindings and three short writing styles were trained. It predicts word/EOS
slots from learned weights; serving reads public facts and verified features,
without teacher strings or response templates. The shopper query changes order;
copy describes selected identity, color and price. New catalogs and unseen
combinations are unproven; earlier heldout-color failures remain preserved.
The status line explicitly labels catalog-trained text and distinguishes it
from copy generated earlier. The original fresh autoregressive implementation
and all its outputs remain available for historical comparisons. Its older
frontend receipts do not cover the current image-delivery integration.

The shop serves responsive WebP derivatives with a compact geometry manifest.
All original images remain unchanged, and gallery/color selection is preserved.
`node scripts/prepare-product-runtime-projection.mjs --check` verifies the
projection against the full source/derivative audit manifests.

## Measured status

The matched loaded-shop comparison completed 20 pairs with all 40 observations
retained. Generic first-three visibility and decoded images took p50/p95
64.3/65.1 ms; fresh local personalization took 281.1/297.7 ms. Paired additional
delay was 217.4/234.4 ms. These are loopback, already-loaded collection views
with browser-cache-eligible images, measured through the same visibility and
two-frame observer. They exclude cold navigation and physical paint. The local
arm produced all 20 fresh fields; the generic arm used static catalog copy.

An isolated host profile attributes 170.62 ms exclusive median to fresh Qwen
query encoding and 0.145 ms to the parallel text-head forward pass. Original
full-six p50/p95 was 192.78/202.37 ms; instrumentation added paired median
3.78 ms. See the [profile review](experiments/ranker/pretrained/STORE_FAST_PATH_PROFILE_V1_INDEPENDENT_REVIEW.md).
Neither a tiny head timing nor ranking alone describes complete storefront readiness.

The [balanced ranker audit](experiments/ranker/pretrained/function_v5_audit_v1/INDEPENDENT_AUDIT.md)
reproduces 29/32 correct first choices and 32/32 targets in six on independently
authored known-catalog queries. The prior ranker scores 14/32 and 29/32 on those
same requests. No hidden persona goals enter inference. The
[latest browser audit](experiments/ranker/pretrained/STORE_COMPACT_V5_BROWSER_INDEPENDENT_AUDIT_V1.md)
verifies eight navigation requests with all 20 factual fields and a separate
four-query loaded view: three fully visible cards with decoded images in
269.95 ms median. Navigation through full text plus six primary images takes
533.10 ms median in new contexts and 362.40 ms on reloads. These small development
samples preserve startup outliers and Vite errors; no physical-paint or
conversion effect is claimed.

The [audited parallel-head benchmark](experiments/ranker/pretrained/STORE_PARALLEL_TEXT_TRAIN_ALL_FRESH_V1_AUDIT.md)
completed 32 curated public requests, all with six cards and 20 factual fields.
Median backend first-three/full-six readiness was 193.43/193.50 ms, including
fresh query encoding, ranking, generation, guards and JSON. Offline top-one
relevance was 30/32; one notebook target was absent from six. These are trained
catalog results, excluding startup, network, browser and images. The first HTTP
request after resident startup took 456.81 ms for full copy. The separate
[browser baseline](experiments/ranker/pretrained/STORE_FULL_PERSONALIZED_CONTENT_BASELINE_V1.md)
measures navigation through all 20 fields and six decoded primary images:
497.80 ms median in four new contexts and 348.85 ms across four reloads. A loaded,
pre-scrolled view takes 237.95 ms median for three fully visible factual cards.
The hero leaves card copy below the fold during navigation; none of these
measurements establishes physical paint or human conversion.
The [native candidate comparison](experiments/ranker/pretrained/STORE_NATIVE_CANDIDATES_V1_RESULTS.md)
preserves slower fresh-AR and fast generated-earlier cache attempts with their
copy failures.

Earlier autoregressive measurements follow:

Warm six-card smoke requests recorded ordering around 0.2 seconds and fresh
copy payloads around 5 to 6 seconds. Ordering latency excludes copy; serialized
copy completion can include withheld fields. These are local inspected smoke
results, with startup separate from request timing. Browser instrumentation
records DOM commit plus two animation frames, not physical paint or image
readiness. The [native smoke summary](experiments/ranker/pretrained/store_demo_results/v1/summary.json)
and [qualitative audit](experiments/ranker/pretrained/STORE_DEMO_V1_QUALITATIVE_AUDIT.md)
preserve failures, including a notebook query that ranked the organizer first.

The [matched native comparison](experiments/ranker/pretrained/STORE_AR_COMPARISON_V1_RESULTS.md)
completed three six-card queries per system. Median full attempts took 21.20
seconds for conventional autoregressive generation and 5.94 seconds for the
hybrid. Both failed the frozen complete-copy readiness proxy. An independent
audit found valid conventional dollar wording falsely rejected, plus real
hybrid copy and ranking failures; the 3.57× attempt ratio is not a measured
useful-page speedup. The separate [v2 offline usefulness scorer](experiments/ranker/pretrained/store_usefulness_v2_candidate/README.md)
recognizes equivalent correct dollar wording and keeps optional headings/badges
separate from title/body completeness. It preserves all original outputs and
readiness timestamps; its retrospective scores are not new timing measurements.
The [native component profile](experiments/ranker/pretrained/STORE_DEMO_COMPONENT_PROFILE_V1_RESULTS.md)
attributes most copy time to suffix prefill and autoregressive steps. Profiling
adds synchronization overhead; the ordinary six-card attempt median was5.58s.

## Independent shopper evaluation

The [ten-pair packet](experiments/ranker/pretrained/store_shopper_eval_v1/README.md)
separates public recommender observations, private shopper needs and evaluator
gold. It fixes five AB and five BA orders, 24 actions and 120 seconds per arm.
The [shopper CLI](apps/runner/SHOPPER_EVAL.md) prepares assignments and aggregates
evidence without supplying gold answers to a shopper. A separate runner uses
isolated headed Chrome and the installed official Codex CLI with its existing
ChatGPT login for real screenshot-based Sol decisions. No extra API key or SDK
installation is required.

```sh
node --import tsx apps/runner/src/shopper-runner.ts plan --packet experiments/ranker/pretrained/store_shopper_eval_v1 --out /tmp/new-plan
node --import tsx apps/runner/src/shopper-runner.ts run --packet experiments/ranker/pretrained/store_shopper_eval_v1 --out /tmp/new-run --seed 11 --limit 2
```

Coordinate local-model runs with the model owner to keep GPU timing isolated.
The earlier [CUA pair](experiments/ranker/pretrained/store_shopper_eval_smoke_v1/README.md)
completed exploratory fake checkouts with protocol deviations. The first
[isolated Sol pair](experiments/ranker/pretrained/store_shopper_sol_smoke_v1/aggregate.json)
selected the bottle but timed out after repeated Add-to-cart actions. The
[repaired-feedback pair](experiments/ranker/pretrained/store_shopper_sol_smoke_v2/smoke-summary.json)
then completed both independently verified fake checkouts: generic 72.553s/12
actions and local-model 44.817s/8 actions. Both selected one Graphite bottle at
$36. The [full twenty-session suite](experiments/ranker/pretrained/store_shopper_sol_full_v1/summary.json)
is now complete: expected outcomes8/10 generic and7/10 local, including two
correct abstentions per arm. Timeouts, an interrupted session and incorrect
abstentions remain in their fixed denominators. This small synthetic suite does
not establish human conversion or comparative lift. Its browser timing entries
are request-weighted, include zero-copy abstentions, and are not ten comparable
fresh-copy samples.
See the [session contract](apps/runner/SHOPPER_CONTRACT_V1.md) for evidence rules.

## Development and architecture

| Area | Responsibility |
|---|---|
| `apps/storefront/` | Client-rendered shop, deterministic generic search, optional streamed local-model results, fake cart/checkout |
| `apps/web/` | Dashboard, comparisons and run/artifact inspection |
| `apps/api/` | Go HTTP API, bounded queue, SQLite metadata, mock evaluation and structured proposals |
| `apps/runner/` | Legacy scripted Playwright adapter plus separate shopper evidence contracts/CLI |
| `packages/contracts/` | Shared v1 schemas and local service addresses |
| `experiments/ranker/pretrained/` | Native model experiments, resident service, frozen data, audits and receipts |

```sh
pnpm typecheck
pnpm test
pnpm build
node --import tsx --test apps/runner/src/shopper-contract.test.ts apps/runner/src/shopper-eval-cli.test.ts
```

`pnpm test` includes the runner's source test files. The direct Node command
focuses on the shopper contract and CLI. `pnpm test:smoke` starts its own named
services and needs the Playwright browser; stop `pnpm dev` before running it.

Start with the [documentation guide](docs/README.md). The [project vision](VISION.md)
describes the broader goal; some implementation-status sections predate the
current integration. Historical experiment reports stay at their original
paths, indexed in [docs/archive](docs/archive/README.md).
