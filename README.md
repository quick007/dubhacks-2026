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

In the dashboard, open **All sessions** and choose **Run a round** to queue
the deterministic mock comparisons. The separate legacy Playwright adapter
exercises a scripted fake checkout and requires
`pnpm exec playwright install chromium`. Its first-item purchase script does
not represent an independent shopper choosing a suitable product.

## Enable the local model

The verified native environment uses Apple silicon, Python 3.14.7, MLX 0.32.4
and MLX-LM 0.32.2. One frozen Qwen3.5-9B four-bit backbone supplies features to
a trained ranking head and runs an adapted decoder for copy. The resident
service handles requests serially on `127.0.0.1:8766`.

This optional path requires the existing local environment, pinned Qwen weights
and completed serving checkpoints. They are not supplied by a fresh Git clone:

| Local prerequisite | Location |
|---|---|
| Python environment | `.tools/pretrained-venv/` |
| Pinned model snapshot | `.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53/` |
| Actual-store ranking run | `data/ranker/pretrained/actual-store-function-v3-attempt-1/` |
| Copy adapter run | `data/ranker/pretrained/actual-store-copy-fit-v2-attempt-1/` |

With those artifacts available, start one resident worker from the repository
root. Choose a new output directory for each service lifetime:

```sh
.tools/pretrained-venv/bin/python -B -m experiments.ranker.pretrained.store_demo_service \
  --width 10 --output data/ranker/pretrained/store-demo-new-session
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

## Measured status

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
