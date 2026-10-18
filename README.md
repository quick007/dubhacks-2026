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

A conventional autoregressive LLM comparison has a CPU preparation driver.
The paired native comparison has not run, so there is no measured
LLM speedup ratio.

## Independent shopper evaluation

The [ten-pair packet](experiments/ranker/pretrained/store_shopper_eval_v1/README.md)
separates public recommender observations, private shopper needs and evaluator
gold. It fixes five AB and five BA orders, 24 actions and 120 seconds per arm.
The CPU-only [prepare/aggregate CLI](apps/runner/SHOPPER_EVAL.md) does not launch
browsers or models and never supplies gold answers to a shopper assignment.

One actual Chrome bottle-shopping pair completed exploratory fake checkouts.
Viewport, timing and durable-evidence deviations keep it outside the formal
benchmark. All 20 formal slots remain unattempted. See the
[smoke evidence](experiments/ranker/pretrained/store_shopper_eval_smoke_v1/README.md)
and [session contract](apps/runner/SHOPPER_CONTRACT_V1.md) for the recorded limits.
The contract and CLI have CPU tests; fixture sidecars do not count as browser
execution or conversion evidence.

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
