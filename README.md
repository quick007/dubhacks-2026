# Storefront Lab

A local DubHacks scaffold for testing a demo storefront with synthetic shopper personas. Baseline and structured variants share a fixed experiment, disjoint development/held-out personas, and a recorded seed. Everything runs without API keys.

See [VISION.md](VISION.md) for the product direction, demo workflow, evaluation methodology, and team workstreams.

Local model work is separate from the app: see the [synthetic ranking baseline](experiments/ranker/README.md) and [pretrained decision-head experiment](experiments/ranker/pretrained/README.md) for commands, data boundaries, and measured limitations.

**All results are synthetic.** Mock scores are deterministic rule outputs. Browser runs exercise a fake local checkout. Neither demonstrates human behavior or proven conversion lift.

## Run locally

Use Node 24.11+ in the 24.x line, or Node 26+ (tested: 26.9), pnpm 12.5.1, and Go 1.24+ (tested: 1.27.2). SQLite uses the small `go-sqlite3` driver and requires CGO plus a C compiler; this Mac already has Apple clang. No database server is needed. The Go helper uses `.tools/go/bin/go` when available, then system `go`, or the explicit `STOREFRONT_GO_BINARY` override. A checksum-verified official Go toolchain was installed only under this project’s ignored `.tools/` directory.

```sh
pnpm install
pnpm exec playwright install chromium
pnpm seed
pnpm dev
```

Open **http://dubhacks.localhost:1356** and click **Run mock comparison**. The dashboard polls for queued/running/completed runs and displays separate development and held-out comparisons. Inspect each run's steps, failure reasons, result artifact, and evidence-linked proposals. **Run browser smoke** runs one real Playwright path through the local demo shop, saving a trace and screenshot. No API key is used by either adapter.

| Service                                                    | Address                             |
| ---------------------------------------------------------- | ----------------------------------- |
| Dashboard, Vite+ + React + Tailwind + shadcn/ui primitives | http://dubhacks.localhost:1356      |
| Go net/http API                                            | http://api.dubhacks.localhost:1356  |
| Local demo shop                                            | http://shop.dubhacks.localhost:1356 |

`pnpm dev` starts [Portless](https://github.com/vercel-labs/portless) in **plain HTTP on unprivileged port 1356**, then starts the three named apps with dynamically assigned backend ports. All listeners bind only to loopback. The launcher owns the proxy and app processes and stops them on Ctrl-C. It refuses an occupied proxy port and never force-takes a route or kills another project. The shop accepts only demo interactions; its order button creates no purchase and collects no payment information.

The project uses isolated state in ignored `.cache/portless/`. The launcher explicitly disables TLS, LAN mode, hosts-file syncing and network sharing, and never installs an OS service or runs certificate trust commands. It leaves the existing proxy on 1355 and global `~/.portless` state alone. No sudo, trusted CA, `/etc/hosts` edit or global Portless installation is needed. Use Chrome, Edge or Firefox, whose built-in `.localhost` resolution works without hosts changes; Safari can depend on system DNS.

Stable names live in `portless.json`; shared URLs and proxy port live in `packages/contracts/src/local.ts`. Both Vite+ apps honor `PORT` and bind `127.0.0.1`; the Go API also reads and validates `PORT` while keeping loopback binding. Vite allows the named hosts and sends HMR WebSockets through the named proxy. Dashboard `/api` requests use the named API route through loopback with the correct `Host` header, avoiding macOS Node DNS failures and proxy loops. The browser runner permits only the named demo-shop origin.

## Verify and reset

```sh
pnpm typecheck          # standard TypeScript tsc + Go vet
pnpm typecheck:tsc-ts   # requested vercel-labs/tsc-ts compiler, separate check
pnpm check              # Vite+ formatting, lint and its own type checking
pnpm test
pnpm build
pnpm test:smoke
```

The small test suite checks shared schemas, runner protocol, storefront settings, and the Go API mock lifecycle/persistence. `pnpm typecheck:tsc-ts` invokes the published `tsc-ts` package; it is distinct from Vite+’s built-in TypeScript Go checks. Standard TypeScript checks remain available too. Playwright starts all three services itself and checks the named dashboard comparison, fake checkout, API browser trace/artifact path, and real CSS hot updates through both frontend HMR WebSockets. Node request checks use loopback transport plus the named `Host` header, while browser navigation uses the named URLs. Stop `pnpm dev` before `pnpm test:smoke`, since tests use the same project proxy port. Browser installation is only needed for browser runs and smoke tests; mock runs require no browser binary.

```sh
pnpm seed       # idempotent initial catalog; preserves runs
pnpm reset      # clears local demo metadata/artifacts and reseeds
```

Stop the API before seed/reset; the API data lock prevents concurrent mutation. The default metadata and artifacts live in `data/` at the workspace root and are gitignored. `STOREFRONT_DATA_DIR` overrides the data root; relative values resolve from the API's working directory. Reset deletes only the app-owned database filenames and artifacts in that selected root. Smoke tests use a fresh `data/smoke/<timestamp>-<pid>/` directory for each invocation. Test isolation for API unit tests uses temporary directories.

Open a downloaded trace with `pnpm exec playwright show-trace /path/to/trace.zip`. Browser and mock artifacts are saved locally. Server restarts mark interrupted in-process jobs failed; the queue does not resume work. This is a single-process prototype.

## Teammate ownership and frozen interfaces

`packages/contracts/src/index.ts` is **v1 of the shared boundary**. Coordinate any contract or endpoint changes before editing it. It exports Zod schemas and types for `Persona`, `Experiment`, `Variant`, `Run`, `StepEvent`, `Artifact`, `EvaluationResult`, `ProposedChange`, catalog, requests, and comparison rows. API route documentation is at the bottom of that file.

| Owner area                | Files and responsibility                                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Dashboard                 | `apps/web/`: polling, run inspection, comparison; consume contracts                                                                        |
| Orchestration/persistence | `apps/api/app.go`, `queue.go`, `store.go`: validated HTTP, bounded queue, SQLite and artifact metadata                                     |
| Runner                    | `apps/api/internal/runner/`: Go `Adapter`, deterministic mock, JSON subprocess adapter; `apps/runner/`: TypeScript Playwright browser path |
| Evaluator                 | `apps/api/internal/evaluator/`: Go evaluator interface, explicit synthetic scores                                                          |
| Optimizer                 | `apps/api/internal/optimizer/`: Go optimizer interface, evidence-linked structured config proposals from development runs                  |
| Recommendations           | `apps/api/internal/recommendation/`: Go recommendation interface, ranking stub using explicitly supplied persona history                   |
| Model integration         | `apps/api/internal/model/`: Go model interface, no-network mock; future API provider belongs on server only                                |
| Demo shop                 | `apps/storefront/`: allowed config query parameters and stable fake-checkout roles                                                         |
| Shared contracts          | `packages/contracts/`: schema owner, coordinate before edits                                                                               |

The Go API invokes the TypeScript browser runner only when needed, sending one JSON request on stdin and receiving one JSON result on stdout. This is a child process, not another long-running service. See `apps/runner/README.md` for the protocol. The runner, evaluator, optimizer, recommendation, and model adapters are intentionally small boundaries for teammates. The recommendation stub can rank supplied items/history; the scaffold does not implement a user-history ingestion service or a learned recommendation model. The browser path is a smoke check, not a general-purpose autonomous shopper.

Run snapshots record immutable random 128-bit IDs, experiment/baseline/variant IDs, persona and split, config, seed, adapter, and contract version. The API exposes no mutation of those snapshots. Only status, steps, artifacts, and evaluation advance. Results are grouped by variant and persona split; optimizer suggestions use development evidence only. Held-out results are never fed to the optimizer. Comparison rows aggregate completed mock runs; reruns add observations rather than replacing them. Scripted browser smoke runs stay separate from mock comparisons because they use a different task policy.

Variants are validated config (`headline`, `showShipping`, `recommendationOrder`), never generated code. There is no endpoint for arbitrary URLs, script execution, deployment, authentication, cloud storage, checkout submissions, or model credentials. `.env.example` reserves a future server-side model boundary; it is optional and requires no secret. No deployment, Docker, Redis, or external database is needed.

## Toolchain and repository handoff

Both frontends use [Vite+](https://viteplus.dev/guide) 1.3.0 through the local `vp` CLI for dev/build and shared checks. The scaffold was migrated with the official `vp migrate` workflow. [tsc-ts](https://github.com/vercel-labs/tsc-ts) is pinned separately to its published package and invoked by `pnpm typecheck:tsc-ts`; it is not an alias for Vite+’s checker. No global Vite+ installation is required. Go source is formatted with gofmt and checked with Go vet/tests. Go dependencies are recorded in `apps/api/go.mod` and `go.sum`; JS dependencies use the pnpm lockfile.

Project checkout: `/Users/lukas/Documents/Projects/dubhacks-2026`, with origin `https://github.com/quick007/dubhacks-2026.git`. The checkout contained only `.git` with a clean status immediately before integration. Source, configuration, documentation, and lockfiles were added without overwriting existing files; `.git` was preserved. Dependencies and a workspace-local Go toolchain were installed fresh here. No database, artifact, secret, dependency directory, or generated build output was copied from the checkpoint. The initial scaffold was built in a temporary Codex task workspace, then integrated here for team contribution. After verification and a confirmed push, the duplicate task scaffold is removed; useful local browser evidence is preserved under ignored `data/checkpoint-evidence/`. This checkout is the project source of truth.
