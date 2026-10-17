# Go local API ownership

`NewApp(AppOptions)` is the `httptest` entry point. `main.go` uses Go's `net/http` and binds only `127.0.0.1` on the validated `PORT` assigned by Portless (4311 only when run directly without `PORT`). `database/sql` plus `github.com/mattn/go-sqlite3` stores metadata in SQLite; the only Go dependency requires CGO and a C compiler. The default data directory is `../../data` from `apps/api`; `STOREFRONT_DATA_DIR` overrides it. `go run . seed` preserves existing runs; `go run . reset` removes only this app's database files and artifacts. Both refuse while an API process holds the directory lock.

Keep the v1 schemas and HTTP routes in `packages/contracts` stable. `internal/contracts` mirrors their JSON field names in Go. `fixtures/catalog.json` freezes the demo catalog, including disjoint development and held-out persona IDs. The API snapshots persona, split, baseline/variant IDs, configuration, seed and adapter into each immutable run ID. Lifecycle, events and results are then persisted under that ID. Comparison rows include only completed mock runs and keep splits separate.

Independent teammate modules:

- `internal/runner`: deterministic Go mock adapter and the browser subprocess bridge. `apps/runner` owns the small TypeScript Playwright task. Stdin is `{run, artifactDirectory}`; stdout is `{events, artifacts, result, error?}`. Browser task artifacts are restricted to `trace.zip`, `screenshot.png`, and `result.json`.
- `internal/evaluator`: scores recorded task observations; its current binary score is explicitly synthetic.
- `internal/optimizer`: evidence-linked configuration patches from development runs only. Held-out runs never enter optimization.
- `internal/recommendation`: optional deterministic ranking using explicitly supplied persona history.
- `internal/model`: server-side adapter interface; shipped mock makes no network calls and needs no key.

The in-process queue has concurrency one and a total capacity of twenty. Restarted pending/running jobs become failed while preserving snapshots. Browser work has a 90-second subprocess timeout. This is a foundation for contributions, with no distributed workers, inference service, or research-grade evaluator.

From this directory, `node ../../scripts/go.mjs test ./...` runs the Go checks and `node ../../scripts/go.mjs vet ./...` performs static checks. The root helper selects the local or installed Go toolchain and keeps Go caches inside the workspace.
