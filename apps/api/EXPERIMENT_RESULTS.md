# Experiment results integration

The existing local Go API accepts saved `shopper-evidence-aggregate.v1` reports. Each immutable report keeps all twenty roster rows, both ten-session denominators, the actual `caseId` / `expectedRunId` / `session.run` IDs, raw sessions, evaluator findings, artifact verification results, errors, and harness arm summaries. `generic` retains the store's deterministic search behavior; `local-model` is the personalized arm. This adapter does not run shoppers, models, or an evaluator. Intake checks identity and transport structure; the harness owns evidence validation and outcome calculations.

The original SQLite store, inference path, runner timings, and presentation UI are independent. No automatic schema migration runs at application startup. No extra local database or cloud service is required.

## Local configuration

The only adapter credential is `TIGER_DATABASE_URL` in the gitignored root `.env` (mode `600`). `.env.example` contains an empty placeholder. The API's `dev` command loads this file; compiled binaries consume process environment variables. Existing `DATABASE_URL` and Tiger CLI/OAuth tokens are not used by this integration. Never put the connection string in a `VITE_` variable.

From the repository root:

```sh
pnpm --filter @storefront/api dev
```

Restart the API after configuring the environment. Without the variable, intake and local queries still work, and receipts remain pending. With a valid variable, sync starts asynchronously. There is no connection wait during API startup. A malformed URL is rejected with a redacted configuration error.

TLS is mandatory with no plaintext fallback. The adapter honors `sslmode=require` (encrypted transport) and `sslmode=verify-full` (certificate and hostname verification). For the latter, supply the service's trusted CA using `sslrootcert=/absolute/path/to/ca.pem`. Connections time out after three seconds; each repository operation has a five-second context and a four-second PostgreSQL statement timeout. Errors exposed by the API never include connection details.

## Schema setup

First verify the intended existing hosted database with a metadata-only check, from `apps/api`:

```sh
node --env-file=../../.env ../../scripts/go.mjs run ./cmd/tiger-check
```

The check reports TLS mode and columns in `public` / `experiment_results`; it reads no table rows and changes no schema. There was no experiment schema to reuse on the configured service. The versioned migration is [001_experiment_results.sql](migrations/001_experiment_results.sql). It creates only `experiment_results.reports` and its index, without extensions or hypertables.

For a new, verified target only:

```sh
node --env-file=../../.env ../../scripts/go.mjs run ./cmd/tiger-migrate --apply-v1
```

This explicit command checks connectivity first and refuses an existing `experiment_results` namespace. Review any existing teammate schema manually rather than rerunning it. It does not replace or modify storefront tables. On the service configured during this integration, this migration has already succeeded.

## Intake contract

`POST /api/experiment-results`, `Content-Type: application/json`, maximum 8 MiB:

```json
{
  "version": "experiment-results.v1",
  "reportId": "unique-immutable-snapshot-id",
  "experimentId": "actual-store-isolated-shopper-v1",
  "synthetic": true,
  "humanConversionClaim": false,
  "report": "Replace this string with the entire saved aggregate JSON object"
}
```

Report and experiment IDs use 1–160 letters, digits, underscores or hyphens and begin with a letter or digit. Recorded sessions must match the envelope experiment ID and roster case/run/arm IDs. Keep all missing, unattempted, invalid and failed rows. `humanConversionClaim` must be false in both envelope and report. Saved evidence is synthetic development evidence; task completion does not establish human conversion, causal lift or unseen generalization.

An optional post-run exporter is owned by `apps/runner`: `src/shopper-export.ts`, invoked from the root with `node --import tsx apps/runner/src/shopper-export.ts`. It reads a completed aggregate after shopper timing has ended and sends it only to the loopback API. Its CLI takes `--report PATH --report-id ID` and infers `experimentId` from recorded sessions; supply `--experiment-id ID` for an empty report. Browser-origin writes are rejected; no CORS or credentialed browser intake is enabled.

`202` returns `{reportId, sha256, durableLocal:true, duplicate, cloud:"asynchronous"}`. It confirms a flushed local receipt, **not a cloud write**. Repeating identical JSON evidence (including reordered object keys / whitespace) is idempotent. Different evidence under the same local report ID returns `409`; revised evidence needs a new snapshot ID. PostgreSQL enforces report identity by SHA-256 as well. A conflict on another client's cloud ID remains pending locally and is visible as a sync error; it never overwrites the cloud report.

Receipts are atomically written with file and directory `fsync` under `STOREFRONT_DATA_DIR/experiment-results/`, with files mode `600`. A `.synced` marker is written only after PostgreSQL acknowledges the idempotent write. Receipts remain archived after sync. Restart replays pending reports; outages retain every accepted receipt. Background retry waits sixty seconds after failure, with new intake waking the worker. A failed report does not prevent independent pending reports from syncing. Original harness reports and artifacts are never deleted.

## Query interface for presentation

| Request | Response content |
| --- | --- |
| `GET /api/experiment-results/status` | Cloud configuration state, local/pending counts, redacted last sync error. `configured` is not proof of connectivity. |
| `GET /api/experiment-results/reports?experimentId=ID&limit=50` | Run-history snapshot headers; optional experiment filter; limit 1–100. |
| `GET /api/experiment-results/reports/REPORT_ID` | Entire immutable envelope, digest and received-at timestamp. |
| `GET /api/experiment-results/runs?reportId=REPORT_ID` | All twenty original rows, including missing/invalid outcomes and raw evidence. |
| `GET /api/experiment-results/comparison?reportId=REPORT_ID` | Harness summaries keyed by `generic` and `local-model`; fixed denominators and formal/exploratory distinctions remain intact. |
| `GET /api/experiment-results/latency?reportId=REPORT_ID` | Original duration arrays and precision labels; count/min/max/mean/nearest-rank p50/p95 per arm for session elapsed and each readiness stage, with observed versus verified-artifact samples separate. |

View responses carry `reportId`, `experimentId`, `synthetic:true`, `humanConversionClaim:false`, `source`, and `data`. Raw report responses carry `result` and `source`. History returns `reports`, `limit`, and `source`. Select one explicit report snapshot; revised snapshots are never added together as additional sessions.

The local archive is preferred for a known report. Otherwise reads use parameterized PostgreSQL queries. History merges cloud and local headers, falling back to `source:"local-fallback"` on a database outage; that source may omit reports held only in the cloud. A cloud-only report returns `503` while unavailable, and an unknown report returns `404` when its repository is reachable. Local receipts are not proof that cloud contents match; consult sync status.

Session elapsed may be measured, estimated, or an upper bound: keep the original precision labels when presenting the numerical summaries. Readiness describes a browser after-paint opportunity and excludes physical paint and image transfer. Missing measurements produce count zero and null summaries. Attempt completion is not useful-page latency. Do not infer success from upload time, `202`, an unattempted slot, or a timeout.

## Verification

Credential-free tests use an in-memory repository and temporary local archives. They cover offline acceptance/readback, restart replay, immutable retry/conflict behavior, TLS configuration, rejected claims/identities, browser-origin protection, preserved missing/formal states, timing precision, and empty latency samples.

From `apps/api`:

```sh
node ../../scripts/go.mjs test -race ./...
node ../../scripts/go.mjs vet ./...
```

The real saved Sol smoke aggregate was also exercised through local HTTP handlers with cloud sync disabled: two recorded timeouts, eighteen unattempted sessions, zero formal sessions, identical retry accepted and changed evidence rejected. This transport check makes no shopper-success claim.

After explicit upload approval, both immutable `store-shopper-sol-smoke-v1` and `store-shopper-sol-smoke-v2` were uploaded and read back from Tiger Data. A fresh empty local archive returned `source:"tiger"` for runs, comparison, and latency; remote history found both snapshots. Cloud digests matched the accepted local digests, repeated PostgreSQL writes succeeded without changing identity, and pending reports reached zero. V1 preserves two timeouts/zero formal sessions; v2 preserves two completed simulated checkouts/two formal sessions. Each still contains eighteen unattempted slots. See [experiment-results-verification.json](experiment-results-verification.json) for the credential-free verification record. These smoke observations do not establish conversion lift.
