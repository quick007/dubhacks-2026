# Shopping telemetry and training export

Normal shop telemetry has a separate ingestion path from synthetic experiment reports. The frontend owns automatic collection, deployment gating, a privacy notice/consent setting, opt-out, and DNT/GPC handling. The Go backend validates anonymous structured shopping events, persists receipts locally before acknowledging, and synchronizes with the existing team Tiger database in the background. No model is trained or deployed by ingestion.

The pipeline is:

1. Frontend emits `fieldwork.shopping-batch.v1` to `POST /api/shopping-events`.
2. Backend writes an atomic local receipt and returns `202`; the asynchronous worker commits an immutable Tiger batch and unique event receipts.
3. A local maintenance command exports versioned observations with provenance and stable whole-session splits.
4. A deterministic builder joins explicit choices to their preceding catalog views to produce `fieldwork.shopping-choice.v1` JSONL for a later model-training job.

## Configuration and environment boundaries

The normal local app has telemetry **disabled**. No browser secret or Tiger CLI token is needed. The existing ignored `TIGER_DATABASE_URL` is consumed only by the backend, with the existing TLS validation policy and bounded connection/statement timeouts.

Backend configuration:

| Variable | Meaning |
| --- | --- |
| `SHOPPING_TELEMETRY_ENABLED=1` | Explicitly enable intake and async cloud sync. Default off. |
| `SHOPPING_TELEMETRY_ENVIRONMENT` | Required `production`, `development`, or `test`; incoming classification must match. |
| `SHOPPING_TELEMETRY_ORIGINS` | Exact comma-separated browser origins; no wildcard/CORS credentials. Production requires HTTPS and rejects localhost. |
| `SHOPPING_TELEMETRY_ALLOW_SYNTHETIC=1` | Explicitly permit labeled synthetic traffic. Required for development/test; real traffic is rejected outside production. |

Frontend trusted build configuration is documented in [the storefront README](../storefront/README.md). Production collection requires both frontend deployment configuration and matching server policy. Request fields cannot promote test/development traffic into the real training dataset. Classification establishes collection policy eligibility; client observations are not independently verified human actions.

The primary API registers intake alongside existing routes. For an isolated local receiver without opening the store's SQLite database, from `apps/api`:

```sh
SHOPPING_TELEMETRY_ENABLED=1 SHOPPING_TELEMETRY_ENVIRONMENT=test \
SHOPPING_TELEMETRY_ALLOW_SYNTHETIC=1 \
SHOPPING_TELEMETRY_ORIGINS=http://127.0.0.1:4329 PORT=4330 \
node --env-file=../../.env ../../scripts/go.mjs run ./cmd/shopping-server
```

This binds loopback only. No production hosting or deployment is created by this work.

## Schema and intake

The explicit maintenance migration is [002_shopping_events.sql](migrations/002_shopping_events.sql). It creates only a new `shopping_events` namespace, separate from experiment results and the existing store. The command connects first, checks that the namespace is absent, and refuses existing namespaces:

```sh
node --env-file=../../.env ../../scripts/go.mjs run ./cmd/shopping-export --apply-v1
```

The team database was initialized with this migration during implementation. Application startup never runs DDL.

`fieldwork.shopping-batch.v1` contains UUID `batchId` / anonymous session-scoped `sessionId`, explicit `{environment,traffic}`, epoch-millisecond `startedAt`, and up to 100 ordered events. Each event has UUID `eventId`, integer `sequence`, epoch-millisecond `at`, integer `elapsedMs`, action, structured `{mode,order,shipping,modelVersion}` context, and action-specific allowlisted data. Server validation uses the public sixteen-product/variant catalog frozen in `internal/shoppingevents/catalog.json`; refresh it deliberately when the public catalog changes.

Supported actions are session start/end, catalog view, a product impression visible at least half for 250 ms, search source category, variant selection, cart add/clear, demo checkout, and model readiness. The frontend contract is in `shopping-events.ts`. Queries, typed inputs, names, emails, account/payment fields, persona details, raw model output, request IDs, arbitrary URLs, and unknown event data fields are rejected. Checkout remains an observed **demo** objective; it is never a real purchase label. Model readiness retains its after-paint-opportunity scope, excluding physical paint and image transfer.

The complete body is limited to 512 KiB and an eight-hour session. Optional `replay:{schema:"fieldwork.sanitized-replay.v1",events:[...]}` has at most 100 rrweb events and 384 KiB. Replay passes a second server sanitizer: masked text, allowlisted node/attribute/numeric fields, public product image paths, public stylesheet class names, stripped CSS URLs/comments/imports/content/quoted strings, and no input/plugin/network/console/media sources. Node IDs and mutation ordering remain intact. Replay fidelity excludes private UI and stripped metadata.

Replay bytes live in separate local `.replay` files and Tiger `replay_chunks`, expire after 24 hours, and are pruned by the running worker every minute (at most 100 expired cloud rows per sweep). Expired replay is unavailable to maintenance reads even before deletion. Deletion resumes when the receiver restarts; there is no hosted scheduled job. Replay is never included in an observation archive, training export, or example.

`202 {batchId,durableLocal:true,duplicate,cloud:"asynchronous"}` confirms local file and directory `fsync`, not cloud completion. Same-ID identical normalized evidence is idempotent; changed evidence returns `409`. PostgreSQL also rejects reused event IDs or session sequence numbers across batches. A transaction commits batch, event identity receipts, and optional replay together. `GET /api/shopping-events/status` reports only operational counts, including pending uploads; it exposes no raw session or replay data.

The local spool is capped at 10,000 batches / 256 MiB and returns `429` at capacity. Already accepted observations are retained during database outages. Cloud operations have five-second contexts; browser fetch/retry is independent of rendering/model inference. HTTP errors contain no database credential details. Async sync errors preserve local receipts and retry each minute or on new intake. There is no public dataset-export route.

## Reproducible training inputs

From `apps/api`, choose a fixed exclusive `--until` cutoff to repeat a snapshot. Outputs are created exclusively, mode `600`, and failed exports remove incomplete outputs:

```sh
node --env-file=../../.env ../../scripts/go.mjs run ./cmd/shopping-export \
  --since 2026-01-01T00:00:00Z --until 2027-01-01T00:00:00Z \
  --max-batches 1000 --out ../../data/shopping-observations-v1.jsonl

node scripts/shopping-training.mjs \
  --input ../../data/shopping-observations-v1.jsonl \
  --out ../../data/shopping-choices-v1.jsonl
```

The first command defaults to `production` + `real` only. Synthetic verification requires explicit `--include-synthetic` there **and** `--allow-synthetic` in the second command. Both outputs include separate manifests, versioned provenance, batch/event/session identities, source SHA-256, environment/traffic, source model-version scope, and derivation/split versions. The first manifest records the received-at cutoff/row counts; the choice manifest records its input digest. The batch limit fails rather than silently truncating. File names are examples; use a new output path for each snapshot.

`fieldwork.training-observation.v1` keeps the validated event in `observation` and nullable derived signals in `derived`. All events remain available as raw structured observations. A view, search, timeout, missing click, and absence from a catalog are not dislike labels. `selection` and `cart-add` are explicit positive-action signals, not proof of preference or conversion.

The choice builder joins by session/view identity across out-of-order batches and requires a preceding catalog view containing the chosen product. Color-selection examples preserve the original rendered variant and the separately validated chosen variant with an explicit `choiceScope`; cart-add requires its chosen variant to match the rendered candidate variant. Missing, conflicting or later candidate views yield no derived example while original observations remain preserved. Candidate positions are rendered-catalog denominators, not proof every candidate was visible. Explicit product impressions remain separately available in observations. All `negativeLabels` arrays are empty; this builder does not invent pairwise negatives from nonclicks. The training input is defined by [shopping-choice.v1.schema.json](scripts/shopping-choice.v1.schema.json). A later learner can use its candidate list, chosen IDs, context and positive signal under an appropriate observational objective.

Every session hashes to one stable train/validation/test split using `fieldwork.session-split.v1`, shared across all batches/examples. Do not randomly split individual events or reuse evaluation sessions for optimization. These are observational logs, not randomized causal or human conversion evidence.

## Checks

```sh
node ../../scripts/go.mjs test -race ./...
node ../../scripts/go.mjs vet ./...
node --test scripts/shopping-training.test.mjs
```

Backend tests cover disabled/environment/origin gates, synthetic exclusion, privacy filtering, offline durability and restart replay, immutable retry/conflict behavior, separate replay expiry, and whole-session splits. Training-builder tests cover out-of-order joins, explicit synthetic opt-in, split mismatch/identity rejection, no inferred negatives, and missing/conflicting exposures.

The automatic browser → real backend → Tiger → training-export path was verified using explicitly labeled synthetic traffic only. The final browser session produced 17 shopping observations and 29 sanitized rrweb events; the backend replay rebuilt sixteen products with original public image URLs. Its Tiger replay matched the local accepted replay using an explicitly documented, recursively sorted Node JSON hash, and observation digests matched on readback. Cloud retries were idempotent and pending uploads reached zero. Opt-out, persisted opt-out, DNT, and privacy canaries passed without model calls. Across both synthetic verification sessions, the default real-traffic export and default choice builder emitted zero rows/examples; explicit synthetic flags produced 34 observations and four positive-choice examples. See [shopping-telemetry.verification.json](shopping-telemetry.verification.json). This verifies the pipeline, not training quality or conversion lift; no production deployment was performed.
