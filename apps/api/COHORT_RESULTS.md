# Verified checkout cohorts in Tiger and the presentation

The original twenty-slot smoke exporter remains unchanged. A completed six-panel cohort uses `shopper-checkout-cohort.v1` inside the same `experiment-results.v1` envelope and existing `experiment_results.reports` table. No migration or new service is required.

Each immutable cohort preserves its 120-slot plan, six complete twenty-slot audited reports (each with its original experiment ID), the exact combined `shopper-synthetic-metrics.v1` output, the server-outcome index, and original source-file paths/SHA-256 values. Optional slide metrics are archived verbatim. Missing/failed/unattempted slots remain in their assigned denominators. Large traces, screenshots and raw browser/WebSocket logs are not bundled. Intake checks paired identities and source bindings; the harness owns outcome validation and metric definitions.

## Export after the cohort is finalized

Required files under the cohort directory are `cohort-plan.json`, six `packet-N/aggregate-audited.json` files, `server-outcomes.json`, and `shopper-metrics.json` (legacy `metrics.json` is also accepted, but having both is ambiguous and rejected); optional `slide-metrics.json` is preserved if present. Build the server index after auditing, since it binds exact report bytes. The source packets must match the frozen plan.

From the repository root:

```sh
node --import tsx apps/api/scripts/checkout-cohort-export.mjs \
  --cohort experiments/ranker/pretrained/store_shopper_cohort_v2 \
  --report-id store-shopper-checkout-cohort-v2-final \
  --experiment-id actual-store-checkout-cohort-v2 \
  --out data/store-shopper-checkout-cohort-v2-envelope.json
```

Preparation rehashes the audited files and runs the unchanged CPU harness analyzer again, including checkout receipt artifact verification and exact server-index/gold source bindings. It refuses changed metrics. The evaluator inputs are used only to verify this evaluation, never to generate histories or train a model. The exporter reads no credentials or environment file, dispatches no browser/model calls, and posts nothing unless `--post` is explicitly added. It creates the envelope exclusively with mode `600`; use new snapshot IDs/output paths for revised evidence.

With the locally configured API running, add `--post` to the preparation command. The only allowed destination is `http://127.0.0.1:4311/api/experiment-results`, with redirects disabled. `202` confirms durable local acceptance; cloud synchronization remains asynchronous. Keep the receipt digest and verify the subsequent Tiger read before declaring a cloud write complete. The fixed 120-slot denominator is never truncated to meet the 8 MiB envelope limit.

The normal API registers these routes after an ordinary API restart; no checkout receiver restart is needed. To inspect/upload/read experiment results in isolation without opening SQLite, from `apps/api`:

```sh
node --env-file=../../.env ../../scripts/go.mjs run ./cmd/experiment-results-server
```

This receiver uses the existing private `TIGER_DATABASE_URL`, binds only `127.0.0.1:4311`, and exposes only experiment routes. It cannot replace the deck's ordinary catalog/run API. `EXPERIMENT_RESULTS_DIR` can select a private spool; default is repository `data/experiment-results`. Stop it before starting the normal API on that port. Keep the frozen local checkout service on 4331 unchanged while its evaluation runs.

## Public presentation data contract

`GET /api/experiment-results/presentation?reportId=ID&source=tiger` performs a fresh parameterized Tiger query with bounded timeouts. It does not fall back to the local spool. Missing reports return `404`; unavailable Tiger returns `503`; legacy twenty-slot reports return `422`. Omitting `source` on this presentation endpoint also selects Tiger. No credentials reach the browser.

The successful shape is:

```text
{
  version: "experiment-presentation.v1", reportId, experimentId,
  sha256, receivedAt, source: "tiger", cloudAcknowledged: true,
  synthetic: true, humanConversionClaim: false,
  data: {
    version: "shopper-checkout-cohort.v1", fixedCases: 60, fixedSessions: 120,
    metrics: <verbatim shopper-synthetic-metrics.v1>,
    slideMetrics: <verbatim optional source, otherwise null>,
    provenance: {
      plan: {path, sha256},
      sourceReports: [{path, sha256, experimentId}, ...six sources],
      metrics: {path, sha256}, serverOutcomes: {path, sha256}
    }
  }
}
```

The data-only frontend adapter is `apps/web/src/lib/presentation-results.ts`: `fetchPresentationResults(reportId, {signal})` selects this exact route, checks Tiger acknowledgement, synthetic scope, fixed denominators, outcome rates/revenue, and source hashes, and preserves the raw metric objects/limitations. It supplies no placeholders or fallback metrics. The presentation owner binds existing visual slots; this integration does not add layout, labels, banners or explanatory text.

Use `metrics.arms.generic` / `metrics.arms["local-model"]`. Intent-correct success includes correct abstention; synthetic checkout is separate. `revenue` is valid-scored merchandise revenue, while `allServerConfirmedOrders` also retains accepted demo orders that failed UI/scoring gates. Neither is real revenue. Keep `pairedDifferences` and original uncertainty/limitations attached to their metrics. Browser readiness is an after-paint opportunity, not physical paint or image transfer. Model/native benchmarks remain separate evidence.

Existing raw report/runs/comparison/latency routes accept `source=tiger` for strict cloud readback too. For cohorts, runs retains all 120 original rows, comparison returns the combined metric arm objects, and latency returns the harness's measured-session/readiness distributions verbatim. Original smoke IDs are never promoted or pooled into this cohort.

Validation: full Go race suite/vet, `node --import tsx --test apps/api/scripts/checkout-cohort-export.test.mjs apps/web/src/lib/presentation-results.test.ts`, and web typechecking. Tests use credential-free protocol fixtures, including an offline Tiger with an available local spool to prove there is no fallback. Actual cohort upload/readback waits for finalized source files and is recorded separately from these tests.
