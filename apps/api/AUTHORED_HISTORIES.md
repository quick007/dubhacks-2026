# Authored public histories: Tiger persistence and model export

The illustrative history dataset is stored as `authored-shopping-history.v1` in the existing immutable experiment-report table, under experiment `authored-public-history-v1` and report `authored-public-history-v1-200profiles`. It contains 200 anonymous profiles in 100 groups, 1,572 authored history events, and ten named demo people excluded from fitting. The frozen split assigns whole groups: 160 train profiles and 40 heldout profiles. These are agent-authored synthetic examples, not browser observations, real customers, orders or conversion measurements.

The adapter imports only five public files: `profiles.public.json`, `demo-people.json`, `split.json`, `catalog-binding.json`, and `PUBLIC_SPLIT_FROZEN.json`. It also retains the generation manifest's digest and source hashes. It never opens private train/heldout annotations. It verifies public-file hashes and the pre-annotation split receipt. The API validates the exact public request shape, 4–12 ordered authored events, public catalog IDs/names/categories/colors, external items without a store SKU, unique profile/session identities, and the complete group split. Hidden target fields and observed-traffic sources are rejected. The catalog binding is pinned in `internal/experimentresults/seed-catalog.json`.

The import and export commands require explicit authored-synthetic opt-in. From the repository root, with the configured normal API or isolated experiment receiver on 4311:

```sh
node apps/api/scripts/authored-history.mjs import \
  --seed-dir PATH_TO_FROZEN_SYNTHETIC_HISTORY_V1 \
  --report-id authored-public-history-v1-200profiles \
  --out data/authored-public-history-v1-envelope.json \
  --include-authored-synthetic --post

node apps/api/scripts/authored-history.mjs export \
  --report-id authored-public-history-v1-200profiles \
  --out NEW_PRIVATE_EXPORT_DIRECTORY \
  --include-authored-synthetic
```

Omit `--post` to prepare an envelope without uploading. The exporter reads only the fixed local API, not credentials. Import `202` confirms local durability, not cloud completion. Export requires a fresh `source=tiger` read and never falls back to a local receipt. Report identity is immutable/idempotent; revised content requires a new report ID. Output names/directories are exclusive, with private permissions.

Tiger stores JSON structurally, so exported files use UTF-8 JSON with recursively sorted keys and preserved array order. Their byte hashes differ from the original pretty-printed authoring files. Export `manifest.json` records both new `files` hashes and `originalSourceSha256`, the accepted report digest, source manifest/code hashes, and explicit authored-synthetic eligibility. The original pre-label split receipt is retained unchanged and refers to the original file hashes. Verify parsed document equality as well as each set of hashes.

The completed round trip is recorded in [authored-history.verification.json](authored-history.verification.json). All five public documents read from Tiger matched their frozen originals structurally, and the separate ten-person endpoint succeeded. A fresh default real-traffic telemetry export after import still returned zero batches/rows/sessions. Authored histories never enter `shopping_events`; default production/real observation export and choice construction remain unchanged. Their eligibility is only for an explicitly authored-synthetic model experiment.

The public person endpoint is `GET /api/experiment-results/demo-people?reportId=authored-public-history-v1-200profiles&source=tiger`. It returns `version: authored-demo-people.v1`, `datasetVersion`, report ID/digest/received-at, `source: tiger`, `synthetic: true`, `origin: agent-authored`, `browserObserved: false`, `realCustomerData: false`, `humanConversionClaim: false`, the ten public `people`, and source/split digests. It returns no anonymous fitting profiles or target annotations. Use a same-origin proxy to 4311; no new cross-origin access is enabled. Keep the active evaluation's separate checkout receiver on 4331 unchanged.

The model owner consumes the actual Tiger export's public requests and joins separately frozen local train/heldout annotations by profile ID. Group membership, public/label isolation and demo exclusion must remain intact. Native feature extraction, fitting, heldout evaluation and serving promotion are separate evidence owned by that lane; persistence/export alone does not establish training or model quality. Seed histories must not be presented as recorded shopper behavior or pooled with the independent checkout cohort. The shopper comparison/presentation routes reject these seed reports as evaluation metrics.

Credential-free checks: `node --test apps/api/scripts/authored-history.test.mjs`; from `apps/api`, `node ../../scripts/go.mjs test -race ./...` and `node ../../scripts/go.mjs vet ./...`. They cover private-label isolation, explicit opt-in, changed source rejection, catalog/split validation, authored/observed separation, and fresh-Tiger-only export.
