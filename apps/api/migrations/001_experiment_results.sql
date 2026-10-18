-- Apply manually only after verifying the hosted database and intended scope.
-- No extension, hypertable, existing store table, or automatic startup DDL.
BEGIN;
CREATE SCHEMA IF NOT EXISTS experiment_results;
CREATE TABLE IF NOT EXISTS experiment_results.reports (
    report_id text PRIMARY KEY CHECK (report_id ~ '^[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}$'),
    experiment_id text NOT NULL,
    sha256 text NOT NULL CHECK (sha256 ~ '^[a-f0-9]{64}$'),
    received_at timestamptz NOT NULL,
    payload jsonb NOT NULL CHECK (
        payload->>'version' = 'experiment-results.v1'
        AND payload->>'reportId' = report_id
        AND payload->>'experimentId' = experiment_id
        AND payload->>'synthetic' = 'true'
        AND payload->>'humanConversionClaim' = 'false'
    )
);
CREATE INDEX IF NOT EXISTS reports_experiment_received_idx
    ON experiment_results.reports(experiment_id, received_at DESC, report_id DESC);
COMMIT;
