-- New namespace only; no startup DDL and no changes to experiment/store tables.
BEGIN;
CREATE SCHEMA shopping_events;
CREATE TABLE shopping_events.batches (
 batch_id uuid PRIMARY KEY,
 session_id uuid NOT NULL,
 environment text NOT NULL CHECK (environment IN ('production','development','test')),
 traffic text NOT NULL CHECK (traffic IN ('real','synthetic')),
 sha256 text NOT NULL CHECK (sha256 ~ '^[a-f0-9]{64}$'),
 received_at timestamptz NOT NULL,
 payload jsonb NOT NULL
);
CREATE INDEX batches_export_idx ON shopping_events.batches(environment,traffic,received_at,batch_id);
CREATE TABLE shopping_events.event_receipts (
 event_id uuid PRIMARY KEY,
 session_id uuid NOT NULL,
 sequence integer NOT NULL,
 batch_id uuid NOT NULL REFERENCES shopping_events.batches(batch_id),
 UNIQUE(session_id,sequence)
);
CREATE TABLE shopping_events.replay_chunks (
 batch_id uuid PRIMARY KEY REFERENCES shopping_events.batches(batch_id),
 expires_at timestamptz NOT NULL,
 payload jsonb NOT NULL
);
CREATE INDEX replay_expiry_idx ON shopping_events.replay_chunks(expires_at);
COMMIT;
