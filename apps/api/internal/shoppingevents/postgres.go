package shoppingevents

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/jackc/pgx/v5"
	"storefront/api/internal/experimentresults"
	"storefront/api/migrations"
	"time"
)

type Repository interface {
	Put(context.Context, Saved, *Replay) error
	Prune(context.Context) error
}
type Postgres struct{ database *experimentresults.Postgres }

func NewPostgres(dsn string) (*Postgres, error) {
	p, err := experimentresults.NewPostgres(dsn)
	if err != nil {
		return nil, err
	}
	return &Postgres{p}, nil
}
func closeDB(c *pgx.Conn) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_ = c.Close(ctx)
}
func (p *Postgres) Migrate(ctx context.Context) error {
	c, err := p.database.Open(ctx)
	if err != nil {
		return err
	}
	defer closeDB(c)
	var exists bool
	if err = c.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname=$1)`, "shopping_events").Scan(&exists); err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("shopping_events already exists; inspect manually before migration")
	}
	_, err = c.Exec(ctx, migrations.ShoppingEventsV1)
	return err
}
func (p *Postgres) Put(ctx context.Context, s Saved, replay *Replay) error {
	c, err := p.database.Open(ctx)
	if err != nil {
		return err
	}
	defer closeDB(c)
	tx, err := c.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(context.Background())
	var existing string
	err = tx.QueryRow(ctx, `SELECT sha256 FROM shopping_events.batches WHERE batch_id=$1`, s.Batch.BatchID).Scan(&existing)
	if err == nil {
		if existing != s.Digest {
			return ErrConflict
		}
		return tx.Commit(ctx)
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return err
	}
	b := s.Batch
	b.Replay = nil
	payload, _ := json.Marshal(b)
	_, err = tx.Exec(ctx, `INSERT INTO shopping_events.batches(batch_id,session_id,environment,traffic,sha256,received_at,payload) VALUES($1,$2,$3,$4,$5,$6,$7::jsonb)`, b.BatchID, b.SessionID, b.Classification.Environment, b.Classification.Traffic, s.Digest, s.ReceivedAt, string(payload))
	if err != nil {
		return err
	}
	for _, e := range b.Events {
		_, err = tx.Exec(ctx, `INSERT INTO shopping_events.event_receipts(event_id,session_id,sequence,batch_id) VALUES($1,$2,$3,$4)`, e.EventID, b.SessionID, *e.Sequence, b.BatchID)
		if err != nil {
			return err
		}
	}
	if replay != nil && s.ReplayExpiresAt.After(time.Now()) {
		raw, _ := json.Marshal(replay)
		_, err = tx.Exec(ctx, `INSERT INTO shopping_events.replay_chunks(batch_id,expires_at,payload) VALUES($1,$2,$3::jsonb)`, b.BatchID, s.ReplayExpiresAt, string(raw))
		if err != nil {
			return err
		}
	}
	return tx.Commit(ctx)
}
func (p *Postgres) Prune(ctx context.Context) error {
	c, err := p.database.Open(ctx)
	if err != nil {
		return err
	}
	defer closeDB(c)
	_, err = c.Exec(ctx, `DELETE FROM shopping_events.replay_chunks WHERE batch_id IN (SELECT batch_id FROM shopping_events.replay_chunks WHERE expires_at <= now() LIMIT 100)`)
	return err
}

type Selection struct {
	Since            time.Time
	Until            time.Time
	IncludeSynthetic bool
	Limit            int
}

func (p *Postgres) Walk(ctx context.Context, selection Selection, consume func(Saved) error) error {
	if selection.Limit < 1 || selection.Limit > 10000 || !selection.Until.After(selection.Since) {
		return fmt.Errorf("bounded export selection required")
	}
	c, err := p.database.Open(ctx)
	if err != nil {
		return err
	}
	defer closeDB(c)
	rows, err := c.Query(ctx, `SELECT payload,sha256,received_at FROM shopping_events.batches
WHERE received_at >= $1 AND received_at < $2 AND ($3 OR (environment='production' AND traffic='real'))
ORDER BY received_at,batch_id LIMIT $4`, selection.Since, selection.Until, selection.IncludeSynthetic, selection.Limit+1)
	if err != nil {
		return err
	}
	defer rows.Close()
	count := 0
	for rows.Next() {
		count++
		if count > selection.Limit {
			return fmt.Errorf("export batch limit exceeded; narrow date range")
		}
		var s Saved
		var payload []byte
		if err = rows.Scan(&payload, &s.Digest, &s.ReceivedAt); err != nil {
			return err
		}
		if json.Unmarshal(payload, &s.Batch) != nil {
			return fmt.Errorf("invalid stored shopping batch")
		}
		if err = consume(s); err != nil {
			return err
		}
	}
	return rows.Err()
}

// Replay is a server-side maintenance read, never part of the training export.
func (p *Postgres) Replay(ctx context.Context, id string) (*Replay, error) {
	c, err := p.database.Open(ctx)
	if err != nil {
		return nil, err
	}
	defer closeDB(c)
	var raw []byte
	err = c.QueryRow(ctx, `SELECT payload FROM shopping_events.replay_chunks WHERE batch_id=$1 AND expires_at>now()`, id).Scan(&raw)
	if err != nil {
		return nil, err
	}
	var replay Replay
	err = json.Unmarshal(raw, &replay)
	return &replay, err
}
