package experimentresults

import (
	"context"
	"fmt"
	"storefront/api/migrations"
	"strings"
)

// ApplyV1 is an explicit maintenance operation; the HTTP service never calls it.
// Refuse an existing namespace so teammate tables cannot be silently adopted.
func (p *Postgres) ApplyV1(ctx context.Context) error {
	c, err := p.connection(ctx)
	if err != nil {
		return err
	}
	defer closeConnection(c)
	var exists bool
	if err = c.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname=$1)`, "experiment_results").Scan(&exists); err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("experiment_results already exists; inspect and review manually before any migration")
	}
	// Guard the intended scope once more immediately before executing static DDL.
	if !strings.Contains(migrations.ExperimentResultsV1, "CREATE SCHEMA IF NOT EXISTS experiment_results") {
		return fmt.Errorf("migration scope mismatch")
	}
	_, err = c.Exec(ctx, migrations.ExperimentResultsV1)
	return err
}

// Inspect reads metadata only. It never selects evidence, credentials, or store
// rows, and never applies migrations. It is separate from application startup.
func (p *Postgres) Inspect(ctx context.Context) ([]string, error) {
	c, err := p.connection(ctx)
	if err != nil {
		return nil, err
	}
	defer closeConnection(c)
	rows, err := c.Query(ctx, `SELECT table_schema||'.'||table_name||'.'||column_name||':'||data_type
FROM information_schema.columns WHERE table_schema IN ('public','experiment_results')
AND EXISTS (SELECT 1 FROM information_schema.tables t
 WHERE t.table_schema=columns.table_schema AND t.table_name=columns.table_name AND t.table_type='BASE TABLE')
ORDER BY table_schema,table_name,ordinal_position LIMIT 500`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	columns := []string{}
	for rows.Next() {
		var column string
		if err = rows.Scan(&column); err != nil {
			return nil, err
		}
		columns = append(columns, column)
	}
	return columns, rows.Err()
}
