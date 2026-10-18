package experimentresults

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"time"

	"github.com/jackc/pgx/v5"
)

var ErrConflict = errors.New("report ID already contains different evidence; use a new report ID")
var ErrNotFound = errors.New("report not found")

type Repository interface {
	Put(context.Context, Saved) error
	Get(context.Context, string) (Saved, error)
	List(context.Context, string, int) ([]Header, error)
}
type Postgres struct{ config *pgx.ConnConfig }

// No connection or migration occurs during construction. Do not log parse errors:
// driver errors may contain the original credential-bearing connection string.
func NewPostgres(dsn string) (*Postgres, error) {
	u, err := url.Parse(dsn)
	if err != nil || (u.Scheme != "postgres" && u.Scheme != "postgresql") || u.Hostname() == "" || u.User == nil {
		return nil, fmt.Errorf("TIGER_DATABASE_URL must be a PostgreSQL URL")
	}
	params := u.Query()
	for key := range params {
		if key != "sslmode" && key != "sslrootcert" && key != "connect_timeout" {
			return nil, fmt.Errorf("unsupported Tiger Data connection option")
		}
	}
	if mode := params.Get("sslmode"); mode != "require" && mode != "verify-full" {
		return nil, fmt.Errorf("TIGER_DATABASE_URL must require TLS")
	}
	// Honor the supplied TLS mode. require encrypts; verify-full additionally
	// validates the service certificate using sslrootcert/system roots.
	// Neither mode may fall back to a plaintext connection.
	params.Set("connect_timeout", "3")
	u.RawQuery = params.Encode()
	c, err := pgx.ParseConfig(u.String())
	if err != nil || c.TLSConfig == nil {
		return nil, fmt.Errorf("invalid Tiger Data TLS configuration")
	}
	c.TLSConfig.MinVersion = tls.VersionTLS12
	c.Fallbacks = nil
	c.ConnectTimeout = 3 * time.Second
	c.RuntimeParams["application_name"] = "dubhacks-experiment-results"
	c.RuntimeParams["statement_timeout"] = "4000"
	return &Postgres{c}, nil
}
func (p *Postgres) CertificateVerified() bool { return !p.config.TLSConfig.InsecureSkipVerify }
func (p *Postgres) connection(ctx context.Context) (*pgx.Conn, error) {
	return pgx.ConnectConfig(ctx, p.config.Copy())
}
func closeConnection(c *pgx.Conn) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_ = c.Close(ctx)
}
func (p *Postgres) Put(ctx context.Context, s Saved) error {
	c, err := p.connection(ctx)
	if err != nil {
		return err
	}
	defer closeConnection(c)
	payload, err := json.Marshal(s.Envelope)
	if err != nil {
		return err
	}
	receivedAt, err := time.Parse(time.RFC3339Nano, s.ReceivedAt)
	if err != nil {
		return fmt.Errorf("invalid receipt timestamp")
	}
	var id string
	err = c.QueryRow(ctx, `INSERT INTO experiment_results.reports(report_id,experiment_id,sha256,received_at,payload)
VALUES($1,$2,$3,$4,$5::jsonb)
ON CONFLICT(report_id) DO UPDATE SET report_id=EXCLUDED.report_id
WHERE experiment_results.reports.sha256=EXCLUDED.sha256 RETURNING report_id`, s.ReportID, s.ExperimentID, s.Digest, receivedAt, string(payload)).Scan(&id)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrConflict
	}
	return err
}
func (p *Postgres) Get(ctx context.Context, id string) (Saved, error) {
	var s Saved
	var raw []byte
	var at time.Time
	c, err := p.connection(ctx)
	if err != nil {
		return s, err
	}
	defer closeConnection(c)
	err = c.QueryRow(ctx, `SELECT sha256,received_at,payload FROM experiment_results.reports WHERE report_id=$1`, id).Scan(&s.Digest, &at, &raw)
	if errors.Is(err, pgx.ErrNoRows) {
		return s, ErrNotFound
	}
	if err != nil {
		return s, err
	}
	err = json.Unmarshal(raw, &s.Envelope)
	s.ReceivedAt = at.UTC().Format(time.RFC3339Nano)
	return s, err
}
func (p *Postgres) List(ctx context.Context, experiment string, limit int) ([]Header, error) {
	c, err := p.connection(ctx)
	if err != nil {
		return nil, err
	}
	defer closeConnection(c)
	rows, err := c.Query(ctx, `SELECT report_id,experiment_id,sha256,received_at FROM experiment_results.reports
WHERE ($1='' OR experiment_id=$1) ORDER BY received_at DESC,report_id DESC LIMIT $2`, experiment, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := []Header{}
	for rows.Next() {
		var h Header
		var at time.Time
		if err = rows.Scan(&h.ReportID, &h.ExperimentID, &h.Digest, &at); err != nil {
			return nil, err
		}
		h.ReceivedAt = at.UTC().Format(time.RFC3339Nano)
		result = append(result, h)
	}
	return result, rows.Err()
}
