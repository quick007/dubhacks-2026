// Read-only connection/scope check; intentionally no migrate command.
package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/jackc/pgx/v5/pgconn"
	"net"
	"os"
	"storefront/api/internal/experimentresults"
	"time"
)

func main() {
	p, err := experimentresults.NewPostgres(os.Getenv("TIGER_DATABASE_URL"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "Tiger configuration invalid (connection details redacted)")
		os.Exit(1)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	columns, err := p.Inspect(ctx)
	if err != nil {
		reason := "connection or TLS failure"
		var dbError *pgconn.PgError
		var dnsError *net.DNSError
		var certError *tls.CertificateVerificationError
		var authorityError x509.UnknownAuthorityError
		if errors.Is(err, context.DeadlineExceeded) {
			reason = "connection timed out"
		}
		if errors.As(err, &dnsError) {
			reason = "DNS lookup failed"
		}
		if errors.As(err, &certError) || errors.As(err, &authorityError) {
			reason = "TLS certificate validation failed"
		}
		if pgconn.Timeout(err) {
			reason = "connection timed out"
		}
		if errors.As(err, &dbError) {
			reason = "PostgreSQL SQLSTATE " + dbError.Code
		}
		fmt.Fprintln(os.Stderr, "Tiger read-only check failed: "+reason+" (connection details redacted); no schema changes attempted")
		os.Exit(1)
	}
	_ = json.NewEncoder(os.Stdout).Encode(map[string]any{"connected": true, "tlsRequired": true, "tlsCertificateVerified": p.CertificateVerified(), "schemaChanged": false, "columns": columns})
}
