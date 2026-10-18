// Explicit one-time schema installation; never runs during API startup.
package main

import (
	"context"
	"fmt"
	"os"
	"storefront/api/internal/experimentresults"
	"time"
)

func main() {
	if len(os.Args) != 2 || os.Args[1] != "--apply-v1" {
		fmt.Fprintln(os.Stderr, "usage: tiger-migrate --apply-v1 (creates only experiment_results in configured database)")
		os.Exit(1)
	}
	p, err := experimentresults.NewPostgres(os.Getenv("TIGER_DATABASE_URL"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "Tiger configuration invalid (connection details redacted)")
		os.Exit(1)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err = p.Inspect(ctx); err != nil {
		fmt.Fprintln(os.Stderr, "Read-only connection/scope verification failed; migration not attempted")
		os.Exit(1)
	}
	if err = p.ApplyV1(ctx); err != nil {
		fmt.Fprintln(os.Stderr, "Migration not confirmed; existing namespace or database error (details redacted). Run tiger-check before any retry.")
		os.Exit(1)
	}
	fmt.Println("Applied 001_experiment_results.sql to dedicated experiment_results schema")
}
