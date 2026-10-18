// Local maintenance CLI: no browser-accessible training export endpoint.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"storefront/api/internal/shoppingevents"
	"time"
)

func main() {
	output := flag.String("out", "", "new JSONL output path")
	since := flag.String("since", "2020-01-01T00:00:00Z", "inclusive received-at RFC3339 cutoff")
	until := flag.String("until", time.Now().UTC().Format(time.RFC3339Nano), "exclusive received-at cutoff")
	include := flag.Bool("include-synthetic", false, "explicitly include development/test/synthetic for isolated inspection")
	limit := flag.Int("max-batches", 1000, "maximum 1..10000 batches; fails instead of silently truncating")
	migrate := flag.Bool("apply-v1", false, "install shopping_events namespace after connection/scope check; no export")
	flag.Parse()
	repo, err := shoppingevents.NewPostgres(os.Getenv("TIGER_DATABASE_URL"))
	if err != nil {
		fail("invalid Tiger configuration (details redacted)")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if *migrate {
		if err = repo.Migrate(ctx); err != nil {
			fail("shopping schema migration not confirmed; check existing namespace and connection (details redacted)")
		}
		fmt.Println("Applied 002_shopping_events.sql to new shopping_events namespace")
		return
	}
	start, err := time.Parse(time.RFC3339Nano, *since)
	if err != nil {
		fail("invalid --since")
	}
	end, err := time.Parse(time.RFC3339Nano, *until)
	if err != nil {
		fail("invalid --until")
	}
	if *output == "" || *limit < 1 || *limit > 10000 || !end.After(start) {
		fail("--out NEW_PATH and bounded selection required")
	}
	manifestPath := *output + ".manifest.json"
	// Reserve both names exclusively; failures never leave a valid-looking dataset.
	file, err := os.OpenFile(*output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		fail("output exists or cannot be created")
	}
	manifestFile, err := os.OpenFile(manifestPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		file.Close()
		os.Remove(*output)
		fail("manifest exists or cannot be created")
	}
	selection := shoppingevents.Selection{Since: start, Until: end, IncludeSynthetic: *include, Limit: *limit}
	manifest, err := shoppingevents.Export(func(consume func(shoppingevents.Saved) error) error { return repo.Walk(ctx, selection, consume) }, file, selection)
	if err == nil {
		err = file.Sync()
	}
	file.Close()
	if err == nil {
		err = json.NewEncoder(manifestFile).Encode(manifest)
	}
	if err == nil {
		err = manifestFile.Sync()
	}
	manifestFile.Close()
	if err != nil {
		os.Remove(*output)
		os.Remove(manifestPath)
		fail("training export failed; outputs removed (database details redacted)")
	}
	_ = json.NewEncoder(os.Stdout).Encode(manifest)
}
func fail(message string) { fmt.Fprintln(os.Stderr, message); os.Exit(1) }
