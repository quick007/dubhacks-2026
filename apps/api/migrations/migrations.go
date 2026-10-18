// Package migrations contains reviewed, opt-in experiment evidence DDL.
package migrations

import _ "embed"

//go:embed 001_experiment_results.sql
var ExperimentResultsV1 string

//go:embed 002_shopping_events.sql
var ShoppingEventsV1 string
