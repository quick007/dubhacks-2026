package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	_ "github.com/mattn/go-sqlite3"
	"os"
	"path/filepath"
	"storefront/api/internal/contracts"
)

type Store struct {
	db      *sql.DB
	dataDir string
}

func openStore(dataDir string) (*Store, error) {
	if err := os.MkdirAll(filepath.Join(dataDir, "artifacts"), 0700); err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite3", filepath.Join(dataDir, "metadata.sqlite")+"?_busy_timeout=5000&_journal_mode=WAL")
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	store := &Store{db, dataDir}
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS catalog (id INTEGER PRIMARY KEY CHECK(id = 1), body TEXT NOT NULL); CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, body TEXT NOT NULL);`)
	if err != nil {
		db.Close()
		return nil, err
	}
	// Runs snapshot their persona and config, so refreshing the fixture leaves history intact.
	if _, err = db.Exec("INSERT INTO catalog (id, body) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET body = excluded.body", string(catalogJSON)); err != nil {
		db.Close()
		return nil, err
	}
	runs, err := store.listRuns()
	if err != nil {
		db.Close()
		return nil, err
	}
	for _, run := range runs {
		if run.Status == "queued" || run.Status == "running" {
			run.Status = "failed"
			run.Error = "Local API restarted before this job completed"
			run.CompletedAt = now()
			if err = store.saveRun(run); err != nil {
				db.Close()
				return nil, err
			}
		}
	}
	return store, nil
}
func (s *Store) catalog() (contracts.Catalog, error) {
	var body []byte
	var catalog contracts.Catalog
	if err := s.db.QueryRow("SELECT body FROM catalog WHERE id = 1").Scan(&body); err != nil {
		return catalog, err
	}
	err := json.Unmarshal(body, &catalog)
	return catalog, err
}
func (s *Store) saveRun(run contracts.Run) error {
	body, err := json.Marshal(run)
	if err != nil {
		return err
	}
	_, err = s.db.Exec("INSERT INTO runs (id, created_at, body) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET body = excluded.body", run.ID, run.CreatedAt, body)
	return err
}
func (s *Store) run(id string) (contracts.Run, error) {
	var body []byte
	var run contracts.Run
	if err := s.db.QueryRow("SELECT body FROM runs WHERE id = ?", id).Scan(&body); err != nil {
		return run, err
	}
	err := json.Unmarshal(body, &run)
	return run, err
}
func (s *Store) listRuns() ([]contracts.Run, error) {
	rows, err := s.db.Query("SELECT body FROM runs ORDER BY created_at DESC, rowid DESC")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	runs := []contracts.Run{}
	for rows.Next() {
		var body []byte
		var run contracts.Run
		if err = rows.Scan(&body); err != nil {
			return nil, err
		}
		if err = json.Unmarshal(body, &run); err != nil {
			return nil, fmt.Errorf("corrupt local run: %w", err)
		}
		runs = append(runs, run)
	}
	return runs, rows.Err()
}
