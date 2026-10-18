package experimentresults

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Service keeps immutable receipts even after upload. A durable sync marker is
// written only after PostgreSQL acknowledges the idempotent insert.
type Service struct {
	dir       string
	repo      Repository
	mu        sync.Mutex
	lastError string
	wake      chan struct{}
	cancel    context.CancelFunc
	done      chan struct{}
}

func New(dir string, repo Repository) (*Service, error) {
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	s := &Service{dir: dir, repo: repo, wake: make(chan struct{}, 1), cancel: cancel, done: make(chan struct{})}
	go s.work(ctx)
	return s, nil
}
func (s *Service) Close() { s.cancel(); <-s.done }
func (s *Service) signal() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

// atomicFile fsyncs data and directory before returning success. Temp files are
// ignored by the worker and queries, including after an interrupted process.
func atomicFile(dir, name string, data []byte) error {
	f, err := os.CreateTemp(dir, ".receipt-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err = f.Write(data); err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	if err = os.Rename(tmp, filepath.Join(dir, name)); err != nil {
		return err
	}
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
func (s *Service) Accept(e Envelope, digest string) (Saved, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	old, err := s.local(e.ReportID)
	if err == nil {
		if old.Digest != digest {
			return Saved{}, false, ErrConflict
		}
		s.signal()
		return old, true, nil
	}
	if !errors.Is(err, ErrNotFound) {
		return Saved{}, false, err
	}
	saved := Saved{Envelope: e, Digest: digest, ReceivedAt: time.Now().UTC().Format(time.RFC3339Nano)}
	data, err := json.Marshal(saved)
	if err != nil {
		return Saved{}, false, err
	}
	if err = atomicFile(s.dir, e.ReportID+".json", data); err != nil {
		return Saved{}, false, err
	}
	s.signal()
	return saved, false, nil
}
func (s *Service) local(id string) (Saved, error) {
	var saved Saved
	if !safeID.MatchString(id) {
		return saved, ErrNotFound
	}
	data, err := os.ReadFile(filepath.Join(s.dir, id+".json"))
	if errors.Is(err, os.ErrNotExist) {
		return saved, ErrNotFound
	}
	if err != nil {
		return saved, err
	}
	err = json.Unmarshal(data, &saved)
	return saved, err
}
func (s *Service) localHeaders(experiment string) ([]Header, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, err
	}
	headers := []Header{}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		saved, err := s.local(strings.TrimSuffix(entry.Name(), ".json"))
		if err != nil {
			return nil, err
		}
		if experiment == "" || saved.ExperimentID == experiment {
			headers = append(headers, saved.Header())
		}
	}
	return headers, nil
}
func (s *Service) Get(ctx context.Context, id string) (Saved, string, error) {
	saved, err := s.local(id)
	if !errors.Is(err, ErrNotFound) {
		return saved, "local", err
	}
	if s.repo == nil {
		return saved, "local", ErrNotFound
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	saved, err = s.repo.Get(ctx, id)
	return saved, "tiger", err
}
func (s *Service) List(ctx context.Context, experiment string, limit int) ([]Header, string, error) {
	local, err := s.localHeaders(experiment)
	if err != nil {
		return nil, "local", err
	}
	source := "local"
	merged := map[string]Header{}
	if s.repo != nil {
		ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		remote, err := s.repo.List(ctx, experiment, limit)
		if err != nil {
			source = "local-fallback"
		} else {
			source = "tiger+local"
			for _, h := range remote {
				merged[h.ReportID] = h
			}
		}
	}
	for _, h := range local {
		merged[h.ReportID] = h
	}
	result := []Header{}
	for _, h := range merged {
		result = append(result, h)
	}
	sort.Slice(result, func(i, j int) bool {
		iTime, _ := time.Parse(time.RFC3339Nano, result[i].ReceivedAt)
		jTime, _ := time.Parse(time.RFC3339Nano, result[j].ReceivedAt)
		if iTime.Equal(jTime) {
			return result[i].ReportID > result[j].ReportID
		}
		return iTime.After(jTime)
	})
	if len(result) > limit {
		result = result[:limit]
	}
	return result, source, nil
}
func (s *Service) syncPending(ctx context.Context) error {
	if s.repo == nil {
		return nil
	}
	headers, err := s.localHeaders("")
	if err != nil {
		return err
	}
	var firstError error
	for _, h := range headers {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		marker := filepath.Join(s.dir, h.ReportID+".synced")
		if _, err := os.Stat(marker); err == nil {
			continue
		}
		saved, err := s.local(h.ReportID)
		if err != nil {
			return err
		}
		operation, cancel := context.WithTimeout(ctx, 5*time.Second)
		err = s.repo.Put(operation, saved)
		cancel()
		if err != nil {
			if firstError == nil {
				firstError = err
			}
			continue
		}
		if err = atomicFile(s.dir, h.ReportID+".synced", []byte(h.Digest)); err != nil {
			if firstError == nil {
				firstError = err
			}
		}
	}
	return firstError
}
func (s *Service) work(ctx context.Context) {
	defer close(s.done)
	delay := time.Duration(0)
	for {
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return
		case <-timer.C:
		case <-s.wake:
			timer.Stop()
		}
		err := s.syncPending(ctx)
		s.mu.Lock()
		if err != nil {
			s.lastError = "Tiger sync unavailable; local receipts retained (check connection, migration, or report ID conflict)"
		} else {
			s.lastError = ""
		}
		s.mu.Unlock()
		if err == nil {
			delay = 5 * time.Second
		} else {
			delay = 60 * time.Second
		}
	}
}
func (s *Service) Status() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	pending := 0
	headers, err := s.localHeaders("")
	for _, h := range headers {
		if _, e := os.Stat(filepath.Join(s.dir, h.ReportID+".synced")); e != nil {
			pending++
		}
	}
	state := "disabled"
	if s.repo != nil {
		state = "configured"
	}
	return map[string]any{"cloud": state, "pendingReports": pending, "localReports": len(headers), "lastSyncError": s.lastError, "localReadable": err == nil, "synthetic": true, "humanConversionClaim": false}
}
func publicError(err error) error {
	if errors.Is(err, ErrNotFound) || errors.Is(err, ErrConflict) {
		return err
	}
	return fmt.Errorf("experiment results unavailable; local receipts retained")
}
