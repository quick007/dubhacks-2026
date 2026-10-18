package shoppingevents

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const maxArchiveBytes = 256 << 20
const maxArchiveBatches = 10000

type Service struct {
	dir       string
	repo      Repository
	config    Config
	mu        sync.Mutex
	wake      chan struct{}
	cancel    context.CancelFunc
	done      chan struct{}
	lastError bool
}

func New(dir string, c Config, repo Repository) (*Service, error) {
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	s := &Service{dir: dir, repo: repo, config: c, wake: make(chan struct{}, 1), cancel: cancel, done: make(chan struct{})}
	go s.work(ctx)
	return s, nil
}
func (s *Service) Close() { s.cancel(); <-s.done }
func atomic(dir, name string, data []byte) error {
	f, err := os.CreateTemp(dir, ".shopping-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if _, err = f.Write(data); err == nil {
		err = f.Sync()
	}
	closed := f.Close()
	if err != nil {
		return err
	}
	if closed != nil {
		return closed
	}
	if err = os.Rename(f.Name(), filepath.Join(dir, name)); err != nil {
		return err
	}
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
func (s *Service) load(id string) (Saved, error) {
	var saved Saved
	if !uuid.MatchString(id) {
		return saved, os.ErrNotExist
	}
	raw, err := os.ReadFile(filepath.Join(s.dir, id+".json"))
	if err != nil {
		return saved, err
	}
	err = json.Unmarshal(raw, &saved)
	return saved, err
}
func (s *Service) entries() ([]Saved, int64, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, 0, err
	}
	saved := []Saved{}
	var size int64
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			return nil, 0, err
		}
		size += info.Size()
		if strings.HasSuffix(e.Name(), ".json") {
			record, err := s.load(strings.TrimSuffix(e.Name(), ".json"))
			if err != nil {
				return nil, 0, err
			}
			saved = append(saved, record)
		}
	}
	return saved, size, nil
}
func (s *Service) Accept(b Batch, digest string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	existing, err := s.load(b.BatchID)
	if err == nil {
		if existing.Digest != digest {
			return false, ErrConflict
		}
		s.signal()
		return true, nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return false, err
	}
	entries, size, err := s.entries()
	if err != nil {
		return false, err
	}
	if len(entries) >= maxArchiveBatches || size+MaxBody > maxArchiveBytes {
		return false, ErrCapacity
	}
	received := time.Now().UTC()
	replay := b.Replay
	b.Replay = nil
	saved := Saved{Batch: b, Digest: digest, ReceivedAt: received, ReplayExpiresAt: received.Add(24 * time.Hour)}
	// Short-lived replay bytes never enter the durable observations JSON archive.
	if replay != nil {
		raw, _ := json.Marshal(replay)
		if err = atomic(s.dir, b.BatchID+".replay", raw); err != nil {
			return false, err
		}
	}
	raw, _ := json.Marshal(saved)
	if err = atomic(s.dir, b.BatchID+".json", raw); err != nil {
		return false, err
	}
	s.signal()
	return false, nil
}
func (s *Service) signal() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}
func (s *Service) sweep(ctx context.Context) error {
	// Remove orphan/expired replay files too, including a crash between replay
	// persistence and observation receipt persistence. Observation bytes remain.
	files, _ := os.ReadDir(s.dir)
	for _, file := range files {
		if strings.HasSuffix(file.Name(), ".replay") || strings.HasPrefix(file.Name(), ".shopping-") {
			if info, err := file.Info(); err == nil && time.Since(info.ModTime()) >= 24*time.Hour {
				_ = os.Remove(filepath.Join(s.dir, file.Name()))
			}
		}
	}
	entries, _, err := s.entries()
	if err != nil {
		return err
	}
	var first error
	for _, entry := range entries {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		replayPath := filepath.Join(s.dir, entry.Batch.BatchID+".replay")
		var replay *Replay
		if !entry.ReplayExpiresAt.After(time.Now()) {
			_ = os.Remove(replayPath)
		} else if raw, err := os.ReadFile(replayPath); err == nil {
			replay = &Replay{}
			if json.Unmarshal(raw, replay) != nil {
				replay = nil
			}
		}
		if s.repo == nil || !s.config.Enabled {
			continue
		}
		if _, err := os.Stat(filepath.Join(s.dir, entry.Batch.BatchID+".synced")); err == nil {
			continue
		}
		op, cancel := context.WithTimeout(ctx, 5*time.Second)
		err = s.repo.Put(op, entry, replay)
		cancel()
		if err == nil {
			err = atomic(s.dir, entry.Batch.BatchID+".synced", []byte(entry.Digest))
		}
		if err != nil && first == nil {
			first = err
		}
	}
	if s.repo != nil && s.config.Enabled {
		op, cancel := context.WithTimeout(ctx, 5*time.Second)
		err = s.repo.Prune(op)
		cancel()
		if first == nil {
			first = err
		}
	}
	return first
}
func (s *Service) work(ctx context.Context) {
	defer close(s.done)
	timer := time.NewTimer(0)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-s.wake:
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
		case <-timer.C:
		}
		err := s.sweep(ctx)
		s.mu.Lock()
		s.lastError = err != nil
		s.mu.Unlock()
		timer.Reset(time.Minute)
	}
}
func (s *Service) Status() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	entries, _, err := s.entries()
	pending := 0
	for _, entry := range entries {
		if _, err := os.Stat(filepath.Join(s.dir, entry.Batch.BatchID+".synced")); err != nil {
			pending++
		}
	}
	return map[string]any{"enabled": s.config.Enabled, "environment": s.config.Environment, "cloudConfigured": s.repo != nil, "batches": len(entries), "pending": pending, "syncUnavailable": s.lastError, "localReadable": err == nil}
}
