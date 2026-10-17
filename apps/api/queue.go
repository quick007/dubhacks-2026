package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"storefront/api/internal/contracts"
	"storefront/api/internal/evaluator"
	"storefront/api/internal/optimizer"
	"storefront/api/internal/runner"
	"sync"
	"time"
)

type JobQueue struct {
	store         *Store
	capacity      int
	delay         time.Duration
	workspaceRoot string
	mu            sync.Mutex
	count         int
	stopped       bool
	jobs          chan string
	done          chan struct{}
}

func newQueue(store *Store, capacity int, delay time.Duration, root string) *JobQueue {
	q := &JobQueue{store: store, capacity: capacity, delay: delay, workspaceRoot: root, jobs: make(chan string, capacity), done: make(chan struct{})}
	go q.loop()
	return q
}

// submit serializes capacity reservation and persistence; a rejected run is never saved.
func (q *JobQueue) submit(run contracts.Run) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.stopped {
		return fmt.Errorf("local queue is stopped")
	}
	if q.count >= q.capacity {
		return errQueueFull
	}
	if err := q.store.saveRun(run); err != nil {
		return err
	}
	q.count++
	q.jobs <- run.ID
	return nil
}
func (q *JobQueue) loop() {
	defer close(q.done)
	for id := range q.jobs {
		// Make the queued state observable to polling clients, including very fast mock jobs.
		time.Sleep(15 * time.Millisecond)
		q.execute(id)
		q.mu.Lock()
		q.count--
		q.mu.Unlock()
	}
}
func (q *JobQueue) execute(id string) {
	run, err := q.store.run(id)
	if err != nil {
		return
	}
	emit := func(event runner.Event) error {
		run.Events = append(run.Events, contracts.StepEvent{ID: newID(), RunID: run.ID, Sequence: len(run.Events), At: now(), Kind: event.Kind, Message: event.Message})
		return q.store.saveRun(run)
	}
	fail := func(err error) {
		run.Status = "failed"
		run.Error = err.Error()
		run.CompletedAt = now()
		_ = emit(runner.Event{Kind: "failure", Message: run.Error})
	}
	run.Status = "running"
	run.StartedAt = now()
	if err = emit(runner.Event{Kind: "status", Message: "running"}); err != nil {
		fail(err)
		return
	}
	directory := filepath.Join(q.store.dataDir, "artifacts", run.ID)
	if err = os.MkdirAll(directory, 0700); err != nil {
		fail(err)
		return
	}
	saveArtifact := func(item runner.Artifact) error {
		if filepath.Base(item.FileName) != item.FileName {
			return fmt.Errorf("invalid artifact filename")
		}
		file := filepath.Join(directory, item.FileName)
		info, err := os.Lstat(file)
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("artifact is not a regular local file")
		}
		run.Artifacts = append(run.Artifacts, contracts.Artifact{ID: newID(), RunID: run.ID, Kind: item.Kind, Path: filepath.ToSlash(filepath.Join("artifacts", run.ID, item.FileName)), MimeType: item.MimeType, CreatedAt: now()})
		return q.store.saveRun(run)
	}
	var adapter runner.Adapter = runner.Mock{StepDelay: q.delay}
	if run.Adapter == "playwright" {
		adapter = runner.Browser{WorkspaceRoot: q.workspaceRoot, NodeBinary: os.Getenv("NODE_BINARY")}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	result, err := adapter.Execute(ctx, runner.Context{Run: run, ArtifactDirectory: directory, Emit: emit, SaveArtifact: saveArtifact})
	if err != nil {
		fail(err)
		return
	}
	evaluation := (evaluator.Deterministic{}).Evaluate(run, result)
	run.Evaluation = &evaluation
	run.Status = "completed"
	run.CompletedAt = now()
	run.ProposedChanges = (optimizer.Factual{}).Propose(run)
	if err = emit(runner.Event{Kind: "status", Message: "completed"}); err != nil {
		fail(err)
	}
}
func (q *JobQueue) close() {
	q.mu.Lock()
	if !q.stopped {
		q.stopped = true
		close(q.jobs)
	}
	q.mu.Unlock()
	<-q.done
}
