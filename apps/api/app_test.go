package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"storefront/api/internal/contracts"
	"storefront/api/internal/optimizer"
	"strings"
	"testing"
	"time"
)

func request(t *testing.T, app *App, method, url string, body any) *httptest.ResponseRecorder {
	t.Helper()
	var input []byte
	if body != nil {
		var err error
		input, err = json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
	}
	req := httptest.NewRequest(method, url, bytes.NewReader(input))
	req.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	app.ServeHTTP(response, req)
	return response
}
func decode[T any](t *testing.T, response *httptest.ResponseRecorder) T {
	t.Helper()
	var value T
	if err := json.Unmarshal(response.Body.Bytes(), &value); err != nil {
		t.Fatalf("invalid response %s: %v", response.Body.String(), err)
	}
	return value
}
func testApp(t *testing.T, dir string, capacity int) *App {
	t.Helper()
	app, err := NewApp(AppOptions{DataDir: dir, MockStepDelay: time.Millisecond, QueueCapacity: capacity})
	if err != nil {
		t.Fatal(err)
	}
	return app
}
func waitCompleted(t *testing.T, app *App, count int) []contracts.Run {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		runs := decode[[]contracts.Run](t, request(t, app, "GET", "/api/runs", nil))
		if len(runs) == count {
			done := true
			for _, run := range runs {
				if run.Status == "failed" {
					t.Fatalf("run failed: %s", run.Error)
				}
				if run.Status != "completed" {
					done = false
				}
			}
			if done {
				return runs
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("mock jobs did not complete")
	return nil
}
func TestMockEndToEnd(t *testing.T) {
	directory := t.TempDir()
	app := testApp(t, directory, 20)
	defer func() { app.Close() }()
	catalog := decode[contracts.Catalog](t, request(t, app, "GET", "/api/catalog", nil))
	for _, id := range catalog.Experiment.DevelopmentPersonaIDs {
		if slices.Contains(catalog.Experiment.HeldOutPersonaIDs, id) {
			t.Fatal("overlapping persona splits")
		}
	}
	snapshots := map[string]contracts.Run{}
	for _, variant := range catalog.Variants {
		for _, persona := range []contracts.Persona{catalog.Personas[0], catalog.Personas[5]} {
			response := request(t, app, "POST", "/api/runs", contracts.RunRequest{ExperimentID: catalog.Experiment.ID, VariantID: variant.ID, PersonaID: persona.ID, Adapter: "mock"})
			if response.Code != 202 {
				t.Fatal(response.Body.String())
			}
			run := decode[contracts.Run](t, response)
			if run.Status != "queued" {
				t.Fatal("initial run is not queued")
			}
			snapshots[run.ID] = run
		}
	}
	runs := waitCompleted(t, app, 4)
	for _, run := range runs {
		snapshot := snapshots[run.ID]
		if !reflect.DeepEqual(run.Config, snapshot.Config) || !reflect.DeepEqual(run.Persona, snapshot.Persona) || run.Seed != snapshot.Seed || run.BaselineVariantID != catalog.Experiment.BaselineVariantID {
			t.Fatal("immutable snapshot changed")
		}
		resultIndex := slices.IndexFunc(run.Artifacts, func(item contracts.Artifact) bool { return item.Kind == "result" })
		if run.Evaluation == nil || !run.Evaluation.Synthetic || run.Evaluation.Findability == nil || resultIndex < 0 {
			t.Fatal("missing synthetic evaluation, findability or result artifact")
		}
		for _, event := range run.Events {
			if event.Page != nil && !slices.ContainsFunc(run.Artifacts, func(item contracts.Artifact) bool { return item.ID == event.ScreenshotArtifactID }) {
				t.Fatal("step screenshot does not resolve to an artifact")
			}
		}
		frame := request(t, app, "GET", "/api/artifacts/"+run.Events[1].ScreenshotArtifactID, nil)
		if frame.Code != 200 || frame.Header().Get("Content-Type") != "image/jpeg" || frame.Header().Get("X-Content-Type-Options") != "nosniff" || !strings.Contains(frame.Header().Get("Content-Security-Policy"), "sandbox") {
			t.Fatalf("frame served unsafely: %d %v", frame.Code, frame.Header())
		}
		statuses := []string{}
		for i, event := range run.Events {
			if event.Sequence != i || event.RunID != run.ID {
				t.Fatal("invalid event identity")
			}
			if event.Kind == "status" {
				statuses = append(statuses, event.Message)
			}
		}
		if !reflect.DeepEqual(statuses, []string{"running", "completed"}) {
			t.Fatalf("wrong lifecycle: %v", statuses)
		}
		artifact := request(t, app, "GET", "/api/artifacts/"+run.Artifacts[resultIndex].ID, nil)
		if artifact.Code != 200 {
			t.Fatal(artifact.Body.String())
		}
		result := decode[map[string]any](t, artifact)
		if result["runId"] != run.ID || result["synthetic"] != true {
			t.Fatal("invalid result download")
		}
		if run.PersonaSplit == "held-out" && len(run.ProposedChanges) != 0 {
			t.Fatal("held-out run entered optimizer")
		}
		if run.PersonaSplit == "development" {
			run.PersonaSplit = "held-out"
			if len((optimizer.Factual{}).Propose(run)) != 0 {
				t.Fatal("optimizer accepted held-out persona")
			}
		}
	}
	comparison := decode[[]contracts.ComparisonRow](t, request(t, app, "GET", "/api/comparison", nil))
	if len(comparison) != 4 {
		t.Fatalf("wrong comparison: %+v", comparison)
	}
	for _, row := range comparison {
		if row.Runs != 1 || !row.Synthetic {
			t.Fatal("invalid comparison counts")
		}
	}
	app.Close()
	app = testApp(t, directory, 20)
	persisted := decode[[]contracts.Run](t, request(t, app, "GET", "/api/runs", nil))
	if !reflect.DeepEqual(persisted, runs) {
		t.Fatal("completed run persistence changed")
	}
	if request(t, app, "GET", "/api/artifacts/unknown", nil).Code != 404 {
		t.Fatal("unknown artifact accepted")
	}
}
func TestQueueAndStrictInput(t *testing.T) {
	app := testApp(t, t.TempDir(), 1)
	defer app.Close()
	body := map[string]any{"experimentId": "fieldwork-study", "variantId": "default", "personaId": "dev-cup"}
	if response := request(t, app, "POST", "/api/runs", body); response.Code != 202 {
		t.Fatal(response.Body.String())
	}
	if response := request(t, app, "POST", "/api/runs", body); response.Code != 429 {
		t.Fatal("queue did not reject excess work")
	}
	body["url"] = "https://example.com"
	if request(t, app, "POST", "/api/runs", body).Code != 400 {
		t.Fatal("unknown request field accepted")
	}
	delete(body, "url")
	body["adapter"] = "remote"
	if request(t, app, "POST", "/api/runs", body).Code != 400 {
		t.Fatal("unknown adapter accepted")
	}
	for _, adapter := range []any{nil, "", 42} {
		body["adapter"] = adapter
		if request(t, app, "POST", "/api/runs", body).Code != 400 {
			t.Fatal("invalid explicit adapter accepted")
		}
	}
	if request(t, app, "GET", "/api/runs/unknown", nil).Code != 404 {
		t.Fatal("unknown run accepted")
	}
}
func TestRecoveryLockAndArtifactBoundary(t *testing.T) {
	directory := t.TempDir()
	app := testApp(t, directory, 20)
	if second, err := NewApp(AppOptions{DataDir: directory}); err == nil {
		second.Close()
		t.Fatal("concurrent data writer accepted")
	}
	app.Close()
	store, err := openStore(directory)
	if err != nil {
		t.Fatal(err)
	}
	run := contracts.Run{ID: "interrupted", Status: "running", CreatedAt: now(), Events: []contracts.StepEvent{}, Artifacts: []contracts.Artifact{}, ProposedChanges: []contracts.ProposedChange{}}
	if err = store.saveRun(run); err != nil {
		t.Fatal(err)
	}
	store.db.Close()
	app = testApp(t, directory, 20)
	defer app.Close()
	recovered := decode[contracts.Run](t, request(t, app, "GET", "/api/runs/interrupted", nil))
	if recovered.Status != "failed" || recovered.Error == "" {
		t.Fatal("interrupted run not recovered")
	}
	outside := filepath.Join(t.TempDir(), "private.txt")
	if err = os.WriteFile(outside, []byte("private"), 0600); err != nil {
		t.Fatal(err)
	}
	run.Artifacts = []contracts.Artifact{{ID: "escape", Path: fmt.Sprintf("../%s", filepath.Base(outside))}}
	if err = app.store.saveRun(run); err != nil {
		t.Fatal(err)
	}
	response := request(t, app, http.MethodGet, "/api/artifacts/escape", nil)
	if response.Code == 200 {
		t.Fatal("artifact escaped allowlist")
	}
}

func TestListenAddress(t *testing.T) {
	for _, value := range []string{"", "4567", "0", "65536", "abc"} {
		t.Setenv("PORT", value)
		t.Setenv("HOST", "0.0.0.0")
		address, err := listenAddress()
		switch value {
		case "":
			if err != nil || address != "127.0.0.1:4311" {
				t.Fatal(address, err)
			}
		case "4567":
			if err != nil || address != "127.0.0.1:4567" {
				t.Fatal(address, err)
			}
		default:
			if err == nil {
				t.Fatalf("accepted invalid PORT %q", value)
			}
		}
	}
}

func TestComparisonIgnoresOtherExperiments(t *testing.T) {
	app := testApp(t, t.TempDir(), 20)
	defer app.Close()
	old := contracts.Run{ID: "old", ExperimentID: "retired-study", VariantID: "baseline", PersonaSplit: "development", Adapter: "mock", Synthetic: true, ContractVersion: 1, Status: "completed", CreatedAt: now(),
		Events: []contracts.StepEvent{}, Artifacts: []contracts.Artifact{}, ProposedChanges: []contracts.ProposedChange{},
		Evaluation: &contracts.EvaluationResult{RunID: "old", Synthetic: true, TaskCompleted: true, Score: 1, FailureReasons: []string{}, Notes: []string{}}}
	if err := app.store.saveRun(old); err != nil {
		t.Fatal(err)
	}
	if rows := decode[[]contracts.ComparisonRow](t, request(t, app, "GET", "/api/comparison", nil)); len(rows) != 0 {
		t.Fatalf("comparison mixed in another experiment: %+v", rows)
	}
}

func TestMockSessionsRunInParallel(t *testing.T) {
	app, err := NewApp(AppOptions{DataDir: t.TempDir(), MockStepDelay: 40 * time.Millisecond, QueueCapacity: 20})
	if err != nil {
		t.Fatal(err)
	}
	defer app.Close()
	catalog := decode[contracts.Catalog](t, request(t, app, "GET", "/api/catalog", nil))
	for _, persona := range catalog.Personas[:4] {
		response := request(t, app, "POST", "/api/runs", contracts.RunRequest{ExperimentID: catalog.Experiment.ID, VariantID: catalog.Experiment.BaselineVariantID, PersonaID: persona.ID, Adapter: "mock"})
		if response.Code != 202 {
			t.Fatal(response.Body.String())
		}
	}
	runs := waitCompleted(t, app, 4)
	at := func(value string) time.Time {
		parsed, err := time.Parse(time.RFC3339Nano, value)
		if err != nil {
			t.Fatal(err)
		}
		return parsed
	}
	firstDone := at(runs[0].CompletedAt)
	for _, run := range runs {
		if done := at(run.CompletedAt); done.Before(firstDone) {
			firstDone = done
		}
	}
	overlapping := 0
	for _, run := range runs {
		if at(run.StartedAt).Before(firstDone) {
			overlapping++
		}
	}
	if overlapping < 4 {
		t.Fatalf("only %d of 4 sessions started before the first finished", overlapping)
	}
}
