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
		for _, persona := range []contracts.Persona{catalog.Personas[0], catalog.Personas[2]} {
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
		if !reflect.DeepEqual(run.Config, snapshot.Config) || !reflect.DeepEqual(run.Persona, snapshot.Persona) || run.Seed != snapshot.Seed || run.BaselineVariantID != "baseline" {
			t.Fatal("immutable snapshot changed")
		}
		if run.Evaluation == nil || !run.Evaluation.Synthetic || len(run.Artifacts) != 1 {
			t.Fatal("missing synthetic evaluation or result artifact")
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
		artifact := request(t, app, "GET", "/api/artifacts/"+run.Artifacts[0].ID, nil)
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
		if run.PersonaSplit == "development" && run.VariantID == "baseline" {
			if len(run.ProposedChanges) != 1 {
				t.Fatal("missing evidence-based development proposal")
			}
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
	body := map[string]any{"experimentId": "demo-experiment", "variantId": "baseline", "personaId": "dev-budget"}
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
