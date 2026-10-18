package main

import (
	"crypto/rand"
	"database/sql"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"storefront/api/internal/contracts"
	"storefront/api/internal/experimentresults"
	"storefront/api/internal/scenario"
	"storefront/api/internal/shoppingevents"
	"strings"
	"sync"
	"time"
)

//go:embed fixtures/catalog.json
var catalogJSON []byte

//go:embed fixtures/scenarios/jacket.json
var scenarioJSON []byte
var errQueueFull = errors.New("local queue is full; wait for an active run to complete")

func now() string { return time.Now().UTC().Format(time.RFC3339Nano) }
func newID() string {
	var body [16]byte
	if _, err := rand.Read(body[:]); err != nil {
		panic(err)
	}
	return hex.EncodeToString(body[:])
}

type AppOptions struct {
	DataDir       string
	WorkspaceRoot string
	MockStepDelay time.Duration
	QueueCapacity int
}
type App struct {
	store       *Store
	results     *experimentresults.Service
	shopping    *shoppingevents.Service
	queue       *JobQueue
	mux         *http.ServeMux
	releaseLock func()
	closeOnce   sync.Once
}

func NewApp(options AppOptions) (*App, error) {
	mockScenario, err := scenario.Load(scenarioJSON)
	if err != nil {
		return nil, err
	}
	if options.DataDir == "" {
		options.DataDir = os.Getenv("STOREFRONT_DATA_DIR")
		if options.DataDir == "" {
			options.DataDir = "../../data"
		}
	}
	dataDir, err := filepath.Abs(options.DataDir)
	if err != nil {
		return nil, err
	}
	if options.WorkspaceRoot == "" {
		options.WorkspaceRoot, err = filepath.Abs("../..")
		if err != nil {
			return nil, err
		}
	}
	if options.QueueCapacity <= 0 {
		options.QueueCapacity = 20
	}
	if options.MockStepDelay == 0 {
		options.MockStepDelay = 200 * time.Millisecond
	}
	release, err := acquireLock(dataDir)
	if err != nil {
		return nil, err
	}
	store, err := openStore(dataDir)
	if err != nil {
		release()
		return nil, err
	}
	var resultsRepo experimentresults.Repository
	if dsn := os.Getenv("TIGER_DATABASE_URL"); dsn != "" {
		resultsRepo, err = experimentresults.NewPostgres(dsn)
		if err != nil {
			store.db.Close()
			release()
			return nil, err
		}
	}
	results, err := experimentresults.New(filepath.Join(dataDir, "experiment-results"), resultsRepo)
	if err != nil {
		store.db.Close()
		release()
		return nil, err
	}
	shoppingConfig, err := shoppingevents.ConfigFromEnv()
	if err != nil {
		results.Close()
		store.db.Close()
		release()
		return nil, err
	}
	var shoppingRepo shoppingevents.Repository
	if shoppingConfig.Enabled && os.Getenv("TIGER_DATABASE_URL") != "" {
		shoppingRepo, err = shoppingevents.NewPostgres(os.Getenv("TIGER_DATABASE_URL"))
		if err != nil {
			results.Close()
			store.db.Close()
			release()
			return nil, err
		}
	}
	shopping, err := shoppingevents.New(filepath.Join(dataDir, "shopping-events"), shoppingConfig, shoppingRepo)
	if err != nil {
		results.Close()
		store.db.Close()
		release()
		return nil, err
	}
	app := &App{store: store, results: results, shopping: shopping, queue: newQueue(store, options.QueueCapacity, options.MockStepDelay, options.WorkspaceRoot, mockScenario), mux: http.NewServeMux(), releaseLock: release}
	app.routes()
	return app, nil
}
func (a *App) Close() {
	a.closeOnce.Do(func() { a.queue.close(); a.results.Close(); a.shopping.Close(); a.store.db.Close(); a.releaseLock() })
}
func (a *App) ServeHTTP(w http.ResponseWriter, r *http.Request) { a.mux.ServeHTTP(w, r) }
func jsonResponse(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
func jsonError(w http.ResponseWriter, status int, err error) {
	jsonResponse(w, status, map[string]string{"error": err.Error()})
}
func (a *App) routes() {
	a.results.Register(a.mux)
	a.shopping.Register(a.mux)
	a.mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { jsonResponse(w, 200, map[string]bool{"ok": true}) })
	a.mux.HandleFunc("GET /api/catalog", func(w http.ResponseWriter, r *http.Request) {
		catalog, err := a.store.catalog()
		if err != nil {
			jsonError(w, 500, err)
			return
		}
		jsonResponse(w, 200, catalog)
	})
	a.mux.HandleFunc("GET /api/runs", func(w http.ResponseWriter, r *http.Request) {
		runs, err := a.store.listRuns()
		if err != nil {
			jsonError(w, 500, err)
			return
		}
		jsonResponse(w, 200, runs)
	})
	a.mux.HandleFunc("GET /api/runs/{id}", func(w http.ResponseWriter, r *http.Request) {
		run, err := a.store.run(r.PathValue("id"))
		if errors.Is(err, sql.ErrNoRows) {
			jsonError(w, 404, fmt.Errorf("run not found"))
			return
		}
		if err != nil {
			jsonError(w, 500, err)
			return
		}
		jsonResponse(w, 200, run)
	})
	a.mux.HandleFunc("POST /api/runs", a.createRun)
	a.mux.HandleFunc("GET /api/comparison", a.comparison)
	a.mux.HandleFunc("GET /api/artifacts/{id}", a.artifact)
	a.mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) { jsonError(w, 404, fmt.Errorf("not found")) })
}
func validID(value string) bool { return len(value) > 0 && len(value) <= 100 }
func (a *App) createRun(w http.ResponseWriter, r *http.Request) {
	var input struct {
		ExperimentID string          `json:"experimentId"`
		VariantID    string          `json:"variantId"`
		PersonaID    string          `json:"personaId"`
		Adapter      json.RawMessage `json:"adapter"`
	}
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16384))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&input); err != nil {
		jsonError(w, 400, fmt.Errorf("invalid run request: %w", err))
		return
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		jsonError(w, 400, fmt.Errorf("expected one JSON request object"))
		return
	}
	request := contracts.RunRequest{ExperimentID: input.ExperimentID, VariantID: input.VariantID, PersonaID: input.PersonaID, Adapter: "mock"}
	if input.Adapter != nil {
		if err := json.Unmarshal(input.Adapter, &request.Adapter); err != nil || string(input.Adapter) == "null" {
			jsonError(w, 400, fmt.Errorf("invalid adapter"))
			return
		}
	}
	if !validID(request.ExperimentID) || !validID(request.VariantID) || !validID(request.PersonaID) || (request.Adapter != "mock" && request.Adapter != "playwright") {
		jsonError(w, 400, fmt.Errorf("invalid experiment, variant, persona, or adapter"))
		return
	}
	catalog, err := a.store.catalog()
	if err != nil {
		jsonError(w, 500, err)
		return
	}
	var variant *contracts.Variant
	var persona *contracts.Persona
	for i := range catalog.Variants {
		v := &catalog.Variants[i]
		if v.ID == request.VariantID && v.ExperimentID == request.ExperimentID {
			variant = v
		}
	}
	for i := range catalog.Personas {
		if catalog.Personas[i].ID == request.PersonaID {
			persona = &catalog.Personas[i]
		}
	}
	if request.ExperimentID != catalog.Experiment.ID || variant == nil || persona == nil {
		jsonError(w, 400, fmt.Errorf("unknown experiment, variant, or persona"))
		return
	}
	run := contracts.Run{ID: newID(), ExperimentID: request.ExperimentID, BaselineVariantID: catalog.Experiment.BaselineVariantID, VariantID: variant.ID, PersonaID: persona.ID, PersonaSplit: persona.Split, Seed: catalog.Experiment.Seed, Config: variant.Config, Persona: *persona, Adapter: request.Adapter, Synthetic: true, ContractVersion: 1, Status: "queued", CreatedAt: now(), Events: []contracts.StepEvent{}, Artifacts: []contracts.Artifact{}, ProposedChanges: []contracts.ProposedChange{}}
	if err = a.queue.submit(run); err != nil {
		status := 500
		if errors.Is(err, errQueueFull) {
			status = 429
		}
		jsonError(w, status, err)
		return
	}
	jsonResponse(w, 202, run)
}
func (a *App) comparison(w http.ResponseWriter, r *http.Request) {
	runs, err := a.store.listRuns()
	if err != nil {
		jsonError(w, 500, err)
		return
	}
	// Runs from earlier catalog fixtures stay stored; compare only the current experiment.
	catalog, err := a.store.catalog()
	if err != nil {
		jsonError(w, 500, err)
		return
	}
	rows := []contracts.ComparisonRow{}
	indices := map[string]int{}
	for _, run := range runs {
		if run.ExperimentID != catalog.Experiment.ID || run.Status != "completed" || run.Evaluation == nil || run.Adapter != "mock" {
			continue
		}
		key := run.VariantID + "/" + run.PersonaSplit
		index, found := indices[key]
		if !found {
			index = len(rows)
			indices[key] = index
			rows = append(rows, contracts.ComparisonRow{VariantID: run.VariantID, PersonaSplit: run.PersonaSplit, Synthetic: true})
		}
		row := &rows[index]
		row.Runs++
		row.MeanScore += run.Evaluation.Score
		if run.Evaluation.TaskCompleted {
			row.CompletedTasks++
		}
	}
	for i := range rows {
		rows[i].MeanScore /= float64(rows[i].Runs)
	}
	jsonResponse(w, 200, rows)
}
func (a *App) artifact(w http.ResponseWriter, r *http.Request) {
	runs, err := a.store.listRuns()
	if err != nil {
		jsonError(w, 500, err)
		return
	}
	var artifact *contracts.Artifact
	for _, run := range runs {
		for _, item := range run.Artifacts {
			if item.ID == r.PathValue("id") {
				copy := item
				artifact = &copy
			}
		}
	}
	if artifact == nil {
		jsonError(w, 404, fmt.Errorf("artifact not found"))
		return
	}
	file := filepath.Join(a.store.dataDir, filepath.FromSlash(artifact.Path))
	artifactRoot, err := filepath.EvalSymlinks(filepath.Join(a.store.dataDir, "artifacts"))
	if err != nil {
		jsonError(w, 404, fmt.Errorf("artifact directory missing"))
		return
	}
	root := artifactRoot + string(os.PathSeparator)
	real, err := filepath.EvalSymlinks(file)
	if err != nil {
		jsonError(w, 404, fmt.Errorf("artifact file missing"))
		return
	}
	if !strings.HasPrefix(real, root) {
		jsonError(w, 403, fmt.Errorf("artifact outside local data directory"))
		return
	}
	handle, err := os.Open(real)
	if err != nil {
		jsonError(w, 404, fmt.Errorf("artifact file missing"))
		return
	}
	defer handle.Close()
	info, err := handle.Stat()
	if err != nil || !info.Mode().IsRegular() {
		jsonError(w, 404, fmt.Errorf("artifact file missing"))
		return
	}
	w.Header().Set("Content-Type", artifact.MimeType)
	w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; sandbox")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", filepath.Base(real)))
	http.ServeContent(w, r, info.Name(), info.ModTime(), handle)
}
