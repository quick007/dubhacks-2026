package experimentresults

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func packet(t *testing.T) []byte {
	t.Helper()
	rows := []any{}
	for i := 1; i <= 10; i++ {
		for _, arm := range []string{"generic", "local-model"} {
			row := map[string]any{"caseId": fmt.Sprintf("pair-%02d", i), "arm": arm, "expectedRunId": fmt.Sprintf("run-%02d-%s", i, arm), "status": "unattempted", "formalStatus": "unattempted", "errors": []string{}}
			if i == 1 {
				row["status"] = "recorded"
				row["session"] = map[string]any{"contractVersion": "shopper-session.v1", "caseId": row["caseId"], "arm": arm, "run": map[string]any{"id": row["expectedRunId"], "experimentId": "actual16"}, "elapsedMs": 41, "model": map[string]any{"readiness": []any{}}}
				row["evaluator"] = map[string]any{"expectedOutcomeMet": false, "evidenceVerified": false}
			}
			rows = append(rows, row)
		}
	}
	arms := map[string]any{}
	for _, arm := range []string{"generic", "local-model"} {
		arms[arm] = map[string]any{"fixedDenominator": 10, "recorded": 1, "formalRecorded": 0, "unattempted": 9, "evidenceVerifiedExpectedOutcomeRateFixedDenominator": 0, "attemptsElapsedMs": []int{41}, "attemptsTimingPrecision": []any{map[string]any{"sessionElapsed": "upper-bound"}}, "readiness": map[string]any{"firstThree": map[string]any{"observedDurationsMs": []int{7}, "verifiedArtifactDurationsMs": []int{}, "scope": "after-paint opportunity"}}}
	}
	body, _ := json.Marshal(map[string]any{"version": "experiment-results.v1", "reportId": "snapshot-1", "experimentId": "actual16", "synthetic": true, "humanConversionClaim": false, "report": map[string]any{"version": "shopper-evidence-aggregate.v1", "fixedCases": 10, "fixedSessions": 20, "humanConversionClaim": false, "rows": rows, "arms": arms}})
	return body
}
func decoded(t *testing.T, body []byte) (Envelope, string) {
	t.Helper()
	e, h, err := Decode(bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	return e, h
}

type fakeRepo struct {
	mu      sync.Mutex
	offline bool
	records map[string]Saved
	calls   int
}

func (f *fakeRepo) Put(ctx context.Context, s Saved) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls++
	if f.offline {
		return errors.New("secret-like connection details must never escape")
	}
	if old, ok := f.records[s.ReportID]; ok && old.Digest != s.Digest {
		return ErrConflict
	}
	f.records[s.ReportID] = s
	return nil
}
func (f *fakeRepo) Get(ctx context.Context, id string) (Saved, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.offline {
		return Saved{}, errors.New("unavailable")
	}
	s, ok := f.records[id]
	if !ok {
		return s, ErrNotFound
	}
	return s, nil
}
func (f *fakeRepo) List(ctx context.Context, experiment string, limit int) ([]Header, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.offline {
		return nil, errors.New("unavailable")
	}
	result := []Header{}
	for _, s := range f.records {
		if experiment == "" || s.ExperimentID == experiment {
			result = append(result, s.Header())
		}
	}
	return result, nil
}
func until(t *testing.T, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for !ok() {
		if time.Now().After(deadline) {
			t.Fatal("background operation did not finish")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestDurableOfflineRetryRestartAndConflict(t *testing.T) {
	dir := t.TempDir()
	repo := &fakeRepo{offline: true, records: map[string]Saved{}}
	s, err := New(dir, repo)
	if err != nil {
		t.Fatal(err)
	}
	e, h := decoded(t, packet(t))
	first, duplicate, err := s.Accept(e, h)
	if err != nil || duplicate {
		t.Fatalf("accept: %v %v", duplicate, err)
	}
	until(t, func() bool { repo.mu.Lock(); defer repo.mu.Unlock(); return repo.calls > 0 })
	if got, _, err := s.Get(context.Background(), e.ReportID); err != nil || got.Digest != h {
		t.Fatal("offline receipt lost", err)
	}
	_, source, err := s.List(context.Background(), "actual16", 10)
	if err != nil || source != "local-fallback" {
		t.Fatal(source, err)
	}
	s.Close()
	repo.mu.Lock()
	repo.offline = false
	repo.mu.Unlock()
	s, err = New(dir, repo)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	until(t, func() bool { _, err := os.Stat(filepath.Join(dir, e.ReportID+".synced")); return err == nil })
	if _, duplicate, err = s.Accept(e, h); err != nil || !duplicate {
		t.Fatal("identical retry", err)
	}
	changed := e
	changed.Report = bytes.ReplaceAll(e.Report, []byte(`"elapsedMs":41`), []byte(`"elapsedMs":42`))
	encoded, _ := json.Marshal(changed)
	changed, hash := decoded(t, encoded)
	if _, _, err = s.Accept(changed, hash); !errors.Is(err, ErrConflict) {
		t.Fatal("expected conflict", err)
	}
	if got, _, _ := s.Get(context.Background(), e.ReportID); got.ReceivedAt != first.ReceivedAt || got.Digest != h {
		t.Fatal("immutable receipt changed")
	}
	if info, err := os.Stat(filepath.Join(dir, e.ReportID+".json")); err != nil || info.Mode().Perm() != 0600 {
		t.Fatal("receipt permissions", err)
	}
}
func TestDecodeRejectsIdentityStateAndClaims(t *testing.T) {
	body := packet(t)
	for _, pair := range [][2]string{
		{`"synthetic":true`, `"synthetic":false`},
		{`"humanConversionClaim":false`, `"humanConversionClaim":true`},
		{`"fixedSessions":20`, `"fixedSessions":19`},
		{`"expectedRunId":"run-02-generic"`, `"expectedRunId":"run-01-generic"`},
		{`"elapsedMs":41`, `"elapsedMs":-1`},
		{`"reportId":"snapshot-1"`, `"reportId":"../secret"`},
		{`"fixedDenominator":10`, `"fixedDenominator":1`},
	} {
		if _, _, err := Decode(bytes.NewReader(bytes.ReplaceAll(body, []byte(pair[0]), []byte(pair[1])))); err == nil {
			t.Errorf("accepted invalid %s", pair[0])
		}
	}
	if _, _, err := Decode(bytes.NewReader(append(body, []byte(` {}`)...))); err == nil {
		t.Fatal("accepted trailing JSON")
	}
	e, h := decoded(t, body)
	pretty, _ := json.MarshalIndent(e, "", "  ")
	_, h2 := decoded(t, pretty)
	if h != h2 {
		t.Fatal("JSON whitespace changed identity")
	}
}
func TestHTTPViewsAndOriginProtection(t *testing.T) {
	s, err := New(t.TempDir(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	mux := http.NewServeMux()
	s.Register(mux)
	post := func(body []byte, origin string) *httptest.ResponseRecorder {
		r := httptest.NewRequest("POST", "/api/experiment-results", bytes.NewReader(body))
		r.Header.Set("Content-Type", "application/json")
		if origin != "" {
			r.Header.Set("Origin", origin)
		}
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, r)
		return w
	}
	if w := post(packet(t), "https://example.com"); w.Code != 403 {
		t.Fatal(w.Code)
	}
	if w := post(packet(t), ""); w.Code != 202 || !strings.Contains(w.Body.String(), `"durableLocal":true`) {
		t.Fatal(w.Code, w.Body.String())
	}
	for _, view := range []string{"runs", "comparison", "latency"} {
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest("GET", "/api/experiment-results/"+view+"?reportId=snapshot-1", nil))
		if w.Code != 200 || !strings.Contains(w.Body.String(), `"humanConversionClaim":false`) {
			t.Fatal(view, w.Code, w.Body.String())
		}
		if view == "runs" && (!strings.Contains(w.Body.String(), "unattempted") || !strings.Contains(w.Body.String(), `"formalStatus":"unattempted"`)) {
			t.Fatal("missing state discarded")
		}
		if view == "latency" && (!strings.Contains(w.Body.String(), "upper-bound") || !strings.Contains(w.Body.String(), "verifiedArtifactDurationsMs")) {
			t.Fatal("timing precision discarded")
		}
	}
	for _, url := range []string{"/api/experiment-results/runs", "/api/experiment-results/reports?limit=101"} {
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest("GET", url, nil))
		if w.Code != 400 {
			t.Fatal(url, w.Code)
		}
	}
}
func TestTLSConfigurationAndRedaction(t *testing.T) {
	p, err := NewPostgres("postgresql://example:placeholder@db.example:5432/evidence?sslmode=verify-full")
	if err != nil {
		t.Fatal(err)
	}
	if p.config.TLSConfig.InsecureSkipVerify || p.config.TLSConfig.ServerName != "db.example" || len(p.config.Fallbacks) != 0 || p.config.ConnectTimeout > 3*time.Second {
		t.Fatal("unsafe TLS or timeout")
	}
	encrypted, err := NewPostgres("postgresql://example:placeholder@db.example/evidence?sslmode=require")
	if err != nil || encrypted.config.TLSConfig == nil || len(encrypted.config.Fallbacks) != 0 {
		t.Fatal("require must use TLS without plaintext fallback")
	}
	for _, dsn := range []string{"postgresql://example:placeholder@db.example/evidence?sslmode=disable", "bad-placeholder", "postgresql://example:placeholder@db.example/evidence?sslmode=require&service=secret"} {
		_, err := NewPostgres(dsn)
		if err == nil || strings.Contains(err.Error(), "placeholder") {
			t.Fatal("invalid config not safely rejected")
		}
	}
}

func TestLatencyStatisticsPreserveMissingMeasurements(t *testing.T) {
	empty := statistics(nil)
	if empty["count"] != 0 || empty["mean"] != nil || empty["p95"] != nil {
		t.Fatal("missing timings became zero measurements")
	}
	stats := statistics([]float64{10, 2, 6, 4})
	if stats["mean"] != 5.5 || stats["p50"] != float64(4) || stats["p95"] != float64(10) {
		t.Fatal(stats)
	}
}
