package shoppingevents

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

func fixture(t *testing.T) []byte {
	t.Helper()
	now := time.Now().UnixMilli()
	body := map[string]any{"schema": "fieldwork.shopping-batch.v1", "batchId": "00000000-0000-4000-8000-000000000001", "sessionId": "00000000-0000-4000-8000-000000000002", "classification": map[string]any{"environment": "test", "traffic": "synthetic"}, "startedAt": now, "events": []any{map[string]any{"eventId": "00000000-0000-4000-8000-000000000003", "sequence": 0, "at": now, "elapsedMs": 0, "action": "session-start", "context": map[string]any{"mode": "baseline", "order": "popular", "shipping": false, "modelVersion": "deterministic-v1"}, "data": map[string]any{"consent": "notice", "replay": true}}}, "replay": map[string]any{"schema": "fieldwork.sanitized-replay.v1", "events": []any{map[string]any{"type": 2, "timestamp": now, "data": map[string]any{"node": map[string]any{"type": 2, "id": 1, "tagName": "div", "attributes": map[string]any{"title": "PRIVATE_TITLE", "src": "https://private.example", "class": "user-secret"}, "childNodes": []any{map[string]any{"type": 3, "id": 2, "textContent": "PRIVATE_BODY"}}}, "initialOffset": map[string]any{"left": 0, "top": 0}}}}}}
	raw, _ := json.Marshal(body)
	return raw
}
func decode(t *testing.T, raw []byte) (Batch, string) {
	t.Helper()
	b, d, err := Decode(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	return b, d
}

type fake struct {
	mu      sync.Mutex
	offline bool
	records map[string]Saved
	replays map[string]*Replay
	calls   int
}

func (f *fake) Put(ctx context.Context, s Saved, r *Replay) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls++
	if f.offline {
		return errors.New("secret driver error")
	}
	if old, ok := f.records[s.Batch.BatchID]; ok && old.Digest != s.Digest {
		return ErrConflict
	}
	f.records[s.Batch.BatchID] = s
	f.replays[s.Batch.BatchID] = r
	return nil
}
func (f *fake) Prune(context.Context) error { return nil }
func waitFor(t *testing.T, condition func() bool) {
	t.Helper()
	until := time.Now().Add(2 * time.Second)
	for !condition() {
		if time.Now().After(until) {
			t.Fatal("worker timeout")
		}
		time.Sleep(time.Millisecond)
	}
}
func TestOfflineDurabilityReplayExpiryAndRestart(t *testing.T) {
	dir := t.TempDir()
	repo := &fake{offline: true, records: map[string]Saved{}, replays: map[string]*Replay{}}
	config := Config{Enabled: true, Environment: "test", AllowSynthetic: true}
	s, err := New(dir, config, repo)
	if err != nil {
		t.Fatal(err)
	}
	batch, digest := decode(t, fixture(t))
	if duplicate, err := s.Accept(batch, digest); err != nil || duplicate {
		t.Fatal(err)
	}
	waitFor(t, func() bool { repo.mu.Lock(); defer repo.mu.Unlock(); return repo.calls > 0 })
	s.Close()
	raw, err := os.ReadFile(filepath.Join(dir, batch.BatchID+".json"))
	if err != nil || bytes.Contains(raw, []byte(`"replay":{`)) || bytes.Contains(raw, []byte("PRIVATE")) {
		t.Fatal("replay leaked into observations archive", err)
	}
	repo.mu.Lock()
	repo.offline = false
	repo.mu.Unlock()
	s, err = New(dir, config, repo)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	waitFor(t, func() bool { _, err := os.Stat(filepath.Join(dir, batch.BatchID+".synced")); return err == nil })
	if duplicate, err := s.Accept(batch, digest); err != nil || !duplicate {
		t.Fatal("retry", err)
	}
	if _, err := s.Accept(batch, strings.Repeat("f", 64)); !errors.Is(err, ErrConflict) {
		t.Fatal("conflict", err)
	}
	saved, err := s.load(batch.BatchID)
	if err != nil {
		t.Fatal(err)
	}
	saved.ReplayExpiresAt = time.Now().Add(-time.Second)
	raw, _ = json.Marshal(saved)
	if err = atomic(dir, batch.BatchID+".json", raw); err != nil {
		t.Fatal(err)
	}
	if err = s.sweep(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err = os.Stat(filepath.Join(dir, batch.BatchID+".replay")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("expired local replay remains")
	}
}
func TestValidationAndReplayPrivacy(t *testing.T) {
	raw := fixture(t)
	batch, _ := decode(t, raw)
	safe, _ := json.Marshal(batch.Replay)
	for _, secret := range []string{"PRIVATE", "private.example", "user-secret", "title"} {
		if bytes.Contains(safe, []byte(secret)) {
			t.Fatal("replay retained", secret)
		}
	}
	for _, change := range [][2]string{{`"action":"session-start"`, `"action":"raw-input"`}, {`"sequence":0`, `"sequence":-1`}, {`"consent":"notice"`, `"consent":"maybe"`}, {`"replay":true`, `"replay":true,"query":"private"`}, {`"modelVersion":"deterministic-v1"`, `"modelVersion":"user@example.com"`}} {
		if _, _, err := Decode(bytes.NewReader(bytes.Replace(raw, []byte(change[0]), []byte(change[1]), 1))); err == nil {
			t.Fatal("accepted invalid", change[0])
		}
	}
	for _, source := range []int{5, 6, 9, 14} {
		var doc map[string]any
		_ = json.Unmarshal(raw, &doc)
		doc["replay"] = map[string]any{"schema": "fieldwork.sanitized-replay.v1", "events": []any{map[string]any{"type": 3, "timestamp": time.Now().UnixMilli(), "data": map[string]any{"source": source, "text": "private"}}}}
		changed, _ := json.Marshal(doc)
		if _, _, err := Decode(bytes.NewReader(changed)); err == nil {
			t.Fatal("private replay source accepted", source)
		}
	}
}

func TestReplayKeepsPublicImagesAndStripsPrivateCSS(t *testing.T) {
	for _, src := range []string{"/products/mug-sage.jpg", "/products/pack-sand.webp", "/products/example.png"} {
		if replayAttrs(map[string]any{"src": src})["src"] != src {
			t.Fatal("public product image lost", src)
		}
	}
	for _, src := range []string{"https://example.com/private.jpg", "/products/mug.jpg?secret=1", "/products/mug.jpg#secret"} {
		if _, ok := replayAttrs(map[string]any{"src": src})["src"]; ok {
			t.Fatal("private URL retained")
		}
	}
	sanitized := css(`.product { color: red; background:url(https://PRIVATE); content:'PRIVATE'; } @import "PRIVATE"; /* PRIVATE */`)
	if strings.Contains(sanitized, "PRIVATE") || strings.Contains(sanitized, "url(") || !strings.Contains(sanitized, "color: red") {
		t.Fatal("CSS privacy/layout regression", sanitized)
	}
}
func TestHTTPGatesCORSAndClaims(t *testing.T) {
	config := Config{Enabled: true, Environment: "test", AllowSynthetic: true, Origins: []string{"http://127.0.0.1:4329"}}
	s, err := New(t.TempDir(), config, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	mux := http.NewServeMux()
	s.Register(mux)
	request := func(method, origin string, raw []byte) *httptest.ResponseRecorder {
		r := httptest.NewRequest(method, "/api/shopping-events", bytes.NewReader(raw))
		r.Header.Set("Origin", origin)
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("Access-Control-Request-Method", "POST")
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, r)
		return w
	}
	if w := request("OPTIONS", config.Origins[0], nil); w.Code != 204 || w.Header().Get("Access-Control-Allow-Credentials") != "" {
		t.Fatal(w.Code, w.Header())
	}
	if w := request("POST", "https://evil.example", fixture(t)); w.Code != 403 {
		t.Fatal(w.Code)
	}
	body := fixture(t)
	if w := request("POST", config.Origins[0], body); w.Code != 202 || !strings.Contains(w.Body.String(), `"durableLocal":true`) {
		t.Fatal(w.Code, w.Body)
	}
	if w := request("POST", config.Origins[0], bytes.Replace(body, []byte(`"traffic":"synthetic"`), []byte(`"traffic":"real"`), 1)); w.Code != 403 {
		t.Fatal("dev claimed real", w.Code)
	}
	disabled, err := New(t.TempDir(), Config{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer disabled.Close()
	disabledMux := http.NewServeMux()
	disabled.Register(disabledMux)
	w := httptest.NewRecorder()
	disabledMux.ServeHTTP(w, httptest.NewRequest("POST", "/api/shopping-events", bytes.NewReader(body)))
	if w.Code != 404 {
		t.Fatal("disabled gate", w.Code)
	}
}
func TestTrainingExclusionsAndWholeSessionSplit(t *testing.T) {
	batch, digest := decode(t, fixture(t))
	saved := Saved{Batch: batch, Digest: digest, ReceivedAt: time.Now()}
	if len(TrainingRows(saved, false)) != 0 {
		t.Fatal("synthetic entered default training")
	}
	rows := TrainingRows(saved, true)
	if len(rows) != 1 || rows[0].Derived["preferenceSignal"] != nil || rows[0].Derived["negativePreference"] != nil {
		t.Fatal("invented preference")
	}
	batch.Classification = Classification{"production", "real"}
	saved.Batch = batch
	rows = TrainingRows(saved, false)
	if len(rows) != 1 {
		t.Fatal("real excluded")
	}
	before := rows[0].Split
	other := saved
	other.Batch.BatchID = "00000000-0000-4000-8000-000000000099"
	if TrainingRows(other, false)[0].Split != before {
		t.Fatal("session split leaked across batches")
	}
	var output bytes.Buffer
	manifest, err := Export(func(consume func(Saved) error) error { return consume(saved) }, &output, Selection{IncludeSynthetic: false})
	if err != nil || manifest.ReplayIncluded || bytes.Contains(output.Bytes(), []byte(`"replay":{`)) {
		t.Fatal("export replay leak", err)
	}
	for i := 0; i < 100; i++ {
		id := fmt.Sprintf("00000000-0000-4000-8000-%012d", i)
		if !oneOf(Split(id), "train", "validation", "test") {
			t.Fatal("bad split")
		}
	}
}
