package experimentresults

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Synthetic unit fixtures only. Never uploaded or counted as observed activity.
func cohortPacket(t *testing.T) []byte {
	t.Helper()
	panels := []any{}
	schedule := []any{}
	indexRows := []any{}
	sources := []any{}
	pairs := []any{}
	for i := 0; i < 6; i++ {
		var e map[string]any
		_ = json.Unmarshal(packet(t), &e)
		report := e["report"].(map[string]any)
		report["synthetic"] = true
		rows := report["rows"].([]any)
		for _, raw := range rows {
			r := raw.(map[string]any)
			r["caseId"] = fmt.Sprintf("p%d-%s", i, r["caseId"])
			r["expectedRunId"] = fmt.Sprintf("p%d-%s", i, r["expectedRunId"])
			if s, ok := r["session"].(map[string]any); ok {
				s["caseId"] = r["caseId"]
				s["run"].(map[string]any)["id"] = r["expectedRunId"]
				s["realPurchase"] = false
			}
			s := map[string]any{"caseId": r["caseId"], "arm": r["arm"], "runId": r["expectedRunId"]}
			schedule = append(schedule, s)
			indexRows = append(indexRows, map[string]any{"caseId": r["caseId"], "arm": r["arm"], "runId": r["expectedRunId"], "serverConfirmed": false, "visibleCheckoutConfirmed": false, "receipt": nil, "readback": nil})
		}
		for j := 0; j < 20; j += 2 {
			a := rows[j].(map[string]any)
			b := rows[j+1].(map[string]any)
			pairs = append(pairs, map[string]any{"caseId": a["caseId"], "generic": map[string]any{"caseId": a["caseId"], "arm": "generic", "runId": a["expectedRunId"]}, "local-model": map[string]any{"caseId": b["caseId"], "arm": "local-model", "runId": b["expectedRunId"]}})
		}
		path := fmt.Sprintf("/fixture/packet-%d/aggregate-audited.json", i+1)
		sources = append(sources, map[string]any{"path": path, "sha256": strings.Repeat("a", 64)})
		panels = append(panels, map[string]any{"path": path, "sha256": strings.Repeat("a", 64), "experimentId": "actual16", "report": report})
	}
	document := func(path string, data any) map[string]any {
		return map[string]any{"path": path, "sha256": strings.Repeat("a", 64), "data": data}
	}
	plan := document("/fixture/cohort-plan.json", map[string]any{"version": "shopper-fixed-cohort.v1", "synthetic": true, "humanConversionClaim": false, "fixedPairs": 60, "fixedSessions": 120, "schedule": schedule})
	index := document("/fixture/server-outcomes.json", map[string]any{"schema": "shopper-server-outcomes.v1", "synthetic": true, "humanConversionClaim": false, "reports": sources, "rows": indexRows})
	arms := map[string]any{}
	for _, arm := range []string{"generic", "local-model"} {
		arms[arm] = map[string]any{"assigned": 60, "allServerConfirmedOrders": map[string]any{"evaluated": true}, "effort": map[string]any{"measuredSessionElapsedMs": map[string]any{"count": 0, "median": nil}}, "browserReadinessMs": map[string]any{"complete": map[string]any{"count": 0, "median": nil}}}
	}
	metrics := document("/fixture/metrics.json", map[string]any{"version": "shopper-synthetic-metrics.v1", "synthetic": true, "humanConversionClaim": false, "fixedCases": 60, "fixedSessions": 120, "sources": sources, "serverIndexSource": map[string]any{"path": index["path"], "sha256": index["sha256"]}, "pairs": pairs, "arms": arms})
	body, _ := json.Marshal(map[string]any{"version": "experiment-results.v1", "reportId": "cohort-test", "experimentId": "cohort-v2", "synthetic": true, "humanConversionClaim": false, "report": map[string]any{"version": "shopper-checkout-cohort.v1", "synthetic": true, "humanConversionClaim": false, "fixedCases": 60, "fixedSessions": 120, "plan": plan, "sourceReports": panels, "metrics": metrics, "serverOutcomes": index}})
	return body
}

func TestCohortExactRosterSourcesAndFakeClaims(t *testing.T) {
	raw := cohortPacket(t)
	decoded(t, raw)
	for _, change := range [][2]string{
		{`"fixedCases":60`, `"fixedCases":59`},
		{`"assigned":60`, `"assigned":10`},
		{`"evaluated":true`, `"evaluated":false`},
		{`"runId":"p0-run-01-generic"`, `"runId":"unknown-run"`},
		{`"experimentId":"actual16"`, `"experimentId":"different"`},
		{`"realPurchase":false`, `"realPurchase":true`},
		{`"path":"/fixture/server-outcomes.json","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"`, `"path":"/fixture/another.json","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"`},
	} {
		changed := bytes.Replace(raw, []byte(change[0]), []byte(change[1]), 1)
		if bytes.Equal(changed, raw) {
			t.Fatal("fixture mutation did not apply")
		}
		if _, _, err := Decode(bytes.NewReader(changed)); err == nil {
			t.Fatal("accepted corrupted cohort", change[0])
		}
	}
}

func TestPresentationStrictTigerNeverFallsBackAndPreservesRawViews(t *testing.T) {
	e, h := decoded(t, cohortPacket(t))
	repo := &fakeRepo{offline: true, records: map[string]Saved{}}
	s, err := New(t.TempDir(), repo)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	saved, _, err := s.Accept(e, h)
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	s.Register(mux)
	get := func(url string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest("GET", url, nil))
		return w
	}
	if w := get("/api/experiment-results/presentation?reportId=cohort-test&source=tiger"); w.Code != 503 {
		t.Fatal("local spool disguised as Tiger", w.Code)
	}
	repo.mu.Lock()
	repo.offline = false
	repo.records[e.ReportID] = saved
	repo.mu.Unlock()
	if w := get("/api/experiment-results/presentation?reportId=cohort-test"); w.Code != 200 || !strings.Contains(w.Body.String(), `"source":"tiger"`) || !strings.Contains(w.Body.String(), `"cloudAcknowledged":true`) {
		t.Fatal(w.Code, w.Body)
	}
	for _, view := range []string{"runs", "comparison", "latency"} {
		w := get("/api/experiment-results/" + view + "?reportId=cohort-test&source=tiger")
		if w.Code != 200 {
			t.Fatal(view, w.Code, w.Body)
		}
		if view == "runs" && bytes.Count(w.Body.Bytes(), []byte(`"expectedRunId"`)) != 120 {
			t.Fatal("roster truncated")
		}
		if view == "latency" && !strings.Contains(w.Body.String(), `"median":null`) {
			t.Fatal("missing measurement fabricated")
		}
	}
	if w := get("/api/experiment-results/reports/cohort-test?source=unknown"); w.Code != 400 {
		t.Fatal("unknown source accepted")
	}
	legacy, lh := decoded(t, packet(t))
	_, _, _ = s.Accept(legacy, lh)
	repo.mu.Lock()
	repo.records[legacy.ReportID] = Saved{Envelope: legacy, Digest: lh}
	repo.mu.Unlock()
	if w := get("/api/experiment-results/presentation?reportId=snapshot-1"); w.Code != 422 {
		t.Fatal("legacy smoke silently promoted", w.Code)
	}
}
