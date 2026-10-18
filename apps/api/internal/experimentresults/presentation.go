package experimentresults

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"time"
)

var errBadSource = errors.New("source must be omitted or tiger")

// Explicit Tiger reads never succeed from the local spool or fall back offline.
func (s *Service) selected(ctx context.Context, id, source string) (Saved, string, error) {
	if !safeID.MatchString(id) {
		return Saved{}, source, ErrNotFound
	}
	if source == "" {
		return s.Get(ctx, id)
	}
	if source != "tiger" {
		return Saved{}, source, errBadSource
	}
	if s.repo == nil {
		return Saved{}, "tiger", errors.New("Tiger repository unavailable")
	}
	operation, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	saved, err := s.repo.Get(operation, id)
	return saved, "tiger", err
}

func (s *Service) registerPresentation(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/experiment-results/presentation", func(w http.ResponseWriter, r *http.Request) {
		id := r.URL.Query().Get("reportId")
		if !safeID.MatchString(id) {
			reply(w, 400, map[string]string{"error": "reportId is required; choose one immutable cohort"})
			return
		}
		source := r.URL.Query().Get("source")
		if source == "" {
			source = "tiger"
		}
		saved, source, err := s.selected(r.Context(), id, source)
		if err != nil {
			failure(w, err)
			return
		}
		if validateCohort(saved.Envelope) != nil {
			reply(w, 422, map[string]string{"error": "selected report is not a verified checkout cohort"})
			return
		}
		var cohort CheckoutCohort
		_ = json.Unmarshal(saved.Report, &cohort)
		sources := []map[string]any{}
		for _, v := range cohort.SourceReports {
			sources = append(sources, map[string]any{"path": v.Path, "sha256": v.SHA256, "experimentId": v.ExperimentID})
		}
		var slide any
		if cohort.SlideMetrics != nil {
			slide = cohort.SlideMetrics.Data
		}
		reply(w, 200, map[string]any{
			"version": "experiment-presentation.v1", "reportId": saved.ReportID, "experimentId": saved.ExperimentID,
			"sha256": saved.Digest, "receivedAt": saved.ReceivedAt, "source": source, "cloudAcknowledged": source == "tiger",
			"synthetic": true, "humanConversionClaim": false,
			"data": map[string]any{"version": cohort.Version, "fixedCases": cohort.FixedCases, "fixedSessions": cohort.FixedSessions,
				"metrics": cohort.Metrics.Data, "slideMetrics": slide,
				"provenance": map[string]any{"plan": cohort.Plan.EvidenceSource, "sourceReports": sources, "metrics": cohort.Metrics.EvidenceSource, "serverOutcomes": cohort.ServerOutcomes.EvidenceSource},
			},
		})
	})
}

func cohortView(raw json.RawMessage, view string) any {
	var c CheckoutCohort
	_ = json.Unmarshal(raw, &c)
	if view == "runs" {
		rows := []json.RawMessage{}
		for _, s := range c.SourceReports {
			var r struct {
				Rows []json.RawMessage `json:"rows"`
			}
			_ = json.Unmarshal(s.Report, &r)
			rows = append(rows, r.Rows...)
		}
		return rows
	}
	var m struct {
		Arms map[string]json.RawMessage `json:"arms"`
	}
	_ = json.Unmarshal(c.Metrics.Data, &m)
	if view == "comparison" {
		return m.Arms
	}
	result := map[string]any{}
	for _, arm := range []string{"generic", "local-model"} {
		var a struct {
			Effort    json.RawMessage `json:"effort"`
			Readiness json.RawMessage `json:"browserReadinessMs"`
		}
		_ = json.Unmarshal(m.Arms[arm], &a)
		result[arm] = map[string]any{"effort": a.Effort, "browserReadinessMs": a.Readiness}
	}
	return map[string]any{"arms": result, "scope": "verbatim harness measured-monotonic session distributions and verified browser after-paint opportunity; excludes physical paint/image transfer", "physicalPaintMeasured": false, "imageTransferMeasured": false}
}
