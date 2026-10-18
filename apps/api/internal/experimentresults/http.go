package experimentresults

import (
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"net/http"
	"sort"
	"strconv"
)

func reply(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
func failure(w http.ResponseWriter, err error) {
	status := 503
	if errors.Is(err, ErrNotFound) {
		status = 404
	}
	if errors.Is(err, ErrConflict) {
		status = 409
	}
	if errors.Is(err, errBadSource) {
		reply(w, 400, map[string]string{"error": err.Error()})
		return
	}
	reply(w, status, map[string]string{"error": publicError(err).Error()})
}
func (s *Service) Register(mux *http.ServeMux) {
	s.registerPresentation(mux)
	mux.HandleFunc("POST /api/experiment-results", func(w http.ResponseWriter, r *http.Request) {
		// Local exporter sends no Origin. Reject browser-initiated writes rather
		// than allowing cross-site requests to this unauthenticated loopback API.
		media, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if r.Header.Get("Origin") != "" || media != "application/json" {
			reply(w, 403, map[string]string{"error": "use the local JSON exporter"})
			return
		}
		e, digest, err := Decode(http.MaxBytesReader(w, r.Body, MaxBody))
		if err != nil {
			reply(w, 400, map[string]string{"error": err.Error()})
			return
		}
		saved, duplicate, err := s.Accept(e, digest)
		if err != nil {
			failure(w, err)
			return
		}
		reply(w, 202, map[string]any{"reportId": saved.ReportID, "sha256": saved.Digest, "durableLocal": true, "duplicate": duplicate, "cloud": "asynchronous"})
	})
	mux.HandleFunc("GET /api/experiment-results/status", func(w http.ResponseWriter, r *http.Request) { reply(w, 200, s.Status()) })
	mux.HandleFunc("GET /api/experiment-results/reports", func(w http.ResponseWriter, r *http.Request) {
		limit := 50
		if q := r.URL.Query().Get("limit"); q != "" {
			n, err := strconv.Atoi(q)
			if err != nil || n < 1 || n > 100 {
				reply(w, 400, map[string]string{"error": "limit must be 1..100"})
				return
			}
			limit = n
		}
		experiment := r.URL.Query().Get("experimentId")
		if experiment != "" && !safeID.MatchString(experiment) {
			reply(w, 400, map[string]string{"error": "invalid experimentId"})
			return
		}
		items, source, err := s.List(r.Context(), experiment, limit)
		if err != nil {
			failure(w, err)
			return
		}
		reply(w, 200, map[string]any{"reports": items, "source": source, "limit": limit, "humanConversionClaim": false})
	})
	mux.HandleFunc("GET /api/experiment-results/reports/{id}", func(w http.ResponseWriter, r *http.Request) {
		saved, source, err := s.selected(r.Context(), r.PathValue("id"), r.URL.Query().Get("source"))
		if err != nil {
			failure(w, err)
			return
		}
		reply(w, 200, map[string]any{"result": saved, "source": source})
	})
	for _, view := range []string{"runs", "comparison", "latency"} {
		mux.HandleFunc("GET /api/experiment-results/"+view, func(w http.ResponseWriter, r *http.Request) {
			id := r.URL.Query().Get("reportId")
			if !safeID.MatchString(id) {
				reply(w, 400, map[string]string{"error": "reportId is required; choose one immutable report snapshot"})
				return
			}
			saved, source, err := s.selected(r.Context(), id, r.URL.Query().Get("source"))
			if err != nil {
				failure(w, err)
				return
			}
			var report Report
			if err = json.Unmarshal(saved.Report, &report); err != nil {
				failure(w, fmt.Errorf("invalid archived report"))
				return
			}
			var value any
			if report.Version == "shopper-checkout-cohort.v1" {
				value = cohortView(saved.Report, view)
			} else {
				switch view {
				case "runs":
					var raw struct {
						Rows json.RawMessage `json:"rows"`
					}
					_ = json.Unmarshal(saved.Report, &raw)
					value = raw.Rows
				case "comparison":
					value = report.Arms
				case "latency":
					value = latency(saved.Report)
				}
			}
			reply(w, 200, map[string]any{"reportId": id, "experimentId": saved.ExperimentID, "source": source, "synthetic": true, "humanConversionClaim": false, "data": value})
		})
	}
}

// Return harness-owned measurements verbatim with their precision/scope labels.
// No timing is derived from upload time, model attempts or unattempted sessions.
func latency(raw json.RawMessage) any {
	var report struct {
		Arms map[string]map[string]json.RawMessage `json:"arms"`
	}
	_ = json.Unmarshal(raw, &report)
	result := map[string]any{}
	for _, arm := range []string{"generic", "local-model"} {
		summary := report.Arms[arm]
		fields := map[string]json.RawMessage{}
		for _, key := range []string{"attemptsElapsedMs", "attemptsTimingPrecision", "readiness"} {
			if data, ok := summary[key]; ok {
				fields[key] = data
			}
		}
		var elapsed []float64
		_ = json.Unmarshal(summary["attemptsElapsedMs"], &elapsed)
		var readiness map[string]struct {
			Observed []float64 `json:"observedDurationsMs"`
			Verified []float64 `json:"verifiedArtifactDurationsMs"`
		}
		_ = json.Unmarshal(summary["readiness"], &readiness)
		stages := map[string]any{}
		for _, stage := range []string{"rank", "firstThree", "complete"} {
			measure := readiness[stage]
			stages[stage] = map[string]any{"observed": statistics(measure.Observed), "verifiedArtifacts": statistics(measure.Verified)}
		}
		result[arm] = map[string]any{"measurements": fields, "sessionElapsedMs": statistics(elapsed), "readinessMs": stages}
	}
	return map[string]any{"arms": result, "scope": "recorded session elapsed and browser after-paint opportunity; excludes physical paint and image transfer", "physicalPaintMeasured": false, "imageTransferMeasured": false}
}

// Nearest-rank percentiles. Empty measurements retain null summaries rather than
// treating absent, invalid or unattempted sessions as zero latency.
func statistics(values []float64) map[string]any {
	numbers := append([]float64{}, values...)
	sort.Float64s(numbers)
	result := map[string]any{"count": len(numbers), "min": nil, "max": nil, "mean": nil, "p50": nil, "p95": nil}
	if len(numbers) == 0 {
		return result
	}
	total := 0.0
	for _, n := range numbers {
		total += n
	}
	result["min"] = numbers[0]
	result["max"] = numbers[len(numbers)-1]
	result["mean"] = total / float64(len(numbers))
	result["p50"] = numbers[(len(numbers)*50+99)/100-1]
	result["p95"] = numbers[(len(numbers)*95+99)/100-1]
	return result
}
