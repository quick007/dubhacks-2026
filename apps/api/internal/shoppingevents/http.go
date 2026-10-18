package shoppingevents

import (
	"encoding/json"
	"errors"
	"mime"
	"net/http"
)

func respond(w http.ResponseWriter, code int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(body)
}
func (s *Service) origin(w http.ResponseWriter, r *http.Request) bool {
	origin := r.Header.Get("Origin")
	if !oneOf(origin, s.config.Origins...) {
		respond(w, 403, map[string]string{"error": "shopping origin is not configured"})
		return false
	}
	w.Header().Set("Access-Control-Allow-Origin", origin)
	w.Header().Add("Vary", "Origin")
	return true
}
func (s *Service) Register(mux *http.ServeMux) {
	mux.HandleFunc("OPTIONS /api/shopping-events", func(w http.ResponseWriter, r *http.Request) {
		if !s.config.Enabled {
			respond(w, 404, map[string]string{"error": "shopping telemetry disabled"})
			return
		}
		if !s.origin(w, r) {
			return
		}
		if r.Header.Get("Access-Control-Request-Method") != "POST" {
			respond(w, 405, map[string]string{"error": "POST required"})
			return
		}
		w.Header().Set("Access-Control-Allow-Methods", "POST")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.Header().Set("Access-Control-Max-Age", "600")
		w.WriteHeader(204)
	})
	mux.HandleFunc("POST /api/shopping-events", func(w http.ResponseWriter, r *http.Request) {
		if !s.config.Enabled {
			respond(w, 404, map[string]string{"error": "shopping telemetry disabled"})
			return
		}
		if !s.origin(w, r) {
			return
		}
		media, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if media != "application/json" {
			respond(w, 415, map[string]string{"error": "JSON shopping batch required"})
			return
		}
		b, digest, err := Decode(http.MaxBytesReader(w, r.Body, MaxBody))
		if err != nil {
			respond(w, 400, map[string]string{"error": err.Error()})
			return
		}
		if !s.config.permits(b) {
			respond(w, 403, map[string]string{"error": "batch classification differs from server telemetry policy"})
			return
		}
		duplicate, err := s.Accept(b, digest)
		if err != nil {
			code := 503
			message := "shopping archive unavailable"
			if errors.Is(err, ErrConflict) {
				code = 409
				message = err.Error()
			}
			if errors.Is(err, ErrCapacity) {
				code = 429
				message = err.Error()
			}
			respond(w, code, map[string]string{"error": message})
			return
		}
		respond(w, 202, map[string]any{"batchId": b.BatchID, "durableLocal": true, "duplicate": duplicate, "cloud": "asynchronous"})
	})
	// Aggregate operational status only; no observation/replay export route.
	mux.HandleFunc("GET /api/shopping-events/status", func(w http.ResponseWriter, r *http.Request) { respond(w, 200, s.Status()) })
}
