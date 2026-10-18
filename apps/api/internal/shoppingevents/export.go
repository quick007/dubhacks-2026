package shoppingevents

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"time"
)

// TrainingObservation preserves raw structured observations and derived signals
// separately. A view/nonclick never becomes a negative preference label.
type TrainingObservation struct {
	Schema      string         `json:"schema"`
	Split       string         `json:"split"`
	Provenance  map[string]any `json:"provenance"`
	Observation Event          `json:"observation"`
	Derived     map[string]any `json:"derived"`
}

func Split(session string) string {
	hash := sha256.Sum256([]byte("fieldwork.session-split.v1/" + session))
	bucket := int(hash[0]) % 10
	if bucket == 8 {
		return "validation"
	}
	if bucket == 9 {
		return "test"
	}
	return "train"
}
func TrainingRows(s Saved, includeSynthetic bool) []TrainingObservation {
	b := s.Batch
	if !includeSynthetic && (b.Classification.Environment != "production" || b.Classification.Traffic != "real") {
		return nil
	}
	rows := []TrainingObservation{}
	for _, event := range b.Events {
		var preference any
		if event.Action == "selection" || event.Action == "cart-add" {
			preference = "explicit-positive-action"
		}
		derived := map[string]any{"schema": "fieldwork.behavior-signal.v1", "preferenceSignal": preference, "negativePreference": nil, "realPurchaseObserved": false, "demoCheckoutObserved": event.Action == "checkout-success"}
		rows = append(rows, TrainingObservation{Schema: "fieldwork.training-observation.v1", Split: Split(b.SessionID), Provenance: map[string]any{"sourceSchema": b.Schema, "batchId": b.BatchID, "sessionId": b.SessionID, "eventId": event.EventID, "sourceSha256": s.Digest, "receivedAt": s.ReceivedAt.UTC().Format(time.RFC3339Nano), "classification": b.Classification, "splitVersion": "fieldwork.session-split.v1", "modelVersionSource": "client-reported-trusted-build-config"}, Observation: event, Derived: derived})
	}
	return rows
}

type Manifest struct {
	Schema            string         `json:"schema"`
	ObservationSchema string         `json:"observationSchema"`
	Since             string         `json:"since"`
	Until             string         `json:"until"`
	IncludeSynthetic  bool           `json:"includeSynthetic"`
	Batches           int            `json:"batches"`
	Rows              int            `json:"rows"`
	Sessions          int            `json:"sessions"`
	Splits            map[string]int `json:"splits"`
	ReplayIncluded    bool           `json:"replayIncluded"`
}

func Export(ctxWalk func(func(Saved) error) error, out io.Writer, selection Selection) (Manifest, error) {
	manifest := Manifest{Schema: "fieldwork.training-export.v1", ObservationSchema: "fieldwork.training-observation.v1", Since: selection.Since.UTC().Format(time.RFC3339Nano), Until: selection.Until.UTC().Format(time.RFC3339Nano), IncludeSynthetic: selection.IncludeSynthetic, Splits: map[string]int{"train": 0, "validation": 0, "test": 0}}
	sessions := map[string]bool{}
	seen := map[string]bool{}
	encoder := json.NewEncoder(out)
	err := ctxWalk(func(saved Saved) error {
		rows := TrainingRows(saved, selection.IncludeSynthetic)
		if len(rows) == 0 {
			return nil
		}
		manifest.Batches++
		sessions[saved.Batch.SessionID] = true
		for _, row := range rows {
			if seen[row.Observation.EventID] {
				return fmt.Errorf("duplicate training observation ID")
			}
			seen[row.Observation.EventID] = true
			if err := encoder.Encode(row); err != nil {
				return err
			}
			manifest.Rows++
			manifest.Splits[row.Split]++
		}
		return nil
	})
	manifest.Sessions = len(sessions)
	return manifest, err
}
