// Package shoppingevents collects bounded anonymous, structured observations.
package shoppingevents

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"strings"
	"time"
)

const MaxBody = 512 << 10
const MaxSessionMS = 8 * 60 * 60 * 1000

var uuid = regexp.MustCompile(`^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$`)
var versionID = regexp.MustCompile(`^[A-Za-z0-9._-]{1,64}$`)

//go:embed catalog.json
var catalogJSON []byte
var catalog = func() map[string][]string { var c map[string][]string; _ = json.Unmarshal(catalogJSON, &c); return c }()
var ErrConflict = errors.New("immutable shopping batch identity conflicts with saved data")
var ErrCapacity = errors.New("shopping telemetry local capacity reached")

type Classification struct {
	Environment string `json:"environment"`
	Traffic     string `json:"traffic"`
}
type Context struct {
	Mode         string  `json:"mode"`
	Order        string  `json:"order"`
	Shipping     *bool   `json:"shipping"`
	ModelVersion *string `json:"modelVersion"`
}
type Event struct {
	EventID   string          `json:"eventId"`
	Sequence  *int            `json:"sequence"`
	At        int64           `json:"at"`
	ElapsedMS *int64          `json:"elapsedMs"`
	Action    string          `json:"action"`
	Context   Context         `json:"context"`
	Data      json.RawMessage `json:"data"`
}
type Replay struct {
	Schema string            `json:"schema"`
	Events []json.RawMessage `json:"events"`
}
type Batch struct {
	Schema         string         `json:"schema"`
	BatchID        string         `json:"batchId"`
	SessionID      string         `json:"sessionId"`
	Classification Classification `json:"classification"`
	StartedAt      int64          `json:"startedAt"`
	Events         []Event        `json:"events"`
	Replay         *Replay        `json:"replay,omitempty"`
}
type Saved struct {
	Batch           Batch     `json:"batch"`
	Digest          string    `json:"sha256"`
	ReceivedAt      time.Time `json:"receivedAt"`
	ReplayExpiresAt time.Time `json:"replayExpiresAt"`
}

func Decode(reader io.Reader) (Batch, string, error) {
	var b Batch
	d := json.NewDecoder(reader)
	d.DisallowUnknownFields()
	if err := d.Decode(&b); err != nil {
		return b, "", fmt.Errorf("invalid shopping batch fields")
	}
	if d.Decode(new(any)) != io.EOF {
		return b, "", fmt.Errorf("expected one shopping batch")
	}
	if b.Schema != "fieldwork.shopping-batch.v1" || !uuid.MatchString(b.BatchID) || !uuid.MatchString(b.SessionID) || !oneOf(b.Classification.Environment, "production", "development", "test") || !oneOf(b.Classification.Traffic, "real", "synthetic") || b.StartedAt < 1577836800000 || b.StartedAt > time.Now().Add(5*time.Minute).UnixMilli() || b.Events == nil || len(b.Events) > 100 || len(b.Events) == 0 && b.Replay == nil {
		return b, "", fmt.Errorf("bounded versioned classified batch required")
	}
	b.BatchID = strings.ToLower(b.BatchID)
	b.SessionID = strings.ToLower(b.SessionID)
	seen := map[string]bool{}
	last := -1
	for i := range b.Events {
		e := &b.Events[i]
		e.EventID = strings.ToLower(e.EventID)
		if !uuid.MatchString(e.EventID) || seen[e.EventID] || e.Sequence == nil || *e.Sequence <= last || *e.Sequence > 1000000 || e.ElapsedMS == nil || *e.ElapsedMS < 0 || *e.ElapsedMS > MaxSessionMS || e.At < b.StartedAt || e.At > b.StartedAt+MaxSessionMS || abs(e.At-b.StartedAt-*e.ElapsedMS) > 5000 {
			return b, "", fmt.Errorf("invalid event identity, sequence or timing")
		}
		seen[e.EventID] = true
		last = *e.Sequence
		if !oneOf(e.Context.Mode, "baseline", "local-model") || !oneOf(e.Context.Order, "popular", "budget", "history") || e.Context.Shipping == nil || e.Context.ModelVersion != nil && !versionID.MatchString(*e.Context.ModelVersion) {
			return b, "", fmt.Errorf("invalid structured shopping context")
		}
		if err := validateData(e.Action, e.Data); err != nil {
			return b, "", err
		}
		var normalized any
		_ = json.Unmarshal(e.Data, &normalized)
		e.Data, _ = json.Marshal(normalized)
	}
	if b.Replay != nil {
		if err := sanitizeReplay(b.Replay); err != nil {
			return b, "", err
		}
	}
	encoded, _ := json.Marshal(b)
	hash := sha256.Sum256(encoded)
	return b, hex.EncodeToString(hash[:]), nil
}
func abs(n int64) int64 {
	if n < 0 {
		return -n
	}
	return n
}
func oneOf(s string, values ...string) bool {
	for _, v := range values {
		if s == v {
			return true
		}
	}
	return false
}
func product(p, v string) bool {
	for _, variant := range catalog[p] {
		if variant == v {
			return true
		}
	}
	return false
}
func object(raw json.RawMessage, required []string, allowed ...string) (map[string]any, error) {
	var data map[string]any
	if json.Unmarshal(raw, &data) != nil || data == nil {
		return nil, fmt.Errorf("structured event data required")
	}
	for k := range data {
		if !oneOf(k, allowed...) {
			return nil, fmt.Errorf("unknown event data field")
		}
	}
	for _, k := range required {
		if _, ok := data[k]; !ok {
			return nil, fmt.Errorf("missing event data field")
		}
	}
	return data, nil
}
func number(v any, min, max float64) bool {
	n, ok := v.(float64)
	return ok && !math.IsNaN(n) && !math.IsInf(n, 0) && n >= min && n <= max
}
func integer(v any, min, max float64) bool {
	n, ok := v.(float64)
	return ok && number(v, min, max) && math.Trunc(n) == n
}
func str(v any) string { s, _ := v.(string); return s }
func validateData(action string, raw json.RawMessage) error {
	keys := map[string][]string{
		"session-start": {"consent", "replay"}, "catalog-view": {"viewId", "products"}, "product-view": {"viewId", "productId", "variantId", "position", "visibility", "visibleMs"},
		"search": {"source"}, "selection": {"viewId", "productId", "variantId"}, "cart-add": {"viewId", "productId", "variantId", "count"}, "cart-clear": {"count"}, "checkout-success": {"count", "demo"},
		"model-readiness": {"stage", "durationMs", "scope", "copySource"}, "session-end": {"reason"},
	}
	allowed, ok := keys[action]
	if !ok {
		return fmt.Errorf("unsupported shopping action")
	}
	d, err := object(raw, allowed, allowed...)
	if err != nil {
		return err
	}
	valid := true
	switch action {
	case "session-start":
		_, ok := d["replay"].(bool)
		valid = ok && oneOf(str(d["consent"]), "notice", "granted")
	case "catalog-view":
		ps, ok := d["products"].([]any)
		valid = ok && len(ps) <= 16
		positions, products := map[float64]bool{}, map[string]bool{}
		for _, p := range ps {
			bytes, _ := json.Marshal(p)
			item, e := object(bytes, []string{"productId", "variantId", "position"}, "productId", "variantId", "position")
			if e != nil {
				return e
			}
			position, _ := item["position"].(float64)
			id := str(item["productId"])
			valid = valid && integer(item["position"], 0, 15) && !positions[position] && !products[id] && product(id, str(item["variantId"]))
			positions[position] = true
			products[id] = true
		}
	case "product-view":
		valid = integer(d["position"], 0, 15) && d["visibility"] == "intersection-at-least-half" && d["visibleMs"] == float64(250)
	case "search":
		valid = oneOf(str(d["source"]), "submit", "example", "clear")
	case "cart-add":
		valid = integer(d["count"], 1, 1000)
	case "cart-clear":
		valid = d["count"] == float64(0)
	case "checkout-success":
		valid = integer(d["count"], 1, 1000) && d["demo"] == true
	case "model-readiness":
		valid = oneOf(str(d["stage"]), "rank", "firstThree", "complete") && number(d["durationMs"], 0, MaxSessionMS) && d["scope"] == "after-paint-opportunity" && (d["copySource"] == nil || oneOf(str(d["copySource"]), "fresh-native", "fresh-parallel-head", "generated-earlier", "mixed-native-cache", "no-generation"))
	case "session-end":
		valid = oneOf(str(d["reason"]), "pagehide", "duration-limit")
	}
	if _, ok := d["viewId"]; ok {
		valid = valid && uuid.MatchString(str(d["viewId"]))
	}
	if _, ok := d["productId"]; ok {
		valid = valid && product(str(d["productId"]), str(d["variantId"]))
	}
	if !valid {
		return fmt.Errorf("invalid allowlisted shopping data")
	}
	return nil
}
