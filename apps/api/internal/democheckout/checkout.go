// Package democheckout issues local-only fake-order receipts from public facts.
// No payment, real purchase, experiment score, or cloud upload occurs here.
package democheckout

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

//go:embed catalog.json
var catalogJSON []byte
var catalog = func() struct {
	Products map[string]struct {
		UnitCents int      `json:"unitCents"`
		Variants  []string `json:"variants"`
	} `json:"products"`
	FreeOver int               `json:"freeShippingOverCents"`
	Flat     int               `json:"flatShippingCents"`
	Sources  map[string]string `json:"sourceSha256"`
} {
	var c struct {
		Products map[string]struct {
			UnitCents int      `json:"unitCents"`
			Variants  []string `json:"variants"`
		} `json:"products"`
		FreeOver int               `json:"freeShippingOverCents"`
		Flat     int               `json:"flatShippingCents"`
		Sources  map[string]string `json:"sourceSha256"`
	}
	if json.Unmarshal(catalogJSON, &c) != nil {
		panic("invalid embedded demo catalog")
	}
	return c
}()
var uuid = regexp.MustCompile(`^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$`)
var conflict = errors.New("checkout ID already has different items or shipping configuration")

type Item struct {
	ProductID string `json:"productId"`
	VariantID string `json:"variantId"`
	Quantity  int    `json:"quantity"`
}
type Request struct {
	Schema          string `json:"schema"`
	CheckoutID      string `json:"checkoutId"`
	Items           []Item `json:"items"`
	ShippingVisible *bool  `json:"shippingVisible"`
}
type Line struct {
	Item
	UnitPrice float64 `json:"unitPrice"`
	LineTotal float64 `json:"lineTotal"`
}
type Shipping struct {
	Status string   `json:"status"`
	Cost   *float64 `json:"cost"`
}
type Receipt struct {
	Schema       string   `json:"schema"`
	CheckoutID   string   `json:"checkoutId"`
	ReceiptID    string   `json:"receiptId"`
	CreatedAt    string   `json:"createdAt"`
	Items        []Line   `json:"items"`
	Subtotal     float64  `json:"subtotal"`
	Shipping     Shipping `json:"shipping"`
	Total        float64  `json:"total"`
	RealPurchase bool     `json:"realPurchase"`
}
type saved struct {
	Digest  string  `json:"sha256"`
	Receipt Receipt `json:"receipt"`
}
type Service struct {
	dir string
	mu  sync.Mutex
}

func syncDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return err
	}
	defer directory.Close()
	return directory.Sync()
}

func New(dir string) (*Service, error) {
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, err
	}
	return &Service{dir: dir}, nil
}
func decode(reader io.Reader) (Request, string, error) {
	var r Request
	d := json.NewDecoder(reader)
	d.DisallowUnknownFields()
	if d.Decode(&r) != nil || d.Decode(new(any)) != io.EOF || r.Schema != "fieldwork.demo-checkout.v1" || !uuid.MatchString(r.CheckoutID) || r.ShippingVisible == nil || len(r.Items) == 0 || len(r.Items) > 48 {
		return r, "", fmt.Errorf("invalid bounded demo checkout request")
	}
	seen := map[string]bool{}
	total := 0
	for _, item := range r.Items {
		p, ok := catalog.Products[item.ProductID]
		variant := false
		for _, id := range p.Variants {
			if id == item.VariantID {
				variant = true
			}
		}
		if !ok || !variant || item.Quantity < 1 || item.Quantity > 1000 || total+item.Quantity > 1000 || seen[item.VariantID] {
			return r, "", fmt.Errorf("known product/variant, unique lines and total quantity 1..1000 required")
		}
		seen[item.VariantID] = true
		total += item.Quantity
	}
	sort.Slice(r.Items, func(i, j int) bool { return r.Items[i].VariantID < r.Items[j].VariantID })
	raw, _ := json.Marshal(r)
	digest := sha256.Sum256(raw)
	return r, hex.EncodeToString(digest[:]), nil
}
func (s *Service) read(id string) (saved, error) {
	var record saved
	if !uuid.MatchString(id) {
		return record, os.ErrNotExist
	}
	raw, err := os.ReadFile(filepath.Join(s.dir, id+".json"))
	if err != nil {
		return record, err
	}
	err = json.Unmarshal(raw, &record)
	return record, err
}
func (s *Service) accept(r Request, digest string) (Receipt, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	old, err := s.read(r.CheckoutID)
	if err == nil {
		if old.Digest != digest {
			return Receipt{}, false, conflict
		}
		return old.Receipt, true, syncDirectory(s.dir)
	}
	if !errors.Is(err, os.ErrNotExist) {
		return Receipt{}, false, err
	}
	files, err := os.ReadDir(s.dir)
	if err != nil || len(files) >= 10000 {
		return Receipt{}, false, fmt.Errorf("local demo receipt capacity unavailable")
	}
	receipt := Receipt{Schema: "fieldwork.demo-checkout-receipt.v1", CheckoutID: r.CheckoutID, ReceiptID: "demo-" + digest[:32], CreatedAt: time.Now().UTC().Format(time.RFC3339Nano), Items: []Line{}, Shipping: Shipping{Status: "hidden"}}
	subtotal := 0
	for _, item := range r.Items {
		unit := catalog.Products[item.ProductID].UnitCents
		line := unit * item.Quantity
		subtotal += line
		receipt.Items = append(receipt.Items, Line{Item: item, UnitPrice: float64(unit) / 100, LineTotal: float64(line) / 100})
	}
	cost := 0
	if *r.ShippingVisible {
		receipt.Shipping.Status = "flat"
		cost = catalog.Flat
		if subtotal > catalog.FreeOver {
			receipt.Shipping.Status = "free"
			cost = 0
		}
		value := float64(cost) / 100
		receipt.Shipping.Cost = &value
	}
	receipt.Subtotal = float64(subtotal) / 100
	receipt.Total = float64(subtotal+cost) / 100
	raw, _ := json.Marshal(saved{Digest: digest, Receipt: receipt})
	f, err := os.CreateTemp(s.dir, ".checkout-*")
	if err != nil {
		return Receipt{}, false, err
	}
	defer os.Remove(f.Name())
	_, err = f.Write(raw)
	if err == nil {
		err = f.Sync()
	}
	closed := f.Close()
	if err == nil {
		err = closed
	}
	duplicate := false
	if err == nil {
		// A hard-link publishes exclusively: another process can never replace
		// the receipt under an existing checkout ID.
		err = os.Link(f.Name(), filepath.Join(s.dir, r.CheckoutID+".json"))
		if errors.Is(err, os.ErrExist) {
			old, readErr := s.read(r.CheckoutID)
			if readErr != nil {
				return Receipt{}, false, readErr
			}
			if old.Digest != digest {
				return Receipt{}, false, conflict
			}
			receipt = old.Receipt
			duplicate = true
			err = nil
		}
	}
	if err != nil {
		return Receipt{}, false, err
	}
	if err = syncDirectory(s.dir); err != nil {
		return Receipt{}, false, err
	}
	return receipt, duplicate, nil
}
func reply(w http.ResponseWriter, code int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(value)
}
func (s *Service) Register(mux *http.ServeMux) {
	mux.HandleFunc("POST /api/demo-checkout", func(w http.ResponseWriter, r *http.Request) {
		// Same-origin proxy clients send JSON. No cross-origin CORS is enabled.
		media, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if media != "application/json" || r.Header.Get("Sec-Fetch-Site") == "cross-site" {
			reply(w, 403, map[string]string{"error": "same-origin JSON demo checkout required"})
			return
		}
		input, digest, err := decode(http.MaxBytesReader(w, r.Body, 16<<10))
		if err != nil {
			reply(w, 400, map[string]string{"error": err.Error()})
			return
		}
		receipt, duplicate, err := s.accept(input, digest)
		if err != nil {
			code := 503
			message := "local demo receipt unavailable"
			if errors.Is(err, conflict) {
				code = 409
				message = err.Error()
			}
			reply(w, code, map[string]string{"error": message})
			return
		}
		reply(w, 202, map[string]any{"receipt": receipt, "duplicate": duplicate})
	})
	mux.HandleFunc("GET /api/demo-checkout/{id}", func(w http.ResponseWriter, r *http.Request) {
		record, err := s.read(strings.ToLower(r.PathValue("id")))
		if err != nil {
			code := 503
			if errors.Is(err, os.ErrNotExist) {
				code = 404
			}
			reply(w, code, map[string]string{"error": "local demo receipt unavailable"})
			return
		}
		reply(w, 200, map[string]any{"receipt": record.Receipt})
	})
}
