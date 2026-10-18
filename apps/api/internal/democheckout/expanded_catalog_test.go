package democheckout

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// These are real handler calls against local temporary receipt storage, not cloud writes.
func TestExpandedCatalogNewFixturesHaveAuthoritativeReceipts(t *testing.T) {
	if len(catalog.Products) != 35 {
		t.Fatalf("expected35 public products, got %d", len(catalog.Products))
	}
	for _, id := range []string{"rain-anorak", "commuter-raincoat", "work-briefcase", "work-tote", "commuter-pack"} {
		t.Run(id, func(t *testing.T) {
			p := catalog.Products[id]
			if len(p.Variants) != 3 {
				t.Fatal("expected three color variants", id)
			}
			s, err := New(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			mux := http.NewServeMux()
			s.Register(mux)
			shipping := false
			raw, _ := json.Marshal(Request{Schema: "fieldwork.demo-checkout.v1", CheckoutID: fixtureID, Items: []Item{{ProductID: id, VariantID: p.Variants[0], Quantity: 1}}, ShippingVisible: &shipping})
			req := httptest.NewRequest("POST", "/api/demo-checkout", bytes.NewReader(raw))
			req.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, req)
			if response.Code != 202 {
				t.Fatal(response.Code, response.Body.String())
			}
			var got struct {
				Receipt Receipt `json:"receipt"`
			}
			if json.Unmarshal(response.Body.Bytes(), &got) != nil {
				t.Fatal("invalid handler receipt")
			}
			if len(got.Receipt.Items) != 1 || got.Receipt.Subtotal != float64(p.UnitCents)/100 || got.Receipt.RealPurchase {
				t.Fatal("wrong authoritative public facts", got.Receipt)
			}
			saved, err := s.read(fixtureID)
			if err != nil || saved.Receipt.ReceiptID != got.Receipt.ReceiptID {
				t.Fatal("durable local receipt mismatch", err)
			}
			get := httptest.NewRecorder()
			mux.ServeHTTP(get, httptest.NewRequest("GET", "/api/demo-checkout/"+fixtureID, nil))
			if get.Code != 200 {
				t.Fatal(get.Code, get.Body.String())
			}
		})
	}
}
