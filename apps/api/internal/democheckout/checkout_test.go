package democheckout

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

const fixtureID = "00000000-0000-4000-8000-000000000001"

func input(items string, shipping bool) []byte {
	value := "false"
	if shipping {
		value = "true"
	}
	return []byte(`{"schema":"fieldwork.demo-checkout.v1","checkoutId":"` + fixtureID + `","items":` + items + `,"shippingVisible":` + value + `}`)
}
func TestAuthoritativeReceiptRetryReadbackAndRestart(t *testing.T) {
	dir := t.TempDir()
	s, err := New(dir)
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	s.Register(mux)
	post := func(raw []byte) *httptest.ResponseRecorder {
		r := httptest.NewRequest("POST", "/api/demo-checkout", bytes.NewReader(raw))
		r.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, r)
		return w
	}
	raw := input(`[{"productId":"mug","variantId":"mug-cobalt","quantity":1}]`, true)
	first := post(raw)
	if first.Code != 202 {
		t.Fatal(first.Code, first.Body)
	}
	var got struct {
		Receipt   Receipt `json:"receipt"`
		Duplicate bool    `json:"duplicate"`
	}
	_ = json.Unmarshal(first.Body.Bytes(), &got)
	if got.Duplicate || got.Receipt.Subtotal != 24 || got.Receipt.Total != 29 || got.Receipt.Shipping.Status != "flat" || got.Receipt.RealPurchase || len(got.Receipt.Items) != 1 || got.Receipt.Items[0].UnitPrice != 24 {
		t.Fatal(got)
	}
	again := post(raw)
	var duplicate struct {
		Receipt   Receipt `json:"receipt"`
		Duplicate bool    `json:"duplicate"`
	}
	_ = json.Unmarshal(again.Body.Bytes(), &duplicate)
	if again.Code != 202 || !duplicate.Duplicate || !reflect.DeepEqual(got.Receipt, duplicate.Receipt) {
		t.Fatal("immutable retry failed")
	}
	if changed := post(bytes.Replace(raw, []byte(`"quantity":1`), []byte(`"quantity":2`), 1)); changed.Code != 409 {
		t.Fatal("changed checkout ID accepted")
	}
	restarted, err := New(dir)
	if err != nil {
		t.Fatal(err)
	}
	r, err := restarted.read(fixtureID)
	if err != nil || !reflect.DeepEqual(r.Receipt, got.Receipt) {
		t.Fatal("restart readback", err)
	}
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("GET", "/api/demo-checkout/"+fixtureID, nil))
	if w.Code != 200 || !strings.Contains(w.Body.String(), got.Receipt.ReceiptID) {
		t.Fatal("GET failed")
	}
	info, _ := os.Stat(filepath.Join(dir, fixtureID+".json"))
	if info.Mode().Perm() != 0600 {
		t.Fatal("receipt permissions")
	}
}
func TestInvalidUnknownAndClientPriceFieldsRejected(t *testing.T) {
	for _, items := range []string{`[]`, `[{"productId":"mug","variantId":"pack-sand","quantity":1}]`, `[{"productId":"mug","variantId":"mug-sage","quantity":0}]`, `[{"productId":"mug","variantId":"mug-sage","quantity":1,"unitPrice":0}]`, `[{"productId":"unknown","variantId":"unknown","quantity":1}]`, `[{"productId":"mug","variantId":"mug-sage","quantity":1},{"productId":"mug","variantId":"mug-sage","quantity":1}]`} {
		if _, _, err := decode(bytes.NewReader(input(items, false))); err == nil {
			t.Fatal("invalid request accepted", items)
		}
	}
}
func TestShippingHiddenFreeAndExactThreshold(t *testing.T) {
	for _, scenario := range []struct {
		items   string
		visible bool
		total   float64
		status  string
	}{{`[{"productId":"mug","variantId":"mug-sage","quantity":1}]`, false, 24, "hidden"}, {`[{"productId":"pack","variantId":"pack-sand","quantity":1}]`, true, 68, "free"}} {
		s, _ := New(t.TempDir())
		r, digest, err := decode(bytes.NewReader(input(scenario.items, scenario.visible)))
		if err != nil {
			t.Fatal(err)
		}
		receipt, _, err := s.accept(r, digest)
		if err != nil || receipt.Total != scenario.total || receipt.Shipping.Status != scenario.status {
			t.Fatal(receipt, err)
		}
	}
	found := false
	for first, a := range catalog.Products {
		for second, b := range catalog.Products {
			if first == second || a.UnitCents+b.UnitCents != 5000 {
				continue
			}
			items, _ := json.Marshal([]Item{{first, a.Variants[0], 1}, {second, b.Variants[0], 1}})
			r, digest, err := decode(bytes.NewReader(input(string(items), true)))
			if err != nil {
				t.Fatal(err)
			}
			s, _ := New(t.TempDir())
			receipt, _, err := s.accept(r, digest)
			if err != nil || receipt.Total != 55 || receipt.Shipping.Status != "flat" {
				t.Fatal("threshold must be strict >50", receipt, err)
			}
			found = true
			break
		}
		if found {
			break
		}
	}
	if !found {
		t.Fatal("public catalog no longer covers exact shipping threshold fixture")
	}
}
func TestFrozenPublicCatalogSourceParity(t *testing.T) {
	for name, expected := range catalog.Sources {
		raw, err := os.ReadFile(filepath.Join("../../../..", name))
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.Sum256(raw)
		if hex.EncodeToString(h[:]) != expected {
			t.Fatal("refresh demo checkout public catalog after source change", name)
		}
	}
}

func TestConcurrentIndependentServicesNeverReplaceReceipt(t *testing.T) {
	dir := t.TempDir()
	a, _ := New(dir)
	b, _ := New(dir)
	request, digest, err := decode(bytes.NewReader(input(`[{"productId":"mug","variantId":"mug-sage","quantity":1}]`, true)))
	if err != nil {
		t.Fatal(err)
	}
	var group sync.WaitGroup
	receipts := make(chan Receipt, 2)
	errors := make(chan error, 2)
	for _, service := range []*Service{a, b} {
		group.Add(1)
		go func(service *Service) {
			defer group.Done()
			receipt, _, err := service.accept(request, digest)
			receipts <- receipt
			errors <- err
		}(service)
	}
	group.Wait()
	first, second := <-receipts, <-receipts
	if err := <-errors; err != nil {
		t.Fatal(err)
	}
	if err := <-errors; err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("independent writers replaced immutable receipt")
	}
}
