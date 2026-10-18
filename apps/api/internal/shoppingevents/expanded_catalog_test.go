package shoppingevents

import (
	"encoding/json"
	"testing"
)

func TestExpandedCatalogViewBoundsAndNewSKUValidation(t *testing.T) {
	if len(catalog) != 35 {
		t.Fatalf("expected 35 public products, got %d", len(catalog))
	}
	products := []map[string]any{}
	for id, variants := range catalog {
		products = append(products, map[string]any{"productId": id, "variantId": variants[0], "position": len(products)})
	}
	encode := func(value any) json.RawMessage { raw, _ := json.Marshal(value); return raw }
	if err := validateData("catalog-view", encode(map[string]any{"viewId": "00000000-0000-4000-8000-000000000001", "products": products})); err != nil {
		t.Fatal("35-product view must be accepted", err)
	}
	if err := validateData("product-view", encode(map[string]any{"viewId": "00000000-0000-4000-8000-000000000001", "productId": "work-tote", "variantId": catalog["work-tote"][0], "position": 34, "visibility": "intersection-at-least-half", "visibleMs": 250})); err != nil {
		t.Fatal("last legal expanded position must be accepted", err)
	}
	if err := validateData("product-view", encode(map[string]any{"viewId": "00000000-0000-4000-8000-000000000001", "productId": "work-tote", "variantId": catalog["work-tote"][0], "position": 35, "visibility": "intersection-at-least-half", "visibleMs": 250})); err == nil {
		t.Fatal("out-of-range product position must be rejected")
	}
	products[0]["position"] = 35
	if err := validateData("catalog-view", encode(map[string]any{"viewId": "00000000-0000-4000-8000-000000000001", "products": products})); err == nil {
		t.Fatal("position beyond public catalog must be rejected")
	}
	for _, id := range []string{"rain-anorak", "commuter-raincoat", "work-briefcase", "work-tote", "commuter-pack"} {
		if !product(id, catalog[id][0]) || product(id, "mug-sage") {
			t.Fatal("variant binding validation broken", id)
		}
	}
}
