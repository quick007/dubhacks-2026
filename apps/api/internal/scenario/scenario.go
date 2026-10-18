// Package scenario holds the scripted mock shopping scenario: catalog facts plus one
// recorded path per persona and variant. Real agent runs will replace the paths.
// scripts/capture-fieldwork.mjs records the paths and screenshots on the real demo shop.
package scenario

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path"
	"slices"
	"storefront/api/internal/contracts"
)

type Product struct {
	ID         string              `json:"id"`
	Name       string              `json:"name"`
	Price      float64             `json:"price"`
	Attributes map[string][]string `json:"attributes"`
}
type Step struct {
	Page     contracts.Page    `json:"page"`
	Message  string            `json:"message"`
	Click    *contracts.Point  `json:"click,omitempty"`
	Search   *contracts.Search `json:"search,omitempty"`
	Products []string          `json:"products,omitempty"`
	// Screenshot names a captured frame of the real shop; without one the mock draws a stand-in.
	Screenshot string `json:"screenshot,omitempty"`
}

// Path is what one mock shopper did. Purchase names the product bought; Reason explains giving up.
type Path struct {
	Steps    []Step `json:"steps"`
	Purchase string `json:"purchase,omitempty"`
	Reason   string `json:"reason,omitempty"`
}
type Scenario struct {
	ID           string                     `json:"id"`
	Products     []Product                  `json:"products"`
	OptimalSteps map[string]int             `json:"optimalSteps"`
	Paths        map[string]map[string]Path `json:"paths"`
}

var pageKinds = []string{"home", "category", "search", "product", "cart", "checkout", "confirmation", "other"}

func Load(data []byte) (*Scenario, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var s Scenario
	if err := decoder.Decode(&s); err != nil {
		return nil, fmt.Errorf("invalid scenario: %w", err)
	}
	for persona, variants := range s.Paths {
		for variant, recorded := range variants {
			if (recorded.Purchase == "") == (recorded.Reason == "") {
				return nil, fmt.Errorf("scenario path %s/%s: set exactly one of purchase or reason", persona, variant)
			}
			ids := []string{}
			if recorded.Purchase != "" {
				ids = append(ids, recorded.Purchase)
			}
			for _, step := range recorded.Steps {
				if !slices.Contains(pageKinds, step.Page.Kind) {
					return nil, fmt.Errorf("scenario path %s/%s: unknown page kind %q", persona, variant, step.Page.Kind)
				}
				if step.Screenshot != "" && (path.Base(step.Screenshot) != step.Screenshot || path.Ext(step.Screenshot) != ".jpg") {
					return nil, fmt.Errorf("scenario path %s/%s: screenshot %q is not a .jpg file name", persona, variant, step.Screenshot)
				}
				ids = append(ids, step.Products...)
			}
			for _, id := range ids {
				if _, ok := s.Product(id); !ok {
					return nil, fmt.Errorf("scenario path %s/%s: unknown product %q", persona, variant, id)
				}
			}
		}
	}
	return &s, nil
}

func (s *Scenario) Product(id string) (Product, bool) {
	for _, product := range s.Products {
		if product.ID == id {
			return product, true
		}
	}
	return Product{}, false
}

func (s *Scenario) Path(personaID, variantID string) (Path, bool) {
	if s == nil {
		return Path{}, false
	}
	path, ok := s.Paths[personaID][variantID]
	return path, ok
}
