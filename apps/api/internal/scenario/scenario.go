// Package scenario holds the scripted mock shopping scenario: catalog facts plus one
// recorded path per persona and variant. Real agent runs will replace the paths.
package scenario

import (
	"bytes"
	"encoding/json"
	"fmt"
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
		for variant, path := range variants {
			if (path.Purchase == "") == (path.Reason == "") {
				return nil, fmt.Errorf("scenario path %s/%s: set exactly one of purchase or reason", persona, variant)
			}
			ids := []string{}
			if path.Purchase != "" {
				ids = append(ids, path.Purchase)
			}
			for _, step := range path.Steps {
				if !slices.Contains(pageKinds, step.Page.Kind) {
					return nil, fmt.Errorf("scenario path %s/%s: unknown page kind %q", persona, variant, step.Page.Kind)
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
