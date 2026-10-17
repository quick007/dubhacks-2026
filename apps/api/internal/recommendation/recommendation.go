package recommendation

import (
	"slices"
	"sort"
	"storefront/api/internal/contracts"
)

type Item struct {
	ID         string
	Price      float64
	Popularity int
}
type Adapter interface {
	Rank([]Item, contracts.Persona, contracts.VariantConfig) []Item
}
type Deterministic struct{}

func (Deterministic) Rank(items []Item, persona contracts.Persona, config contracts.VariantConfig) []Item {
	ranked := slices.Clone(items)
	sort.SliceStable(ranked, func(i, j int) bool {
		if config.RecommendationOrder == "budget" {
			return ranked[i].Price < ranked[j].Price
		}
		if config.RecommendationOrder == "history" {
			a, b := slices.Contains(persona.History, ranked[i].ID), slices.Contains(persona.History, ranked[j].ID)
			if a != b {
				return a
			}
		}
		return ranked[i].Popularity > ranked[j].Popularity
	})
	return ranked
}
