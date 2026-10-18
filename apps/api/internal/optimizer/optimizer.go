package optimizer

import (
	"fmt"
	"storefront/api/internal/contracts"
)

type Adapter interface {
	Propose(contracts.Run) []contracts.ProposedChange
}
type Factual struct{}

// Propose suggests showing existing product facts on cards when a development shopper had to open
// several product pages to check them and then gave up.
func (Factual) Propose(run contracts.Run) []contracts.ProposedChange {
	proposals := []contracts.ProposedChange{}
	if run.PersonaSplit != "development" || run.Status != "completed" || run.Config.ShowAttributes ||
		run.Evaluation == nil || run.Evaluation.Findability == nil || run.Evaluation.Findability.Outcome != "abandoned" {
		return proposals
	}
	evidence := []string{}
	for _, event := range run.Events {
		if event.Page != nil && event.Page.Kind == "product" {
			evidence = append(evidence, event.ID)
		}
	}
	if len(evidence) < 2 {
		return proposals
	}
	return append(proposals, contracts.ProposedChange{
		ID:              run.ID + "-attributes",
		SourceRunID:     run.ID,
		Kind:            "page",
		Rationale:       fmt.Sprintf("The shopper opened %d product pages to check details the product cards leave out, then gave up. Show the existing size, color and weatherproofing facts on the cards.", len(evidence)),
		EvidenceStepIDs: evidence,
		ConfigPatch:     map[string]any{"showAttributes": true},
		Status:          "suggested",
	})
}
