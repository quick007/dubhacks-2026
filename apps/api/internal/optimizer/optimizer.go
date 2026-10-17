package optimizer

import (
	"storefront/api/internal/contracts"
	"strings"
)

type Adapter interface {
	Propose(contracts.Run) []contracts.ProposedChange
}
type Factual struct{}

func (Factual) Propose(run contracts.Run) []contracts.ProposedChange {
	proposals := []contracts.ProposedChange{}
	if run.PersonaSplit != "development" || run.Status != "completed" || run.Config.ShowShipping || run.Evaluation == nil {
		return proposals
	}
	missing := false
	for _, failure := range run.Evaluation.FailureReasons {
		if strings.Contains(failure, "Shipping information hidden") {
			missing = true
		}
	}
	if !missing {
		return proposals
	}
	evidence := []string{}
	for _, event := range run.Events {
		if event.Kind == "observation" || event.Kind == "failure" {
			evidence = append(evidence, event.ID)
		}
	}
	return append(proposals, contracts.ProposedChange{ID: run.ID + "-shipping", SourceRunID: run.ID, Kind: "page", Rationale: "The development mock run recorded hidden shipping information and its synthetic persona stopped. Try displaying existing delivery facts; no human lift is established.", EvidenceStepIDs: evidence, ConfigPatch: map[string]any{"showShipping": true}, Status: "suggested"})
}
