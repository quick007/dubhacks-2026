package optimizer

import (
	"storefront/api/internal/contracts"
	"testing"
)

func TestProposesCardDetailsAfterRepeatedProductChecks(t *testing.T) {
	product := func(id, label string) contracts.StepEvent {
		return contracts.StepEvent{ID: id, Page: &contracts.Page{Kind: "product", Label: label}}
	}
	run := contracts.Run{ID: "r", PersonaSplit: "development", Status: "completed",
		Events:     []contracts.StepEvent{product("a", "Harbor"), product("b", "Trail")},
		Evaluation: &contracts.EvaluationResult{Findability: &contracts.Findability{Outcome: "abandoned"}}}
	proposals := (Factual{}).Propose(run)
	if len(proposals) != 1 || proposals[0].ConfigPatch["showAttributes"] != true || len(proposals[0].EvidenceStepIDs) != 2 {
		t.Fatalf("wrong proposals %+v", proposals)
	}
	run.PersonaSplit = "held-out"
	if len((Factual{}).Propose(run)) != 0 {
		t.Fatal("held-out run entered optimizer")
	}
	run.PersonaSplit, run.Events = "development", run.Events[:1]
	if len((Factual{}).Propose(run)) != 0 {
		t.Fatal("proposed without repeated product checks")
	}
}
