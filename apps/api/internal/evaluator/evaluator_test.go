package evaluator

import (
	"storefront/api/internal/contracts"
	"storefront/api/internal/runner"
	"storefront/api/internal/scenario"
	"testing"
)

func TestFindability(t *testing.T) {
	s := &scenario.Scenario{
		Products: []scenario.Product{
			{ID: "ridge", Name: "Ridgeline shell", Price: 138, Attributes: map[string][]string{"color": {"navy"}, "size": {"M"}}},
			{ID: "harbor", Name: "Harbor rain jacket", Price: 120, Attributes: map[string][]string{"color": {"navy"}, "size": {"S"}}},
		},
		OptimalSteps: map[string]int{"p": 2},
	}
	task := &contracts.ShoppingTask{Budget: 150, Requirements: []contracts.Requirement{
		{ID: "color", Label: "Navy", Kind: "hard", Attribute: "color", Value: "navy"},
		{ID: "size", Label: "Size M", Kind: "hard", Attribute: "size", Value: "M"},
	}}
	jackets := &contracts.Page{Kind: "category", Label: "Jackets"}
	run := contracts.Run{ID: "r", PersonaID: "p", Persona: contracts.Persona{Task: task}, Events: []contracts.StepEvent{
		{ID: "1", Page: jackets},
		{ID: "2", Page: &contracts.Page{Kind: "search", Label: "Search: navy"}, Search: &contracts.Search{Query: "navy", ResultCount: 0}},
		{ID: "3", Page: jackets},
		{ID: "4", Kind: "failure"},
	}}
	evaluate := func(product string) contracts.EvaluationResult {
		return Deterministic{Scenario: s}.Evaluate(run, runner.Result{TaskCompleted: true, PurchasedProductID: product, FailureReasons: []string{}})
	}

	bought := evaluate("ridge")
	if f := bought.Findability; f == nil || f.Outcome != "purchased" || !bought.TaskCompleted || bought.Score != 1 || f.AbandonedAt != "" || f.OrderValue != 138 {
		t.Fatalf("expected purchase: %+v", bought)
	}
	wrong := evaluate("harbor")
	if wrong.Findability.Outcome != "constraint-failure" || wrong.TaskCompleted || len(wrong.FailureReasons) != 1 {
		t.Fatalf("expected constraint failure: %+v", wrong)
	}
	gave := evaluate("")
	f := gave.Findability
	if f.Outcome != "abandoned" || f.Steps != 3 || f.OptimalSteps != 2 || f.Searches != 1 || f.EmptySearches != 1 || f.Revisits != 1 || f.AbandonedAt != "3" || len(f.Requirements) != 3 || f.OrderValue != 0 {
		t.Fatalf("wrong abandon findability: %+v", f)
	}

	// Runs without recorded page steps (today's scripted browser check) keep their own completion.
	browser := Deterministic{Scenario: s}.Evaluate(contracts.Run{Persona: contracts.Persona{Task: task}}, runner.Result{TaskCompleted: true})
	if browser.Findability != nil || !browser.TaskCompleted {
		t.Fatalf("browser run was judged by the scenario: %+v", browser)
	}
}
