package evaluator

import (
	"fmt"
	"slices"
	"storefront/api/internal/contracts"
	"storefront/api/internal/runner"
	"storefront/api/internal/scenario"
	"strings"
)

type Adapter interface {
	Evaluate(contracts.Run, runner.Result) contracts.EvaluationResult
}
type Deterministic struct{ Scenario *scenario.Scenario }

func (d Deterministic) Evaluate(run contracts.Run, result runner.Result) contracts.EvaluationResult {
	evaluation := contracts.EvaluationResult{RunID: run.ID, Synthetic: true, TaskCompleted: result.TaskCompleted, FailureReasons: result.FailureReasons, Notes: result.Notes}
	recordedPath := slices.ContainsFunc(run.Events, func(event contracts.StepEvent) bool { return event.Page != nil })
	if task := run.Persona.Task; task != nil && d.Scenario != nil && recordedPath {
		findability, reasons := d.findability(run, *task, result.PurchasedProductID)
		evaluation.Findability = &findability
		evaluation.TaskCompleted = findability.Outcome == "purchased"
		evaluation.FailureReasons = append(evaluation.FailureReasons, reasons...)
	}
	if evaluation.TaskCompleted {
		evaluation.Score = 1
	}
	return evaluation
}

// findability judges the recorded path and purchase against scenario facts, never the shopper's own claim.
func (d Deterministic) findability(run contracts.Run, task contracts.ShoppingTask, purchased string) (contracts.Findability, []string) {
	f := contracts.Findability{OptimalSteps: d.Scenario.OptimalSteps[run.PersonaID], Requirements: []contracts.RequirementResult{}}
	seen := map[contracts.Page]bool{}
	for _, event := range run.Events {
		if event.Search != nil {
			f.Searches++
			if event.Search.ResultCount == 0 {
				f.EmptySearches++
			}
		}
		if event.Page == nil {
			continue
		}
		f.Steps++
		if seen[*event.Page] {
			f.Revisits++
		}
		seen[*event.Page] = true
		f.AbandonedAt = event.ID
	}
	product, bought := d.Scenario.Product(purchased)
	if bought {
		f.OrderValue = product.Price
	}
	reasons := []string{}
	for _, requirement := range task.Requirements {
		values := product.Attributes[requirement.Attribute]
		met := bought && slices.Contains(values, requirement.Value)
		f.Requirements = append(f.Requirements, contracts.RequirementResult{ID: requirement.ID, Met: met, Observed: strings.Join(values, ", ")})
		if bought && !met && requirement.Kind == "hard" {
			reasons = append(reasons, fmt.Sprintf("%s misses %s", product.Name, requirement.Label))
		}
	}
	if task.Budget > 0 {
		result := contracts.RequirementResult{ID: "budget", Met: bought && product.Price <= task.Budget}
		if bought {
			result.Observed = fmt.Sprintf("$%.0f", product.Price)
		}
		f.Requirements = append(f.Requirements, result)
		if bought && !result.Met {
			reasons = append(reasons, fmt.Sprintf("%s costs more than $%.0f", product.Name, task.Budget))
		}
	}
	switch {
	case !bought:
		f.Outcome = "abandoned"
	case len(reasons) > 0:
		f.Outcome, f.AbandonedAt = "constraint-failure", ""
	default:
		f.Outcome, f.AbandonedAt = "purchased", ""
	}
	return f, reasons
}
