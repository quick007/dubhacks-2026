package evaluator

import (
	"storefront/api/internal/contracts"
	"storefront/api/internal/runner"
)

type Adapter interface {
	Evaluate(contracts.Run, runner.Result) contracts.EvaluationResult
}
type Deterministic struct{}

func (Deterministic) Evaluate(run contracts.Run, result runner.Result) contracts.EvaluationResult {
	score := 0.0
	if result.TaskCompleted {
		score = 1
	}
	return contracts.EvaluationResult{RunID: run.ID, Synthetic: true, TaskCompleted: result.TaskCompleted, Score: score, FailureReasons: result.FailureReasons, Notes: result.Notes}
}
