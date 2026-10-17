package model

import (
	"context"
	"fmt"
)

type Input struct {
	Task     string
	Evidence []string
}
type Output struct {
	Synthetic bool   `json:"synthetic"`
	Text      string `json:"text"`
}
type Adapter interface {
	Complete(context.Context, Input) (Output, error)
}
type Mock struct{}

func (Mock) Complete(_ context.Context, input Input) (Output, error) {
	return Output{true, fmt.Sprintf("Mock adapter: %s. %d evidence items supplied.", input.Task, len(input.Evidence))}, nil
}
