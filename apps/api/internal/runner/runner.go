package runner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"storefront/api/internal/contracts"
	"time"
)

type Result struct {
	TaskCompleted  bool     `json:"taskCompleted"`
	FailureReasons []string `json:"failureReasons"`
	Notes          []string `json:"notes"`
}
type Event struct {
	Kind    string `json:"kind"`
	Message string `json:"message"`
}
type Artifact struct {
	Kind     string `json:"kind"`
	FileName string `json:"fileName"`
	MimeType string `json:"mimeType"`
}
type Output struct {
	Events    []Event    `json:"events"`
	Artifacts []Artifact `json:"artifacts"`
	Result    Result     `json:"result"`
	Error     string     `json:"error,omitempty"`
}
type Context struct {
	Run               contracts.Run
	ArtifactDirectory string
	Emit              func(Event) error
	SaveArtifact      func(Artifact) error
}
type Adapter interface {
	Execute(context.Context, Context) (Result, error)
}

type Mock struct{ StepDelay time.Duration }

func (m Mock) Execute(ctx context.Context, c Context) (Result, error) {
	missing := slices.Contains(c.Run.Persona.Traits, "shipping-sensitive") && !c.Run.Config.ShowShipping
	observation := "Variant configuration hides shipping information"
	if c.Run.Config.ShowShipping {
		observation = "Variant configuration displays shipping information"
	}
	last := Event{"action", "Synthetic persona completed the fake checkout task"}
	if missing {
		last = Event{"failure", "Synthetic shipping-sensitive persona stopped because delivery information was hidden"}
	}
	for _, event := range []Event{{"navigation", "Synthetic agent opened the local demo storefront"}, {"observation", observation}, {"action", "Synthetic persona inspected recommendations ordered by " + c.Run.Config.RecommendationOrder}, last} {
		if err := c.Emit(event); err != nil {
			return Result{}, err
		}
		select {
		case <-ctx.Done():
			return Result{}, ctx.Err()
		case <-time.After(m.StepDelay):
		}
	}
	failures := []string{}
	if missing {
		failures = append(failures, "Shipping information hidden for a shipping-sensitive synthetic persona")
	}
	result := Result{!missing, failures, []string{"Deterministic mock rules; this is not measured human conversion.", fmt.Sprintf("Recorded seed: %d", c.Run.Seed)}}
	body, err := json.MarshalIndent(struct {
		Synthetic bool   `json:"synthetic"`
		RunID     string `json:"runId"`
		Result
	}{true, c.Run.ID, result}, "", "  ")
	if err != nil {
		return Result{}, err
	}
	if err = os.WriteFile(filepath.Join(c.ArtifactDirectory, "result.json"), body, 0600); err != nil {
		return Result{}, err
	}
	return result, c.SaveArtifact(Artifact{"result", "result.json", "application/json"})
}

// Browser delegates only the local scripted task to a small TypeScript process.
type Browser struct {
	WorkspaceRoot string
	NodeBinary    string
}

func (b Browser) Execute(ctx context.Context, c Context) (Result, error) {
	node := b.NodeBinary
	if node == "" {
		node = "node"
	}
	input, err := json.Marshal(struct {
		Run               contracts.Run `json:"run"`
		ArtifactDirectory string        `json:"artifactDirectory"`
	}{c.Run, c.ArtifactDirectory})
	if err != nil {
		return Result{}, err
	}
	cmd := exec.CommandContext(ctx, node, "--import", filepath.Join(b.WorkspaceRoot, "node_modules/tsx/dist/loader.mjs"), filepath.Join(b.WorkspaceRoot, "apps/runner/src/cli.ts"))
	cmd.Dir = b.WorkspaceRoot
	cmd.Stdin = bytes.NewReader(input)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	processErr := cmd.Run()
	var output Output
	decoder := json.NewDecoder(&stdout)
	decoder.DisallowUnknownFields()
	if err = decoder.Decode(&output); err != nil {
		return Result{}, fmt.Errorf("invalid browser runner JSON: %w; process: %v; %.2000s", err, processErr, stderr.String())
	}
	for _, event := range output.Events {
		if !slices.Contains([]string{"navigation", "action", "observation", "failure", "status"}, event.Kind) {
			return Result{}, fmt.Errorf("invalid browser event kind")
		}
		if err = c.Emit(event); err != nil {
			return Result{}, err
		}
	}
	for _, artifact := range output.Artifacts {
		allowed := map[string]string{"trace.zip": "trace", "screenshot.png": "screenshot", "result.json": "result"}
		if allowed[artifact.FileName] != artifact.Kind || artifact.FileName == "" {
			return Result{}, fmt.Errorf("invalid browser artifact")
		}
		if err = c.SaveArtifact(artifact); err != nil {
			return Result{}, err
		}
	}
	if processErr != nil || output.Error != "" {
		return Result{}, fmt.Errorf("local browser runner failed: %v: %s; %.2000s", processErr, output.Error, stderr.String())
	}
	if output.Result.FailureReasons == nil {
		output.Result.FailureReasons = []string{}
	}
	if output.Result.Notes == nil {
		output.Result.Notes = []string{}
	}
	return output.Result, nil
}
