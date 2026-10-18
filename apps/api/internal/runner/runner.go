package runner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"storefront/api/internal/contracts"
	"storefront/api/internal/scenario"
	"time"
)

type Result struct {
	TaskCompleted      bool     `json:"taskCompleted"`
	FailureReasons     []string `json:"failureReasons"`
	Notes              []string `json:"notes"`
	PurchasedProductID string   `json:"purchasedProductId,omitempty"`
}
type Event struct {
	Kind           string            `json:"kind"`
	Message        string            `json:"message"`
	Page           *contracts.Page   `json:"page,omitempty"`
	Click          *contracts.Point  `json:"click,omitempty"`
	Search         *contracts.Search `json:"search,omitempty"`
	ScreenshotFile string            `json:"screenshotFile,omitempty"`
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

// Mock replays the scripted scenario path for the run's persona and variant, saving one frame per step:
// the step's captured screenshot from Frames when it names one, otherwise a drawn stand-in.
type Mock struct {
	StepDelay time.Duration
	Scenario  *scenario.Scenario
	Frames    fs.FS
}

func (m Mock) Execute(ctx context.Context, c Context) (Result, error) {
	path, ok := m.Scenario.Path(c.Run.PersonaID, c.Run.VariantID)
	if !ok {
		return Result{}, fmt.Errorf("no mock scenario path for persona %s on variant %s", c.Run.PersonaID, c.Run.VariantID)
	}
	for i, step := range path.Steps {
		file, mimeType, frame := fmt.Sprintf("step-%02d.svg", i), "image/svg+xml", m.Scenario.Frame(step, c.Run.Config.ShowAttributes)
		if step.Screenshot != "" {
			if m.Frames == nil {
				return Result{}, fmt.Errorf("no captured frames for screenshot %s", step.Screenshot)
			}
			captured, err := fs.ReadFile(m.Frames, step.Screenshot)
			if err != nil {
				return Result{}, err
			}
			file, mimeType, frame = fmt.Sprintf("step-%02d.jpg", i), "image/jpeg", captured
		}
		if err := os.WriteFile(filepath.Join(c.ArtifactDirectory, file), frame, 0600); err != nil {
			return Result{}, err
		}
		if err := c.SaveArtifact(Artifact{"screenshot", file, mimeType}); err != nil {
			return Result{}, err
		}
		kind := "navigation"
		if step.Search != nil || slices.Contains([]string{"cart", "checkout", "confirmation"}, step.Page.Kind) {
			kind = "action"
		}
		if err := c.Emit(Event{Kind: kind, Message: step.Message, Page: &step.Page, Click: step.Click, Search: step.Search, ScreenshotFile: file}); err != nil {
			return Result{}, err
		}
		if err := wait(ctx, m.StepDelay); err != nil {
			return Result{}, err
		}
	}
	result := Result{
		TaskCompleted:      path.Purchase != "",
		FailureReasons:     []string{},
		Notes:              []string{"Scripted mock shopper replayed on recorded page frames; not measured human behavior.", fmt.Sprintf("Recorded seed: %d", c.Run.Seed)},
		PurchasedProductID: path.Purchase,
	}
	if path.Reason != "" {
		result.FailureReasons = append(result.FailureReasons, path.Reason)
		if err := c.Emit(Event{Kind: "failure", Message: path.Reason}); err != nil {
			return Result{}, err
		}
	}
	return result, saveResult(c, result)
}

func wait(ctx context.Context, delay time.Duration) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(delay):
		return nil
	}
}

func saveResult(c Context, result Result) error {
	body, err := json.MarshalIndent(struct {
		Synthetic bool   `json:"synthetic"`
		RunID     string `json:"runId"`
		Result
	}{true, c.Run.ID, result}, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(c.ArtifactDirectory, "result.json"), body, 0600); err != nil {
		return err
	}
	return c.SaveArtifact(Artifact{"result", "result.json", "application/json"})
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
