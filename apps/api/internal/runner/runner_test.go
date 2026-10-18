package runner

import (
	"context"
	"os"
	"path/filepath"
	"storefront/api/internal/contracts"
	"storefront/api/internal/scenario"
	"testing"
)

func TestMockReplaysScenarioPath(t *testing.T) {
	s, err := scenario.Load([]byte(`{"id":"t","products":[{"id":"p","name":"Shell","price":100,"attributes":{}}],"optimalSteps":{"a":3},
"paths":{"a":{"v":{"steps":[
{"page":{"kind":"category","label":"Jackets"},"message":"Opened jackets","click":{"x":0.1,"y":0.4},"products":["p"]},
{"page":{"kind":"search","label":"Search: shell"},"message":"Searched","search":{"query":"shell","resultCount":0}}],
"reason":"Gave up"}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	events, artifacts := []Event{}, []Artifact{}
	c := Context{
		Run:               contracts.Run{ID: "r", PersonaID: "a", VariantID: "v"},
		ArtifactDirectory: dir,
		Emit:              func(e Event) error { events = append(events, e); return nil },
		SaveArtifact:      func(a Artifact) error { artifacts = append(artifacts, a); return nil },
	}
	result, err := Mock{Scenario: s}.Execute(context.Background(), c)
	if err != nil {
		t.Fatal(err)
	}
	if result.TaskCompleted || result.PurchasedProductID != "" || len(result.FailureReasons) != 1 {
		t.Fatalf("wrong result %+v", result)
	}
	if len(events) != 3 || events[0].Page == nil || events[0].Click == nil || events[0].ScreenshotFile != "step-00.svg" ||
		events[0].Kind != "navigation" || events[1].Kind != "action" || events[2].Kind != "failure" {
		t.Fatalf("wrong events %+v", events)
	}
	if len(artifacts) != 3 || artifacts[0].MimeType != "image/svg+xml" || artifacts[2].Kind != "result" {
		t.Fatalf("wrong artifacts %+v", artifacts)
	}
	if _, err := os.Stat(filepath.Join(dir, "step-01.svg")); err != nil {
		t.Fatal(err)
	}
	if _, err := (Mock{}).Execute(context.Background(), Context{Run: contracts.Run{PersonaID: "zzz"}}); err == nil {
		t.Fatal("ran without a scenario path")
	}
}
