# Case File Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the results-grid dashboard with a scrolling "case file" story that shows simulated shoppers struggling to find a waterproof navy jacket, the factual fix, and a held-out before/after result, all backed by new findability fields in the shared contract and a deterministic mock scenario.

**Architecture:** Additive optional fields in the Zod contract and its Go mirror carry the shopper's task, per-step page/click/search/screenshot data, and an evaluator-computed findability result. The Go mock runner replays a scripted scenario file, drawing an SVG frame per step; the evaluator judges each purchase against scenario product facts; the optimizer proposes showing existing product facts on cards. The web app derives story data with pure functions in `lib/story.ts` and renders one section component per story beat.

**Tech Stack:** Go 1.23 (`net/http`, SQLite), TypeScript, Zod 4, React 19, Vite+, Tailwind v4, Base UI, Lucide, `@fontsource-variable` fonts, Node test runner, Playwright.

**Spec:** `docs/superpowers/specs/2026-10-17-case-file-demo-design.md`

## Global Constraints

- `CONTRACT_VERSION` stays 1; every new contract field is optional; Zod (`packages/contracts/src/index.ts`) and Go (`apps/api/internal/contracts/types.go`) change together.
- One React component per file (a tangential helper under ~20 lines may share a file).
- Base UI for interactive primitives, Lucide for icons; no text glyph icons (no `↗`, `✗`, `^`).
- No all-caps text, no eyebrow labels; sentence case everywhere.
- No card layers. The only framed objects are screenshot frames.
- Colors: fog `#ECEFEA` background, ink `#17241D` text, graphite `#8B938D` baseline, cobalt `#2448D6` candidate, signal `#E2462A` give-up marks only, paper `#FFFFFF` frames; supporting muted text `#5E6A63`, rules `#D5DAD3`.
- Fonts: Big Shoulders Display (display), Atkinson Hyperlegible Next (text), self-hosted via `@fontsource-variable`.
- Results are labeled synthetic. The headline result uses held-out shoppers only. Regressions are named. Technical failures never count as a shopper giving up. Findability is judged from scenario facts, never the shopper's claim.
- `prefers-reduced-motion` is respected live (no autoplay, no typing).
- Works at 390 px wide with 16 px gutters and no horizontal page scroll.
- TypeScript lib is ES2022: no `findLast`, `toSorted`.
- Clean, YAGNI, readable, correct, performant. Delete code the new page no longer uses.
- No SSR. Do not change storefront styling.
- Do not commit unless the user asks; each task ends with a checks step instead.

## Review Focus

- Runs saved before this change (old persona ids, no `task`, no `findability`) sit in the same database: the page must ignore them, not crash. Pinned in Task 7 (`buildPairs` and `ledger` tests).
- A rerun is in progress (newest run per cell is `queued`): the story must keep showing the previous finished results. Pinned in Task 7 (`buildPairs` prefers the latest finished run).
- A run fails for technical reasons (`status: "failed"`): it must not count as giving up and must be reported separately. Pinned in Task 7 (`verdict` test).
- A step has no screenshot, or its artifact id is missing: the replay shows a placeholder instead of a broken image. Pinned in Task 7 (`frames` test).
- A product name contains `&` or `<`: the generated SVG must stay valid and escaped. Pinned in Task 3.

---

### Task 1: Findability fields in the Zod contract

**Files:**

- Modify: `packages/contracts/src/index.ts`
- Test: `packages/contracts/src/index.test.ts`

**Interfaces:**

- Produces: `ShoppingTaskSchema`, `ShoppingTask`, `PageKindSchema`, `PageKind`, `FindabilitySchema`, `Findability`; `Persona.task?`, `VariantConfig.showAttributes?`, `StepEvent.page?/click?/screenshotArtifactId?/search?`, `EvaluationResult.findability?`.

- [ ] **Step 1: Write the failing test** (append to `index.test.ts`, add `RunSchema, StepEventSchema` to the import)

```ts
void test("findability fields round-trip and older runs still parse", () => {
  const at = new Date().toISOString();
  const step = {
    id: "s",
    runId: "r",
    sequence: 0,
    at,
    kind: "action",
    message: "Searched for navy jacket",
    page: { kind: "search", label: "Search: navy jacket" },
    click: { x: 0.5, y: 0.2 },
    screenshotArtifactId: "a",
    search: { query: "navy jacket", resultCount: 0 },
  };
  const run = {
    id: "r",
    experimentId: "e",
    baselineVariantId: "b",
    variantId: "b",
    personaId: "p",
    personaSplit: "development",
    seed: 1,
    config: {
      headline: "h",
      showShipping: false,
      recommendationOrder: "popular",
      showAttributes: true,
    },
    persona: {
      id: "p",
      name: "P",
      goal: "g",
      split: "development",
      traits: [],
      history: [],
      task: {
        query: "navy jacket",
        budget: 150,
        requirements: [
          { id: "color", label: "Navy", kind: "hard", attribute: "color", value: "navy" },
        ],
      },
    },
    adapter: "mock",
    synthetic: true,
    contractVersion: 1,
    status: "completed",
    createdAt: at,
    events: [step],
    artifacts: [],
    evaluation: {
      runId: "r",
      synthetic: true,
      taskCompleted: false,
      score: 0,
      failureReasons: [],
      notes: [],
      findability: {
        outcome: "abandoned",
        requirements: [{ id: "color", met: false }],
        steps: 1,
        optimalSteps: 6,
        searches: 1,
        emptySearches: 1,
        revisits: 0,
        abandonedAt: "s",
      },
    },
    proposedChanges: [],
  };
  assert.deepEqual(RunSchema.parse(run), run);
  const { task: _task, ...oldPersona } = run.persona;
  const { findability: _findability, ...oldEvaluation } = run.evaluation;
  assert.equal(
    RunSchema.safeParse({
      ...run,
      persona: oldPersona,
      evaluation: oldEvaluation,
      events: [{ id: "s", runId: "r", sequence: 0, at, kind: "status", message: "running" }],
    }).success,
    true,
  );
  assert.equal(StepEventSchema.safeParse({ ...step, click: { x: 2, y: 0 } }).success, false);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --filter @storefront/contracts test`
Expected: FAIL (unknown keys `task`, `page`, `findability` rejected by `.strict()`).

- [ ] **Step 3: Implement**

In `index.ts`, add after `const id = ...`:

```ts
export const RequirementSchema = z
  .object({
    id,
    label: z.string(),
    kind: z.enum(["hard", "preference"]),
    attribute: z.string(),
    value: z.string(),
  })
  .strict();
export const ShoppingTaskSchema = z
  .object({
    query: z.string(),
    requirements: z.array(RequirementSchema),
    budget: z.number().positive().optional(),
  })
  .strict();
export type ShoppingTask = z.infer<typeof ShoppingTaskSchema>;
```

Add `task: ShoppingTaskSchema.optional(),` to `PersonaSchema` and `showAttributes: z.boolean().optional(),` to `VariantConfigSchema`.

Before `StepEventSchema` add:

```ts
export const PageKindSchema = z.enum([
  "home",
  "category",
  "search",
  "product",
  "cart",
  "checkout",
  "confirmation",
  "other",
]);
export type PageKind = z.infer<typeof PageKindSchema>;
```

Add to `StepEventSchema`'s object:

```ts
    page: z.object({ kind: PageKindSchema, label: z.string().optional() }).strict().optional(),
    click: z.object({ x: z.number().min(0).max(1), y: z.number().min(0).max(1) }).strict().optional(),
    screenshotArtifactId: id.optional(),
    search: z
      .object({ query: z.string(), resultCount: z.number().int().nonnegative() })
      .strict()
      .optional(),
```

Before `EvaluationResultSchema` add, then add `findability: FindabilitySchema.optional(),` to it:

```ts
const count = z.number().int().nonnegative();
export const FindabilitySchema = z
  .object({
    outcome: z.enum([
      "purchased",
      "abandoned",
      "constraint-failure",
      "technical-failure",
      "timeout",
    ]),
    requirements: z.array(
      z.object({ id, met: z.boolean(), observed: z.string().optional() }).strict(),
    ),
    steps: count,
    optimalSteps: count,
    searches: count,
    emptySearches: count,
    revisits: count,
    abandonedAt: id.optional(),
  })
  .strict();
export type Findability = z.infer<typeof FindabilitySchema>;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm --filter @storefront/contracts test && pnpm --filter @storefront/contracts typecheck`
Expected: PASS.

- [ ] **Step 5: Checks** — `pnpm exec vp fmt packages/contracts/src`.

---

### Task 2: Go contract mirror and scenario loader

**Files:**

- Modify: `apps/api/internal/contracts/types.go`
- Create: `apps/api/internal/scenario/scenario.go`
- Test: `apps/api/internal/scenario/scenario_test.go`

**Interfaces:**

- Consumes: Task 1 field names (JSON tags must match exactly).
- Produces: `contracts.Requirement`, `contracts.ShoppingTask`, `contracts.Page`, `contracts.Point`, `contracts.Search`, `contracts.RequirementResult`, `contracts.Findability`; `scenario.Product`, `scenario.Step`, `scenario.Path`, `scenario.Scenario`, `scenario.Load([]byte) (*Scenario, error)`, `(*Scenario).Product(id string) (Product, bool)`, `(*Scenario).Path(personaID, variantID string) (Path, bool)`.

- [ ] **Step 1: Write the failing test** (`scenario_test.go`)

```go
package scenario

import (
	"strings"
	"testing"
)

const valid = `{"id":"t","products":[{"id":"p","name":"Shell","price":100,"attributes":{"size":["M"]}}],
"optimalSteps":{"a":2},
"paths":{"a":{"v":{"steps":[{"page":{"kind":"category","label":"Jackets"},"message":"Opened jackets","products":["p"]}],"purchase":"p"}}}}`

func TestLoad(t *testing.T) {
	s, err := Load([]byte(valid))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Path("a", "v"); !ok {
		t.Fatal("missing path")
	}
	if _, ok := (*Scenario)(nil).Path("a", "v"); ok {
		t.Fatal("nil scenario returned a path")
	}
	for _, bad := range []string{
		strings.Replace(valid, `"products":["p"]`, `"products":["x"]`, 1),
		strings.Replace(valid, `"kind":"category"`, `"kind":"basement"`, 1),
		strings.Replace(valid, `"purchase":"p"`, `"purchase":""`, 1),
		strings.Replace(valid, `"id":"t"`, `"id":"t","extra":1`, 1),
	} {
		if _, err := Load([]byte(bad)); err == nil {
			t.Fatalf("accepted invalid scenario: %s", bad)
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --filter @storefront/api test`
Expected: FAIL (package `scenario` has no `Load`).

- [ ] **Step 3: Implement**

Append to `types.go`, and add the marked fields to existing structs:

```go
type Requirement struct {
	ID        string `json:"id"`
	Label     string `json:"label"`
	Kind      string `json:"kind"`
	Attribute string `json:"attribute"`
	Value     string `json:"value"`
}
type ShoppingTask struct {
	Query        string        `json:"query"`
	Requirements []Requirement `json:"requirements"`
	Budget       float64       `json:"budget,omitempty"`
}
type Page struct {
	Kind  string `json:"kind"`
	Label string `json:"label,omitempty"`
}
type Point struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}
type Search struct {
	Query       string `json:"query"`
	ResultCount int    `json:"resultCount"`
}
type RequirementResult struct {
	ID       string `json:"id"`
	Met      bool   `json:"met"`
	Observed string `json:"observed,omitempty"`
}
type Findability struct {
	Outcome       string              `json:"outcome"`
	Requirements  []RequirementResult `json:"requirements"`
	Steps         int                 `json:"steps"`
	OptimalSteps  int                 `json:"optimalSteps"`
	Searches      int                 `json:"searches"`
	EmptySearches int                 `json:"emptySearches"`
	Revisits      int                 `json:"revisits"`
	AbandonedAt   string              `json:"abandonedAt,omitempty"`
}
```

- `Persona`: `Task *ShoppingTask \`json:"task,omitempty"\``
- `VariantConfig`: `ShowAttributes bool \`json:"showAttributes,omitempty"\``
- `StepEvent`: `Page *Page \`json:"page,omitempty"\``, `Click *Point \`json:"click,omitempty"\``, `ScreenshotArtifactID string \`json:"screenshotArtifactId,omitempty"\``, `Search *Search \`json:"search,omitempty"\``
- `EvaluationResult`: `Findability *Findability \`json:"findability,omitempty"\``

`scenario.go`:

```go
// Package scenario holds the scripted mock shopping scenario: catalog facts plus one
// recorded path per persona and variant. Real agent runs will replace the paths.
package scenario

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"storefront/api/internal/contracts"
)

type Product struct {
	ID         string              `json:"id"`
	Name       string              `json:"name"`
	Price      float64             `json:"price"`
	Attributes map[string][]string `json:"attributes"`
}
type Step struct {
	Page     contracts.Page    `json:"page"`
	Message  string            `json:"message"`
	Click    *contracts.Point  `json:"click,omitempty"`
	Search   *contracts.Search `json:"search,omitempty"`
	Products []string          `json:"products,omitempty"`
}

// Path is what one mock shopper did. Purchase names the product bought; Reason explains giving up.
type Path struct {
	Steps    []Step `json:"steps"`
	Purchase string `json:"purchase,omitempty"`
	Reason   string `json:"reason,omitempty"`
}
type Scenario struct {
	ID           string                     `json:"id"`
	Products     []Product                  `json:"products"`
	OptimalSteps map[string]int             `json:"optimalSteps"`
	Paths        map[string]map[string]Path `json:"paths"`
}

var pageKinds = []string{"home", "category", "search", "product", "cart", "checkout", "confirmation", "other"}

func Load(data []byte) (*Scenario, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var s Scenario
	if err := decoder.Decode(&s); err != nil {
		return nil, fmt.Errorf("invalid scenario: %w", err)
	}
	for persona, variants := range s.Paths {
		for variant, path := range variants {
			if (path.Purchase == "") == (path.Reason == "") {
				return nil, fmt.Errorf("scenario path %s/%s: set exactly one of purchase or reason", persona, variant)
			}
			ids := []string{}
			if path.Purchase != "" {
				ids = append(ids, path.Purchase)
			}
			for _, step := range path.Steps {
				if !slices.Contains(pageKinds, step.Page.Kind) {
					return nil, fmt.Errorf("scenario path %s/%s: unknown page kind %q", persona, variant, step.Page.Kind)
				}
				ids = append(ids, step.Products...)
			}
			for _, id := range ids {
				if _, ok := s.Product(id); !ok {
					return nil, fmt.Errorf("scenario path %s/%s: unknown product %q", persona, variant, id)
				}
			}
		}
	}
	return &s, nil
}

func (s *Scenario) Product(id string) (Product, bool) {
	for _, product := range s.Products {
		if product.ID == id {
			return product, true
		}
	}
	return Product{}, false
}

func (s *Scenario) Path(personaID, variantID string) (Path, bool) {
	if s == nil {
		return Path{}, false
	}
	path, ok := s.Paths[personaID][variantID]
	return path, ok
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm --filter @storefront/api test`
Expected: PASS.

- [ ] **Step 5: Checks** — `pnpm --filter @storefront/api typecheck` (go vet) and `gofmt -l apps/api` prints nothing.

---

### Task 3: SVG mock frames

**Files:**

- Create: `apps/api/internal/scenario/frame.go`
- Test: `apps/api/internal/scenario/frame_test.go`

**Interfaces:**

- Consumes: Task 2 `Scenario`, `Step`, `Product`.
- Produces: `(*Scenario).Frame(step Step, showAttributes bool) []byte`. Layout coordinates used by the scenario's click points: tile `i` center `((180+300*(i%4))/1280, (310+300*(i/4))/800)`; home button center `(0.125, 0.44)`; product "Add to cart" center `(0.625, 0.615)`; cart/checkout button center `(0.141, 0.365)`.

- [ ] **Step 1: Write the failing test** (`frame_test.go`)

```go
package scenario

import (
	"encoding/xml"
	"storefront/api/internal/contracts"
	"strings"
	"testing"
)

func TestFrameEscapesTextAndTogglesAttributes(t *testing.T) {
	s := &Scenario{Products: []Product{{ID: "p", Name: "Rain & <Shine>", Price: 90, Attributes: map[string][]string{"size": {"M"}}}}}
	step := Step{Page: contracts.Page{Kind: "category", Label: "Jackets"}, Products: []string{"p"}}
	hidden, shown := string(s.Frame(step, false)), string(s.Frame(step, true))
	if !strings.Contains(hidden, "Rain &amp; &lt;Shine&gt;") || strings.Contains(hidden, "<Shine>") {
		t.Fatal("product name not escaped")
	}
	if strings.Contains(hidden, "size: M") || !strings.Contains(shown, "size: M") {
		t.Fatal("card attributes not toggled")
	}
	product := string(s.Frame(Step{Page: contracts.Page{Kind: "product", Label: "Rain"}, Products: []string{"p"}}, false))
	if !strings.Contains(product, "size: M") {
		t.Fatal("product page must always show attributes")
	}
	if err := xml.Unmarshal([]byte(shown), new(struct{})); err != nil {
		t.Fatalf("invalid SVG: %v", err)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --filter @storefront/api test`
Expected: FAIL (`s.Frame undefined`).

- [ ] **Step 3: Implement** (`frame.go`)

```go
package scenario

import (
	"fmt"
	"html"
	"maps"
	"slices"
	"strings"
)

var buttonLabels = map[string]string{"product": "Add to cart", "cart": "Checkout", "checkout": "Place demo order"}

// Frame draws a plain SVG stand-in for the page a mock shopper saw. Real runs store real screenshots.
func (s *Scenario) Frame(step Step, showAttributes bool) []byte {
	var b strings.Builder
	text := func(x, y, size int, weight, fill, value string) {
		fmt.Fprintf(&b, `<text x="%d" y="%d" font-size="%d" font-weight="%s" fill="%s">%s</text>`, x, y, size, weight, fill, html.EscapeString(value))
	}
	button := func(x, y int, label string) {
		fmt.Fprintf(&b, `<rect x="%d" y="%d" width="280" height="64" rx="6" fill="#283e33"/>`, x, y)
		text(x+24, y+40, 22, "700", "#e9ead8", label)
	}
	b.WriteString(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 800" font-family="sans-serif">`)
	b.WriteString(`<rect width="1280" height="800" fill="#f7f7f0"/><rect width="1280" height="72" fill="#283e33"/>`)
	text(40, 46, 26, "700", "#e9ead8", "fieldwork")
	text(1100, 46, 18, "400", "#e9ead8", "Mock frame")
	text(40, 130, 34, "700", "#263d32", step.Page.Label)
	products := []Product{}
	for _, id := range step.Products {
		if product, ok := s.Product(id); ok {
			products = append(products, product)
		}
	}
	switch step.Page.Kind {
	case "home":
		text(40, 260, 56, "700", "#263d32", "Jackets for every kind of weather")
		button(40, 320, "Shop jackets")
	case "category", "search":
		if step.Search != nil && step.Search.ResultCount == 0 {
			text(40, 200, 24, "400", "#5e6a63", fmt.Sprintf("No results for “%s”", step.Search.Query))
		}
		for i, product := range products {
			x, y := 40+(i%4)*300, 170+(i/4)*300
			fmt.Fprintf(&b, `<rect x="%d" y="%d" width="280" height="280" rx="8" fill="#ffffff" stroke="#d5dad3"/>`, x, y)
			fmt.Fprintf(&b, `<rect x="%d" y="%d" width="248" height="120" rx="4" fill="#e3e8df"/>`, x+16, y+16)
			text(x+16, y+170, 20, "700", "#263d32", product.Name)
			text(x+16, y+198, 18, "400", "#263d32", fmt.Sprintf("$%.0f", product.Price))
			if showAttributes {
				for j, line := range attributeLines(product) {
					text(x+16, y+226+j*18, 14, "400", "#5e6a63", line)
				}
			}
		}
	default:
		if len(products) == 0 {
			break
		}
		product := products[0]
		if step.Page.Kind == "product" {
			b.WriteString(`<rect x="40" y="170" width="560" height="560" rx="8" fill="#e3e8df"/>`)
			text(660, 220, 34, "700", "#263d32", product.Name)
			text(660, 270, 26, "400", "#263d32", fmt.Sprintf("$%.0f", product.Price))
			for j, line := range attributeLines(product) {
				text(660, 320+j*32, 20, "400", "#5e6a63", line)
			}
			button(660, 460, buttonLabels["product"])
			break
		}
		text(40, 220, 24, "400", "#263d32", fmt.Sprintf("1 × %s   $%.0f", product.Name, product.Price))
		if label, ok := buttonLabels[step.Page.Kind]; ok {
			button(40, 260, label)
		}
	}
	b.WriteString(`</svg>`)
	return []byte(b.String())
}

func attributeLines(product Product) []string {
	lines := []string{}
	for _, key := range slices.Sorted(maps.Keys(product.Attributes)) {
		lines = append(lines, fmt.Sprintf("%s: %s", key, strings.Join(product.Attributes[key], ", ")))
	}
	return lines
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm --filter @storefront/api test`
Expected: PASS.

- [ ] **Step 5: Checks** — `gofmt -l apps/api` prints nothing.

---

### Task 4: Mock runner replays scenario paths

**Files:**

- Modify: `apps/api/internal/runner/runner.go`
- Modify: `apps/api/queue.go` (emit mapping, adapter and evaluator wiring)
- Modify: `apps/api/app.go` (embed and load scenario, artifact headers)
- Modify: `apps/api/store.go:33` (refresh catalog fixture on open)
- Test: `apps/api/internal/runner/runner_test.go`

**Interfaces:**

- Consumes: Task 2 `scenario.Load`, `(*Scenario).Path`; Task 3 `(*Scenario).Frame`.
- Produces: `runner.Event{Kind, Message, Page *contracts.Page, Click *contracts.Point, Search *contracts.Search, ScreenshotFile string}`; `runner.Result.PurchasedProductID string`; `runner.Mock{StepDelay, Scenario *scenario.Scenario}`; `newQueue(store, capacity, delay, root, scenario *scenario.Scenario)`. `evaluator.Deterministic` gains a `Scenario` field in Task 5; this task constructs it as `evaluator.Deterministic{Scenario: q.scenario}` after Task 5, so do Task 5's struct change first if executing out of order.

- [ ] **Step 1: Write the failing test** (`runner_test.go`)

```go
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --filter @storefront/api test`
Expected: FAIL (unknown fields `Scenario`, `Page`, `ScreenshotFile`).

- [ ] **Step 3: Implement**

In `runner.go`: extend `Event` and `Result`, replace `Mock` and its `Execute` entirely (the shipping rule goes away; every catalog persona now has a scenario path), and add `wait` and `saveResult`:

```go
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
```

```go
// Mock replays the scripted scenario path for the run's persona and variant, drawing one frame per step.
type Mock struct {
	StepDelay time.Duration
	Scenario  *scenario.Scenario
}

func (m Mock) Execute(ctx context.Context, c Context) (Result, error) {
	path, ok := m.Scenario.Path(c.Run.PersonaID, c.Run.VariantID)
	if !ok {
		return Result{}, fmt.Errorf("no mock scenario path for persona %s on variant %s", c.Run.PersonaID, c.Run.VariantID)
	}
	for i, step := range path.Steps {
		file := fmt.Sprintf("step-%02d.svg", i)
		if err := os.WriteFile(filepath.Join(c.ArtifactDirectory, file), m.Scenario.Frame(step, c.Run.Config.ShowAttributes), 0600); err != nil {
			return Result{}, err
		}
		if err := c.SaveArtifact(Artifact{"screenshot", file, "image/svg+xml"}); err != nil {
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
		Notes:              []string{"Scripted mock shopper on drawn page frames; not measured human behavior.", fmt.Sprintf("Recorded seed: %d", c.Run.Seed)},
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
```

Add `"storefront/api/internal/scenario"` to imports.

In `queue.go`: add `scenario *scenario.Scenario` to `JobQueue`, a `scenario *scenario.Scenario` parameter to `newQueue` (stored on the struct), and replace `emit`:

```go
	emit := func(event runner.Event) error {
		step := contracts.StepEvent{ID: newID(), RunID: run.ID, Sequence: len(run.Events), At: now(), Kind: event.Kind, Message: event.Message, Page: event.Page, Click: event.Click, Search: event.Search}
		if event.ScreenshotFile != "" {
			index := slices.IndexFunc(run.Artifacts, func(item contracts.Artifact) bool { return path.Base(item.Path) == event.ScreenshotFile })
			if index < 0 {
				return fmt.Errorf("step screenshot %s was not saved", event.ScreenshotFile)
			}
			step.ScreenshotArtifactID = run.Artifacts[index].ID
		}
		run.Events = append(run.Events, step)
		return q.store.saveRun(run)
	}
```

Use `runner.Mock{StepDelay: q.delay, Scenario: q.scenario}` and `evaluator.Deterministic{Scenario: q.scenario}`. Imports: `path`, `slices`, `storefront/api/internal/scenario`.

In `app.go`: add next to the catalog embed

```go
//go:embed fixtures/scenarios/jacket.json
var scenarioJSON []byte
```

At the top of `NewApp` (before acquiring the lock):

```go
	mockScenario, err := scenario.Load(scenarioJSON)
	if err != nil {
		return nil, err
	}
```

(the later `dataDir, err := ...` becomes `dataDir, err = ...`), and pass `mockScenario` to `newQueue`. In `artifact`, before `ServeContent`:

```go
	w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; sandbox")
	w.Header().Set("X-Content-Type-Options", "nosniff")
```

Until Task 6 adds the real file, create a placeholder `apps/api/fixtures/scenarios/jacket.json` containing `{"id":"jacket","products":[],"optimalSteps":{},"paths":{}}` so the embed compiles.

In `store.go`, replace the `INSERT OR IGNORE` with an upsert so fixture edits apply on restart:

```go
	// Runs snapshot their persona and config, so refreshing the fixture leaves history intact.
	if _, err = db.Exec("INSERT INTO catalog (id, body) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET body = excluded.body", string(catalogJSON)); err != nil {
```

- [ ] **Step 4: Run tests**

Run: `pnpm --filter @storefront/api test`
Expected: `TestMockReplaysScenarioPath` PASS. `TestMockEndToEnd` FAILS until Task 6 (catalog personas have no scenario paths yet); that is expected.

- [ ] **Step 5: Checks** — `pnpm --filter @storefront/api typecheck`, `gofmt -l apps/api`.

---

### Task 5: Evaluator findability and optimizer proposal

**Files:**

- Modify: `apps/api/internal/evaluator/evaluator.go`
- Modify: `apps/api/internal/optimizer/optimizer.go`
- Test: `apps/api/internal/evaluator/evaluator_test.go`, `apps/api/internal/optimizer/optimizer_test.go`

**Interfaces:**

- Consumes: `contracts.Findability`, `scenario.Scenario`, `runner.Result.PurchasedProductID`.
- Produces: `evaluator.Deterministic{Scenario *scenario.Scenario}`; findability outcomes `purchased | abandoned | constraint-failure`; requirement result id `"budget"` when the task has a budget; proposal patch `{"showAttributes": true}`.

- [ ] **Step 1: Write the failing tests**

`evaluator_test.go`:

```go
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
	if f := bought.Findability; f == nil || f.Outcome != "purchased" || !bought.TaskCompleted || bought.Score != 1 || f.AbandonedAt != "" {
		t.Fatalf("expected purchase: %+v", bought)
	}
	wrong := evaluate("harbor")
	if wrong.Findability.Outcome != "constraint-failure" || wrong.TaskCompleted || len(wrong.FailureReasons) != 1 {
		t.Fatalf("expected constraint failure: %+v", wrong)
	}
	gave := evaluate("")
	f := gave.Findability
	if f.Outcome != "abandoned" || f.Steps != 3 || f.OptimalSteps != 2 || f.Searches != 1 || f.EmptySearches != 1 || f.Revisits != 1 || f.AbandonedAt != "3" || len(f.Requirements) != 3 {
		t.Fatalf("wrong abandon findability: %+v", f)
	}

	// Runs without recorded page steps (today's scripted browser check) keep their own completion.
	browser := Deterministic{Scenario: s}.Evaluate(contracts.Run{Persona: contracts.Persona{Task: task}}, runner.Result{TaskCompleted: true})
	if browser.Findability != nil || !browser.TaskCompleted {
		t.Fatalf("browser run was judged by the scenario: %+v", browser)
	}
}
```

`optimizer_test.go`:

```go
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm --filter @storefront/api test`
Expected: FAIL (unknown field `Scenario`, no findability, no attributes proposal).

- [ ] **Step 3: Implement**

`evaluator.go`:

```go
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
```

`optimizer.go` (the shipping rule is removed; no persona triggers it any more):

```go
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
```

- [ ] **Step 4: Run tests**

Run: `pnpm --filter @storefront/api test`
Expected: evaluator and optimizer tests PASS; `TestMockEndToEnd` still fails until Task 6.

- [ ] **Step 5: Checks** — `pnpm --filter @storefront/api typecheck`, `gofmt -l apps/api`.

---

### Task 6: Jacket scenario fixtures and end-to-end API test

**Files:**

- Modify: `apps/api/fixtures/catalog.json` (full replacement)
- Modify: `apps/api/fixtures/scenarios/jacket.json` (full replacement)
- Modify: `apps/api/app_test.go`

**Interfaces:**

- Consumes: everything in Tasks 2–5.
- Produces: experiment `jacket-study`; variants `baseline` (cards hide details) and `card-details` (cards show details); personas `dev-commuter, dev-hiker, dev-student, dev-searcher, dev-gifter` (development) and `ho-angler, ho-cyclist, ho-walker, ho-hurried, ho-parent` (held-out), in that order. Expected outcomes: development baseline 3/5 → candidate 5/5; held-out baseline 2/5 (`ho-cyclist`, `ho-hurried`) → candidate 4/5, with `ho-hurried` worse (buys the non-waterproof Trail windbreaker).

- [ ] **Step 1: Update `app_test.go` to the new fixtures (failing first)**

In `TestMockEndToEnd`:

- Replace `catalog.Personas[2]` with `catalog.Personas[5]` (one development, one held-out persona).
- Replace the artifact assertion and download block:

```go
		resultIndex := slices.IndexFunc(run.Artifacts, func(item contracts.Artifact) bool { return item.Kind == "result" })
		if run.Evaluation == nil || !run.Evaluation.Synthetic || run.Evaluation.Findability == nil || resultIndex < 0 {
			t.Fatal("missing synthetic evaluation, findability or result artifact")
		}
		for _, event := range run.Events {
			if event.Page != nil && !slices.ContainsFunc(run.Artifacts, func(item contracts.Artifact) bool { return item.ID == event.ScreenshotArtifactID }) {
				t.Fatal("step screenshot does not resolve to an artifact")
			}
		}
		frame := request(t, app, "GET", "/api/artifacts/"+run.Events[1].ScreenshotArtifactID, nil)
		if frame.Code != 200 || frame.Header().Get("Content-Type") != "image/svg+xml" || frame.Header().Get("X-Content-Type-Options") != "nosniff" || !strings.Contains(frame.Header().Get("Content-Security-Policy"), "sandbox") {
			t.Fatalf("frame served unsafely: %d %v", frame.Code, frame.Header())
		}
		artifact := request(t, app, "GET", "/api/artifacts/"+run.Artifacts[resultIndex].ID, nil)
```

(`run.Events[0]` is the `running` status event, so `Events[1]` is the first page step.) Keep the remaining assertions: the development baseline run of `dev-commuter` gives up after three product pages, so it still yields exactly one proposal. Add `"strings"` to imports.

In `TestQueueAndStrictInput` (line ~163) replace `"demo-experiment"` with `"jacket-study"` and `"dev-budget"` with `"dev-commuter"`. Grep the file for any other old ids (`grep -n "demo-experiment\|dev-budget\|dev-explorer\|heldout-" apps/api/app_test.go`) and update them the same way.

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm --filter @storefront/api test`
Expected: FAIL (no scenario paths, unknown persona ids).

- [ ] **Step 3: Replace the fixtures**

`apps/api/fixtures/catalog.json`:

```json
{
  "experiment": {
    "id": "jacket-study",
    "name": "Finding the right rain jacket",
    "baselineVariantId": "baseline",
    "variantIds": ["baseline", "card-details"],
    "developmentPersonaIds": [
      "dev-commuter",
      "dev-hiker",
      "dev-student",
      "dev-searcher",
      "dev-gifter"
    ],
    "heldOutPersonaIds": ["ho-angler", "ho-cyclist", "ho-walker", "ho-hurried", "ho-parent"],
    "seed": 42,
    "createdAt": "2026-10-17T00:00:00.000Z",
    "contractVersion": 1
  },
  "personas": [
    {
      "id": "dev-commuter",
      "name": "Rainy commuter",
      "goal": "Waterproof navy jacket, medium, under $150",
      "split": "development",
      "traits": ["specific", "patient"],
      "history": [],
      "task": {
        "query": "waterproof navy jacket, medium, under $150",
        "budget": 150,
        "requirements": [
          { "id": "color", "label": "Navy", "kind": "hard", "attribute": "color", "value": "navy" },
          { "id": "size", "label": "Size M", "kind": "hard", "attribute": "size", "value": "M" },
          {
            "id": "waterproof",
            "label": "Waterproof",
            "kind": "hard",
            "attribute": "waterproof",
            "value": "yes"
          }
        ]
      }
    },
    {
      "id": "dev-hiker",
      "name": "Weekend hiker",
      "goal": "Waterproof jacket in large, olive if possible, under $120",
      "split": "development",
      "traits": ["outdoors"],
      "history": [],
      "task": {
        "query": "waterproof jacket in large, olive if possible, under $120",
        "budget": 120,
        "requirements": [
          {
            "id": "waterproof",
            "label": "Waterproof",
            "kind": "hard",
            "attribute": "waterproof",
            "value": "yes"
          },
          { "id": "size", "label": "Size L", "kind": "hard", "attribute": "size", "value": "L" },
          {
            "id": "color",
            "label": "Olive",
            "kind": "preference",
            "attribute": "color",
            "value": "olive"
          }
        ]
      }
    },
    {
      "id": "dev-student",
      "name": "Student on a budget",
      "goal": "Any medium jacket under $100",
      "split": "development",
      "traits": ["budget"],
      "history": [],
      "task": {
        "query": "any medium jacket under $100",
        "budget": 100,
        "requirements": [
          { "id": "size", "label": "Size M", "kind": "hard", "attribute": "size", "value": "M" }
        ]
      }
    },
    {
      "id": "dev-searcher",
      "name": "Search-first shopper",
      "goal": "Navy rain shell, medium",
      "split": "development",
      "traits": ["searches first"],
      "history": [],
      "task": {
        "query": "navy rain shell, medium",
        "budget": 200,
        "requirements": [
          { "id": "color", "label": "Navy", "kind": "hard", "attribute": "color", "value": "navy" },
          { "id": "size", "label": "Size M", "kind": "hard", "attribute": "size", "value": "M" },
          {
            "id": "waterproof",
            "label": "Waterproof",
            "kind": "hard",
            "attribute": "waterproof",
            "value": "yes"
          }
        ]
      }
    },
    {
      "id": "dev-gifter",
      "name": "Gift buyer",
      "goal": "Small waterproof jacket under $150",
      "split": "development",
      "traits": ["gift"],
      "history": [],
      "task": {
        "query": "small waterproof jacket under $150",
        "budget": 150,
        "requirements": [
          { "id": "size", "label": "Size S", "kind": "hard", "attribute": "size", "value": "S" },
          {
            "id": "waterproof",
            "label": "Waterproof",
            "kind": "hard",
            "attribute": "waterproof",
            "value": "yes"
          }
        ]
      }
    },
    {
      "id": "ho-angler",
      "name": "Coastal angler",
      "goal": "Navy waterproof jacket, large, under $150",
      "split": "held-out",
      "traits": ["specific"],
      "history": [],
      "task": {
        "query": "navy waterproof jacket, large, under $150",
        "budget": 150,
        "requirements": [
          { "id": "color", "label": "Navy", "kind": "hard", "attribute": "color", "value": "navy" },
          { "id": "size", "label": "Size L", "kind": "hard", "attribute": "size", "value": "L" },
          {
            "id": "waterproof",
            "label": "Waterproof",
            "kind": "hard",
            "attribute": "waterproof",
            "value": "yes"
          }
        ]
      }
    },
    {
      "id": "ho-cyclist",
      "name": "Night cyclist",
      "goal": "Black waterproof jacket, medium, under $250",
      "split": "held-out",
      "traits": ["searches first"],
      "history": [],
      "task": {
        "query": "black waterproof jacket, medium, under $250",
        "budget": 250,
        "requirements": [
          {
            "id": "color",
            "label": "Black",
            "kind": "hard",
            "attribute": "color",
            "value": "black"
          },
          { "id": "size", "label": "Size M", "kind": "hard", "attribute": "size", "value": "M" },
          {
            "id": "waterproof",
            "label": "Waterproof",
            "kind": "hard",
            "attribute": "waterproof",
            "value": "yes"
          }
        ]
      }
    },
    {
      "id": "ho-walker",
      "name": "Dog walker",
      "goal": "Waterproof medium jacket under $120",
      "split": "held-out",
      "traits": ["budget"],
      "history": [],
      "task": {
        "query": "waterproof medium jacket under $120",
        "budget": 120,
        "requirements": [
          { "id": "size", "label": "Size M", "kind": "hard", "attribute": "size", "value": "M" },
          {
            "id": "waterproof",
            "label": "Waterproof",
            "kind": "hard",
            "attribute": "waterproof",
            "value": "yes"
          }
        ]
      }
    },
    {
      "id": "ho-hurried",
      "name": "Hurried shopper",
      "goal": "Waterproof medium jacket under $150",
      "split": "held-out",
      "traits": ["impatient"],
      "history": [],
      "task": {
        "query": "waterproof medium jacket under $150",
        "budget": 150,
        "requirements": [
          { "id": "size", "label": "Size M", "kind": "hard", "attribute": "size", "value": "M" },
          {
            "id": "waterproof",
            "label": "Waterproof",
            "kind": "hard",
            "attribute": "waterproof",
            "value": "yes"
          }
        ]
      }
    },
    {
      "id": "ho-parent",
      "name": "Rain-ready parent",
      "goal": "Navy waterproof jacket, medium, under $150",
      "split": "held-out",
      "traits": ["specific"],
      "history": [],
      "task": {
        "query": "navy waterproof jacket, medium, under $150",
        "budget": 150,
        "requirements": [
          { "id": "color", "label": "Navy", "kind": "hard", "attribute": "color", "value": "navy" },
          { "id": "size", "label": "Size M", "kind": "hard", "attribute": "size", "value": "M" },
          {
            "id": "waterproof",
            "label": "Waterproof",
            "kind": "hard",
            "attribute": "waterproof",
            "value": "yes"
          }
        ]
      }
    }
  ],
  "variants": [
    {
      "id": "baseline",
      "experimentId": "jacket-study",
      "name": "Today's shop",
      "isBaseline": true,
      "config": {
        "headline": "Jackets for every kind of weather",
        "showShipping": true,
        "recommendationOrder": "popular"
      }
    },
    {
      "id": "card-details",
      "experimentId": "jacket-study",
      "name": "Details on cards",
      "isBaseline": false,
      "config": {
        "headline": "Jackets for every kind of weather",
        "showShipping": true,
        "recommendationOrder": "popular",
        "showAttributes": true
      }
    }
  ]
}
```

`apps/api/fixtures/scenarios/jacket.json` — category grid order is fixed: `harbor` (tile 0), `trail` (1), `alpine` (2), `coastline` (3), `summit` (4), `ridgeline` (5). Tile click centers: 0 `(0.141,0.388)`, 1 `(0.375,0.388)`, 2 `(0.609,0.388)`, 3 `(0.844,0.388)`, 4 `(0.141,0.763)`, 5 `(0.375,0.763)`.

```json
{
  "id": "jacket",
  "products": [
    {
      "id": "harbor",
      "name": "Harbor rain jacket",
      "price": 120,
      "attributes": { "color": ["navy"], "size": ["S", "L"], "waterproof": ["yes"] }
    },
    {
      "id": "trail",
      "name": "Trail windbreaker",
      "price": 95,
      "attributes": { "color": ["navy"], "size": ["M"], "waterproof": ["no"] }
    },
    {
      "id": "alpine",
      "name": "Alpine parka",
      "price": 210,
      "attributes": { "color": ["black"], "size": ["M", "L"], "waterproof": ["yes"] }
    },
    {
      "id": "coastline",
      "name": "Coastline anorak",
      "price": 110,
      "attributes": { "color": ["olive"], "size": ["M", "L"], "waterproof": ["yes"] }
    },
    {
      "id": "summit",
      "name": "Summit fleece",
      "price": 70,
      "attributes": { "color": ["navy"], "size": ["M"], "waterproof": ["no"] }
    },
    {
      "id": "ridgeline",
      "name": "Ridgeline shell",
      "price": 138,
      "attributes": { "color": ["navy"], "size": ["S", "M", "L"], "waterproof": ["yes"] }
    }
  ],
  "optimalSteps": {
    "dev-commuter": 6,
    "dev-hiker": 6,
    "dev-student": 6,
    "dev-searcher": 6,
    "dev-gifter": 6,
    "ho-angler": 6,
    "ho-cyclist": 6,
    "ho-walker": 6,
    "ho-hurried": 6,
    "ho-parent": 6
  },
  "paths": {
    "dev-commuter": {
      "baseline": {
        "steps": [
          {
            "page": { "kind": "home", "label": "Home" },
            "message": "Opened the shop",
            "click": { "x": 0.125, "y": 0.44 }
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Opened Jackets; the cards show only name and price",
            "click": { "x": 0.141, "y": 0.388 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Harbor rain jacket" },
            "message": "Harbor rain jacket comes in S and L, not M",
            "products": ["harbor"]
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Went back to Jackets",
            "click": { "x": 0.375, "y": 0.388 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Trail windbreaker" },
            "message": "Trail windbreaker isn't waterproof",
            "products": ["trail"]
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Went back to Jackets again",
            "click": { "x": 0.141, "y": 0.763 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Summit fleece" },
            "message": "Summit fleece isn't waterproof either",
            "products": ["summit"]
          }
        ],
        "reason": "Gave up after three product pages: the cards don't say which jackets are waterproof or come in medium."
      },
      "card-details": {
        "steps": [
          {
            "page": { "kind": "home", "label": "Home" },
            "message": "Opened the shop",
            "click": { "x": 0.125, "y": 0.44 }
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Spotted navy, medium and waterproof on the Ridgeline shell card",
            "click": { "x": 0.375, "y": 0.763 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Ridgeline shell" },
            "message": "Added the Ridgeline shell in M to the cart",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["ridgeline"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["ridgeline"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["ridgeline"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["ridgeline"]
          }
        ],
        "purchase": "ridgeline"
      }
    },
    "dev-hiker": {
      "baseline": {
        "steps": [
          { "page": { "kind": "home", "label": "Home" }, "message": "Opened the shop" },
          {
            "page": { "kind": "search", "label": "Search: waterproof jacket" },
            "message": "Searched for waterproof jacket",
            "click": { "x": 0.609, "y": 0.388 },
            "search": { "query": "waterproof jacket", "resultCount": 4 },
            "products": ["harbor", "alpine", "coastline", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Coastline anorak" },
            "message": "Coastline anorak: olive, large, waterproof, $110",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["coastline"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["coastline"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["coastline"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["coastline"]
          }
        ],
        "purchase": "coastline"
      },
      "card-details": {
        "steps": [
          { "page": { "kind": "home", "label": "Home" }, "message": "Opened the shop" },
          {
            "page": { "kind": "search", "label": "Search: waterproof jacket" },
            "message": "Searched for waterproof jacket",
            "click": { "x": 0.609, "y": 0.388 },
            "search": { "query": "waterproof jacket", "resultCount": 4 },
            "products": ["harbor", "alpine", "coastline", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Coastline anorak" },
            "message": "Coastline anorak: olive, large, waterproof, $110",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["coastline"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["coastline"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["coastline"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["coastline"]
          }
        ],
        "purchase": "coastline"
      }
    },
    "dev-student": {
      "baseline": {
        "steps": [
          {
            "page": { "kind": "home", "label": "Home" },
            "message": "Opened the shop",
            "click": { "x": 0.125, "y": 0.44 }
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Looked for the cheapest jacket",
            "click": { "x": 0.375, "y": 0.388 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Trail windbreaker" },
            "message": "Trail windbreaker: medium, $95",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["trail"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["trail"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["trail"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["trail"]
          }
        ],
        "purchase": "trail"
      },
      "card-details": {
        "steps": [
          {
            "page": { "kind": "home", "label": "Home" },
            "message": "Opened the shop",
            "click": { "x": 0.125, "y": 0.44 }
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Looked for the cheapest jacket",
            "click": { "x": 0.375, "y": 0.388 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Trail windbreaker" },
            "message": "Trail windbreaker: medium, $95",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["trail"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["trail"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["trail"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["trail"]
          }
        ],
        "purchase": "trail"
      }
    },
    "dev-searcher": {
      "baseline": {
        "steps": [
          { "page": { "kind": "home", "label": "Home" }, "message": "Opened the shop" },
          {
            "page": { "kind": "search", "label": "Search: rain shell navy" },
            "message": "Searched for rain shell navy and got nothing",
            "search": { "query": "rain shell navy", "resultCount": 0 }
          },
          {
            "page": { "kind": "search", "label": "Search: navy jacket" },
            "message": "Searched for navy jacket",
            "click": { "x": 0.141, "y": 0.388 },
            "search": { "query": "navy jacket", "resultCount": 4 },
            "products": ["harbor", "trail", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Harbor rain jacket" },
            "message": "Harbor rain jacket has no medium",
            "products": ["harbor"]
          },
          {
            "page": { "kind": "search", "label": "Search: navy jacket" },
            "message": "Went back to the search results",
            "click": { "x": 0.375, "y": 0.388 },
            "products": ["harbor", "trail", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Trail windbreaker" },
            "message": "Trail windbreaker isn't waterproof",
            "products": ["trail"]
          }
        ],
        "reason": "Gave up after a search with no results and two products that didn't fit."
      },
      "card-details": {
        "steps": [
          { "page": { "kind": "home", "label": "Home" }, "message": "Opened the shop" },
          {
            "page": { "kind": "search", "label": "Search: rain shell navy" },
            "message": "Searched for rain shell navy and got nothing",
            "search": { "query": "rain shell navy", "resultCount": 0 }
          },
          {
            "page": { "kind": "search", "label": "Search: navy jacket" },
            "message": "The Ridgeline card shows medium and waterproof",
            "click": { "x": 0.844, "y": 0.388 },
            "search": { "query": "navy jacket", "resultCount": 4 },
            "products": ["harbor", "trail", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Ridgeline shell" },
            "message": "Added the Ridgeline shell in M to the cart",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["ridgeline"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["ridgeline"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["ridgeline"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["ridgeline"]
          }
        ],
        "purchase": "ridgeline"
      }
    },
    "dev-gifter": {
      "baseline": {
        "steps": [
          {
            "page": { "kind": "home", "label": "Home" },
            "message": "Opened the shop",
            "click": { "x": 0.125, "y": 0.44 }
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Opened Jackets",
            "click": { "x": 0.141, "y": 0.388 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Harbor rain jacket" },
            "message": "Harbor rain jacket: small, waterproof, $120",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["harbor"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["harbor"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["harbor"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["harbor"]
          }
        ],
        "purchase": "harbor"
      },
      "card-details": {
        "steps": [
          {
            "page": { "kind": "home", "label": "Home" },
            "message": "Opened the shop",
            "click": { "x": 0.125, "y": 0.44 }
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Opened Jackets",
            "click": { "x": 0.141, "y": 0.388 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Harbor rain jacket" },
            "message": "Harbor rain jacket: small, waterproof, $120",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["harbor"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["harbor"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["harbor"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["harbor"]
          }
        ],
        "purchase": "harbor"
      }
    },
    "ho-angler": {
      "baseline": {
        "steps": [
          {
            "page": { "kind": "home", "label": "Home" },
            "message": "Opened the shop",
            "click": { "x": 0.125, "y": 0.44 }
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Opened Jackets",
            "click": { "x": 0.375, "y": 0.388 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Trail windbreaker" },
            "message": "Trail windbreaker isn't waterproof",
            "products": ["trail"]
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Went back to Jackets",
            "click": { "x": 0.609, "y": 0.388 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Alpine parka" },
            "message": "Alpine parka is black and over budget",
            "products": ["alpine"]
          }
        ],
        "reason": "Gave up after two product pages without finding a navy waterproof jacket in large."
      },
      "card-details": {
        "steps": [
          {
            "page": { "kind": "home", "label": "Home" },
            "message": "Opened the shop",
            "click": { "x": 0.125, "y": 0.44 }
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "The Harbor card shows navy, large and waterproof",
            "click": { "x": 0.141, "y": 0.388 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Harbor rain jacket" },
            "message": "Added the Harbor rain jacket in L to the cart",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["harbor"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["harbor"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["harbor"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["harbor"]
          }
        ],
        "purchase": "harbor"
      }
    },
    "ho-cyclist": {
      "baseline": {
        "steps": [
          { "page": { "kind": "home", "label": "Home" }, "message": "Opened the shop" },
          {
            "page": { "kind": "search", "label": "Search: black waterproof jacket" },
            "message": "Searched for black waterproof jacket",
            "click": { "x": 0.141, "y": 0.388 },
            "search": { "query": "black waterproof jacket", "resultCount": 1 },
            "products": ["alpine"]
          },
          {
            "page": { "kind": "product", "label": "Alpine parka" },
            "message": "Alpine parka: black, medium, waterproof, $210",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["alpine"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["alpine"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["alpine"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["alpine"]
          }
        ],
        "purchase": "alpine"
      },
      "card-details": {
        "steps": [
          { "page": { "kind": "home", "label": "Home" }, "message": "Opened the shop" },
          {
            "page": { "kind": "search", "label": "Search: black waterproof jacket" },
            "message": "Searched for black waterproof jacket",
            "click": { "x": 0.141, "y": 0.388 },
            "search": { "query": "black waterproof jacket", "resultCount": 1 },
            "products": ["alpine"]
          },
          {
            "page": { "kind": "product", "label": "Alpine parka" },
            "message": "Alpine parka: black, medium, waterproof, $210",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["alpine"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["alpine"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["alpine"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["alpine"]
          }
        ],
        "purchase": "alpine"
      }
    },
    "ho-walker": {
      "baseline": {
        "steps": [
          {
            "page": { "kind": "home", "label": "Home" },
            "message": "Opened the shop",
            "click": { "x": 0.125, "y": 0.44 }
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Opened Jackets",
            "click": { "x": 0.141, "y": 0.763 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Summit fleece" },
            "message": "Summit fleece isn't waterproof",
            "products": ["summit"]
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Went back to Jackets",
            "click": { "x": 0.375, "y": 0.388 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Trail windbreaker" },
            "message": "Trail windbreaker isn't waterproof either",
            "products": ["trail"]
          }
        ],
        "reason": "Gave up after two product pages that weren't waterproof."
      },
      "card-details": {
        "steps": [
          {
            "page": { "kind": "home", "label": "Home" },
            "message": "Opened the shop",
            "click": { "x": 0.125, "y": 0.44 }
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "The Coastline card shows medium and waterproof",
            "click": { "x": 0.844, "y": 0.388 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Coastline anorak" },
            "message": "Added the Coastline anorak in M to the cart",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["coastline"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["coastline"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["coastline"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["coastline"]
          }
        ],
        "purchase": "coastline"
      }
    },
    "ho-hurried": {
      "baseline": {
        "steps": [
          {
            "page": { "kind": "home", "label": "Home" },
            "message": "Opened the shop",
            "click": { "x": 0.125, "y": 0.44 }
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Picked the jacket that looked most like a rain shell",
            "click": { "x": 0.375, "y": 0.763 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Ridgeline shell" },
            "message": "Ridgeline shell: medium, waterproof, $138",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["ridgeline"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["ridgeline"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["ridgeline"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["ridgeline"]
          }
        ],
        "purchase": "ridgeline"
      },
      "card-details": {
        "steps": [
          {
            "page": { "kind": "home", "label": "Home" },
            "message": "Opened the shop",
            "click": { "x": 0.125, "y": 0.44 }
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Picked the cheapest card listing medium",
            "click": { "x": 0.375, "y": 0.388 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Trail windbreaker" },
            "message": "Added the Trail windbreaker without checking waterproofing",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["trail"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["trail"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["trail"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["trail"]
          }
        ],
        "purchase": "trail"
      }
    },
    "ho-parent": {
      "baseline": {
        "steps": [
          {
            "page": { "kind": "home", "label": "Home" },
            "message": "Opened the shop",
            "click": { "x": 0.125, "y": 0.44 }
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Opened Jackets",
            "click": { "x": 0.141, "y": 0.388 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Harbor rain jacket" },
            "message": "Harbor rain jacket has no medium",
            "products": ["harbor"]
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "Went back to Jackets",
            "click": { "x": 0.375, "y": 0.388 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Trail windbreaker" },
            "message": "Trail windbreaker isn't waterproof",
            "products": ["trail"]
          }
        ],
        "reason": "Gave up after two product pages without a waterproof medium."
      },
      "card-details": {
        "steps": [
          {
            "page": { "kind": "home", "label": "Home" },
            "message": "Opened the shop",
            "click": { "x": 0.125, "y": 0.44 }
          },
          {
            "page": { "kind": "category", "label": "Jackets" },
            "message": "The Ridgeline card shows navy, medium and waterproof",
            "click": { "x": 0.375, "y": 0.763 },
            "products": ["harbor", "trail", "alpine", "coastline", "summit", "ridgeline"]
          },
          {
            "page": { "kind": "product", "label": "Ridgeline shell" },
            "message": "Added the Ridgeline shell in M to the cart",
            "click": { "x": 0.625, "y": 0.615 },
            "products": ["ridgeline"]
          },
          {
            "page": { "kind": "cart", "label": "Cart" },
            "message": "Reviewed the cart",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["ridgeline"]
          },
          {
            "page": { "kind": "checkout", "label": "Checkout" },
            "message": "Placed the demo order",
            "click": { "x": 0.141, "y": 0.365 },
            "products": ["ridgeline"]
          },
          {
            "page": { "kind": "confirmation", "label": "Order confirmed" },
            "message": "Saw the demo order confirmation",
            "products": ["ridgeline"]
          }
        ],
        "purchase": "ridgeline"
      }
    }
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm --filter @storefront/api test`
Expected: all PASS.

- [ ] **Step 5: Checks** — `pnpm exec vp fmt apps/api/fixtures` (formatting only; contents unchanged), `pnpm --filter @storefront/api typecheck`.

---

### Task 7: Web foundation and story derivations

**Files:**

- Modify: `apps/web/package.json` (fonts, test script)
- Modify: `apps/web/src/styles.css`
- Modify: `apps/web/src/lib/format.ts`, `apps/web/src/lib/runs.ts`, `apps/web/src/lib/runs.test.ts`, `apps/web/src/lib/api.ts`
- Modify: `apps/web/src/components/RunDetail.tsx`, `RunSteps.tsx`, `RunHistory.tsx`, `ui/button.tsx` (token rename)
- Create: `apps/web/src/lib/story.ts`, `apps/web/src/lib/useReducedMotion.ts`
- Test: `apps/web/src/lib/story.test.ts`

**Interfaces:**

- Consumes: Task 1 types.
- Produces (`lib/story.ts`): `type Arm = "baseline" | "candidate"`; `interface Pair { persona: Persona; baseline?: Run; candidate?: Run }`; `buildPairs(catalog, runs): Pair[]`; `purchased(run?): boolean`; `featured(pairs, personaId?): Pair | undefined`; `verdict(pairs): Verdict` with `{ total, before, after, better: Pair[], worse: Pair[], technical: Pair[] }`; `ledger(pairs): Ledger` with `{ rows: { id, label, before: Rate, after: Rate }[], extraSteps: Record<Arm, number | undefined>, emptySearches: Record<Arm, number>, giveUpPage: Record<Arm, string | undefined> }`, `Rate = { met, total }`; `flows(pairs, arm): { edges: { from: PageKind; to: PageKind; count: number }[]; giveUps: Map<PageKind, number> }`; `interface Frame { event: StepEvent; src?: string }`; `frames(run?): Frame[]`; `counters(list, index): { steps, searches, revisits }`; `requirementLabel(persona, id): string`. (`lib/format.ts`): `outcomeLabels: Record<Findability["outcome"], string>`; `configFields` gains `showAttributes`. (`lib/useReducedMotion.ts`): `useReducedMotion(): boolean`.

- [ ] **Step 1: Dependencies, tokens, cleanup**

```bash
cd apps/web
pnpm add @fontsource-variable/big-shoulders-display @fontsource-variable/atkinson-hyperlegible-next
pnpm remove @fontsource-variable/instrument-sans
```

Confirm the family names in `node_modules/@fontsource-variable/*/index.css` (`font-family:` lines) and use them below.

`styles.css`:

```css
@import "tailwindcss";
@import "@fontsource-variable/big-shoulders-display";
@import "@fontsource-variable/atkinson-hyperlegible-next";
@theme {
  --font-sans: "Atkinson Hyperlegible Next Variable", ui-sans-serif, system-ui, sans-serif;
  --font-display: "Big Shoulders Display Variable", ui-sans-serif, sans-serif;
  --color-fog: #ecefea;
  --color-ink: #17241d;
  --color-muted: #5e6a63;
  --color-line: #d5dad3;
  --color-graphite: #8b938d;
  --color-cobalt: #2448d6;
  --color-signal: #e2462a;
  --color-paper: #ffffff;
}
body {
  margin: 0;
  background: var(--color-fog);
  color: var(--color-ink);
  font-family: var(--font-sans);
  font-size: 16px;
  font-feature-settings: "tnum";
  -webkit-font-smoothing: antialiased;
}
button,
a {
  -webkit-tap-highlight-color: transparent;
}
:where(a, button, [role="button"], [role="slider"]):focus-visible {
  outline: 2px solid var(--color-cobalt);
  outline-offset: 2px;
}
@media (prefers-reduced-motion: reduce) {
  * {
    animation: none !important;
    transition: none !important;
    scroll-behavior: auto !important;
  }
}
```

Rename old color tokens in the kept components:

```bash
sed -i -E 's/-(pass|fail|warn)-soft\b/-fog/g; s/-pass\b/-cobalt/g; s/-fail\b/-signal/g; s/-warn\b/-muted/g; s/-canvas\b/-fog/g; s/-surface\b/-paper/g; s/-pine\b/-ink/g; s/-faint\b/-graphite/g' \
  src/components/RunDetail.tsx src/components/RunSteps.tsx src/components/RunHistory.tsx src/components/ui/button.tsx
```

Then `grep -n "canvas\|surface\|pine\|faint\|pass\b\|fail\b\|warn" src/components/RunDetail.tsx src/components/RunSteps.tsx src/components/RunHistory.tsx src/components/ui/button.tsx` and fix any leftovers by hand (object keys such as `passed:` stay).

In `RunDetail.tsx`, show the final screen instead of the first:
`const screenshot = [...run.artifacts].reverse().find((artifact) => artifact.kind === "screenshot");`

`lib/runs.ts`: delete `tally` and `paired`; delete their test in `runs.test.ts`.
`lib/api.ts`: delete `storefrontUrl` (no remaining users after Task 10) and the `VariantConfig`/`LOCAL_ORIGINS` imports it needed.
`lib/format.ts`: add to `configFields`:

```ts
  {
    key: "showAttributes",
    label: "Details on cards",
    format: (c) => (c.showAttributes ? "Shown" : "Hidden"),
  },
```

and add:

```ts
export const outcomeLabels: Record<Findability["outcome"], string> = {
  purchased: "Found it and bought it",
  abandoned: "Gave up",
  "constraint-failure": "Bought the wrong thing",
  "technical-failure": "Run broke",
  timeout: "Ran out of time",
};
```

(import `type Findability` from `@storefront/contracts`).

`package.json` test script: `"test": "node --import tsx --test src/lib/*.test.ts"`.

`lib/useReducedMotion.ts`:

```ts
import { useSyncExternalStore } from "react";

const query = "(prefers-reduced-motion: reduce)";
function subscribe(onChange: () => void) {
  const media = window.matchMedia(query);
  media.addEventListener("change", onChange);
  return () => media.removeEventListener("change", onChange);
}
export function useReducedMotion() {
  return useSyncExternalStore(subscribe, () => window.matchMedia(query).matches);
}
```

- [ ] **Step 2: Write the failing test** (`story.test.ts`)

```ts
import assert from "node:assert/strict";
import test from "node:test";
import type { Catalog, Findability, Persona, Run, StepEvent } from "@storefront/contracts";
import { buildPairs, counters, featured, flows, frames, ledger, verdict } from "./story.ts";

const task = {
  query: "navy jacket",
  budget: 150,
  requirements: [
    { id: "color", label: "Navy", kind: "hard" as const, attribute: "color", value: "navy" },
  ],
};
const persona = (id: string, split: Persona["split"], withTask = true): Persona => ({
  id,
  name: id,
  goal: "",
  split,
  traits: [],
  history: [],
  ...(withTask ? { task } : {}),
});
const catalog: Catalog = {
  experiment: {
    id: "e",
    name: "E",
    baselineVariantId: "base",
    variantIds: ["base", "cand"],
    developmentPersonaIds: ["dev", "dev2"],
    heldOutPersonaIds: ["up", "down", "broken"],
    seed: 1,
    createdAt: "2026-01-01T00:00:00Z",
    contractVersion: 1,
  },
  personas: [
    persona("legacy", "development", false),
    persona("dev", "development"),
    persona("up", "held-out"),
    persona("down", "held-out"),
    persona("broken", "held-out"),
  ],
  variants: [],
};
const page = (
  id: string,
  kind: NonNullable<StepEvent["page"]>["kind"],
  extra: Partial<StepEvent> = {},
): StepEvent => ({
  id,
  runId: "r",
  sequence: 0,
  at: "2026-01-01T00:00:00Z",
  kind: "navigation",
  message: id,
  page: { kind, label: kind },
  ...extra,
});
let clock = 0;
function run(
  personaId: string,
  variantId: string,
  outcome: Findability["outcome"] | "queued" | "failed",
  events: StepEvent[] = [],
): Run {
  clock++;
  const finished = outcome !== "queued" && outcome !== "failed";
  return {
    id: `${personaId}-${variantId}-${clock}`,
    experimentId: "e",
    baselineVariantId: "base",
    variantId,
    personaId,
    personaSplit: catalog.personas.find((p) => p.id === personaId)!.split,
    seed: 1,
    config: { headline: "h", showShipping: true, recommendationOrder: "popular" },
    persona: catalog.personas.find((p) => p.id === personaId)!,
    adapter: "mock",
    synthetic: true,
    contractVersion: 1,
    status: finished ? "completed" : outcome,
    createdAt: new Date(Date.UTC(2026, 0, 1, 0, 0, clock)).toISOString(),
    events,
    artifacts: [],
    evaluation: finished
      ? {
          runId: "r",
          synthetic: true,
          taskCompleted: outcome === "purchased",
          score: outcome === "purchased" ? 1 : 0,
          failureReasons: [],
          notes: [],
          findability: {
            outcome,
            requirements: [
              { id: "color", met: outcome === "purchased" },
              { id: "budget", met: outcome === "purchased" },
            ],
            steps: events.filter((event) => event.page).length,
            optimalSteps: 2,
            searches: events.filter((event) => event.search).length,
            emptySearches: events.filter((event) => event.search?.resultCount === 0).length,
            revisits: 0,
            abandonedAt: outcome === "abandoned" ? events.at(-1)?.id : undefined,
          },
        }
      : undefined,
    proposedChanges: [],
  };
}

const devBase = run("dev", "base", "abandoned", [
  page("h", "home"),
  page("c", "category"),
  page("p", "product"),
]);
const devCand = run("dev", "cand", "purchased", [
  page("h", "home"),
  page("c", "category"),
  page("p", "product"),
  page("k", "cart"),
]);
const runs = [
  devBase,
  devCand,
  run("dev", "cand", "queued"), // rerun in progress: must not hide the finished run
  run("legacy", "base", "purchased"),
  run("up", "base", "abandoned", [
    page("h2", "home"),
    page("s", "search", { search: { query: "x", resultCount: 0 } }),
  ]),
  run("up", "cand", "purchased", [page("h3", "home"), page("c3", "category")]),
  run("down", "base", "purchased", [page("h4", "home")]),
  run("down", "cand", "constraint-failure", [page("h5", "home")]),
  run("broken", "base", "purchased"),
  run("broken", "cand", "failed"),
];
const pairs = buildPairs(catalog, runs);

void test("pairs skip personas without a task and keep finished runs during a rerun", () => {
  assert.deepEqual(
    pairs.map((pair) => pair.persona.id),
    ["dev", "up", "down", "broken"],
  );
  assert.equal(pairs[0].candidate?.id, devCand.id);
});

void test("featured shopper is the first development shopper the fix helped", () => {
  assert.equal(featured(pairs)?.persona.id, "dev");
  assert.equal(featured(pairs, "down")?.persona.id, "down");
});

void test("verdict is held-out only, names regressions and sets technical failures aside", () => {
  const result = verdict(pairs);
  assert.equal(result.total, 2);
  assert.equal(result.before, 1);
  assert.equal(result.after, 1);
  assert.deepEqual(
    result.better.map((pair) => pair.persona.id),
    ["up"],
  );
  assert.deepEqual(
    result.worse.map((pair) => pair.persona.id),
    ["down"],
  );
  assert.deepEqual(
    result.technical.map((pair) => pair.persona.id),
    ["broken"],
  );
});

void test("ledger rates requirements and summarises effort", () => {
  const result = ledger(pairs);
  const color = result.rows.find((row) => row.id === "color")!;
  assert.deepEqual(color.before, { met: 2, total: 4 });
  assert.deepEqual(color.after, { met: 2, total: 3 });
  assert.equal(result.rows.find((row) => row.id === "budget")?.label, "Within budget");
  assert.equal(result.extraSteps.candidate, 1);
  assert.equal(result.emptySearches.baseline, 1);
  assert.equal(result.giveUpPage.baseline, "product");
});

void test("flows count shoppers per route and where they gave up", () => {
  const baseline = flows(pairs, "baseline");
  assert.equal(
    baseline.edges.find((edge) => edge.from === "home" && edge.to === "category")?.count,
    1,
  );
  assert.equal(baseline.giveUps.get("product"), 1);
  assert.equal(baseline.giveUps.get("search"), 1);
});

void test("frames fall back when a screenshot is missing and counters track revisits", () => {
  const withShot = {
    ...devBase,
    artifacts: [
      {
        id: "a",
        runId: "r",
        kind: "screenshot" as const,
        path: "x",
        mimeType: "image/svg+xml",
        createdAt: "2026-01-01T00:00:00Z",
      },
    ],
  };
  withShot.events = [
    page("1", "category", { screenshotArtifactId: "a" }),
    page("2", "category", { screenshotArtifactId: "gone" }),
  ];
  const list = frames(withShot);
  assert.equal(list[0].src, "/api/artifacts/a");
  assert.equal(list[1].src, undefined);
  assert.deepEqual(counters(list, 1), { steps: 2, searches: 0, revisits: 1 });
  assert.deepEqual(frames(undefined), []);
});
```

Ledger expectation check: color `before` counts baseline settled runs for dev (no), up (no), down (yes), broken (yes) = 2/4; `after` counts dev (yes), up (yes), down (no) = 2/3 (broken's candidate failed technically and is skipped). `extraSteps.candidate` = median of purchased candidate runs' `steps − optimalSteps`: dev 4−2=2, up 2−2=0 → median 1. `giveUpPage.baseline`: dev gave up on `product`, up on `search` → tie; first maximum in insertion order wins, which is `product`.

- [ ] **Step 3: Run test to verify it fails**

Run: `pnpm --filter @storefront/web test`
Expected: FAIL (cannot find `./story.ts`).

- [ ] **Step 4: Implement** (`lib/story.ts`)

```ts
import type { Catalog, Persona, Run, StepEvent } from "@storefront/contracts";
import { artifactUrl } from "./api";
import { cellKey, groupRuns } from "./runs";

export type Arm = "baseline" | "candidate";
export type PageKind = NonNullable<StepEvent["page"]>["kind"];
export interface Pair {
  persona: Persona;
  baseline?: Run;
  candidate?: Run;
}
const arms: Arm[] = ["baseline", "candidate"];

function finished(run: Run) {
  return run.status === "completed" || run.status === "failed";
}
function findabilityOf(run?: Run) {
  return run?.status === "completed" ? run.evaluation?.findability : undefined;
}
export function purchased(run?: Run) {
  return findabilityOf(run)?.outcome === "purchased";
}

/** Latest finished mock run per arm for every persona with a shopping task, so a rerun never blanks the story. */
export function buildPairs(catalog: Catalog, runs: Run[]): Pair[] {
  const cells = groupRuns(runs);
  const baselineId = catalog.experiment.baselineVariantId;
  const candidateId = catalog.experiment.variantIds.find((id) => id !== baselineId);
  const latest = (personaId: string, variantId?: string) =>
    variantId ? cells.get(cellKey("mock", personaId, variantId))?.find(finished) : undefined;
  return catalog.personas
    .filter((persona) => persona.task)
    .map((persona) => ({
      persona,
      baseline: latest(persona.id, baselineId),
      candidate: latest(persona.id, candidateId),
    }));
}

export function featured(pairs: Pair[], personaId?: string) {
  const development = pairs.filter((pair) => pair.persona.split === "development");
  return (
    pairs.find((pair) => pair.persona.id === personaId) ??
    development.find(
      (pair) =>
        findabilityOf(pair.baseline) && !purchased(pair.baseline) && purchased(pair.candidate),
    ) ??
    development[0]
  );
}

export interface Verdict {
  total: number;
  before: number;
  after: number;
  better: Pair[];
  worse: Pair[];
  technical: Pair[];
}

/** Held-out shoppers whose two runs both finished; technical failures are reported, never counted as giving up. */
export function verdict(pairs: Pair[]): Verdict {
  const heldOut = pairs.filter((pair) => pair.persona.split === "held-out");
  const both = heldOut.filter(
    (pair) => findabilityOf(pair.baseline) && findabilityOf(pair.candidate),
  );
  return {
    total: both.length,
    before: both.filter((pair) => purchased(pair.baseline)).length,
    after: both.filter((pair) => purchased(pair.candidate)).length,
    better: both.filter((pair) => !purchased(pair.baseline) && purchased(pair.candidate)),
    worse: both.filter((pair) => purchased(pair.baseline) && !purchased(pair.candidate)),
    technical: heldOut.filter(
      (pair) => pair.baseline?.status === "failed" || pair.candidate?.status === "failed",
    ),
  };
}

export interface Rate {
  met: number;
  total: number;
}
export interface Ledger {
  rows: { id: string; label: string; before: Rate; after: Rate }[];
  extraSteps: Record<Arm, number | undefined>;
  emptySearches: Record<Arm, number>;
  giveUpPage: Record<Arm, string | undefined>;
}

export function ledger(pairs: Pair[]): Ledger {
  const rows = new Map<string, Ledger["rows"][number]>();
  const extra: Record<Arm, number[]> = { baseline: [], candidate: [] };
  const giveUps: Record<Arm, Map<string, number>> = { baseline: new Map(), candidate: new Map() };
  const emptySearches: Record<Arm, number> = { baseline: 0, candidate: 0 };
  for (const pair of pairs) {
    for (const arm of arms) {
      const run = pair[arm];
      const findability = findabilityOf(run);
      if (!run || !findability) continue;
      for (const requirement of findability.requirements) {
        const row = rows.get(requirement.id) ?? {
          id: requirement.id,
          label: requirement.id === "budget" ? "Within budget" : capitalize(requirement.id),
          before: { met: 0, total: 0 },
          after: { met: 0, total: 0 },
        };
        const rate = arm === "baseline" ? row.before : row.after;
        rate.total++;
        if (requirement.met) rate.met++;
        rows.set(requirement.id, row);
      }
      if (findability.outcome === "purchased")
        extra[arm].push(Math.max(0, findability.steps - findability.optimalSteps));
      emptySearches[arm] += findability.emptySearches;
      const page = run.events.find((event) => event.id === findability.abandonedAt)?.page?.label;
      if (page) giveUps[arm].set(page, (giveUps[arm].get(page) ?? 0) + 1);
    }
  }
  return {
    rows: [...rows.values()],
    extraSteps: { baseline: median(extra.baseline), candidate: median(extra.candidate) },
    emptySearches,
    giveUpPage: {
      baseline: mostCommon(giveUps.baseline),
      candidate: mostCommon(giveUps.candidate),
    },
  };
}

/** Shoppers per route between page kinds, and where they gave up. */
export function flows(pairs: Pair[], arm: Arm) {
  const edges = new Map<string, { from: PageKind; to: PageKind; shoppers: Set<string> }>();
  const giveUps = new Map<PageKind, number>();
  for (const pair of pairs) {
    const run = pair[arm];
    const findability = findabilityOf(run);
    if (!run || !findability) continue;
    let previous: PageKind | undefined;
    for (const event of run.events) {
      const kind = event.page?.kind;
      if (!kind) continue;
      if (previous && previous !== kind) {
        const key = `${previous}>${kind}`;
        const edge = edges.get(key) ?? { from: previous, to: kind, shoppers: new Set<string>() };
        edge.shoppers.add(pair.persona.id);
        edges.set(key, edge);
      }
      if (event.id === findability.abandonedAt) giveUps.set(kind, (giveUps.get(kind) ?? 0) + 1);
      previous = kind;
    }
  }
  return {
    edges: [...edges.values()].map(({ from, to, shoppers }) => ({
      from,
      to,
      count: shoppers.size,
    })),
    giveUps,
  };
}

export interface Frame {
  event: StepEvent;
  src?: string;
}
export function frames(run?: Run): Frame[] {
  if (!run) return [];
  const saved = new Set(run.artifacts.map((artifact) => artifact.id));
  return run.events
    .filter((event) => event.page)
    .map((event) => ({
      event,
      src:
        event.screenshotArtifactId && saved.has(event.screenshotArtifactId)
          ? artifactUrl(event.screenshotArtifactId)
          : undefined,
    }));
}

/** Running totals up to and including step `index`; revisits match the evaluator (same page kind and label). */
export function counters(list: Frame[], index: number) {
  const seen = new Set<string>();
  let searches = 0;
  let revisits = 0;
  for (const { event } of list.slice(0, index + 1)) {
    if (event.search) searches++;
    const key = `${event.page?.kind}:${event.page?.label ?? ""}`;
    if (seen.has(key)) revisits++;
    seen.add(key);
  }
  return { steps: Math.min(index + 1, list.length), searches, revisits };
}

export function requirementLabel(persona: Persona, id: string) {
  if (id === "budget") return `Under $${persona.task?.budget}`;
  return persona.task?.requirements.find((requirement) => requirement.id === id)?.label ?? id;
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
function median(values: number[]) {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
function mostCommon(counts: Map<string, number>) {
  let best: string | undefined;
  for (const [key, count] of counts)
    if (best === undefined || count > counts.get(best)!) best = key;
  return best;
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pnpm --filter @storefront/web test && pnpm --filter @storefront/web typecheck`
Expected: PASS. (Typecheck may still flag old components that reference deleted helpers; those files are deleted in Task 10. If so, delete `OutcomeMatrix.tsx`, `OutcomeCell.tsx`, `ConfigDiff.tsx`, `SuggestionList.tsx`, `AppHeader.tsx` now and stub `App.tsx` to `export function App() { return null; }` until Task 10.)

- [ ] **Step 6: Checks** — `pnpm exec vp fmt apps/web/src`.

---

### Task 8: Opening and race

**Files:**

- Create: `apps/web/src/story/Opening.tsx`, `apps/web/src/story/Race.tsx`, `apps/web/src/story/RaceLane.tsx`, `apps/web/src/story/RaceFrame.tsx`

**Interfaces:**

- Consumes: `Pair`, `Frame`, `frames`, `counters`, `requirementLabel` (Task 7), `outcomeLabels`, `useReducedMotion`, `Button`, `cn`.
- Produces: `<Opening persona />`, `<Race pair busy />` (section `id="race"`), `<RaceLane title tone frames index run persona />`, `<RaceFrame frame? className? />`. Every story section root carries `data-beat`.

- [ ] **Step 1: `RaceFrame.tsx`**

```tsx
import { cn } from "../lib/utils";
import type { Frame } from "../lib/story";

export function RaceFrame({ frame, className }: { frame?: Frame; className?: string }) {
  const click = frame?.event.click;
  return (
    <figure className={className}>
      <div className="relative aspect-[8/5] overflow-hidden rounded-md border border-line bg-paper">
        {frame?.src ? (
          <img src={frame.src} alt={frame.event.message} className="size-full object-cover" />
        ) : (
          <p className="grid size-full place-items-center text-muted">
            No screenshot for this step
          </p>
        )}
        {click && (
          <span
            aria-hidden
            className="absolute size-7 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-ink bg-ink/15"
            style={{ left: `${click.x * 100}%`, top: `${click.y * 100}%` }}
          />
        )}
      </div>
      <figcaption className={cn("mt-2 min-h-12 text-muted", !frame && "invisible")}>
        {frame?.event.message}
      </figcaption>
    </figure>
  );
}
```

- [ ] **Step 2: `RaceLane.tsx`**

```tsx
import type { Persona, Run } from "@storefront/contracts";
import { Check, X } from "lucide-react";
import { outcomeLabels } from "../lib/format";
import { counters, requirementLabel, type Frame } from "../lib/story";
import { cn } from "../lib/utils";
import { RaceFrame } from "./RaceFrame";

export function RaceLane({
  title,
  tone,
  frames,
  index,
  run,
  persona,
}: {
  title: string;
  tone: "graphite" | "cobalt";
  frames: Frame[];
  index: number;
  run?: Run;
  persona: Persona;
}) {
  const at = Math.min(index, frames.length - 1);
  const findability = index >= frames.length - 1 ? run?.evaluation?.findability : undefined;
  const totals = counters(frames, at);
  return (
    <div className="min-w-0">
      <h2
        className={cn(
          "font-display text-4xl font-bold",
          tone === "cobalt" ? "text-cobalt" : "text-graphite",
        )}
      >
        {title}
      </h2>
      <RaceFrame frame={frames[at]} className="mt-3" />
      <dl className="mt-2 flex gap-8">
        {(
          [
            ["Steps", totals.steps],
            ["Searches", totals.searches],
            ["Pages revisited", totals.revisits],
          ] as const
        ).map(([label, value]) => (
          <div key={label}>
            <dt className="text-sm text-muted">{label}</dt>
            <dd className="font-display text-5xl font-bold">{value}</dd>
          </div>
        ))}
      </dl>
      {findability && (
        <div className="mt-4">
          <p
            className={cn(
              "text-xl font-bold",
              findability.outcome === "purchased" ? "text-cobalt" : "text-signal",
            )}
          >
            {outcomeLabels[findability.outcome]}
          </p>
          {findability.outcome === "abandoned" ? (
            <p className="mt-1 max-w-[50ch] text-muted">{run?.evaluation?.failureReasons[0]}</p>
          ) : (
            <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
              {findability.requirements.map((requirement) => (
                <li
                  key={requirement.id}
                  className={cn("flex items-center gap-1", !requirement.met && "text-signal")}
                >
                  {requirement.met ? <Check className="size-4" /> : <X className="size-4" />}
                  {requirementLabel(persona, requirement.id)}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: `Race.tsx`**

```tsx
import { Slider } from "@base-ui/react/slider";
import { Pause, Play } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Button } from "../components/ui/button";
import { frames, type Pair } from "../lib/story";
import { useReducedMotion } from "../lib/useReducedMotion";
import { RaceLane } from "./RaceLane";

const STEP_MS = 1100;

export function Race({ pair, busy }: { pair: Pair; busy: boolean }) {
  const baseline = frames(pair.baseline);
  const candidate = frames(pair.candidate);
  const last = Math.max(baseline.length, candidate.length) - 1;
  const reduced = useReducedMotion();
  const [index, setIndex] = useState(() => (reduced ? Math.max(last, 0) : 0));
  const [playing, setPlaying] = useState(false);
  const section = useRef<HTMLElement>(null);
  const running = playing && index < last;

  // Play once when the race first scrolls into view.
  useEffect(() => {
    const element = section.current;
    if (!element || reduced) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        setPlaying(true);
        observer.disconnect();
      },
      { threshold: 0.5 },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [reduced]);

  useEffect(() => {
    if (!running) return;
    const timer = window.setTimeout(() => setIndex(index + 1), STEP_MS);
    return () => window.clearTimeout(timer);
  }, [running, index]);

  if (last < 0)
    return (
      <section id="race" data-beat className="py-16 text-lg text-muted">
        No runs yet. Run the shoppers from the bar below.
      </section>
    );

  return (
    <section id="race" ref={section} data-beat className="py-16">
      <div className="grid gap-10 md:grid-cols-2">
        <RaceLane
          title="Today's shop"
          tone="graphite"
          frames={baseline}
          index={index}
          run={pair.baseline}
          persona={pair.persona}
        />
        <RaceLane
          title="With the fix"
          tone="cobalt"
          frames={candidate}
          index={index}
          run={pair.candidate}
          persona={pair.persona}
        />
      </div>
      <div className="mt-8 flex items-center gap-4">
        <Button
          variant="outline"
          size="sm"
          aria-label={running ? "Pause" : "Play"}
          onClick={() => {
            if (running) return setPlaying(false);
            if (index >= last) setIndex(0);
            setPlaying(true);
          }}
        >
          {running ? <Pause /> : <Play />}
        </Button>
        <Slider.Root
          value={index}
          min={0}
          max={last}
          onValueChange={(value) => {
            setPlaying(false);
            setIndex(Array.isArray(value) ? value[0] : value);
          }}
          className="flex-1"
        >
          <Slider.Control className="flex h-8 items-center">
            <Slider.Track className="h-1 w-full rounded bg-line">
              <Slider.Indicator className="rounded bg-ink" />
              <Slider.Thumb aria-label="Step" className="size-4 rounded-full bg-ink" />
            </Slider.Track>
          </Slider.Control>
        </Slider.Root>
        <span className="text-sm whitespace-nowrap text-muted">
          Step {index + 1} of {last + 1}
        </span>
      </div>
      {busy && (
        <p role="status" className="mt-3 text-sm text-muted">
          Running the shoppers again. New results replace these as they finish.
        </p>
      )}
    </section>
  );
}
```

- [ ] **Step 4: `Opening.tsx`**

```tsx
import type { Persona } from "@storefront/contracts";
import { useEffect, useState } from "react";
import { useReducedMotion } from "../lib/useReducedMotion";

const TYPE_MS = 45;

export function Opening({ persona }: { persona: Persona }) {
  const query = persona.task?.query ?? persona.goal;
  const reduced = useReducedMotion();
  const [typed, setTyped] = useState(0);
  useEffect(() => {
    if (reduced || typed >= query.length) return;
    const timer = window.setTimeout(() => setTyped(typed + 1), TYPE_MS);
    return () => window.clearTimeout(timer);
  }, [typed, query, reduced]);
  return (
    <section data-beat className="flex min-h-[70vh] flex-col justify-center py-16">
      <p className="text-xl text-muted">{persona.name} is looking for</p>
      <h1
        aria-label={query}
        className="mt-3 max-w-[16ch] font-display text-6xl leading-[0.95] font-bold sm:text-8xl"
      >
        <span aria-hidden>{reduced ? query : query.slice(0, typed)}</span>
        <span
          aria-hidden
          className="ml-1 inline-block h-[0.8em] w-[0.08em] animate-pulse bg-cobalt align-baseline"
        />
      </h1>
      <p className="mt-8 max-w-[50ch] text-xl">
        Two versions of the same shop. Watch which one gets them there.
      </p>
    </section>
  );
}
```

- [ ] **Step 5: Checks** — `pnpm --filter @storefront/web typecheck` (verify the Base UI `Slider.Root` `onValueChange` value type; adjust the `Array.isArray` guard only if the type differs), `pnpm exec vp fmt apps/web/src`.

---

### Task 9: Snag, fix and shop map

**Files:**

- Create: `apps/web/src/story/Snag.tsx`, `apps/web/src/story/Fix.tsx`, `apps/web/src/story/ShopMap.tsx`

**Interfaces:**

- Consumes: `Pair`, `frames`, `flows`, `purchased`, `Arm`, `PageKind` (Task 7), `RaceFrame` (Task 8), `configFields`, `Button`, Base UI `Toggle`/`ToggleGroup`.
- Produces: `<Snag pair onEvidence={(run: Run) => void} />`, `<Fix pair />`, `<ShopMap pairs featuredId onSelect={(personaId: string) => void} />`.

- [ ] **Step 1: `Snag.tsx`**

```tsx
import type { Run } from "@storefront/contracts";
import { Button } from "../components/ui/button";
import { frames, type Pair } from "../lib/story";
import { RaceFrame } from "./RaceFrame";

export function Snag({ pair, onEvidence }: { pair: Pair; onEvidence: (run: Run) => void }) {
  const run = pair.baseline;
  const findability = run?.evaluation?.findability;
  if (!run || findability?.outcome !== "abandoned") return null;
  const list = frames(run);
  const stuck = list.find((frame) => frame.event.id === findability.abandonedAt) ?? list.at(-1);
  return (
    <section data-beat className="grid items-center gap-10 py-16 md:grid-cols-[3fr_2fr]">
      <RaceFrame frame={stuck} />
      <div>
        <h2 className="font-display text-5xl font-bold sm:text-6xl">Where it went wrong</h2>
        <p className="mt-4 max-w-[45ch] text-xl">{run.evaluation?.failureReasons[0]}</p>
        <Button variant="outline" className="mt-6" onClick={() => onEvidence(run)}>
          See every step
        </Button>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: `Fix.tsx`**

```tsx
import type { Run } from "@storefront/contracts";
import { configFields } from "../lib/format";
import { frames, type Pair } from "../lib/story";
import { RaceFrame } from "./RaceFrame";

const listing = (run?: Run) =>
  frames(run).find(
    (frame) => frame.event.page?.kind === "category" || frame.event.page?.kind === "search",
  );

export function Fix({ pair }: { pair: Pair }) {
  const proposal = pair.baseline?.proposedChanges[0];
  if (!pair.baseline || !proposal) return null;
  const before = pair.baseline.config;
  const after = { ...before, ...proposal.configPatch };
  return (
    <section data-beat className="py-16">
      <h2 className="font-display text-5xl font-bold sm:text-6xl">The fix</h2>
      <p className="mt-4 max-w-[60ch] text-xl">{proposal.rationale}</p>
      <ul className="mt-3 text-muted">
        {configFields
          .filter((field) => field.key in proposal.configPatch)
          .map((field) => (
            <li key={field.key}>
              {field.label}: {field.format(before)} to {field.format(after)}
            </li>
          ))}
      </ul>
      <div className="mt-8 grid gap-8 md:grid-cols-2">
        <div>
          <h3 className="mb-2 font-bold text-graphite">Today's shop</h3>
          <RaceFrame frame={listing(pair.baseline)} />
        </div>
        <div>
          <h3 className="mb-2 font-bold text-cobalt">With the fix</h3>
          <RaceFrame frame={listing(pair.candidate)} />
        </div>
      </div>
      <p className="mt-4 text-muted">
        Uses only facts already in the catalog. No new claims, discounts or reviews.
      </p>
    </section>
  );
}
```

- [ ] **Step 3: `ShopMap.tsx`**

```tsx
import { Toggle } from "@base-ui/react/toggle";
import { ToggleGroup } from "@base-ui/react/toggle-group";
import { Check, X } from "lucide-react";
import { useMemo, useState } from "react";
import { flows, purchased, type Arm, type Pair, type PageKind } from "../lib/story";
import { cn } from "../lib/utils";

type Station = { x: number; y: number; label: string };
const stations: Partial<Record<PageKind, Station>> = {
  home: { x: 60, y: 150, label: "Home" },
  category: { x: 240, y: 70, label: "Category" },
  search: { x: 240, y: 230, label: "Search" },
  product: { x: 430, y: 150, label: "Product" },
  cart: { x: 600, y: 150, label: "Cart" },
  checkout: { x: 750, y: 150, label: "Checkout" },
  confirmation: { x: 900, y: 150, label: "Bought" },
};
const armLabels: Record<Arm, string> = { baseline: "Today's shop", candidate: "With the fix" };

// Bend each route to its left so a return trip (b to a) never sits on top of a to b.
function curve(from: Station, to: Station) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy) || 1;
  const cx = (from.x + to.x) / 2 - (dy / length) * 26;
  const cy = (from.y + to.y) / 2 + (dx / length) * 26;
  return `M${from.x},${from.y} Q${cx},${cy} ${to.x},${to.y}`;
}

export function ShopMap({
  pairs,
  featuredId,
  onSelect,
}: {
  pairs: Pair[];
  featuredId: string;
  onSelect: (personaId: string) => void;
}) {
  const [arm, setArm] = useState<Arm>("baseline");
  const { edges, giveUps } = useMemo(() => flows(pairs, arm), [pairs, arm]);
  const stroke = arm === "baseline" ? "var(--color-graphite)" : "var(--color-cobalt)";
  return (
    <section data-beat className="py-16">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <h2 className="font-display text-5xl font-bold sm:text-6xl">Every shopper's route</h2>
        <ToggleGroup
          value={[arm]}
          onValueChange={(value) => value[0] && setArm(value[0] as Arm)}
          className="flex rounded-md bg-paper p-1"
        >
          {(["baseline", "candidate"] as const).map((option) => (
            <Toggle
              key={option}
              value={option}
              className="h-9 rounded px-3 text-muted data-[pressed]:bg-ink data-[pressed]:text-paper"
            >
              {armLabels[option]}
            </Toggle>
          ))}
        </ToggleGroup>
      </div>
      <svg
        viewBox="0 0 960 290"
        className="mt-8 w-full"
        role="img"
        aria-label={`Shopper routes through ${armLabels[arm].toLowerCase()}`}
      >
        {edges.map((edge) => {
          const from = stations[edge.from];
          const to = stations[edge.to];
          return from && to ? (
            <path
              key={`${edge.from}-${edge.to}`}
              d={curve(from, to)}
              fill="none"
              stroke={stroke}
              strokeOpacity={0.8}
              strokeWidth={2 + edge.count * 3}
              strokeLinecap="round"
            />
          ) : null;
        })}
        {Object.entries(stations).map(([kind, station]) => {
          const gaveUp = giveUps.get(kind as PageKind);
          return (
            <g key={kind}>
              <circle
                cx={station.x}
                cy={station.y}
                r={11}
                fill="var(--color-paper)"
                stroke="var(--color-ink)"
                strokeWidth={3}
              />
              <text
                x={station.x}
                y={station.y + 36}
                textAnchor="middle"
                fill="var(--color-ink)"
                fontSize={16}
              >
                {station.label}
              </text>
              {gaveUp && (
                <g color="var(--color-signal)">
                  <X x={station.x + 12} y={station.y - 36} width={20} height={20} strokeWidth={3} />
                  <text
                    x={station.x + 34}
                    y={station.y - 20}
                    fill="var(--color-signal)"
                    fontSize={18}
                    fontWeight={700}
                  >
                    {gaveUp}
                  </text>
                </g>
              )}
            </g>
          );
        })}
      </svg>
      <p className="mt-2 text-sm text-muted">
        Thicker lines carry more shoppers. Pick a shopper to replay their race.
      </p>
      <ul className="mt-4 flex flex-wrap gap-2">
        {pairs.map((pair) => {
          const selected = pair.persona.id === featuredId;
          return (
            <li key={pair.persona.id}>
              <button
                type="button"
                aria-pressed={selected}
                onClick={() => onSelect(pair.persona.id)}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1 text-sm hover:border-ink",
                  selected && "border-ink bg-ink text-paper",
                )}
              >
                {purchased(pair[arm]) ? (
                  <Check className={cn("size-3.5", !selected && "text-cobalt")} />
                ) : (
                  <X className={cn("size-3.5", !selected && "text-signal")} />
                )}
                {pair.persona.name}
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
```

- [ ] **Step 4: Checks** — `pnpm --filter @storefront/web typecheck`, `pnpm exec vp fmt apps/web/src`.

---

### Task 10: Ledger, verdict, fine print, presenter bar, evidence drawer, and App

**Files:**

- Create: `apps/web/src/story/Ledger.tsx`, `Verdict.tsx`, `FinePrint.tsx`, `PresenterBar.tsx`, `EvidenceDrawer.tsx`
- Modify: `apps/web/src/App.tsx` (full replacement)
- Delete: `apps/web/src/components/OutcomeMatrix.tsx`, `OutcomeCell.tsx`, `ConfigDiff.tsx`, `SuggestionList.tsx`, `AppHeader.tsx`

**Interfaces:**

- Consumes: everything from Tasks 7–9; `RunDetail` (existing).
- Produces: the finished page.

- [ ] **Step 1: `Ledger.tsx`**

```tsx
import { useMemo, type ReactNode } from "react";
import { ledger, type Pair, type Rate } from "../lib/story";

const rate = (value: Rate) => `${value.met} of ${value.total}`;

export function Ledger({ pairs }: { pairs: Pair[] }) {
  const data = useMemo(() => ledger(pairs), [pairs]);
  if (data.rows.length === 0) return null;
  const rows: [string, ReactNode, ReactNode][] = [
    ...data.rows.map((row): [string, ReactNode, ReactNode] => [
      row.label,
      rate(row.before),
      rate(row.after),
    ]),
    [
      "Extra steps for buyers, median",
      data.extraSteps.baseline ?? "None",
      data.extraSteps.candidate ?? "None",
    ],
    ["Searches with no results", data.emptySearches.baseline, data.emptySearches.candidate],
    [
      "Most common give-up page",
      data.giveUpPage.baseline ?? "None",
      data.giveUpPage.candidate ?? "None",
    ],
  ];
  return (
    <section data-beat className="py-16">
      <h2 className="font-display text-5xl font-bold sm:text-6xl">How easy it was to find</h2>
      <table className="mt-8 w-full max-w-3xl text-left text-lg">
        <thead>
          <tr className="text-sm text-muted">
            <th className="pb-2 font-normal">Every shopper, requirement met</th>
            <th className="pb-2 font-normal">Today's shop</th>
            <th className="pb-2 font-normal">With the fix</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, before, after]) => (
            <tr key={label} className="border-t border-line">
              <th scope="row" className="py-3 pr-4 font-normal">
                {label}
              </th>
              <td className="py-3 pr-4 text-graphite">{before}</td>
              <td className="py-3 font-bold text-cobalt">{after}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
```

- [ ] **Step 2: `Verdict.tsx`**

```tsx
import type { Run } from "@storefront/contracts";
import { ArrowRight } from "lucide-react";
import { useMemo } from "react";
import { verdict, type Pair } from "../lib/story";

function Names({ pairs, onEvidence }: { pairs: Pair[]; onEvidence: (run: Run) => void }) {
  return pairs.map((pair, i) => (
    <span key={pair.persona.id}>
      {i > 0 && ", "}
      <button
        type="button"
        className="underline underline-offset-4"
        onClick={() => pair.candidate && onEvidence(pair.candidate)}
      >
        {pair.persona.name}
      </button>
    </span>
  ));
}

export function Verdict({ pairs, onEvidence }: { pairs: Pair[]; onEvidence: (run: Run) => void }) {
  const result = useMemo(() => verdict(pairs), [pairs]);
  if (result.total === 0) return null;
  return (
    <section data-beat className="py-24">
      <p className="flex flex-wrap items-center gap-x-6 font-display text-7xl leading-none font-bold sm:text-9xl">
        <span className="text-graphite">
          {result.before} of {result.total}
        </span>
        <ArrowRight className="size-[0.6em]" aria-label="to" />
        <span className="text-cobalt">
          {result.after} of {result.total}
        </span>
      </p>
      <p className="mt-6 max-w-[36ch] text-2xl">
        new shoppers found and bought what they came for.
      </p>
      <p className="mt-2 max-w-[60ch] text-muted">
        None of these shoppers were used to find or choose the fix.
      </p>
      <div className="mt-8 space-y-2 text-lg">
        {result.better.length > 0 && (
          <p>
            Better with the fix: <Names pairs={result.better} onEvidence={onEvidence} />
          </p>
        )}
        {result.worse.length > 0 && (
          <p className="text-signal">
            Worse with the fix: <Names pairs={result.worse} onEvidence={onEvidence} />
          </p>
        )}
        {result.technical.length > 0 && (
          <p className="text-muted">
            {result.technical.length} shopper runs broke for technical reasons and are not counted.
          </p>
        )}
      </div>
    </section>
  );
}
```

- [ ] **Step 3: `FinePrint.tsx`**

```tsx
import type { Experiment } from "@storefront/contracts";

export function FinePrint({ experiment }: { experiment: Experiment }) {
  return (
    <footer data-beat className="border-t border-line pt-10 pb-28 text-sm text-muted">
      <p className="max-w-[68ch]">
        Synthetic results: scripted mock shoppers moving through drawn page frames, checked against
        catalog facts by a deterministic evaluator. They show how the loop works, not how real
        people shop or how revenue changes.
      </p>
      <p className="mt-2">
        {experiment.name}, seed {experiment.seed}, contract version {experiment.contractVersion}
      </p>
    </footer>
  );
}
```

- [ ] **Step 4: `PresenterBar.tsx`**

```tsx
import { RotateCcw } from "lucide-react";
import { useEffect } from "react";
import { Button } from "../components/ui/button";

// Arrow keys scroll to the next or previous story section; the page stays a scroll, not slides.
function onArrow(event: KeyboardEvent) {
  if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
  if (
    event.target instanceof HTMLElement &&
    event.target.closest("input, textarea, [role=slider], [role=dialog]")
  )
    return;
  const beats = [...document.querySelectorAll<HTMLElement>("[data-beat]")];
  const tops = beats.map((beat) => beat.getBoundingClientRect().top);
  const target =
    event.key === "ArrowRight"
      ? beats.find((_, i) => tops[i] > 8)
      : [...beats].reverse().find((beat) => beat.getBoundingClientRect().top < -8);
  if (!target) return;
  event.preventDefault();
  target.scrollIntoView({ behavior: "smooth", block: "start" });
}

export function PresenterBar({
  busy,
  hasRuns,
  onRun,
}: {
  busy: boolean;
  hasRuns: boolean;
  onRun: () => void;
}) {
  useEffect(() => {
    window.addEventListener("keydown", onArrow);
    return () => window.removeEventListener("keydown", onArrow);
  }, []);
  return (
    <div className="fixed inset-x-0 bottom-0 z-10 border-t border-line bg-fog/90 backdrop-blur print:hidden">
      <div className="mx-auto flex max-w-6xl items-center gap-4 px-4 py-2 text-sm sm:px-8">
        <span className="font-bold">Synthetic results</span>
        <span className="hidden text-muted sm:inline">Arrow keys move between sections</span>
        <Button size="sm" className="ml-auto" disabled={busy} onClick={onRun}>
          <RotateCcw />
          {busy ? "Running shoppers" : hasRuns ? "Run shoppers again" : "Run shoppers"}
        </Button>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: `EvidenceDrawer.tsx`**

```tsx
import { Dialog } from "@base-ui/react/dialog";
import type { Run } from "@storefront/contracts";
import { X } from "lucide-react";
import { RunDetail } from "../components/RunDetail";

const noop = () => {};

export function EvidenceDrawer({
  run,
  variantName,
  onClose,
}: {
  run?: Run;
  variantName: string;
  onClose: () => void;
}) {
  return (
    <Dialog.Root open={run !== undefined} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 z-20 bg-ink/30" />
        <Dialog.Popup className="fixed inset-y-0 right-0 z-30 w-full max-w-md overflow-y-auto bg-fog p-4 shadow-xl">
          <div className="mb-3 flex items-center justify-between">
            <Dialog.Title className="font-display text-3xl font-bold">Evidence</Dialog.Title>
            <Dialog.Close
              aria-label="Close"
              className="rounded-md p-1.5 text-muted hover:bg-paper hover:text-ink"
            >
              <X className="size-5" />
            </Dialog.Close>
          </div>
          {run && <RunDetail run={run} variantName={variantName} history={[]} onSelectRun={noop} />}
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
```

- [ ] **Step 6: `App.tsx`** (replace the whole file) and delete the old components

```tsx
import {
  CatalogSchema,
  RunSchema,
  type Catalog,
  type Run,
  type RunRequest,
} from "@storefront/contracts";
import { useEffect, useMemo, useState } from "react";
import { api } from "./lib/api";
import { buildPairs, featured } from "./lib/story";
import { EvidenceDrawer } from "./story/EvidenceDrawer";
import { FinePrint } from "./story/FinePrint";
import { Fix } from "./story/Fix";
import { Ledger } from "./story/Ledger";
import { Opening } from "./story/Opening";
import { PresenterBar } from "./story/PresenterBar";
import { Race } from "./story/Race";
import { ShopMap } from "./story/ShopMap";
import { Snag } from "./story/Snag";
import { Verdict } from "./story/Verdict";

const message = (cause: unknown) =>
  cause instanceof Error ? cause.message : "Could not reach the local API.";

export function App() {
  const [catalog, setCatalog] = useState<Catalog>();
  const [runs, setRuns] = useState<Run[]>([]);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [featuredId, setFeaturedId] = useState<string>();
  const [evidence, setEvidence] = useState<Run>();
  const busy =
    submitting || runs.some((run) => run.status === "queued" || run.status === "running");

  // Poll fast only while shoppers are running.
  useEffect(() => {
    let cancelled = false;
    async function refresh() {
      try {
        const [catalogData, runData] = await Promise.all([
          catalog ? undefined : api("/catalog"),
          api("/runs"),
        ]);
        if (cancelled) return;
        if (catalogData) setCatalog(CatalogSchema.parse(catalogData));
        setRuns(RunSchema.array().parse(runData));
        setError("");
      } catch (cause) {
        if (!cancelled) setError(message(cause));
      }
    }
    void refresh();
    const timer = window.setInterval(() => void refresh(), busy ? 1000 : 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [catalog, busy]);

  const pairs = useMemo(() => (catalog ? buildPairs(catalog, runs) : []), [catalog, runs]);
  const feature = featured(pairs, featuredId);

  async function runShoppers() {
    if (!catalog) return;
    const requests: RunRequest[] = catalog.experiment.variantIds.flatMap((variantId) =>
      catalog.personas
        .filter((persona) => persona.task)
        .map((persona) => ({
          experimentId: catalog.experiment.id,
          variantId,
          personaId: persona.id,
          adapter: "mock",
        })),
    );
    setSubmitting(true);
    setError("");
    try {
      await Promise.all(
        requests.map(async (request) => {
          const run = RunSchema.parse(await api("/runs", request));
          setRuns((current) => [run, ...current]);
        }),
      );
    } catch (cause) {
      setError(message(cause));
    } finally {
      setSubmitting(false);
    }
  }

  function replay(personaId: string) {
    setFeaturedId(personaId);
    document.getElementById("race")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <>
      <main className="mx-auto max-w-6xl px-4 sm:px-8">
        {error && (
          <p role="alert" className="mt-6 rounded-md bg-paper px-4 py-3 text-signal">
            {error}
          </p>
        )}
        {!catalog ? (
          <p className="py-24 text-lg text-muted">Loading the study</p>
        ) : !feature ? (
          <p className="py-24 text-lg">This study has no shoppers with a goal yet.</p>
        ) : (
          <>
            <Opening key={feature.persona.id} persona={feature.persona} />
            <Race
              key={`${feature.baseline?.id}-${feature.candidate?.id}`}
              pair={feature}
              busy={busy}
            />
            <Snag pair={feature} onEvidence={setEvidence} />
            <Fix pair={feature} />
            <ShopMap pairs={pairs} featuredId={feature.persona.id} onSelect={replay} />
            <Ledger pairs={pairs} />
            <Verdict pairs={pairs} onEvidence={setEvidence} />
            <FinePrint experiment={catalog.experiment} />
          </>
        )}
      </main>
      <PresenterBar busy={busy} hasRuns={runs.length > 0} onRun={() => void runShoppers()} />
      <EvidenceDrawer
        run={evidence}
        variantName={
          catalog?.variants.find((variant) => variant.id === evidence?.variantId)?.name ?? ""
        }
        onClose={() => setEvidence(undefined)}
      />
    </>
  );
}
```

```bash
git rm -q apps/web/src/components/OutcomeMatrix.tsx apps/web/src/components/OutcomeCell.tsx \
  apps/web/src/components/ConfigDiff.tsx apps/web/src/components/SuggestionList.tsx apps/web/src/components/AppHeader.tsx
```

(Files that were never committed: use `rm` instead of `git rm`.)

- [ ] **Step 7: Checks**

Run: `pnpm --filter @storefront/web typecheck && pnpm --filter @storefront/web test && pnpm exec vp fmt apps/web/src && pnpm check`
Expected: PASS (the pre-existing `experiments/ranker/dataset/schema.md` format warning is out of scope).

---

### Task 11: Browser test and visual verification

**Files:**

- Modify: `tests/local-demo.spec.ts`

**Interfaces:**

- Consumes: the finished page and API.

- [ ] **Step 1: Replace the dashboard test**

```ts
test("case file shows the race and a held-out result after running shoppers", async ({
  page,
  request,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Run shoppers/ }).click();
  await expect
    .poll(
      async () => {
        const runs = await (await request.get("/api/runs")).json();
        return runs.filter(
          (run: { adapter: string; status: string }) =>
            run.adapter === "mock" && run.status === "completed",
        ).length;
      },
      { timeout: 30_000 },
    )
    .toBeGreaterThanOrEqual(20);
  await expect(page.getByText("new shoppers found and bought what they came for.")).toBeVisible({
    timeout: 10_000,
  });
  await expect(page.locator("#race img").first()).toBeVisible();
  await page.getByRole("button", { name: "See every step" }).click();
  await expect(page.getByRole("dialog")).toContainText("Task failed");
});
```

- [ ] **Step 2: Verify visually**

Restart the API so the new catalog fixture loads (stop and rerun `pnpm dev`; the catalog upsert applies on open). Open `http://dubhacks.localhost:1356`, click "Run shoppers", and screenshot at 1440×900 and 390×844. Check: no horizontal scroll at 390 (`document.documentElement.scrollWidth === 390`), race autoplays, shop map toggles, evidence drawer opens, verdict reads "2 of 5 → 4 of 5" with Hurried shopper named as worse.

- [ ] **Step 3: Run the Playwright suite**

Stop `pnpm dev` first (`reuseExistingServer: false`), then `pnpm test:smoke`.
Expected: PASS.

- [ ] **Step 4: Final checks** — `pnpm typecheck && pnpm test && pnpm check`.
