// Package contracts mirrors the frozen v1 JSON schemas in packages/contracts.
// Change the shared schemas first; keep HTTP shapes compatible across languages.
package contracts

type Persona struct {
	ID      string        `json:"id"`
	Name    string        `json:"name"`
	Goal    string        `json:"goal"`
	Split   string        `json:"split"`
	Traits  []string      `json:"traits"`
	History []string      `json:"history"`
	Bio     string        `json:"bio,omitempty"`
	Task    *ShoppingTask `json:"task,omitempty"`
}
type VariantConfig struct {
	Headline            string `json:"headline"`
	ShowShipping        bool   `json:"showShipping"`
	RecommendationOrder string `json:"recommendationOrder"`
	ShowAttributes      bool   `json:"showAttributes,omitempty"`
}
type Variant struct {
	ID           string        `json:"id"`
	ExperimentID string        `json:"experimentId"`
	Name         string        `json:"name"`
	IsBaseline   bool          `json:"isBaseline"`
	Config       VariantConfig `json:"config"`
}
type Experiment struct {
	ID                    string   `json:"id"`
	Name                  string   `json:"name"`
	BaselineVariantID     string   `json:"baselineVariantId"`
	VariantIDs            []string `json:"variantIds"`
	DevelopmentPersonaIDs []string `json:"developmentPersonaIds"`
	HeldOutPersonaIDs     []string `json:"heldOutPersonaIds"`
	Seed                  int      `json:"seed"`
	CreatedAt             string   `json:"createdAt"`
	ContractVersion       int      `json:"contractVersion"`
}
type Catalog struct {
	Experiment Experiment `json:"experiment"`
	Personas   []Persona  `json:"personas"`
	Variants   []Variant  `json:"variants"`
}
type StepEvent struct {
	ID                   string  `json:"id"`
	RunID                string  `json:"runId"`
	Sequence             int     `json:"sequence"`
	At                   string  `json:"at"`
	Kind                 string  `json:"kind"`
	Message              string  `json:"message"`
	Page                 *Page   `json:"page,omitempty"`
	Click                *Point  `json:"click,omitempty"`
	ScreenshotArtifactID string  `json:"screenshotArtifactId,omitempty"`
	Search               *Search `json:"search,omitempty"`
}
type Artifact struct {
	ID        string `json:"id"`
	RunID     string `json:"runId"`
	Kind      string `json:"kind"`
	Path      string `json:"path"`
	MimeType  string `json:"mimeType"`
	CreatedAt string `json:"createdAt"`
}
type EvaluationResult struct {
	RunID          string       `json:"runId"`
	Synthetic      bool         `json:"synthetic"`
	TaskCompleted  bool         `json:"taskCompleted"`
	Score          float64      `json:"score"`
	FailureReasons []string     `json:"failureReasons"`
	Notes          []string     `json:"notes"`
	Findability    *Findability `json:"findability,omitempty"`
}
type ProposedChange struct {
	ID              string         `json:"id"`
	SourceRunID     string         `json:"sourceRunId"`
	Kind            string         `json:"kind"`
	Rationale       string         `json:"rationale"`
	EvidenceStepIDs []string       `json:"evidenceStepIds"`
	ConfigPatch     map[string]any `json:"configPatch"`
	Status          string         `json:"status"`
}
type RunRequest struct {
	ExperimentID string `json:"experimentId"`
	VariantID    string `json:"variantId"`
	PersonaID    string `json:"personaId"`
	Adapter      string `json:"adapter"`
}
type Run struct {
	ID                string            `json:"id"`
	ExperimentID      string            `json:"experimentId"`
	BaselineVariantID string            `json:"baselineVariantId"`
	VariantID         string            `json:"variantId"`
	PersonaID         string            `json:"personaId"`
	PersonaSplit      string            `json:"personaSplit"`
	Seed              int               `json:"seed"`
	Config            VariantConfig     `json:"config"`
	Persona           Persona           `json:"persona"`
	Adapter           string            `json:"adapter"`
	Synthetic         bool              `json:"synthetic"`
	ContractVersion   int               `json:"contractVersion"`
	Status            string            `json:"status"`
	CreatedAt         string            `json:"createdAt"`
	StartedAt         string            `json:"startedAt,omitempty"`
	CompletedAt       string            `json:"completedAt,omitempty"`
	Events            []StepEvent       `json:"events"`
	Artifacts         []Artifact        `json:"artifacts"`
	Evaluation        *EvaluationResult `json:"evaluation,omitempty"`
	ProposedChanges   []ProposedChange  `json:"proposedChanges"`
	Error             string            `json:"error,omitempty"`
}
type ComparisonRow struct {
	VariantID      string  `json:"variantId"`
	PersonaSplit   string  `json:"personaSplit"`
	Runs           int     `json:"runs"`
	CompletedTasks int     `json:"completedTasks"`
	MeanScore      float64 `json:"meanScore"`
	Synthetic      bool    `json:"synthetic"`
}
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
	OrderValue    float64             `json:"orderValue,omitempty"`
}
