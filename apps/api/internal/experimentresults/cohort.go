package experimentresults

import (
	"encoding/json"
	"fmt"
	"regexp"
)

var sha256Text = regexp.MustCompile(`^[a-f0-9]{64}$`)

type EvidenceSource struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}
type EvidenceDocument struct {
	EvidenceSource
	Data json.RawMessage `json:"data"`
}
type CohortReport struct {
	EvidenceSource
	ExperimentID string          `json:"experimentId"`
	Report       json.RawMessage `json:"report"`
}
type CheckoutCohort struct {
	Version              string            `json:"version"`
	Synthetic            bool              `json:"synthetic"`
	HumanConversionClaim *bool             `json:"humanConversionClaim"`
	FixedCases           int               `json:"fixedCases"`
	FixedSessions        int               `json:"fixedSessions"`
	Plan                 EvidenceDocument  `json:"plan"`
	SourceReports        []CohortReport    `json:"sourceReports"`
	Metrics              EvidenceDocument  `json:"metrics"`
	ServerOutcomes       EvidenceDocument  `json:"serverOutcomes"`
	SlideMetrics         *EvidenceDocument `json:"slideMetrics,omitempty"`
}

func validSource(s EvidenceSource) bool {
	return s.Path != "" && len(s.Path) <= 4000 && sha256Text.MatchString(s.SHA256)
}

// Intake validates transport identities and exact source bindings. The local
// exporter rehashes evidence and reruns the harness analyzer; this API does not
// invent scores, re-evaluate private goals, or claim that a client flag is truth.
func validateCohort(e Envelope) error {
	bad := func() error {
		return fmt.Errorf("verified six-panel synthetic checkout cohort and exact source/run bindings required")
	}
	var c CheckoutCohort
	if json.Unmarshal(e.Report, &c) != nil || c.Version != "shopper-checkout-cohort.v1" || !c.Synthetic || c.HumanConversionClaim == nil || *c.HumanConversionClaim || c.FixedCases != 60 || c.FixedSessions != 120 || len(c.SourceReports) != 6 {
		return bad()
	}
	for _, d := range []EvidenceDocument{c.Plan, c.Metrics, c.ServerOutcomes} {
		if !validSource(d.EvidenceSource) || len(d.Data) == 0 {
			return bad()
		}
	}
	if c.SlideMetrics != nil && (!validSource(c.SlideMetrics.EvidenceSource) || len(c.SlideMetrics.Data) == 0) {
		return bad()
	}
	type slot struct {
		CaseID string `json:"caseId"`
		Arm    string `json:"arm"`
		RunID  string `json:"runId"`
	}
	var plan struct {
		Version   string `json:"version"`
		Synthetic bool   `json:"synthetic"`
		Human     *bool  `json:"humanConversionClaim"`
		Pairs     int    `json:"fixedPairs"`
		Sessions  int    `json:"fixedSessions"`
		Schedule  []slot `json:"schedule"`
	}
	if json.Unmarshal(c.Plan.Data, &plan) != nil || plan.Version != "shopper-fixed-cohort.v1" || !plan.Synthetic || plan.Human == nil || *plan.Human || plan.Pairs != 60 || plan.Sessions != 120 || len(plan.Schedule) != 120 {
		return bad()
	}
	planned := map[string]slot{}
	for _, s := range plan.Schedule {
		if !safeID.MatchString(s.RunID) || !safeID.MatchString(s.CaseID) || (s.Arm != "generic" && s.Arm != "local-model") {
			return bad()
		}
		if _, ok := planned[s.RunID]; ok {
			return bad()
		}
		planned[s.RunID] = s
	}
	assigned := map[string]slot{}
	sources := map[string]string{}
	cases := map[string]bool{}
	for _, source := range c.SourceReports {
		if !validSource(source.EvidenceSource) || !safeID.MatchString(source.ExperimentID) || sources[source.Path] != "" {
			return bad()
		}
		sources[source.Path] = source.SHA256
		var r Report
		if json.Unmarshal(source.Report, &r) != nil {
			return bad()
		}
		child := e
		child.ExperimentID = source.ExperimentID
		if validate(child, r) != nil {
			return bad()
		}
		for _, row := range r.Rows {
			s := slot{row.CaseID, row.Arm, row.ExpectedRunID}
			if assigned[s.RunID].RunID != "" || planned[s.RunID] != s {
				return bad()
			}
			assigned[s.RunID] = s
			cases[s.CaseID] = true
		}
		var raw struct {
			Synthetic bool `json:"synthetic"`
			Rows      []struct {
				Session *struct {
					RealPurchase *bool `json:"realPurchase"`
				} `json:"session"`
			} `json:"rows"`
		}
		if json.Unmarshal(source.Report, &raw) != nil || !raw.Synthetic {
			return bad()
		}
		for _, r := range raw.Rows {
			if r.Session != nil && (r.Session.RealPurchase == nil || *r.Session.RealPurchase) {
				return bad()
			}
		}
	}
	if len(assigned) != 120 || len(cases) != 60 {
		return bad()
	}
	var index struct {
		Schema    string           `json:"schema"`
		Synthetic bool             `json:"synthetic"`
		Human     *bool            `json:"humanConversionClaim"`
		Reports   []EvidenceSource `json:"reports"`
		Rows      []struct {
			slot
			Confirmed *bool           `json:"serverConfirmed"`
			Visible   *bool           `json:"visibleCheckoutConfirmed"`
			Receipt   json.RawMessage `json:"receipt"`
			Readback  *struct {
				HTTPStatus int  `json:"httpStatus"`
				Identical  bool `json:"identical"`
			} `json:"readback"`
		} `json:"rows"`
	}
	if json.Unmarshal(c.ServerOutcomes.Data, &index) != nil || index.Schema != "shopper-server-outcomes.v1" || !index.Synthetic || index.Human == nil || *index.Human || len(index.Reports) != 6 || len(index.Rows) != 120 {
		return bad()
	}
	matched := map[string]bool{}
	for _, s := range index.Reports {
		if sources[s.Path] != s.SHA256 || matched[s.Path] {
			return bad()
		}
		matched[s.Path] = true
	}
	seen := map[string]bool{}
	for _, r := range index.Rows {
		if assigned[r.RunID] != r.slot || seen[r.RunID] || r.Confirmed == nil || r.Visible == nil || (*r.Visible && !*r.Confirmed) {
			return bad()
		}
		seen[r.RunID] = true
		if *r.Confirmed {
			var receipt struct {
				Schema string `json:"schema"`
				Real   *bool  `json:"realPurchase"`
			}
			if json.Unmarshal(r.Receipt, &receipt) != nil || receipt.Schema != "fieldwork.demo-checkout-receipt.v1" || receipt.Real == nil || *receipt.Real || r.Readback == nil || r.Readback.HTTPStatus != 200 || !r.Readback.Identical {
				return bad()
			}
		}
	}
	var metrics struct {
		Version   string           `json:"version"`
		Synthetic bool             `json:"synthetic"`
		Human     *bool            `json:"humanConversionClaim"`
		Cases     int              `json:"fixedCases"`
		Sessions  int              `json:"fixedSessions"`
		Sources   []EvidenceSource `json:"sources"`
		Server    EvidenceSource   `json:"serverIndexSource"`
		Pairs     []struct {
			CaseID  string `json:"caseId"`
			Generic slot   `json:"generic"`
			Local   slot   `json:"local-model"`
		} `json:"pairs"`
		Arms map[string]struct {
			Assigned int `json:"assigned"`
			Server   struct {
				Evaluated bool `json:"evaluated"`
			} `json:"allServerConfirmedOrders"`
		} `json:"arms"`
	}
	if json.Unmarshal(c.Metrics.Data, &metrics) != nil || metrics.Version != "shopper-synthetic-metrics.v1" || !metrics.Synthetic || metrics.Human == nil || *metrics.Human || metrics.Cases != 60 || metrics.Sessions != 120 || len(metrics.Sources) != 6 || metrics.Server != c.ServerOutcomes.EvidenceSource || len(metrics.Pairs) != 60 {
		return bad()
	}
	matched = map[string]bool{}
	seen = map[string]bool{}
	for _, s := range metrics.Sources {
		if sources[s.Path] != s.SHA256 || matched[s.Path] {
			return bad()
		}
		matched[s.Path] = true
	}
	for _, p := range metrics.Pairs {
		if !cases[p.CaseID] || seen[p.CaseID] || p.Generic.CaseID != p.CaseID || p.Local.CaseID != p.CaseID || p.Generic.Arm != "generic" || p.Local.Arm != "local-model" || assigned[p.Generic.RunID] != p.Generic || assigned[p.Local.RunID] != p.Local {
			return bad()
		}
		seen[p.CaseID] = true
	}
	for _, arm := range []string{"generic", "local-model"} {
		if metrics.Arms[arm].Assigned != 60 || !metrics.Arms[arm].Server.Evaluated {
			return bad()
		}
	}
	return nil
}
