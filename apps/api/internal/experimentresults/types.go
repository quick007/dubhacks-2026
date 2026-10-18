// Package experimentresults archives independent shopper evidence, outside inference.
package experimentresults

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"regexp"
)

const MaxBody = 8 << 20

var safeID = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}$`)

type Envelope struct {
	Version              string          `json:"version"`
	ReportID             string          `json:"reportId"`
	ExperimentID         string          `json:"experimentId"`
	Synthetic            bool            `json:"synthetic"`
	HumanConversionClaim bool            `json:"humanConversionClaim"`
	Report               json.RawMessage `json:"report"`
}
type Report struct {
	Version              string                     `json:"version"`
	FixedCases           int                        `json:"fixedCases"`
	FixedSessions        int                        `json:"fixedSessions"`
	HumanConversionClaim *bool                      `json:"humanConversionClaim"`
	Rows                 []Row                      `json:"rows"`
	Arms                 map[string]json.RawMessage `json:"arms"`
}

// Row mirrors the identity/state fields in shopper-eval-cli.ts SessionRow.
// The full original row/session/evaluator/artifact receipts remain in Report.
type Row struct {
	CaseID        string `json:"caseId"`
	Arm           string `json:"arm"`
	ExpectedRunID string `json:"expectedRunId"`
	Status        string `json:"status"`
	FormalStatus  string `json:"formalStatus"`
	Session       *struct {
		ContractVersion string `json:"contractVersion"`
		CaseID          string `json:"caseId"`
		Arm             string `json:"arm"`
		Run             struct {
			ID           string `json:"id"`
			ExperimentID string `json:"experimentId"`
		} `json:"run"`
		ElapsedMS *float64 `json:"elapsedMs"`
		Model     struct {
			Readiness []struct {
				Stage      string   `json:"stage"`
				DurationMS *float64 `json:"durationMs"`
				ArtifactID string   `json:"artifactId"`
			} `json:"readiness"`
		} `json:"model"`
	} `json:"session"`
}
type Saved struct {
	Envelope
	Digest     string `json:"sha256"`
	ReceivedAt string `json:"receivedAt"`
}
type Header struct {
	ReportID     string `json:"reportId"`
	ExperimentID string `json:"experimentId"`
	Digest       string `json:"sha256"`
	ReceivedAt   string `json:"receivedAt"`
}

func (s Saved) Header() Header { return Header{s.ReportID, s.ExperimentID, s.Digest, s.ReceivedAt} }

// Canonicalize JSON so whitespace/object key ordering does not change retry identity.
func Decode(reader io.Reader) (Envelope, string, error) {
	var e Envelope
	d := json.NewDecoder(reader)
	d.DisallowUnknownFields()
	if err := d.Decode(&e); err != nil {
		return e, "", fmt.Errorf("invalid experiment results envelope")
	}
	if d.Decode(new(any)) != io.EOF {
		return e, "", fmt.Errorf("expected one JSON object")
	}
	if e.Version != "experiment-results.v1" || !safeID.MatchString(e.ReportID) || !safeID.MatchString(e.ExperimentID) || !e.Synthetic || e.HumanConversionClaim {
		return e, "", fmt.Errorf("versioned synthetic evidence and safe report/experiment IDs required")
	}
	var raw any
	d = json.NewDecoder(bytes.NewReader(e.Report))
	d.UseNumber()
	if err := d.Decode(&raw); err != nil {
		return e, "", fmt.Errorf("invalid report")
	}
	e.Report, _ = json.Marshal(raw)
	var report Report
	if err := json.Unmarshal(e.Report, &report); err != nil {
		return e, "", fmt.Errorf("invalid report fields")
	}
	var err error
	if report.Version == "shopper-checkout-cohort.v1" {
		err = validateCohort(e)
	} else if report.Version == "authored-shopping-history.v1" {
		err = validateAuthored(e)
	} else {
		err = validate(e, report)
	}
	if err != nil {
		return e, "", err
	}
	encoded, _ := json.Marshal(e)
	hash := sha256.Sum256(encoded)
	return e, hex.EncodeToString(hash[:]), nil
}
func validate(e Envelope, r Report) error {
	if r.Version != "shopper-evidence-aggregate.v1" || r.FixedCases != 10 || r.FixedSessions != 20 || len(r.Rows) != 20 || r.HumanConversionClaim == nil || *r.HumanConversionClaim {
		return fmt.Errorf("fixed ten-pair shopper-evidence-aggregate.v1 with humanConversionClaim=false required")
	}
	runs, slots, cases := map[string]bool{}, map[string]bool{}, map[string]int{}
	for _, row := range r.Rows {
		key := row.CaseID + "/" + row.Arm
		if !safeID.MatchString(row.CaseID) || !safeID.MatchString(row.ExpectedRunID) || runs[row.ExpectedRunID] || slots[key] || (row.Arm != "generic" && row.Arm != "local-model") {
			return fmt.Errorf("unique paired case/arm and actual run IDs required")
		}
		runs[row.ExpectedRunID] = true
		slots[key] = true
		cases[row.CaseID]++
		if row.Status != "recorded" && row.Status != "unattempted" && row.Status != "invalid-sidecar" {
			return fmt.Errorf("invalid row state")
		}
		if row.FormalStatus != "recorded" && row.FormalStatus != "unattempted" {
			return fmt.Errorf("invalid formal state")
		}
		if row.FormalStatus == "recorded" && row.Status != "recorded" {
			return fmt.Errorf("formal state requires recorded evidence")
		}
		if row.Status == "recorded" {
			s := row.Session
			if s == nil || s.ContractVersion != "shopper-session.v1" || s.CaseID != row.CaseID || s.Arm != row.Arm || s.Run.ID != row.ExpectedRunID || s.Run.ExperimentID != e.ExperimentID || !duration(s.ElapsedMS) {
				return fmt.Errorf("recorded session identity/timing mismatch")
			}
			for _, m := range s.Model.Readiness {
				if (m.Stage != "rank" && m.Stage != "firstThree" && m.Stage != "complete") || !duration(m.DurationMS) {
					return fmt.Errorf("invalid readiness measure")
				}
			}
		}
	}
	if len(cases) != 10 {
		return fmt.Errorf("ten paired cases required")
	}
	for _, n := range cases {
		if n != 2 {
			return fmt.Errorf("both arms required per case")
		}
	}
	for _, arm := range []string{"generic", "local-model"} {
		var summary struct {
			FixedDenominator int `json:"fixedDenominator"`
		}
		if json.Unmarshal(r.Arms[arm], &summary) != nil || summary.FixedDenominator != 10 {
			return fmt.Errorf("arm summaries must retain fixed denominator ten")
		}
	}
	return nil
}
func duration(n *float64) bool { return n != nil && *n >= 0 && !math.IsNaN(*n) && !math.IsInf(*n, 0) }
