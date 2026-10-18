package experimentresults

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"time"
)

//go:embed seed-catalog.json
var seedCatalog []byte

const seedSchema = "agent-authored-shopping-history.v1"

type AuthoredHistories struct {
	Version        string            `json:"version"`
	DatasetVersion string            `json:"datasetVersion"`
	Origin         string            `json:"origin"`
	Synthetic      bool              `json:"synthetic"`
	Human          *bool             `json:"humanConversionClaim"`
	Observed       *bool             `json:"browserObserved"`
	Real           *bool             `json:"realCustomerData"`
	Profiles       EvidenceDocument  `json:"profiles"`
	DemoPeople     EvidenceDocument  `json:"demoPeople"`
	Split          EvidenceDocument  `json:"split"`
	Catalog        EvidenceDocument  `json:"catalog"`
	PublicFreeze   EvidenceDocument  `json:"publicFreeze"`
	SourceManifest EvidenceSource    `json:"sourceManifest"`
	SourceHashes   map[string]string `json:"sourceHashes"`
}
type seedRequest struct {
	Query        string   `json:"query"`
	Budget       *float64 `json:"itemBudget,omitempty"`
	Requirements []struct {
		Attribute string `json:"attribute"`
		Value     string `json:"value"`
	} `json:"hardRequirements,omitempty"`
	Persona struct {
		Background []string `json:"background"`
		Colors     []string `json:"colorPreferences"`
	} `json:"persona"`
	History []struct {
		ID       string `json:"eventId"`
		Type     string `json:"eventType"`
		Name     string `json:"itemName"`
		Category string `json:"category"`
		Color    string `json:"selectedColor,omitempty"`
		At       string `json:"occurredAt"`
		Source   string `json:"source"`
		Product  string `json:"productId,omitempty"`
		Variant  string `json:"variantId,omitempty"`
		External *bool  `json:"external,omitempty"`
	} `json:"publicHistory"`
}

func strictDecode(raw []byte, value any) error {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	return d.Decode(value)
}
func boundedPublic(s string) bool { return len(s) > 0 && len(s) <= 600 }
func validateSeedRequest(raw json.RawMessage, catalog map[string]struct {
	Name, Category string
	Variants       map[string]string
}) bool {
	var q seedRequest
	if strictDecode(raw, &q) != nil || !boundedPublic(q.Query) || q.Budget != nil && (*q.Budget <= 0) || len(q.Requirements) > 12 || len(q.Persona.Background) < 1 || len(q.Persona.Background) > 6 || len(q.Persona.Colors) > 4 || len(q.History) < 4 || len(q.History) > 12 {
		return false
	}
	for _, s := range append(q.Persona.Background, q.Persona.Colors...) {
		if !boundedPublic(s) {
			return false
		}
	}
	for _, r := range q.Requirements {
		if !boundedPublic(r.Attribute) || !boundedPublic(r.Value) {
			return false
		}
	}
	seen := map[string]bool{}
	var previous time.Time
	for _, v := range q.History {
		at, err := time.Parse(time.RFC3339Nano, v.At)
		if !boundedPublic(v.ID) || !boundedPublic(v.Name) || !boundedPublic(v.Category) || seen[v.ID] || err != nil || at.Before(previous) || v.Source != "demo-seed" || (v.Type != "view" && v.Type != "cart" && v.Type != "purchase") {
			return false
		}
		seen[v.ID] = true
		previous = at
		if v.External != nil {
			if !*v.External || v.Product != "" || v.Variant != "" {
				return false
			}
		} else {
			p, ok := catalog[v.Product]
			if !ok || p.Variants[v.Variant] != v.Color || p.Name != v.Name || p.Category != v.Category {
				return false
			}
		}
	}
	return true
}
func validateAuthored(e Envelope) error {
	bad := func() error {
		return fmt.Errorf("versioned authored-only public histories, exact public catalog and frozen group split required")
	}
	var a AuthoredHistories
	if strictDecode(e.Report, &a) != nil || a.Version != "authored-shopping-history.v1" || a.DatasetVersion != seedSchema || a.Origin != "agent-authored" || !a.Synthetic || a.Human == nil || *a.Human || a.Observed == nil || *a.Observed || a.Real == nil || *a.Real {
		return bad()
	}
	for _, d := range []EvidenceDocument{a.Profiles, a.DemoPeople, a.Split, a.Catalog, a.PublicFreeze} {
		if !validSource(d.EvidenceSource) {
			return bad()
		}
	}
	if !validSource(a.SourceManifest) || len(a.SourceHashes) == 0 {
		return bad()
	}
	for path, sha := range a.SourceHashes {
		if path == "" || len(path) > 4000 || !sha256Text.MatchString(sha) {
			return bad()
		}
	}
	var actual, want any
	if json.Unmarshal(a.Catalog.Data, &actual) != nil || json.Unmarshal(seedCatalog, &want) != nil || !reflect.DeepEqual(actual, want) {
		return bad()
	}
	var catalogDocument struct {
		Products []struct {
			ID       string `json:"id"`
			Name     string `json:"name"`
			Category string `json:"category"`
			Variants []struct {
				ID    string `json:"id"`
				Color string `json:"colorName"`
			} `json:"variants"`
		} `json:"products"`
	}
	_ = json.Unmarshal(seedCatalog, &catalogDocument)
	catalog := map[string]struct {
		Name, Category string
		Variants       map[string]string
	}{}
	for _, p := range catalogDocument.Products {
		v := map[string]string{}
		for _, c := range p.Variants {
			v[c.ID] = c.Color
		}
		catalog[p.ID] = struct {
			Name, Category string
			Variants       map[string]string
		}{p.Name, p.Category, v}
	}
	type profile struct {
		Profile string          `json:"profileId"`
		Group   string          `json:"userGroupId"`
		Session string          `json:"sessionId"`
		Request json.RawMessage `json:"request"`
	}
	var profiles struct {
		Schema     string          `json:"schemaVersion"`
		Provenance json.RawMessage `json:"provenance"`
		Profiles   []profile       `json:"profiles"`
	}
	var demos struct {
		Schema     string          `json:"schemaVersion"`
		Provenance json.RawMessage `json:"provenance"`
		People     []struct {
			ID      string          `json:"id"`
			Name    string          `json:"displayName"`
			Role    string          `json:"role"`
			Summary string          `json:"summary"`
			Request json.RawMessage `json:"request"`
		} `json:"people"`
	}
	if strictDecode(a.Profiles.Data, &profiles) != nil || strictDecode(a.DemoPeople.Data, &demos) != nil || profiles.Schema != seedSchema || demos.Schema != seedSchema || len(profiles.Profiles) != 200 || len(demos.People) != 10 {
		return bad()
	}
	for _, raw := range []json.RawMessage{profiles.Provenance, demos.Provenance} {
		var p struct {
			Kind     string `json:"kind"`
			Observed *bool  `json:"browserObserved"`
			Human    *bool  `json:"humanConversionClaim"`
		}
		if json.Unmarshal(raw, &p) != nil || p.Kind != "agent-authored-synthetic" || p.Observed == nil || *p.Observed || p.Human == nil || *p.Human {
			return bad()
		}
	}
	seen := map[string]bool{}
	sessions := map[string]bool{}
	groups := map[string][]string{}
	for _, p := range profiles.Profiles {
		if !safeID.MatchString(p.Profile) || !safeID.MatchString(p.Group) || !safeID.MatchString(p.Session) || seen[p.Profile] || sessions[p.Session] || !validateSeedRequest(p.Request, catalog) {
			return bad()
		}
		seen[p.Profile] = true
		sessions[p.Session] = true
		groups[p.Group] = append(groups[p.Group], p.Profile)
	}
	if len(groups) != 100 {
		return bad()
	}
	for _, members := range groups {
		if len(members) != 2 {
			return bad()
		}
	}
	demoIDs := map[string]bool{}
	for _, p := range demos.People {
		if !safeID.MatchString(p.ID) || demoIDs[p.ID] || !boundedPublic(p.Name) || !boundedPublic(p.Role) || !boundedPublic(p.Summary) || !validateSeedRequest(p.Request, catalog) {
			return bad()
		}
		demoIDs[p.ID] = true
	}
	var split struct {
		Schema        string   `json:"schemaVersion"`
		Train         []string `json:"trainGroupIds"`
		Held          []string `json:"heldoutGroupIds"`
		TrainProfiles []string `json:"trainProfileIds"`
		HeldProfiles  []string `json:"heldoutProfileIds"`
		Groups        []struct {
			Group    string   `json:"userGroupId"`
			Split    string   `json:"split"`
			Profiles []string `json:"profileIds"`
		} `json:"groups"`
	}
	if json.Unmarshal(a.Split.Data, &split) != nil || split.Schema != seedSchema || len(split.Train) != 80 || len(split.Held) != 20 || len(split.TrainProfiles) != 160 || len(split.HeldProfiles) != 40 || len(split.Groups) != 100 {
		return bad()
	}
	groupSplit := map[string]string{}
	profileSplit := map[string]string{}
	for kind, list := range map[string][]string{"train": split.Train, "heldout": split.Held} {
		for _, id := range list {
			if groupSplit[id] != "" || len(groups[id]) != 2 {
				return bad()
			}
			groupSplit[id] = kind
		}
	}
	for kind, list := range map[string][]string{"train": split.TrainProfiles, "heldout": split.HeldProfiles} {
		for _, id := range list {
			if profileSplit[id] != "" || !seen[id] {
				return bad()
			}
			profileSplit[id] = kind
		}
	}
	seenGroups := map[string]bool{}
	for _, g := range split.Groups {
		if seenGroups[g.Group] || groupSplit[g.Group] != g.Split || len(g.Profiles) != 2 {
			return bad()
		}
		seenGroups[g.Group] = true
		members := map[string]bool{}
		for _, id := range g.Profiles {
			if members[id] || profileSplit[id] != g.Split {
				return bad()
			}
			members[id] = true
		}
		for _, id := range groups[g.Group] {
			if !members[id] {
				return bad()
			}
		}
	}
	var frozen struct {
		Schema string            `json:"schemaVersion"`
		Before bool              `json:"publicAndSplitPublishedBeforeLabels"`
		Files  map[string]string `json:"files"`
	}
	if json.Unmarshal(a.PublicFreeze.Data, &frozen) != nil || frozen.Schema != seedSchema || !frozen.Before || frozen.Files["profiles.public.json"] != a.Profiles.SHA256 || frozen.Files["demo-people.json"] != a.DemoPeople.SHA256 || frozen.Files["split.json"] != a.Split.SHA256 {
		return bad()
	}
	return nil
}
func (s *Service) registerAuthored(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/experiment-results/demo-people", func(w http.ResponseWriter, r *http.Request) {
		id := r.URL.Query().Get("reportId")
		if !safeID.MatchString(id) {
			reply(w, 400, map[string]string{"error": "immutable seed reportId required"})
			return
		}
		source := r.URL.Query().Get("source")
		if source == "" {
			source = "tiger"
		}
		saved, source, err := s.selected(r.Context(), id, source)
		if err != nil {
			failure(w, err)
			return
		}
		if validateAuthored(saved.Envelope) != nil {
			reply(w, 422, map[string]string{"error": "selected report is not an authored public history dataset"})
			return
		}
		var a AuthoredHistories
		_ = json.Unmarshal(saved.Report, &a)
		var d struct {
			People json.RawMessage `json:"people"`
		}
		_ = json.Unmarshal(a.DemoPeople.Data, &d)
		reply(w, 200, map[string]any{"version": "authored-demo-people.v1", "datasetVersion": a.DatasetVersion, "reportId": id, "sha256": saved.Digest, "receivedAt": saved.ReceivedAt, "source": source, "synthetic": true, "origin": "agent-authored", "browserObserved": false, "realCustomerData": false, "humanConversionClaim": false, "people": d.People, "provenance": map[string]any{"source": a.DemoPeople.EvidenceSource, "split": a.Split.EvidenceSource}})
	})
}
