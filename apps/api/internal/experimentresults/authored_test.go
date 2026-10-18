package experimentresults

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Repetitive protocol fixtures, never seed content or behavioral observations.
func authoredPacket(t *testing.T) []byte {
	t.Helper()
	hash := strings.Repeat("a", 64)
	events := []any{}
	for i := 0; i < 4; i++ {
		events = append(events, map[string]any{"eventId": fmt.Sprintf("fixture-event-%d", i), "eventType": "view", "itemName": "Trail mug", "category": "For the slow mornings", "selectedColor": "Sage", "occurredAt": fmt.Sprintf("2026-09-01T00:00:0%dZ", i), "source": "demo-seed", "productId": "mug", "variantId": "mug-sage"})
	}
	request := map[string]any{"query": "Useful everyday things", "persona": map[string]any{"background": []string{"Synthetic protocol fixture"}, "colorPreferences": []string{}}, "publicHistory": events}
	profiles := []any{}
	people := []any{}
	trainGroups := []string{}
	heldGroups := []string{}
	trainProfiles := []string{}
	heldProfiles := []string{}
	groups := []any{}
	for i := 0; i < 100; i++ {
		group := fmt.Sprintf("fixture-group-%d", i)
		ids := []string{}
		split := "train"
		if i >= 80 {
			split = "heldout"
			heldGroups = append(heldGroups, group)
		} else {
			trainGroups = append(trainGroups, group)
		}
		for j := 0; j < 2; j++ {
			id := fmt.Sprintf("fixture-profile-%d-%d", i, j)
			ids = append(ids, id)
			profiles = append(profiles, map[string]any{"profileId": id, "userGroupId": group, "sessionId": "session-" + id, "request": request})
			if split == "train" {
				trainProfiles = append(trainProfiles, id)
			} else {
				heldProfiles = append(heldProfiles, id)
			}
		}
		groups = append(groups, map[string]any{"userGroupId": group, "split": split, "profileIds": ids})
	}
	for i := 0; i < 10; i++ {
		people = append(people, map[string]any{"id": fmt.Sprintf("person-%d", i), "displayName": "Example", "role": "Protocol test", "summary": "Synthetic fixture", "request": request})
	}
	provenance := map[string]any{"kind": "agent-authored-synthetic", "browserObserved": false, "humanConversionClaim": false}
	doc := func(name string, data any) map[string]any {
		return map[string]any{"path": "/fixture/" + name, "sha256": hash, "data": data}
	}
	var catalog any
	_ = json.Unmarshal(seedCatalog, &catalog)
	report := map[string]any{"version": "authored-shopping-history.v1", "datasetVersion": seedSchema, "origin": "agent-authored", "synthetic": true, "browserObserved": false, "realCustomerData": false, "humanConversionClaim": false,
		"profiles":   doc("profiles.public.json", map[string]any{"schemaVersion": seedSchema, "provenance": provenance, "profiles": profiles}),
		"demoPeople": doc("demo-people.json", map[string]any{"schemaVersion": seedSchema, "provenance": provenance, "people": people}),
		"split":      doc("split.json", map[string]any{"schemaVersion": seedSchema, "trainGroupIds": trainGroups, "heldoutGroupIds": heldGroups, "trainProfileIds": trainProfiles, "heldoutProfileIds": heldProfiles, "groups": groups}),
		"catalog":    doc("catalog-binding.json", catalog), "publicFreeze": doc("PUBLIC_SPLIT_FROZEN.json", map[string]any{"schemaVersion": seedSchema, "publicAndSplitPublishedBeforeLabels": true, "files": map[string]string{"profiles.public.json": hash, "demo-people.json": hash, "split.json": hash}}),
		"sourceManifest": map[string]any{"path": "/fixture/manifest.json", "sha256": hash}, "sourceHashes": map[string]string{"fixture-authoring": hash}}
	raw, _ := json.Marshal(map[string]any{"version": "experiment-results.v1", "reportId": "authored-fixture", "experimentId": "authored-public-history-v1", "synthetic": true, "humanConversionClaim": false, "report": report})
	return raw
}
func TestAuthoredPublicHistorySplitCatalogAndLabelIsolation(t *testing.T) {
	raw := authoredPacket(t)
	decoded(t, raw)
	for _, change := range [][2]string{{`"browserObserved":false`, `"browserObserved":true`}, {`"source":"demo-seed"`, `"source":"storefront-action"`}, {`"variantId":"mug-sage"`, `"variantId":"pack-sand"`}, {`"query":"Useful everyday things"`, `"query":"Useful everyday things","targetProductId":"mug"`}, {`"userGroupId":"fixture-group-0"`, `"userGroupId":"unknown-group"`}, {`"publicAndSplitPublishedBeforeLabels":true`, `"publicAndSplitPublishedBeforeLabels":false`}} {
		changed := bytes.Replace(raw, []byte(change[0]), []byte(change[1]), 1)
		if bytes.Equal(raw, changed) {
			t.Fatal("mutation missing")
		}
		if _, _, err := Decode(bytes.NewReader(changed)); err == nil {
			t.Fatal("invalid authored dataset accepted", change[0])
		}
	}
}
func TestPublicDemoEndpointOmitsAnonymousTrainingProfiles(t *testing.T) {
	e, h := decoded(t, authoredPacket(t))
	repo := &fakeRepo{records: map[string]Saved{e.ReportID: {Envelope: e, Digest: h, ReceivedAt: "2026-10-18T00:00:00Z"}}}
	s, err := New(t.TempDir(), repo)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	mux := http.NewServeMux()
	s.Register(mux)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("GET", "/api/experiment-results/demo-people?reportId=authored-fixture&source=tiger", nil))
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"origin":"agent-authored"`) || !strings.Contains(w.Body.String(), `"browserObserved":false`) || strings.Contains(w.Body.String(), `"profileId"`) || bytes.Count(w.Body.Bytes(), []byte(`"displayName"`)) != 10 {
		t.Fatal(w.Code, w.Body)
	}
}
