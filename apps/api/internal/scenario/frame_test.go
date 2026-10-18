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
