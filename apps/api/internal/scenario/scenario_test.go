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
