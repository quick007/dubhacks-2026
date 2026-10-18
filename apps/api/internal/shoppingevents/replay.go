package shoppingevents

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"unicode"
)

var asset = regexp.MustCompile(`^/products/[a-z0-9-]+\.(jpg|png|webp)$`)
var cssPrivate = regexp.MustCompile(`(?is)url\s*\([^)]*\)|@import[^;]*;|content\s*:[^;}]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'`)
var cssComments = regexp.MustCompile(`(?s)/\*.*?\*/`)

//go:embed replay-classes.json
var classJSON []byte
var classNames = func() map[string]bool {
	var names []string
	_ = json.Unmarshal(classJSON, &names)
	set := map[string]bool{}
	for _, name := range names {
		set[name] = true
	}
	return set
}()

func mask(text string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) {
			return r
		}
		return '*'
	}, text)
}
func css(text string) string {
	return cssPrivate.ReplaceAllString(cssComments.ReplaceAllString(text, ""), "")
}
func replayAttrs(v any) map[string]any {
	result := map[string]any{}
	attrs, _ := v.(map[string]any)
	for k, v := range attrs {
		if oneOf(k, "width", "height", "rr_width", "rr_height", "rr_scrollLeft", "rr_scrollTop") {
			if number(v, -1000000, 1000000) {
				result[k] = v
			} else if s, ok := v.(string); ok && len(s) < 32 && regexp.MustCompile(`^[0-9. px%+-]*$`).MatchString(s) {
				result[k] = s
			}
		}
		if k == "src" && asset.MatchString(str(v)) {
			result[k] = v
		}
		if k == "class" && len(str(v)) < 1000 {
			allowed := []string{}
			for _, name := range strings.Fields(str(v)) {
				if classNames[name] {
					allowed = append(allowed, name)
				}
			}
			result[k] = strings.Join(allowed, " ")
		}
		if (k == "style" || k == "_cssText") && len(str(v)) <= 65536 {
			result[k] = css(str(v))
		}
	}
	return result
}
func replayNode(v any, depth int, nodes *int, parent string) (any, error) {
	*nodes++
	if depth > 64 || *nodes > 6000 {
		return nil, fmt.Errorf("replay DOM limit exceeded")
	}
	n, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("invalid replay node")
	}
	result := map[string]any{}
	tag := strings.ToLower(str(n["tagName"]))
	if oneOf(tag, "input", "textarea", "select", "iframe", "video", "audio", "script") {
		return map[string]any{"type": 2, "id": n["id"], "tagName": "div", "attributes": map[string]any{}, "childNodes": []any{}}, nil
	}
	for _, key := range []string{"type", "id"} {
		if number(n[key], -1, 1e9) {
			result[key] = n[key]
		}
	}
	if tag != "" && regexp.MustCompile(`^[a-z0-9-]{1,40}$`).MatchString(tag) {
		result["tagName"] = tag
	}
	for _, key := range []string{"isSVG", "isShadow", "isShadowHost"} {
		if value, ok := n[key].(bool); ok {
			result[key] = value
		}
	}
	if mode := str(n["compatMode"]); oneOf(mode, "CSS1Compat", "BackCompat") {
		result["compatMode"] = mode
	}
	if value, ok := n["textContent"].(string); ok {
		if parent == "style" {
			result["textContent"] = css(value)
		} else {
			result["textContent"] = mask(value)
		}
	}
	if attrs, ok := n["attributes"]; ok {
		result["attributes"] = replayAttrs(attrs)
	}
	if children, ok := n["childNodes"].([]any); ok {
		items := []any{}
		for _, child := range children {
			safe, err := replayNode(child, depth+1, nodes, tag)
			if err != nil {
				return nil, err
			}
			items = append(items, safe)
		}
		result["childNodes"] = items
	}
	return result, nil
}
func numbers(value any, keys ...string) map[string]any {
	v, _ := value.(map[string]any)
	result := map[string]any{}
	for _, key := range keys {
		if number(v[key], -1e9, 1e12) {
			result[key] = v[key]
		}
	}
	return result
}
func sanitizeReplay(r *Replay) error {
	raw, _ := json.Marshal(r)
	if r.Schema != "fieldwork.sanitized-replay.v1" || r.Events == nil || len(r.Events) > 100 || len(raw) > 384<<10 {
		return fmt.Errorf("bounded versioned replay required")
	}
	safe := []json.RawMessage{}
	nodes := 0
	for _, raw := range r.Events {
		var event struct {
			Type      int            `json:"type"`
			Timestamp int64          `json:"timestamp"`
			Data      map[string]any `json:"data"`
		}
		if json.Unmarshal(raw, &event) != nil || event.Timestamp < 1577836800000 || event.Data == nil {
			return fmt.Errorf("invalid replay event")
		}
		data := map[string]any{}
		switch event.Type {
		case 0, 1:
		case 4:
			data = numbers(event.Data, "width", "height")
			data["href"] = "http://localhost/"
		case 2:
			node, err := replayNode(event.Data["node"], 0, &nodes, "")
			if err != nil {
				return err
			}
			data["node"] = node
			data["initialOffset"] = numbers(event.Data["initialOffset"], "left", "top")
		case 3:
			source, ok := event.Data["source"].(float64)
			if !ok {
				return fmt.Errorf("invalid replay mutation source")
			}
			if source == 0 {
				data["source"] = 0
				for _, key := range []string{"texts", "attributes", "removes", "adds"} {
					items := []any{}
					values, _ := event.Data[key].([]any)
					for _, value := range values {
						v, _ := value.(map[string]any)
						item := numbers(v, "id", "parentId", "nextId")
						switch key {
						case "texts":
							item["value"] = mask(str(v["value"]))
						case "attributes":
							item["attributes"] = replayAttrs(v["attributes"])
						case "adds":
							node, err := replayNode(v["node"], 0, &nodes, "")
							if err != nil {
								return err
							}
							item["node"] = node
						}
						items = append(items, item)
					}
					data[key] = items
				}
			} else if source == 2 || source == 3 || source == 4 {
				data = numbers(event.Data, "source", "type", "id", "x", "y", "width", "height")
			} else {
				return fmt.Errorf("unsupported replay source")
			}
		default:
			return fmt.Errorf("unsupported replay event type")
		}
		encoded, _ := json.Marshal(map[string]any{"type": event.Type, "timestamp": event.Timestamp, "data": data})
		safe = append(safe, encoded)
	}
	r.Events = safe
	return nil
}
