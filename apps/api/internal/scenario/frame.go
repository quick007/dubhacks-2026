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
