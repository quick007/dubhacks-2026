package shoppingevents

import (
	"fmt"
	"net/url"
	"os"
	"strings"
)

type Config struct {
	Enabled        bool
	Environment    string
	Origins        []string
	AllowSynthetic bool
}

func ConfigFromEnv() (Config, error) {
	c := Config{Enabled: os.Getenv("SHOPPING_TELEMETRY_ENABLED") == "1", Environment: os.Getenv("SHOPPING_TELEMETRY_ENVIRONMENT"), AllowSynthetic: os.Getenv("SHOPPING_TELEMETRY_ALLOW_SYNTHETIC") == "1"}
	if !c.Enabled {
		return c, nil
	}
	if !oneOf(c.Environment, "production", "development", "test") {
		return c, fmt.Errorf("explicit shopping telemetry server environment required")
	}
	for _, origin := range strings.Split(os.Getenv("SHOPPING_TELEMETRY_ORIGINS"), ",") {
		origin = strings.TrimSpace(origin)
		u, err := url.Parse(origin)
		if err != nil || u.Host == "" || u.User != nil || u.Path != "" || u.RawQuery != "" || u.Fragment != "" || !oneOf(u.Scheme, "http", "https") {
			return c, fmt.Errorf("exact shopping telemetry origins required")
		}
		hostname := strings.ToLower(u.Hostname())
		if c.Environment == "production" && (u.Scheme != "https" || oneOf(hostname, "localhost", "127.0.0.1", "::1") || strings.HasSuffix(hostname, ".localhost")) {
			return c, fmt.Errorf("production telemetry requires a trusted HTTPS public origin")
		}
		c.Origins = append(c.Origins, origin)
	}
	if c.Environment != "production" && !c.AllowSynthetic {
		return c, fmt.Errorf("nonproduction telemetry requires explicit synthetic mode")
	}
	return c, nil
}
func (c Config) permits(b Batch) bool {
	return c.Enabled && b.Classification.Environment == c.Environment && (b.Classification.Traffic == "synthetic" && c.AllowSynthetic || b.Classification.Traffic == "real" && c.Environment == "production")
}
