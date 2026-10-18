import { useEffect, useState } from "react";
import { DEFAULT_CONFIG, type VariantConfig } from "./config";

const THEMES = [
  { id: "blue", label: "Blue", color: "#245edb" },
  { id: "forest", label: "Forest", color: "#326750" },
  { id: "clay", label: "Clay", color: "#a54f35" },
  { id: "plum", label: "Plum", color: "#79529c" },
  { id: "graphite", label: "Graphite", color: "#4b5563" },
] as const;
type Theme = (typeof THEMES)[number]["id"];
const storageKey = "fieldwork.store-color";

const ORDER_OPTIONS: { id: VariantConfig["recommendationOrder"]; label: string }[] = [
  { id: "popular", label: "Popular" },
  { id: "budget", label: "Lowest price" },
  { id: "history", label: "Supplied history" },
];

function savedTheme(): Theme {
  try {
    const value = localStorage.getItem(storageKey);
    return THEMES.find((theme) => theme.id === value)?.id ?? "blue";
  } catch {
    return "blue";
  }
}

/** Demo-only controls for the structured variant settings the runner can also pass in the URL. */
export function DemoControls({
  config,
  onChange,
}: {
  config: VariantConfig;
  onChange: (next: VariantConfig) => void;
}) {
  const [theme, setTheme] = useState<Theme>(savedTheme);
  const [headline, setHeadline] = useState(config.headline);

  useEffect(() => {
    document.documentElement.dataset.storeTheme = theme;
    try {
      localStorage.setItem(storageKey, theme);
    } catch {
      // Color selection still works when the browser cannot store a preference.
    }
  }, [theme]);

  useEffect(() => {
    setHeadline(config.headline);
  }, [config.headline]);

  return (
    <div className="demo-controls">
      <button type="button" className="demo-button" popoverTarget="demo-controls" aria-label="Demo controls">
        <span className="theme-swatch current-theme" aria-hidden="true" />
        Demo
      </button>
      <div
        id="demo-controls"
        className="demo-popover"
        popover="auto"
        role="group"
        aria-label="Demo controls"
      >
        <span className="demo-title">Demo controls</span>
        <div className="demo-section">
          <span className="theme-label">Store color</span>
          <div className="theme-options" role="group" aria-label="Store color">
            {THEMES.map((option) => (
              <button
                type="button"
                key={option.id}
                aria-pressed={theme === option.id}
                onClick={() => setTheme(option.id)}
              >
                <span className="theme-swatch" style={{ background: option.color }} aria-hidden="true" />
                {option.label}
              </button>
            ))}
          </div>
        </div>
        <form
          className="demo-section"
          onSubmit={(event) => {
            event.preventDefault();
            onChange({ ...config, headline: headline.trim() || DEFAULT_CONFIG.headline });
          }}
        >
          <label className="theme-label" htmlFor="demo-headline">
            Headline
          </label>
          <div className="demo-headline">
            <input
              id="demo-headline"
              value={headline}
              maxLength={120}
              placeholder={DEFAULT_CONFIG.headline}
              onChange={(event) => setHeadline(event.target.value)}
            />
            <button type="submit">Apply</button>
          </div>
        </form>
        <div className="demo-section">
          <span className="theme-label">Shipping details</span>
          <div className="theme-options" role="group" aria-label="Shipping details">
            <button
              type="button"
              aria-pressed={!config.showShipping}
              onClick={() => onChange({ ...config, showShipping: false })}
            >
              Hidden
            </button>
            <button
              type="button"
              aria-pressed={config.showShipping}
              onClick={() => onChange({ ...config, showShipping: true })}
            >
              Shown in cart
            </button>
          </div>
        </div>
        <div className="demo-section">
          <span className="theme-label">Product order</span>
          <div className="theme-options" role="group" aria-label="Product order">
            {ORDER_OPTIONS.map((option) => (
              <button
                type="button"
                key={option.id}
                aria-pressed={config.recommendationOrder === option.id}
                onClick={() => onChange({ ...config, recommendationOrder: option.id })}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
        <button type="button" className="demo-reset" onClick={() => onChange({ ...DEFAULT_CONFIG })}>
          Reset demo settings
        </button>
      </div>
    </div>
  );
}
