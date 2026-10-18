import type { Findability, VariantConfig } from "@storefront/contracts";

export function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}
export function splitLabel(value: "development" | "held-out") {
  return value === "development" ? "Development" : "Held-out";
}

export const configFields: {
  key: keyof VariantConfig;
  label: string;
  format: (config: VariantConfig) => string;
}[] = [
  { key: "headline", label: "Headline", format: (c) => c.headline },
  {
    key: "showShipping",
    label: "Shipping",
    format: (c) => (c.showShipping ? "Visible" : "Hidden"),
  },
  {
    key: "recommendationOrder",
    label: "Recommendations",
    format: (c) => c.recommendationOrder[0].toUpperCase() + c.recommendationOrder.slice(1),
  },
  {
    key: "showAttributes",
    label: "Details on cards",
    format: (c) => (c.showAttributes ? "Shown" : "Hidden"),
  },
];

export function describePatch(before: VariantConfig, patch: Partial<VariantConfig>) {
  const after = { ...before, ...patch };
  return configFields
    .filter((field) => field.key in patch)
    .map((field) => `${field.label}: ${field.format(before)} to ${field.format(after)}`);
}

export const outcomeLabels: Record<Findability["outcome"], string> = {
  purchased: "Found it and bought it",
  abandoned: "Gave up",
  "constraint-failure": "Bought the wrong thing",
  "technical-failure": "Run broke",
  timeout: "Ran out of time",
};

export function relativeTime(iso: string, now = Date.now()) {
  const seconds = Math.round((now - Date.parse(iso)) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}
