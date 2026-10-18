/** Public deployment settings only. No URL flag can promote traffic to real. */
export type ShoppingTelemetryConfig = {
  endpoint: string;
  classification: { environment: "production" | "development" | "test"; traffic: "real" | "synthetic" };
  consent: "required" | "notice";
  modelVersion: string | null;
};
export function shoppingTelemetryConfig(env: Record<string, unknown>, location: Pick<Location, "hostname" | "origin" | "protocol">): ShoppingTelemetryConfig | null {
  const mode = env.VITE_SHOP_TELEMETRY;
  if (mode !== "production" && mode !== "synthetic") return null;
  const local = /^(localhost|127\.0\.0\.1|\[::1\]|[a-z0-9.-]+\.localhost)$/.test(location.hostname);
  const hosts = String(env.VITE_SHOP_TELEMETRY_HOSTS ?? "").split(",").map((item) => item.trim()).filter(Boolean);
  if (!hosts.includes(location.hostname)) return null;
  if (mode === "production" && (env.PROD !== true || local || location.protocol !== "https:")) return null;
  if (mode === "synthetic" && !local) return null;
  const apiOrigin = String(env.VITE_SHOP_TELEMETRY_API_ORIGIN ?? location.origin);
  try {
    const api = new URL(apiOrigin);
    if (api.origin !== apiOrigin || api.username || api.password || (mode === "production" && api.protocol !== "https:")) return null;
    if (mode === "synthetic" && !/^(localhost|127\.0\.0\.1|\[::1\]|[a-z0-9.-]+\.localhost)$/.test(api.hostname)) return null;
  } catch { return null; }
  const model = env.VITE_SHOP_MODEL_VERSION;
  return { endpoint: `${apiOrigin}/api/shopping-events`,
    classification: mode === "production" ? { environment: "production", traffic: "real" }
      : { environment: env.VITE_SHOP_TELEMETRY_ENVIRONMENT === "development" ? "development" : "test", traffic: "synthetic" },
    consent: env.VITE_SHOP_TELEMETRY_CONSENT === "notice" ? "notice" : "required",
    modelVersion: typeof model === "string" && /^[A-Za-z0-9._-]{1,64}$/.test(model) ? model : null };
}
export function privacyChoice(config: ShoppingTelemetryConfig, stored: string | null, signals: { globalPrivacyControl?: boolean; doNotTrack?: string | null }): "notice" | "granted" | "denied" | "pending" {
  if (signals.globalPrivacyControl || signals.doNotTrack === "1" || stored === "denied") return "denied";
  if (stored === "granted") return "granted";
  return config.consent === "notice" ? "notice" : "pending";
}
