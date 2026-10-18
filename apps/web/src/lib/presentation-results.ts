/** Data-only binding for one immutable, server-confirmed synthetic cohort in Tiger. */
export type EvidenceSource = { path: string; sha256: string };
export type Fraction = { numerator: number; denominator: number; rate: number | null; interval95: unknown };
export type CohortArm = {
  assigned: number;
  recorded: number;
  verifiedPurchases: number;
  intentCorrectSuccess: Fraction;
  syntheticConversion: Fraction;
  prespecifiedFeasibleCheckout: Fraction | null;
  revenue: { currency: "USD"; totalCents: number; revenuePerAssignedSessionCents: number; [key: string]: unknown };
  allServerConfirmedOrders: { evaluated: true; count: number; merchandiseSubtotalCents: number;
    merchandiseRevenuePerAssignedSessionCents: number; [key: string]: unknown };
  [key: string]: unknown;
};
export type PresentationResults = {
  version: "experiment-presentation.v1";
  reportId: string;
  experimentId: string;
  sha256: string;
  receivedAt: string;
  source: "tiger";
  cloudAcknowledged: true;
  synthetic: true;
  humanConversionClaim: false;
  data: {
    version: "shopper-checkout-cohort.v1";
    fixedCases: 60;
    fixedSessions: 120;
    metrics: { version: "shopper-synthetic-metrics.v1"; synthetic: true; humanConversionClaim: false;
      fixedCases: 60; fixedSessions: 120; arms: Record<"generic" | "local-model", CohortArm>;
      pairedDifferences: Record<string, unknown>; limitations: string[]; [key: string]: unknown };
    slideMetrics: unknown;
    provenance: { plan: EvidenceSource; sourceReports: (EvidenceSource & { experimentId: string })[];
      metrics: EvidenceSource; serverOutcomes: EvidenceSource };
  };
};
const id = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}$/;
const hash = /^[a-f0-9]{64}$/;
type Obj = Record<string, unknown>;
function object(value: unknown): Obj {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Expected versioned experiment result object");
  return value as Obj;
}
function requireValue(value: unknown, message: string): asserts value { if (!value) throw new Error(message); }
function finite(value: unknown): value is number { return typeof value === "number" && Number.isFinite(value) && value >= 0; }
function source(value: unknown) { const s = object(value); requireValue(typeof s.path === "string" && s.path.length > 0 && typeof s.sha256 === "string" && hash.test(s.sha256), "Missing evidence source digest"); }
function fraction(value: unknown, denominator?: number) {
  const f = object(value);
  requireValue(Number.isSafeInteger(f.numerator) && Number.isSafeInteger(f.denominator) && finite(f.numerator)
    && finite(f.denominator) && f.numerator <= f.denominator && (denominator === undefined || f.denominator === denominator), "Invalid assigned outcome denominator");
  requireValue(f.denominator === 0 ? f.rate === null : finite(f.rate) && Math.abs(f.rate - f.numerator / f.denominator) < 1e-12, "Outcome rate differs from recorded counts");
}

/** Reject missing/unavailable data; no mock, cached or zero-valued fallback. */
export function parsePresentationResults(input: unknown, expectedReportId: string): PresentationResults {
  const value = object(input), data = object(value.data), metrics = object(data.metrics), arms = object(metrics.arms);
  requireValue(id.test(expectedReportId) && value.reportId === expectedReportId && typeof value.experimentId === "string" && id.test(value.experimentId)
    && value.version === "experiment-presentation.v1" && value.source === "tiger" && value.cloudAcknowledged === true
    && value.synthetic === true && value.humanConversionClaim === false && typeof value.sha256 === "string" && hash.test(value.sha256)
    && typeof value.receivedAt === "string" && Number.isFinite(Date.parse(value.receivedAt)), "Fresh Tiger cohort provenance required");
  requireValue(data.version === "shopper-checkout-cohort.v1" && data.fixedCases === 60 && data.fixedSessions === 120
    && metrics.version === "shopper-synthetic-metrics.v1" && metrics.synthetic === true && metrics.humanConversionClaim === false
    && metrics.fixedCases === 60 && metrics.fixedSessions === 120, "Fixed synthetic cohort scope required");
  for (const arm of ["generic", "local-model"]) {
    const a = object(arms[arm]), revenue = object(a.revenue), server = object(a.allServerConfirmedOrders);
    requireValue(a.assigned === 60 && Number.isSafeInteger(a.recorded) && finite(a.recorded) && a.recorded <= 60
      && Number.isSafeInteger(a.verifiedPurchases) && finite(a.verifiedPurchases) && a.verifiedPurchases <= 60, "All sixty assigned sessions must remain in each arm");
    fraction(a.intentCorrectSuccess, 60); fraction(a.syntheticConversion, 60);
    if (a.prespecifiedFeasibleCheckout !== null) fraction(a.prespecifiedFeasibleCheckout);
    requireValue(revenue.currency === "USD" && finite(revenue.totalCents) && finite(revenue.revenuePerAssignedSessionCents)
      && Math.abs(revenue.totalCents / 60 - revenue.revenuePerAssignedSessionCents) < 1e-9, "Recorded merchandise revenue required");
    requireValue(server.evaluated === true && Number.isSafeInteger(server.count) && finite(server.count) && server.count <= 60
      && finite(server.merchandiseSubtotalCents) && finite(server.merchandiseRevenuePerAssignedSessionCents)
      && Math.abs(server.merchandiseSubtotalCents / 60 - server.merchandiseRevenuePerAssignedSessionCents) < 1e-9, "Evaluated server-confirmed orders required");
  }
  const provenance = object(data.provenance);
  source(provenance.plan); source(provenance.metrics); source(provenance.serverOutcomes);
  requireValue(Array.isArray(provenance.sourceReports) && provenance.sourceReports.length === 6, "Six source panels required");
  for (const report of provenance.sourceReports) source(report);
  requireValue(Array.isArray(metrics.limitations) && metrics.limitations.every((v) => typeof v === "string"), "Original evidence limitations required");
  object(metrics.pairedDifferences);
  return input as PresentationResults;
}

export async function fetchPresentationResults(reportId: string, options: { signal?: AbortSignal; fetchImpl?: typeof fetch } = {}) {
  requireValue(id.test(reportId), "Choose a safe immutable report ID");
  const response = await (options.fetchImpl ?? fetch)(`/api/experiment-results/presentation?reportId=${encodeURIComponent(reportId)}&source=tiger`,
    { signal: options.signal ?? AbortSignal.timeout(10000), cache: "no-store", credentials: "omit", redirect: "error", referrerPolicy: "no-referrer" });
  requireValue(response.ok, `Tiger presentation results unavailable (HTTP ${response.status})`);
  return parsePresentationResults(await response.json(), reportId);
}
