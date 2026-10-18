import assert from "node:assert/strict";
import { test } from "node:test";
import { fetchPresentationResults, parsePresentationResults } from "./presentation-results";

function fixture() {
  const source = { path: "/fixture/metrics.json", sha256: "a".repeat(64) };
  const fraction = { numerator: 0, denominator: 60, rate: 0, interval95: null };
  const arm = { assigned: 60, recorded: 0, verifiedPurchases: 0, intentCorrectSuccess: fraction, syntheticConversion: fraction,
    prespecifiedFeasibleCheckout: null, revenue: { currency: "USD", totalCents: 0, revenuePerAssignedSessionCents: 0 },
    allServerConfirmedOrders: { evaluated: true, count: 0, merchandiseSubtotalCents: 0, merchandiseRevenuePerAssignedSessionCents: 0 } };
  return { version: "experiment-presentation.v1", reportId: "cohort-fixture", experimentId: "cohort-v2", sha256: "b".repeat(64),
    receivedAt: "2026-10-18T00:00:00Z", source: "tiger", cloudAcknowledged: true, synthetic: true, humanConversionClaim: false,
    data: { version: "shopper-checkout-cohort.v1", fixedCases: 60, fixedSessions: 120,
      metrics: { version: "shopper-synthetic-metrics.v1", synthetic: true, humanConversionClaim: false, fixedCases: 60, fixedSessions: 120,
        arms: { generic: arm, "local-model": arm }, pairedDifferences: {}, limitations: ["Synthetic protocol fixture only"] },
      slideMetrics: null, provenance: { plan: source, metrics: source, serverOutcomes: source,
        sourceReports: Array.from({ length: 6 }, (_, i) => ({ ...source, path: `/fixture/panel-${i}.json`, experimentId: `packet-${i}` })) } } };
}
test("preserves exact immutable Tiger source, synthetic scope, nulls and raw metric objects", () => {
  const value = fixture(); assert.equal(parsePresentationResults(value, "cohort-fixture"), value);
  assert.equal(parsePresentationResults(value, "cohort-fixture").data.slideMetrics, null);
});
test("rejects local fallback, missing metrics, altered denominators or fabricated rates", () => {
  for (const change of [
    (f: ReturnType<typeof fixture>) => { f.source = "local"; },
    (f: ReturnType<typeof fixture>) => { f.cloudAcknowledged = false; },
    (f: ReturnType<typeof fixture>) => { f.humanConversionClaim = true; },
    (f: ReturnType<typeof fixture>) => { f.data.metrics.arms.generic.assigned = 2; },
    (f: ReturnType<typeof fixture>) => { f.data.metrics.arms.generic.syntheticConversion.rate = .8; },
    (f: ReturnType<typeof fixture>) => { f.data.provenance.sourceReports.pop(); },
  ]) { const value = structuredClone(fixture()); change(value); assert.throws(() => parsePresentationResults(value, "cohort-fixture")); }
});
test("fetch explicitly selects Tiger and an immutable report; failure never produces mock data", async () => {
  const fetchImpl: typeof fetch = async (url, init) => {
    assert.equal(url, "/api/experiment-results/presentation?reportId=cohort-fixture&source=tiger");
    assert.equal(init?.cache, "no-store"); assert.equal(init?.credentials, "omit");
    return new Response(JSON.stringify(fixture()), { status: 200 });
  };
  const value = await fetchPresentationResults("cohort-fixture", { fetchImpl }); assert.equal(value.source, "tiger");
  await assert.rejects(fetchPresentationResults("cohort-fixture", { fetchImpl: async () => new Response("{}", { status: 503 }) }), /503/);
});
