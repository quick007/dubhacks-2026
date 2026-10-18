import assert from "node:assert/strict";
import { test } from "node:test";
import { buildCohortEnvelope, postCohort } from "./checkout-cohort-export.mjs";

// Protocol fixtures only; never uploaded or represented as shopper observations.
function fixture() {
  const sha256 = "a".repeat(64), schedule = [], pairs = [], rows = [], sourceReports = [];
  for (let packet = 1; packet <= 6; packet++) {
    const reportRows = [];
    for (let i = 0; i < 10; i++) {
      const caseId = `p${packet}-c${i}`, pair = { caseId };
      for (const arm of ["generic", "local-model"]) {
        const runId = `${caseId}-${arm}`, slot = { caseId, arm, runId };
        schedule.push(slot); pair[arm] = slot;
        reportRows.push({ caseId, arm, expectedRunId: runId, status: "unattempted", formalStatus: "unattempted" });
        rows.push({ ...slot, serverConfirmed: false, visibleCheckoutConfirmed: false, receipt: null });
      }
      pairs.push(pair);
    }
    sourceReports.push({ path: `/fixture/packet-${packet}/aggregate-audited.json`, sha256, experimentId: `packet-${packet}`,
      report: { version: "shopper-evidence-aggregate.v1", synthetic: true, humanConversionClaim: false, fixedCases: 10, fixedSessions: 20,
        rows: reportRows, arms: { generic: { fixedDenominator: 10 }, "local-model": { fixedDenominator: 10 } } } });
  }
  const sources = sourceReports.map(({ path, sha256 }) => ({ path, sha256 }));
  const serverSource = { path: "/fixture/server-outcomes.json", sha256 };
  return { reportId: "cohort-fixture", experimentId: "cohort-v2",
    plan: { path: "/fixture/cohort-plan.json", sha256, data: { version: "shopper-fixed-cohort.v1", synthetic: true, humanConversionClaim: false, fixedPairs: 60, fixedSessions: 120, schedule } },
    sourceReports,
    metrics: { path: "/fixture/metrics.json", sha256, data: { version: "shopper-synthetic-metrics.v1", synthetic: true, humanConversionClaim: false,
      fixedCases: 60, fixedSessions: 120, sources, serverIndexSource: serverSource, pairs,
      arms: Object.fromEntries(["generic", "local-model"].map((arm) => [arm, { assigned: 60, allServerConfirmedOrders: { evaluated: true } }])) } },
    serverOutcomes: { ...serverSource, data: { schema: "shopper-server-outcomes.v1", synthetic: true, humanConversionClaim: false, reports: sources, rows } } };
}

test("retains all120 slots and per-panel experiment identities without promoting missing data", () => {
  const input = fixture(), result = buildCohortEnvelope(input);
  assert.equal(result.report.sourceReports.flatMap((r) => r.report.rows).length, 120);
  assert.equal(result.report.sourceReports[5].experimentId, "packet-6");
  input.metrics.data.pairs.length = 0;
  assert.equal(result.report.metrics.data.pairs.length, 60, "caller mutations must not alter validated envelope");
});
test("changed source hashes, roster, claims and denominator fail closed", () => {
  const changes = [
    (f) => { f.sourceReports[0].report.rows.pop(); },
    (f) => { f.metrics.data.fixedCases = 59; },
    (f) => { f.metrics.data.sources[0].sha256 = "b".repeat(64); },
    (f) => { f.plan.data.schedule[0].runId = "different"; },
    (f) => { f.serverOutcomes.data.rows[0].serverConfirmed = true; },
    (f) => { f.metrics.data.humanConversionClaim = true; },
    (f) => { f.metrics.data.pairs[0].generic.runId = "different"; },
    (f) => { f.metrics.data.arms.generic.assigned = 10; },
  ];
  for (const change of changes) { const f = structuredClone(fixture()); change(f); assert.throws(() => buildCohortEnvelope(f)); }
});
test("post goes only to fixed loopback without redirects/credentials and confirms local202 only", async () => {
  const envelope = buildCohortEnvelope(fixture()); let called = false;
  const receipt = await postCohort(envelope, async (url, init) => {
    called = true; assert.equal(url, "http://127.0.0.1:4311/api/experiment-results"); assert.equal(init.redirect, "error");
    assert.equal(init.method, "POST"); assert.equal(init.headers.Authorization, undefined);
    return new Response(JSON.stringify({ reportId: envelope.reportId, sha256: "c".repeat(64), durableLocal: true, duplicate: false, cloud: "asynchronous" }), { status: 202 });
  });
  assert.equal(called, true); assert.equal(receipt.cloud, "asynchronous");
  await assert.rejects(postCohort(envelope, async () => new Response("{}", { status: 409 })), /HTTP409/);
});
