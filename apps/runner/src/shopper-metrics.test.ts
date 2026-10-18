import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { analyzeShopperMetrics, analyzeShopperMetricReports, pairedBootstrap95, shopperMetricsMain, wilson95 } from "./shopper-metrics.js";

const sha = "a".repeat(64);
function fixture(count = 2, prefix = "case") {
  return { version: "shopper-evidence-aggregate.v1", synthetic: true, humanConversionClaim: false,
    fixedCases: count, fixedSessions: count * 2, catalogSnapshotSha256: sha,
    rows: Array.from({ length: count }, (_, i) => (["generic", "local-model"] as const).map((arm) => {
      const caseId = `${prefix}-${i}`, runId = `${caseId}-${arm}`;
      return { caseId, arm, expectedRunId: runId, status: "recorded", formalStatus: "recorded", outcome: "completed",
        artifacts: ["screen", "trace", "dom"].map((artifactId) => ({ artifactId, verified: true })),
        evaluator: { evidenceVerified: true, acceptableSelection: true, allHardConstraintsMet: true, budgetRespected: true, expectedOutcomeMet: true },
        session: { caseId, arm, run: { id: runId }, realPurchase: false, status: "completed", fakeOrderConfirmed: true,
          entryUrl: arm === "generic" ? "http://127.0.0.1:4312/?model=off" : "http://127.0.0.1:4312/?model=local",
          startedAt: "2026-10-18T00:00:00Z" as string | null, elapsedMs: 1000, protocol: { passed: true, deviations: [] as string[] },
          timingPrecision: { sessionElapsed: "measured-monotonic", stepElapsed: "measured-monotonic" }, steps: [{}, {}],
          artifacts: [{ artifact: { id: "screen", kind: "screenshot" }, status: "captured" }, { artifact: { id: "trace", kind: "trace" }, status: "captured" }],
          verification: { catalogSha256: sha, independentOfShopperAgent: true, hardConstraints: [{ needId: "price", status: "met" }],
            preferences: [{ needId: "color", status: "met" }, { needId: "material", status: "unknown" }] },
          choice: { productId: "bottle", variantId: "bottle-blue", catalogPrice: 12.34, quantity: 1, observationSource: "independent-dom", evidenceArtifactIds: ["screen", "trace", "dom"] } as null | { productId: string; variantId: string; catalogPrice: number; quantity: number; observationSource: string; evidenceArtifactIds: string[] },
          model: { readiness: [] as Record<string, unknown>[] } } };
    })).flat() };
}
function armMetrics(report: ReturnType<typeof analyzeShopperMetrics>, arm: "generic" | "local-model") { return report.arms[arm]; }
function serverFixture(report: ReturnType<typeof fixture>) {
  return Object.fromEntries(report.rows.map((row) => [row.expectedRunId, { artifactHashVerified: true, row: {
    runId: row.expectedRunId, caseId: row.caseId, arm: row.arm, serverConfirmed: true, errors: [],
    readback: { httpStatus: 200, identical: true }, receipt: { schema: "fieldwork.demo-checkout-receipt.v1",
      checkoutId: `checkout-${row.expectedRunId}`, receiptId: `receipt-${row.expectedRunId}`, createdAt: "2026-10-18T00:00:00Z",
      items: [{ productId: "bottle", variantId: "bottle-blue", quantity: 1, unitPrice: 12.34, lineTotal: 12.34 }], subtotal: 12.34,
      shipping: { status: "hidden", cost: null as number | null }, total: 12.34, realPurchase: false },
  } }]));
}

void test("fixed assigned denominator includes timeout, missing and invalid sessions with zero revenue", () => {
  const report = fixture(3);
  report.rows[2].session.status = report.rows[2].outcome = "timeout";
  report.rows[2].session.fakeOrderConfirmed = false; report.rows[2].formalStatus = "unattempted";
  report.rows[4].status = "unattempted";
  report.rows[5].status = "invalid-sidecar";
  const result = analyzeShopperMetrics(report, { replicates: 1000 });
  assert.equal(result.arms.generic.assigned, 3); assert.equal(result.arms.generic.verifiedPurchases, 1);
  assert.equal(result.arms.generic.observedFakeCheckouts, 1);
  assert.equal(result.arms.generic.syntheticConversion.rate, 1 / 3);
  assert.equal(result.arms.generic.revenue.totalCents, 1234);
  assert.equal(result.arms.generic.revenue.revenuePerAssignedSessionCents, 1234 / 3);
  assert.equal(result.arms.generic.revenue.nonpurchaseSessions, 2);
  assert.equal(result.arms["local-model"].verifiedPurchases, 2);
  assert.equal(result.arms["local-model"].revenue.nonpurchaseSessions, 1);
});

void test("purchase gate excludes self report, wrong gold, unknown constraints, protocol, missing evidence, wrong catalog and quantity", () => {
  const changes = [
    (r: ReturnType<typeof fixture>["rows"][number]) => { r.session.choice!.observationSource = "agent-reported"; },
    (r: ReturnType<typeof fixture>["rows"][number]) => { r.evaluator.acceptableSelection = false; },
    (r: ReturnType<typeof fixture>["rows"][number]) => { r.session.verification.hardConstraints[0].status = "unknown"; },
    (r: ReturnType<typeof fixture>["rows"][number]) => { r.session.protocol.passed = false; r.session.protocol.deviations = ["shared storage"]; },
    (r: ReturnType<typeof fixture>["rows"][number]) => { r.artifacts[1].verified = false; },
    (r: ReturnType<typeof fixture>["rows"][number]) => { r.session.verification.catalogSha256 = "b".repeat(64); },
    (r: ReturnType<typeof fixture>["rows"][number]) => { r.session.choice!.quantity = 2; },
    (r: ReturnType<typeof fixture>["rows"][number]) => { r.session.choice!.catalogPrice = 1.234; },
    (r: ReturnType<typeof fixture>["rows"][number]) => { r.evaluator.budgetRespected = false; },
  ];
  for (const change of changes) {
    const report = fixture(1); change(report.rows[0]);
    const result = analyzeShopperMetrics(report, { replicates: 1000 }).arms.generic;
    assert.equal(result.observedFakeCheckouts, 1); assert.equal(result.verifiedPurchases, 0); assert.equal(result.revenue.totalCents, 0);
    assert.ok(Object.keys(result.purchaseExclusions).length > 0);
  }
});

void test("correct abstention is reported separately and never a conversion or revenue", () => {
  const report = fixture(1), row = report.rows[0];
  row.session.choice = null; row.session.fakeOrderConfirmed = false; row.session.status = row.outcome = "abandoned";
  row.evaluator.acceptableSelection = false; row.evaluator.allHardConstraintsMet = false;
  const result = analyzeShopperMetrics(report, { replicates: 1000 }).arms.generic;
  assert.deepEqual(result.abstentions, { observed: 1, verifiedExpected: 1 });
  assert.equal(result.verifiedPurchases, 0); assert.equal(result.revenue.totalCents, 0);
});

void test("unknown preference checks remain denominator and failed purchases receive zero credit", () => {
  const report = fixture(2); report.rows[2].evaluator.evidenceVerified = false;
  const result = analyzeShopperMetrics(report, { replicates: 1000 }).arms.generic.preference;
  assert.equal(result.observedChecks.unknown, 2);
  assert.equal(result.purchasedMetFractionPerAssignedSession, .25);
  assert.equal(result.purchasedMetFractionAmongPurchasersWithSpecifiedPreferences, .5);
  assert.equal(result.verifiedPurchasedMetChecksPerAssignedSession, .5);
  assert.equal(result.purchasedUnknownChecks, 1);
  report.rows[0].session.verification.preferences = [];
  const noPrefs = analyzeShopperMetrics(report, { replicates: 1000 }).arms.generic.preference;
  assert.equal(noPrefs.purchasersWithoutSpecifiedPreferences, 1);
  assert.equal(noPrefs.purchasedMetFractionAmongPurchasersWithSpecifiedPreferences, null);
});

void test("interruption is distinct from timeout; partial/estimated and upper-bound times excluded from comparable runtime", () => {
  const report = fixture(3);
  const interrupted = report.rows[0]; interrupted.session.status = interrupted.outcome = "technical-failure";
  interrupted.session.startedAt = null; interrupted.session.timingPrecision.sessionElapsed = "estimated";
  interrupted.session.timingPrecision.stepElapsed = "not-recorded"; interrupted.session.elapsedMs = 349;
  interrupted.session.protocol = { passed: false, deviations: ["Browser interrupted after transport closed"] };
  report.rows[2].session.timingPrecision.sessionElapsed = "upper-bound";
  report.rows[4].session.status = report.rows[4].outcome = "timeout";
  const result = analyzeShopperMetrics(report, { replicates: 1000 }).arms.generic;
  assert.equal(result.interruptions, 1); assert.equal(result.outcomes.timeout, 1);
  assert.equal(result.technicalFailuresExcludingInterruptions, 0);
  assert.equal(result.effort.measuredSessionElapsedMs.count, 1);
  assert.equal(result.effort.excludedSessionTiming.length, 2);
  assert.equal(result.effort.observedSteps.count, 3); assert.equal(result.effort.fullyTimedSteps.count, 2);
});

void test("paired case bootstrap keeps equal arm outcomes paired and is deterministic", () => {
  assert.deepEqual(pairedBootstrap95([0, 0, 0], 42, 1000), pairedBootstrap95([0, 0, 0], 42, 1000));
  const report = fixture(2);
  for (const row of report.rows.slice(2)) row.evaluator.budgetRespected = false;
  const paired = analyzeShopperMetrics(report, { seed: 42, replicates: 1000 }).pairedDifferences;
  assert.equal(paired.conversion.difference, 0); assert.equal(paired.conversion.lower, 0); assert.equal(paired.conversion.upper, 0);
  assert.equal(paired.revenuePerSessionCents.lower, 0); assert.equal(paired.purchasedPreferenceFraction.upper, 0);
  assert.equal(paired.conversion.pairs, 2);
  const interval = wilson95(0, 10); assert.equal(interval.lower, 0); assert.ok(interval.upper > .27 && interval.upper < .28);
  assert.throws(() => wilson95(11, 10), /invalid/);
});

void test("reports combine distinct case clusters and reject duplicate IDs or catalog changes", () => {
  const one = fixture(2, "first"), two = fixture(2, "second");
  const combined = analyzeShopperMetricReports([one, two], { replicates: 1000 });
  assert.equal(combined.fixedCases, 4); assert.equal(combined.fixedSessions, 8); assert.equal(combined.sourceReports, 2);
  assert.equal(combined.pairedDifferences.conversion.pairs, 4);
  assert.throws(() => analyzeShopperMetricReports([one, one]), /Case count|Duplicate/);
  two.catalogSnapshotSha256 = "b".repeat(64);
  assert.throws(() => analyzeShopperMetricReports([one, two]), /catalog and prices/);
});

void test("prespecified feasible denominator and intent success distinguish valid checkout from justified abstention", () => {
  const report = fixture(2);
  for (const row of report.rows.slice(2)) {
    row.session.choice = null; row.session.fakeOrderConfirmed = false; row.session.status = row.outcome = "abandoned";
    row.evaluator.acceptableSelection = false; row.evaluator.allHardConstraintsMet = false;
  }
  const result = analyzeShopperMetrics(report, { replicates: 1000, caseEligibility: { "case-0": "satisfiable", "case-1": "known-empty" } });
  assert.equal(result.arms.generic.intentCorrectSuccess.numerator, 2);
  assert.equal(result.arms.generic.syntheticConversion.numerator, 1);
  assert.equal(result.arms.generic.prespecifiedFeasibleCheckout!.denominator, 1);
  assert.equal(result.arms.generic.prespecifiedFeasibleCheckout!.rate, 1);
  assert.equal(result.feasibleCheckoutPairedDifference!.pairs, 1);
  assert.deepEqual(result.prespecifiedEligibility, { satisfiable: 1, "known-empty": 1, "unverifiable-required-fact": 0 });
  assert.throws(() => analyzeShopperMetrics(report, { caseEligibility: { "case-0": "known-empty", "case-1": "satisfiable" } }), /contradicts/);
  assert.throws(() => analyzeShopperMetrics(report, { caseEligibility: { "case-0": "satisfiable" } }), /exactly/);
});

void test("invalid pairing, malformed evidence and unsafe numeric values fail closed", () => {
  for (const change of [
    (r: ReturnType<typeof fixture>) => { r.rows.pop(); },
    (r: ReturnType<typeof fixture>) => { r.rows[1].arm = "generic"; },
    (r: ReturnType<typeof fixture>) => { r.rows[1].expectedRunId = r.rows[0].expectedRunId; },
    (r: ReturnType<typeof fixture>) => { r.rows[0].session.elapsedMs = Number.NaN; },
    (r: ReturnType<typeof fixture>) => { r.rows[0].session.realPurchase = true; },
    (r: ReturnType<typeof fixture>) => { r.rows[0].session.verification.preferences[0].status = "other"; },
  ]) { const report = fixture(); change(report); assert.throws(() => analyzeShopperMetrics(report), Error); }
});

void test("readiness uses verified scoped evidence, deduplicates by request and rejects conflicting timings", () => {
  const report = fixture(1), row = report.rows[1];
  const measure = { stage: "rank", requestId: "request", durationMs: 193, artifactId: "dom", capturedAtElapsedMs: 500,
    scope: "after-paint-opportunity", physicalPaintMeasured: false, imageTransferMeasured: false };
  row.session.model.readiness = [measure, { ...measure }];
  const result = analyzeShopperMetrics(report, { replicates: 1000 });
  assert.equal(result.arms["local-model"].browserReadinessMs.rank.count, 1);
  row.session.model.readiness[1].durationMs = 194;
  assert.throws(() => analyzeShopperMetrics(report), /Conflicting duplicate/);
});

void test("CLI repeated reports preserves source bytes/digests and refuses overwrite", async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "shopper-metrics-"));
  try {
    const first = path.join(directory, "first.json"), second = path.join(directory, "second.json"), output = path.join(directory, "metrics.json");
    const bytes = JSON.stringify(fixture(1, "first")); await writeFile(first, bytes); await writeFile(second, JSON.stringify(fixture(1, "second")));
    const result = await shopperMetricsMain(["--report", first, "--report", second, "--out", output]);
    assert.equal(result.fixedCases, 2); assert.equal(result.sources.length, 2); assert.match(result.sources[0].sha256, /^[a-f0-9]{64}$/);
    assert.equal(await readFile(first, "utf8"), bytes);
    await assert.rejects(shopperMetricsMain(["--report", first, "--out", output]), /EEXIST/);
    await assert.rejects(shopperMetricsMain(["--report", first, "--out", first]), /differ/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

void test("CLI loads gold eligibility offline only after matching manifest and aggregate hashes", async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "shopper-metrics-gold-"));
  try {
    const first = path.join(directory, "report.json"), output = path.join(directory, "metrics.json"), report = fixture(1);
    const gold = JSON.stringify([{ caseId: "case-0", eligibilityStatus: "satisfiable" }]);
    const hash = createHash("sha256").update(gold).digest("hex");
    await writeFile(first, JSON.stringify({ ...report, evaluatorGoldSha256: hash }));
    await writeFile(path.join(directory, "evaluator-gold.json"), gold);
    await writeFile(path.join(directory, "manifest.json"), JSON.stringify({ files: { "evaluator-gold.json": hash } }));
    const result = await shopperMetricsMain(["--report", first, "--packet", directory, "--out", output]);
    assert.equal(result.arms.generic.prespecifiedFeasibleCheckout!.denominator, 1);
    assert.equal(result.packetSources[0].evaluatorGoldSha256, hash);
    await writeFile(path.join(directory, "evaluator-gold.json"), `${gold} `);
    await assert.rejects(shopperMetricsMain(["--report", first, "--packet", directory, "--out", path.join(directory, "other.json")]), /gold differs/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

void test("server policy gates purchases and separately retains confirmed revenue from constraint-excluded orders", () => {
  const report = fixture(1), serverOutcomes = serverFixture(report);
  report.rows[0].evaluator.allHardConstraintsMet = false;
  const result = analyzeShopperMetrics(report, { serverOutcomes, replicates: 1000 });
  assert.match(result.checkoutEvidencePolicy, /Server-confirmed/);
  assert.equal(result.arms.generic.verifiedPurchases, 0);
  assert.equal(result.arms.generic.revenue.totalCents, 0);
  assert.equal(result.arms.generic.allServerConfirmedOrders.count, 1);
  assert.equal(result.arms.generic.allServerConfirmedOrders.merchandiseRevenuePerAssignedSessionCents, 1234);
  assert.equal(result.arms["local-model"].verifiedPurchases, 1);
  serverOutcomes[report.rows[1].expectedRunId].row.serverConfirmed = false;
  const unconfirmed = analyzeShopperMetrics(report, { serverOutcomes, replicates: 1000 });
  assert.equal(unconfirmed.arms["local-model"].observedFakeCheckouts, 1);
  assert.equal(unconfirmed.arms["local-model"].verifiedPurchases, 0);
});

void test("server confirmation rejects mismatched price, quantity, GET readback or missing artifact verification", () => {
  for (const change of [
    (index: ReturnType<typeof serverFixture>, id: string) => { index[id].row.receipt.items[0].unitPrice = 999; },
    (index: ReturnType<typeof serverFixture>, id: string) => { index[id].row.receipt.items[0].quantity = 2; },
    (index: ReturnType<typeof serverFixture>, id: string) => { index[id].row.readback.identical = false; },
    (index: ReturnType<typeof serverFixture>, id: string) => { index[id].artifactHashVerified = false; },
  ]) {
    const report = fixture(1), serverOutcomes = serverFixture(report); change(serverOutcomes, report.rows[0].expectedRunId);
    assert.throws(() => analyzeShopperMetrics(report, { serverOutcomes }), Error);
  }
  const report = fixture(1), serverOutcomes = serverFixture(report); delete serverOutcomes[report.rows[0].expectedRunId];
  assert.throws(() => analyzeShopperMetrics(report, { serverOutcomes }), /every assigned run/);
});

void test("API hidden shipping receipt preserves unknown charge while confirmed merchandise revenue remains valid", () => {
  const report = fixture(1), serverOutcomes = serverFixture(report);
  const result = analyzeShopperMetrics(report, { serverOutcomes, replicates: 1000 });
  const orders = result.arms.generic.allServerConfirmedOrders;
  assert.equal(result.arms.generic.verifiedPurchases, 1);
  assert.equal(result.arms.generic.revenue.totalCents, 1234);
  assert.equal(orders.merchandiseSubtotalCents, 1234);
  assert.equal(orders.shippingKnown, false); assert.equal(orders.shippingCents, null);
  assert.equal(orders.shippingKnownOrders, 0); assert.equal(orders.shippingUnknownOrders, 1);
  assert.equal(result.pairs[0].generic.serverShippingCents, null);
  const receipt = serverOutcomes[report.rows[0].expectedRunId].row.receipt;
  receipt.shipping.cost = 0;
  assert.throws(() => analyzeShopperMetrics(report, { serverOutcomes }), /Hidden shipping cost must remain null/);
  receipt.shipping.cost = null; receipt.total = 17.34;
  assert.throws(() => analyzeShopperMetrics(report, { serverOutcomes }), /total must equal merchandise subtotal/);
});

void test("visible flat and free shipping preserve observed amounts separately from merchandise revenue", () => {
  const report = fixture(1), serverOutcomes = serverFixture(report);
  const generic = serverOutcomes[report.rows[0].expectedRunId].row.receipt;
  generic.shipping = { status: "flat", cost: 5 }; generic.total = 17.34;
  const local = serverOutcomes[report.rows[1].expectedRunId].row.receipt;
  report.rows[1].session.choice!.catalogPrice = 64;
  local.items[0].unitPrice = local.items[0].lineTotal = local.subtotal = local.total = 64;
  local.shipping = { status: "free", cost: 0 };
  const result = analyzeShopperMetrics(report, { serverOutcomes, replicates: 1000 });
  assert.equal(result.arms.generic.allServerConfirmedOrders.shippingKnown, true);
  assert.equal(result.arms.generic.allServerConfirmedOrders.shippingCents, 500);
  assert.equal(result.arms.generic.revenue.totalCents, 1234);
  assert.equal(result.arms["local-model"].allServerConfirmedOrders.shippingKnown, true);
  assert.equal(result.arms["local-model"].allServerConfirmedOrders.shippingCents, 0);
  assert.equal(result.arms["local-model"].revenue.totalCents, 6400);
});

void test("server policy preserves justified abstention without requiring an order receipt", () => {
  const report = fixture(1);
  const serverOutcomes: Record<string, { artifactHashVerified: boolean; row: Record<string, unknown> }> = {};
  for (const row of report.rows) {
    row.session.choice = null; row.session.fakeOrderConfirmed = false; row.session.status = row.outcome = "abandoned";
    row.evaluator.acceptableSelection = false; row.evaluator.allHardConstraintsMet = false;
    serverOutcomes[row.expectedRunId] = { artifactHashVerified: false, row: { runId: row.expectedRunId,
      caseId: row.caseId, arm: row.arm, serverConfirmed: false, receipt: null, artifact: null, readback: null, errors: [] } };
  }
  const result = analyzeShopperMetrics(report, { serverOutcomes, caseEligibility: { "case-0": "known-empty" }, replicates: 1000 });
  assert.equal(result.arms.generic.intentCorrectSuccess.numerator, 1);
  assert.equal(result.arms.generic.abstentions.verifiedExpected, 1);
  assert.equal(result.arms.generic.verifiedPurchases, 0);
  assert.equal(result.arms.generic.allServerConfirmedOrders.count, 0);
});

void test("CLI server index binds report roster and verifies actual artifact hash before granting purchase credit", async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "shopper-metrics-server-"));
  try {
    const report = fixture(1), source = path.join(directory, "report.json"), output = path.join(directory, "metrics.json"), indexPath = path.join(directory, "index.json");
    const reportBytes = JSON.stringify(report); await writeFile(source, reportBytes);
    const evidenceFiles: { path: string; bytes: string }[] = [];
    const rows = [];
    for (const { row } of Object.values(serverFixture(report))) {
      const evidencePath = path.join(directory, `${row.arm}-checkout-receipt.json`);
      const evidenceBytes = JSON.stringify({ schema: "shopper-demo-checkout-evidence.v1", synthetic: true, errors: [],
        post: { url: "http://127.0.0.1:4312/api/demo-checkout", httpStatus: 202, request: {}, body: { receipt: row.receipt, duplicate: false } },
        get: { url: `http://127.0.0.1:4312/api/demo-checkout/${row.receipt.checkoutId}`, httpStatus: 200, body: { receipt: row.receipt } },
        confirmation: { visible: true, checkoutId: row.receipt.checkoutId, receiptId: row.receipt.receiptId } });
      await writeFile(evidencePath, evidenceBytes); evidenceFiles.push({ path: evidencePath, bytes: evidenceBytes });
      rows.push({ ...row, artifact: { path: evidencePath, sha256: createHash("sha256").update(evidenceBytes).digest("hex"), bytes: Buffer.byteLength(evidenceBytes) } });
    }
    const index = { schema: "shopper-server-outcomes.v1", synthetic: true, humanConversionClaim: false,
      reports: [{ path: source, sha256: createHash("sha256").update(reportBytes).digest("hex") }],
      rows };
    await writeFile(indexPath, JSON.stringify(index));
    const args = ["--report", source, "--server-outcomes", indexPath, "--out", output];
    assert.equal((await shopperMetricsMain(args)).arms.generic.verifiedPurchases, 1);
    const evidencePath = evidenceFiles[0].path, evidenceBytes = evidenceFiles[0].bytes;
    await writeFile(evidencePath, `${evidenceBytes} `);
    await assert.rejects(shopperMetricsMain([...args.slice(0, -1), path.join(directory, "other.json")]), /byte count differs/);
    await writeFile(evidencePath, evidenceBytes);
    const parsed = JSON.parse(evidenceBytes); parsed.post.body.receipt.total = 99;
    const changed = JSON.stringify(parsed); await writeFile(evidencePath, changed);
    index.rows[0].artifact.sha256 = createHash("sha256").update(changed).digest("hex"); index.rows[0].artifact.bytes = Buffer.byteLength(changed);
    await writeFile(indexPath, JSON.stringify(index));
    await assert.rejects(shopperMetricsMain([...args.slice(0, -1), path.join(directory, "other.json")]), /bodies differ/);
    await writeFile(evidencePath, evidenceBytes);
    index.reports[0].sha256 = "f".repeat(64); await writeFile(indexPath, JSON.stringify(index));
    await assert.rejects(shopperMetricsMain([...args.slice(0, -1), path.join(directory, "other.json")]), /exact report bytes/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
