/** Mock-only exporter tests; no browser, model, local server, or cloud traffic. */
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { buildEnvelope, postShopperResults, shopperExportMain, SHOPPER_RESULTS_ENDPOINT,
  type ExportFetch } from "./shopper-export.js";

function reportFixture() {
  return { version: "shopper-evidence-aggregate.v1", fixedCases: 10, fixedSessions: 20, status: "partial",
    humanConversionClaim: false, physicalPaintClaim: false, limitations: ["Synthetic CPU fixture; not browser evidence"],
    rows: Array.from({ length: 10 }, (_, index) => ["generic", "local-model"].map((arm) => ({
      caseId: `case-${index}`, arm, expectedRunId: `case-${index}-${arm}`, status: index === 1 ? "invalid-sidecar" : "unattempted",
      formalStatus: "unattempted", errors: index === 1 ? ["Missing screenshot; preserve error"] : [],
      ...(index === 0 ? { status: "recorded", session: { contractVersion: "shopper-session.v1", caseId: `case-${index}`, arm,
        realPurchase: false, run: { id: `case-${index}-${arm}`, experimentId: "synthetic-test" }, elapsedMs: 123,
        timingPrecision: { sessionElapsed: "measured-monotonic" }, model: { readiness: [{ stage: "complete", durationMs: 100 }] },
        artifacts: [{ status: "missing", error: "No trace; fixture" }] }, evaluator: { evidenceVerified: false } } : {}),
    }))).flat(), arms: { generic: { fixedDenominator: 10, unattempted: 8 }, "local-model": { fixedDenominator: 10, unattempted: 8 } },
  };
}
const receipt = (reportId = "snapshot-1", duplicate = false) => ({ reportId, sha256: "a".repeat(64), durableLocal: true, duplicate, cloud: "asynchronous" });
const jsonResponse = (body: unknown, status = 202) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

void test("envelope preserves the whole report including invalid/missing evidence and derives experiment identity", () => {
  const report = reportFixture(), envelope = buildEnvelope(report, { reportId: "snapshot-1" });
  assert.deepEqual(envelope, { version: "experiment-results.v1", reportId: "snapshot-1", experimentId: "synthetic-test",
    synthetic: true, humanConversionClaim: false, report });
  report.limitations.push("Caller mutation");
  assert.equal((envelope.report.limitations as string[]).length, 1);
  assert.equal((envelope.report.rows as unknown[]).length, 20);
});

void test("empty evidence retains every slot and requires explicit experiment ID", () => {
  const report = reportFixture();
  for (const row of report.rows) { delete row.session; delete row.evaluator; row.status = "unattempted"; }
  assert.throws(() => buildEnvelope(report, { reportId: "snapshot-1" }), /Safe.*ID/);
  const envelope = buildEnvelope(report, { reportId: "snapshot-1", experimentId: "empty-suite" });
  assert.equal(envelope.experimentId, "empty-suite");
  assert.deepEqual(envelope.report, report);
});

void test("rejects changed experiment, purchase flags, altered pairing, denominator, version, and missing session", () => {
  const invalid: [string, (report: ReturnType<typeof reportFixture>) => void][] = [
    ["experiment", (r) => { r.rows[0].session!.run.experimentId = "other"; }],
    ["purchase", (r) => { r.rows[0].session!.realPurchase = true; }],
    ["conversion", (r) => { r.humanConversionClaim = true; }],
    ["pair", (r) => { r.rows[0].caseId = r.rows[2].caseId; }],
    ["denominator", (r) => { r.arms.generic.fixedDenominator = 9; }],
    ["version", (r) => { r.version = "other"; }],
    ["missing", (r) => { delete r.rows[0].session; }],
    ["identity", (r) => { r.rows[0].session!.run.id = "another-run"; }],
    ["duration", (r) => { r.rows[0].session!.elapsedMs = Number.NaN; }],
  ];
  for (const [name, change] of invalid) { const report = reportFixture(); change(report); assert.throws(() => buildEnvelope(report, { reportId: "snapshot-1" }), Error, name); }
  assert.throws(() => buildEnvelope(reportFixture(), { reportId: "snapshot-1", experimentId: "different" }), /differs/);
  assert.throws(() => buildEnvelope(reportFixture(), { reportId: "unsafe/id" }), /Safe/);
  const report = reportFixture(); report.rows.pop();
  assert.throws(() => buildEnvelope(report, { reportId: "snapshot-1" }), /twenty/);
  assert.throws(() => buildEnvelope({ ...reportFixture(), synthetic: false }, { reportId: "snapshot-1" }), /Synthetic/);
});

void test("8 MiB envelope limit includes report evidence and metadata", () => {
  assert.throws(() => buildEnvelope({ ...reportFixture(), rawEvidence: "x".repeat(8 * 1024 * 1024) }, { reportId: "snapshot-1" }), /8 MiB/);
});

void test("POST is fixed loopback, JSON without Origin, redirect-free, and accepts durable duplicate receipts", async () => {
  const envelope = buildEnvelope(reportFixture(), { reportId: "snapshot-1" });
  let calls = 0;
  const fetchImpl: ExportFetch = async (url, init) => {
    calls++;
    assert.equal(url, SHOPPER_RESULTS_ENDPOINT); assert.equal(init.method, "POST"); assert.equal(init.redirect, "error");
    assert.deepEqual(init.headers, { "Content-Type": "application/json" });
    assert.equal(new Headers(init.headers).has("Origin"), false);
    assert.ok(init.signal instanceof AbortSignal); assert.deepEqual(JSON.parse(String(init.body)), envelope);
    return jsonResponse(receipt("snapshot-1", calls === 2));
  };
  assert.equal((await postShopperResults(envelope, { fetchImpl })).duplicate, false);
  assert.equal((await postShopperResults(envelope, { fetchImpl })).duplicate, true);
  assert.equal(calls, 2);
});

void test("forbids external origins, credentials, query/hash, changed path, and redirects before leaking evidence", async () => {
  const envelope = buildEnvelope(reportFixture(), { reportId: "snapshot-1" }); let calls = 0;
  const fetchImpl: ExportFetch = async () => { calls++; return jsonResponse(receipt()); };
  for (const endpoint of ["https://example.com/api/experiment-results", "http://localhost:4311/api/experiment-results",
    "http://127.0.0.1:4312/api/experiment-results", "http://user:password@127.0.0.1:4311/api/experiment-results",
    `${SHOPPER_RESULTS_ENDPOINT}?token=no`, `${SHOPPER_RESULTS_ENDPOINT}#private`, `${SHOPPER_RESULTS_ENDPOINT}/other`]) {
    await assert.rejects(postShopperResults(envelope, { endpoint, fetchImpl }), /fixed local/);
  }
  assert.equal(calls, 0);
  await assert.rejects(postShopperResults(envelope, { fetchImpl: async () => jsonResponse({}, 302) }), /HTTP 302/);
});

void test("conflict/failure or non-durable receipts never report export success", async () => {
  const envelope = buildEnvelope(reportFixture(), { reportId: "snapshot-1" });
  await assert.rejects(postShopperResults(envelope, { fetchImpl: async () => jsonResponse({}, 409) }), /new snapshot report ID/);
  await assert.rejects(postShopperResults(envelope, { fetchImpl: async () => jsonResponse({}, 503) }), /HTTP 503/);
  for (const response of [{ ...receipt(), durableLocal: false }, { ...receipt(), reportId: "wrong" },
    { ...receipt(), cloud: "synced" }, { ...receipt(), sha256: "bad" }]) {
    await assert.rejects(postShopperResults(envelope, { fetchImpl: async () => jsonResponse(response) }), /durable asynchronous/);
  }
});

void test("timeout aborts local request and validates bounds", async () => {
  const envelope = buildEnvelope(reportFixture(), { reportId: "snapshot-1" });
  const fetchImpl: ExportFetch = async (_url, init) => new Promise((_resolve, reject) => {
    init.signal!.addEventListener("abort", () => reject(init.signal!.reason), { once: true });
  });
  await assert.rejects(postShopperResults(envelope, { timeoutMs: 10, fetchImpl }), /timed out/);
  for (const timeoutMs of [0, -1, 60_001, 1.1]) await assert.rejects(postShopperResults(envelope, { timeoutMs, fetchImpl }), /timeout/);
});

void test("standalone CLI reads an already saved report and posts only on explicit invocation", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "shopper-export-test-"));
  try {
    const reportPath = path.join(root, "aggregate.json"); await writeFile(reportPath, JSON.stringify(reportFixture()));
    let calls = 0;
    const fetchImpl: ExportFetch = async () => { calls++; return jsonResponse(receipt()); };
    await assert.rejects(shopperExportMain(["--report", reportPath, "--report-id", "snapshot-1", "--endpoint", "https://example.com"], fetchImpl), /Unknown/);
    await assert.rejects(shopperExportMain(["--report", reportPath, "--report", reportPath], fetchImpl), /repeated/);
    await assert.rejects(shopperExportMain([], fetchImpl), /Use --report/);
    assert.equal(calls, 0);
    assert.deepEqual(await shopperExportMain(["--report", reportPath, "--report-id", "snapshot-1"], fetchImpl), receipt());
    assert.equal(calls, 1);
  } finally { await rm(root, { recursive: true, force: true }); }
});
