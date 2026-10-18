import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { generateServerOutcomes, runServerOutcomesCli } from "./shopper-server-outcomes.js";

const UUID = "12345678-1234-4234-8234-123456789abc", CATALOG = "c".repeat(64);
const sha = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
const encode = (value: unknown) => Buffer.from(JSON.stringify(value, null, 2) + "\n");
type Obj = Record<string, any>;
function receipt(): Obj {
  return { schema: "fieldwork.demo-checkout-receipt.v1", checkoutId: UUID, receiptId: "demo-" + "a".repeat(32), createdAt: "2026-10-18T10:00:00Z",
    items: [{ productId: "bottle", variantId: "graphite", quantity: 1, unitPrice: 36, lineTotal: 36 }], subtotal: 36,
    shipping: { status: "flat", cost: 5 }, total: 41, realPurchase: false };
}
async function fixture(t: { after: (fn: () => Promise<void>) => void }, mutate?: (evidence: Obj, session: Obj, report: Obj) => void) {
  const root = await mkdtemp(path.join(os.tmpdir(), "shopper-server-outcomes-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const directory = path.join(root, "case-1", "generic"); await mkdir(directory, { recursive: true });
  const savedReceipt = receipt();
  const evidence: Obj = { schema: "shopper-demo-checkout-evidence.v1", synthetic: true,
    post: { url: "http://127.0.0.1:4312/api/demo-checkout", httpStatus: 202,
      request: { schema: "fieldwork.demo-checkout.v1", checkoutId: UUID, items: [{ productId: "bottle", variantId: "graphite", quantity: 1 }], shippingVisible: true },
      body: { receipt: savedReceipt, duplicate: false } },
    get: { url: `http://127.0.0.1:4312/api/demo-checkout/${UUID}`, httpStatus: 200, body: { receipt: structuredClone(savedReceipt) } },
    confirmation: { visible: true, checkoutId: UUID, receiptId: savedReceipt.receiptId }, errors: [] };
  const session: Obj = { caseId: "case-1", arm: "generic", run: { id: "run-1" }, realPurchase: false,
    status: "completed", entryUrl: "http://127.0.0.1:4312/", verification: { catalogSha256: CATALOG, independentOfShopperAgent: true },
    choice: { productId: "bottle", variantId: "graphite", quantity: 1, catalogPrice: 36, observationSource: "independent-dom" }, artifacts: [] };
  const report: Obj = { version: "shopper-evidence-aggregate.v1", synthetic: true, humanConversionClaim: false, fixedSessions: 2,
    catalogSnapshotSha256: CATALOG, rows: [{ expectedRunId: "run-1", caseId: "case-1", arm: "generic", path: "case-1/generic/session.json", status: "recorded", session },
      { expectedRunId: "run-2", caseId: "case-1", arm: "local-model", status: "unattempted", path: "case-1/local-model/session.json" }] };
  mutate?.(evidence, session, report);
  const artifactBytes = encode(evidence), artifactPath = path.join(directory, "checkout-receipt.json");
  session.artifacts.push({ artifact: { id: "run-1-checkout", runId: "run-1", kind: "result", path: "checkout-receipt.json" },
    status: "captured", sha256: sha(artifactBytes), bytes: artifactBytes.length, error: null });
  const sessionBytes = encode(session), sessionPath = path.join(directory, "session.json"); report.rows[0].sidecarSha256 = sha(sessionBytes);
  await writeFile(artifactPath, artifactBytes); await writeFile(sessionPath, sessionBytes);
  const reportPath = path.join(root, "aggregate.json"), outputPath = path.join(root, "index.json"); await writeFile(reportPath, encode(report));
  const calls: { url: string; init: RequestInit | undefined }[] = [];
  const mock = (async (url: URL | RequestInfo, init?: RequestInit) => { calls.push({ url: String(url), init });
    return new Response(JSON.stringify({ receipt: evidence.post?.body.receipt ?? savedReceipt }), { status: 200 }); }) as typeof fetch;
  return { root, reportPath, outputPath, artifactPath, sessionPath, report, evidence, session, calls, mock };
}

test("indexes all assigned rows and confirms only matching durable local receipt", async (t) => {
  const f = await fixture(t), result = await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock });
  assert.equal(result.schema, "shopper-server-outcomes.v1"); assert.equal(result.rows.length, 2);
  assert.equal(result.rows[0].serverConfirmed, true); assert.deepEqual(result.rows[0].errors, []);
  assert.equal(result.rows[0].visibleCheckoutConfirmed, true);
  assert.deepEqual(result.rows[0].readback, { httpStatus: 200, identical: true });
  assert.equal(result.rows[0].artifact?.sha256, sha(await readFile(f.artifactPath)));
  assert.equal(result.reports[0].sha256, sha(await readFile(f.reportPath)));
  assert.equal(result.rows[1].serverConfirmed, false); assert.equal(result.rows[1].visibleCheckoutConfirmed, false); assert.equal(result.rows[1].receipt, null);
  assert.equal(f.calls.length, 1); assert.equal(f.calls[0].init?.method, "GET"); assert.equal(f.calls[0].init?.redirect, "error");
  assert.equal(f.calls[0].url, `http://127.0.0.1:4312/api/demo-checkout/${UUID}`);
  await assert.rejects(generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock }), /EEXIST/);
});

test("receipt confirmation does not depend on agent status or independent gold success", async (t) => {
  const f = await fixture(t, (_e, s, r) => { s.status = "constraint-failure"; r.rows[0].evaluator = { acceptableSelection: false, allHardConstraintsMet: false }; });
  assert.equal((await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock })).rows[0].serverConfirmed, true);
});

test("no-order evidence preserves abstention with no technical error or network call", async (t) => {
  const f = await fixture(t, (e, s) => { e.post = null; e.get = null; e.confirmation = { visible: false, checkoutId: null, receiptId: null }; s.choice = null; s.status = "abandoned"; });
  const row = (await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock })).rows[0];
  assert.equal(row.serverConfirmed, false); assert.equal(row.visibleCheckoutConfirmed, false); assert.ok(row.artifact); assert.deepEqual(row.errors, []); assert.equal(f.calls.length, 0);
});

test("durable accepted order survives missing DOM confirmation and a render timeout", async (t) => {
  const f = await fixture(t, (e, s) => { e.confirmation = { visible: false, checkoutId: null, receiptId: null };
    e.errors = ["Confirmation render timed out"]; s.status = "timeout"; });
  const row = (await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock })).rows[0];
  assert.equal(row.serverConfirmed, true); assert.equal(row.visibleCheckoutConfirmed, false);
  assert.deepEqual(row.errors, ["Confirmation render timed out"]); assert.equal(row.receipt?.subtotal, 36);
  assert.deepEqual(row.readback, { httpStatus: 200, identical: true }); assert.equal(f.calls.length, 1);
});

test("matching visible IDs plus recorded UI errors do not claim valid visible completion", async (t) => {
  const f = await fixture(t, (e) => { e.errors = ["UI render exception"]; });
  const row = (await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock })).rows[0];
  assert.equal(row.serverConfirmed, true); assert.equal(row.visibleCheckoutConfirmed, false); assert.deepEqual(row.errors, ["UI render exception"]);
});

test("missing DOM confirmation alone does not negate a durable server order", async (t) => {
  const f = await fixture(t, (e) => { e.confirmation = { visible: false, checkoutId: null, receiptId: null }; });
  const row = (await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock })).rows[0];
  assert.equal(row.serverConfirmed, true); assert.equal(row.visibleCheckoutConfirmed, false); assert.deepEqual(row.errors, []);
});

test("mismatching visible IDs are recorded without negating the durable order", async (t) => {
  const f = await fixture(t, (e) => { e.confirmation.receiptId = "demo-" + "b".repeat(32); });
  const row = (await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock })).rows[0];
  assert.equal(row.serverConfirmed, true); assert.equal(row.visibleCheckoutConfirmed, false);
  assert.match(row.errors[0], /Visible confirmation IDs differ/); assert.equal(f.calls.length, 1);
});

test("malformed DOM confirmation is separate from verified server existence", async (t) => {
  const f = await fixture(t, (e) => { e.confirmation = null; });
  const row = (await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock })).rows[0];
  assert.equal(row.serverConfirmed, true); assert.equal(row.visibleCheckoutConfirmed, false); assert.match(row.errors[0], /confirmation: object required/);
});

test("hidden shipping retains unknown delivery cost without inventing revenue", async (t) => {
  const f = await fixture(t, (e) => { e.post.request.shippingVisible = false;
    for (const r of [e.post.body.receipt, e.get.body.receipt]) { r.shipping = { status: "hidden", cost: null }; r.total = 36; } });
  const row = (await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock })).rows[0];
  assert.equal(row.serverConfirmed, true); assert.deepEqual(row.receipt?.shipping, { status: "hidden", cost: null });
});

const invalidCases: [string, (e: Obj, s: Obj, r: Obj) => void, RegExp][] = [
  ["POST failure", (e) => { e.post.httpStatus = 500; }, /POST202/],
  ["GET failure", (e) => { e.get.httpStatus = 404; }, /GET200/],
  ["different captured GET", (e) => { e.get.body.receipt.total = 99; }, /POST\/GET receipts differ/],
  ["different request item", (e) => { e.post.request.items[0].productId = "wrong"; }, /Request\/receipt items differ/],
  ["client price injection", (e) => { e.post.request.items[0].unitPrice = 1; }, /unexpected or missing fields/],
  ["price mismatch", (e) => { e.post.body.receipt.items[0].unitPrice = 1; }, /price\/subtotal/],
  ["quantity mismatch", (e) => { e.post.body.receipt.items[0].quantity = 2; }, /identity\/quantity/],
  ["shipping wrong", (e) => { e.post.body.receipt.shipping.cost = 0; }, /Shipping cost/],
  ["not independently observed", (_e, s) => { s.choice.observationSource = "agent-reported"; }, /DOM choice/],
  ["wrong catalog SHA", (_e, s) => { s.verification.catalogSha256 = "a".repeat(64); }, /catalog binding/],
  ["external assigned origin", (e, s) => { s.entryUrl = "https://example.com/"; e.post.url = "https://example.com/api/demo-checkout"; }, /not allowlisted/],
  ["external readback", (e) => { e.get.url = `http://127.0.0.1:4331/api/demo-checkout/${UUID}`; }, /assigned storefront/],
  ["readback query", (e) => { e.get.url += "?extra=1"; }, /assigned storefront/],
  ["readback credentials", (e) => { e.get.url = e.get.url.replace("http://", "http://user:pass@"); }, /assigned storefront/],
  ["real purchase", (e) => { e.post.body.receipt.realPurchase = true; }, /synthetic demo/],
];
for (const [name, mutate, expected] of invalidCases) test(`rejects ${name} before durable request`, async (t) => {
  const f = await fixture(t, mutate), row = (await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock })).rows[0];
  assert.equal(row.serverConfirmed, false); assert.equal(row.visibleCheckoutConfirmed, false); assert.equal(row.receipt, null); assert.match(row.errors.join(" "), expected); assert.equal(f.calls.length, 0);
});

test("checks actual artifact digest and byte length", async (t) => {
  const f = await fixture(t); await writeFile(f.artifactPath, encode({ ...f.evidence, synthetic: false }));
  const row = (await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock })).rows[0];
  assert.match(row.errors[0], /bytes or SHA-256 mismatch/); assert.equal(f.calls.length, 0);
});

test("checks actual sidecar digest", async (t) => {
  const f = await fixture(t); await writeFile(f.sessionPath, encode({ ...f.session, status: "timeout" }));
  const row = (await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock })).rows[0];
  assert.match(row.errors[0], /sidecar SHA-256 mismatch/); assert.equal(f.calls.length, 0);
});

test("requires embedded report session to match hashed sidecar", async (t) => {
  const f = await fixture(t); f.report.rows[0].session.status = "timeout"; await writeFile(f.reportPath, encode(f.report));
  const row = (await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock })).rows[0];
  assert.match(row.errors[0], /embedded session differs/); assert.equal(f.calls.length, 0);
});

test("does not follow a session artifact symlink outside its directory", async (t) => {
  const f = await fixture(t), escaped = path.join(f.root, "escaped.json");
  await writeFile(escaped, await readFile(f.artifactPath)); await rm(f.artifactPath); await symlink(escaped, f.artifactPath);
  const row = (await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: f.mock })).rows[0];
  assert.match(row.errors[0], /symlink escapes/); assert.equal(f.calls.length, 0);
});

test("durable readback network failure remains an unconfirmed assigned zero", async (t) => {
  const f = await fixture(t), row = (await generateServerOutcomes([f.reportPath], f.outputPath,
    { fetch: (async () => { throw new Error("offline"); }) as typeof fetch })).rows[0];
  assert.equal(row.serverConfirmed, false); assert.equal(row.readback, null); assert.match(row.errors[0], /offline/); assert.ok(row.artifact);
});

test("rejects durable missing or changed receipt", async (t) => {
  const f = await fixture(t);
  const missing = await generateServerOutcomes([f.reportPath], f.outputPath, { fetch: (async () => new Response("{}", { status: 404 })) as typeof fetch });
  assert.match(missing.rows[0].errors[0], /HTTP 404/);
  const changed = await generateServerOutcomes([f.reportPath], path.join(f.root, "changed.json"),
    { fetch: (async () => new Response(JSON.stringify({ receipt: { ...receipt(), receiptId: "changed" } }))) as typeof fetch });
  assert.match(changed.rows[0].errors[0], /Durable GET receipt differs/);
});

test("CLI repeats report inputs, forbids duplicates, and requires complete assigned roster", async (t) => {
  const first = await fixture(t), second = await fixture(t, (_e, s, r) => { s.run.id = "run-3"; r.rows[0].expectedRunId = "run-3"; r.rows[1].expectedRunId = "run-4"; });
  // Fixture artifact belongs to run-1: second report is retained but correctly unconfirmed.
  const index = await runServerOutcomesCli(["--report", first.reportPath, "--report", second.reportPath, "--out", first.outputPath], { fetch: first.mock });
  assert.equal(index.rows.length, 4); assert.equal(index.reports.length, 2);
  await assert.rejects(runServerOutcomesCli(["--report", first.reportPath, "--report", first.reportPath, "--out", path.join(first.root, "duplicate.json")], { fetch: first.mock }), /Duplicate report/);
  await assert.rejects(runServerOutcomesCli(["--report", first.reportPath]), /Usage/);
  first.report.fixedSessions = 3; await writeFile(first.reportPath, encode(first.report));
  await assert.rejects(generateServerOutcomes([first.reportPath], path.join(first.root, "bad-roster.json"), { fetch: first.mock }), /All assigned aggregate rows/);
});
