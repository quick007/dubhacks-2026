/** CPU fixtures exercise ingestion only; these files are never real shopper evidence. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { aggregateShopperResults, loadShopperPacket, prepareShopperAssignments, shopperEvalMain } from "./shopper-eval-cli.js";
import { SHOPPER_CONTRACT_VERSION, shopperEntryUrl, verifyShopperNeeds,
  type CatalogSnapshot, type ShopperArm, type ShopperCase, type ShopperSession } from "./shopper-contract.js";

const SOURCE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../experiments/ranker/pretrained/store_shopper_eval_v1");
const sha = (value: Buffer | string) => createHash("sha256").update(value).digest("hex");
const names = ["pair-plan.json", "public-observations.json", "shopper-instructions.json", "evaluator-gold.json", "catalog-binding.json"];
const save = (file: string, value: unknown) => writeFile(file, `${JSON.stringify(value, null, 2)}\n`);
async function fixture(run: (root: string, packet: string) => Promise<void>) {
  const root = await mkdtemp(path.join(os.tmpdir(), "shopper-cli-fixture-")), packet = path.join(root, "packet");
  await mkdir(packet);
  try {
    const files: Record<string, string> = {};
    for (const name of names) {
      const bytes = await readFile(path.join(SOURCE, name)); await writeFile(path.join(packet, name), bytes); files[name] = sha(bytes);
    }
    await save(path.join(packet, "manifest.json"), { version: "actual16-paired-shopper-packet.v1", files });
    await run(root, packet);
  } finally { await rm(root, { recursive: true, force: true }); }
}

async function sessionFixture(c: ShopperCase, arm: ShopperArm, catalog: CatalogSnapshot, directory: string,
  beforeNavigation = false): Promise<ShopperSession> {
  await mkdir(directory, { recursive: true });
  const runId = `${c.caseId}-${arm}`, stamp = "2026-10-18T12:00:00.000Z", artifactId = `${runId}-screenshot`;
  const bytes = Buffer.from("CPU test fixture; no Chrome execution");
  if (!beforeNavigation) await writeFile(path.join(directory, "screenshot.fixture"), bytes);
  const screenshot = { artifact: { id: artifactId, runId, kind: "screenshot" as const, path: "screenshot.fixture", mimeType: "text/plain", createdAt: stamp },
    status: beforeNavigation ? "missing" as const : "captured" as const, sha256: beforeNavigation ? null : sha(bytes),
    bytes: beforeNavigation ? 0 : bytes.length, error: beforeNavigation ? "Browser did not open; CPU fixture" : null };
  const trace = { artifact: { id: `${runId}-trace`, runId, kind: "trace" as const, path: "trace.zip", mimeType: "application/zip", createdAt: stamp },
    status: "missing" as const, sha256: null, bytes: 0, error: "No trace captured; CPU fixture" };
  const product = catalog.products.find((p) => p.productId === "bottle")!, variant = product.variants[0];
  const choice = beforeNavigation ? null : { productId: product.productId, variantId: variant.variantId, originalName: product.originalName,
    catalogPrice: product.itemPrice, selectedColor: variant.selectedColor, quantity: 1 as const, observationSource: "agent-reported" as const,
    sourceStepIds: [`${runId}-2`], evidenceArtifactIds: [artifactId] };
  const actions = [{ type: "navigate" as const, url: shopperEntryUrl(arm, "http://127.0.0.1:4312/") },
    { type: "choose-variant" as const, productId: product.productId, variantId: variant.variantId },
    { type: "add-to-cart" as const, productId: product.productId, variantId: variant.variantId },
    { type: "confirm-demo-order" as const, target: "demo confirmation" }];
  const evidenceIds = beforeNavigation ? [] : [artifactId];
  return { contractVersion: SHOPPER_CONTRACT_VERSION, caseId: c.caseId, arm,
    run: { id: runId, experimentId: "CPU-fixture", personaId: c.caseId, personaSplit: c.split, seed: c.seed },
    browser: { name: "Chrome", headed: true, actualNavigation: !beforeNavigation, version: beforeNavigation ? null : "CPU fixture version", automation: "cua" },
    isolation: beforeNavigation ? { kind: "not-created", tabId: null, contextId: null, sharedStorage: null, storageCleared: null,
      initialCartEmpty: null, evidenceArtifactIds: [], note: "Browser not created; CPU fixture" }
      : { kind: "fresh-tab", tabId: `${runId}-tab`, contextId: "shared-fixture", sharedStorage: true, storageCleared: false,
        initialCartEmpty: true, evidenceArtifactIds: [artifactId], note: "Fresh tab shares storage; CPU fixture" },
    entryUrl: shopperEntryUrl(arm, "http://127.0.0.1:4312/"), startedAt: stamp, elapsedMs: beforeNavigation ? 0 : 100,
    timingPrecision: { sessionElapsed: "measured-monotonic", stepElapsed: "measured-monotonic", source: "CPU fixture only" },
    protocol: { passed: true, deviations: [] },
    status: beforeNavigation ? "technical-failure" : "completed", error: beforeNavigation ? "CPU fixture launch failure" : null,
    realPurchase: false, fakeOrderConfirmed: !beforeNavigation,
    steps: beforeNavigation ? [] : actions.map((action, index) => ({ elapsedMs: index * 20, action,
      event: { id: `${runId}-${index}`, runId, sequence: index, at: stamp, kind: index === 0 ? "navigation" : "action", message: "CPU fixture action" } })),
    artifacts: [screenshot, trace], choice,
    verification: { verifierId: "independent-CPU-fixture", independentOfShopperAgent: true, catalogSha256: catalog.sha256,
      hardConstraints: verifyShopperNeeds(c.privateShopperNeeds.hardConstraints, choice, catalog).map((r) => ({ ...r, evidenceArtifactIds: evidenceIds })),
      preferences: verifyShopperNeeds(c.privateShopperNeeds.preferences, choice, catalog).map((r) => ({ ...r, evidenceArtifactIds: evidenceIds })) },
    model: { status: arm === "generic" ? "not-enabled" : "not-observed", readiness: [] },
    fallback: { used: false, kind: "none", reason: null } };
}

void test("prepare writes twenty private/public assignments without reading evaluator files or dispatching", async () => fixture(async (root, packet) => {
  await rm(path.join(packet, "evaluator-gold.json")); await rm(path.join(packet, "catalog-binding.json"));
  const output = path.join(root, "assignments"), result = await prepareShopperAssignments(packet, output);
  assert.equal(result.fixedSessions, 20); assert.equal(result.evaluatorFilesRead, false);
  assert.equal(result.browserDispatched, false); assert.equal(result.modelCalled, false);
  assert.equal(result.assignments.length, 20);
  for (const row of result.assignments) {
    const privateAssignment = JSON.parse(await readFile(path.join(output, row.privateAssignment), "utf8"));
    const publicContext = JSON.parse(await readFile(path.join(output, row.publicContext), "utf8"));
    assert.ok(privateAssignment.shopperNeeds.goal); assert.equal("shopperNeeds" in publicContext, false);
    assert.equal("acceptableProductIds" in privateAssignment, false); assert.equal("productId" in privateAssignment, false);
    assert.equal(privateAssignment.recommenderContext.query, publicContext.recommenderContext.query);
    assert.equal(new URL(publicContext.entryUrl).origin, "http://127.0.0.1:4312");
  }
  await assert.rejects(prepareShopperAssignments(packet, output), /EEXIST/);
}));

void test("all missing sidecars retain twenty unattempted rows and ten denominators", async () => fixture(async (root, packet) => {
  const report = await aggregateShopperResults(packet, path.join(root, "missing-sessions"), path.join(root, "report.json"));
  assert.deepEqual(report.setupErrors, []); assert.equal(report.status, "partial"); assert.equal(report.rows.length, 20);
  assert.ok(report.rows.every((row) => row.status === "unattempted"));
  const arms = report.arms as Record<string, { fixedDenominator: number; unattempted: number; evidenceVerifiedExpectedOutcomeRateFixedDenominator: number }>;
  for (const arm of ["generic", "local-model"]) { assert.equal(arms[arm].fixedDenominator, 10); assert.equal(arms[arm].unattempted, 10); assert.equal(arms[arm].evidenceVerifiedExpectedOutcomeRateFixedDenominator, 0); }
  const empty = (report.arms as Record<string, { attemptsActionCounts: number[]; attemptsPreferenceFit: unknown[];
    actionCountSummary: { total: number; minimum: number | null; maximum: number | null; mean: number | null } }>).generic;
  assert.deepEqual(empty.attemptsActionCounts, []); assert.deepEqual(empty.attemptsPreferenceFit, []);
  assert.equal(empty.actionCountSummary.total, 0); assert.equal(empty.actionCountSummary.minimum, null);
  assert.equal(empty.actionCountSummary.maximum, null); assert.equal(empty.actionCountSummary.mean, null);
}));

void test("actual-sidecar schema ingestion keeps missing trace distinct from captured evidence", async () => fixture(async (root, packet) => {
  const loaded = await loadShopperPacket(packet), c = loaded.cases.find((row) => row.caseId === "shopper-pair-02")!;
  const catalog = JSON.parse(await readFile(path.join(packet, "catalog-binding.json"), "utf8")).catalogSnapshot as CatalogSnapshot;
  const sessions = path.join(root, "sessions"), directory = path.join(sessions, c.caseId, "generic");
  const session = await sessionFixture(c, "generic", catalog, directory); await save(path.join(directory, "session.json"), session);
  const report = await aggregateShopperResults(packet, sessions, path.join(root, "result.json"));
  const row = report.rows.find((item) => item.caseId === c.caseId && item.arm === "generic")!;
  assert.equal(row.status, "recorded"); assert.equal(row.evaluator?.expectedOutcomeMet, true);
  assert.equal(row.evaluator?.evidenceVerified, false); assert.deepEqual(row.evaluator?.missingEvidenceKinds, ["trace"]);
  assert.ok(row.errors.some((error) => error.includes("No trace captured")));
  assert.equal(report.rows.filter((item) => item.status === "unattempted").length, 19);
  await writeFile(path.join(directory, "screenshot.fixture"), "corrupted fixture screenshot");
  const corrupted = await aggregateShopperResults(packet, sessions, path.join(root, "corrupt-result.json"));
  assert.ok(corrupted.rows.find((item) => item.caseId === c.caseId && item.arm === "generic")!.errors.some((error) => error.includes("SHA-256 mismatch")));
}));

void test("pre-navigation technical failures remain recorded failures with zero fabricated actions", async () => fixture(async (root, packet) => {
  const loaded = await loadShopperPacket(packet), c = loaded.cases[1];
  const catalog = JSON.parse(await readFile(path.join(packet, "catalog-binding.json"), "utf8")).catalogSnapshot as CatalogSnapshot;
  const sessions = path.join(root, "sessions"), directory = path.join(sessions, c.caseId, "local-model");
  await save(path.join(directory, "session.json"), await sessionFixture(c, "local-model", catalog, directory, true));
  const report = await aggregateShopperResults(packet, sessions, path.join(root, "result.json"));
  const row = report.rows.find((item) => item.caseId === c.caseId && item.arm === "local-model")!;
  assert.equal(row.status, "recorded"); assert.equal(row.outcome, "technical-failure");
  assert.equal(row.session?.steps.length, 0); assert.equal(row.evaluator?.expectedOutcomeMet, false);
}));

void test("exploratory completions preserve observations without filling formal roster slots", async () => fixture(async (root, packet) => {
  const loaded = await loadShopperPacket(packet), c = loaded.cases[1];
  const catalog = JSON.parse(await readFile(path.join(packet, "catalog-binding.json"), "utf8")).catalogSnapshot as CatalogSnapshot;
  const sessions = path.join(root, "sessions"), directory = path.join(sessions, c.caseId, "generic");
  const session = await sessionFixture(c, "generic", catalog, directory);
  session.protocol = { passed: false, deviations: ["Provider viewport differs from predeclared1280x960; CPU fixture"] };
  session.timingPrecision = { sessionElapsed: "upper-bound", stepElapsed: "not-recorded", source: "CPU fixture upper bound" };
  session.steps.forEach((step) => { step.elapsedMs = null; step.event.at = null; });
  await save(path.join(directory, "session.json"), session);
  const report = await aggregateShopperResults(packet, sessions, path.join(root, "exploratory.json"));
  const generic = (report.arms as Record<string, Record<string, number>>).generic;
  assert.equal(generic.reportedCompleted, 1); assert.equal(generic.completed, 0);
  assert.equal(generic.exploratoryRecorded, 1); assert.equal(generic.formalUnattempted, 10);
  assert.equal(generic.evidenceVerifiedExpectedOutcome, 0); assert.equal(generic.verifiedDomCompleted, 0);
  assert.equal(report.rows.find((row) => row.caseId === c.caseId && row.arm === "generic")?.formalStatus, "unattempted");
  assert.equal(report.status, "partial"); assert.equal("completePairedContract" in report, false);
}));

void test("action and preference summaries retain completed, unknown and zero-action attempts", async () => fixture(async (root, packet) => {
  const loaded = await loadShopperPacket(packet);
  const catalog = JSON.parse(await readFile(path.join(packet, "catalog-binding.json"), "utf8")).catalogSnapshot as CatalogSnapshot;
  const sessions = path.join(root, "sessions");
  for (const caseId of ["shopper-pair-01", "shopper-pair-02", "shopper-pair-03", "shopper-pair-10"]) {
    const c = loaded.cases.find((item) => item.caseId === caseId)!;
    const directory = path.join(sessions, c.caseId, "generic");
    await save(path.join(directory, "session.json"), await sessionFixture(c, "generic", catalog, directory,
      caseId === "shopper-pair-01" || caseId === "shopper-pair-10"));
  }
  const report = await aggregateShopperResults(packet, sessions, path.join(root, "summaries.json"));
  const generic = (report.arms as Record<string, { attemptsActionCounts: number[];
    attemptsPreferenceFit: { caseId: string; runId: string; met: number; known: number; total: number }[];
    actionCountSummary: { total: number; minimum: number; maximum: number; mean: number } }>).generic;
  assert.deepEqual(generic.attemptsActionCounts, [0, 4, 4, 0]);
  assert.equal(generic.actionCountSummary.total, 8); assert.equal(generic.actionCountSummary.minimum, 0);
  assert.equal(generic.actionCountSummary.maximum, 4); assert.equal(generic.actionCountSummary.mean, 2);
  assert.deepEqual(generic.attemptsPreferenceFit, [
    { caseId: "shopper-pair-01", runId: "shopper-pair-01-generic", met: 0, known: 0, total: 0 },
    { caseId: "shopper-pair-02", runId: "shopper-pair-02-generic", met: 1, known: 1, total: 1 },
    { caseId: "shopper-pair-03", runId: "shopper-pair-03-generic", met: 0, known: 1, total: 1 },
    { caseId: "shopper-pair-10", runId: "shopper-pair-10-generic", met: 0, known: 0, total: 1 },
  ]);
}));

void test("budget boundaries pass while action or wall overruns cannot increase formal success", async () => fixture(async (root, packet) => {
  const loaded = await loadShopperPacket(packet), c = loaded.cases.find((item) => item.caseId === "shopper-pair-02")!;
  const catalog = JSON.parse(await readFile(path.join(packet, "catalog-binding.json"), "utf8")).catalogSnapshot as CatalogSnapshot;
  const sessions = path.join(root, "sessions"), directory = path.join(sessions, c.caseId, "generic");
  for (const [name, actionCount, elapsedMs, expectedFormal] of [
    ["boundary", 24, 120000, 1], ["actions", 25, 100, 0], ["wall", 4, 120001, 0],
  ] as const) {
    const session = await sessionFixture(c, "generic", catalog, directory);
    while (session.steps.length < actionCount) {
      const sequence = session.steps.length;
      session.steps.push({ event: { id: `${session.run.id}-${sequence}`, runId: session.run.id, sequence,
        at: session.startedAt, kind: "action", message: "CPU fixture inspection" }, elapsedMs: 80,
        action: { type: "inspect", target: "CPU fixture product" } });
    }
    session.elapsedMs = elapsedMs; session.choice!.observationSource = "independent-dom";
    const traceBytes = Buffer.from("CPU fixture trace; no browser execution");
    await writeFile(path.join(directory, "trace.zip"), traceBytes);
    Object.assign(session.artifacts.find((artifact) => artifact.artifact.kind === "trace")!, {
      status: "captured", sha256: sha(traceBytes), bytes: traceBytes.length, error: null,
    });
    await save(path.join(directory, "session.json"), session);
    const report = await aggregateShopperResults(packet, sessions, path.join(root, `${name}.json`));
    const row = report.rows.find((item) => item.caseId === c.caseId && item.arm === "generic")!;
    const generic = (report.arms as Record<string, Record<string, number>>).generic;
    assert.equal(row.status, "recorded"); assert.equal(row.session?.protocol.passed, true);
    assert.equal(row.evaluator?.expectedOutcomeMet, true); assert.equal(row.evaluator?.evidenceVerified, true);
    assert.equal(row.evaluator?.budgetRespected, expectedFormal === 1);
    assert.equal(row.formalStatus, expectedFormal ? "recorded" : "unattempted");
    assert.equal(generic.reportedCompleted, 1); assert.equal(generic.observedExpectedOutcome, 1);
    assert.equal(generic.completed, expectedFormal); assert.equal(generic.formalRecorded, expectedFormal);
    assert.equal(generic.formalUnattempted, 10 - expectedFormal); assert.equal(generic.budgetViolations, 1 - expectedFormal);
    assert.equal(generic.evidenceVerifiedExpectedOutcome, expectedFormal); assert.equal(generic.verifiedDomCompleted, expectedFormal);
    assert.equal(row.errors.some((error) => error.includes("Shopper budget exceeded")), expectedFormal === 0);
  }
}));

void test("an over-budget receipt cannot complete the formal twenty-session suite", async () => fixture(async (root, packet) => {
  const loaded = await loadShopperPacket(packet);
  const catalog = JSON.parse(await readFile(path.join(packet, "catalog-binding.json"), "utf8")).catalogSnapshot as CatalogSnapshot;
  const sessions = path.join(root, "sessions");
  for (const c of loaded.cases) {
    for (const arm of c.armOrder) {
      const directory = path.join(sessions, c.caseId, arm);
      const session = await sessionFixture(c, arm, catalog, directory, true);
      if (c.caseId === "shopper-pair-02" && arm === "generic") session.elapsedMs = 120001;
      await save(path.join(directory, "session.json"), session);
    }
  }
  const report = await aggregateShopperResults(packet, sessions, path.join(root, "suite.json"));
  assert.equal(report.rows.filter((row) => row.status === "recorded").length, 20);
  assert.equal(report.rows.filter((row) => row.formalStatus === "recorded").length, 19);
  assert.equal(report.status, "partial"); assert.equal("completePairedContract" in report, false);
}));

void test("malformed sidecar is retained as invalid and raw result logs are never synthesized", async () => fixture(async (root, packet) => {
  const sessions = path.join(root, "sessions"), directory = path.join(sessions, "shopper-pair-02", "generic");
  await mkdir(directory, { recursive: true }); await save(path.join(directory, "result.json"), { rawManualLog: true });
  const first = await aggregateShopperResults(packet, sessions, path.join(root, "before.json"));
  assert.ok(first.rows.every((row) => row.status === "unattempted"));
  await writeFile(path.join(directory, "session.json"), '{"fake":true}');
  const second = await aggregateShopperResults(packet, sessions, path.join(root, "after.json"));
  const row = second.rows.find((item) => item.caseId === "shopper-pair-02" && item.arm === "generic")!;
  assert.equal(row.status, "invalid-sidecar"); assert.ok(row.sidecarSha256); assert.ok(row.errors.length);
}));

void test("changed packet/evaluator bindings fail closed without dropping the fixed denominator", async () => fixture(async (root, packet) => {
  const gold = path.join(packet, "evaluator-gold.json"); await writeFile(gold, "[]");
  const report = await aggregateShopperResults(packet, path.join(root, "sessions"), path.join(root, "blocked.json"));
  assert.equal(report.status, "blocked"); assert.equal(report.rows.length, 20); assert.ok(report.setupErrors.length);
  await writeFile(path.join(packet, "public-observations.json"), "[]");
  await assert.rejects(loadShopperPacket(packet), /packet file changed/);
}));

void test("CLI exposes only prepare/aggregate and refuses unknown or repeated options", async () => {
  await assert.rejects(shopperEvalMain(["dispatch"]), /no browser\/model dispatch/);
  await assert.rejects(shopperEvalMain(["prepare", "--packet", "x", "--out", "y", "--gpu", "yes"]), /Unknown/);
  await assert.rejects(shopperEvalMain(["aggregate", "--packet", "x", "--packet", "y"]), /Unknown/);
  await assert.rejects(shopperEvalMain(["aggregate", "--packet", "x", "--out", "y"]), /sessions/);
});
