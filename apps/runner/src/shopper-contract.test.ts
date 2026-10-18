import assert from "node:assert/strict";
import { test } from "node:test";
import { createHash } from "node:crypto";
import { mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { buildShopperAssignment, shopperEntryUrl, SHOPPER_CONTRACT_VERSION, validatePairedShopperSuite,
  validateShopperCase, validateShopperCatalog, validateShopperSession, verifyShopperArtifactFiles, verifyShopperNeeds,
  type CatalogSnapshot, type ShopperArm, type ShopperCase, type ShopperSession } from "./shopper-contract.js";

const digest = "a".repeat(64), timestamp = "2026-10-18T12:00:00.000Z";
const catalog: CatalogSnapshot = { sha256: digest, products: [{ productId: "bottle", originalName: "Store bottle", itemPrice: 36,
  facts: { functions: ["drinking-water bottle"], waterproofDeclared: false },
  variants: [{ variantId: "bottle-blue", selectedColor: "Blue" }, { variantId: "bottle-cream", selectedColor: "Cream" }] }] };
const shopperCase: ShopperCase = { contractVersion: SHOPPER_CONTRACT_VERSION, caseId: "pair-01", split: "development", seed: 7,
  armOrder: ["generic", "local-model"], privateShopperNeeds: { goal: "Choose one bottle that fits my needs, then complete a fake order.",
    hardConstraints: [{ id: "budget", label: "Below40", attribute: "itemPrice", operator: "less-than", value: 40 }],
    preferences: [{ id: "drink", label: "For water", attribute: "functions", operator: "includes", value: "drinking-water bottle" }] },
  observableContext: { query: "a bottle for water", itemBudget: null, hardRequirements: [] } };

function session(arm: ShopperArm = "generic", c = shopperCase, index = 0): ShopperSession {
  const runId = `run-${c.caseId}-${arm}`, entryUrl = shopperEntryUrl(arm);
  const artifact = (kind: "screenshot" | "trace", captured: boolean) => ({ artifact: { id: `${runId}-${kind}`, runId,
    kind, path: `${kind}.${kind === "trace" ? "zip" : "png"}`, mimeType: kind === "trace" ? "application/zip" : "image/png", createdAt: timestamp },
    status: captured ? "captured" as const : "missing" as const, sha256: captured ? digest : null,
    bytes: captured ? 10 : 0, error: captured ? null : "Chrome UI adapter supplied no trace export; screenshot retained." });
  const screenshot = `${runId}-screenshot`;
  const choice = { productId: "bottle", variantId: "bottle-blue", originalName: "Store bottle", catalogPrice: 36,
    selectedColor: "Blue", quantity: 1 as const, observationSource: "independent-dom" as const,
    sourceStepIds: ["step-2"], evidenceArtifactIds: [screenshot] };
  const actions = [{ type: "navigate" as const, url: entryUrl }, { type: "search" as const, query: c.observableContext.query },
    { type: "add-to-cart" as const, productId: "bottle", variantId: "bottle-blue" }, { type: "confirm-demo-order" as const, target: "Demo order confirmed" }];
  return { contractVersion: SHOPPER_CONTRACT_VERSION, caseId: c.caseId, arm,
    run: { id: runId, experimentId: "paired-shopper-v1", personaId: c.caseId, personaSplit: c.split, seed: c.seed },
    browser: { name: "Chrome", headed: true, actualNavigation: true, version: "test-fixture", automation: "cua" },
    isolation: { kind: "fresh-tab", tabId: `tab-${index}-${arm}`, contextId: "existing-chrome-profile",
      sharedStorage: true, storageCleared: false, initialCartEmpty: true, evidenceArtifactIds: [screenshot], note: "Fresh Chrome tab; origin storage shared." },
    entryUrl, startedAt: timestamp, elapsedMs: 100,
    timingPrecision: { sessionElapsed: "measured-monotonic", stepElapsed: "measured-monotonic", source: "CPU fixture monotonic clock" },
    protocol: { passed: true, deviations: [] },
    status: "completed", error: null, realPurchase: false, fakeOrderConfirmed: true,
    steps: actions.map((action, sequence) => ({ event: { id: `step-${sequence}`, runId, sequence, at: timestamp,
      kind: sequence === 0 ? "navigation" : "action", message: "Observed fixture action" }, elapsedMs: sequence * 20, action })),
    artifacts: [artifact("screenshot", true), artifact("trace", false)], choice,
    verification: { verifierId: "independent-catalog-check", independentOfShopperAgent: true, catalogSha256: digest,
      hardConstraints: verifyShopperNeeds(c.privateShopperNeeds.hardConstraints, choice, catalog).map((n) => ({ ...n, evidenceArtifactIds: [screenshot] })),
      preferences: verifyShopperNeeds(c.privateShopperNeeds.preferences, choice, catalog).map((n) => ({ ...n, evidenceArtifactIds: [screenshot] })) },
    model: { status: arm === "generic" ? "not-enabled" : "not-observed", readiness: [] },
    fallback: { used: false, kind: "none", reason: null } };
}

void test("shopper assignment separates private needs from observable request and contains no gold IDs", () => {
  const assignment = buildShopperAssignment(shopperCase, "local-model", "http://127.0.0.1:4312/");
  assert.equal(assignment.entryUrl, "http://127.0.0.1:4312/?model=local");
  assert.deepEqual(assignment.recommenderContext, shopperCase.observableContext);
  assert.deepEqual(assignment.shopperNeeds, shopperCase.privateShopperNeeds);
  assert.equal("run" in assignment, false); assert.equal("gold" in assignment, false);
  assert.equal(JSON.stringify(assignment).includes('"productId"'), false);
  assert.match(assignment.instructions.join(" "), /existing deterministic query\/search behavior/);
  assert.throws(() => buildShopperAssignment({ ...shopperCase, expectedProductIds: ["bottle"] }, "generic"), /Unexpected/);
  assert.throws(() => shopperEntryUrl("generic", "https://example.com/"), /local shopper base/);
  const bad = structuredClone(shopperCase); bad.privateShopperNeeds.hardConstraints[0].attribute = "productId";
  assert.throws(() => validateShopperCase(bad), /Gold identity/);
});

void test("real Chrome session validates while trace absence stays explicit", () => {
  const parsed = validateShopperSession(session(), shopperCase, catalog);
  assert.equal(parsed.isolation.kind, "fresh-tab"); assert.equal(parsed.isolation.sharedStorage, true);
  assert.equal(parsed.artifacts[1].status, "missing"); assert.equal(parsed.artifacts[1].sha256, null);
  assert.equal(parsed.choice?.catalogPrice, 36);
  const bad = session(); bad.browser.headed = false as true;
  assert.throws(() => validateShopperSession(bad, shopperCase, catalog), /headed Chrome/);
  bad.browser.headed = true; bad.isolation.sharedStorage = false;
  assert.throws(() => validateShopperSession(bad, shopperCase, catalog), /shared storage honestly/);
});

void test("actual choice must match independent product, variant, price and color", () => {
  for (const patch of [{ productId: "gold-other" }, { variantId: "wrong-color" }, { catalogPrice: 35 }, { selectedColor: "Cream" }, { originalName: "Other bottle" }]) {
    const bad = session(); Object.assign(bad.choice!, patch);
    assert.throws(() => validateShopperSession(bad, shopperCase, catalog), /catalog/);
  }
  const bad = session(); bad.steps[2].action = { type: "add-to-cart", productId: "bottle", variantId: "bottle-cream" };
  assert.throws(() => validateShopperSession(bad, shopperCase, catalog), /actual cart action/);
});

void test("need checks use typed catalog facts; unknown stock and text substrings never become proof", () => {
  const choice = session().choice;
  const checks = verifyShopperNeeds([
    { id: "stock", label: "In stock", attribute: "stockVerified", operator: "equals", value: true },
    { id: "exact", label: "36inclusive", attribute: "itemPrice", operator: "at-most", value: 36 },
    { id: "strict", label: "below36", attribute: "itemPrice", operator: "less-than", value: 36 },
    { id: "no-waterproof", label: "Waterproof", attribute: "waterproofDeclared", operator: "equals", value: true },
  ], choice, catalog);
  assert.deepEqual(checks.map((n) => n.status), ["unknown", "met", "not-met", "not-met"]);
  const unsupported = { ...catalog, products: [{ ...catalog.products[0], facts: { functions: "not a drinking-water bottle" } }] };
  assert.equal(verifyShopperNeeds(shopperCase.privateShopperNeeds.preferences, choice, unsupported)[0].status, "unknown");
  const bad = session(); bad.verification.hardConstraints[0].status = "unknown";
  assert.throws(() => validateShopperSession(bad, shopperCase, catalog), /disagrees/);
  assert.throws(() => validateShopperCatalog({ ...catalog, products: [{ ...catalog.products[0], facts: { itemPrice: 1 } }] }), /override/);
});

void test("browser actions keep monotonic times and cannot disclose private search needs", () => {
  const bad = session(); bad.steps[2].elapsedMs = 1;
  assert.throws(() => validateShopperSession(bad, shopperCase, catalog), /monotonic/);
  bad.steps[2].elapsedMs = 40; bad.steps[1].action = { type: "search", query: "My private hidden need" };
  assert.throws(() => validateShopperSession(bad, shopperCase, catalog), /assigned observable query/);
  bad.steps[1].action = { type: "navigate", url: shopperEntryUrl("local-model") }; bad.steps[1].event.kind = "navigation";
  assert.throws(() => validateShopperSession(bad, shopperCase, catalog), /paired model arm/);
});

void test("navigation rejects private side channels while preserving public storefront routes", () => {
  for (const arm of ["generic", "local-model"] as const) {
    const entry = shopperEntryUrl(arm);
    const query = encodeURIComponent(shopperCase.observableContext.query);
    for (const route of ["#collection", `#collection?q=${query}&c=food`, `#cart?q=${query}`, "#confirmed"]) {
      const s = session(arm);
      s.steps[1].action = { type: "navigate", url: entry + route }; s.steps[1].event.kind = "navigation";
      assert.doesNotThrow(() => validateShopperSession(s, shopperCase, catalog));
    }
    for (const suffix of [
      `${arm === "generic" ? "?" : "&"}privateNeed=hidden`,
      `${arm === "generic" ? "?" : "&"}headline=hidden`,
      `${arm === "generic" ? "?" : "&"}model=other`,
      "#private-hidden-need", "#collection?privateNeed=hidden", `#collection?q=${query}&q=hidden`,
      "#collection?c=private-hidden-need", "#collection?c=food&c=home", `#collection?q=${query}?hidden`,
    ]) {
      const s = session(arm);
      s.steps[1].action = { type: "navigate", url: entry + suffix }; s.steps[1].event.kind = "navigation";
      assert.throws(() => validateShopperSession(s, shopperCase, catalog), /parameters|model arm|hash routes|collections/);
    }
    const s = session(arm), credentials = new URL(entry); credentials.username = "private"; credentials.password = "hidden";
    s.steps[1].action = { type: "navigate", url: credentials.href }; s.steps[1].event.kind = "navigation";
    assert.throws(() => validateShopperSession(s, shopperCase, catalog), /without credentials/);
  }
});

void test("readiness retains first/full counts and rejects ranking-only complete or physical paint claims", () => {
  const s = session("local-model");
  s.model.status = "complete";
  assert.throws(() => validateShopperSession(s, shopperCase, catalog), /not ranking-only/);
  s.model.readiness.push({ name: "fieldwork.local-model.complete.after-paint-opportunity", stage: "complete", requestId: "browser-request",
    startTimeMs: 10, durationMs: 50, capturedAtElapsedMs: 75, acceptedFields: 18, withheldFields: 2, requiredFields: 20,
    source: "data-local-model-readiness", artifactId: s.artifacts[0].artifact.id,
    scope: "after-paint-opportunity", physicalPaintMeasured: false, imageTransferMeasured: false });
  assert.equal(validateShopperSession(s, shopperCase, catalog).model.readiness[0].acceptedFields, 18);
  s.model.readiness[0].physicalPaintMeasured = true as false;
  assert.throws(() => validateShopperSession(s, shopperCase, catalog), /physical paint claims/);
});

void test("prelaunch failures retain honest null browser observations and no fake successful choice", () => {
  const s = session(); s.status = "technical-failure"; s.error = "Chrome tool unavailable";
  s.browser.actualNavigation = false; s.browser.version = null;
  s.isolation = { kind: "not-created", tabId: null, contextId: null, sharedStorage: null, storageCleared: null,
    initialCartEmpty: null, evidenceArtifactIds: [], note: "No browser action completed." };
  s.steps = []; s.choice = null; s.fakeOrderConfirmed = false;
  s.artifacts.forEach((a) => { a.status = "missing"; a.sha256 = null; a.bytes = 0; a.error = "No capture available"; });
  for (const key of ["hardConstraints", "preferences"] as const) s.verification[key] = verifyShopperNeeds(shopperCase.privateShopperNeeds[key], null, catalog).map((n) => ({ ...n, evidenceArtifactIds: [] }));
  assert.equal(validateShopperSession(s, shopperCase, catalog).status, "technical-failure");
  s.status = "completed";
  assert.throws(() => validateShopperSession(s, shopperCase, catalog), /Prelaunch failure/);
});

void test("unknown or nonempty initial cart is retained only as a disclosed technical failure or timeout", () => {
  for (const initialCartEmpty of [null, false]) {
    for (const status of ["technical-failure", "timeout"] as const) {
      const s = session();
      s.status = status; s.error = "Initial empty cart could not be verified";
      s.protocol = { passed: false, deviations: ["Initial empty cart was not verified"] };
      s.isolation.initialCartEmpty = initialCartEmpty;
      s.steps = s.steps.slice(0, 1); s.choice = null; s.fakeOrderConfirmed = false;
      for (const key of ["hardConstraints", "preferences"] as const) {
        s.verification[key] = verifyShopperNeeds(shopperCase.privateShopperNeeds[key], null, catalog)
          .map((need) => ({ ...need, evidenceArtifactIds: [s.artifacts[0].artifact.id] }));
      }
      assert.equal(validateShopperSession(s, shopperCase, catalog).isolation.initialCartEmpty, initialCartEmpty);
      s.protocol = { passed: true, deviations: [] };
      assert.throws(() => validateShopperSession(s, shopperCase, catalog), /empty initial cart/);
      s.protocol = { passed: false, deviations: [] };
      assert.throws(() => validateShopperSession(s, shopperCase, catalog), /empty initial cart/);
      s.protocol.deviations = ["Initial empty cart was not verified"];
      for (const outcome of ["completed", "abandoned", "constraint-failure"] as const) {
        s.status = outcome;
        assert.throws(() => validateShopperSession(s, shopperCase, catalog), /empty initial cart/);
      }
    }
  }
});

void test("CUA unknown provider metadata and untimed actions stay null with an honest upper bound", () => {
  const s = session(); s.browser.version = null; s.isolation.contextId = null;
  s.timingPrecision = { sessionElapsed: "upper-bound", stepElapsed: "not-recorded", source: "Host start/end tool wall time; no per-action clock exposed" };
  s.steps.forEach((step) => { step.elapsedMs = null; step.event.at = null; });
  s.choice!.observationSource = "agent-reported";
  s.protocol = { passed: false, deviations: ["Exploratory viewport differed from frozen pair protocol"] };
  s.artifacts[0].artifact.kind = "result"; s.artifacts[0].artifact.mimeType = "application/json"; s.artifacts[0].artifact.path = "raw-result.json";
  s.artifacts.push({ artifact: { ...s.artifacts[1].artifact, id: "screenshot-not-exported", kind: "screenshot", path: "missing.png", mimeType: "image/png" },
    status: "missing", sha256: null, bytes: 0, error: "Screenshot exists only in transcript; no durable file exposed" });
  const parsed = validateShopperSession(s, shopperCase, catalog);
  assert.equal(parsed.browser.version, null); assert.equal(parsed.isolation.contextId, null);
  assert.equal(parsed.steps[0].elapsedMs, null); assert.equal(parsed.steps[0].event.at, null);
  assert.equal(parsed.choice?.observationSource, "agent-reported"); assert.equal(parsed.timingPrecision.sessionElapsed, "upper-bound");
  s.timingPrecision.stepElapsed = "measured-monotonic";
  assert.throws(() => validateShopperSession(s, shopperCase, catalog), /precision disagrees/);
});

void test("fixed ten-pair denominator preserves failed attempts and rejects reused tabs", () => {
  const cases = Array.from({ length: 10 }, (_, i) => ({ ...structuredClone(shopperCase), caseId: `pair-${i}`, seed: i,
    armOrder: i % 2 ? ["local-model", "generic"] as [ShopperArm, ShopperArm] : ["generic", "local-model"] as [ShopperArm, ShopperArm] }));
  const sessions = cases.flatMap((c, i) => c.armOrder.map((arm) => session(arm, c, i)));
  sessions[0].status = "abandoned";
  const summary = validatePairedShopperSuite(cases, sessions, catalog);
  assert.equal(summary.fixedSessions, 20); assert.equal(summary.arms.generic.fixedDenominator, 10);
  assert.equal(summary.arms.generic.completedRate, .9); assert.equal(summary.arms["local-model"].completedRate, 1);
  sessions[1].protocol = { passed: false, deviations: ["Exploratory smoke, viewport not controlled"] };
  const exploratory = validatePairedShopperSuite(cases, sessions, catalog);
  assert.equal(exploratory.arms["local-model"].completedRate, .9);
  assert.equal(exploratory.arms["local-model"].reportedCompleted, 10);
  assert.equal(exploratory.arms["local-model"].exploratorySessions, 1);
  assert.throws(() => validatePairedShopperSuite(cases, sessions.slice(1), catalog), /including failures/);
  sessions[1].isolation.tabId = sessions[0].isolation.tabId;
  assert.throws(() => validatePairedShopperSuite(cases, sessions, catalog), /Repeated/);
});

void test("exploratory estimates and unknown start/capture clocks do not become formal latency evidence", () => {
  const s = session("local-model"); s.startedAt = null; s.elapsedMs = 35_000;
  s.protocol = { passed: false, deviations: ["Session clock was not captured; 35 seconds is an estimate, not an upper bound"] };
  s.timingPrecision = { sessionElapsed: "estimated", stepElapsed: "not-recorded", source: "Approximate operator estimate only" };
  s.steps.forEach((step) => { step.elapsedMs = null; step.event.at = null; });
  s.model.status = "complete";
  s.model.readiness = [{ name: "fieldwork.local-model.complete.after-paint-opportunity", stage: "complete", requestId: "exact-browser-entry",
    startTimeMs: 100, durationMs: 6200, capturedAtElapsedMs: null, acceptedFields: 19, withheldFields: 1, requiredFields: 20,
    source: "data-local-model-readiness", artifactId: s.artifacts[0].artifact.id,
    scope: "after-paint-opportunity", physicalPaintMeasured: false, imageTransferMeasured: false }];
  const parsed = validateShopperSession(s, shopperCase, catalog);
  assert.equal(parsed.startedAt, null); assert.equal(parsed.model.readiness[0].capturedAtElapsedMs, null);
  assert.equal(parsed.model.readiness[0].durationMs, 6200); assert.equal(parsed.timingPrecision.sessionElapsed, "estimated");
  s.protocol = { passed: true, deviations: [] };
  assert.throws(() => validateShopperSession(s, shopperCase, catalog), /exploratory/);
});

void test("artifact verification checks actual bytes, mismatches, missing traces and symlink escapes", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "shopper-contract-"));
  const outside = await mkdtemp(path.join(tmpdir(), "shopper-outside-"));
  try {
    const bytes = Buffer.from("test evidence bytes"); await writeFile(path.join(directory, "screenshot.png"), bytes);
    const artifacts = session().artifacts;
    artifacts[0].sha256 = createHash("sha256").update(bytes).digest("hex"); artifacts[0].bytes = bytes.length;
    const verified = await verifyShopperArtifactFiles(artifacts, directory);
    assert.equal(verified[0].verified, true); assert.equal(verified[1].verified, false);
    artifacts[0].sha256 = "b".repeat(64);
    assert.match((await verifyShopperArtifactFiles(artifacts, directory))[0].error!, /mismatch/);
    await writeFile(path.join(outside, "outside.png"), bytes);
    await symlink(path.join(outside, "outside.png"), path.join(directory, "linked.png"));
    artifacts[0].artifact.path = "linked.png";
    assert.match((await verifyShopperArtifactFiles(artifacts, directory))[0].error!, /symlink escapes/);
  } finally { await rm(directory, { recursive: true, force: true }); await rm(outside, { recursive: true, force: true }); }
});
