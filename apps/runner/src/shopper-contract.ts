/** External shopper evidence sidecars. Frozen Run/runner wire contracts stay v1. */
import { createHash } from "node:crypto";
import { readFile, realpath } from "node:fs/promises";
import path from "node:path";
import { ArtifactSchema, LOCAL_ORIGINS, StepEventSchema, type Artifact, type Run, type StepEvent } from "@storefront/contracts";

export const SHOPPER_CONTRACT_VERSION = "shopper-session.v1" as const;
export const SHOPPER_ALLOWED_ORIGINS = [LOCAL_ORIGINS.storefront, "http://127.0.0.1:4312", "http://localhost:4312"] as const;
export type ShopperArm = "generic" | "local-model";
export type ShopperNeed = { id: string; label: string; attribute: string;
  operator: "equals" | "includes" | "at-most" | "less-than"; value: string | number | boolean };
export type ShopperCase = {
  contractVersion: typeof SHOPPER_CONTRACT_VERSION;
  caseId: string;
  split: Run["personaSplit"];
  seed: Run["seed"];
  armOrder: [ShopperArm, ShopperArm];
  privateShopperNeeds: { goal: string; hardConstraints: ShopperNeed[]; preferences: ShopperNeed[] };
  observableContext: { query: string; itemBudget: number | null; hardRequirements: { attribute: string; value: string }[] };
};
export type ShopperAssignment = {
  contractVersion: typeof SHOPPER_CONTRACT_VERSION;
  caseId: string;
  arm: ShopperArm;
  entryUrl: string;
  shopperNeeds: ShopperCase["privateShopperNeeds"];
  recommenderContext: ShopperCase["observableContext"];
  instructions: readonly string[];
};
export type CatalogSnapshot = { sha256: string; products: { productId: string; originalName: string;
  itemPrice: number; facts: Record<string, string | number | boolean | string[]>;
  variants: { variantId: string; selectedColor: string }[] }[] };
export type ShopperAction =
  | { type: "navigate"; url: string }
  | { type: "search"; query: string }
  | { type: "choose-variant" | "add-to-cart"; productId: string; variantId: string }
  | { type: "inspect" | "open-cart" | "confirm-demo-order" | "abandon"; target: string };
export type ShopperStep = { event: Omit<StepEvent, "at"> & { at: StepEvent["at"] | null };
  elapsedMs: number | null; action: ShopperAction };
export type ShopperArtifact = { artifact: Artifact; status: "captured" | "missing" | "failed";
  sha256: string | null; bytes: number; error: string | null };
export type ShopperChoice = { productId: string; variantId: string; originalName: string;
  catalogPrice: number; selectedColor: string; quantity: 1; observationSource: "agent-reported" | "independent-dom";
  sourceStepIds: string[]; evidenceArtifactIds: string[] };
export type NeedVerification = { needId: string; status: "met" | "not-met" | "unknown";
  observed: string; evidenceArtifactIds: string[] };
export type ReadinessEvidence = { name: string; stage: "rank" | "firstThree" | "complete";
  requestId: string; startTimeMs: number; durationMs: number; capturedAtElapsedMs: number | null;
  acceptedFields: number | null; withheldFields: number | null; requiredFields: number | null;
  source: "performance-measure" | "data-local-model-readiness"; artifactId: string;
  scope: "after-paint-opportunity"; physicalPaintMeasured: false; imageTransferMeasured: false };
export type ShopperSession = {
  contractVersion: typeof SHOPPER_CONTRACT_VERSION;
  caseId: string;
  arm: ShopperArm;
  run: Pick<Run, "id" | "experimentId" | "personaId" | "personaSplit" | "seed">;
  browser: { name: "Chrome"; headed: true; actualNavigation: boolean; version: string | null; automation: "cua" | "human" | "playwright-chrome" };
  isolation: { kind: "fresh-tab" | "fresh-storage-context" | "not-created"; tabId: string | null; contextId: string | null;
    sharedStorage: boolean | null; storageCleared: boolean | null; initialCartEmpty: boolean | null; evidenceArtifactIds: string[]; note: string };
  entryUrl: string;
  startedAt: string | null;
  elapsedMs: number;
  timingPrecision: { sessionElapsed: "measured-monotonic" | "upper-bound" | "estimated";
    stepElapsed: "measured-monotonic" | "partially-recorded" | "not-recorded"; source: string };
  protocol: { passed: boolean; deviations: string[] };
  status: "completed" | "abandoned" | "constraint-failure" | "technical-failure" | "timeout";
  error: string | null;
  realPurchase: false;
  fakeOrderConfirmed: boolean;
  steps: ShopperStep[];
  artifacts: ShopperArtifact[];
  choice: ShopperChoice | null;
  verification: { verifierId: string; independentOfShopperAgent: true; catalogSha256: string;
    hardConstraints: NeedVerification[]; preferences: NeedVerification[] };
  model: { status: "not-enabled" | "complete" | "partial" | "failed" | "not-observed"; readiness: ReadinessEvidence[] };
  fallback: { used: boolean; kind: "none" | "last-safe-model" | "original-catalog" | "service-unavailable"; reason: string | null };
};

function requireValue(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }
function record(value: unknown): Record<string, unknown> {
  requireValue(value && typeof value === "object" && !Array.isArray(value), "Expected shopper contract object");
  return value as Record<string, unknown>;
}
function exact(value: unknown, keys: string[]) {
  const object = record(value);
  requireValue(Object.keys(object).length === keys.length && Object.keys(object).every((key) => keys.includes(key)), "Unexpected or missing shopper contract fields");
}
function text(value: unknown): asserts value is string { requireValue(typeof value === "string" && value.trim().length > 0 && value.length <= 4000, "Nonempty bounded shopper text required"); }
function finite(value: unknown): asserts value is number { requireValue(typeof value === "number" && Number.isFinite(value) && value >= 0, "Finite nonnegative elapsed, price or count required"); }
function hash(value: unknown): asserts value is string { requireValue(typeof value === "string" && /^[a-f0-9]{64}$/.test(value), "SHA-256 evidence digest required"); }
function list(value: unknown): asserts value is unknown[] { requireValue(Array.isArray(value), "Shopper contract array required"); }
function ids(value: unknown): asserts value is string[] {
  list(value); value.forEach(text); requireValue(new Set(value).size === value.length, "Repeated evidence or requirement IDs");
}
function need(value: unknown): asserts value is ShopperNeed {
  exact(value, ["id", "label", "attribute", "operator", "value"]);
  const n = value as ShopperNeed; text(n.id); text(n.label); text(n.attribute);
  requireValue(!/^(?:product.?id|variant.?id|catalog.?index|gold|expected.?product)/i.test(n.attribute), "Gold identity selectors cannot be shopper needs");
  requireValue(["equals", "includes", "at-most", "less-than"].includes(n.operator), "Unsupported need comparison");
  requireValue(["string", "number", "boolean"].includes(typeof n.value), "Typed need value required");
  if (typeof n.value === "number") finite(n.value);
  if (typeof n.value === "string") text(n.value);
  if (n.operator === "at-most" || n.operator === "less-than") requireValue(typeof n.value === "number", "Numeric bound required");
}

export function validateShopperCase(value: unknown): ShopperCase {
  exact(value, ["contractVersion", "caseId", "split", "seed", "armOrder", "privateShopperNeeds", "observableContext"]);
  const c = value as ShopperCase;
  requireValue(c.contractVersion === SHOPPER_CONTRACT_VERSION, "Unsupported shopper sidecar version"); text(c.caseId);
  requireValue(["development", "held-out"].includes(c.split), "Explicit existing persona split required");
  finite(c.seed); requireValue(Number.isInteger(c.seed), "Integer seed required");
  list(c.armOrder); requireValue(c.armOrder.length === 2 && new Set(c.armOrder).size === 2 && c.armOrder.every((arm) => ["generic", "local-model"].includes(arm)), "Each case requires both paired arms in declared order");
  exact(c.privateShopperNeeds, ["goal", "hardConstraints", "preferences"]); text(c.privateShopperNeeds.goal);
  list(c.privateShopperNeeds.hardConstraints); list(c.privateShopperNeeds.preferences);
  const needs = [...c.privateShopperNeeds.hardConstraints, ...c.privateShopperNeeds.preferences]; needs.forEach(need); ids(needs.map((n) => n.id));
  exact(c.observableContext, ["query", "itemBudget", "hardRequirements"]); text(c.observableContext.query);
  if (c.observableContext.itemBudget !== null) finite(c.observableContext.itemBudget);
  list(c.observableContext.hardRequirements);
  for (const requirement of c.observableContext.hardRequirements) { exact(requirement, ["attribute", "value"]); text(requirement.attribute); text(requirement.value); }
  return structuredClone(c);
}

export function shopperEntryUrl(arm: ShopperArm, baseUrl = `${LOCAL_ORIGINS.storefront}/`): string {
  requireValue(arm === "generic" || arm === "local-model", "Unknown shopper arm");
  const base = new URL(baseUrl);
  requireValue(SHOPPER_ALLOWED_ORIGINS.some((origin) => origin === base.origin) && base.pathname === "/" && !base.search && !base.hash && !base.username && !base.password,
    "Only declared local shopper base origins with a clean root URL are allowed");
  return `${base.origin}/${arm === "local-model" ? "?model=local" : ""}`;
}

/** Pass only this packet to the shopper. Never pass a Run, catalog IDs or gold labels. */
export function buildShopperAssignment(value: unknown, arm: ShopperArm, baseUrl?: string): ShopperAssignment {
  const c = validateShopperCase(value);
  return { contractVersion: SHOPPER_CONTRACT_VERSION, caseId: c.caseId, arm, entryUrl: shopperEntryUrl(arm, baseUrl),
    shopperNeeds: c.privateShopperNeeds, recommenderContext: c.observableContext, instructions: [
      "Navigate in actual headed Chrome; inspect products and select a product/color using the visible store.",
      "Private shopper needs guide your choice. Only assigned observable context may be entered into search or sent to the recommender; do not reveal private needs there.",
      "The generic arm retains the store's existing deterministic query/search behavior; it is not a blank-query or no-personalization control.",
      "Do not use gold product IDs, expected selections or evaluator answers. No first-button purchase script.",
      "Only a fake demo order may be confirmed. Never make a real purchase or payment.",
      "Report actual actions and any failure honestly. A fresh Chrome tab does not imply fresh cookies or isolated storage.",
    ] };
}

function catalogProduct(choice: ShopperChoice, catalog: CatalogSnapshot) {
  const product = catalog.products.find((p) => p.productId === choice.productId);
  requireValue(product, "Chosen product is absent from independently supplied catalog");
  const variant = product.variants.find((v) => v.variantId === choice.variantId);
  requireValue(variant && choice.originalName === product.originalName && choice.catalogPrice === product.itemPrice && choice.selectedColor === variant.selectedColor,
    "Actual chosen variant, catalog name, price or color differs from catalog evidence");
  return product;
}

export function validateShopperCatalog(value: unknown): CatalogSnapshot {
  exact(value, ["sha256", "products"]);
  const catalog = value as CatalogSnapshot; hash(catalog.sha256); list(catalog.products);
  requireValue(catalog.products.length > 0, "Independent catalog cannot be empty");
  ids(catalog.products.map((p) => p.productId));
  for (const product of catalog.products) {
    exact(product, ["productId", "originalName", "itemPrice", "facts", "variants"]);
    text(product.originalName); finite(product.itemPrice); record(product.facts); list(product.variants);
    requireValue(product.variants.length > 0, "Catalog product variants required");
    ids(product.variants.map((v) => v.variantId));
    product.variants.forEach((variant) => { exact(variant, ["variantId", "selectedColor"]); text(variant.selectedColor); });
    for (const [attribute, fact] of Object.entries(product.facts)) {
      requireValue(!["itemPrice", "selectedColor", "productId", "variantId"].includes(attribute), "Typed semantic facts cannot override actual catalog identity, price or variant");
      if (attribute === "originalName") requireValue(fact === product.originalName, "Exact-name fact must match catalog originalName");
      if (Array.isArray(fact)) fact.forEach(text);
      else if (typeof fact === "number") finite(fact);
      else requireValue(typeof fact === "string" && fact.trim().length > 0 || typeof fact === "boolean", "Typed independent catalog fact required");
    }
  }
  return structuredClone(catalog);
}

export function verifyShopperNeeds(needs: ShopperNeed[], choice: ShopperChoice | null, catalog: CatalogSnapshot): Omit<NeedVerification, "evidenceArtifactIds">[] {
  const product = choice ? catalogProduct(choice, catalog) : null;
  return needs.map((n) => {
    const facts: Record<string, unknown> = product && choice ? { ...product.facts, originalName: product.originalName,
      itemPrice: product.itemPrice, selectedColor: choice.selectedColor } : {};
    const actual = facts[n.attribute];
    if (actual === undefined) return { needId: n.id, status: "unknown", observed: "No independently supplied catalog fact" };
    let met: boolean | null = null;
    if (n.operator === "equals") met = actual === n.value;
    else if (n.operator === "includes" && Array.isArray(actual) && typeof n.value === "string") met = actual.includes(n.value);
    else if ((n.operator === "at-most" || n.operator === "less-than") && typeof actual === "number" && typeof n.value === "number") met = n.operator === "at-most" ? actual <= n.value : actual < n.value;
    return { needId: n.id, status: met === null ? "unknown" : met ? "met" : "not-met", observed: JSON.stringify(actual) };
  });
}

function allowedNavigation(url: string, arm: ShopperArm, c: ShopperCase, origin: string) {
  const parsed = new URL(url);
  requireValue(parsed.origin === origin && parsed.pathname === "/" && !parsed.username && !parsed.password,
    "Only actual assigned local storefront navigation without credentials is allowed");
  const model = parsed.searchParams.getAll("model");
  requireValue(arm === "local-model" ? model.length === 1 && model[0] === "local" : model.length === 0,
    "Browser navigation changed paired model arm");
  requireValue([...parsed.searchParams.keys()].every((key) => key === "model"),
    "Unassigned storefront query parameters can disclose private context or change the paired configuration");
  const route = parsed.hash.slice(1).split("?");
  requireValue(route.length <= 2 && ["", "collection", "cart", "confirmed"].includes(route[0]),
    "Only existing storefront hash routes are allowed");
  const params = new URLSearchParams(route[1] ?? "");
  const keys = [...params.keys()];
  requireValue(keys.every((key) => key === "q" || key === "c") && new Set(keys).size === keys.length,
    "Unknown or repeated storefront hash parameters can disclose private context");
  const collection = params.get("c");
  // Existing public collection routes; never allow a private need as route data.
  requireValue(collection === null || ["all", "bags", "clothing", "home", "food", "accessories"].includes(collection),
    "Only existing public storefront collections are allowed");
  const query = params.get("q");
  requireValue(!query || query === c.observableContext.query, "Private needs or unassigned search leaked into observable browser navigation");
}

/** Validate a completed or failed sidecar; this never launches a browser or model. */
export function validateShopperSession(value: unknown, caseValue: unknown, catalog: CatalogSnapshot): ShopperSession {
  const c = validateShopperCase(caseValue);
  validateShopperCatalog(catalog);
  exact(value, ["contractVersion", "caseId", "arm", "run", "browser", "isolation", "entryUrl", "startedAt", "elapsedMs", "timingPrecision", "protocol", "status", "error", "realPurchase", "fakeOrderConfirmed", "steps", "artifacts", "choice", "verification", "model", "fallback"]);
  const s = value as ShopperSession;
  requireValue(s.contractVersion === SHOPPER_CONTRACT_VERSION && s.caseId === c.caseId, "Session/case version or identity mismatch");
  exact(s.run, ["id", "experimentId", "personaId", "personaSplit", "seed"]); text(s.run.id); text(s.run.experimentId); text(s.run.personaId);
  requireValue(s.run.personaSplit === c.split && s.run.seed === c.seed, "Run split/seed differs from frozen case");
  exact(s.browser, ["name", "headed", "actualNavigation", "version", "automation"]);
  requireValue(s.browser.name === "Chrome" && s.browser.headed === true && typeof s.browser.actualNavigation === "boolean" && ["cua", "human", "playwright-chrome"].includes(s.browser.automation), "Actual headed Chrome required; headless Chromium is insufficient");
  if (s.browser.version !== null) text(s.browser.version);
  if (!s.browser.actualNavigation) requireValue(s.browser.version === null, "Unopened Chrome cannot claim an observed version");
  exact(s.isolation, ["kind", "tabId", "contextId", "sharedStorage", "storageCleared", "initialCartEmpty", "evidenceArtifactIds", "note"]);
  text(s.isolation.note);
  if (s.browser.actualNavigation) {
    text(s.isolation.tabId);
    if (s.isolation.contextId !== null) text(s.isolation.contextId);
    requireValue(s.isolation.initialCartEmpty === true ||
      ((s.isolation.initialCartEmpty === null || s.isolation.initialCartEmpty === false)
        && ["technical-failure", "timeout"].includes(s.status) && s.protocol?.passed === false
        && Array.isArray(s.protocol.deviations) && s.protocol.deviations.length > 0),
    "Fresh session must observe an empty initial cart, or retain an explicit nonformal technical/timeout failure");
    requireValue(s.isolation.kind === "fresh-tab" ? s.isolation.sharedStorage === true && s.isolation.storageCleared === false
    : s.isolation.kind === "fresh-storage-context" && s.isolation.sharedStorage === false && s.isolation.storageCleared === true,
    "Declare fresh-tab shared storage honestly, or provide a genuinely fresh storage context");
  } else requireValue(["technical-failure", "timeout"].includes(s.status) && s.isolation.kind === "not-created" && s.isolation.tabId === null && s.isolation.contextId === null
    && s.isolation.sharedStorage === null && s.isolation.storageCleared === null && s.isolation.initialCartEmpty === null,
    "Prelaunch failure must honestly declare no browser/storage observations");
  const entryOrigin = new URL(s.entryUrl).origin;
  requireValue(s.entryUrl === shopperEntryUrl(s.arm, `${entryOrigin}/`), "Session entry URL differs from declared paired arm");
  if (s.startedAt === null) requireValue(s.protocol.passed === false, "Unknown session start is allowed only for an exploratory protocol deviation");
  else requireValue(Number.isFinite(Date.parse(s.startedAt)), "ISO session start timestamp required, or explicit exploratory null");
  finite(s.elapsedMs);
  exact(s.timingPrecision, ["sessionElapsed", "stepElapsed", "source"]); text(s.timingPrecision.source);
  requireValue(["measured-monotonic", "upper-bound", "estimated"].includes(s.timingPrecision.sessionElapsed)
    && ["measured-monotonic", "partially-recorded", "not-recorded"].includes(s.timingPrecision.stepElapsed),
    "Explicit observed timing precision required; missing per-action times remain null");
  exact(s.protocol, ["passed", "deviations"]); list(s.protocol.deviations); s.protocol.deviations.forEach(text);
  requireValue(typeof s.protocol.passed === "boolean" && s.protocol.passed === (s.protocol.deviations.length === 0),
    "Protocol eligibility must disclose deviations; exploratory sessions cannot pass formal protocol");
  if (s.timingPrecision.sessionElapsed === "estimated") requireValue(s.protocol.passed === false,
    "Estimated session time is exploratory, never a proven bound or formal latency sample");
  requireValue(["completed", "abandoned", "constraint-failure", "technical-failure", "timeout"].includes(s.status), "Explicit session outcome required");
  requireValue(s.realPurchase === false && typeof s.fakeOrderConfirmed === "boolean", "Only fake demo confirmation is authorized");
  requireValue(s.error === null || typeof s.error === "string", "Explicit nullable session error required");
  if (s.status === "technical-failure" || s.status === "timeout") text(s.error);
  list(s.artifacts); ids(s.artifacts.map((a) => a.artifact.id));
  for (const evidence of s.artifacts) {
    exact(evidence, ["artifact", "status", "sha256", "bytes", "error"]); ArtifactSchema.parse(evidence.artifact);
    requireValue(evidence.artifact.runId === s.run.id && !path.isAbsolute(evidence.artifact.path) && !evidence.artifact.path.split(/[\\/]/).includes(".."), "Evidence paths must stay within the session artifact directory");
    finite(evidence.bytes); requireValue(Number.isInteger(evidence.bytes), "Integer artifact byte length required");
    if (evidence.status === "captured") { hash(evidence.sha256); requireValue(evidence.bytes > 0 && evidence.error === null, "Captured evidence requires bytes and no error"); }
    else { requireValue(["missing", "failed"].includes(evidence.status) && evidence.sha256 === null && evidence.bytes === 0, "Missing evidence cannot carry invented hashes"); text(evidence.error); }
  }
  requireValue(["screenshot", "trace"].every((kind) => s.artifacts.some((a) => a.artifact.kind === kind)), "Explicit screenshot and trace captured/missing/error receipts required");
  const evidenceIds = new Set(s.artifacts.filter((a) => a.status === "captured").map((a) => a.artifact.id));
  const checkEvidence = (values: unknown) => { ids(values); requireValue(values.every((id) => evidenceIds.has(id)), "Referenced evidence is absent or not captured"); };
  checkEvidence(s.isolation.evidenceArtifactIds);
  list(s.steps); requireValue(s.browser.actualNavigation ? s.steps.length > 0 : s.steps.length === 0 && !s.choice && !s.fakeOrderConfirmed,
    "Actual Chrome navigation steps required; prelaunch failures cannot invent shopping actions");
  let elapsed = -1;
  for (const [index, step] of s.steps.entries()) {
    exact(step, ["event", "elapsedMs", "action"]);
    const { at, ...eventWithoutTimestamp } = step.event;
    StepEventSchema.omit({ at: true }).parse(eventWithoutTimestamp);
    if (at !== null) requireValue(Number.isFinite(Date.parse(at)), "Observed action wall timestamp must be valid, or explicitly null");
    requireValue(step.event.runId === s.run.id && step.event.sequence === index, "Step sequence/run linkage differs");
    if (step.elapsedMs !== null) { finite(step.elapsedMs); requireValue(step.elapsedMs >= elapsed && step.elapsedMs <= s.elapsedMs, "Step elapsed time must be monotonic within session"); elapsed = step.elapsedMs; }
    if (step.event.screenshotArtifactId) checkEvidence([step.event.screenshotArtifactId]);
    const action = step.action; record(action);
    if (action.type === "navigate") { exact(action, ["type", "url"]); allowedNavigation(action.url, s.arm, c, entryOrigin); requireValue(step.event.kind === "navigation", "Navigation must use an existing navigation StepEvent"); }
    else if (action.type === "search") { exact(action, ["type", "query"]); requireValue(action.query === c.observableContext.query, "Only assigned observable query may enter search"); }
    else if (action.type === "choose-variant" || action.type === "add-to-cart") { exact(action, ["type", "productId", "variantId"]); text(action.productId); text(action.variantId); }
    else { exact(action, ["type", "target"]); requireValue(["inspect", "open-cart", "confirm-demo-order", "abandon"].includes(action.type), "Unknown shopper action"); text((action as { target?: unknown }).target); }
  }
  const timedSteps = s.steps.filter((step) => step.elapsedMs !== null).length;
  requireValue(s.timingPrecision.stepElapsed === "not-recorded" ? timedSteps === 0
    : s.timingPrecision.stepElapsed === "measured-monotonic" ? timedSteps === s.steps.length
      : timedSteps > 0 && timedSteps < s.steps.length, "Per-action timing precision disagrees with known/null values");
  if (s.browser.actualNavigation) requireValue(s.steps[0].action.type === "navigate" && s.steps[0].action.url === s.entryUrl, "First observed action must navigate actual Chrome to the paired entry URL");
  requireValue(s.fakeOrderConfirmed === s.steps.some((step) => step.action.type === "confirm-demo-order"), "Fake confirmation claim requires matching browser action");
  if (s.choice) {
    exact(s.choice, ["productId", "variantId", "originalName", "catalogPrice", "selectedColor", "quantity", "observationSource", "sourceStepIds", "evidenceArtifactIds"]);
    requireValue(["agent-reported", "independent-dom"].includes(s.choice.observationSource), "Actual choice evidence must distinguish agent report from independent DOM verification");
    requireValue(s.choice.quantity === 1, "Version1 evaluates one actual chosen product per shopper"); catalogProduct(s.choice, catalog);
    ids(s.choice.sourceStepIds); requireValue(s.choice.sourceStepIds.length > 0 && s.choice.sourceStepIds.every((id) => s.steps.some((step) => step.event.id === id)), "Choice requires actual browser step evidence");
    checkEvidence(s.choice.evidenceArtifactIds); requireValue(s.choice.evidenceArtifactIds.length > 0, "Choice requires captured independent evidence");
    if (s.fakeOrderConfirmed) requireValue(s.steps.some((step) => step.action.type === "add-to-cart" && step.action.productId === s.choice?.productId && step.action.variantId === s.choice.variantId), "Confirmed choice must match actual cart action");
  }
  exact(s.verification, ["verifierId", "independentOfShopperAgent", "catalogSha256", "hardConstraints", "preferences"]);
  text(s.verification.verifierId); requireValue(s.verification.independentOfShopperAgent === true, "Independent verification required; shopper self-report is insufficient");
  hash(catalog.sha256); requireValue(s.verification.catalogSha256 === catalog.sha256, "Independent catalog snapshot mismatch");
  for (const key of ["hardConstraints", "preferences"] as const) {
    const expected = verifyShopperNeeds(c.privateShopperNeeds[key], s.choice, catalog); list(s.verification[key]);
    requireValue(s.verification[key].length === expected.length, "Every private requirement needs an independent check");
    s.verification[key].forEach((check, index) => { exact(check, ["needId", "status", "observed", "evidenceArtifactIds"]);
      requireValue(check.needId === expected[index].needId && check.status === expected[index].status && check.observed === expected[index].observed, "Independent check disagrees with actual chosen catalog facts");
      checkEvidence(check.evidenceArtifactIds); requireValue(!s.browser.actualNavigation || check.evidenceArtifactIds.length > 0, "Each independent check requires captured evidence"); });
  }
  if (s.status === "completed") requireValue(s.choice && s.fakeOrderConfirmed && s.verification.hardConstraints.every((check) => check.status === "met"), "Completion requires actual fake order and every hard constraint independently met");
  exact(s.model, ["status", "readiness"]); list(s.model.readiness);
  requireValue(["not-enabled", "complete", "partial", "failed", "not-observed"].includes(s.model.status), "Explicit model observation status required");
  if (s.arm === "generic") requireValue(s.model.status === "not-enabled" && s.model.readiness.length === 0, "Generic arm cannot claim model readiness");
  else requireValue(s.model.status !== "not-enabled", "Local arm must declare observed model status honestly");
  if (s.model.status === "complete") requireValue(s.model.readiness.some((m) => m.stage === "complete"), "Complete model readiness requires complete-copy browser evidence, not ranking-only timing");
  for (const measure of s.model.readiness) {
    exact(measure, ["name", "stage", "requestId", "startTimeMs", "durationMs", "capturedAtElapsedMs", "acceptedFields", "withheldFields", "requiredFields", "source", "artifactId", "scope", "physicalPaintMeasured", "imageTransferMeasured"]);
    requireValue(["rank", "firstThree", "complete"].includes(measure.stage) && measure.name === `fieldwork.local-model.${measure.stage}.after-paint-opportunity`, "Expected browser readiness measure name/stage");
    text(measure.requestId); finite(measure.startTimeMs); finite(measure.durationMs);
    if (measure.capturedAtElapsedMs === null) requireValue(s.protocol.passed === false && s.timingPrecision.stepElapsed === "not-recorded",
      "Unknown readiness capture clock requires exploratory unrecorded action timing");
    else { finite(measure.capturedAtElapsedMs);
      requireValue(measure.durationMs <= measure.capturedAtElapsedMs && measure.capturedAtElapsedMs <= s.elapsedMs, "Readiness must fit actual session elapsed timeline"); }
    requireValue(["performance-measure", "data-local-model-readiness"].includes(measure.source) && measure.scope === "after-paint-opportunity" && measure.physicalPaintMeasured === false && measure.imageTransferMeasured === false, "Browser readiness is a paint opportunity; image/physical paint claims forbidden");
    checkEvidence([measure.artifactId]);
    if (measure.stage === "rank") requireValue(measure.acceptedFields === null && measure.withheldFields === null && measure.requiredFields === null, "Ranking latency cannot stand in for generated field readiness");
    else { [measure.acceptedFields, measure.withheldFields, measure.requiredFields].forEach(finite);
      requireValue([measure.acceptedFields, measure.withheldFields, measure.requiredFields].every(Number.isInteger)
        && measure.requiredFields! <= (measure.stage === "firstThree" ? 10 : 20)
        && measure.acceptedFields! + measure.withheldFields! === measure.requiredFields, "Exact fresh/withheld field counts required"); }
  }
  exact(s.fallback, ["used", "kind", "reason"]);
  requireValue(s.fallback.used === (s.fallback.kind !== "none") && ["none", "last-safe-model", "original-catalog", "service-unavailable"].includes(s.fallback.kind), "Explicit fallback state required");
  if (s.fallback.used) text(s.fallback.reason); else requireValue(s.fallback.reason === null, "No fallback requires null reason");
  return structuredClone(s);
}

/** Missing traces are honest missing evidence, never synthesized files. */
export async function verifyShopperArtifactFiles(artifacts: ShopperArtifact[], directory: string) {
  const root = path.resolve(directory);
  return Promise.all(artifacts.map(async (evidence) => {
    if (evidence.status !== "captured") return { artifactId: evidence.artifact.id, verified: false, error: evidence.error };
    try {
      const file = path.resolve(root, evidence.artifact.path);
      requireValue(file.startsWith(root + path.sep), "Artifact escapes session directory");
      const [actualRoot, actualFile] = await Promise.all([realpath(root), realpath(file)]);
      requireValue(actualFile.startsWith(actualRoot + path.sep), "Artifact symlink escapes session directory");
      const bytes = await readFile(actualFile), sha256 = createHash("sha256").update(bytes).digest("hex");
      requireValue(bytes.length === evidence.bytes && sha256 === evidence.sha256, "Captured artifact bytes or SHA-256 mismatch");
      return { artifactId: evidence.artifact.id, verified: true, error: null };
    } catch (error) { return { artifactId: evidence.artifact.id, verified: false, error: String(error) }; }
  }));
}

export function validatePairedShopperSuite(cases: ShopperCase[], sessions: ShopperSession[], catalog: CatalogSnapshot) {
  requireValue(cases.length === 10 && sessions.length === 20, "Fixed10 paired cases require20 receipts, including failures");
  ids(cases.map((c) => c.caseId)); ids(sessions.map((s) => s.run.id)); ids(sessions.flatMap((s) => s.isolation.tabId ? [s.isolation.tabId] : []));
  const checkedCases = cases.map(validateShopperCase);
  const checked = checkedCases.flatMap((c) => {
    const pair = sessions.filter((s) => s.caseId === c.caseId);
    requireValue(pair.length === 2 && c.armOrder.every((arm, i) => pair[i].arm === arm), "Both arms must preserve declared pair order without dropping failures");
    return pair.map((s) => validateShopperSession(s, c, catalog));
  });
  return { contractVersion: SHOPPER_CONTRACT_VERSION, synthetic: true, fixedCases: 10, fixedSessions: 20,
    arms: Object.fromEntries((["generic", "local-model"] as const).map((arm) => {
      const rows = checked.filter((s) => s.arm === arm);
      return [arm, { fixedDenominator: 10, completed: rows.filter((s) => s.protocol.passed && s.status === "completed").length,
        reportedCompleted: rows.filter((s) => s.status === "completed").length,
        exploratorySessions: rows.filter((s) => !s.protocol.passed).length,
        verifiedDomCompleted: rows.filter((s) => s.protocol.passed && s.status === "completed" && s.choice?.observationSource === "independent-dom").length,
        completionEvidenceScope: "formal completion excludes protocol deviations; reportedCompleted also retains exploratory outcomes; independent DOM verification is separate",
        completedRate: rows.filter((s) => s.protocol.passed && s.status === "completed").length / 10,
        failures: rows.filter((s) => !s.protocol.passed || s.status !== "completed").length,
        storageIsolation: rows.map((s) => s.isolation.kind), elapsedMsAllAttempts: rows.map((s) => s.elapsedMs),
        elapsedTimingPrecision: rows.map((s) => s.timingPrecision) }];
    })), humanConversionClaim: false, physicalPaintClaim: false };
}
