/** CPU-only descriptive analysis of validated synthetic shopper aggregates. No browser or network calls. */
import { createHash } from "node:crypto";
import { readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import type { ShopperArm } from "./shopper-contract.js";

const ARMS: ShopperArm[] = ["generic", "local-model"];
const HASH = /^[a-f0-9]{64}$/;
type Obj = Record<string, unknown>;
function requireValue(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }
function object(value: unknown, label: string): Obj {
  requireValue(value && typeof value === "object" && !Array.isArray(value), `${label}: object required`); return value as Obj;
}
function array(value: unknown, label: string): unknown[] { requireValue(Array.isArray(value), `${label}: array required`); return value; }
function text(value: unknown, label: string): string { requireValue(typeof value === "string" && value.length > 0, `${label}: text required`); return value; }
function number(value: unknown, label: string): number { requireValue(typeof value === "number" && Number.isFinite(value) && value >= 0, `${label}: finite nonnegative number required`); return value; }
function integer(value: unknown, label: string): number { const result = number(value, label); requireValue(Number.isSafeInteger(result), `${label}: safe integer required`); return result; }
function bool(value: unknown, label: string): boolean { requireValue(typeof value === "boolean", `${label}: boolean required`); return value; }
function strings(value: unknown, label: string) { return array(value, label).map((item) => text(item, label)); }
const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);
const mean = (values: number[]) => values.length ? sum(values) / values.length : null;
function quantile(sorted: number[], p: number) {
  if (!sorted.length) return null;
  const position = (sorted.length - 1) * p, low = Math.floor(position), high = Math.ceil(position);
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}
export function distribution(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  return { count: sorted.length, total: sum(sorted), mean: mean(sorted), minimum: sorted[0] ?? null,
    median: quantile(sorted, .5), p95: quantile(sorted, .95), maximum: sorted.at(-1) ?? null, values };
}
export function wilson95(successes: number, assigned: number) {
  requireValue(Number.isInteger(assigned) && assigned > 0 && Number.isInteger(successes) && successes >= 0 && successes <= assigned, "Wilson counts invalid");
  const z = 1.959963984540054, p = successes / assigned, denominator = 1 + z * z / assigned;
  const center = (p + z * z / (2 * assigned)) / denominator;
  const half = z * Math.sqrt(p * (1 - p) / assigned + z * z / (4 * assigned * assigned)) / denominator;
  return { level: .95, method: "Wilson score", lower: Math.max(0, center - half), upper: Math.min(1, center + half) };
}
export function pairedBootstrap95(differences: number[], seed = 20261018, replicates = 20000) {
  requireValue(differences.length > 0 && differences.every(Number.isFinite), "Paired bootstrap requires finite case differences");
  requireValue(Number.isSafeInteger(seed) && seed >= 0 && seed <= 0xffffffff && Number.isInteger(replicates) && replicates >= 1000 && replicates <= 100000, "Bootstrap settings invalid");
  let state = seed >>> 0;
  const random = () => { state = (state + 0x6D2B79F5) >>> 0; let t = state; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const draws = Array.from({ length: replicates }, () => {
    let total = 0; for (let index = 0; index < differences.length; index++) total += differences[Math.floor(random() * differences.length)];
    return total / differences.length;
  }).sort((a, b) => a - b);
  return { difference: mean(differences)!, lower: quantile(draws, .025)!, upper: quantile(draws, .975)!, level: .95,
    method: "Paired case percentile bootstrap", pairs: differences.length, seed, replicates, direction: "local-model minus generic" };
}

type Preference = { met: number; notMet: number; unknown: number; total: number };
type Eligibility = "satisfiable" | "known-empty" | "unverifiable-required-fact";
type ServerEvidence = { row: Obj; artifactHashVerified: boolean };
export type MetricsOptions = { seed?: number; replicates?: number; caseEligibility?: Record<string, Eligibility>;
  serverOutcomes?: Record<string, ServerEvidence> };
type Row = { caseId: string; arm: ShopperArm; runId: string; status: string; outcome: string | null;
  observedCheckout: boolean; purchase: boolean; revenueCents: number; exclusions: string[];
  preference: Preference | null; purchasedPreferenceCredit: number; abstained: boolean; correctAbstention: boolean;
  interrupted: boolean; observedSteps: number | null; comparableSteps: number | null; elapsedMs: number | null;
  timingPrecision: string | null; serverConfirmed: boolean; serverMerchandiseRevenueCents: number; serverShippingCents: number | null;
  readiness: { stage: string; requestId: string; durationMs: number }[] };

function moneyCents(value: unknown, label: string) {
  const amount = number(value, label), cents = Math.round(amount * 100);
  requireValue(Number.isSafeInteger(cents) && Math.abs(amount * 100 - cents) < 1e-7, `${label}: safe whole cents required`); return cents;
}
function serverReceipt(evidence: ServerEvidence, row: Row, choice: Obj | null) {
  const item = evidence.row;
  requireValue(item.runId === row.runId && item.caseId === row.caseId && item.arm === row.arm, "Server outcome identity mismatch");
  const confirmed = bool(item.serverConfirmed, "serverConfirmed"); strings(item.errors, "server errors");
  if (!confirmed) return null;
  requireValue(evidence.artifactHashVerified && choice, "Server confirmation requires verified receipt artifact and catalog-bound choice");
  const receipt = object(item.receipt, "server receipt"), readback = object(item.readback, "readback");
  requireValue(receipt.schema === "fieldwork.demo-checkout-receipt.v1" && receipt.realPurchase === false, "Only versioned fake demo receipt accepted");
  text(receipt.checkoutId, "checkoutId"); text(receipt.receiptId, "receiptId");
  const createdAt = text(receipt.createdAt, "receipt createdAt"); requireValue(Number.isFinite(Date.parse(createdAt)), "Invalid receipt creation time");
  requireValue(readback.httpStatus === 200 && readback.identical === true, "Server confirmation requires identical successful GET readback");
  const items = array(receipt.items, "receipt items"); requireValue(items.length === 1, "Exactly one demo item required");
  const product = object(items[0], "receipt item");
  requireValue(product.productId === choice.productId && product.variantId === choice.variantId && product.quantity === 1 && choice.quantity === 1,
    "Server receipt product/variant/quantity differs from independently observed choice");
  const unit = moneyCents(product.unitPrice, "receipt unitPrice"), subtotal = moneyCents(receipt.subtotal, "receipt subtotal");
  requireValue(unit === moneyCents(choice.catalogPrice, "choice catalogPrice") && moneyCents(product.lineTotal, "receipt lineTotal") === unit && subtotal === unit,
    "Server receipt price differs from catalog-bound choice");
  const shipping = object(receipt.shipping, "receipt shipping"), status = text(shipping.status, "shipping status");
  requireValue(["hidden", "free", "flat"].includes(status), "Unknown receipt shipping status");
  let shippingCents: number | null;
  if (status === "hidden") {
    requireValue(shipping.cost === null, "Hidden shipping cost must remain null, not an observed zero charge");
    shippingCents = null;
    requireValue(moneyCents(receipt.total, "receipt total") === subtotal, "Hidden-shipping receipt total must equal merchandise subtotal");
  } else {
    shippingCents = moneyCents(shipping.cost, "shipping cost");
    requireValue(status !== "free" || shippingCents === 0, "Free shipping must have an observed zero charge");
    requireValue(moneyCents(receipt.total, "receipt total") === subtotal + shippingCents, "Receipt total differs from subtotal plus shipping");
  }
  return { subtotal, shippingCents };
}

function verifyServerArtifactContent(input: unknown, indexed: Obj, assigned: Obj) {
  const evidence = object(input, "server evidence JSON");
  requireValue(evidence.schema === "shopper-demo-checkout-evidence.v1" && evidence.synthetic === true, "Versioned synthetic checkout evidence required");
  strings(evidence.errors, "checkout evidence errors");
  if (indexed.serverConfirmed !== true) return;
  const post = object(evidence.post, "POST evidence"), get = object(evidence.get, "GET evidence"), confirmation = object(evidence.confirmation, "visible confirmation");
  const receipt = object(indexed.receipt, "indexed receipt"), postBody = object(post.body, "POST body"), getBody = object(get.body, "GET body");
  requireValue(post.httpStatus === 202 && get.httpStatus === 200 && typeof postBody.duplicate === "boolean", "Actual successful POST202 and GET200 required");
  requireValue(isDeepStrictEqual(postBody.receipt, receipt) && isDeepStrictEqual(getBody.receipt, receipt), "Hashed POST/GET receipt bodies differ from indexed receipt");
  requireValue(confirmation.visible === true && confirmation.checkoutId === receipt.checkoutId && confirmation.receiptId === receipt.receiptId,
    "Visible confirmation does not match server receipt IDs");
  requireValue((evidence.errors as unknown[]).length === 0, "Confirmed checkout evidence contains errors");
  const session = object(assigned.session, "assigned session"), entry = new URL(text(session.entryUrl, "session entryUrl"));
  requireValue(entry.protocol === "http:" && ["127.0.0.1", "localhost", "[::1]"].includes(entry.hostname) && !entry.username && !entry.password, "Local storefront origin required");
  for (const [action, requestPath] of [[post, "/api/demo-checkout"], [get, `/api/demo-checkout/${encodeURIComponent(text(receipt.checkoutId, "checkoutId"))}`]] as const) {
    const url = new URL(text(action.url, "checkout evidence URL"));
    requireValue(url.origin === entry.origin && url.pathname === requestPath && !url.search && !url.hash && !url.username && !url.password,
      "Checkout traffic must match assigned local storefront and exact demo endpoint");
  }
}

function checks(value: unknown, label: string): Preference {
  const rows = array(value, label).map((v) => object(v, label));
  const ids = rows.map((r) => text(r.needId, `${label}.needId`));
  requireValue(new Set(ids).size === ids.length, `${label}: duplicate need ID`);
  for (const row of rows) requireValue(["met", "not-met", "unknown"].includes(String(row.status)), `${label}: invalid status`);
  return { met: rows.filter((r) => r.status === "met").length, notMet: rows.filter((r) => r.status === "not-met").length,
    unknown: rows.filter((r) => r.status === "unknown").length, total: rows.length };
}

function parseRow(value: unknown, catalogSha: unknown, serverOutcomes?: Record<string, ServerEvidence>): Row {
  const row = object(value, "row"), caseId = text(row.caseId, "caseId"), runId = text(row.expectedRunId, "expectedRunId");
  requireValue(ARMS.includes(row.arm as ShopperArm), "Unknown arm");
  requireValue(["recorded", "unattempted", "invalid-sidecar"].includes(String(row.status)), "Unknown row status");
  requireValue(["recorded", "unattempted"].includes(String(row.formalStatus)), "Unknown formal status");
  const result: Row = { caseId, arm: row.arm as ShopperArm, runId, status: String(row.status), outcome: null,
    observedCheckout: false, purchase: false, revenueCents: 0, exclusions: [], preference: null, purchasedPreferenceCredit: 0,
    abstained: false, correctAbstention: false, interrupted: false, observedSteps: null, comparableSteps: null,
    elapsedMs: null, timingPrecision: null, serverConfirmed: false, serverMerchandiseRevenueCents: 0, serverShippingCents: null, readiness: [] };
  if (row.status !== "recorded") {
    if (serverOutcomes) { const server = serverOutcomes[runId]; requireValue(server && server.row.serverConfirmed === false, "Unrecorded slot cannot claim server receipt without independently bound choice"); }
    result.exclusions.push(String(row.status)); return result;
  }
  const session = object(row.session, "session"), evaluator = object(row.evaluator, "evaluator"), run = object(session.run, "run");
  requireValue(session.caseId === caseId && session.arm === row.arm && run.id === runId, "Recorded row/session identity mismatch");
  requireValue(session.realPurchase === false, "Only synthetic fake checkout evidence is accepted");
  requireValue(["completed", "abandoned", "constraint-failure", "technical-failure", "timeout"].includes(String(session.status)), "Unknown session outcome");
  requireValue(row.outcome === session.status, "Row/session outcome mismatch");
  result.outcome = String(session.status); result.observedCheckout = bool(session.fakeOrderConfirmed, "fakeOrderConfirmed");
  const protocol = object(session.protocol, "protocol"), timing = object(session.timingPrecision, "timingPrecision");
  const passed = bool(protocol.passed, "protocol.passed"), deviations = strings(protocol.deviations, "deviations");
  requireValue(passed === (deviations.length === 0), "Protocol/deviations contradiction");
  const precision = text(timing.sessionElapsed, "sessionElapsed");
  requireValue(["measured-monotonic", "estimated", "upper-bound"].includes(precision), "Unknown session timing precision");
  requireValue(["measured-monotonic", "partially-recorded", "not-recorded"].includes(String(timing.stepElapsed)), "Unknown step timing precision");
  const elapsed = number(session.elapsedMs, "elapsedMs"), steps = array(session.steps, "steps");
  result.observedSteps = steps.length; result.timingPrecision = precision;
  result.comparableSteps = timing.stepElapsed === "measured-monotonic" ? steps.length : null;
  result.elapsedMs = precision === "measured-monotonic" && typeof session.startedAt === "string" ? elapsed : null;
  result.interrupted = session.status === "technical-failure" && (session.startedAt === null || /interrupt|transport|disconnect|reconnect|terminat/i.test([session.error, ...deviations].join(" ")));
  const verification = object(session.verification, "verification"), hard = checks(verification.hardConstraints, "hardConstraints");
  result.preference = checks(verification.preferences, "preferences");
  const evidence = bool(evaluator.evidenceVerified, "evidenceVerified"), acceptable = bool(evaluator.acceptableSelection, "acceptableSelection");
  const hardMet = bool(evaluator.allHardConstraintsMet, "allHardConstraintsMet"), budget = bool(evaluator.budgetRespected, "budgetRespected");
  const expected = bool(evaluator.expectedOutcomeMet, "expectedOutcomeMet");
  const artifacts = array(row.artifacts, "artifacts").map((a) => object(a, "artifact check"));
  const verifiedIds = new Set(artifacts.filter((a) => bool(a.verified, "artifact.verified")).map((a) => text(a.artifactId, "artifactId")));
  const receipts = array(session.artifacts, "session.artifacts").map((a) => object(a, "artifact receipt"));
  const requiredEvidence = ["screenshot", "trace"].every((kind) => receipts.some((a) => {
    const artifact = object(a.artifact, "artifact"); return artifact.kind === kind && a.status === "captured" && verifiedIds.has(String(artifact.id));
  }));
  const allArtifactsVerified = artifacts.length > 0 && artifacts.every((a) => a.verified === true);
  const allEvidence = evidence && requiredEvidence && allArtifactsVerified;
  const budgetKnown = budget && precision !== "estimated" && session.startedAt !== null;
  const formal = row.formalStatus === "recorded" && passed && budgetKnown;
  result.abstained = session.choice === null && !result.observedCheckout && ["abandoned", "constraint-failure"].includes(String(session.status));
  result.correctAbstention = result.abstained && expected && formal && allEvidence;
  const choice = session.choice === null ? null : object(session.choice, "choice");
  let cents = 0, priceValid = false, choiceEvidence = false;
  if (choice) {
    text(choice.productId, "productId"); text(choice.variantId, "variantId");
    const price = number(choice.catalogPrice, "catalogPrice"), rounded = Math.round(price * 100);
    priceValid = Number.isSafeInteger(rounded) && Math.abs(price * 100 - rounded) < 1e-7 && choice.quantity === 1;
    if (priceValid) cents = rounded;
    const ids = strings(choice.evidenceArtifactIds, "choice.evidenceArtifactIds");
    choiceEvidence = ids.length > 0 && ids.every((id) => verifiedIds.has(id));
  }
  const linkedCatalog = typeof catalogSha === "string" && HASH.test(catalogSha) && verification.catalogSha256 === catalogSha;
  if (serverOutcomes) {
    const evidence = serverOutcomes[runId]; requireValue(evidence, "Server outcomes must retain every assigned run");
    const confirmed = serverReceipt(evidence, result, choice);
    if (confirmed) { result.serverConfirmed = true; result.serverMerchandiseRevenueCents = confirmed.subtotal; result.serverShippingCents = confirmed.shippingCents; }
  }
  const requirements: [string, boolean][] = [["no-observed-checkout", result.observedCheckout], ["not-completed", session.status === "completed"],
    ["not-formal", formal], ["budget-unverified-or-exceeded", budgetKnown], ["evidence-unverified", allEvidence],
    ["choice-not-independent-dom", choice?.observationSource === "independent-dom"], ["choice-evidence-unverified", choiceEvidence],
    ["catalog-binding-unverified", linkedCatalog && verification.independentOfShopperAgent === true], ["unacceptable-gold-selection", acceptable && expected],
    ["hard-constraints-unmet-or-unknown", hardMet && hard.met === hard.total], ["invalid-unit-price-or-quantity", priceValid],
    ...(serverOutcomes ? [["server-receipt-unconfirmed", result.serverConfirmed] as [string, boolean]] : [])];
  result.exclusions = requirements.filter(([, met]) => !met).map(([label]) => label);
  result.purchase = result.exclusions.length === 0; result.revenueCents = result.purchase ? cents : 0;
  if (result.purchase && result.preference.total) result.purchasedPreferenceCredit = result.preference.met / result.preference.total;
  const model = object(session.model, "model"), seen = new Map<string, number>();
  for (const value of array(model.readiness, "readiness")) {
    const measure = object(value, "readiness measure"), stage = text(measure.stage, "stage"), requestId = text(measure.requestId, "requestId");
    requireValue(["rank", "firstThree", "complete"].includes(stage), "Unknown readiness stage");
    const durationMs = number(measure.durationMs, "durationMs"), key = `${stage}/${requestId}`;
    if (seen.has(key)) { requireValue(seen.get(key) === durationMs, "Conflicting duplicate readiness timing"); continue; }
    seen.set(key, durationMs);
    if (verifiedIds.has(String(measure.artifactId)) && typeof measure.capturedAtElapsedMs === "number" && Number.isFinite(measure.capturedAtElapsedMs)
      && measure.capturedAtElapsedMs >= 0 && measure.scope === "after-paint-opportunity"
      && measure.physicalPaintMeasured === false && measure.imageTransferMeasured === false) result.readiness.push({ stage, requestId, durationMs });
  }
  return result;
}

function summarize(rows: Row[], assigned: number, serverEvaluated: boolean) {
  const purchases = rows.filter((r) => r.purchase), preferences = rows.flatMap((r) => r.preference ? [r.preference] : []);
  const purchasedPreferences = purchases.filter((r) => r.preference!.total > 0);
  const confirmed = rows.filter((r) => r.serverConfirmed), knownShipping = confirmed.flatMap((r) => r.serverShippingCents === null ? [] : [r.serverShippingCents]);
  requireValue([sum(rows.map((r) => r.revenueCents)), sum(rows.map((r) => r.serverMerchandiseRevenueCents)), sum(knownShipping)].every(Number.isSafeInteger), "Revenue total exceeds safe whole cents");
  return { assigned, recorded: rows.filter((r) => r.status === "recorded").length,
    observedFakeCheckouts: rows.filter((r) => r.observedCheckout).length, verifiedPurchases: purchases.length,
    intentCorrectSuccess: { numerator: rows.filter((r) => r.purchase || r.correctAbstention).length, denominator: assigned,
      rate: rows.filter((r) => r.purchase || r.correctAbstention).length / assigned,
      interval95: wilson95(rows.filter((r) => r.purchase || r.correctAbstention).length, assigned) },
    syntheticConversion: { numerator: purchases.length, denominator: assigned, rate: purchases.length / assigned, interval95: wilson95(purchases.length, assigned) },
    revenue: { currency: "USD", quantityPerPurchase: 1, totalCents: sum(rows.map((r) => r.revenueCents)),
      revenuePerAssignedSessionCents: sum(rows.map((r) => r.revenueCents)) / assigned, nonpurchaseSessions: assigned - purchases.length },
    allServerConfirmedOrders: { evaluated: serverEvaluated, count: rows.filter((r) => r.serverConfirmed).length,
      assigned, rate: serverEvaluated ? rows.filter((r) => r.serverConfirmed).length / assigned : null,
      interval95: serverEvaluated ? wilson95(rows.filter((r) => r.serverConfirmed).length, assigned) : null,
      merchandiseSubtotalCents: sum(rows.map((r) => r.serverMerchandiseRevenueCents)),
      merchandiseRevenuePerAssignedSessionCents: sum(rows.map((r) => r.serverMerchandiseRevenueCents)) / assigned,
      shippingKnown: confirmed.length > 0 && knownShipping.length === confirmed.length,
      shippingCents: confirmed.length > 0 && knownShipping.length === confirmed.length ? sum(knownShipping) : null,
      knownShippingCents: sum(knownShipping), shippingKnownOrders: knownShipping.length, shippingUnknownOrders: confirmed.length - knownShipping.length,
      scope: "Server-confirmed quantity-one orders, including confirmed orders excluded from valid-purchase scoring; revenue is merchandise subtotal only. Hidden shipping remains unknown, never an observed zero charge." },
    outcomes: Object.fromEntries(["completed", "abandoned", "constraint-failure", "timeout", "technical-failure", "unattempted", "invalid-sidecar"].map((outcome) => [outcome, rows.filter((r) => (r.outcome ?? r.status) === outcome).length])),
    abstentions: { observed: rows.filter((r) => r.abstained).length, verifiedExpected: rows.filter((r) => r.correctAbstention).length },
    interruptions: rows.filter((r) => r.interrupted).length,
    technicalFailuresExcludingInterruptions: rows.filter((r) => r.outcome === "technical-failure" && !r.interrupted).length,
    purchaseExclusions: Object.fromEntries([...new Set(rows.flatMap((r) => r.exclusions))].sort().map((reason) => [reason, rows.filter((r) => r.exclusions.includes(reason)).length])),
    preference: { observedChecks: { met: sum(preferences.map((p) => p.met)), notMet: sum(preferences.map((p) => p.notMet)),
        unknown: sum(preferences.map((p) => p.unknown)), total: sum(preferences.map((p) => p.total)), unavailableSessions: rows.filter((r) => !r.preference).length },
      verifiedPurchasedMetChecks: sum(purchases.map((r) => r.preference!.met)),
      verifiedPurchasedMetChecksPerAssignedSession: sum(purchases.map((r) => r.preference!.met)) / assigned,
      purchasedMetFractionPerAssignedSession: sum(rows.map((r) => r.purchasedPreferenceCredit)) / assigned,
      purchasedMetFractionAmongPurchasersWithSpecifiedPreferences: mean(purchasedPreferences.map((r) => r.purchasedPreferenceCredit)),
      purchasersWithSpecifiedPreferences: purchasedPreferences.length, purchasersWithoutSpecifiedPreferences: purchases.length - purchasedPreferences.length,
      purchasedUnknownChecks: sum(purchases.map((r) => r.preference!.unknown)),
      scoreDefinition: "For each assigned slot: verified purchase ? met preference checks / specified preference checks : 0; slots with no specified preferences score 0. Unknown checks remain in the fraction denominator; no-preference counts are disclosed." },
    effort: { observedSteps: distribution(rows.flatMap((r) => r.observedSteps === null ? [] : [r.observedSteps])),
      fullyTimedSteps: distribution(rows.flatMap((r) => r.comparableSteps === null ? [] : [r.comparableSteps])),
      measuredSessionElapsedMs: distribution(rows.flatMap((r) => r.elapsedMs === null ? [] : [r.elapsedMs])),
      excludedSessionTiming: rows.filter((r) => r.elapsedMs === null).map((r) => ({ caseId: r.caseId, precision: r.timingPrecision, status: r.status })) },
    browserReadinessMs: Object.fromEntries(["rank", "firstThree", "complete"].map((stage) => [stage,
      distribution(rows.flatMap((r) => r.readiness.filter((m) => m.stage === stage).map((m) => m.durationMs)))])),
  };
}

export function analyzeShopperMetrics(input: unknown, options: MetricsOptions = {}) {
  const report = object(input, "aggregate");
  requireValue(report.version === "shopper-evidence-aggregate.v1" && report.synthetic === true && report.humanConversionClaim === false, "Validated synthetic aggregate required");
  const assigned = integer(report.fixedCases, "fixedCases"); requireValue(assigned > 0 && assigned <= 10000, "Bounded positive fixedCases required");
  requireValue(report.fixedSessions === assigned * 2, "Fixed assigned session denominator differs");
  const rows = array(report.rows, "rows").map((r) => parseRow(r, report.catalogSnapshotSha256, options.serverOutcomes));
  requireValue(rows.length === assigned * 2, "Every assigned slot must be retained, including missing/failure rows");
  const caseIds = [...new Set(rows.map((r) => r.caseId))].sort(); requireValue(caseIds.length === assigned, "Case count differs from fixed denominator");
  if (options.caseEligibility) {
    requireValue(Object.keys(options.caseEligibility).length === assigned && caseIds.every((id) => ["satisfiable", "known-empty", "unverifiable-required-fact"].includes(options.caseEligibility![id])), "Eligibility metadata must cover exactly the assigned case IDs");
    for (const row of rows) {
      requireValue(!(row.purchase && options.caseEligibility[row.caseId] !== "satisfiable"), "Purchase contradicts prespecified eligibility");
      requireValue(!(row.correctAbstention && options.caseEligibility[row.caseId] === "satisfiable"), "Correct abstention contradicts prespecified feasible scenario");
    }
  }
  requireValue(new Set(rows.map((r) => r.runId)).size === rows.length, "Duplicate run ID");
  if (options.serverOutcomes) {
    requireValue(Object.keys(options.serverOutcomes).length === rows.length && rows.every((row) => Object.hasOwn(options.serverOutcomes!, row.runId)), "Server outcome roster differs from assigned runs");
    const confirmed = Object.values(options.serverOutcomes).filter((entry) => entry.row.serverConfirmed === true).map((entry) => object(entry.row.receipt, "receipt"));
    for (const field of ["checkoutId", "receiptId"]) requireValue(new Set(confirmed.map((r) => r[field])).size === confirmed.length, "Server receipt IDs reused across assigned sessions");
  }
  for (const caseId of caseIds) for (const arm of ARMS) requireValue(rows.filter((r) => r.caseId === caseId && r.arm === arm).length === 1, "Exactly one slot per case/arm required");
  const pairs = caseIds.map((caseId) => {
    const generic = rows.find((r) => r.caseId === caseId && r.arm === "generic")!, local = rows.find((r) => r.caseId === caseId && r.arm === "local-model")!;
    return { caseId, eligibility: options.caseEligibility?.[caseId] ?? null, generic, "local-model": local, differences: { conversion: Number(local.purchase) - Number(generic.purchase),
      intentCorrectSuccess: Number(local.purchase || local.correctAbstention) - Number(generic.purchase || generic.correctAbstention),
      serverConfirmedConversion: Number(local.serverConfirmed) - Number(generic.serverConfirmed),
      allServerRevenuePerSessionCents: local.serverMerchandiseRevenueCents - generic.serverMerchandiseRevenueCents,
      revenuePerSessionCents: local.revenueCents - generic.revenueCents, purchasedPreferenceFraction: local.purchasedPreferenceCredit - generic.purchasedPreferenceCredit } };
  });
  return { version: "shopper-synthetic-metrics.v1", synthetic: true, humanConversionClaim: false, fixedCases: assigned, fixedSessions: assigned * 2,
    currency: "USD", sourceEvidenceReverified: false, sourceVersion: report.version,
    prespecifiedEligibility: options.caseEligibility ? Object.fromEntries(["satisfiable", "known-empty", "unverifiable-required-fact"].map((status) => [status, caseIds.filter((id) => options.caseEligibility![id] === status).length])) : null,
    primaryOutcome: "Intent-correct success: verified valid fake checkout or independently justified and evidenced abstention",
    checkoutEvidencePolicy: options.serverOutcomes ? "Server-confirmed fake order plus independent-DOM, catalog/gold, constraints, protocol and budget gates; receipt artifacts hashed and GET readback verified"
      : "Legacy independent-DOM receipt evidence; server order confirmation is not established by this report",
    arms: Object.fromEntries(ARMS.map((arm) => {
      const selected = rows.filter((r) => r.arm === arm), feasible = selected.filter((r) => options.caseEligibility?.[r.caseId] === "satisfiable");
      const purchases = feasible.filter((r) => r.purchase).length;
      return [arm, { ...summarize(selected, assigned, !!options.serverOutcomes), prespecifiedFeasibleCheckout: options.caseEligibility ? {
        numerator: purchases, denominator: feasible.length, rate: feasible.length ? purchases / feasible.length : null,
        interval95: feasible.length ? wilson95(purchases, feasible.length) : null,
      } : null }];
    })),
    pairedDifferences: Object.fromEntries((["intentCorrectSuccess", "conversion", "revenuePerSessionCents", "purchasedPreferenceFraction"] as const)
      .map((metric) => { const differences = pairs.map((p) => p.differences[metric]), interval = pairedBootstrap95(differences, options.seed, options.replicates); return [metric, {
        ...interval,
        localWins: differences.filter((d) => d > 0).length, genericWins: differences.filter((d) => d < 0).length, ties: differences.filter((d) => d === 0).length,
        ...(metric === "conversion" || metric === "intentCorrectSuccess" ? { absolutePercentagePoints: 100 * interval.difference,
          lowerPercentagePoints: 100 * interval.lower, upperPercentagePoints: 100 * interval.upper } : {}),
      }]; })),
    feasibleCheckoutPairedDifference: options.caseEligibility && pairs.some((p) => p.eligibility === "satisfiable")
      ? pairedBootstrap95(pairs.filter((p) => p.eligibility === "satisfiable").map((p) => p.differences.conversion), options.seed, options.replicates) : null,
    serverConfirmedPairedDifferences: options.serverOutcomes ? Object.fromEntries((["serverConfirmedConversion", "allServerRevenuePerSessionCents"] as const).map((metric) => {
      const differences = pairs.map((p) => p.differences[metric]); return [metric, { ...pairedBootstrap95(differences, options.seed, options.replicates),
        localWins: differences.filter((d) => d > 0).length, genericWins: differences.filter((d) => d < 0).length, ties: differences.filter((d) => d === 0).length }];
    })) : null, pairs,
    limitations: ["Synthetic fixed scenario panel, not human conversion lift or independent identically distributed customers.",
      "Descriptive Wilson intervals and a paired case bootstrap show finite-panel uncertainty; they do not establish population effects, causal lift, or generalization. Small samples can produce degenerate bootstrap intervals.",
      "Every assigned slot remains in conversion and revenue denominators; nonpurchases, missing slots, invalid evidence, interruptions and timeouts contribute zero revenue.",
      "Purchases require validated aggregate gold/constraint/catalog checks plus verified independent-DOM fake checkout, formal protocol and budget. This analyzer consumes those recorded checks; it does not rehash artifacts or independently rescore gold/catalog.",
      "Preference credits count only verified purchases. Unobserved checks are not fabricated, unknown checks do not earn credit, and no-preference cases are reported separately.",
      "Comparable elapsed times require measured monotonic total runtime and a known start. Estimated, upper-bound and missing timings are excluded; partial observed step counts remain separately visible.",
      "Browser readiness is a verified after-paint opportunity measure, not physical paint, image transfer, native GPU latency or end-to-end shopper latency."] };
}

/** Combines distinct frozen panels; duplicate case/run IDs and changed catalogs fail closed. */
export function analyzeShopperMetricReports(inputs: unknown[], options: MetricsOptions = {}) {
  requireValue(inputs.length > 0 && inputs.length <= 100, "One to 100 aggregate reports required");
  const reports = inputs.map((input) => object(input, "aggregate"));
  let fixedCases = 0;
  for (const report of reports) {
    requireValue(report.version === "shopper-evidence-aggregate.v1" && report.synthetic === true && report.humanConversionClaim === false, "Validated synthetic aggregate required");
    const count = integer(report.fixedCases, "fixedCases");
    requireValue(count > 0 && report.fixedSessions === count * 2 && array(report.rows, "rows").length === count * 2, "Each source must retain its fixed assigned slots");
    requireValue(report.catalogSnapshotSha256 === reports[0].catalogSnapshotSha256, "Combined reports must share the frozen catalog and prices");
    // Source-local denominator and pairing checks are also required, not only pooled checks.
    const sourceRows = array(report.rows, "rows").map((r) => object(r, "row"));
    const cases = new Set(sourceRows.map((r) => text(r.caseId, "caseId")));
    requireValue(cases.size === count, "Source case count differs from denominator");
    for (const caseId of cases) for (const arm of ARMS) requireValue(sourceRows.filter((r) => r.caseId === caseId && r.arm === arm).length === 1, "Source pairing differs");
    fixedCases += count;
  }
  return { ...analyzeShopperMetrics({ ...reports[0], fixedCases, fixedSessions: fixedCases * 2,
    rows: reports.flatMap((r) => array(r.rows, "rows")) }, options), sourceReports: reports.length };
}

export async function shopperMetricsMain(args: string[]) {
  const usage = "Usage: shopper-metrics.ts --report AGGREGATE.json [--packet PACKET_DIR] [--report OTHER.json --packet OTHER_DIR ...] [--server-outcomes INDEX.json] --out NEW.json";
  requireValue(args.length >= 4 && args.length % 2 === 0 && args.at(-2) === "--out", usage);
  const reportPaths: string[] = [], packetPaths: string[] = []; let serverIndexPath: string | null = null;
  for (let index = 0; index < args.length - 2; index += 2) {
    requireValue(["--report", "--packet", "--server-outcomes"].includes(args[index]) && !!args[index + 1], usage);
    if (args[index] === "--server-outcomes") { requireValue(serverIndexPath === null, "Only one server index allowed"); serverIndexPath = path.resolve(args[index + 1]); }
    else (args[index] === "--report" ? reportPaths : packetPaths).push(path.resolve(args[index + 1]));
  }
  requireValue(reportPaths.length > 0 && (packetPaths.length === 0 || packetPaths.length === reportPaths.length), "Supply one packet per report or omit all packets");
  const outputPath = path.resolve(args.at(-1)!);
  requireValue(!reportPaths.includes(outputPath), "Output must differ from source report");
  requireValue(new Set(reportPaths).size === reportPaths.length, "Duplicate source report path");
  const inputs = await Promise.all(reportPaths.map(async (reportPath) => {
    requireValue((await stat(reportPath)).size <= 16 * 1024 * 1024, "Aggregate exceeds 16 MiB");
    const bytes = await readFile(reportPath); return { report: JSON.parse(bytes.toString("utf8")) as unknown,
      source: { path: reportPath, sha256: createHash("sha256").update(bytes).digest("hex") } };
  }));
  const caseEligibility: Record<string, Eligibility> = {};
  const packetSources = await Promise.all(packetPaths.map(async (directory, index) => {
    const manifestBytes = await readFile(path.join(directory, "manifest.json")), manifest = object(JSON.parse(manifestBytes.toString("utf8")), "packet manifest");
    const goldBytes = await readFile(path.join(directory, "evaluator-gold.json")), goldSha = createHash("sha256").update(goldBytes).digest("hex");
    requireValue(object(manifest.files, "manifest files")["evaluator-gold.json"] === goldSha, "Evaluator gold differs from frozen manifest");
    const aggregate = object(inputs[index].report, "aggregate");
    requireValue(aggregate.evaluatorGoldSha256 === goldSha, "Packet gold differs from aggregate gold binding");
    const ids = new Set(array(aggregate.rows, "rows").map((r) => text(object(r, "row").caseId, "caseId")));
    const gold = array(JSON.parse(goldBytes.toString("utf8")), "gold").map((g) => object(g, "gold row"));
    requireValue(gold.length === ids.size && new Set(gold.map((g) => g.caseId)).size === ids.size, "Gold case count differs");
    for (const row of gold) {
      const id = text(row.caseId, "gold caseId"); requireValue(ids.has(id) && !Object.hasOwn(caseEligibility, id), "Gold case membership duplicated or differs");
      requireValue(["satisfiable", "known-empty", "unverifiable-required-fact"].includes(String(row.eligibilityStatus)), "Unknown gold eligibility");
      caseEligibility[id] = row.eligibilityStatus as Eligibility;
    }
    return { path: directory, evaluatorGoldSha256: goldSha, manifestSha256: createHash("sha256").update(manifestBytes).digest("hex") };
  }));
  let serverIndexSource: { path: string; sha256: string } | null = null;
  const serverOutcomes: Record<string, ServerEvidence> = {};
  if (serverIndexPath) {
    requireValue(outputPath !== serverIndexPath, "Output must differ from server index");
    requireValue((await stat(serverIndexPath)).size <= 16 * 1024 * 1024, "Server index exceeds 16 MiB");
    const bytes = await readFile(serverIndexPath), index = object(JSON.parse(bytes.toString("utf8")), "server index");
    requireValue(index.schema === "shopper-server-outcomes.v1" && index.synthetic === true && index.humanConversionClaim === false, "Versioned synthetic server index required");
    const indexedReports = array(index.reports, "server index reports").map((r) => object(r, "server report"));
    requireValue(indexedReports.length === inputs.length && new Set(indexedReports.map((r) => r.path)).size === inputs.length, "Server index report roster differs");
    for (const input of inputs) requireValue(indexedReports.some((r) => r.path === input.source.path && r.sha256 === input.source.sha256), "Server index does not bind exact report bytes/path");
    const assignedRows = inputs.flatMap((input) => array(object(input.report, "aggregate").rows, "rows").map((r) => object(r, "row")));
    const indexedRows = array(index.rows, "server index rows").map((r) => object(r, "server outcome"));
    requireValue(indexedRows.length === assignedRows.length, "Server index must retain every assigned slot");
    for (const row of indexedRows) {
      const runId = text(row.runId, "server runId");
      requireValue(!Object.hasOwn(serverOutcomes, runId) && assignedRows.some((r) => r.expectedRunId === runId && r.caseId === row.caseId && r.arm === row.arm), "Duplicate or mismatched server run identity");
      const confirmed = bool(row.serverConfirmed, "serverConfirmed"); strings(row.errors, "server errors");
      let artifactHashVerified = false;
      if (row.artifact !== null) {
        const artifact = object(row.artifact, "server artifact"), artifactPath = text(artifact.path, "server artifact path");
        requireValue(path.isAbsolute(artifactPath) && typeof artifact.sha256 === "string" && HASH.test(artifact.sha256), "Absolute hashed server evidence path required");
        const expectedBytes = integer(artifact.bytes, "server artifact bytes"); requireValue(expectedBytes <= 16 * 1024 * 1024, "Server artifact exceeds 16 MiB");
        requireValue((await stat(artifactPath)).size === expectedBytes, "Server artifact byte count differs");
        const evidenceBytes = await readFile(artifactPath);
        requireValue(createHash("sha256").update(evidenceBytes).digest("hex") === artifact.sha256, "Server artifact hash differs");
        const assigned = assignedRows.find((r) => r.expectedRunId === runId)!;
        verifyServerArtifactContent(JSON.parse(evidenceBytes.toString("utf8")), row, assigned);
        artifactHashVerified = true;
      }
      requireValue(!confirmed || artifactHashVerified, "Server confirmation requires hashed evidence artifact");
      serverOutcomes[runId] = { row, artifactHashVerified };
    }
    serverIndexSource = { path: serverIndexPath, sha256: createHash("sha256").update(bytes).digest("hex") };
  }
  const output = { ...analyzeShopperMetricReports(inputs.map((input) => input.report), {
    ...(packetPaths.length ? { caseEligibility } : {}), ...(serverIndexPath ? { serverOutcomes } : {}) }),
    sources: inputs.map((input) => input.source), packetSources, serverIndexSource };
  await writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`, { flag: "wx" }); return output;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  shopperMetricsMain(process.argv.slice(2)).then((result) => process.stdout.write(`${JSON.stringify({ version: result.version, fixedCases: result.fixedCases, arms: result.arms, pairedDifferences: result.pairedDifferences }, null, 2)}\n`))
    .catch((error: unknown) => { process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`); process.exitCode = 1; });
}
