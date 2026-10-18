/** Result-side only. Descriptive deck data from final saved evidence; never dispatches agents or network requests. */
import { createHash } from "node:crypto";
import { readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { pairedBootstrap95, wilson95 } from "../../../../apps/runner/src/shopper-metrics.ts";

type Obj = Record<string, any>;
const ARMS = ["generic", "local-model"] as const;
const WORKSPACE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");
const HASH = /^[a-f0-9]{64}$/;
const sha = (bytes: Buffer | string) => createHash("sha256").update(bytes).digest("hex");
function requireValue(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }
function object(value: unknown, label: string): Obj { requireValue(value && typeof value === "object" && !Array.isArray(value), `${label}: object required`); return value as Obj; }
function list(value: unknown, label: string): any[] { requireValue(Array.isArray(value), `${label}: array required`); return value; }
function text(value: unknown, label: string): string { requireValue(typeof value === "string" && value.length > 0, `${label}: text required`); return value; }
function number(value: unknown, label: string): number { requireValue(typeof value === "number" && Number.isFinite(value) && value >= 0, `${label}: nonnegative finite number required`); return value; }
function integer(value: unknown, label: string): number { const result = number(value, label); requireValue(Number.isSafeInteger(result), `${label}: whole number required`); return result; }
function bool(value: unknown, label: string): boolean { requireValue(typeof value === "boolean", `${label}: boolean required`); return value; }
function digest(value: unknown, label: string): string { requireValue(typeof value === "string" && HASH.test(value), `${label}: SHA-256 required`); return value; }
const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);
const round = (n: number) => Math.round(n * 1e6) / 1e6;
function quantile(values: number[], p: number) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b), at = (sorted.length - 1) * p, low = Math.floor(at), high = Math.ceil(at);
  return round(sorted[low] + (sorted[high] - sorted[low]) * (at - low));
}
const measured = (values: number[]) => ({ n: values.length, median: quantile(values, .5) });
function rate(n: number, d: number, includeInterval = true) {
  const interval = d ? wilson95(n, d) : null;
  return { n, d, rate: d ? round(n / d) : null, ...(includeInterval ? { ci95: interval ? [round(interval.lower), round(interval.upper)] : null } : {}) };
}
function contrast(values: number[], unit: "pp" | "cents" | "fraction") {
  if (!values.length) return { n: 0, unit, difference: null, ci95: null, winsLossesTies: [0, 0, 0] };
  const result = pairedBootstrap95(values, 20261018, 20000), scale = unit === "pp" ? 100 : 1;
  return { n: values.length, unit, difference: round(result.difference * scale), ci95: [round(result.lower * scale), round(result.upper * scale)],
    winsLossesTies: [values.filter((n) => n > 0).length, values.filter((n) => n < 0).length, values.filter((n) => n === 0).length] };
}
function checkedRow(value: unknown, caseId: string, arm: string): Obj {
  const row = object(value, "derived metrics row");
  requireValue(row.caseId === caseId && row.arm === arm, "Derived row identity differs from pair");
  text(row.runId, "runId"); requireValue(["recorded", "unattempted", "invalid-sidecar"].includes(row.status), "Invalid row status");
  for (const key of ["purchase", "correctAbstention", "abstained", "interrupted", "serverConfirmed", "visibleCheckoutConfirmed", "observedCheckout"]) bool(row[key], key);
  for (const key of ["revenueCents", "serverMerchandiseRevenueCents"]) integer(row[key], key);
  number(row.purchasedPreferenceCredit, "preference credit");
  requireValue(row.purchasedPreferenceCredit <= 1, "Preference fraction exceeds one");
  for (const key of ["observedSteps", "comparableSteps", "elapsedMs"]) if (row[key] !== null) number(row[key], key);
  requireValue(!row.visibleCheckoutConfirmed || row.serverConfirmed, "Visible confirmation without server order");
  requireValue(row.purchase || row.revenueCents === 0, "Nonpurchase revenue must remain zero");
  requireValue(row.serverConfirmed || row.serverMerchandiseRevenueCents === 0, "Unconfirmed server revenue must remain zero");
  requireValue(!row.purchase || (row.serverConfirmed && row.visibleCheckoutConfirmed && row.status === "recorded" && row.outcome === "completed"), "Purchase lacks recorded server and visible completion");
  requireValue(!row.correctAbstention || (row.status === "recorded" && row.abstained && !row.purchase && !row.serverConfirmed), "Correct abstention contradicts purchase evidence");
  if (row.status !== "recorded") requireValue(!row.purchase && !row.correctAbstention && !row.serverConfirmed && row.elapsedMs === null, "Unrecorded slot claims an observed success");
  else requireValue(["completed", "abandoned", "constraint-failure", "technical-failure", "timeout"].includes(row.outcome), "Invalid recorded outcome");
  if (row.elapsedMs !== null) requireValue(row.timingPrecision === "measured-monotonic", "Comparable runtime must be measured monotonic");
  if (row.preference !== null) {
    const preference = object(row.preference, "preferences");
    for (const key of ["met", "notMet", "unknown", "total"]) integer(preference[key], `preferences.${key}`);
    requireValue(preference.met + preference.notMet + preference.unknown === preference.total, "Preference check counts differ");
    const expectedCredit = row.purchase && preference.total ? preference.met / preference.total : 0;
    requireValue(Math.abs(row.purchasedPreferenceCredit - expectedCredit) < 1e-10, "Preference credit differs from validated derived checks");
  } else requireValue(row.purchasedPreferenceCredit === 0, "Unavailable preference evidence earned credit");
  for (const item of list(row.readiness, "readiness")) { const measure = object(item, "readiness measure"); requireValue(["rank", "firstThree", "complete"].includes(measure.stage), "Invalid readiness stage"); number(measure.durationMs, "durationMs"); }
  return row;
}
function difference(pair: Obj, metric: string): number {
  const g = pair.generic, l = pair["local-model"];
  if (metric === "intent") return Number(l.purchase || l.correctAbstention) - Number(g.purchase || g.correctAbstention);
  if (metric === "checkout") return Number(l.purchase) - Number(g.purchase);
  if (metric === "validRps") return l.revenueCents - g.revenueCents;
  if (metric === "serverRps") return l.serverMerchandiseRevenueCents - g.serverMerchandiseRevenueCents;
  return l.purchasedPreferenceCredit - g.purchasedPreferenceCredit;
}
function panel(pairs: Obj[], includeLatency: boolean) {
  const feasible = pairs.filter((p) => p.eligibility === "satisfiable");
  return { pairs: pairs.length, assignedSessions: pairs.length * 2, arms: Object.fromEntries(ARMS.map((arm) => {
    const rows = pairs.map((p) => p[arm]), purchases = rows.filter((r) => r.purchase), server = rows.filter((r) => r.serverConfirmed);
    const preferred = purchases.filter((r) => r.preference?.total > 0);
    const outcomes = Object.fromEntries(["completed", "abandoned", "constraint-failure", "technical-failure", "timeout", "unattempted", "invalid-sidecar"].map((status) => [status, rows.filter((r) => (r.outcome ?? r.status) === status).length]));
    return [arm, { assigned: rows.length, recorded: rows.filter((r) => r.status === "recorded").length,
      intent: rate(rows.filter((r) => r.purchase || r.correctAbstention).length, rows.length, includeLatency),
      validCheckout: rate(purchases.length, rows.length, includeLatency), feasibleCheckout: rate(feasible.filter((p) => p[arm].purchase).length, feasible.length, includeLatency),
      revenue: { validTotalCents: sum(rows.map((r) => r.revenueCents)), validPerAssignedCents: round(sum(rows.map((r) => r.revenueCents)) / rows.length),
        serverOrders: server.length, serverTotalCents: sum(rows.map((r) => r.serverMerchandiseRevenueCents)),
        serverPerAssignedCents: round(sum(rows.map((r) => r.serverMerchandiseRevenueCents)) / rows.length),
        serverWithoutVisibleConfirmation: server.filter((r) => !r.visibleCheckoutConfirmed).length },
      preferenceProxy: { perAssigned: round(sum(rows.map((r) => r.purchasedPreferenceCredit)) / rows.length),
        ...(includeLatency ? {
        amongPurchasersWithPreferences: preferred.length ? round(sum(preferred.map((r) => r.purchasedPreferenceCredit)) / preferred.length) : null,
        purchasersWithPreferences: preferred.length, purchasersWithoutPreferences: purchases.length - preferred.length,
        observedUnknownChecks: sum(rows.map((r) => r.preference?.unknown ?? 0)), unavailableSessions: rows.filter((r) => r.preference === null).length } : {}) },
      ...(includeLatency ? { effort: { steps: measured(rows.filter((r) => r.comparableSteps !== null).map((r) => r.comparableSteps)),
        elapsedMs: measured(rows.filter((r) => r.elapsedMs !== null).map((r) => r.elapsedMs)), excludedElapsed: rows.filter((r) => r.elapsedMs === null).length } } : {}),
      outcomes: { ...outcomes, interruptions: rows.filter((r) => r.interrupted).length, abstentions: rows.filter((r) => r.abstained).length,
        justifiedAbstentions: rows.filter((r) => r.correctAbstention).length },
      ...(includeLatency ? { browserReadinessMs: Object.fromEntries(["rank", "firstThree", "complete"].map((stage) => {
        const values = rows.flatMap((r) => r.readiness.filter((m: Obj) => m.stage === stage).map((m: Obj) => m.durationMs));
        return [stage, { ...measured(values), p95: quantile(values, .95) }];
      })) } : {}),
    }];
  })), paired: { intent: contrast(pairs.map((p) => difference(p, "intent")), "pp"),
    feasibleCheckout: contrast(feasible.map((p) => difference(p, "checkout")), "pp"),
    validRps: contrast(pairs.map((p) => difference(p, "validRps")), "cents"),
    allServerRps: contrast(pairs.map((p) => difference(p, "serverRps")), "cents"),
    preferenceProxy: contrast(pairs.map((p) => difference(p, "preference")), "fraction") } };
}

/** Pure projection of already validated derived rows. No new product/constraint/gold scoring. */
export function buildSlideMetrics(metricsInput: unknown, planInput: unknown, serverInput: unknown) {
  const metrics = object(metricsInput, "metrics"), plan = object(planInput, "plan"), server = object(serverInput, "server index");
  requireValue(metrics.version === "shopper-synthetic-metrics.v1" && metrics.synthetic === true && metrics.humanConversionClaim === false, "Synthetic full metrics required");
  requireValue(plan.version === "shopper-fixed-cohort.v1" && plan.synthetic === true && plan.humanConversionClaim === false, "Frozen synthetic cohort plan required");
  requireValue(plan.fixedPairs === 60 && plan.fixedSessions === 120 && plan.pilotPairs === 5 && metrics.fixedCases === 60 && metrics.fixedSessions === 120, "Only full 60-pair/120-assigned cohort accepted");
  requireValue(server.schema === "shopper-server-outcomes.v1" && server.synthetic === true && server.humanConversionClaim === false, "Synthetic server index required");
  const schedule = list(plan.schedule, "schedule").map((r) => object(r, "scheduled row"));
  requireValue(schedule.length === 120 && new Set(schedule.map((r) => r.runId)).size === 120, "Exactly 120 unique scheduled runs required");
  const pilotIds = [...new Set(schedule.filter((r) => bool(r.pilot, "pilot")).map((r) => text(r.caseId, "scheduled caseId")))];
  requireValue(pilotIds.length === 5 && schedule.filter((r) => r.pilot).length === 10, "Frozen first-five pilot membership required");
  requireValue(schedule.slice(0, 10).every((r) => r.pilot === true) && schedule.slice(10).every((r) => r.pilot === false), "Pilot must be the prespecified first five pairs");
  const pairs = list(metrics.pairs, "metrics pairs").map((value) => {
    const pair = object(value, "pair"), id = text(pair.caseId, "caseId");
    requireValue(["satisfiable", "known-empty", "unverifiable-required-fact"].includes(pair.eligibility), "Prespecified eligibility required");
    for (const arm of ARMS) {
      const row = checkedRow(pair[arm], id, arm);
      requireValue(schedule.filter((s) => s.caseId === id && s.arm === arm && s.runId === row.runId).length === 1, "Metrics row differs from frozen schedule");
    }
    requireValue(schedule.filter((s) => s.caseId === id).length === 2, "Case does not preserve two assigned arms");
    return pair;
  });
  requireValue(pairs.length === 60 && new Set(pairs.map((p) => p.caseId)).size === 60 && pairs.filter((p) => p.eligibility === "satisfiable").length === 48, "60 unique pairs and 48 prespecified feasible cases required");
  const serverRows = list(server.rows, "server rows").map((r) => object(r, "server row"));
  requireValue(serverRows.length === 120 && new Set(serverRows.map((r) => r.runId)).size === 120, "Server roster must retain all assigned runs");
  for (const pair of pairs) for (const arm of ARMS) {
    const row = pair[arm], indexed = serverRows.find((r) => r.runId === row.runId);
    requireValue(indexed?.caseId === pair.caseId && indexed?.arm === arm && indexed?.serverConfirmed === row.serverConfirmed && indexed?.visibleCheckoutConfirmed === row.visibleCheckoutConfirmed, "Metrics/server outcome flags or identities differ");
    if (row.serverConfirmed) requireValue(Math.round(number(object(indexed.receipt, "receipt").subtotal, "receipt subtotal") * 100) === row.serverMerchandiseRevenueCents, "Server merchandise revenue differs");
  }
  const full = panel(pairs, true), pilot = pairs.filter((p) => pilotIds.includes(p.caseId)), remaining = pairs.filter((p) => !pilotIds.includes(p.caseId));
  for (const arm of ARMS) {
    const original = object(object(metrics.arms, "arms")[arm], "original arm"), derived = full.arms[arm];
    requireValue(original.assigned === 60 && original.recorded === derived.recorded && original.verifiedPurchases === derived.validCheckout.n
      && original.intentCorrectSuccess.numerator === derived.intent.n && original.intentCorrectSuccess.denominator === 60
      && original.syntheticConversion.numerator === derived.validCheckout.n && original.syntheticConversion.denominator === 60
      && original.prespecifiedFeasibleCheckout.denominator === 48
      && original.prespecifiedFeasibleCheckout.numerator === derived.feasibleCheckout.n
      && original.revenue.totalCents === derived.revenue.validTotalCents
      && original.allServerConfirmedOrders.evaluated === true && original.allServerConfirmedOrders.assigned === 60
      && original.allServerConfirmedOrders.count === derived.revenue.serverOrders
      && original.allServerConfirmedOrders.merchandiseSubtotalCents === derived.revenue.serverTotalCents, "Full metrics arm counts/revenue differ from derived rows");
  }
  const recorded = ARMS.reduce((total, arm) => total + full.arms[arm].recorded, 0);
  const health = object(object(object(object(plan.frozenRuntime, "frozen runtime").runtime, "runtime").model, "model").healthIdentity, "health identity");
  return { schema: "shopper-slide-metrics.v1", synthetic: true, humanConversionClaim: false,
    status: recorded === 120 ? "all-assigned-recorded" : "partial", assigned: 120, recorded,
    provenance: { metricsVersion: metrics.version, planVersion: plan.version, serverVersion: server.schema,
      catalogSnapshotSha256: digest(plan.catalogSnapshotSha256, "catalog digest"), runtimeSnapshotSha256: digest(object(plan.snapshot, "snapshot").sha256, "snapshot digest"),
      modelReceiptSha256: digest(object(plan.frozenRuntime.modelOwnerReceipt, "model owner receipt").sha256, "model receipt digest"),
      shopperModel: text(plan.model, "shopper model"), rankCheckpointSha256: digest(health["startup.rankCheckpointSha256"], "rank checkpoint"),
      rankExportSha256: digest(health["startup.rankSessionProvenance.rankHead.servingExportSha256"], "rank export"),
      copyCheckpointSha256: digest(health["startup.copyCandidate.checkpointSha256"], "copy checkpoint"),
      budget: { actions: integer(plan.budget.maxActions, "action budget"), seconds: integer(plan.budget.maxWallSeconds, "time budget") } },
    full, sensitivity: { pilot5: { caseIds: pilotIds, ...panel(pilot, false) },
      remaining55: { caseIds: remaining.map((p) => p.caseId), ...panel(remaining, false) } },
    definitions: { primary: "Intent-correct success includes verified valid checkout or justified abstention.",
      revenue: "USD merchandise cents per assigned session; nonpurchase slots contribute zero valid revenue. All-server revenue also retains UI-failed accepted orders.",
      preference: "Independent preference-match proxy on purchased products; not observed customer satisfaction.",
      effort: "Recorded fully timed steps and measured monotonic elapsed only; exclusions retained.",
      uncertainty: "Wilson95 full-panel rates; seeded20000 paired-case bootstrap, local minus generic. winsLossesTies=[local wins,generic wins,ties]. Small repeated-catalog panel, not IID customers or human lift.",
      sensitivity: "Prespecified first5 pilot versus remaining55; descriptive subsets, original case clusters, no controls pooled.",
      latency: "Browser after-paint opportunity; not physical paint, image transfer or native GPU latency." } };
}

async function json(file: string) {
  requireValue((await stat(file)).size <= 16 * 1024 * 1024, "JSON input exceeds 16 MiB");
  const bytes = await readFile(file); return { value: JSON.parse(bytes.toString("utf8")), sha256: sha(bytes), path: path.resolve(file) };
}
export async function slideReportMain(args: string[]) {
  requireValue(args.length === 8 && args[0] === "--metrics" && args[2] === "--plan" && args[4] === "--server-outcomes" && args[6] === "--out",
    "Usage: slide-report.mts --metrics FINAL_METRICS.json --plan COHORT_PLAN.json --server-outcomes FINAL_INDEX.json --out NEW_SLIDE_METRICS.json");
  const [metrics, plan, server] = await Promise.all([json(args[1]), json(args[3]), json(args[5])]), out = path.resolve(args[7]);
  requireValue(![metrics.path, plan.path, server.path].includes(out), "Output must differ from all source files");
  requireValue(metrics.value.serverIndexSource?.sha256 === server.sha256 && path.resolve(metrics.value.serverIndexSource.path) === server.path, "Metrics must bind exact server index bytes/path");
  const sources = list(metrics.value.sources, "metrics report sources").map((r) => object(r, "report source"));
  const indexedReports = list(server.value.reports, "server report sources");
  requireValue(sources.length === 6 && isDeepStrictEqual([...sources].sort((a, b) => a.path.localeCompare(b.path)), [...indexedReports].sort((a, b) => a.path.localeCompare(b.path))), "Exact six source report bindings required");
  for (const source of sources) {
    const report = await json(text(source.path, "source report path")); requireValue(report.sha256 === digest(source.sha256, "report digest"), "Source report hash differs");
    requireValue(report.value.catalogSnapshotSha256 === plan.value.catalogSnapshotSha256 && report.value.synthetic === true, "Source report catalog differs from frozen plan");
  }
  const snapshot = object(plan.value.snapshot, "snapshot"), frozenReceipt = object(plan.value.frozenRuntime.modelOwnerReceipt, "model receipt");
  requireValue((await json(text(snapshot.file, "snapshot file"))).sha256 === snapshot.sha256, "Frozen runtime snapshot bytes differ");
  requireValue((await json(path.resolve(WORKSPACE, text(frozenReceipt.path, "model receipt path")))).sha256 === frozenReceipt.sha256, "Model owner receipt bytes differ");
  const result = { ...buildSlideMetrics(metrics.value, plan.value, server.value), sources: {
    metrics: { path: metrics.path, sha256: metrics.sha256 }, plan: { path: plan.path, sha256: plan.sha256 },
    serverOutcomes: { path: server.path, sha256: server.sha256 }, reportSetSha256: sha(JSON.stringify(sources)) } };
  const bytes = `${JSON.stringify(result)}\n`; requireValue(Buffer.byteLength(bytes) <= 10 * 1024, "Slide JSON exceeds 10 KiB; never silently drop evidence fields");
  await writeFile(out, bytes, { flag: "wx" }); return result;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  slideReportMain(process.argv.slice(2)).then((result) => process.stdout.write(`${JSON.stringify({ schema: result.schema, status: result.status, assigned: result.assigned, recorded: result.recorded })}\n`))
    .catch((error: unknown) => { process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`); process.exitCode = 1; });
}
