/** CPU-only shopper assignment preparation and evidence aggregation. Never dispatches a browser/model. */
import { createHash } from "node:crypto";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildShopperAssignment, SHOPPER_CONTRACT_VERSION, validatePairedShopperSuite,
  validateShopperCase, validateShopperCatalog, validateShopperSession, verifyShopperArtifactFiles,
  type CatalogSnapshot, type ShopperArm, type ShopperCase, type ShopperSession } from "./shopper-contract.js";

const WORKSPACE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const ARMS: ShopperArm[] = ["generic", "local-model"];
const INPUT_LIMIT = 4 * 1024 * 1024;
const sha = (bytes: Buffer | string) => createHash("sha256").update(bytes).digest("hex");
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);
const canonical = (value: unknown): string => {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}`;
  return JSON.stringify(value);
};
function requireValue(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }
function object(value: unknown): Record<string, unknown> {
  requireValue(value && typeof value === "object" && !Array.isArray(value), "Expected JSON object");
  return value as Record<string, unknown>;
}
async function readJson(file: string): Promise<{ value: unknown; sha256: string }> {
  requireValue((await stat(file)).size <= INPUT_LIMIT, "Shopper JSON input exceeds 4 MiB");
  const bytes = await readFile(file);
  return { value: JSON.parse(bytes.toString("utf8")), sha256: sha(bytes) };
}
async function writeJson(file: string, value: unknown) {
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" });
}
function safeId(value: unknown): asserts value is string {
  requireValue(typeof value === "string" && /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}$/.test(value), "Safe case/run ID required");
}

export type Packet = { cases: ShopperCase[]; runIds: Record<string, string>; defaultBaseUrl: string;
  budget: { maxActions: number; maxWallSeconds: number }; sources: Record<string, string>; experimentId?: string };

/** Deliberately reads only public context, private shopper instructions and schedule; never evaluator gold/catalog. */
export async function loadShopperPacket(directory: string): Promise<Packet> {
  const names = ["pair-plan.json", "public-observations.json", "shopper-instructions.json"];
  const inputs = await Promise.all(names.map((name) => readJson(path.join(directory, name))));
  const manifestInput = await readJson(path.join(directory, "manifest.json")), manifest = object(manifestInput.value);
  requireValue(manifest.version === "actual16-paired-shopper-packet.v1", "Frozen shopper packet manifest required");
  const fileHashes = object(manifest.files);
  names.forEach((name, index) => requireValue(fileHashes[name] === inputs[index].sha256, `Shopper packet file changed: ${name}`));
  const plan = object(inputs[0].value), schedule = plan.schedule;
  if (plan.experimentId !== undefined) safeId(plan.experimentId);
  requireValue(plan.contractVersion === SHOPPER_CONTRACT_VERSION && plan.initialPairs === 10 && plan.initialRuns === 20,
    "Fixed ten-pair/twenty-run shopper plan required");
  requireValue(Array.isArray(schedule) && schedule.length === 10 && Array.isArray(inputs[1].value) && Array.isArray(inputs[2].value), "Fixed ten-row packet arrays required");
  const publicRows = inputs[1].value as Record<string, unknown>[], privateRows = inputs[2].value as Record<string, unknown>[];
  requireValue(publicRows.length === 10 && privateRows.length === 10 && new Set(publicRows.map((r) => r.caseId)).size === 10
    && new Set(privateRows.map((r) => r.caseId)).size === 10, "Exactly ten unique public/private case rows required");
  const runIds: Record<string, string> = {};
  const cases = schedule.map((row: unknown) => {
    const item = object(row); safeId(item.caseId);
    const publicRow = publicRows.find((r) => r.caseId === item.caseId), privateRow = privateRows.find((r) => r.caseId === item.caseId);
    requireValue(publicRow && privateRow, "Schedule/public/private case membership differs");
    const c = validateShopperCase({ contractVersion: plan.contractVersion, caseId: item.caseId, split: item.split,
      seed: item.seed, armOrder: item.armOrder, privateShopperNeeds: privateRow.privateShopperNeeds,
      observableContext: publicRow.observableContext });
    requireValue(Array.isArray(item.runs) && item.runs.length === 2, "Two explicit run IDs required per pair");
    for (const [index, runValue] of item.runs.entries()) {
      const run = object(runValue); safeId(run.runId);
      requireValue(run.arm === c.armOrder[index] && run.initialStatus === "unrun", "Declared paired order and unrun initial status required");
      runIds[`${c.caseId}/${c.armOrder[index]}`] = run.runId;
    }
    return c;
  });
  requireValue(new Set(cases.map((c) => c.caseId)).size === 10 && new Set(Object.values(runIds)).size === 20, "Unique ten cases and twenty run IDs required");
  const budget = object(plan.budget);
  requireValue(budget.maxActions === 24 && budget.maxWallSeconds === 120, "Unchanged24-action/120-second shopper budget required");
  requireValue(typeof plan.defaultBaseUrl === "string", "Declared local base URL required");
  // The contract applies the local-origin allowlist and URL normalization.
  buildShopperAssignment(cases[0], "generic", plan.defaultBaseUrl);
  return { cases, runIds, defaultBaseUrl: plan.defaultBaseUrl, ...(plan.experimentId ? { experimentId: plan.experimentId as string } : {}),
    budget: { maxActions: 24, maxWallSeconds: 120 },
    sources: { ...Object.fromEntries(names.map((name, index) => [name, inputs[index].sha256])), "manifest.json": manifestInput.sha256 } };
}

export async function prepareShopperAssignments(packetDirectory: string, outputDirectory: string, baseUrl?: string) {
  const packet = await loadShopperPacket(packetDirectory), output = path.resolve(outputDirectory);
  // Exclusive creation preserves every old assignment and manual result.
  await mkdir(output);
  await mkdir(path.join(output, "private")); await mkdir(path.join(output, "public"));
  const assignments = [];
  for (const c of packet.cases) {
    for (const arm of c.armOrder) {
      const assignment = buildShopperAssignment(c, arm, baseUrl ?? packet.defaultBaseUrl);
      const privatePath = `private/${c.caseId}.${arm}.json`, publicPath = `public/${c.caseId}.${arm}.json`;
      await writeJson(path.join(output, privatePath), assignment);
      await writeJson(path.join(output, publicPath), { contractVersion: SHOPPER_CONTRACT_VERSION,
        caseId: c.caseId, arm, entryUrl: assignment.entryUrl, recommenderContext: assignment.recommenderContext });
      assignments.push({ caseId: c.caseId, arm, runId: packet.runIds[`${c.caseId}/${arm}`],
        privateAssignment: privatePath, publicContext: publicPath, status: "unattempted" });
    }
  }
  const manifest = { version: "shopper-assignment-preparation.v1", contractVersion: SHOPPER_CONTRACT_VERSION,
    fixedCases: 10, fixedSessions: 20, browserDispatched: false, modelCalled: false,
    evaluatorFilesRead: false, packetSources: packet.sources, budget: packet.budget, assignments,
    instructions: ["Dispatch each private assignment to a fresh independent authorized shopper agent using actual headed Chrome.",
      "Only the separate public context may enter recommender inputs. Never send evaluator gold/catalog bindings to the shopper.",
      "Keep declared pair order and observed isolation; a fresh tab shares browser storage unless actually isolated.",
      "Normalize only observed actions into session.json. Missing screenshot/trace receipts must remain missing; never invent a trace."] };
  await writeJson(path.join(output, "manifest.json"), manifest);
  return manifest;
}

type Gold = { caseId: string; acceptableProductIds: string[]; acceptableVariantIds: string[];
  eligibilityStatus: "satisfiable" | "known-empty" | "unverifiable-required-fact"; requiredOutcome: string };
type SessionRow = { caseId: string; arm: ShopperArm; expectedRunId: string; path: string;
  status: "unattempted" | "invalid-sidecar" | "recorded"; errors: string[];
  formalStatus: "unattempted" | "recorded";
  sidecarSha256?: string; outcome?: ShopperSession["status"]; session?: ShopperSession;
  artifacts?: Awaited<ReturnType<typeof verifyShopperArtifactFiles>>;
  evaluator?: { acceptableSelection: boolean; expectedOutcomeMet: boolean; allHardConstraintsMet: boolean;
    evidenceVerified: boolean; budgetRespected: boolean; missingEvidenceKinds: string[] } };

function checkedGold(value: unknown, cases: ShopperCase[], catalog: CatalogSnapshot): Gold[] {
  requireValue(Array.isArray(value) && value.length === 10, "Exactly ten evaluator gold rows required");
  const gold = value as Gold[];
  requireValue(new Set(gold.map((g) => g.caseId)).size === 10 && cases.every((c) => gold.some((g) => g.caseId === c.caseId)), "Gold/packet case membership differs");
  for (const row of gold) {
    requireValue(["satisfiable", "known-empty", "unverifiable-required-fact"].includes(row.eligibilityStatus), "Explicit gold eligibility required");
    requireValue(Array.isArray(row.acceptableProductIds) && Array.isArray(row.acceptableVariantIds), "Explicit product/variant evaluator sets required");
    requireValue(row.acceptableProductIds.every((id) => catalog.products.some((p) => p.productId === id)) &&
      row.acceptableVariantIds.every((id) => catalog.products.some((p) => row.acceptableProductIds.includes(p.productId) && p.variants.some((v) => v.variantId === id))),
      "Evaluator selectors differ from independently bound catalog");
    requireValue(row.eligibilityStatus === "satisfiable" ? row.acceptableProductIds.length > 0 && row.acceptableVariantIds.length > 0
      : row.acceptableProductIds.length === 0 && row.acceptableVariantIds.length === 0, "Gold eligibility contradicts acceptable selector sets");
  }
  return gold;
}

async function checkedCatalog(file: string): Promise<{ catalog: CatalogSnapshot; sha256: string }> {
  const input = await readJson(file), binding = object(input.value), catalog = validateShopperCatalog(binding.catalogSnapshot);
  requireValue(catalog.products.length === 16, "Independent actual16 catalog binding required");
  requireValue(sha(canonical({ products: catalog.products })) === catalog.sha256, "Catalog snapshot content digest differs");
  requireValue(binding.runtimePublicExportVerifiedCPU === true, "Catalog binding must record public export verification");
  const hashes = object(binding.liveInputSourceHashes);
  requireValue(Object.keys(hashes).length > 0, "Catalog source freshness hashes required");
  for (const [name, expected] of Object.entries(hashes)) {
    const source = path.resolve(WORKSPACE, name);
    requireValue(source.startsWith(WORKSPACE + path.sep) && typeof expected === "string", "Contained catalog source hash required");
    requireValue(sha(await readFile(source)) === expected, `Catalog source changed: ${name}`);
  }
  return { catalog, sha256: input.sha256 };
}

function armSummary(rows: SessionRow[], arm: ShopperArm) {
  const selected = rows.filter((row) => row.arm === arm), recorded = selected.filter((row) => row.status === "recorded");
  const formal = recorded.filter((row) => row.formalStatus === "recorded");
  const accepted = formal.filter((row) => row.evaluator?.expectedOutcomeMet && row.evaluator.evidenceVerified);
  const actionCounts = recorded.map((row) => row.session!.steps.length);
  const local = recorded.flatMap((row) => (row.session?.model.readiness ?? []).map((measure) => ({ measure,
    artifactVerified: row.artifacts?.some((a) => a.artifactId === measure.artifactId && a.verified) === true })));
  return { fixedDenominator: 10, recorded: recorded.length, unattempted: selected.filter((row) => row.status === "unattempted").length,
    formalRecorded: formal.length, formalUnattempted: 10 - formal.length,
    exploratoryRecorded: recorded.filter((row) => !row.session?.protocol.passed).length,
    budgetViolations: recorded.filter((row) => !row.evaluator?.budgetRespected).length,
    invalidSidecars: selected.filter((row) => row.status === "invalid-sidecar").length,
    completed: formal.filter((row) => row.outcome === "completed").length,
    reportedCompleted: recorded.filter((row) => row.outcome === "completed").length,
    verifiedDomCompleted: accepted.filter((row) => row.outcome === "completed" && row.session?.choice?.observationSource === "independent-dom").length,
    technicalFailures: recorded.filter((row) => row.outcome === "technical-failure" || row.outcome === "timeout").length,
    observedExpectedOutcome: recorded.filter((row) => row.evaluator?.expectedOutcomeMet).length,
    evidenceVerifiedExpectedOutcome: accepted.length, evidenceVerifiedExpectedOutcomeRateFixedDenominator: accepted.length / 10,
    attemptsElapsedMs: recorded.map((row) => row.session!.elapsedMs),
    attemptsActionCounts: actionCounts,
    actionCountSummary: { total: actionCounts.reduce((sum, count) => sum + count, 0),
      minimum: actionCounts.length ? Math.min(...actionCounts) : null,
      maximum: actionCounts.length ? Math.max(...actionCounts) : null,
      mean: actionCounts.length ? actionCounts.reduce((sum, count) => sum + count, 0) / actionCounts.length : null,
      scope: "Recorded normalized shopper steps, including navigation; unrecorded browser actions are not inferred" },
    attemptsPreferenceFit: recorded.map((row) => {
      const preferences = row.session!.verification.preferences;
      return { caseId: row.caseId, runId: row.expectedRunId,
        met: preferences.filter((check) => check.status === "met").length,
        known: preferences.filter((check) => check.status !== "unknown").length, total: preferences.length };
    }),
    attemptsTimingPrecision: recorded.map((row) => row.session!.timingPrecision),
    readiness: Object.fromEntries((["rank", "firstThree", "complete"] as const).map((stage) => [stage,
      { observedDurationsMs: local.filter((r) => r.measure.stage === stage).map((r) => r.measure.durationMs),
        verifiedArtifactDurationsMs: local.filter((r) => r.measure.stage === stage && r.artifactVerified).map((r) => r.measure.durationMs),
        scope: "browser after-paint opportunity; excludes physical paint and image transfer" }])),
    fallbackUsed: recorded.filter((row) => row.session?.fallback.used).length };
}

/** Only consumes actual normalized sidecars. Missing/manual raw logs never become invented browser evidence. */
export async function aggregateShopperResults(packetDirectory: string, sessionsDirectory: string, outputFile: string,
  goldFile = path.join(packetDirectory, "evaluator-gold.json"), catalogFile = path.join(packetDirectory, "catalog-binding.json")) {
  const packet = await loadShopperPacket(packetDirectory);
  const rows: SessionRow[] = packet.cases.flatMap((c) => c.armOrder.map((arm) => ({ caseId: c.caseId, arm,
    expectedRunId: packet.runIds[`${c.caseId}/${arm}`], path: `${c.caseId}/${arm}/session.json`,
    status: "unattempted", formalStatus: "unattempted", errors: [] })));
  const report: { version: string; status: string; fixedCases: number; fixedSessions: number; packetSources: Record<string, string>;
    setupErrors: string[]; rows: SessionRow[]; [key: string]: unknown } = {
    version: "shopper-evidence-aggregate.v1", synthetic: true, status: "partial", fixedCases: 10, fixedSessions: 20,
    packetSources: packet.sources, setupErrors: [], rows, browserDispatched: false, modelCalled: false,
    humanConversionClaim: false, unseenGeneralizationClaim: false, physicalPaintClaim: false,
    limitations: ["Synthetic development smoke with fixed20 denominator; sidecar validation cannot reconstruct unrecorded actions.",
      "Contract/catalog checks and observed expected outcome are separate from verified evidence. Missing trace is retained explicitly.",
      "Evidence file hashes establish saved bytes; they do not establish that an unverified browser claim happened."] };
  try {
    const [{ catalog, sha256 }, goldInput] = await Promise.all([checkedCatalog(catalogFile), readJson(goldFile)]);
    const manifest = object((await readJson(path.join(packetDirectory, "manifest.json"))).value), fileHashes = object(manifest.files);
    requireValue(fileHashes["evaluator-gold.json"] === goldInput.sha256 && fileHashes["catalog-binding.json"] === sha256,
      "Evaluator gold/catalog bytes differ from frozen packet manifest");
    const gold = checkedGold(goldInput.value, packet.cases, catalog);
    report.catalogBindingSha256 = sha256; report.catalogSnapshotSha256 = catalog.sha256; report.evaluatorGoldSha256 = goldInput.sha256;
    for (const row of rows) {
      const file = path.resolve(sessionsDirectory, row.path);
      try {
        const input = await readJson(file); row.sidecarSha256 = input.sha256;
        const c = packet.cases.find((item) => item.caseId === row.caseId)!;
        const session = validateShopperSession(input.value, c, catalog);
        requireValue(session.arm === row.arm && session.run.id === row.expectedRunId, "Sidecar run/arm differs from declared roster slot");
        const artifactChecks = await verifyShopperArtifactFiles(session.artifacts, path.dirname(file));
        const evaluator = gold.find((g) => g.caseId === c.caseId)!;
        const acceptable = !!session.choice && evaluator.acceptableProductIds.includes(session.choice.productId)
          && evaluator.acceptableVariantIds.includes(session.choice.variantId);
        const abstained = session.choice === null && !session.fakeOrderConfirmed
          && (session.status === "abandoned" || session.status === "constraint-failure");
        const expectedOutcomeMet = evaluator.eligibilityStatus === "satisfiable" ? session.status === "completed" && acceptable : abstained;
        const missingEvidenceKinds = ["screenshot", "trace"].filter((kind) => !session.artifacts.some((a) => a.artifact.kind === kind && a.status === "captured"));
        const evidenceVerified = missingEvidenceKinds.length === 0 && artifactChecks.every((a) => a.verified);
        const budgetRespected = session.steps.length <= packet.budget.maxActions && session.elapsedMs <= packet.budget.maxWallSeconds * 1000;
        row.status = "recorded"; row.formalStatus = session.protocol.passed && budgetRespected ? "recorded" : "unattempted";
        row.session = session; row.outcome = session.status; row.artifacts = artifactChecks;
        row.evaluator = { acceptableSelection: acceptable, expectedOutcomeMet,
          allHardConstraintsMet: session.verification.hardConstraints.every((check) => check.status === "met"),
          evidenceVerified, missingEvidenceKinds,
          budgetRespected };
        if (!budgetRespected) row.errors.push(`Shopper budget exceeded: ${session.steps.length}/${packet.budget.maxActions} actions; ${session.elapsedMs}/${packet.budget.maxWallSeconds * 1000} ms`);
        row.errors.push(...artifactChecks.filter((a) => !a.verified).map((a) => `${a.artifactId}: ${a.error}`));
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ENOENT" && row.sidecarSha256 === undefined) continue;
        row.status = "invalid-sidecar"; row.errors.push(errorText(error));
      }
    }
    const sessions = rows.filter((row) => row.status === "recorded" && row.formalStatus === "recorded").map((row) => row.session!);
    // The strict paired suite is meaningful only once all20 actual receipts exist.
    if (sessions.length === 20) {
      try { report.completePairedContract = validatePairedShopperSuite(packet.cases, sessions, catalog); }
      catch (error) { report.setupErrors.push(`Paired suite: ${errorText(error)}`); }
    }
    report.status = report.setupErrors.length ? "blocked" : sessions.length !== 20 ? "partial" : "complete-recorded-suite";
  } catch (error) {
    report.status = "blocked"; report.setupErrors.push(errorText(error));
  }
  report.arms = Object.fromEntries(ARMS.map((arm) => [arm, armSummary(rows, arm)]));
  await writeJson(outputFile, report);
  return report;
}

export async function shopperEvalMain(args: string[]) {
  const [command, ...rest] = args;
  requireValue(command === "prepare" || command === "aggregate", "Use prepare or aggregate; no browser/model dispatch command exists");
  const allowed = command === "prepare" ? ["packet", "out", "base-url"] : ["packet", "out", "sessions", "gold", "catalog"];
  const options: Record<string, string> = {};
  for (let i = 0; i < rest.length; i += 2) {
    const flag = rest[i], value = rest[i + 1];
    requireValue(flag.startsWith("--") && allowed.includes(flag.slice(2)) && value && !value.startsWith("--") && !options[flag.slice(2)], "Unknown, repeated or missing CLI option");
    options[flag.slice(2)] = value;
  }
  requireValue(options.packet && options.out, "--packet DIRECTORY and --out NEW_PATH required");
  if (command === "prepare") return prepareShopperAssignments(options.packet, options.out, options["base-url"]);
  requireValue(options.sessions, "aggregate requires --sessions DIRECTORY");
  return aggregateShopperResults(options.packet, options.sessions, options.out, options.gold, options.catalog);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  shopperEvalMain(process.argv.slice(2)).then((result) => {
    process.stdout.write(`${JSON.stringify({ command: process.argv[2], output: result })}\n`);
    if ("status" in result && result.status === "blocked") process.exitCode = 1;
  }).catch((error) => { process.stderr.write(`${errorText(error)}\n`); process.exitCode = 1; });
}
