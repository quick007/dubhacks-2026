/** Result-only audit. Never launches a shopper, browser, model, or evaluation rerun. */
import { createHash } from "node:crypto";
import { access, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadShopperPacket, type aggregateShopperResults } from "../../../../apps/runner/src/shopper-eval-cli.ts";
import { validateShopperCatalog, validateShopperSession, verifyShopperArtifactFiles,
  type ShopperArm, type ShopperSession } from "../../../../apps/runner/src/shopper-contract.ts";

const DEFAULT_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const WORKSPACE = path.resolve(DEFAULT_DIRECTORY, "../../../..");
const PACKET_DIRECTORY = path.resolve(DEFAULT_DIRECTORY, "../store_shopper_eval_v1");
const ARMS: ShopperArm[] = ["generic", "local-model"];
const sha = (bytes: Buffer | string) => createHash("sha256").update(bytes).digest("hex");
const encode = (value: unknown) => JSON.stringify(value, null, 2) + "\n";
function requireValue(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }
type Aggregate = Awaited<ReturnType<typeof aggregateShopperResults>>;
type Row = Aggregate["rows"][number];
type Gold = { caseId: string; eligibilityStatus: string; acceptableProductIds: string[]; acceptableVariantIds: string[] };
type Plan = { schedule: { caseId: string; arm: ShopperArm; runId: string }[];
  implementationSources: Record<string, string>; packetSources: Record<string, string>;
  budget: { maxActions: number; maxWallSeconds: number } };
const unknownRuntime = (session: ShopperSession) => session.startedAt === null || session.timingPrecision.sessionElapsed === "estimated";
function distribution(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  const total = values.reduce((sum, value) => sum + value, 0);
  return { count: values.length, total, minimum: sorted[0] ?? null, maximum: sorted.at(-1) ?? null,
    mean: values.length ? total / values.length : null,
    median: sorted.length ? sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2 : null };
}

/** Exported for CPU-only temporary-fixture tests. The CLI uses the saved suite directory. */
export async function auditFinal(directory = DEFAULT_DIRECTORY, workspace = WORKSPACE, packetDirectory = PACKET_DIRECTORY) {
  // Read the final file first. An active/partial suite never creates audit output.
  const aggregateFile = path.join(directory, "aggregate.json");
  const aggregateBytes = await readFile(aggregateFile);
  const original = JSON.parse(aggregateBytes.toString("utf8")) as Aggregate;
  const planBytes = await readFile(path.join(directory, "execution-plan.json"));
  const plan = JSON.parse(planBytes.toString("utf8")) as Plan;
  const packet = await loadShopperPacket(packetDirectory);
  requireValue(original.fixedCases === 10 && original.fixedSessions === 20 && original.rows.length === 20,
    "Final aggregate must preserve the fixed ten-pair/twenty-session roster");
  requireValue(Array.isArray(original.setupErrors) && original.setupErrors.length === 0, "Final aggregate has unresolved setup errors");
  requireValue(plan.schedule.length === 20 && new Set(plan.schedule.map((item) => item.runId)).size === 20,
    "Execution plan must contain twenty unique scheduled runs");
  requireValue(plan.budget.maxActions === packet.budget.maxActions && plan.budget.maxWallSeconds === packet.budget.maxWallSeconds,
    "Execution and packet budgets differ");
  for (const [name, expected] of Object.entries(packet.sources)) {
    requireValue(plan.packetSources[name] === expected && original.packetSources[name] === expected, `Packet provenance differs: ${name}`);
  }
  const requiredCore = ["shopper-runner.ts", "shopper-agent.ts", "shopper-codex-agent.ts", "shopper-chrome.ts", "shopper-contract.ts", "shopper-eval-cli.ts"];
  requireValue(requiredCore.every((name) => typeof plan.implementationSources[name] === "string"), "Core implementation hashes are incomplete");
  const coreSources = [];
  for (const [name, expected] of Object.entries(plan.implementationSources)) {
    requireValue(path.basename(name) === name && /^[a-f0-9]{64}$/.test(expected), "Unsafe core source binding");
    const actual = sha(await readFile(path.join(workspace, "apps/runner/src", name)));
    requireValue(actual === expected, `Core implementation changed: ${name}`);
    coreSources.push({ name, sha256: actual, verified: true });
  }
  const manifest = JSON.parse(await readFile(path.join(packetDirectory, "manifest.json"), "utf8"));
  const catalogBytes = await readFile(path.join(packetDirectory, "catalog-binding.json"));
  const goldBytes = await readFile(path.join(packetDirectory, "evaluator-gold.json"));
  requireValue(sha(catalogBytes) === manifest.files["catalog-binding.json"] && sha(catalogBytes) === original.catalogBindingSha256,
    "Independent catalog provenance differs");
  requireValue(sha(goldBytes) === manifest.files["evaluator-gold.json"] && sha(goldBytes) === original.evaluatorGoldSha256,
    "Independent evaluator provenance differs");
  const catalog = validateShopperCatalog(JSON.parse(catalogBytes.toString("utf8")).catalogSnapshot);
  const gold = JSON.parse(goldBytes.toString("utf8")) as Gold[];
  requireValue(gold.length === 10 && new Set(gold.map((item) => item.caseId)).size === 10, "Independent evaluator roster differs");
  const audited = structuredClone(original);
  const sessions: { row: Row; session: ShopperSession; primary: boolean; checks: Awaited<ReturnType<typeof verifyShopperArtifactFiles>> }[] = [];
  const corrections: { runId: string; field: string; before: boolean; after: false; reason: string }[] = [];
  const seen = new Set<string>();
  const tabIds = new Set<string>(), contextIds = new Set<string>();
  for (const row of audited.rows) {
    const slot = plan.schedule.find((item) => item.runId === row.expectedRunId);
    requireValue(slot && slot.caseId === row.caseId && slot.arm === row.arm && !seen.has(slot.runId), "Duplicate or mismatched final roster slot");
    seen.add(slot.runId);
    const c = packet.cases.find((item) => item.caseId === row.caseId);
    requireValue(c && packet.runIds[`${row.caseId}/${row.arm}`] === slot.runId, "Final roster differs from frozen packet");
    requireValue(row.status === "recorded" && row.session && row.evaluator, `Missing or invalid actual receipt: ${slot.runId}`);
    requireValue(row.path === `${row.caseId}/${row.arm}/session.json`, "Unexpected session path");
    const sessionDirectory = path.join(directory, row.caseId, row.arm);
    const sessionBytes = await readFile(path.join(sessionDirectory, "session.json"));
    requireValue(sha(sessionBytes) === row.sidecarSha256, `Actual sidecar bytes changed: ${slot.runId}`);
    const session = validateShopperSession(JSON.parse(sessionBytes.toString("utf8")), c, catalog);
    requireValue(JSON.stringify(session) === JSON.stringify(row.session) && session.run.id === slot.runId && session.arm === row.arm,
      `Embedded and saved session differ: ${slot.runId}`);
    if (session.browser.actualNavigation) {
      requireValue(session.isolation.tabId && !tabIds.has(session.isolation.tabId), "Reused browser tab in actual receipts");
      tabIds.add(session.isolation.tabId);
      if (session.isolation.kind === "fresh-storage-context") {
        requireValue(session.isolation.contextId && !contextIds.has(session.isolation.contextId), "Reused or unknown isolated browser context");
        contextIds.add(session.isolation.contextId);
      }
    }
    const checks = await verifyShopperArtifactFiles(session.artifacts, sessionDirectory);
    session.artifacts.forEach((artifact, index) => {
      requireValue(artifact.status !== "captured" || checks[index].verified, `Captured artifact failed verification: ${artifact.artifact.id}`);
    });
    const evidenceVerified = ["screenshot", "trace"].every((kind) => session.artifacts.some((item) => item.artifact.kind === kind && item.status === "captured"))
      && checks.every((check) => check.verified);
    requireValue(row.evaluator.evidenceVerified === evidenceVerified, `Aggregate evidence flag differs: ${slot.runId}`);
    const g = gold.find((item) => item.caseId === c.caseId);
    requireValue(g, "Missing independent evaluator case");
    const acceptable = !!session.choice && g.acceptableProductIds.includes(session.choice.productId) && g.acceptableVariantIds.includes(session.choice.variantId);
    const abstained = !session.choice && !session.fakeOrderConfirmed && ["abandoned", "constraint-failure"].includes(session.status);
    const primary = g.eligibilityStatus === "satisfiable" ? session.status === "completed" && acceptable : abstained;
    requireValue(row.evaluator.acceptableSelection === acceptable && row.evaluator.expectedOutcomeMet === primary,
      `Independent expected outcome differs: ${slot.runId}`);
    if (unknownRuntime(session)) {
      requireValue(!session.protocol.passed && row.formalStatus === "unattempted", "Unknown runtime was improperly eligible for formal scoring");
      const reason = "Budget compliance unverified: estimated timing or missing session start cannot establish total runtime; recorded elapsed span is not a comparable latency sample";
      corrections.push({ runId: slot.runId, field: "evaluator.budgetRespected", before: row.evaluator.budgetRespected, after: false, reason });
      row.evaluator.budgetRespected = false;
      row.errors.push(reason);
    }
    sessions.push({ row, session, primary, checks });
  }
  requireValue(seen.size === 20 && ARMS.every((arm) => sessions.filter((item) => item.row.arm === arm).length === 10), "Final denominator changed");
  const originalArms = original.arms as Record<ShopperArm, Record<string, unknown>>;
  const auditedArms = audited.arms as Record<ShopperArm, Record<string, unknown>>;
  const summaryArms = Object.fromEntries(ARMS.map((arm) => {
    const rows = sessions.filter((item) => item.row.arm === arm);
    const unverified = rows.filter((item) => unknownRuntime(item.session));
    const violations = rows.filter(({ session }) => session.steps.length > packet.budget.maxActions
      || !unknownRuntime(session) && session.elapsedMs > packet.budget.maxWallSeconds * 1000).length;
    requireValue(originalArms[arm].fixedDenominator === 10 && originalArms[arm].budgetViolations === violations,
      `Original actual budget violation count differs: ${arm}`);
    auditedArms[arm].budgetUnverified = unverified.length;
    // Keep actual violations distinct from conservative false flags for unproven compliance.
    auditedArms[arm].budgetViolations = violations;
    const accepted = rows.filter(({ row, primary }) => row.formalStatus === "recorded" && primary && row.evaluator!.evidenceVerified && row.evaluator!.budgetRespected);
    const formal = rows.filter(({ row }) => row.formalStatus === "recorded");
    requireValue(originalArms[arm].recorded === rows.length && originalArms[arm].formalRecorded === formal.length
      && originalArms[arm].formalUnattempted === 10 - formal.length
      && originalArms[arm].completed === formal.filter(({ session }) => session.status === "completed").length
      && originalArms[arm].reportedCompleted === rows.filter(({ session }) => session.status === "completed").length
      && originalArms[arm].observedExpectedOutcome === rows.filter((item) => item.primary).length
      && originalArms[arm].verifiedDomCompleted === accepted.filter(({ session }) => session.status === "completed"
        && session.choice?.observationSource === "independent-dom").length,
      `Original formal or reported counts differ from actual receipts: ${arm}`);
    requireValue(originalArms[arm].evidenceVerifiedExpectedOutcome === accepted.length
      && originalArms[arm].evidenceVerifiedExpectedOutcomeRateFixedDenominator === accepted.length / 10,
      `Primary success count changed: ${arm}`);
    const measured = rows.filter(({ session }) => !unknownRuntime(session) && session.timingPrecision.sessionElapsed === "measured-monotonic");
    const preferenceFit = rows.map(({ session }) => ({ runId: session.run.id,
      met: session.verification.preferences.filter((check) => check.status === "met").length,
      known: session.verification.preferences.filter((check) => check.status !== "unknown").length,
      total: session.verification.preferences.length }));
    return [arm, { fixedDenominator: 10, recorded: rows.length,
      primaryExpectedOutcome: { observed: rows.filter((item) => item.primary).length, verifiedFormal: accepted.length,
        verifiedFormalRate: accepted.length / 10, definition: "Independent acceptable product/variant plus simulated checkout for satisfiable cases; correct abstention for known-empty or unverifiable cases" },
      checkout: { fakeOrdersConfirmed: rows.filter(({ session }) => session.fakeOrderConfirmed).length,
        reportedCompleted: rows.filter(({ session }) => session.status === "completed").length,
        formalCompleted: originalArms[arm].completed },
      outcomes: Object.fromEntries((["completed", "abandoned", "constraint-failure", "technical-failure", "timeout"] as const)
        .map((status) => [status, rows.filter(({ session }) => session.status === status).length])),
      expectedOutcomeNotMet: rows.filter((item) => !item.primary).length,
      abstentions: rows.filter(({ session }) => !session.choice && !session.fakeOrderConfirmed && ["abandoned", "constraint-failure"].includes(session.status)).length,
      budgetViolations: violations, budgetUnverified: unverified.length,
      preferences: { met: preferenceFit.reduce((sum, item) => sum + item.met, 0), known: preferenceFit.reduce((sum, item) => sum + item.known, 0),
        total: preferenceFit.reduce((sum, item) => sum + item.total, 0), perRun: preferenceFit },
      steps: { ...distribution(rows.map(({ session }) => session.steps.length)), scope: "Recorded normalized shopper steps including navigation; partial receipts do not imply unobserved actions" },
      measuredTotalElapsedMs: { ...distribution(measured.map(({ session }) => session.elapsedMs)),
        perRun: measured.map(({ session }) => ({ runId: session.run.id, elapsedMs: session.elapsedMs })),
        excludedRuns: rows.filter((item) => !measured.includes(item)).map(({ session }) => session.run.id),
        scope: "Measured monotonic total attempts only; excludes estimates, partial spans, missing starts, upper bounds, cleanup and scoring" },
      readiness: Object.fromEntries((["rank", "firstThree", "complete"] as const).map((stage) => {
        const samples = rows.flatMap(({ session, checks }) => session.model.readiness.filter((measure) => measure.stage === stage)
          .map((measure) => ({ runId: session.run.id, requestId: measure.requestId, durationMs: measure.durationMs,
            acceptedFields: measure.acceptedFields, withheldFields: measure.withheldFields, requiredFields: measure.requiredFields,
            artifactVerified: checks.some((check) => check.artifactId === measure.artifactId && check.verified) })));
        return [stage, { ...distribution(samples.filter((sample) => sample.artifactVerified).map((sample) => sample.durationMs)), samples,
          scope: "Browser-native after-paint opportunity entries; excludes physical paint, image transfer and isolated GPU benchmarks" }];
      })),
    }];
  }));
  const provenance = { originalAggregateSha256: sha(aggregateBytes), executionPlanSha256: sha(planBytes),
    auditUtilitySha256: sha(await readFile(fileURLToPath(import.meta.url))), coreSources, corrections,
    fullRosterPreserved: true, primaryAndFormalCountsUnchanged: true, rawElapsedValuesAndPrecisionPreserved: true };
  audited.resultAudit = provenance;
  const auditedText = encode(audited);
  const independentAudit = { version: "shopper-independent-result-audit.v1", synthetic: true, humanConversionClaim: false,
    ...provenance, auditedAggregateSha256: sha(auditedText), artifactReceipts: sessions.map(({ session, checks }) => ({ runId: session.run.id, checks })),
    missingArtifacts: sessions.flatMap(({ session }) => session.artifacts.filter((item) => item.status !== "captured")
      .map((item) => ({ runId: session.run.id, artifactId: item.artifact.id, status: item.status, reason: item.error }))) };
  const summary = { version: "shopper-synthetic-suite-summary.v1", synthetic: true, humanConversionClaim: false, humanLiftClaim: false,
    fixedCases: 10, fixedSessions: 20, originalAggregateSha256: sha(aggregateBytes), auditedAggregateSha256: sha(auditedText),
    arms: summaryArms, limitations: ["Synthetic shopper-agent development evidence, not human conversion lift or unseen generalization",
      "A verified saved artifact establishes bytes, not independent proof of every browser claim", "Unknown-runtime attempts remain in the fixed denominator and outside comparable total-latency statistics"] };
  const outputs = [["aggregate-audited.json", auditedText], ["independent-audit.json", encode(independentAudit)], ["summary.json", encode(summary)]] as const;
  for (const [name] of outputs) {
    try { await access(path.join(directory, name)); throw new Error(`Audit output already exists: ${name}`); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
  }
  requireValue(sha(await readFile(aggregateFile)) === sha(aggregateBytes), "Final aggregate changed during audit");
  for (const [name, content] of outputs) await writeFile(path.join(directory, name), content, { flag: "wx" });
  return { synthetic: true, fullRosterPreserved: true, primaryAndFormalCountsUnchanged: true,
    budgetUnverified: corrections.length, outputs: outputs.map(([name]) => path.join(directory, name)) };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 2) throw new Error("Run audit-final.mts without arguments after the final aggregate.json exists");
  auditFinal().then((result) => process.stdout.write(encode(result)))
    .catch((error) => { process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`); process.exitCode = 1; });
}
