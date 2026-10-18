/** Optional, preselected controls. Never imported by the active main cohort; no automatic invocation. */
import { createHash } from "node:crypto";
import { appendFile, lstat, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ChromeShopperBrowser } from "../../../../apps/runner/src/shopper-chrome.ts";
import { CodexShopperAgent, codexModelAvailability, findCodexBinary } from "../../../../apps/runner/src/shopper-codex-agent.ts";
import { inspectCohortAttempts, verifyCohortPins, verifyRuntimeIdentity, type CohortPlan } from "../../../../apps/runner/src/shopper-cohort.ts";
import { loadShopperPacket, type Packet } from "../../../../apps/runner/src/shopper-eval-cli.ts";
import { executeShopperSession } from "../../../../apps/runner/src/shopper-runner.ts";
import { validateShopperCatalog, validateShopperSession, verifyShopperArtifactFiles, type ShopperArm } from "../../../../apps/runner/src/shopper-contract.ts";
import { generateServerOutcomes, type ServerOutcomesIndex } from "../../../../apps/runner/src/shopper-server-outcomes.ts";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PRESELECTED = path.join(ROOT, "optional-controls-plan.json");
const PRESELECTED_SHA = "2851c95a7eb58b9f1991a859f49651c347d0b219024dceba4db105cfcb11b2f6";
const DEADLINE = "2026-10-18T13:00:00.000Z";
const sha = (bytes: Buffer | string) => createHash("sha256").update(bytes).digest("hex");
const message = (error: unknown) => error instanceof Error ? error.message : String(error);
const writeJson = (file: string, value: unknown) => writeFile(file, JSON.stringify(value, null, 2) + "\n", { flag: "wx" });
function requireValue(ok: unknown, error: string): asserts ok { if (!ok) throw new Error(error); }
async function absent(file: string) {
  try { await lstat(file); throw new Error(`Active main dispatcher exists: ${file}`); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
}
type ControlSlot = { kind: "aa" | "repeats"; caseId: string; arm: ShopperArm; label: string;
  runId: string; packetIndex: number; experimentId: string; path: string; clusterId: string };
type Obj = Record<string, any>;

/** Selection and ordering are frozen independently of the observed main outcomes. */
export function controlSchedule(main: CohortPlan): ControlSlot[] {
  requireValue(main.version === "shopper-fixed-cohort.v1" && main.fixedPairs === 60 && main.fixedSessions === 120
    && main.schedule.length === 120 && new Set(main.schedule.map((slot) => slot.runId)).size === 120, "Fixed 60-pair/120-session main plan required");
  requireValue(main.model === "gpt-6.1-sol" && main.settings.effort === "low" && main.settings.concurrency === 1
    && main.settings.quantity === 1 && main.settings.viewport.width === 1280 && main.settings.viewport.height === 960
    && main.budget.maxActions === 24 && main.budget.maxWallSeconds === 120 && main.baseUrl === "http://127.0.0.1:4312/", "Frozen Sol model, budget and local storefront settings required");
  const slots: ControlSlot[] = [];
  const packetIndex = (caseId: string) => {
    const pair = main.schedule.filter((slot) => slot.caseId === caseId);
    requireValue(pair.length === 2 && pair[0].packetIndex === pair[1].packetIndex
      && new Set(pair.map((slot) => slot.arm)).size === 2, `Preselected case missing from main roster: ${caseId}`);
    return pair[0].packetIndex;
  };
  for (const caseId of ["holdout-v2-02", "holdout-v3-06"]) for (const label of ["A", "B"]) slots.push({
    kind: "aa", caseId, arm: "generic", label, runId: `${caseId}-aa-${label}`, packetIndex: packetIndex(caseId),
    experimentId: "actual-store-aa-v2", path: `${caseId}/${label}/session.json`, clusterId: caseId,
  });
  for (const caseId of ["holdout-v2-01", "holdout-v3-03"]) {
    const pair = main.schedule.filter((slot) => slot.caseId === caseId).reverse();
    packetIndex(caseId);
    for (const slot of pair) slots.push({ kind: "repeats", caseId, arm: slot.arm, label: slot.arm,
      runId: `${caseId}-repeat-${slot.arm}`, packetIndex: slot.packetIndex, experimentId: "actual-store-repeat-v2",
      path: `${caseId}/${slot.arm}/session.json`, clusterId: caseId });
  }
  requireValue(slots.length === 8 && new Set(slots.map((slot) => slot.runId)).size === 8, "Exactly eight unique optional assignments required");
  return slots;
}

/** CPU-only admission check. No Chrome, Codex or network calls. */
export async function inspectControlsStart(mainDirectory: string, now = new Date()) {
  await absent(path.join(mainDirectory, "active-run-lock.json"));
  const [prescribedBytes, mainBytes, launchBytes] = await Promise.all([
    readFile(PRESELECTED), readFile(path.join(mainDirectory, "cohort-plan.json")), readFile(path.join(mainDirectory, "launch.json")),
  ]);
  requireValue(sha(prescribedBytes) === PRESELECTED_SHA, "Preselected optional-controls plan changed");
  const prescribed = JSON.parse(prescribedBytes.toString("utf8"));
  requireValue(prescribed.schema === "shopper-optional-controls-plan.v1" && prescribed.synthetic === true
    && prescribed.selectedBeforeTheseScenariosWereExecuted === true && path.basename(path.resolve(mainDirectory)) === prescribed.mainCohort,
    "Canonical main cohort and preselected controls plan required");
  const main = JSON.parse(mainBytes.toString("utf8")) as CohortPlan, launch = JSON.parse(launchBytes.toString("utf8"));
  requireValue(launch.schema === "shopper-cohort-launch.v1" && launch.planSha256 === sha(mainBytes), "Main plan differs from its launch receipt");
  requireValue(main.deadline === DEADLINE && Date.parse(DEADLINE) - now.getTime() >= 25 * 60_000,
    "Optional controls require at least 25 minutes before 2026-10-18T13:00:00Z");
  const schedule = controlSchedule(main);
  await verifyCohortPins(main);
  const completed = await inspectCohortAttempts(main, mainDirectory);
  requireValue(completed.size === 120, `All 120 main receipts required; observed ${completed.size}`);
  await absent(path.join(mainDirectory, "active-run-lock.json"));
  return { main, schedule, mainPlanSha256: sha(mainBytes), mainReceipts: Object.fromEntries(completed), preselectedPlanSha256: PRESELECTED_SHA };
}

/** Catalog/gold is read for scoring only after the isolated shopper has stopped. */
export async function scoreControl(slot: ControlSlot, packetDirectory: string, groupDirectory: string) {
  const row: Obj = { caseId: slot.caseId, arm: slot.arm, expectedRunId: slot.runId, path: slot.path, controlLabel: slot.label,
    clusterId: slot.clusterId, status: "unattempted", formalStatus: "unattempted", errors: [] };
  const directory = path.dirname(path.join(groupDirectory, slot.path));
  let input: Buffer;
  try { input = await readFile(path.join(directory, "session.json")); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    try { await lstat(path.join(directory, "attempt-lock.json")); row.status = "invalid-sidecar";
      row.errors.push("Attempt started without a valid final receipt; interrupted/unknown outcome retained without retry"); }
    catch (lockError) { if ((lockError as NodeJS.ErrnoException).code !== "ENOENT") throw lockError; }
    return row;
  }
  row.sidecarSha256 = sha(input);
  try {
    const packet = await loadShopperPacket(packetDirectory), c = packet.cases.find((value) => value.caseId === slot.caseId)!;
    const [bindingBytes, goldBytes, manifestBytes] = await Promise.all(["catalog-binding.json", "evaluator-gold.json", "manifest.json"]
      .map((name) => readFile(path.join(packetDirectory, name))));
    const manifest = JSON.parse(manifestBytes.toString("utf8"));
    requireValue(manifest.files["catalog-binding.json"] === sha(bindingBytes) && manifest.files["evaluator-gold.json"] === sha(goldBytes), "Control scoring bindings changed");
    const catalog = validateShopperCatalog(JSON.parse(bindingBytes.toString("utf8")).catalogSnapshot);
    const gold = (JSON.parse(goldBytes.toString("utf8")) as Obj[]).find((value) => value.caseId === slot.caseId)!;
    requireValue(c && gold && ["satisfiable", "known-empty", "unverifiable-required-fact"].includes(gold.eligibilityStatus), "Independent scenario and gold required");
    const session = validateShopperSession(JSON.parse(input.toString("utf8")), c, catalog);
    requireValue(session.arm === slot.arm && session.run.id === slot.runId && session.run.experimentId === slot.experimentId, "Control receipt identity mismatch");
    const artifacts = await verifyShopperArtifactFiles(session.artifacts, directory);
    const acceptable = !!session.choice && gold.acceptableProductIds.includes(session.choice.productId) && gold.acceptableVariantIds.includes(session.choice.variantId);
    const abstained = session.choice === null && !session.fakeOrderConfirmed && ["abandoned", "constraint-failure"].includes(session.status);
    const missing = ["screenshot", "trace"].filter((kind) => !session.artifacts.some((artifact) => artifact.artifact.kind === kind && artifact.status === "captured"));
    const budget = session.steps.length <= 24 && session.elapsedMs <= 120_000 && session.timingPrecision.sessionElapsed === "measured-monotonic" && session.startedAt !== null;
    Object.assign(row, { status: "recorded", formalStatus: session.protocol.passed && budget ? "recorded" : "unattempted",
      session, outcome: session.status, artifacts, evaluator: { acceptableSelection: acceptable,
        expectedOutcomeMet: gold.eligibilityStatus === "satisfiable" ? session.status === "completed" && acceptable : abstained,
        allHardConstraintsMet: session.verification.hardConstraints.every((check) => check.status === "met"), evidenceVerified: !missing.length && artifacts.every((check) => check.verified),
        budgetRespected: budget, missingEvidenceKinds: missing }, eligibilityStatus: gold.eligibilityStatus });
    row.errors.push(...artifacts.filter((artifact) => !artifact.verified).map((artifact) => `${artifact.artifactId}: ${artifact.error}`));
  } catch (error) { row.status = "invalid-sidecar"; row.errors.push(message(error)); }
  return row;
}

function credit(row: Obj, server: ServerOutcomesIndex["rows"][number]) {
  // scoreControl has independently validated catalog/price/choice/artifact bindings before this gate.
  const eligible = row.status === "recorded" && row.formalStatus === "recorded" && row.session?.protocol?.passed === true
    && row.evaluator?.evidenceVerified === true && row.evaluator?.budgetRespected === true;
  const purchase = !!(eligible && row.eligibilityStatus === "satisfiable" && row.outcome === "completed"
    && row.session?.status === "completed" && row.session?.fakeOrderConfirmed === true && server.serverConfirmed && server.visibleCheckoutConfirmed
    && row.session?.choice?.observationSource === "independent-dom" && row.session?.choice?.quantity === 1
    && row.evaluator.acceptableSelection === true && row.evaluator.expectedOutcomeMet === true && row.evaluator.allHardConstraintsMet === true);
  const abstention = !!(eligible && row.eligibilityStatus !== "satisfiable" && row.evaluator.expectedOutcomeMet && !server.serverConfirmed);
  return { intentCorrect: purchase || abstention, validPurchase: purchase, serverConfirmed: server.serverConfirmed,
    visibleCheckoutConfirmed: server.visibleCheckoutConfirmed,
    serverAcceptedMerchandiseRevenueCents: server.serverConfirmed ? Math.round(Number(server.receipt!.subtotal) * 100) : 0,
    validPurchaseRevenueCents: purchase ? Math.round(Number(server.receipt!.subtotal) * 100) : 0 };
}

export function summarizeControls(reports: Obj[], server: ServerOutcomesIndex) {
  return { schema: "shopper-optional-controls-summary.v1", synthetic: true, humanConversionClaim: false,
    independentFreshCohort: false, pooledWithMain: false,
    groups: reports.map((report) => {
      const rows = (report.rows as Obj[]).map((row) => {
        const confirmed = server.rows.find((item) => item.runId === row.expectedRunId);
        requireValue(confirmed, `Missing assigned server-outcome row: ${row.expectedRunId}`);
        return { runId: row.expectedRunId, caseId: row.caseId, clusterId: row.clusterId,
          label: row.controlLabel, arm: row.arm, status: row.status, outcome: row.outcome ?? null, ...credit(row, confirmed) };
      });
      const pairs = [...new Set(rows.map((row) => row.caseId))].map((caseId) => {
        const selected = rows.filter((row) => row.caseId === caseId);
        const first = selected.find((row) => row.label === (report.controlKind === "aa" ? "A" : "generic"))!;
        const second = selected.find((row) => row.label === (report.controlKind === "aa" ? "B" : "local-model"))!;
        return { caseId, clusterId: caseId, firstRunId: first.runId, secondRunId: second.runId,
          direction: report.controlKind === "aa" ? "B minus A (both generic)" : "local-model minus generic (repeat only)",
          intentCorrectDifference: Number(second.intentCorrect) - Number(first.intentCorrect),
          validPurchaseDifference: Number(second.validPurchase) - Number(first.validPurchase),
          validPurchaseRevenueCentsDifference: second.validPurchaseRevenueCents - first.validPurchaseRevenueCents };
      });
      const labels = report.controlKind === "aa" ? ["A", "B"] : ["generic", "local-model"];
      return { kind: report.controlKind, fixedAssignedSessions: 4, recorded: rows.filter((row) => row.status === "recorded").length,
        unattempted: rows.filter((row) => row.status === "unattempted").length, invalidOrInterrupted: rows.filter((row) => row.status === "invalid-sidecar").length,
        timeouts: rows.filter((row) => row.outcome === "timeout").length, technicalFailures: rows.filter((row) => row.outcome === "technical-failure").length,
        abstentions: rows.filter((row) => row.outcome === "abandoned").length, rows, pairs,
        byLabel: Object.fromEntries(labels.map((label) => {
          const selected = rows.filter((row) => row.label === label), revenue = selected.reduce((sum, row) => sum + row.validPurchaseRevenueCents, 0);
          const acceptedRevenue = selected.reduce((sum, row) => sum + row.serverAcceptedMerchandiseRevenueCents, 0);
          return [label, { assigned: 2, intentCorrect: selected.filter((row) => row.intentCorrect).length,
            validPurchases: selected.filter((row) => row.validPurchase).length, serverConfirmed: selected.filter((row) => row.serverConfirmed).length,
            visibleCheckoutConfirmed: selected.filter((row) => row.visibleCheckoutConfirmed).length,
            serverAcceptedMerchandiseRevenueCents: acceptedRevenue, serverAcceptedMerchandiseRevenueCentsPerAssignedSession: acceptedRevenue / 2,
            validPurchaseRevenueCents: revenue, validPurchaseRevenueCentsPerAssignedSession: revenue / 2 }];
        })),
        secondVersusFirst: { intentWins: pairs.filter((pair) => pair.intentCorrectDifference > 0).length,
          intentLosses: pairs.filter((pair) => pair.intentCorrectDifference < 0).length, intentTies: pairs.filter((pair) => pair.intentCorrectDifference === 0).length },
        caution: "Two scenario clusters only; descriptive sanity/repeatability evidence, no precise noise-rate or uplift claim. Repeats retain original clusters." };
    }) };
}

export async function runControls(mainDirectory: string, out: string) {
  const admitted = await inspectControlsStart(mainDirectory);
  await mkdir(out); // Exclusive: no resume, no rerun, no overwrite of a previous control attempt.
  // One permanent claim prevents rerunning controls merely by choosing a different output directory.
  await writeJson(path.join(mainDirectory, "optional-controls-claim.json"), { schema: "shopper-optional-controls-claim.v1", synthetic: true,
    claimedAt: new Date().toISOString(), pid: process.pid, output: path.resolve(out), planSha256: PRESELECTED_SHA, fixedAssignedSessions: 8 });
  await writeFile(path.join(out, ".gitignore"), "**/trace.zip\n**/observation-*.png\n**/browser.jsonl\n", { flag: "wx" });
  const selfSha = sha(await readFile(fileURLToPath(import.meta.url)));
  await writeJson(path.join(out, "execution-plan.json"), { schema: "shopper-optional-controls-execution.v1", synthetic: true, humanConversionClaim: false,
    mainCohort: path.resolve(mainDirectory), ...admitted, scriptSha256: selfSha, fixedAssignedSessions: 8, separateFromMain: true });
  const binary = await findCodexBinary(admitted.main.requestedCodexBinary ?? undefined);
  await writeJson(path.join(out, "shopper-availability.json"), { binary, model: admitted.main.model,
    availability: await codexModelAvailability(binary, admitted.main.model) });
  await inspectControlsStart(mainDirectory); // Admission can expire during adapter discovery.
  const packets: Packet[] = await Promise.all(admitted.main.packets.map((packet) => loadShopperPacket(packet.directory)));
  let stop: string | null = null, fatal: string | null = null;
  const excluded = new Map<string, string>();
  const sigint = () => { stop ??= "SIGINT"; }, sigterm = () => { stop ??= "SIGTERM"; };
  process.on("SIGINT", sigint); process.on("SIGTERM", sigterm);
  const log = (event: unknown) => appendFile(path.join(out, "dispatch.jsonl"), JSON.stringify(event) + "\n");
  try {
    for (const slot of admitted.schedule) {
      if (stop || Date.now() >= Date.parse(DEADLINE)) break;
      await absent(path.join(mainDirectory, "active-run-lock.json"));
      await verifyCohortPins(admitted.main);
      requireValue(sha(await readFile(path.join(mainDirectory, "cohort-plan.json"))) === admitted.mainPlanSha256, "Frozen main execution plan changed");
      requireValue(sha(await readFile(PRESELECTED)) === PRESELECTED_SHA && sha(await readFile(fileURLToPath(import.meta.url))) === selfSha, "Frozen control implementation/selection changed");
      await log({ type: "runtime-identity", ...await verifyRuntimeIdentity(admitted.main.frozenRuntime), runId: slot.runId });
      await absent(path.join(mainDirectory, "active-run-lock.json"));
      if (stop || Date.now() >= Date.parse(DEADLINE)) break;
      const original = packets[slot.packetIndex], scenario = original.cases.find((item) => item.caseId === slot.caseId)!;
      const cloned: Packet = { ...original, runIds: { ...original.runIds, [`${slot.caseId}/${slot.arm}`]: slot.runId }, experimentId: slot.experimentId };
      await log({ type: "start", at: new Date().toISOString(), ...slot });
      try {
        const result = await executeShopperSession(admitted.main.packets[slot.packetIndex].directory, cloned, scenario, slot.arm,
          path.dirname(path.join(out, slot.kind, slot.path)), { openBrowser: (url, directory) => ChromeShopperBrowser.open(url, directory),
            openAgent: (assignment, event) => new CodexShopperAgent(binary, admitted.main.model, assignment, event) }, admitted.main.baseUrl);
        await log({ type: "finish", at: new Date().toISOString(), runId: slot.runId, status: result.status, elapsedMs: result.elapsedMs, steps: result.steps.length });
        await verifyCohortPins(admitted.main);
        requireValue(sha(await readFile(fileURLToPath(import.meta.url))) === selfSha, "Frozen control implementation changed during session");
      } catch (error) { fatal = `Stopped at ${slot.runId}: ${message(error)}; evidence retained without retry`; excluded.set(slot.runId, fatal);
        await log({ type: "interrupted-or-invalid", runId: slot.runId, error: fatal }); break; }
    }
  } catch (error) { fatal = message(error); await log({ type: "dispatch-blocked", error: fatal }); }
  finally { process.off("SIGINT", sigint); process.off("SIGTERM", sigterm); }
  const reports: Obj[] = [], files: string[] = [];
  for (const kind of ["aa", "repeats"] as const) {
    const directory = path.join(out, kind); await mkdir(directory, { recursive: true });
    const rows = [];
    for (const slot of admitted.schedule.filter((item) => item.kind === kind)) {
      const row = await scoreControl(slot, admitted.main.packets[slot.packetIndex].directory, directory);
      if (excluded.has(slot.runId)) { row.formalStatus = "unattempted"; row.errors.push(excluded.get(slot.runId)); }
      rows.push(row);
    }
    const report = { version: "shopper-evidence-aggregate.v1", auxiliarySchema: "shopper-optional-controls-aggregate.v1", synthetic: true, humanConversionClaim: false,
      controlKind: kind, fixedSessions: 4, fixedCases: 2, independentFreshCohort: false, pooledWithMain: false,
      catalogSnapshotSha256: admitted.main.catalogSnapshotSha256, stopReason: fatal ?? stop ?? (Date.now() >= Date.parse(DEADLINE) ? "deadline" : "roster-finished"), rows };
    const file = path.join(directory, "aggregate.json"); await writeJson(file, report); reports.push(report); files.push(file);
  }
  const server = await generateServerOutcomes(files, path.join(out, "server-outcomes.json"));
  const summary = summarizeControls(reports, server); await writeJson(path.join(out, "controls-summary.json"), summary);
  requireValue(fatal === null, fatal ?? "Controls dispatch failed");
  return { output: out, fixedAssignedSessions: 8, recorded: reports.flatMap((report) => report.rows).filter((row) => row.status === "recorded").length };
}

export async function controlsMain(args: string[]) {
  requireValue(args.length === 4 && args[0] === "--main-cohort" && args[2] === "--out", "Use --main-cohort CANONICAL_MAIN_DIRECTORY --out NEW_CONTROLS_DIRECTORY");
  return runControls(path.resolve(args[1]), path.resolve(args[3]));
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) controlsMain(process.argv.slice(2))
  .then((result) => console.log(JSON.stringify(result))).catch((error) => { console.error(message(error)); process.exitCode = 1; });
