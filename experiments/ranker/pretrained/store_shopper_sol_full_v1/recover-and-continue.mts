/** Recovery after the original process disappeared. Retain its actual attempts; never rerun them. */
import { createHash } from "node:crypto";
import { appendFile, copyFile, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { executeShopperSession, captureArtifact } from "../../../../apps/runner/src/shopper-runner.ts";
import { loadShopperPacket, aggregateShopperResults } from "../../../../apps/runner/src/shopper-eval-cli.ts";
import { ChromeShopperBrowser } from "../../../../apps/runner/src/shopper-chrome.ts";
import { CodexShopperAgent, codexModelAvailability } from "../../../../apps/runner/src/shopper-codex-agent.ts";
import { SHOPPER_CONTRACT_VERSION, validateShopperCatalog, validateShopperSession, verifyShopperNeeds,
  type ShopperSession } from "../../../../apps/runner/src/shopper-contract.ts";

const root = path.dirname(fileURLToPath(import.meta.url));
const workspace = path.resolve(root, "../../../..");
const packetDirectory = path.resolve(root, "../store_shopper_eval_v1");
const sha = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
const save = (file: string, value: unknown) => writeFile(file, JSON.stringify(value, null, 2) + "\n", { flag: "wx" });
const emit = async (event: unknown) => { const line = JSON.stringify({ at: new Date().toISOString(), event });
  await appendFile(path.join(root, "dispatch.jsonl"), line + "\n"); process.stdout.write(line + "\n"); };

const plan = JSON.parse(await readFile(path.join(root, "execution-plan.json"), "utf8"));
for (const [name, expected] of Object.entries(plan.implementationSources)) {
  if (sha(await readFile(path.join(workspace, "apps/runner/src", name))) !== expected) throw new Error(`Committed implementation changed: ${name}`);
}
const packet = await loadShopperPacket(packetDirectory);
const originalEvents = (await readFile(path.join(root, "dispatch.jsonl"), "utf8")).trim().split("\n").map((line) => JSON.parse(line));
const started = originalEvents.filter((row) => row.event.type === "start").map((row) => row.event.runId);
if (JSON.stringify(started) !== JSON.stringify(plan.schedule.slice(2, 4).map((item: {runId:string}) => item.runId))) throw new Error("Unexpected prior dispatch state; do not resume blindly");
const interrupted = plan.schedule[3];
if (interrupted.caseId !== "shopper-pair-04" || interrupted.arm !== "local-model") throw new Error("Unexpected interrupted slot");
await save(path.join(root, "recovery-lock.json"), { at: new Date().toISOString(), originalProcessSession: 81789,
  processCheck: "Elevated ps found no continuation, Codex shopper or automated Chrome process after reconnect; original exec session unavailable",
  policy: "Keep prior completed and failed attempts. Normalize interrupted browser evidence as exploratory technical failure; run only sixteen not-started slots." });

const directory = path.join(root, interrupted.caseId, interrupted.arm);
const rawBytes = await readFile(path.join(directory, "browser.jsonl"));
const browserEvents = rawBytes.toString("utf8").trim().split("\n").map((line) => JSON.parse(line));
const opened = browserEvents.find((row) => row.event.type === "opened");
const observation = browserEvents.find((row) => row.event.type === "observation");
if (!opened || !observation || observation.event.cartEmpty !== true || browserEvents.some((row) => row.event.type === "action")) throw new Error("Interrupted evidence differs; no inferred shopper actions allowed");
await copyFile(path.join(directory, "observation-1.png"), path.join(directory, "recovered-observation.png"));
await save(path.join(directory, "recovery-evidence.json"), { synthetic: true, recoveredAt: new Date().toISOString(),
  opened, observation, originalBrowserLog: { path: "browser.jsonl", bytes: rawBytes.length, sha256: sha(rawBytes) },
  missing: ["Agent decision/action result", "termination clock", "final screenshot", "stopped Playwright trace"],
  outcome: "Execution process no longer exists after Mac/exec-server disconnection. No observed shopper action beyond navigation; total runtime unknown." });
const runId = interrupted.runId;
const artifacts = await Promise.all([
  captureArtifact(directory, runId, "recovered-observation.png", "screenshot", "image/png"),
  captureArtifact(directory, runId, "trace.zip", "trace", "application/zip"),
  captureArtifact(directory, runId, "recovery-evidence.json", "result", "application/json"),
]);
const evidence = artifacts.filter((item) => item.status === "captured").map((item) => item.artifact.id);
const catalog = validateShopperCatalog(JSON.parse(await readFile(path.join(packetDirectory, "catalog-binding.json"), "utf8")).catalogSnapshot);
const c = packet.cases.find((item) => item.caseId === interrupted.caseId)!;
const checks = (needs: typeof c.privateShopperNeeds.hardConstraints) => verifyShopperNeeds(needs, null, catalog).map((item) => ({ ...item, evidenceArtifactIds: evidence }));
const session: ShopperSession = {
  contractVersion: SHOPPER_CONTRACT_VERSION, caseId: c.caseId, arm: interrupted.arm,
  run: { id: runId, experimentId: "actual-store-isolated-shopper-v1", personaId: c.caseId, personaSplit: c.split, seed: c.seed },
  browser: { name: "Chrome", headed: true, actualNavigation: true, version: opened.event.browserVersion, automation: "playwright-chrome" },
  isolation: { kind: "fresh-storage-context", tabId: opened.event.tabId, contextId: opened.event.contextId,
    sharedStorage: false, storageCleared: true, initialCartEmpty: true, evidenceArtifactIds: evidence,
    note: "Actual provider opened a fresh Chrome context and observed empty cart; restored from saved browser receipts after original process disappeared" },
  entryUrl: opened.event.entryUrl, startedAt: null,
  elapsedMs: Math.max(0, Date.parse(observation.at) - Date.parse(opened.at)),
  timingPrecision: { sessionElapsed: "estimated", stepElapsed: "not-recorded",
    source: "Only the recorded browser-open to observation event span is known; this is not total runtime or an upper bound. Termination and original monotonic clocks unavailable." },
  protocol: { passed: false, deviations: ["Original process disappeared during Mac/exec-server interruption", "Total elapsed time and monotonic action clocks unavailable", "No final screenshot or trace; last actual observation screenshot retained"] },
  status: "technical-failure", error: "Original evaluation process no longer existed after reconnect; no final outcome was saved",
  realPurchase: false, fakeOrderConfirmed: false,
  steps: [{ event: { id: `${runId}-step-0`, runId, sequence: 0, at: opened.at, kind: "navigation", message: "Observed fresh headed Chrome navigation from actual saved provider opened event" },
    elapsedMs: null, action: { type: "navigate", url: opened.event.entryUrl } }], artifacts, choice: null,
  verification: { verifierId: "independent-interruption-recovery-v1", independentOfShopperAgent: true,
    catalogSha256: catalog.sha256, hardConstraints: checks(c.privateShopperNeeds.hardConstraints), preferences: checks(c.privateShopperNeeds.preferences) },
  model: { status: "not-observed", readiness: [] }, fallback: { used: false, kind: "none", reason: null },
};
validateShopperSession(session, c, catalog);
await save(path.join(directory, "session.json"), session);
await emit({ type: "recovered-interruption", ...interrupted, status: session.status, actualTotalElapsedKnown: false,
  observationSpanMs: session.elapsedMs, missingTrace: true, rerun: false });
await aggregateShopperResults(packetDirectory, root, path.join(root, "aggregates", "after-recovered-pair-04.json"));
const availability = await codexModelAvailability(plan.shopper.binary, plan.shopper.model);
await save(path.join(root, "recovery-model-availability.json"), availability);
let attempted = 4;
for (const item of plan.schedule.slice(4)) {
  const current = packet.cases.find((candidate) => candidate.caseId === item.caseId)!;
  await emit({ type: "start", ...item, attempted });
  try {
    const actual = await executeShopperSession(packetDirectory, packet, current, item.arm, path.join(root, current.caseId, item.arm), {
      openBrowser: (url, output) => ChromeShopperBrowser.open(url, output),
      openAgent: (assignment, log) => new CodexShopperAgent(plan.shopper.binary, plan.shopper.model, assignment, log),
    }, plan.baseUrl);
    await emit({ type: "finish", ...item, status: actual.status, elapsedMs: actual.elapsedMs, actions: actual.steps.length,
      protocolPassed: actual.protocol.passed, error: actual.error });
  } catch (error) { await emit({ type: "invalid-receipt", ...item, error: error instanceof Error ? error.message : String(error) }); }
  attempted++;
  if (attempted % 2 === 0) {
    const report = await aggregateShopperResults(packetDirectory, root, path.join(root, "aggregates", `after-${item.caseId}.json`));
    await emit({ type: "pair-checkpoint", attempted, fixedSessions: 20, status: report.status, arms: report.arms, setupErrors: report.setupErrors });
  }
}
const report = await aggregateShopperResults(packetDirectory, root, path.join(root, "aggregate.json"));
await emit({ type: "complete", attempted, status: report.status, arms: report.arms, setupErrors: report.setupErrors });
