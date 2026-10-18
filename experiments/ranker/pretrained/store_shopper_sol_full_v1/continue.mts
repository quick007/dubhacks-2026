/** Continue the committed smoke schedule without rerunning its completed first pair.
 * Uses the unchanged runner/agent/provider; no evaluator data enters shopper inputs.
 */
import { createHash } from "node:crypto";
import { appendFile, cp, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { executeShopperSession, pairedExecutionOrder } from "../../../../apps/runner/src/shopper-runner.ts";
import { loadShopperPacket, aggregateShopperResults } from "../../../../apps/runner/src/shopper-eval-cli.ts";
import { ChromeShopperBrowser } from "../../../../apps/runner/src/shopper-chrome.ts";
import { CodexShopperAgent, codexModelAvailability } from "../../../../apps/runner/src/shopper-codex-agent.ts";

const root = path.dirname(fileURLToPath(import.meta.url));
const workspace = path.resolve(root, "../../../..");
const previous = path.resolve(root, "../store_shopper_sol_smoke_v2");
const packetDirectory = path.resolve(root, "../store_shopper_eval_v1");
const sha = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
const save = (name: string, value: unknown) => writeFile(path.join(root, name), JSON.stringify(value, null, 2) + "\n", { flag: "wx" });
const emit = async (event: unknown) => {
  const line = JSON.stringify({ at: new Date().toISOString(), event });
  await appendFile(path.join(root, "dispatch.jsonl"), line + "\n");
  process.stdout.write(line + "\n");
};

const sourceBytes = await readFile(path.join(previous, "execution-plan.json"));
const original = JSON.parse(sourceBytes.toString("utf8"));
const packet = await loadShopperPacket(packetDirectory);
if (JSON.stringify(pairedExecutionOrder(packet, original.seed)) !== JSON.stringify(original.schedule)) throw new Error("Frozen schedule differs");
for (const [name, expected] of Object.entries(original.implementationSources)) {
  if (sha(await readFile(path.join(workspace, "apps/runner/src", name))) !== expected) throw new Error(`Committed implementation changed: ${name}`);
}
if (original.shopper.model !== "gpt-6.1-sol" || original.shopper.effort !== "low" || original.settings.concurrency !== 1) throw new Error("Unchanged committed shopper settings required");
const binary = original.shopper.binary;
const availability = await codexModelAvailability(binary, original.shopper.model);
await save("attempt-lock.json", { startedAt: new Date().toISOString(), sourcePlanSha256: sha(sourceBytes) });
await mkdir(path.join(root, "aggregates"));
const imported = [];
for (const item of original.schedule.slice(0, 2)) {
  const source = path.join(previous, item.caseId, item.arm);
  const bytes = await readFile(path.join(source, "session.json"));
  const session = JSON.parse(bytes.toString("utf8"));
  if (session.status !== "completed" || !session.protocol.passed || session.run.id !== item.runId) throw new Error("Only the actual verified completed first pair may be reused");
  await cp(source, path.join(root, item.caseId, item.arm), { recursive: true, force: false, errorOnExist: true });
  imported.push({ ...item, source: path.relative(root, source), sessionSha256: sha(bytes), originalStartedAt: session.startedAt });
}
await save("execution-plan.json", { ...original, limit: 20, remainingSessions: 18, reusedCompletedSessions: 2,
  sourcePlanSha256: sha(sourceBytes), imported, shopper: { ...original.shopper, modelAvailability: availability },
  continuation: "Identical committed code/settings; first two completed sessions copied byte-for-byte, remaining eighteen sequential in the original seeded order",
  cloudExportPerformed: false });
await aggregateShopperResults(packetDirectory, root, path.join(root, "aggregates", "after-imported-pair.json"));
await emit({ type: "imported-pair", imported, remaining: 18 });
let attempted = 2;
for (const item of original.schedule.slice(2)) {
  const c = packet.cases.find((candidate) => candidate.caseId === item.caseId)!;
  await emit({ type: "start", ...item, attempted });
  try {
    const session = await executeShopperSession(packetDirectory, packet, c, item.arm, path.join(root, c.caseId, item.arm), {
      openBrowser: (url, directory) => ChromeShopperBrowser.open(url, directory),
      openAgent: (assignment, log) => new CodexShopperAgent(binary, original.shopper.model, assignment, log),
    }, original.baseUrl);
    await emit({ type: "finish", ...item, status: session.status, elapsedMs: session.elapsedMs, actions: session.steps.length,
      protocolPassed: session.protocol.passed, error: session.error });
  } catch (error) {
    await emit({ type: "invalid-receipt", ...item, error: error instanceof Error ? error.message : String(error) });
  }
  attempted++;
  if (attempted % 2 === 0) {
    const report = await aggregateShopperResults(packetDirectory, root, path.join(root, "aggregates", `after-${item.caseId}.json`));
    await emit({ type: "pair-checkpoint", attempted, fixedSessions: 20, status: report.status, arms: report.arms, setupErrors: report.setupErrors });
  }
}
const report = await aggregateShopperResults(packetDirectory, root, path.join(root, "aggregate.json"));
await emit({ type: "complete", attempted, status: report.status, arms: report.arms, setupErrors: report.setupErrors });
