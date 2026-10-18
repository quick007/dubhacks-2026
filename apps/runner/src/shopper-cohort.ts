/** Fixed-roster, sequential multi-packet execution. Only the real session executor dispatches shoppers. */
import { createHash } from "node:crypto";
import { appendFile, lstat, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ChromeShopperBrowser } from "./shopper-chrome.js";
import { CodexShopperAgent, codexModelAvailability, findCodexBinary } from "./shopper-codex-agent.js";
import { aggregateShopperResults, loadShopperPacket, type Packet } from "./shopper-eval-cli.js";
import { executeShopperSession, pairedExecutionOrder, type SessionDependencies } from "./shopper-runner.js";
import { buildShopperAssignment, validateShopperCatalog, validateShopperSession, type ShopperArm } from "./shopper-contract.js";

const SOURCE_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const PLAN_NAME = "cohort-plan.json";
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);
const sha = (bytes: Buffer | string) => createHash("sha256").update(bytes).digest("hex");
const json = (value: unknown) => JSON.stringify(value, null, 2) + "\n";
const writeJson = (file: string, value: unknown) => writeFile(file, json(value), { flag: "wx" });
function requireValue(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }
const canonical = (value: unknown): string => Array.isArray(value) ? `[${value.map(canonical).join(",")}]`
  : value && typeof value === "object" ? `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}` : JSON.stringify(value);

export type CohortOptions = {
  packets: string[]; snapshot: string; out: string; seed: number; pilotPairs: number; deadline: string;
  baseUrl?: string; model?: string; codexBinary?: string;
};
export type CohortSlot = { packetIndex: number; packetKey: string; pairIndex: number; caseId: string; arm: ShopperArm; runId: string; pilot: boolean };
type FrozenPacket = { directory: string; key: string; files: Record<string, string> };
export type CohortPlan = {
  version: "shopper-fixed-cohort.v1"; synthetic: true; humanConversionClaim: false; createdAt: string;
  fixedPairs: 60; fixedSessions: 120; seed: number; pilotPairs: number; deadline: string;
  baseUrl: string; model: string; requestedCodexBinary: string | null;
  snapshot: { file: string; sha256: string }; packets: FrozenPacket[]; implementationSources: Record<string, string>;
  schedule: CohortSlot[]; catalogSnapshotSha256: string;
  budget: { maxActions: 24; maxWallSeconds: 120 };
  settings: { concurrency: 1; viewport: { width: 1280; height: 960 }; independentChromeProcess: true; independentStorageContext: true; quantity: 1; effort: "low" };
  pilotPolicy: string; deadlinePolicy: string; comparisonScope: string;
};

async function fileHashes(directory: string, prefix = ""): Promise<Record<string, string>> {
  const result: Record<string, string> = {};
  for (const entry of (await readdir(path.join(directory, prefix), { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    const name = prefix ? `${prefix}/${entry.name}` : entry.name;
    requireValue(!entry.isSymbolicLink(), `Symlink cannot be frozen: ${name}`);
    if (entry.isDirectory()) Object.assign(result, await fileHashes(directory, name));
    else { requireValue(entry.isFile(), `Only regular packet files may be frozen: ${name}`); result[name] = sha(await readFile(path.join(directory, name))); }
  }
  return result;
}

async function implementationHashes() {
  const names = (await readdir(SOURCE_DIRECTORY)).filter((name) => /^shopper-.*\.ts$/.test(name) && !name.endsWith(".test.ts")).sort();
  return Object.fromEntries(await Promise.all(names.map(async (name) => [name, sha(await readFile(path.join(SOURCE_DIRECTORY, name)))])));
}

/** A single global pair shuffle; frozen within-pair orders remain untouched. */
export function cohortExecutionOrder(packets: Packet[], seed: number, pilotPairs: number): CohortSlot[] {
  requireValue(Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff, "Seed must be an unsigned 32-bit integer");
  requireValue(packets.length === 6 && packets.every((packet) => packet.cases.length === 10), "Six strict ten-pair packets required");
  requireValue(Number.isInteger(pilotPairs) && pilotPairs >= 1 && pilotPairs <= 60, "Pilot pairs must be 1..60");
  const pairs = packets.flatMap((packet, packetIndex) => {
    const slots = pairedExecutionOrder(packet, seed);
    return Array.from({ length: 10 }, (_, index) => slots.slice(index * 2, index * 2 + 2)
      .map((slot) => ({ ...slot, packetIndex, packetKey: `packet-${packetIndex + 1}` })));
  });
  requireValue(new Set(pairs.flat().map((slot) => slot.runId)).size === 120, "Run IDs must be unique across all packets");
  requireValue(new Set(pairs.map((pair) => pair[0].caseId)).size === 60, "Case IDs must be unique across all packets");
  let state = seed >>> 0;
  const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 0x100000000; };
  for (let index = pairs.length - 1; index > 0; index--) {
    const target = Math.floor(random() * (index + 1)); [pairs[index], pairs[target]] = [pairs[target], pairs[index]];
  }
  return pairs.flatMap((pair, pairIndex) => pair.map((slot) => ({ ...slot, pairIndex, pilot: pairIndex < pilotPairs })));
}

/** This is a CPU-only preflight. No model metadata, browser or external service is contacted. */
export async function buildCohortPlan(options: CohortOptions): Promise<CohortPlan> {
  requireValue(options.packets.length === 6 && new Set(options.packets.map((file) => path.resolve(file))).size === 6, "Six distinct packet directories required");
  const deadlineMs = Date.parse(options.deadline);
  requireValue(Number.isFinite(deadlineMs) && /(?:Z|[+-]\d\d:\d\d)$/.test(options.deadline), "Deadline must be an ISO time with an explicit timezone");
  const snapshotFile = path.resolve(options.snapshot), snapshotBytes = await readFile(snapshotFile);
  const snapshot = JSON.parse(snapshotBytes.toString("utf8"));
  requireValue(snapshot && typeof snapshot === "object" && !Array.isArray(snapshot), "Snapshot receipt must be a JSON object");
  const packets: FrozenPacket[] = [], loaded: Packet[] = [], catalogs: string[] = [];
  let commonBaseUrl = options.baseUrl;
  for (const [index, inputDirectory] of options.packets.entries()) {
    const directory = path.resolve(inputDirectory);
    requireValue(!(await lstat(directory)).isSymbolicLink(), "Packet root cannot be a symlink");
    const packet = await loadShopperPacket(directory), files = await fileHashes(directory);
    const manifest = JSON.parse(await readFile(path.join(directory, "manifest.json"), "utf8"));
    requireValue(manifest.synthetic === true && manifest.scenarioHeldOut === true, "Only fresh scenario-held-out synthetic packets may enter this cohort");
    requireValue(!/provisional|awaiting|pending/i.test(String(manifest.sourceReceiptPhase ?? "")), "Packet source receipt is provisional; wait for the frozen snapshot");
    for (const [name, expected] of Object.entries(manifest.files ?? {})) requireValue(files[name] === expected, `Packet manifest hash mismatch: ${name}`);
    const binding = JSON.parse(await readFile(path.join(directory, "catalog-binding.json"), "utf8"));
    const catalog = validateShopperCatalog(binding.catalogSnapshot);
    catalogs.push(canonical(catalog));
    requireValue(packet.budget.maxActions === 24 && packet.budget.maxWallSeconds === 120, "Identical 24-action/120-second budgets required");
    requireValue(packet.cases.filter((c) => c.armOrder[0] === "generic").length === 5, "Each packet must retain balanced frozen within-pair order");
    commonBaseUrl ??= packet.defaultBaseUrl;
    if (!options.baseUrl) requireValue(packet.defaultBaseUrl === commonBaseUrl, "Packets must share a base URL unless one explicit common override is supplied");
    buildShopperAssignment(packet.cases[0], "generic", commonBaseUrl);
    packets.push({ directory, key: `packet-${index + 1}`, files }); loaded.push(packet);
  }
  requireValue(new Set(catalogs).size === 1, "Identical catalog, variants and prices required across all packets");
  const schedule = cohortExecutionOrder(loaded, options.seed, options.pilotPairs);
  return { version: "shopper-fixed-cohort.v1", synthetic: true, humanConversionClaim: false, createdAt: new Date().toISOString(),
    fixedPairs: 60, fixedSessions: 120, seed: options.seed, pilotPairs: options.pilotPairs, deadline: new Date(deadlineMs).toISOString(),
    baseUrl: commonBaseUrl!, model: options.model ?? "gpt-6.1-sol", requestedCodexBinary: options.codexBinary ? path.resolve(options.codexBinary) : null,
    snapshot: { file: snapshotFile, sha256: sha(snapshotBytes) }, packets, implementationSources: await implementationHashes(),
    schedule, catalogSnapshotSha256: JSON.parse(catalogs[0]).sha256, budget: { maxActions: 24, maxWallSeconds: 120 },
    settings: { concurrency: 1, viewport: { width: 1280, height: 960 }, independentChromeProcess: true, independentStorageContext: true, quantity: 1, effort: "low" },
    pilotPolicy: "The first prespecified pairs are a runtime feasibility pilot under unchanged settings. No outcome-based resizing or retuning. Pilot and remaining cohort are separately identifiable; all 120 assigned slots remain in reporting.",
    deadlinePolicy: "Launch no session at or after the deadline. Finish a session already launched under its 120-second budget; retain every unattempted slot. SIGINT/SIGTERM stop future launches after the current session.",
    comparisonScope: "Combined personalized experience (ranking, copy and latency), fixed catalog/UI/checkout and agent settings; synthetic outcomes only" };
}

export async function createCohortPlan(options: CohortOptions) {
  const plan = await buildCohortPlan(options);
  await mkdir(options.out); // Exclusive ownership, including plan-only invocations.
  await writeJson(path.join(options.out, PLAN_NAME), plan);
  await writeFile(path.join(options.out, ".gitignore"), "**/trace.zip\n**/observation-*.png\n**/browser.jsonl\n", { flag: "wx" });
  return plan;
}

export async function verifyCohortPins(plan: CohortPlan) {
  requireValue(sha(await readFile(plan.snapshot.file)) === plan.snapshot.sha256, "Frozen model snapshot receipt changed");
  requireValue(canonical(await implementationHashes()) === canonical(plan.implementationSources), "Frozen shopper implementation changed");
  for (const packet of plan.packets) requireValue(canonical(await fileHashes(packet.directory)) === canonical(packet.files), `Frozen packet files changed: ${packet.key}`);
}

/** Safe resume never retries a locked session or invents a receipt for an interrupted process. */
export async function inspectCohortAttempts(plan: CohortPlan, out: string) {
  const completed = new Map<string, string>();
  for (const slot of plan.schedule) {
    const directory = path.join(out, slot.packetKey, slot.caseId, slot.arm);
    let names: string[];
    try { names = await readdir(directory); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") continue; throw error; }
    requireValue(names.includes("attempt-lock.json") && names.includes("session.json"),
      `Unfinished or uncertain attempt ${slot.runId} at ${directory}; no automatic retry or fabricated receipt. Resolve independently before resuming.`);
    const lock = JSON.parse(await readFile(path.join(directory, "attempt-lock.json"), "utf8"));
    requireValue(lock.runId === slot.runId, `Attempt lock run ID mismatch: ${slot.runId}`);
    const packet = await loadShopperPacket(plan.packets[slot.packetIndex].directory);
    const binding = JSON.parse(await readFile(path.join(plan.packets[slot.packetIndex].directory, "catalog-binding.json"), "utf8"));
    const sessionBytes = await readFile(path.join(directory, "session.json"));
    const session = validateShopperSession(JSON.parse(sessionBytes.toString("utf8")), packet.cases.find((c) => c.caseId === slot.caseId)!, validateShopperCatalog(binding.catalogSnapshot));
    requireValue(session.run.id === slot.runId && session.arm === slot.arm, `Preserved receipt identity mismatch: ${slot.runId}`);
    completed.set(slot.runId, sha(sessionBytes));
  }
  return completed;
}

type RunHooks = {
  dependencies: SessionDependencies; now?: () => Date; shouldStop?: () => boolean;
  execute?: typeof executeShopperSession; aggregate?: typeof aggregateShopperResults;
};

/** Test hooks inject CPU fixtures only; the CLI always supplies the installed Chrome/Codex dependencies. */
export async function dispatchCohort(plan: CohortPlan, out: string, hooks: RunHooks) {
  await verifyCohortPins(plan);
  const now = hooks.now ?? (() => new Date()), execute = hooks.execute ?? executeShopperSession;
  const aggregate = hooks.aggregate ?? aggregateShopperResults;
  const completed = await inspectCohortAttempts(plan, out), packets = await Promise.all(plan.packets.map((packet) => loadShopperPacket(packet.directory)));
  const activeLock = path.join(out, "active-run-lock.json"), invocation = `${Date.now()}-${process.pid}`;
  await writeJson(activeLock, { pid: process.pid, startedAt: now().toISOString(), invocation,
    policy: "Exclusive active dispatcher. A stale lock requires independent process/receipt inspection; never steal it automatically." });
  let signal: string | null = null, stopReason = "roster-finished", launched = 0;
  const stop = (received: string) => { signal ??= received; };
  const sigint = () => stop("SIGINT"), sigterm = () => stop("SIGTERM");
  process.on("SIGINT", sigint); process.on("SIGTERM", sigterm);
  const log = (value: unknown) => appendFile(path.join(out, "dispatch.jsonl"), JSON.stringify(value) + "\n");
  const checkpoint = async (packetIndex: number, name: string) => {
    const packet = plan.packets[packetIndex], directory = path.join(out, packet.key);
    await mkdir(path.join(directory, "aggregates"), { recursive: true });
    const result = await aggregate(packet.directory, directory, path.join(directory, "aggregates", `${name}.json`));
    requireValue(result.setupErrors.length === 0, `Aggregation blocked for ${packet.key}: ${result.setupErrors.join("; ")}`);
    return { file: `${packet.key}/aggregates/${name}.json`, status: result.status,
      recorded: result.rows.filter((row) => row.status === "recorded").length,
      invalidSidecars: result.rows.filter((row) => row.status === "invalid-sidecar").length };
  };
  try {
    await log({ type: "invocation", invocation, at: now().toISOString(), preservedReceipts: Object.fromEntries(completed), fixedAssignedSessions: 120 });
    // Scorer freshness/gold validation must succeed before any Chrome process is opened.
    for (let index = 0; index < 6; index++) await checkpoint(index, `preflight-${invocation}`);
    for (let pairIndex = 0; pairIndex < 60; pairIndex++) {
      const pair = plan.schedule.slice(pairIndex * 2, pairIndex * 2 + 2);
      let pairLaunched = false;
      for (const slot of pair) {
        if (completed.has(slot.runId)) continue;
        if (signal || hooks.shouldStop?.()) { stopReason = signal ?? "requested-stop"; break; }
        if (now().getTime() >= Date.parse(plan.deadline)) { stopReason = "deadline"; break; }
        await verifyCohortPins(plan);
        const packet = packets[slot.packetIndex], c = packet.cases.find((item) => item.caseId === slot.caseId)!;
        await log({ type: "start", at: now().toISOString(), ...slot });
        launched++; pairLaunched = true;
        try {
          const session = await execute(plan.packets[slot.packetIndex].directory, packet, c, slot.arm,
            path.join(out, slot.packetKey, slot.caseId, slot.arm), hooks.dependencies, plan.baseUrl);
          await log({ type: "finish", at: now().toISOString(), ...slot, status: session.status,
            steps: session.steps.length, elapsedMs: session.elapsedMs, protocolPassed: session.protocol.passed });
          completed.set(slot.runId, sha(await readFile(path.join(out, slot.packetKey, slot.caseId, slot.arm, "session.json"))));
        } catch (error) {
          await log({ type: "unfinished-or-invalid-receipt", at: now().toISOString(), ...slot, error: errorText(error) });
          throw new Error(`Stopped after ${slot.runId}: ${errorText(error)}. Evidence retained; no retry or receipt invented.`);
        }
      }
      if (pairLaunched) {
        const summary = await checkpoint(pair[0].packetIndex, `after-pair-${String(pairIndex + 1).padStart(2, "0")}-${invocation}`);
        await log({ type: "pair-checkpoint", at: now().toISOString(), pairIndex, ...summary });
      }
      if (stopReason !== "roster-finished") break;
      if (pairIndex + 1 === plan.pilotPairs) {
        await log({ type: "pilot-complete", at: now().toISOString(), pairs: plan.pilotPairs, completedReceipts: completed.size,
          decisionRule: "Continue unchanged until fixed roster/deadline/explicit stop; do not tune to outcomes" });
      }
    }
    for (const slot of plan.schedule.filter((item) => completed.has(item.runId))) {
      requireValue(sha(await readFile(path.join(out, slot.packetKey, slot.caseId, slot.arm, "session.json"))) === completed.get(slot.runId),
        `Preserved session receipt changed during dispatch: ${slot.runId}`);
    }
    const aggregates = [];
    for (let index = 0; index < 6; index++) aggregates.push(await checkpoint(index, `final-${invocation}`));
    const metadata = { version: "shopper-cohort-dispatch.v1", synthetic: true, humanConversionClaim: false,
      invocation, at: now().toISOString(), fixedPairs: 60, fixedSessions: 120, launchedThisInvocation: launched,
      recordedReceipts: completed.size, unattempted: 120 - completed.size, stopReason,
      pilotPairs: plan.pilotPairs, schedule: plan.schedule.map((slot) => ({ ...slot, status: completed.has(slot.runId) ? "recorded" : "unattempted" })),
      aggregateFiles: aggregates, planSha256: sha(await readFile(path.join(out, PLAN_NAME))),
      note: "Recorded is receipt presence, not success or protocol eligibility. See per-packet aggregates; fixed denominators include every unattempted slot." };
    await writeJson(path.join(out, `cohort-result-${invocation}.json`), metadata);
    return metadata;
  } finally {
    process.off("SIGINT", sigint); process.off("SIGTERM", sigterm);
    await rm(activeLock);
  }
}

export async function shopperCohortMain(args: string[]) {
  const command = args.shift(); requireValue(command === "plan" || command === "run", "Use plan or run");
  const options = new Map<string, string>(); let resume = false;
  while (args.length) {
    const key = args.shift()!;
    if (key === "--resume") { requireValue(!resume && command === "run", "--resume is a unique run-only flag"); resume = true; continue; }
    const value = args.shift(); requireValue(key.startsWith("--") && value && !value.startsWith("--") && !options.has(key), "Unique --key value options required");
    options.set(key, value);
  }
  const allowed = ["--packets-json", "--snapshot", "--out", "--seed", "--pilot-pairs", "--deadline", "--base-url", "--model", "--codex-binary"];
  for (const key of options.keys()) requireValue(allowed.includes(key), `Unknown option ${key}`);
  const inputs: unknown = JSON.parse(options.get("--packets-json") ?? "null");
  requireValue(Array.isArray(inputs) && inputs.every((item) => typeof item === "string"), "--packets-json requires a JSON array of six directories");
  requireValue(options.get("--snapshot") && options.get("--out") && options.get("--deadline"), "--snapshot, --out and --deadline required");
  const config: CohortOptions = { packets: inputs, snapshot: options.get("--snapshot")!, out: path.resolve(options.get("--out")!),
    seed: Number(options.get("--seed") ?? "20261018"), pilotPairs: Number(options.get("--pilot-pairs") ?? "5"), deadline: options.get("--deadline")!,
    baseUrl: options.get("--base-url"), model: options.get("--model"), codexBinary: options.get("--codex-binary") };
  let plan: CohortPlan;
  if (resume) {
    plan = JSON.parse(await readFile(path.join(config.out, PLAN_NAME), "utf8"));
    const expected = await buildCohortPlan(config), { createdAt: _a, ...old } = plan, { createdAt: _b, ...fresh } = expected;
    requireValue(canonical(old) === canonical(fresh), "Resume settings, roster or frozen hashes differ from saved plan");
    await inspectCohortAttempts(plan, config.out);
  } else plan = await createCohortPlan(config);
  if (command === "plan") return { command, output: config.out, fixedPairs: 60, fixedSessions: 120, browserDispatched: false };
  // Model metadata is checked once; no paid credentials are configured or read by this orchestrator.
  const binary = await findCodexBinary(plan.requestedCodexBinary ?? undefined);
  const availability = await codexModelAvailability(binary, plan.model);
  await writeJson(path.join(config.out, `shopper-availability-${Date.now()}-${process.pid}.json`), { binary, model: plan.model, availability });
  const result = await dispatchCohort(plan, config.out, {
    dependencies: { openBrowser: (url, directory) => ChromeShopperBrowser.open(url, directory),
      openAgent: (assignment, log) => new CodexShopperAgent(binary, plan.model, assignment, log) },
  });
  return { command, output: config.out, fixedSessions: result.fixedSessions, recordedReceipts: result.recordedReceipts,
    unattempted: result.unattempted, stopReason: result.stopReason, aggregateFiles: result.aggregateFiles };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  shopperCohortMain(process.argv.slice(2)).then((result) => process.stdout.write(JSON.stringify(result) + "\n"))
    .catch((error) => { process.stderr.write(errorText(error) + "\n"); process.exitCode = 1; });
}
