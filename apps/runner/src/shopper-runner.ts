/** Sequential paired shopper execution. Shopper decisions and evaluator inputs stay separate. */
import { createHash } from "node:crypto";
import { access, appendFile, constants, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";
import { StdioShopperAgent, type ShopperAgent } from "./shopper-agent.js";
import { ChromeShopperBrowser } from "./shopper-chrome.js";
import { CodexShopperAgent, codexModelAvailability, findCodexBinary } from "./shopper-codex-agent.js";
import { loadShopperPacket, aggregateShopperResults, type Packet } from "./shopper-eval-cli.js";
import { buildShopperAssignment, validateShopperCatalog, validateShopperSession, verifyShopperNeeds,
  SHOPPER_CONTRACT_VERSION, type ShopperArm, type ShopperCase, type ShopperSession,
  type ShopperArtifact, type ShopperAction, type ShopperChoice } from "./shopper-contract.js";

const sha = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
const errorText = (e: unknown) => e instanceof Error ? e.message : String(e);
const writeJson = (file: string, value: unknown) => writeFile(file, JSON.stringify(value, null, 2) + "\n", { flag: "wx" });

/** Seeded Fisher-Yates permutes pair execution order, retaining each frozen within-pair order. */
export function pairedExecutionOrder(packet: Packet, seed: number) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error("Schedule seed must be an unsigned 32-bit integer");
  let state = seed >>> 0;
  const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 0x100000000; };
  const cases = [...packet.cases];
  for (let i = cases.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [cases[i], cases[j]] = [cases[j], cases[i]]; }
  return cases.flatMap((c) => c.armOrder.map((arm) => ({ caseId: c.caseId, arm, runId: packet.runIds[`${c.caseId}/${arm}`] })));
}

export async function captureArtifact(directory: string, runId: string, name: string,
  kind: ShopperArtifact["artifact"]["kind"], mimeType: string): Promise<ShopperArtifact> {
  const artifact = { id: `${runId}-${name.replace(/[^a-zA-Z0-9]/g, "-")}`, runId, kind,
    path: name, mimeType, createdAt: new Date().toISOString() };
  try { const bytes = await readFile(path.join(directory, name));
    if (!bytes.length) throw new Error("Artifact file is empty");
    return { artifact, status: "captured", sha256: sha(bytes), bytes: bytes.length, error: null };
  } catch (error) { return { artifact, status: "missing", sha256: null, bytes: 0, error: errorText(error) }; }
}

type BrowserPort = Pick<ChromeShopperBrowser, "observe" | "act" | "close"> & {
  browserVersion: string; contextId: string; tabId: string;
};
export type SessionDependencies = {
  openBrowser: (entryUrl: string, directory: string) => Promise<BrowserPort>;
  openAgent: (assignment: ReturnType<typeof buildShopperAssignment>, log: (e: unknown) => void) => ShopperAgent;
};

export async function executeShopperSession(packetDirectory: string, packet: Packet, c: ShopperCase, arm: ShopperArm,
  directory: string, dependencies: SessionDependencies, baseUrl = packet.defaultBaseUrl) {
  await mkdir(directory, { recursive: true });
  const runId = packet.runIds[`${c.caseId}/${arm}`], assignment = buildShopperAssignment(c, arm, baseUrl);
  // Exclusive session ownership: a failed/repeated invocation cannot overwrite earlier evidence.
  await writeFile(path.join(directory, "attempt-lock.json"), JSON.stringify({ runId, startedAt: new Date().toISOString() }), { flag: "wx" });
  const logs: unknown[] = [], steps: ShopperSession["steps"] = [];
  let browser: BrowserPort | undefined, agent: ShopperAgent | undefined;
  let status: ShopperSession["status"] = "technical-failure", error: string | null = null;
  let confirmed = false, initialCartEmpty: boolean | null = null;
  let observedChoice: { originalName: string; catalogPrice: number; selectedColor: string; quantity: 1 } | undefined;
  const deviations: string[] = [], readiness: { entry: unknown; capturedAtElapsedMs: number }[] = [];
  let queryStartBrowserTime: number | null = null, localStatusText = "";
  let fallback: ShopperSession["fallback"] = { used: false, kind: "none", reason: null };
  const startedAt = new Date().toISOString(), start = performance.now();
  const elapsed = () => performance.now() - start;
  let finishedElapsed = 0, stopped = false;
  const emit = (action: ShopperAction, message: string) => {
    const sequence = steps.length;
    steps.push({ event: { id: `${runId}-step-${sequence}`, runId, sequence, at: new Date().toISOString(),
      kind: action.type === "navigate" ? "navigation" : "action", message }, elapsedMs: elapsed(), action });
  };
  const maxMs = packet.budget.maxWallSeconds * 1000;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const work = async () => {
    // A fresh Chrome process/context and independent shopper child are created for each arm.
    const opened = await dependencies.openBrowser(assignment.entryUrl, directory);
    if (stopped) { await opened.close(); return; }
    browser = opened;
    emit({ type: "navigate", url: assignment.entryUrl }, "Navigated a fresh isolated headed Chrome context to assigned storefront");
    let observation = await browser.observe();
    if (stopped) return;
    initialCartEmpty = observation.cartEmpty;
    logs.push({ type: "initial-observation", elapsedMs: elapsed(), ...observation, screenshotBase64: "saved by browser trace" });
    if (!initialCartEmpty) throw new Error("Initial empty cart could not be independently observed");
    agent = dependencies.openAgent(assignment, (event) => logs.push({ elapsedMs: elapsed(), event }));
    while (!stopped && steps.length < packet.budget.maxActions) {
      const current = new URL(observation.url), entry = new URL(assignment.entryUrl);
      const hash = current.hash.slice(1).split("?");
      const query = new URLSearchParams(hash[1] ?? "").get("q");
      if (current.origin !== entry.origin || current.pathname !== "/" || current.search !== entry.search
        || (query && query !== assignment.recommenderContext.query)) throw new Error("Browser changed arm/origin or disclosed an unassigned query");
      const action = await agent.decide(observation, { actions: packet.budget.maxActions - steps.length, wallMs: Math.max(0, maxMs - elapsed()) });
      if (stopped) break;
      logs.push({ type: "agent-action", elapsedMs: elapsed(), action });
      if (action.type === "abandon") { emit({ type: "abandon", target: action.reason }, action.reason); status = "abandoned"; return; }
      if (action.type === "search" && action.query !== assignment.recommenderContext.query) throw new Error("Shopper tried to reveal private/unassigned search context");
      if (action.type === "search") queryStartBrowserTime = observation.observedAtPerformanceMs;
      const result = await browser.act(action);
      if (stopped) break;
      if (result.selected) observedChoice = result.selected;
      // Product IDs are bound only after the shopper exits. Retain literal observed control actions here.
      emit(action.type === "search" ? { type: "search", query: action.query }
        : { type: result.kind === "open-cart" || result.kind === "confirm-demo-order" ? result.kind : "inspect", target: result.target }, result.target);
      logs.push({ type: "browser-result", elapsedMs: elapsed(), stepSequence: steps.length - 1, result });
      if (result.confirmationVisible) { confirmed = true; status = "constraint-failure"; return; }
      observation = await browser.observe();
      if (queryStartBrowserTime !== null) {
        const queryStart = queryStartBrowserTime;
        readiness.push(...observation.readiness.filter((entry) => typeof (entry as {startTime?:unknown})?.startTime === "number"
          && (entry as {startTime:number}).startTime >= queryStart).map((entry) => ({ entry, capturedAtElapsedMs: elapsed() })));
        localStatusText = observation.modelStatusText;
        if (arm === "local-model" && /last available result shown/i.test(localStatusText)) fallback = { used: true, kind: "last-safe-model", reason: localStatusText };
        else if (arm === "local-model" && /local model unavailable/i.test(localStatusText)) fallback = { used: true, kind: "service-unavailable", reason: localStatusText };
      }
      logs.push({ type: "observation", elapsedMs: elapsed(), ...observation, screenshotBase64: "saved by browser trace" });
    }
    if (!stopped) { status = "timeout"; error = "Shopper exhausted the action budget"; }
  };
  try {
    await Promise.race([work(), new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error("Shopper exhausted the wall-clock budget")), maxMs);
    })]);
  } catch (cause) { error = errorText(cause); status = elapsed() >= maxMs ? "timeout" : "technical-failure"; }
  finally {
    stopped = true; finishedElapsed = elapsed(); if (timer) clearTimeout(timer);
    await agent?.close().catch((cause) => logs.push({ type: "agent-close-error", error: errorText(cause) }));
    await browser?.close().catch((cause) => { deviations.push(`Browser artifact cleanup failed: ${errorText(cause)}`); });
  }
  if (finishedElapsed > maxMs) deviations.push("Wall-clock budget exceeded");
  if (!browser) deviations.push("Chrome session was not established");
  if (browser && !initialCartEmpty) deviations.push("Initial empty cart was not verified");
  await writeJson(path.join(directory, "browser-observations.json"), { synthetic: true, logs, readiness, observedChoice: observedChoice ?? null });
  const artifacts = await Promise.all([
    captureArtifact(directory, runId, "screenshot.png", "screenshot", "image/png"),
    captureArtifact(directory, runId, "trace.zip", "trace", "application/zip"),
    captureArtifact(directory, runId, "browser-observations.json", "result", "application/json"),
  ]);
  try {
    await access(path.join(directory, "checkout-receipt.json"));
    artifacts.push(await captureArtifact(directory, runId, "checkout-receipt.json", "result", "application/json"));
  } catch { /* Existing external providers and historical receipts may not expose server checkout evidence. */ }
  const evidence = artifacts.filter((a) => a.status === "captured").map((a) => a.artifact.id);
  // The independent catalog is first opened AFTER the shopper and browser have stopped.
  const binding = JSON.parse(await readFile(path.join(packetDirectory, "catalog-binding.json"), "utf8"));
  const catalog = validateShopperCatalog(binding.catalogSnapshot);
  const selectionSteps: number[] = [];
  for (const value of logs) {
    const log = value as { type?: string; stepSequence?: number; result?: { kind?: string; selected?: typeof observedChoice } };
    const selected = log.result?.selected;
    if (log.type !== "browser-result" || log.stepSequence === undefined || !selected || log.result?.kind !== "add-to-cart") continue;
    const product = catalog.products.find((p) => p.originalName === selected.originalName && p.itemPrice === selected.catalogPrice);
    const variant = product?.variants.find((v) => v.selectedColor === selected.selectedColor);
    if (product && variant) {
      steps[log.stepSequence].action = { type: "add-to-cart", productId: product.productId, variantId: variant.variantId };
      if (selected.originalName === observedChoice?.originalName && selected.selectedColor === observedChoice.selectedColor) selectionSteps.push(log.stepSequence);
    }
  }
  let choice: ShopperChoice | null = null;
  if (observedChoice) {
    const observed = observedChoice;
    const product = catalog.products.find((p) => p.originalName === observed.originalName && p.itemPrice === observed.catalogPrice);
    const variant = product?.variants.find((v) => v.selectedColor === observed.selectedColor);
    if (product && variant) {
      choice = { ...observedChoice, productId: product.productId, variantId: variant.variantId, observationSource: "independent-dom",
        sourceStepIds: [...selectionSteps.map((sequence) => steps[sequence].event.id),
          ...steps.filter((s) => s.action.type === "open-cart").map((s) => s.event.id)], evidenceArtifactIds: evidence };
    } else { deviations.push("Observed cart identity/color/price could not be independently bound"); }
  }
  const check = (needs: ShopperCase["privateShopperNeeds"]["hardConstraints"]) => verifyShopperNeeds(needs, choice, catalog)
    .map((item) => ({ ...item, evidenceArtifactIds: browser ? evidence : [] }));
  const hardConstraints = check(c.privateShopperNeeds.hardConstraints);
  if (confirmed && choice && hardConstraints.every((item) => item.status === "met")) status = "completed";
  const observedReadiness: ShopperSession["model"]["readiness"] = [];
  const seen = new Set<string>();
  for (const capture of readiness) {
    const entry = capture.entry as { name?: unknown; startTime?: unknown; duration?: unknown; detail?: Record<string, unknown> };
    if (!entry || typeof entry !== "object" || !entry.detail) continue;
    const { stage, requestId, acceptedFields, withheldFields, requiredFields } = entry.detail;
    const key = `${requestId}/${stage}`;
    if (seen.has(key) || !["rank", "firstThree", "complete"].includes(String(stage)) || typeof requestId !== "string"
      || entry.name !== `fieldwork.local-model.${stage}.after-paint-opportunity`
      || typeof entry.startTime !== "number" || typeof entry.duration !== "number" || entry.duration < 0
      || entry.duration > capture.capturedAtElapsedMs || capture.capturedAtElapsedMs > finishedElapsed) continue;
    if (stage !== "rank" && ![acceptedFields, withheldFields, requiredFields].every((value) => typeof value === "number" && Number.isInteger(value) && value >= 0)) continue;
    seen.add(key);
    observedReadiness.push({ name: entry.name, stage: stage as "rank" | "firstThree" | "complete", requestId,
      startTimeMs: entry.startTime, durationMs: entry.duration, capturedAtElapsedMs: capture.capturedAtElapsedMs,
      acceptedFields: stage === "rank" ? null : acceptedFields as number,
      withheldFields: stage === "rank" ? null : withheldFields as number,
      requiredFields: stage === "rank" ? null : requiredFields as number,
      source: "data-local-model-readiness", artifactId: artifacts[2].artifact.id, scope: "after-paint-opportunity",
      physicalPaintMeasured: false, imageTransferMeasured: false });
  }
  const session: ShopperSession = {
    contractVersion: SHOPPER_CONTRACT_VERSION, caseId: c.caseId, arm,
    run: { id: runId, experimentId: packet.experimentId ?? "actual-store-isolated-shopper-v1", personaId: c.caseId, personaSplit: c.split, seed: c.seed },
    browser: { name: "Chrome", headed: true, actualNavigation: !!browser, version: browser?.browserVersion ?? null, automation: "playwright-chrome" },
    isolation: browser ? { kind: "fresh-storage-context", tabId: browser.tabId, contextId: browser.contextId, sharedStorage: false,
      storageCleared: true, initialCartEmpty, evidenceArtifactIds: evidence, note: "New headed Chrome process and unshared context; 1280x960 viewport; independent shopper child" }
      : { kind: "not-created", tabId: null, contextId: null, sharedStorage: null, storageCleared: null, initialCartEmpty: null,
        evidenceArtifactIds: [], note: "Chrome failed before an observed navigation" },
    entryUrl: assignment.entryUrl, startedAt, elapsedMs: Math.max(finishedElapsed, ...steps.map((s) => s.elapsedMs ?? 0)),
    timingPrecision: { sessionElapsed: "measured-monotonic", stepElapsed: "measured-monotonic", source: "Node performance.now from Chrome navigation dispatch through observed outcome; excludes cleanup/scoring" },
    protocol: { passed: deviations.length === 0, deviations }, status, error,
    realPurchase: false, fakeOrderConfirmed: confirmed, steps, artifacts, choice,
    verification: { verifierId: "independent-catalog-verifier-v1", independentOfShopperAgent: true,
      catalogSha256: catalog.sha256, hardConstraints, preferences: check(c.privateShopperNeeds.preferences) },
    model: { status: arm === "generic" ? "not-enabled" : /local model unavailable/i.test(localStatusText) ? "failed" : observedReadiness.some((m) => m.stage === "complete") ? "complete"
      : observedReadiness.length ? "partial" : "not-observed", readiness: arm === "generic" ? [] : observedReadiness },
    fallback,
  };
  // Only actual mirrored performance entries become readiness samples; raw entries remain auditable.
  try { validateShopperSession(session, c, catalog); }
  catch (cause) { await writeJson(path.join(directory, "invalid-session.json"), { session, validationError: errorText(cause) }); throw cause; }
  await writeJson(path.join(directory, "session.json"), session);
  return session;
}

export async function shopperRunnerMain(args: string[]) {
  const command = args.shift();
  if (command !== "plan" && command !== "run") throw new Error("Use plan or run");
  const options = new Map<string, string>();
  while (args.length) { const key = args.shift()!, value = args.shift();
    if (!key.startsWith("--") || !value || options.has(key)) throw new Error("Unique --key value options required");
    options.set(key, value); }
  const allowed = ["--packet", "--out", "--seed", "--agent", "--agent-args-json", "--limit", "--base-url", "--codex-binary", "--model"];
  for (const key of options.keys()) if (!allowed.includes(key)) throw new Error(`Unknown option ${key}`);
  const packetDirectory = options.get("--packet"), out = options.get("--out");
  if (!packetDirectory || !out) throw new Error("--packet and --out required");
  const packet = await loadShopperPacket(packetDirectory), seed = Number(options.get("--seed") ?? "20261018");
  const baseUrl = options.get("--base-url") ?? packet.defaultBaseUrl;
  buildShopperAssignment(packet.cases[0], "generic", baseUrl);
  const schedule = pairedExecutionOrder(packet, seed), limit = Number(options.get("--limit") ?? "20");
  if (!Number.isInteger(limit) || limit < 1 || limit > 20) throw new Error("--limit must be 1..20");
  const agentCommand = options.get("--agent"), agentArgs: unknown = JSON.parse(options.get("--agent-args-json") ?? "[]");
  if (!Array.isArray(agentArgs) || !agentArgs.every((item) => typeof item === "string")) throw new Error("Agent args must be a JSON array of strings");
  let codexBinary: string | undefined, modelAvailability: Record<string, unknown> | undefined;
  const model = options.get("--model") ?? "gpt-6.1-sol";
  if (command === "run") {
    if (agentCommand) {
      if (!path.isAbsolute(agentCommand) || options.has("--codex-binary")) throw new Error("Use either an absolute --agent or --codex-binary");
      await access(agentCommand, constants.X_OK);
    } else {
      codexBinary = await findCodexBinary(options.get("--codex-binary"));
      modelAvailability = await codexModelAvailability(codexBinary, model);
    }
  }
  await mkdir(out);
  const sourceDirectory = path.dirname(fileURLToPath(import.meta.url));
  const implementationSources = Object.fromEntries(await Promise.all([
    "shopper-runner.ts", "shopper-agent.ts", "shopper-codex-agent.ts", "shopper-chrome.ts", "shopper-contract.ts", "shopper-eval-cli.ts",
  ].map(async (name) => [name, sha(await readFile(path.join(sourceDirectory, name)))])));
  const plan = { version: "isolated-chrome-shopper-execution.v1", synthetic: true, humanConversionClaim: false,
    seed, baseUrl, randomization: "seeded pair permutation; frozen balanced within-pair arm orders retained", schedule,
    settings: { viewport: { width: 1280, height: 960 }, chrome: "installed headed Google Chrome", quantity: 1,
      independentProcessPerSession: true, independentStorageContextPerSession: true, concurrency: 1 },
    budget: packet.budget, packetSources: packet.sources, implementationSources, limit, agentConfigured: !!agentCommand || !!codexBinary,
    shopper: agentCommand ? { adapter: "external-stdio" } : { adapter: "official-codex-cli", model, effort: "low", binary: codexBinary ?? null, modelAvailability: modelAvailability ?? null },
    browserDispatched: command === "run", limitations: ["Synthetic development scenarios; no human conversion lift", "Coordinate native timing exclusion windows before run", "Missing readiness observations are not latency samples"] };
  await writeJson(path.join(out, "execution-plan.json"), plan);
  if (command === "plan") return plan;
  for (const item of schedule.slice(0, limit)) {
    const c = packet.cases.find((candidate) => candidate.caseId === item.caseId)!;
    await appendFile(path.join(out, "dispatch.jsonl"), JSON.stringify({ ...item, type: "start", at: new Date().toISOString() }) + "\n");
    try {
      const session = await executeShopperSession(packetDirectory, packet, c, item.arm, path.join(out, c.caseId, item.arm), {
        openBrowser: (url, directory) => ChromeShopperBrowser.open(url, directory),
        openAgent: (assignment, log) => agentCommand ? new StdioShopperAgent(agentCommand, agentArgs as string[], assignment, log)
          : new CodexShopperAgent(codexBinary!, model, assignment, log),
      }, baseUrl);
      await appendFile(path.join(out, "dispatch.jsonl"), JSON.stringify({ ...item, type: "finish", at: new Date().toISOString(), status: session.status }) + "\n");
    } catch (error) {
      await appendFile(path.join(out, "dispatch.jsonl"), JSON.stringify({ ...item, type: "invalid-receipt", at: new Date().toISOString(), error: errorText(error) }) + "\n");
    }
  }
  return aggregateShopperResults(packetDirectory, out, path.join(out, "aggregate.json"));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  shopperRunnerMain(process.argv.slice(2)).then((result) => process.stdout.write(JSON.stringify(result, null, 2) + "\n"))
    .catch((error) => { process.stderr.write(errorText(error) + "\n"); process.exitCode = 1; });
}
