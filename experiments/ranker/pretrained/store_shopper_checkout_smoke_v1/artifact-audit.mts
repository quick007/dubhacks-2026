/** CPU-only audit of finalized synthetic cohort evidence. Never dispatches a browser/model.
 * Usage: node --import tsx artifact-audit.mts --cohort-result FINAL.json
 *   --packet PACKET_V2 ... --packet PACKET_V7 --out NEW-AUDIT.json
 * Self-contained ephemeral fixtures: node --import tsx artifact-audit.mts --self-test
 */
import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { lstat, mkdir, mkdtemp, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual, promisify } from "node:util";
import assert from "node:assert/strict";

type Obj = Record<string, any>;
type Source = { path: string; sha256: string; bytes: number } | { path: string; missing: true };
export type AuditOptions = { cohortResult: string; packets: string[]; out: string; python?: string };
const sha = (value: Buffer | string) => createHash("sha256").update(value).digest("hex");
const encode = (value: unknown) => JSON.stringify(value, null, 2) + "\n";
const HASH = /^[a-f0-9]{64}$/;
const ID = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}$/;
const ARMS = ["generic", "local-model"];
const invoke = promisify(execFile);
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);
function need(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }
function obj(value: unknown, label: string): Obj {
  need(value !== null && typeof value === "object" && !Array.isArray(value), `${label}: object required`); return value as Obj;
}
function finite(value: unknown): value is number { return typeof value === "number" && Number.isFinite(value) && value >= 0; }
async function absent(file: string) {
  try { await lstat(file); throw new Error(`Already exists: ${file}`); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
}

/** Reject traversal and every symlink below the real source root, including contained symlinks. */
async function contained(root: string, relative: string): Promise<string> {
  need(typeof relative === "string" && relative.length > 0 && !path.isAbsolute(relative) && !relative.includes("\\")
    && relative.split("/").every((part) => part && part !== "." && part !== ".."), `Unsafe evidence path: ${relative}`);
  let file = await realpath(root);
  for (const part of relative.split("/")) {
    file = path.join(file, part);
    try { need(!(await lstat(file)).isSymbolicLink(), `Symlink evidence path rejected: ${file}`); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
  }
  return file;
}

class Sources {
  entries = new Map<string, Source>();
  hash(file: string): string { const entry = this.entries.get(path.resolve(file)); need(entry && "sha256" in entry, `Unhashed source: ${file}`); return entry.sha256; }
  async bytes(file: string, optional = false, limit = 64 * 1024 * 1024): Promise<Buffer | null> {
    file = path.resolve(file);
    let bytes: Buffer;
    try {
      const before = await lstat(file);
      need(before.isFile() && !before.isSymbolicLink() && before.size <= limit, `Bounded regular file required: ${file}`);
      bytes = await readFile(file);
      const after = await lstat(file);
      need(after.isFile() && !after.isSymbolicLink() && before.ino === after.ino && before.dev === after.dev
        && before.size === after.size && bytes.length === after.size && before.mtimeMs === after.mtimeMs,
      `Source changed while reading: ${file}`);
    } catch (error) {
      if (!optional || (error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      const prior = this.entries.get(file); need(!prior || "missing" in prior, `Source disappeared: ${file}`);
      this.entries.set(file, { path: file, missing: true }); return null;
    }
    const entry = { path: file, bytes: bytes.length, sha256: sha(bytes) }, prior = this.entries.get(file);
    need(!prior || isDeepStrictEqual(prior, entry), `Source changed during audit: ${file}`);
    this.entries.set(file, entry); return bytes;
  }
  async json(file: string): Promise<Obj> { return obj(JSON.parse((await this.bytes(file))!.toString("utf8")), file); }
  async verify() {
    for (const item of [...this.entries.values()]) {
      if ("missing" in item) await this.bytes(item.path, true);
      else await this.bytes(item.path, false, 512 * 1024 * 1024);
    }
  }
}

const ZIP_SCRIPT = String.raw`
import json, sys, zipfile
try:
    with zipfile.ZipFile(sys.argv[1]) as z:
        members = z.infolist()
        total = sum(i.file_size for i in members)
        if len(members) > 100000 or total > 1024 * 1024 * 1024:
            raise ValueError("ZIP audit safety bound exceeded")
        if any(i.flag_bits & 1 for i in members):
            raise ValueError("Encrypted ZIP member")
        names = [i.filename for i in members]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate ZIP member names")
        if not members:
            raise ValueError("Empty ZIP archive")
        bad = z.testzip()
        if bad is not None:
            raise ValueError("ZIP CRC failure: " + bad)
        print(json.dumps(dict(status="passed", members=len(members), uncompressedBytes=total,
            traceEventFiles=sum(n.endswith(".trace") for n in names),
            networkEventFiles=sum(n.endswith(".network") for n in names))))
except Exception as e:
    print(json.dumps(dict(status="invalid", error=type(e).__name__ + ": " + str(e))))
`;
async function zipIntegrity(file: string, python: string): Promise<Obj> {
  try {
    const { stdout } = await invoke(python, ["-I", "-c", ZIP_SCRIPT, file], { timeout: 30000, maxBuffer: 64 * 1024 });
    const result = obj(JSON.parse(stdout), "ZIP integrity result");
    need(["passed", "invalid"].includes(result.status), "Unknown ZIP integrity result");
    return result;
  } catch (error) { return { status: "unverified", error: errorText(error) }; }
}
function screenshotSignature(bytes: Buffer, mime: string): Obj {
  if (mime === "image/png") {
    const valid = bytes.length >= 33 && bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
      && bytes.subarray(12, 16).toString("ascii") === "IHDR" && bytes.readUInt32BE(8) === 13
      && bytes.readUInt32BE(16) > 0 && bytes.readUInt32BE(20) > 0;
    return valid ? { status: "passed", format: "PNG", width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) }
      : { status: "invalid", error: "PNG signature/IHDR absent or invalid" };
  }
  if (mime === "image/jpeg") return bytes.length > 4 && bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255
    && bytes.at(-2) === 255 && bytes.at(-1) === 217 ? { status: "passed", format: "JPEG" }
      : { status: "invalid", error: "JPEG signature/end marker absent" };
  return { status: "unverified", error: `Unsupported screenshot MIME type: ${mime}` };
}

/** Checks recorder consistency, not truthfulness of visible text, completeness, or private-goal fit. */
export function auditBrowserRecords(session: Obj, raw: Obj): Obj {
  const errors: string[] = [], notes: string[] = [];
  const check = (ok: unknown, message: string) => { if (!ok) errors.push(message); };
  const steps: Obj[] = Array.isArray(session.steps) ? session.steps : [];
  check(Array.isArray(session.steps), "Session steps array missing");
  let prior = -1, unknownStepTimes = 0;
  const eventIds = new Set<string>();
  for (const [index, step] of steps.entries()) {
    check(step?.event?.runId === session.run?.id && step?.event?.sequence === index, `Step ${index}: run/sequence differs`);
    check(typeof step?.event?.id === "string" && !eventIds.has(step.event.id), `Step ${index}: event ID missing/duplicate`);
    eventIds.add(step?.event?.id);
    if (step?.elapsedMs === null) unknownStepTimes++;
    else {
      check(finite(step?.elapsedMs) && step.elapsedMs >= prior && step.elapsedMs <= session.elapsedMs, `Step ${index}: nonmonotonic/out-of-session elapsed`);
      if (finite(step?.elapsedMs)) prior = step.elapsedMs;
    }
  }
  check(finite(session.elapsedMs), "Session elapsed is invalid");
  if (session.timingPrecision?.stepElapsed === "measured-monotonic") check(unknownStepTimes === 0, "Measured step timing contains nulls");
  check(session.browser?.actualNavigation ? steps[0]?.action?.type === "navigate" && steps[0]?.action?.url === session.entryUrl : steps.length === 0,
    "Initial navigation record differs from actualNavigation/entry URL");
  check(raw.synthetic === true && Array.isArray(raw.logs), "Synthetic browser logs missing");
  const logs: Obj[] = Array.isArray(raw.logs) ? raw.logs : [];
  let priorLog = -1, priorBrowser = -1, pending: Obj | null = null, agentActions = 0, results = 0, abandonments = 0;
  let confirmations = 0, selected: unknown = null, lastObservation: Obj | null = null, lastResult: Obj | null = null;
  const resultSequences = new Set<number>(), observations: Obj[] = [];
  for (const [index, log] of logs.entries()) {
    if (!log || typeof log !== "object") { errors.push(`Log ${index}: object required`); continue; }
    if (log.elapsedMs !== undefined) {
      check(finite(log.elapsedMs) && log.elapsedMs >= priorLog, `Log ${index}: elapsed time nonmonotonic/invalid`);
      if (finite(log.elapsedMs)) priorLog = log.elapsedMs;
    }
    if (["initial-observation", "observation", "agent-action", "browser-result"].includes(log.type))
      check(finite(log.elapsedMs) && log.elapsedMs <= session.elapsedMs, `Log ${index}: outcome event outside session elapsed`);
    if (log.type === "initial-observation" || log.type === "observation") {
      observations.push(log); lastObservation = log;
      check(typeof log.text === "string" && typeof log.url === "string" && Array.isArray(log.controls), `Log ${index}: observation fields missing`);
      check(Number.isInteger(log.cartItemCount) && log.cartItemCount >= 0 && log.cartEmpty === (log.cartItemCount === 0), `Log ${index}: inconsistent cart state`);
      check(finite(log.observedAtPerformanceMs) && log.observedAtPerformanceMs >= priorBrowser, `Log ${index}: browser clock nonmonotonic`);
      if (finite(log.observedAtPerformanceMs)) priorBrowser = log.observedAtPerformanceMs;
      try { check(new URL(log.url).origin === new URL(session.entryUrl).origin, `Log ${index}: observed origin differs`); }
      catch { errors.push(`Log ${index}: invalid observed/entry URL`); }
    }
    if (log.type === "agent-action") {
      agentActions++; check(!pending, `Log ${index}: next action before prior action completed`);
      pending = log.action;
      if (pending?.type === "abandon") {
        abandonments++;
        check(steps.some((step) => step.action?.type === "abandon" && step.action.target === pending!.reason), "Abandon reason missing from normalized steps");
        pending = null;
      }
    }
    if (log.type === "browser-result") {
      results++; lastResult = log.result;
      check(!!pending, `Log ${index}: browser result lacks pending agent action`);
      const sequence = log.stepSequence, step = steps[sequence], result = log.result;
      check(Number.isInteger(sequence) && sequence > 0 && !resultSequences.has(sequence) && !!step, `Log ${index}: missing/duplicate normalized step`);
      if (step?.elapsedMs !== null) check(finite(step?.elapsedMs) && step.elapsedMs <= log.elapsedMs, `Log ${index}: browser result precedes normalized action`);
      resultSequences.add(sequence);
      if (result?.kind === "search") check(step?.action?.type === "search" && step.action.query === pending?.query, "Search normalization differs");
      else if (["open-cart", "confirm-demo-order"].includes(result?.kind)) check(step?.action?.type === result.kind, "Cart/confirmation normalization differs");
      else if (result?.kind === "add-to-cart" && result.selected && step?.action?.type === "add-to-cart") {
        check(typeof step.action.productId === "string" && typeof step.action.variantId === "string", "Bound cart identity missing");
      } else check(step?.action?.type === "inspect" && step.action.target === result?.target, "Inspection normalization differs");
      if (result?.selected) selected = result.selected;
      if (result?.confirmationVisible === true) confirmations++;
      pending = null;
    }
  }
  check(observations.filter((item) => item.type === "initial-observation").length <= 1, "Multiple initial observations");
  const initial = observations.find((item) => item.type === "initial-observation");
  if (initial) check(initial.cartEmpty === session.isolation?.initialCartEmpty, "Initial cart observation differs from sidecar");
  else if (session.isolation?.initialCartEmpty === true) errors.push("Claimed initial empty cart has no saved observation");
  check(steps.length === (session.browser?.actualNavigation ? 1 : 0) + results + abandonments, "Normalized action count differs from recorded browser results/abandonment");
  check(session.fakeOrderConfirmed === (confirmations > 0), "Sidecar fake confirmation differs from browser results");
  check(isDeepStrictEqual(raw.observedChoice ?? null, selected), "Final recorded cart identity differs from latest browser result");
  if (session.choice) check(["originalName", "catalogPrice", "selectedColor", "quantity"].every((key) => session.choice[key] === (selected as Obj | null)?.[key]),
    "Bound sidecar choice differs from recorded cart identity");
  if (pending) {
    check(["timeout", "technical-failure"].includes(session.status), "Unfinished browser action with a nonfailure outcome");
    notes.push("Last agent action has no completed browser result; retained as pending, not an executed normalized step");
  }
  if (unknownStepTimes) notes.push("Some normalized action times are unrecorded; those times are not reconstructed");
  return { status: errors.length ? "invalid" : unknownStepTimes ? "unverified" : "passed", errors, notes, normalizedSteps: steps.length, unknownStepTimes,
    agentActions, completedBrowserResults: results, abandonments, pendingAction: pending?.type ?? null,
    savedObservations: observations.length, finalState: {
      lastObservation: lastObservation ? { elapsedMs: lastObservation.elapsedMs, url: lastObservation.url,
        cartItemCount: lastObservation.cartItemCount, cartEmpty: lastObservation.cartEmpty } : null,
      lastBrowserResult: lastResult ? { kind: lastResult.kind, confirmationVisible: lastResult.confirmationVisible === true } : null,
      observedChoicePresent: raw.observedChoice != null, visibleConfirmations: confirmations,
      note: "Last observation may precede the final action. Confirmation is a saved browser-result claim; pixels and server orders are independently unverified here." } };
}

async function auditRow(row: Obj, reportRoot: string, sources: Sources, python: string): Promise<Obj> {
  const result: Obj = { caseId: row.caseId, arm: row.arm, runId: row.expectedRunId, declaredStatus: row.status,
    status: row.status === "unattempted" ? "unattempted" : "invalid", errors: [], artifacts: [] };
  try {
    const sidecarPath = await contained(reportRoot, row.path), bytes = await sources.bytes(sidecarPath, true);
    if (!bytes) {
      result.sidecar = { status: "missing" };
      if (row.status !== "unattempted") { result.status = "missing"; result.errors.push("Declared sidecar does not exist"); }
      return result;
    }
    result.sidecar = { status: "present", sha256: sha(bytes), bytes: bytes.length };
    if (row.status === "unattempted") { result.status = "invalid"; result.errors.push("Unattempted slot has an actual sidecar"); return result; }
    const session = obj(JSON.parse(bytes.toString("utf8")), "session");
    result.sidecar.hashMatchesReport = HASH.test(row.sidecarSha256) && sha(bytes) === row.sidecarSha256;
    result.sidecar.embeddedMatches = isDeepStrictEqual(session, row.session);
    if (!result.sidecar.hashMatchesReport) result.errors.push("Sidecar SHA differs/missing in report");
    if (row.status === "invalid-sidecar") {
      result.errors.push("Source report declares invalid sidecar; no promotion or outcome reinterpretation"); return result;
    }
    if (!result.sidecar.embeddedMatches) result.errors.push("Actual sidecar differs from report embedded session");
    need(session.contractVersion === "shopper-session.v1" && session.caseId === row.caseId && session.arm === row.arm
      && session.run?.id === row.expectedRunId && row.outcome === session.status && session.realPurchase === false,
    "Sidecar identity/contract/outcome differs from fixed roster");
    need(Array.isArray(session.artifacts), "Session artifacts array missing");
    const seenPaths = new Set<string>(), seenIds = new Set<string>();
    let raw: Obj | null = null;
    for (const receipt of session.artifacts) {
      const artifact = obj(receipt.artifact, "artifact"), entry: Obj = { id: artifact.id, path: artifact.path,
        kind: artifact.kind, declaredStatus: receipt.status, status: "invalid", errors: [] };
      result.artifacts.push(entry);
      try {
        need(artifact.runId === row.expectedRunId && typeof artifact.id === "string" && !seenIds.has(artifact.id)
          && !seenPaths.has(artifact.path), "Artifact identity/path duplicated or differs from session");
        seenIds.add(artifact.id); seenPaths.add(artifact.path);
        const file = await contained(path.dirname(sidecarPath), artifact.path);
        const actual = await sources.bytes(file, true, 512 * 1024 * 1024);
        entry.presence = actual ? "present" : "missing";
        if (receipt.status !== "captured") {
          need(["missing", "failed"].includes(receipt.status) && receipt.sha256 === null && receipt.bytes === 0
            && typeof receipt.error === "string" && receipt.error.length > 0, "Invalid declared missing/failed artifact receipt");
          entry.status = actual ? "invalid" : "missing";
          if (actual) entry.errors.push("Uncaptured artifact exists without a captured SHA receipt");
          continue;
        }
        if (!actual) { entry.status = "missing"; entry.errors.push("Captured artifact missing"); continue; }
        entry.actualSha256 = sha(actual); entry.actualBytes = actual.length;
        entry.byteHashVerified = HASH.test(receipt.sha256) && receipt.sha256 === entry.actualSha256
          && receipt.bytes === actual.length && actual.length > 0 && receipt.error === null;
        need(entry.byteHashVerified, "Captured artifact SHA/byte count/receipt differs");
        entry.status = "passed";
        if (artifact.kind === "trace") {
          entry.zipIntegrity = await zipIntegrity(file, python);
          if (entry.zipIntegrity.status !== "passed") entry.status = entry.zipIntegrity.status;
        }
        if (artifact.kind === "screenshot") {
          entry.signature = screenshotSignature(actual, artifact.mimeType);
          if (entry.signature.status !== "passed") entry.status = entry.signature.status;
        }
        if (artifact.path === "browser-observations.json") raw = obj(JSON.parse(actual.toString("utf8")), "browser observations");
      } catch (error) { entry.status = "invalid"; entry.errors.push(errorText(error)); }
    }
    result.missingRequiredKinds = ["trace", "screenshot"].filter((kind) => !result.artifacts.some((item: Obj) => item.kind === kind && item.status === "passed"));
    result.browserRecords = raw ? auditBrowserRecords(session, raw) : { status: "missing", errors: ["Captured matching browser-observations.json unavailable"] };
    const states = [...result.artifacts.map((item: Obj) => item.status), result.browserRecords.status];
    result.status = result.errors.length || states.includes("invalid") ? "invalid"
      : states.includes("missing") || result.missingRequiredKinds.length ? "missing" : states.includes("unverified") ? "unverified" : "passed";
  } catch (error) { result.status = "invalid"; result.errors.push(errorText(error)); }
  return result;
}

/** Final receipt + absent dispatcher lock are mandatory; never accepts live checkpoints. */
export async function auditFinalArtifacts(options: AuditOptions) {
  need(options.packets.length === 6 && new Set(options.packets.map((file) => path.resolve(file))).size === 6, "Six distinct frozen packets required");
  await absent(options.out);
  const cohortPath = path.resolve(options.cohortResult), root = await realpath(path.dirname(cohortPath));
  const lock = path.join(root, "active-run-lock.json"); await absent(lock);
  const sources = new Sources(), cohort = await sources.json(cohortPath);
  need(cohort.version === "shopper-cohort-dispatch.v1" && cohort.synthetic === true && cohort.humanConversionClaim === false
    && cohort.fixedPairs === 60 && cohort.fixedSessions === 120 && Array.isArray(cohort.schedule) && cohort.schedule.length === 120
    && Array.isArray(cohort.aggregateFiles) && cohort.aggregateFiles.length === 6, "Final fixed120 synthetic dispatcher receipt required");
  need(["roster-finished", "deadline", "requested-stop", "SIGINT", "SIGTERM"].includes(cohort.stopReason), "Unrecognized final dispatch stop reason");
  const frozenPlanPath = await contained(root, "cohort-plan.json"), frozenPlan = await sources.json(frozenPlanPath);
  need(HASH.test(cohort.planSha256) && cohort.planSha256 === sources.hash(frozenPlanPath)
    && frozenPlan.version === "shopper-fixed-cohort.v1" && frozenPlan.synthetic === true && frozenPlan.fixedPairs === 60
    && frozenPlan.fixedSessions === 120 && Array.isArray(frozenPlan.schedule) && frozenPlan.schedule.length === 120
    && Array.isArray(frozenPlan.packets) && frozenPlan.packets.length === 6, "Final dispatcher receipt does not bind its frozen fixed120 plan");
  const roster = new Map<string, Obj>(), cases = new Set<string>(), rows: Obj[] = [], reports: Obj[] = [];
  for (const slot of cohort.schedule) {
    need(ID.test(slot.caseId) && ID.test(slot.runId) && ID.test(slot.packetKey) && ARMS.includes(slot.arm)
      && ["recorded", "unattempted"].includes(slot.status) && !roster.has(slot.runId), "Invalid/duplicate dispatch roster slot");
    roster.set(slot.runId, slot);
  }
  need(frozenPlan.schedule.every((slot: Obj, index: number) => {
    const { status, ...actual } = cohort.schedule[index]; return isDeepStrictEqual(actual, slot);
  }), "Final dispatcher schedule differs from frozen cohort plan");
  need(cohort.recordedReceipts === cohort.schedule.filter((slot: Obj) => slot.status === "recorded").length
    && cohort.unattempted === 120 - cohort.recordedReceipts, "Final dispatcher presence counts differ from roster");
  const seenRuns = new Set<string>(), seenPacketKeys = new Set<string>();
  for (const [index, checkpoint] of cohort.aggregateFiles.entries()) {
    const relative = obj(checkpoint, "Final aggregate entry").file;
    need(typeof relative === "string" && /^final-[^/]+\.json$/.test(path.basename(relative)), "Only final dispatcher aggregates accepted");
    const reportPath = await contained(root, relative), report = await sources.json(reportPath), packetRoot = path.resolve(options.packets[index]);
    const manifestPath = path.join(packetRoot, "manifest.json"), planPath = path.join(packetRoot, "pair-plan.json");
    const manifest = await sources.json(manifestPath), plan = await sources.json(planPath);
    need(manifest.frozen === true && manifest.synthetic === true && manifest.files?.["pair-plan.json"] === sources.hash(planPath),
      "Frozen packet manifest/pair-plan SHA differs");
    need(report.version === "shopper-evidence-aggregate.v1" && report.synthetic === true && report.humanConversionClaim === false
      && report.fixedCases === 10 && report.fixedSessions === 20 && Array.isArray(report.rows) && report.rows.length === 20
      && Array.isArray(plan.schedule) && plan.schedule.length === 10, "Final report/packet must preserve fixed20 slots");
    need(report.packetSources?.["manifest.json"] === sources.hash(manifestPath)
      && report.packetSources?.["pair-plan.json"] === sources.hash(planPath), "Report packet SHA binding differs");
    const packetKey = path.basename(path.dirname(reportPath)); need(!seenPacketKeys.has(packetKey), "Repeated packet result root"); seenPacketKeys.add(packetKey);
    const frozenPacket = frozenPlan.packets[index];
    need(frozenPacket.key === packetKey && path.resolve(frozenPacket.directory) === packetRoot
      && frozenPacket.files?.["manifest.json"] === sources.hash(manifestPath)
      && frozenPacket.files?.["pair-plan.json"] === sources.hash(planPath), "Packet path/manifest/roster differs from frozen cohort plan");
    const wanted = new Map<string, Obj>();
    for (const pair of plan.schedule) {
      need(ID.test(pair.caseId) && !cases.has(pair.caseId) && Array.isArray(pair.runs) && pair.runs.length === 2
        && new Set(pair.runs.map((run: Obj) => run.arm)).size === 2, "Invalid/repeated paired scenario"); cases.add(pair.caseId);
      for (const run of pair.runs) {
        need(ID.test(run.runId) && ARMS.includes(run.arm) && !wanted.has(run.runId), "Invalid/repeated frozen run ID");
        wanted.set(run.runId, { caseId: pair.caseId, arm: run.arm });
      }
    }
    for (const row of report.rows) {
      const expected = wanted.get(row.expectedRunId), slot = roster.get(row.expectedRunId);
      need(expected && slot && !seenRuns.has(row.expectedRunId) && row.caseId === expected.caseId && row.arm === expected.arm
        && slot.packetKey === packetKey && slot.caseId === row.caseId && slot.arm === row.arm
        && row.path === `${row.caseId}/${row.arm}/session.json` && ["recorded", "unattempted", "invalid-sidecar"].includes(row.status)
        && (row.status === "unattempted") === (slot.status === "unattempted"), "Final report differs from full frozen dispatch roster");
      seenRuns.add(row.expectedRunId);
      rows.push({ packetKey, report: reportPath, ...await auditRow(row, path.dirname(reportPath), sources, options.python ?? "python3") });
    }
    reports.push({ path: reportPath, sha256: sources.hash(reportPath), packet: packetRoot });
  }
  need(rows.length === 120 && cases.size === 60 && seenRuns.size === 120, "Full120 roster was not preserved");
  await sources.verify(); await absent(lock);
  const statuses = ["passed", "missing", "invalid", "unverified", "unattempted"];
  const count = (items: Obj[]) => Object.fromEntries(statuses.map((status) => [status, items.filter((item) => item.status === status).length]));
  const artifacts = rows.flatMap((row) => row.artifacts);
  const result = { schema: "shopper-final-artifact-audit.v1", synthetic: true, humanConversionClaim: false,
    createdAt: new Date().toISOString(), utilitySha256: sha(await readFile(fileURLToPath(import.meta.url))),
    fixedPairs: 60, fixedSessions: 120, reports, sourceFiles: [...sources.entries.values()],
    summary: { rows: count(rows), artifacts: count(artifacts), capturedByteHashVerified: artifacts.filter((item) => item.byteHashVerified).length,
      traceZipIntegrity: count(artifacts.filter((item) => item.kind === "trace").map((item) => ({ status: item.zipIntegrity?.status ?? item.status }))),
      screenshotSignatures: count(artifacts.filter((item) => item.kind === "screenshot").map((item) => ({ status: item.signature?.status ?? item.status }))),
      browserRecords: count(rows.filter((row) => row.declaredStatus === "recorded").map((row) => ({ status: row.browserRecords?.status ?? row.status }))) },
    limitations: ["Synthetic agent evidence; no human conversion lift or sales claim.",
      "SHA/byte checks establish saved file identity, ZIP CRC establishes archive readability, screenshot signatures establish format markers only.",
      "No pixel interpretation, full image decoding, trace event semantic replay, trace completeness, server order verification, or private-goal rescoring is performed.",
      "Browser records are consistency checks on saved recorder claims; last observations may precede final actions. Pending failed actions are not treated as completed actions.",
      "Final dispatch receipt and absent active lock gate execution; this is a source-byte snapshot, not proof that the original recorder or host was trustworthy."], rows };
  await writeFile(options.out, encode(result), { flag: "wx" }); return result;
}

async function selfTest() {
  const directory = await mkdtemp(path.join(tmpdir(), "shopper-artifact-audit-"));
  try {
    const packets: string[] = [], frozenPackets: Obj[] = [], schedule: Obj[] = [], aggregateFiles: Obj[] = [];
    const json = async (file: string, value: unknown) => { await mkdir(path.dirname(file), { recursive: true }); await writeFile(file, encode(value)); };
    let sample: Obj | undefined, sampleSession: Obj | undefined, sampleRoot = "";
    for (let index = 0; index < 6; index++) {
      const packetKey = `packet-${index + 1}`, packet = path.join(directory, `frozen-${index}`); packets.push(packet);
      const plan = { schedule: Array.from({ length: 10 }, (_, pair) => ({ caseId: `c-${index}-${pair}`,
        runs: ARMS.map((arm) => ({ arm, runId: `r-${index}-${pair}-${arm}` })) })) };
      await json(path.join(packet, "pair-plan.json"), plan);
      const planSha = sha(await readFile(path.join(packet, "pair-plan.json")));
      await json(path.join(packet, "manifest.json"), { frozen: true, synthetic: true, files: { "pair-plan.json": planSha } });
      frozenPackets.push({ key: packetKey, directory: packet,
        files: { "manifest.json": sha(await readFile(path.join(packet, "manifest.json"))), "pair-plan.json": planSha } });
      const rows = plan.schedule.flatMap((pair) => pair.runs.map((run) => {
        const row: Obj = { caseId: pair.caseId, arm: run.arm, expectedRunId: run.runId, path: `${pair.caseId}/${run.arm}/session.json`, status: "unattempted" };
        schedule.push({ packetKey, caseId: pair.caseId, ...run, status: "unattempted" }); return row;
      }));
      if (index === 0) {
        sample = rows[0]; sample.status = "recorded"; sample.outcome = "abandoned"; schedule[0].status = "recorded";
        sampleRoot = path.join(directory, packetKey, sample.caseId, sample.arm); await mkdir(sampleRoot, { recursive: true });
        const runId = sample.expectedRunId, entryUrl = "http://127.0.0.1:4312/?arm=generic";
        const steps = [{ event: { id: `${runId}-0`, runId, sequence: 0 }, elapsedMs: 1, action: { type: "navigate", url: entryUrl } },
          { event: { id: `${runId}-1`, runId, sequence: 1 }, elapsedMs: 4, action: { type: "abandon", target: "Required stock unavailable" } }];
        const raw = { synthetic: true, observedChoice: null, logs: [
          { type: "initial-observation", elapsedMs: 2, observedAtPerformanceMs: 100, text: "Store", url: entryUrl, controls: [], cartItemCount: 0, cartEmpty: true },
          { type: "agent-action", elapsedMs: 3, action: { type: "abandon", reason: "Required stock unavailable" } }] };
        await json(path.join(sampleRoot, "browser-observations.json"), raw);
        await invoke("python3", ["-I", "-c", "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1],'w'); z.writestr('trace.trace',b'{}'); z.close()", path.join(sampleRoot, "trace.zip")]);
        const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a53sAAAAASUVORK5CYII=", "base64");
        await writeFile(path.join(sampleRoot, "screenshot.png"), png);
        const artifacts = [];
        for (const [name, kind, mimeType] of [["trace.zip", "trace", "application/zip"], ["screenshot.png", "screenshot", "image/png"], ["browser-observations.json", "result", "application/json"]]) {
          const bytes = await readFile(path.join(sampleRoot, name)); artifacts.push({ artifact: { id: name, runId, path: name, kind, mimeType }, status: "captured", sha256: sha(bytes), bytes: bytes.length, error: null });
        }
        sampleSession = { contractVersion: "shopper-session.v1", caseId: sample.caseId, arm: sample.arm, run: { id: runId }, status: "abandoned", realPurchase: false,
          browser: { actualNavigation: true }, entryUrl, elapsedMs: 5, timingPrecision: { stepElapsed: "measured-monotonic" },
          isolation: { initialCartEmpty: true }, steps, fakeOrderConfirmed: false, choice: null, artifacts };
        await json(path.join(sampleRoot, "session.json"), sampleSession); sample.session = sampleSession; sample.sidecarSha256 = sha(await readFile(path.join(sampleRoot, "session.json")));
      }
      const relative = `${packetKey}/final-test.json`; aggregateFiles.push({ file: relative });
      await json(path.join(directory, relative), { version: "shopper-evidence-aggregate.v1", synthetic: true, humanConversionClaim: false,
        fixedCases: 10, fixedSessions: 20, packetSources: { "pair-plan.json": planSha, "manifest.json": sha(await readFile(path.join(packet, "manifest.json"))) }, rows });
    }
    const cohortResult = path.join(directory, "cohort-result-test.json");
    await json(path.join(directory, "cohort-plan.json"), { version: "shopper-fixed-cohort.v1", synthetic: true,
      fixedPairs: 60, fixedSessions: 120, packets: frozenPackets, schedule: schedule.map(({ status, ...slot }) => slot) });
    await json(cohortResult, { version: "shopper-cohort-dispatch.v1", synthetic: true, humanConversionClaim: false,
      fixedPairs: 60, fixedSessions: 120, schedule, aggregateFiles, stopReason: "deadline", recordedReceipts: 1, unattempted: 119,
      planSha256: sha(await readFile(path.join(directory, "cohort-plan.json"))) });
    const out = path.join(directory, "audit.json"), result = await auditFinalArtifacts({ cohortResult, packets, out });
    assert.equal(result.rows.length, 120); assert.equal(result.summary.rows.passed, 1); assert.equal(result.summary.rows.unattempted, 119);
    assert.equal(result.summary.traceZipIntegrity.passed, 1); assert.equal(result.summary.screenshotSignatures.passed, 1);
    await assert.rejects(auditFinalArtifacts({ cohortResult, packets, out }), /Already exists/);
    await writeFile(path.join(directory, "active-run-lock.json"), "{}");
    await assert.rejects(auditFinalArtifacts({ cohortResult, packets, out: out + "-active" }), /Already exists/);
    await rm(path.join(directory, "active-run-lock.json"));
    const raw = JSON.parse(await readFile(path.join(sampleRoot, "browser-observations.json"), "utf8"));
    const nonmonotonic = structuredClone(raw); nonmonotonic.logs[1].elapsedMs = 1;
    assert.equal(auditBrowserRecords(sampleSession!, nonmonotonic).status, "invalid");
    const pending = structuredClone(raw); pending.logs[1].action = { type: "click", ref: "button-1" };
    const timeout = { ...sampleSession!, status: "timeout", steps: sampleSession!.steps.slice(0, 1) };
    assert.equal(auditBrowserRecords(timeout, pending).status, "passed");
    assert.equal(auditBrowserRecords({ ...timeout, status: "completed" }, pending).status, "invalid");
    const chosen = { originalName: "Fixture bag", catalogPrice: 10, selectedColor: "blue", quantity: 1 };
    const completed = structuredClone(sampleSession!); completed.status = "completed"; completed.elapsedMs = 30;
    completed.fakeOrderConfirmed = true; completed.choice = { ...chosen, productId: "fixture", variantId: "fixture-blue" };
    completed.steps = completed.steps.slice(0, 1);
    const completedRaw = { synthetic: true, observedChoice: chosen, logs: [raw.logs[0]] as Obj[] };
    for (const [index, kind] of ["search", "add-to-cart", "open-cart", "confirm-demo-order"].entries()) {
      const sequence = index + 1, elapsedMs = 4 + index * 5;
      const action = kind === "search" ? { type: "search", query: "bag" } : { type: "click", ref: `fixture-${index}` };
      const normalized = kind === "search" ? action : kind === "add-to-cart" ? { type: kind, productId: "fixture", variantId: "fixture-blue" } : { type: kind, target: kind };
      completed.steps.push({ event: { id: `${completed.run.id}-${sequence}`, runId: completed.run.id, sequence }, elapsedMs: elapsedMs + 1, action: normalized });
      completedRaw.logs.push({ type: "agent-action", elapsedMs, action }, { type: "browser-result", elapsedMs: elapsedMs + 2, stepSequence: sequence,
        result: { kind, target: kind, ...(index > 0 ? { selected: chosen } : {}), confirmationVisible: index === 3 } });
    }
    assert.equal(auditBrowserRecords(completed, completedRaw).status, "passed");
    const falseConfirm = structuredClone(completedRaw); falseConfirm.logs.at(-1)!.result.confirmationVisible = false;
    assert.equal(auditBrowserRecords(completed, falseConfirm).status, "invalid");
    const wrongChoice = structuredClone(completedRaw); wrongChoice.observedChoice = { ...chosen, selectedColor: "red" };
    assert.equal(auditBrowserRecords(completed, wrongChoice).status, "invalid");
    const duplicateResult = structuredClone(completedRaw); duplicateResult.logs.at(-1)!.stepSequence = 2;
    assert.equal(auditBrowserRecords(completed, duplicateResult).status, "invalid");
    assert.equal(screenshotSignature(Buffer.from("not an image"), "image/png").status, "invalid");
    const badZip = path.join(directory, "bad.zip"); await writeFile(badZip, "not a zip");
    assert.equal((await zipIntegrity(badZip, "python3")).status, "invalid");
    assert.equal((await zipIntegrity(badZip, path.join(directory, "missing-python"))).status, "unverified");
    const archive = await readFile(path.join(sampleRoot, "trace.zip")); archive[archive.indexOf(Buffer.from("{}"))] ^= 1;
    await writeFile(badZip, archive); assert.equal((await zipIntegrity(badZip, "python3")).status, "invalid");
    await symlink(path.join(sampleRoot, "screenshot.png"), path.join(sampleRoot, "alias.png"));
    await assert.rejects(contained(sampleRoot, "alias.png"), /Symlink/); await assert.rejects(contained(sampleRoot, "../escape"), /Unsafe/);
    const altered = structuredClone(sample!); altered.session.elapsedMs = 999;
    assert.equal((await auditRow(altered, path.join(directory, "packet-1"), new Sources(), "python3")).status, "invalid");
    const missing = structuredClone(sample!); missing.path = `${missing.caseId}/${missing.arm}/missing.json`;
    assert.equal((await auditRow(missing, path.join(directory, "packet-1"), new Sources(), "python3")).status, "missing");
    const unattempted = { ...sample!, status: "unattempted" };
    assert.equal((await auditRow(unattempted, path.join(directory, "packet-1"), new Sources(), "python3")).status, "invalid");
    const missingArtifact = structuredClone(sampleSession!); missingArtifact.artifacts[0].artifact.path = "missing-trace.zip";
    const missingRow: Obj = { ...sample!, session: missingArtifact }; await json(path.join(sampleRoot, "session.json"), missingArtifact);
    missingRow.sidecarSha256 = sha(await readFile(path.join(sampleRoot, "session.json")));
    assert.equal((await auditRow(missingRow, path.join(directory, "packet-1"), new Sources(), "python3")).status, "missing");
    await json(path.join(sampleRoot, "session.json"), sampleSession);
    const hashMismatch = structuredClone(sampleSession!); hashMismatch.artifacts[0].sha256 = "0".repeat(64);
    const hashRow: Obj = { ...sample!, session: hashMismatch }; await json(path.join(sampleRoot, "session.json"), hashMismatch);
    hashRow.sidecarSha256 = sha(await readFile(path.join(sampleRoot, "session.json")));
    assert.equal((await auditRow(hashRow, path.join(directory, "packet-1"), new Sources(), "python3")).status, "invalid");
    await json(path.join(sampleRoot, "session.json"), sampleSession);
    const finalReceipt = JSON.parse(await readFile(cohortResult, "utf8")); finalReceipt.schedule[0].runId = "wrong-roster-id";
    await json(cohortResult, finalReceipt);
    await assert.rejects(auditFinalArtifacts({ cohortResult, packets, out: out + "-bad-roster" }), /schedule differs/);
    assert.equal(await lstat(out + "-bad-roster").catch(() => null), null);
    const tracker = new Sources(); await tracker.bytes(badZip); await writeFile(badZip, "changed"); await assert.rejects(tracker.verify(), /changed/);
    console.log("artifact-audit self-test passed: temporary fixed120 roster; 119 explicit unattempted; sidecar match; ZIP CRC/corruption; PNG signature; pending timeout; monotonicity; symlinks; missing evidence; drift; exclusive output; active-lock refusal");
  } finally { await rm(directory, { recursive: true, force: true }); }
}

export async function main(args: string[]) {
  if (args.length === 1 && args[0] === "--self-test") return selfTest();
  const options: AuditOptions = { cohortResult: "", packets: [], out: "" };
  while (args.length) {
    const key = args.shift(), value = args.shift(); need(value && !value.startsWith("--"), "Every option requires a value");
    if (key === "--packet") options.packets.push(path.resolve(value));
    else if (key === "--cohort-result" && !options.cohortResult) options.cohortResult = path.resolve(value);
    else if (key === "--out" && !options.out) options.out = path.resolve(value);
    else if (key === "--python" && !options.python) options.python = value;
    else throw new Error(`Unknown/duplicate option: ${key}`);
  }
  need(options.cohortResult && options.out, "--cohort-result FINAL.json, six --packet paths, and --out NEW.json required");
  const result = await auditFinalArtifacts(options); console.log(encode({ output: options.out, summary: result.summary }));
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main(process.argv.slice(2)).catch((error) => { console.error(errorText(error)); process.exitCode = 1; });
