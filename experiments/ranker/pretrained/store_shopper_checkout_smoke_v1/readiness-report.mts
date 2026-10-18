/** Result-side evidence audit only. No browser, agent, model, purchase or network calls. */
import { createHash } from "node:crypto";
import { readFile, realpath, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { verifyShopperArtifactFiles, type ShopperArtifact } from "../../../../apps/runner/src/shopper-contract.ts";

type Obj = Record<string, any>;
const ARMS = ["generic", "local-model"] as const;
const STAGES = ["rank", "firstThree", "complete"] as const;
const HASH = /^[a-f0-9]{64}$/;
const sha = (bytes: Buffer | string) => createHash("sha256").update(bytes).digest("hex");
function requireValue(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }
function object(value: unknown, label: string): Obj { requireValue(value && typeof value === "object" && !Array.isArray(value), `${label}: object required`); return value as Obj; }
function list(value: unknown, label: string): any[] { requireValue(Array.isArray(value), `${label}: array required`); return value; }
function text(value: unknown, label: string): string { requireValue(typeof value === "string" && value.length > 0, `${label}: text required`); return value; }
function bool(value: unknown, label: string): boolean { requireValue(typeof value === "boolean", `${label}: boolean required`); return value; }
function number(value: unknown, label: string): number { requireValue(typeof value === "number" && Number.isFinite(value) && value >= 0, `${label}: nonnegative finite number required`); return value; }
function integer(value: unknown, label: string): number { const result = number(value, label); requireValue(Number.isSafeInteger(result), `${label}: whole nonnegative number required`); return result; }
function digest(value: unknown, label: string): string { requireValue(typeof value === "string" && HASH.test(value), `${label}: SHA-256 required`); return value; }
async function json(file: string) {
  requireValue((await stat(file)).size <= 16 * 1024 * 1024, "JSON exceeds 16 MiB");
  const bytes = await readFile(file); return { value: JSON.parse(bytes.toString("utf8")), sha256: sha(bytes), path: path.resolve(file) };
}
async function contained(root: string, relative: string) {
  requireValue(!path.isAbsolute(relative), "Evidence path must be relative to report root");
  const target = path.resolve(root, relative); requireValue(target.startsWith(path.resolve(root) + path.sep), "Evidence path escapes report root");
  const [actualRoot, actualTarget] = await Promise.all([realpath(root), realpath(target)]);
  requireValue(actualTarget.startsWith(actualRoot + path.sep), "Evidence symlink escapes report root"); return actualTarget;
}
function summary(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const q = (p: number) => { if (!sorted.length) return null; const at = (sorted.length - 1) * p, low = Math.floor(at), high = Math.ceil(at); return sorted[low] + (sorted[high] - sorted[low]) * (at - low); };
  return { count: values.length, minimum: sorted[0] ?? null, median: q(.5), p95: q(.95), maximum: sorted.at(-1) ?? null };
}
function counts(values: string[]) { return Object.fromEntries([...new Set(values)].sort().map((value) => [value, values.filter((v) => v === value).length])); }
function checkedArtifacts(value: unknown): ShopperArtifact[] {
  const artifacts = list(value, "session artifacts");
  const ids = new Set<string>();
  for (const value of artifacts) {
    const receipt = object(value, "artifact receipt"), artifact = object(receipt.artifact, "artifact");
    const id = text(artifact.id, "artifact ID"); requireValue(!ids.has(id), "Duplicate artifact ID"); ids.add(id);
    text(artifact.kind, "artifact kind"); text(artifact.path, "artifact path");
    requireValue(["captured", "missing", "failed"].includes(receipt.status), "Unknown artifact capture status");
    integer(receipt.bytes, "artifact bytes");
    if (receipt.status === "captured") digest(receipt.sha256, "captured artifact digest");
    else requireValue(receipt.sha256 === null, "Uncaptured artifact cannot claim saved-byte digest");
  }
  return artifacts as ShopperArtifact[];
}
function checkedMeasure(value: unknown, session: Obj, artifactChecks: Obj[]): Obj {
  const measure = object(value, "readiness measure");
  requireValue(STAGES.includes(measure.stage) && measure.name === `fieldwork.local-model.${measure.stage}.after-paint-opportunity`, "Readiness stage/name mismatch");
  text(measure.requestId, "request ID"); number(measure.startTimeMs, "startTimeMs"); number(measure.durationMs, "durationMs");
  if (measure.capturedAtElapsedMs !== null) {
    const captured = number(measure.capturedAtElapsedMs, "capturedAtElapsedMs");
    requireValue(measure.durationMs <= captured && captured <= session.elapsedMs, "Readiness capture outside session timeline");
  }
  requireValue(measure.scope === "after-paint-opportunity" && measure.physicalPaintMeasured === false && measure.imageTransferMeasured === false, "Unsupported latency scope claim");
  requireValue(["performance-measure", "data-local-model-readiness"].includes(measure.source), "Unknown readiness source");
  text(measure.artifactId, "readiness artifact ID");
  const artifact = artifactChecks.find((a) => a.artifactId === measure.artifactId); requireValue(artifact, "Readiness refers to undeclared artifact");
  if (measure.stage === "rank") requireValue(measure.acceptedFields === null && measure.withheldFields === null && measure.requiredFields === null, "Rank is not generated-copy field evidence");
  else {
    for (const field of ["acceptedFields", "withheldFields", "requiredFields"]) integer(measure[field], field);
    requireValue(measure.acceptedFields + measure.withheldFields === measure.requiredFields && measure.requiredFields <= (measure.stage === "firstThree" ? 10 : 20), "Copy field counts contradict required fields");
  }
  return { ...measure, artifactVerified: artifact.verified === true, comparableCaptureClock: typeof measure.capturedAtElapsedMs === "number" };
}
async function observedViewport(artifacts: ShopperArtifact[], checks: Obj[], directory: string, note: string) {
  const declared = /\b(\d+)x(\d+)\s+viewport\b/.exec(note);
  const rows: Obj[] = [], screenshotDimensions: Obj[] = [];
  for (const artifact of artifacts) {
    if (artifact.artifact.kind === "screenshot" && checks.some((c) => c.artifactId === artifact.artifact.id && c.verified)) {
      const bytes = await readFile(await contained(directory, artifact.artifact.path));
      if (bytes.length >= 24 && bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) && bytes.toString("ascii", 12, 16) === "IHDR") {
        const width = bytes.readUInt32BE(16), height = bytes.readUInt32BE(20); requireValue(width > 0 && height > 0, "Invalid PNG dimensions");
        screenshotDimensions.push({ artifactId: artifact.artifact.id, width, height, source: "verified saved PNG IHDR header; screenshot pixels, not measured DOM width" });
      }
    }
    if (path.basename(artifact.artifact.path) !== "browser-observations.json" || !checks.some((c) => c.artifactId === artifact.artifact.id && c.verified)) continue;
    const input = await json(await contained(directory, artifact.artifact.path)), evidence = object(input.value, "browser observations");
    requireValue(evidence.synthetic === true, "Synthetic browser observations required");
    for (const log of list(evidence.logs, "observation logs")) {
      if (!["initial-observation", "observation"].includes(log?.type) || !log.viewport) continue;
      const viewport = object(log.viewport, "observed viewport");
      rows.push({ artifactId: artifact.artifact.id, width: viewport.width === undefined ? null : number(viewport.width, "viewport width"),
        height: number(viewport.height, "viewport height"), scrollY: number(viewport.scrollY, "scrollY"),
        documentHeight: number(viewport.documentHeight, "documentHeight") });
    }
  }
  return { declared: declared ? { width: Number(declared[1]), height: Number(declared[2]), source: "session isolation note" } : null,
    observed: rows, widthMeasured: rows.some((r) => r.width !== null), recordedScreenshotDimensions: screenshotDimensions,
    scope: "Declared setup, saved screenshot pixels and artifact-bound observed DOM dimensions remain distinct" };
}
async function inspectRow(value: unknown, root: string) {
  const row = object(value, "aggregate row"), runId = text(row.expectedRunId, "expected run ID"), caseId = text(row.caseId, "case ID");
  requireValue(ARMS.includes(row.arm) && ["recorded", "unattempted", "invalid-sidecar"].includes(row.status), "Invalid arm or slot status");
  const base = { runId, caseId, arm: row.arm, slotStatus: row.status, errors: list(row.errors, "row errors") };
  if (row.status !== "recorded") return { ...base, sessionVerified: false, modelStatus: "not-recorded", session: null, measures: [] };
  const sidecarPath = await contained(root, text(row.path, "session path")), sidecar = await json(sidecarPath), session = object(sidecar.value, "session");
  requireValue(sidecar.sha256 === digest(row.sidecarSha256, "sidecar digest") && isDeepStrictEqual(session, row.session), "Raw session hash or embedded session differs");
  requireValue(session.contractVersion === "shopper-session.v1" && session.realPurchase === false && session.caseId === caseId && session.arm === row.arm && session.run?.id === runId && row.outcome === session.status, "Session identity/synthetic/outcome mismatch");
  const elapsedMs = number(session.elapsedMs, "elapsedMs"), timing = object(session.timingPrecision, "timing precision");
  requireValue(["measured-monotonic", "estimated", "upper-bound"].includes(timing.sessionElapsed), "Unknown session timing precision");
  requireValue(["measured-monotonic", "partially-recorded", "not-recorded"].includes(timing.stepElapsed), "Unknown step timing precision");
  const artifacts = checkedArtifacts(session.artifacts), checks = await verifyShopperArtifactFiles(artifacts, path.dirname(sidecarPath));
  const artifactChecks = artifacts.map((a, index) => ({ artifactId: a.artifact.id, kind: a.artifact.kind, path: a.artifact.path,
    captureStatus: a.status, sha256: a.sha256, bytes: a.bytes, ...checks[index] }));
  const model = object(session.model, "model"), isolation = object(session.isolation, "isolation"), fallback = object(session.fallback, "fallback");
  requireValue(["not-enabled", "complete", "partial", "failed", "not-observed"].includes(model.status), "Unknown model status");
  const fallbackUsed = bool(fallback.used, "fallback used");
  requireValue(["none", "last-safe-model", "original-catalog", "service-unavailable"].includes(fallback.kind) && fallbackUsed === (fallback.kind !== "none"), "Contradictory fallback state");
  if (fallbackUsed) text(fallback.reason, "fallback reason"); else requireValue(fallback.reason === null, "Unused fallback reason must be null");
  const measures = list(model.readiness, "readiness").map((m) => checkedMeasure(m, session, artifactChecks));
  const seen = new Map<string, Obj>();
  for (const measure of measures) {
    const key = `${measure.requestId}/${measure.stage}`, previous = seen.get(key);
    requireValue(!previous || isDeepStrictEqual(previous, measure), "Conflicting repeated readiness request/stage");
    seen.set(key, measure);
  }
  requireValue(row.arm !== "generic" || (model.status === "not-enabled" && measures.length === 0), "Generic session cannot claim local-model readiness");
  requireValue(model.status !== "complete" || measures.some((m) => m.stage === "complete"), "Complete model status lacks complete-stage evidence");
  const measuredTotalRuntimeMs = timing.sessionElapsed === "measured-monotonic" && typeof session.startedAt === "string" && Number.isFinite(Date.parse(session.startedAt)) ? elapsedMs : null;
  const viewport = await observedViewport(artifacts, artifactChecks, path.dirname(sidecarPath), text(isolation.note, "isolation note"));
  return { ...base, sessionVerified: true, sidecar: { path: path.relative(root, sidecarPath), sha256: sidecar.sha256 }, modelStatus: model.status,
    session: { outcome: session.status, startedAt: session.startedAt, elapsedMs, timingPrecision: timing, measuredTotalRuntimeMs,
      runtimeExcludedReason: measuredTotalRuntimeMs !== null ? null : session.startedAt === null ? "unknown-start" : `not-measured-total:${timing.sessionElapsed}`,
      steps: list(session.steps, "steps").length, isolation: { ...isolation }, viewport, fallback: { ...fallback } },
    artifacts: artifactChecks, measures, uniqueRequestStageCount: seen.size,
    stageStatus: Object.fromEntries(STAGES.map((stage) => [stage, measures.some((m) => m.stage === stage) ? "observed" : model.status === "not-enabled" ? "not-enabled" : "not-observed"])) };
}
function armSummary(rows: Obj[]) {
  const recorded = rows.filter((r) => r.sessionVerified), all = recorded.flatMap((r) => r.measures.map((m: Obj) => ({ ...m, runId: r.runId })));
  const unique = [...new Map(all.map((m) => [`${m.runId}/${m.requestId}/${m.stage}`, m])).values()];
  const stageSummary = (measures: Obj[]) => ({ distinctRequests: new Set(measures.map((m) => `${m.runId}/${m.requestId}`)).size,
    distinctSessions: new Set(measures.map((m) => m.runId)).size, durationMs: summary(measures.map((m) => m.durationMs)),
    zeroRequiredFields: measures.filter((m) => m.requiredFields === 0).length,
    positiveRequiredFields: measures.filter((m) => m.requiredFields > 0).length,
    withAcceptedGeneratedFields: measures.filter((m) => m.acceptedFields > 0).length,
    allRequiredFieldsWithheld: measures.filter((m) => m.requiredFields > 0 && m.acceptedFields === 0).length,
    partlyWithheld: measures.filter((m) => m.acceptedFields > 0 && m.withheldFields > 0).length,
    allRequiredFieldsAccepted: measures.filter((m) => m.requiredFields > 0 && m.acceptedFields === m.requiredFields).length,
    fieldTotals: measures.some((m) => m.requiredFields !== null) ? {
      accepted: sum(measures.map((m) => m.acceptedFields ?? 0)), withheld: sum(measures.map((m) => m.withheldFields ?? 0)), required: sum(measures.map((m) => m.requiredFields ?? 0)) } : null });
  const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);
  const tabs = recorded.flatMap((r) => typeof r.session.isolation.tabId === "string" ? [r.session.isolation.tabId] : []);
  const contexts = recorded.flatMap((r) => typeof r.session.isolation.contextId === "string" ? [r.session.isolation.contextId] : []);
  const repeated = (ids: string[]) => [...new Set(ids)].filter((id) => ids.filter((other) => other === id).length > 1);
  return { assigned: rows.length, recorded: recorded.length, slotStatuses: counts(rows.map((r) => r.slotStatus)),
    modelStatuses: counts(rows.map((r) => r.modelStatus)), outcomes: counts(recorded.map((r) => r.session.outcome)),
    fallback: { used: recorded.filter((r) => r.session.fallback.used).length, kinds: counts(recorded.map((r) => r.session.fallback.kind)), unavailableSlots: rows.length - recorded.length },
    artifactCaptureStatuses: counts(recorded.flatMap((r) => r.artifacts.map((a: Obj) => a.captureStatus))),
    artifactVerificationFailures: recorded.flatMap((r) => r.artifacts.filter((a: Obj) => !a.verified)).length,
    isolation: { kinds: counts(recorded.map((r) => r.session.isolation.kind)), uniqueTabIds: new Set(tabs).size,
      uniqueContextIds: new Set(contexts).size, repeatedTabIds: repeated(tabs), repeatedContextIds: repeated(contexts),
      sharedStorageObserved: recorded.filter((r) => r.session.isolation.sharedStorage === true).length,
      initialCartEmptyNotVerified: recorded.filter((r) => r.session.isolation.initialCartEmpty !== true).length },
    measuredTotalRuntimeMs: summary(recorded.flatMap((r) => r.session.measuredTotalRuntimeMs === null ? [] : [r.session.measuredTotalRuntimeMs])),
    excludedRuntimeSlots: rows.filter((r) => !r.sessionVerified || r.session.measuredTotalRuntimeMs === null).map((r) => ({ runId: r.runId, reason: r.session?.runtimeExcludedReason ?? r.slotStatus })),
    distinctObservedRequestClusters: new Set(unique.map((m) => `${m.runId}/${m.requestId}`)).size,
    stages: Object.fromEntries(STAGES.map((stage) => {
      const observed = unique.filter((m) => m.stage === stage), verified = observed.filter((m) => m.artifactVerified && m.comparableCaptureClock);
      return [stage, { observed: stageSummary(observed), artifactVerifiedWithCaptureClock: stageSummary(verified),
        assignedSessionsWithoutObservedStage: rows.length - new Set(observed.map((m) => m.runId)).size,
        unverifiedOrUnknownClockMeasures: observed.length - verified.length }];
    })) };
}
export async function readinessReportMain(args: string[]) {
  requireValue(args.length === 14 && args.at(-2) === "--out", "Usage: readiness-report.mts --report AUDITED1.json ... --report AUDITED6.json --out NEW.json");
  const reportPaths: string[] = [];
  for (let i = 0; i < 12; i += 2) { requireValue(args[i] === "--report" && !!args[i + 1], "Exactly six --report inputs required"); reportPaths.push(path.resolve(args[i + 1])); }
  const out = path.resolve(args.at(-1)!); requireValue(new Set(reportPaths).size === 6 && !reportPaths.includes(out), "Distinct source reports and exclusive output required");
  const sources: Obj[] = [], rows: Obj[] = []; let catalogSha: string | null = null;
  for (const reportPath of reportPaths) {
    const input = await json(reportPath), report = object(input.value, "aggregate");
    requireValue(report.version === "shopper-evidence-aggregate.v1" && report.synthetic === true && report.humanConversionClaim === false && report.fixedCases === 10 && report.fixedSessions === 20, "Frozen synthetic ten-pair report required");
    const currentCatalog = digest(report.catalogSnapshotSha256, "catalog digest"); requireValue(catalogSha === null || currentCatalog === catalogSha, "Catalog differs across panels"); catalogSha = currentCatalog;
    const assigned = list(report.rows, "report rows"); requireValue(assigned.length === 20, "Each report must retain twenty assigned slots");
    sources.push({ path: input.path, sha256: input.sha256 });
    for (const row of assigned) rows.push(await inspectRow(row, path.dirname(reportPath)));
  }
  requireValue(rows.length === 120 && new Set(rows.map((r) => r.runId)).size === 120 && new Set(rows.map((r) => r.caseId)).size === 60, "Exactly 120 unique assigned runs in sixty pairs required");
  for (const id of new Set(rows.map((r) => r.caseId))) for (const arm of ARMS) requireValue(rows.filter((r) => r.caseId === id && r.arm === arm).length === 1, "Pair membership differs");
  const recorded = rows.filter((r) => r.sessionVerified).length;
  const result = { schema: "shopper-readiness-fallback-audit.v1", synthetic: true, humanConversionClaim: false,
    physicalPaintClaim: false, imageTransferClaim: false, nativeGpuLatencyClaim: false,
    status: recorded === 120 ? "all-assigned-recorded" : "partial", assigned: 120, recorded, catalogSnapshotSha256: catalogSha,
    sources, arms: Object.fromEntries(ARMS.map((arm) => [arm, armSummary(rows.filter((r) => r.arm === arm))])), rows,
    limitations: ["Durations are browser after-paint opportunities, not physical paint, image transfer or native GPU latency.",
      "Request/stage measures are clustered within shopper sessions; requests are not independent shoppers. Stage summaries deduplicate identical run/request/stage observations.",
      "Complete with zero required fields does not establish generated-copy readiness. Positive required fields, accepted generated fields and withheld fields are separated.",
      "Fallback state is the recorded explicit browser/session observation; absent fallback evidence is not inferred from withheld fields.",
      "Every raw session hash and embedded session match is verified; every artifact capture status and saved-file hash result is retained. Unverified artifacts/unknown capture clocks do not enter verified latency summaries.",
      "Measured monotonic total-runtime summaries exclude missing starts, estimates and upper bounds; per-run raw timing precision remains visible.",
      "Viewport setup declarations remain separate from artifact-bound observed heights; an unrecorded observed width is not fabricated.",
      "Synthetic fixed scenario panel only; no human conversion lift or generalization claim."] };
  await writeFile(out, `${JSON.stringify(result, null, 2)}\n`, { flag: "wx" }); return result;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  readinessReportMain(process.argv.slice(2)).then((result) => process.stdout.write(`${JSON.stringify({ schema: result.schema, assigned: result.assigned, recorded: result.recorded, status: result.status })}\n`))
    .catch((error: unknown) => { process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`); process.exitCode = 1; });
}
