/** Post-run only: preserves verified source bytes/hashes; no credentials, browsers or model calls. */
import { createHash } from "node:crypto";
import { readFile, realpath, stat, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { buildEnvelope } from "../../runner/src/shopper-export.ts";
import { shopperMetricsMain } from "../../runner/src/shopper-metrics.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const ENDPOINT = "http://127.0.0.1:4311/api/experiment-results";
const LIMIT = 8 * 1024 * 1024;
const HASH = /^[a-f0-9]{64}$/;
const ID = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}$/;
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
function requireValue(value, message) { if (!value) throw new Error(message); }
function source(document) {
  requireValue(typeof document?.path === "string" && document.path.length <= 4000 && HASH.test(document.sha256), "Hashed source document required");
  return { path: document.path, sha256: document.sha256 };
}
const same = (a, b) => isDeepStrictEqual(a, b);

/** Pure transport validation. File verification and metric reanalysis happen before this call. */
export function buildCohortEnvelope({ reportId, experimentId, plan, sourceReports, metrics, serverOutcomes, slideMetrics }) {
  requireValue(ID.test(reportId) && ID.test(experimentId), "Safe immutable report/cohort ID required");
  source(plan); source(metrics); source(serverOutcomes);
  requireValue(plan.data?.version === "shopper-fixed-cohort.v1" && plan.data.synthetic === true && plan.data.humanConversionClaim === false
    && plan.data.fixedPairs === 60 && plan.data.fixedSessions === 120 && plan.data.schedule?.length === 120, "Fixed120slot cohort plan required");
  requireValue(Array.isArray(sourceReports) && sourceReports.length === 6, "Exactly six audited source panels required");
  const reports = new Map(), runs = new Map(), cases = new Set();
  const schedule = new Map(plan.data.schedule.map((s) => [s.runId, `${s.caseId}/${s.arm}`]));
  requireValue(schedule.size === 120, "Unique planned run IDs required");
  for (const item of sourceReports) {
    source(item);
    requireValue(!reports.has(item.path), "Duplicate source report"); reports.set(item.path, item.sha256);
    buildEnvelope(item.report, { reportId, experimentId: item.experimentId });
    requireValue(item.report.synthetic === true, "Explicit synthetic report required");
    for (const r of item.report.rows) {
      const slot = `${r.caseId}/${r.arm}`;
      requireValue(!runs.has(r.expectedRunId) && schedule.get(r.expectedRunId) === slot, "Source runs differ from frozen120slot plan");
      runs.set(r.expectedRunId, slot); cases.add(r.caseId);
    }
  }
  requireValue(runs.size === 120 && cases.size === 60, "All sixty pairs retained");
  const checkSources = (list) => requireValue(Array.isArray(list) && list.length === 6 && new Set(list.map((s) => s.path)).size === 6
    && list.every((s) => reports.get(s.path) === s.sha256), "Exact six source file hashes required");
  const m = metrics.data, index = serverOutcomes.data;
  requireValue(m?.version === "shopper-synthetic-metrics.v1" && m.synthetic === true && m.humanConversionClaim === false
    && m.fixedCases === 60 && m.fixedSessions === 120 && m.pairs?.length === 60 && same(m.serverIndexSource, source(serverOutcomes)), "Server-bound fixed-cohort metrics required");
  checkSources(m.sources);
  requireValue(index?.schema === "shopper-server-outcomes.v1" && index.synthetic === true && index.humanConversionClaim === false
    && index.rows?.length === 120, "All120 server outcome slots required"); checkSources(index.reports);
  const checked = new Set();
  for (const r of index.rows) {
    requireValue(!checked.has(r.runId) && runs.get(r.runId) === `${r.caseId}/${r.arm}` && typeof r.serverConfirmed === "boolean"
      && typeof r.visibleCheckoutConfirmed === "boolean" && (!r.visibleCheckoutConfirmed || r.serverConfirmed), "Server index identity/state mismatch");
    checked.add(r.runId);
    if (r.serverConfirmed) requireValue(r.receipt?.schema === "fieldwork.demo-checkout-receipt.v1" && r.receipt.realPurchase === false
      && r.readback?.httpStatus === 200 && r.readback.identical === true, "Fake receipt and identical GET readback required");
  }
  const paired = new Set();
  for (const pair of m.pairs) {
    requireValue(cases.has(pair.caseId) && !paired.has(pair.caseId), "Unique metric case pairing required"); paired.add(pair.caseId);
    for (const arm of ["generic", "local-model"]) requireValue(pair[arm]?.caseId === pair.caseId && pair[arm]?.arm === arm
      && runs.get(pair[arm].runId) === `${pair.caseId}/${arm}`, "Metric/run pairing mismatch");
  }
  for (const arm of ["generic", "local-model"]) requireValue(m.arms?.[arm]?.assigned === 60 && m.arms[arm].allServerConfirmedOrders?.evaluated === true,
    "Both60session denominators and evaluated server evidence required");
  if (slideMetrics) source(slideMetrics);
  const envelope = { version: "experiment-results.v1", reportId, experimentId, synthetic: true, humanConversionClaim: false,
    report: { version: "shopper-checkout-cohort.v1", synthetic: true, humanConversionClaim: false, fixedCases: 60, fixedSessions: 120,
      plan, sourceReports, metrics, serverOutcomes, ...(slideMetrics ? { slideMetrics } : {}) } };
  const bytes = JSON.stringify(envelope);
  requireValue(Buffer.byteLength(bytes) <= LIMIT, "Cohort envelope exceeds8MiB; do not truncate assigned slots");
  return JSON.parse(bytes);
}

async function readDocument(file, root) {
  const actual = await realpath(file);
  requireValue(actual.startsWith(root + path.sep) && (await stat(actual)).isFile(), "Source must remain within its frozen cohort directory");
  requireValue((await stat(actual)).size <= LIMIT, "Source exceeds8MiB");
  const bytes = await readFile(actual);
  return { path: actual, sha256: sha(bytes), data: JSON.parse(bytes.toString("utf8")) };
}

export async function prepareCohort(directory, reportId, experimentId) {
  const root = await realpath(directory), experiments = await realpath(path.join(ROOT, "experiments/ranker/pretrained"));
  requireValue(root.startsWith(experiments + path.sep), "Use a frozen project experiment directory");
  const candidates = [];
  for (const name of ["shopper-metrics.json", "metrics.json"]) {
    try { await stat(path.join(root, name)); candidates.push(name); }
    catch (error) { if (error.code !== "ENOENT") throw error; }
  }
  requireValue(candidates.length === 1, "Exactly one finalized shopper-metrics.json (or legacy metrics.json) required");
  const [plan, metrics, serverOutcomes] = await Promise.all(["cohort-plan.json", candidates[0], "server-outcomes.json"].map((name) => readDocument(path.join(root, name), root)));
  requireValue(metrics.data.sources?.length === 6 && metrics.data.packetSources?.length === 6 && plan.data.packets?.length === 6, "Six complete source/packet records required");
  requireValue(same(metrics.data.serverIndexSource, source(serverOutcomes)), "Metrics bind a different server index");
  const sourceReports = [];
  for (let i = 0; i < 6; i++) {
    const report = await readDocument(path.join(root, `packet-${i + 1}/aggregate-audited.json`), root);
    requireValue(metrics.data.sources.some((s) => same(s, source(report))), "Metrics must bind exact audited report bytes");
    const packet = await realpath(plan.data.packets[i].directory);
    requireValue(packet.startsWith(experiments + path.sep) && /^store_shopper_eval_v[2-7]$/.test(path.basename(packet)), "Only frozen public shopper packet directories allowed");
    requireValue(metrics.data.packetSources[i].path === packet, "Metric packet source differs from frozen plan");
    const manifestFile = path.join(packet, "manifest.json"), manifestBytes = await readFile(manifestFile), manifest = JSON.parse(manifestBytes);
    requireValue(sha(manifestBytes) === plan.data.packets[i].files["manifest.json"], "Packet manifest changed since plan freeze");
    sourceReports.push({ ...source(report), experimentId: manifest.experimentId, report: report.data });
  }
  // Rehash receipt evidence and rerun the unchanged CPU analyzer before upload.
  // It reads frozen evaluator checks for this evaluation only, never a training input.
  for (const row of serverOutcomes.data.rows ?? []) if (row.artifact !== null) {
    const actual = await realpath(row.artifact.path);
    requireValue(actual.startsWith(root + path.sep) && path.basename(actual) === "checkout-receipt.json", "Only contained checkout receipt evidence may be consumed; raw browser logs excluded");
  }
  const temporary = await mkdtemp(path.join(os.tmpdir(), "fieldwork-cohort-verify-"));
  try {
    const args = sourceReports.flatMap((r, i) => ["--report", r.path, "--packet", metrics.data.packetSources[i].path]);
    const recomputed = await shopperMetricsMain([...args, "--server-outcomes", serverOutcomes.path, "--out", path.join(temporary, "metrics.json")]);
    requireValue(same(recomputed, metrics.data), "Saved metrics differ from rehashed CPU analyzer output; do not upload");
  } finally { await rm(temporary, { recursive: true, force: true }); }
  let slideMetrics;
  try { slideMetrics = await readDocument(path.join(root, "slide-metrics.json"), root); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
  for (const doc of [plan, metrics, serverOutcomes, ...sourceReports, ...(slideMetrics ? [slideMetrics] : [])])
    requireValue(sha(await readFile(doc.path)) === doc.sha256, "Source changed during final verification");
  return buildCohortEnvelope({ reportId, experimentId, plan, metrics, serverOutcomes, sourceReports, slideMetrics });
}

export async function postCohort(envelope, fetchImpl = fetch) {
  const checked = buildCohortEnvelope({ reportId: envelope.reportId, experimentId: envelope.experimentId, ...envelope.report });
  const response = await fetchImpl(ENDPOINT, { method: "POST", redirect: "error", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(checked), signal: AbortSignal.timeout(10000) });
  requireValue(response.status === 202, `Cohort intake failed HTTP${response.status}`);
  const receipt = await response.json();
  requireValue(receipt.reportId === checked.reportId && HASH.test(receipt.sha256) && receipt.durableLocal === true
    && typeof receipt.duplicate === "boolean" && receipt.cloud === "asynchronous", "Durable local receipt not confirmed");
  return receipt;
}

export async function main(args) {
  const options = {}; let post = false;
  for (let i = 0; i < args.length; i++) {
    const flag = args[i];
    if (flag === "--post") { requireValue(!post, "Repeated --post"); post = true; continue; }
    requireValue(["--cohort", "--report-id", "--experiment-id", "--out"].includes(flag) && args[i + 1] && !options[flag], "Use --cohort DIR --report-id ID --experiment-id COHORT_ID --out NEW.json [--post]");
    options[flag] = args[++i];
  }
  requireValue(Object.keys(options).length === 4, "All four preparation arguments required");
  const envelope = await prepareCohort(options["--cohort"], options["--report-id"], options["--experiment-id"]);
  await writeFile(options["--out"], JSON.stringify(envelope) + "\n", { flag: "wx", mode: 0o600 });
  return post ? postCohort(envelope) : { reportId: envelope.reportId, prepared: true, uploaded: false };
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main(process.argv.slice(2))
  .then((result) => process.stdout.write(JSON.stringify(result) + "\n"))
  .catch((error) => { process.stderr.write(error.message + "\n"); process.exitCode = 1; });
