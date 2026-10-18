/** Explicit authored-synthetic import/export. No secret-file access, browser activity or training. */
import { createHash } from "node:crypto";
import { readFile, realpath, stat, mkdir, writeFile, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const API = "http://127.0.0.1:4311/api/experiment-results";
const VERSION = "agent-authored-shopping-history.v1";
const FILES = { profiles: "profiles.public.json", demoPeople: "demo-people.json", split: "split.json", catalog: "catalog-binding.json", publicFreeze: "PUBLIC_SPLIT_FROZEN.json" };
const HASH = /^[a-f0-9]{64}$/, ID = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}$/;
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
function requireValue(ok, message) { if (!ok) throw new Error(message); }
export function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map((k) => [k, canonical(value[k])]));
  return value;
}
const encoded = (value) => JSON.stringify(canonical(value)) + "\n";
function authoredReport(report) {
  requireValue(report?.version === "authored-shopping-history.v1" && report.datasetVersion === VERSION && report.origin === "agent-authored"
    && report.synthetic === true && report.browserObserved === false && report.realCustomerData === false && report.humanConversionClaim === false, "Authored-only synthetic provenance required");
  for (const key of Object.keys(FILES)) requireValue(report[key]?.data && HASH.test(report[key].sha256), "All five hashed public documents required");
  requireValue(report.profiles.data.profiles?.length === 200 && report.demoPeople.data.people?.length === 10
    && report.split.data.trainProfileIds?.length === 160 && report.split.data.heldoutProfileIds?.length === 40, "Frozen200profiles/10demos/160train40heldout required");
  for (const doc of [report.profiles, report.demoPeople]) requireValue(doc.data.schemaVersion === VERSION
    && doc.data.provenance?.kind === "agent-authored-synthetic" && doc.data.provenance.browserObserved === false
    && doc.data.provenance.humanConversionClaim === false, "Authored requests must not masquerade as observed customer data");
  const allowed = new Set(["profileId", "userGroupId", "sessionId", "request"]);
  for (const p of report.profiles.data.profiles) requireValue(Object.keys(p).every((k) => allowed.has(k)), "Public profile cannot contain outcome labels");
  return report;
}

async function publicDocument(root, name) {
  const file = await realpath(path.join(root, name));
  requireValue(file.startsWith(root + path.sep) && (await stat(file)).size < 4 * 1024 * 1024, "Bounded contained public seed file required");
  const bytes = await readFile(file);
  return { path: file, sha256: sha(bytes), data: JSON.parse(bytes.toString("utf8")) };
}
export async function prepareSeed(directory, reportId) {
  requireValue(ID.test(reportId), "Safe immutable seed report ID required");
  const root = await realpath(directory), manifest = await publicDocument(root, "manifest.json"), documents = {};
  requireValue(manifest.data.schemaVersion === VERSION && manifest.data.provenance?.kind === "agent-authored-synthetic"
    && manifest.data.provenance.browserObserved === false && manifest.data.provenance.humanConversionClaim === false, "Frozen authored manifest required");
  // This allowlist never opens labels.train/heldout, evaluation gold, or raw browser logs.
  for (const [key, name] of Object.entries(FILES)) {
    const d = await publicDocument(root, name);
    requireValue(manifest.data.files?.[name] === d.sha256, "Public document changed since frozen manifest"); documents[key] = d;
  }
  const frozen = documents.publicFreeze.data;
  requireValue(frozen.publicAndSplitPublishedBeforeLabels === true && ["profiles", "demoPeople", "split"].every((key) => frozen.files?.[FILES[key]] === documents[key].sha256), "Public split must be frozen before private labels");
  const report = authoredReport({ version: "authored-shopping-history.v1", datasetVersion: VERSION, origin: "agent-authored", synthetic: true,
    humanConversionClaim: false, browserObserved: false, realCustomerData: false,
    ...documents, sourceManifest: { path: manifest.path, sha256: manifest.sha256 }, sourceHashes: manifest.data.sourceSha256 });
  const envelope = { version: "experiment-results.v1", reportId, experimentId: "authored-public-history-v1", synthetic: true, humanConversionClaim: false, report };
  requireValue(Buffer.byteLength(JSON.stringify(envelope)) <= 8 * 1024 * 1024, "Seed envelope exceeds8MiB");
  return envelope;
}
async function request(url, options = {}, fetchImpl = fetch) {
  return fetchImpl(url, { redirect: "error", signal: AbortSignal.timeout(10000), ...options });
}
export async function postSeed(envelope, fetchImpl = fetch) {
  authoredReport(envelope.report);
  const response = await request(API, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(envelope) }, fetchImpl);
  requireValue(response.status === 202, `Authored seed intake failed HTTP${response.status}`);
  const receipt = await response.json();
  requireValue(receipt.reportId === envelope.reportId && HASH.test(receipt.sha256) && receipt.durableLocal === true
    && typeof receipt.duplicate === "boolean" && receipt.cloud === "asynchronous", "Durable seed receipt required");
  return receipt;
}
export async function exportTigerSeed(reportId, output, fetchImpl = fetch) {
  requireValue(ID.test(reportId), "Safe immutable seed report ID required");
  const response = await request(`${API}/reports/${reportId}?source=tiger`, {}, fetchImpl);
  requireValue(response.status === 200, `Fresh Tiger seed read failed HTTP${response.status}`);
  const body = await response.json();
  requireValue(body.source === "tiger" && body.result?.reportId === reportId && HASH.test(body.result.sha256), "Fresh Tiger seed provenance required");
  const report = authoredReport(body.result.report), files = {}, originalSourceSha256 = {};
  await mkdir(output, { mode: 0o700 });
  try {
    for (const [key, filename] of Object.entries(FILES)) {
      const bytes = encoded(report[key].data);
      await writeFile(path.join(output, filename), bytes, { flag: "wx", mode: 0o600 });
      files[filename] = sha(bytes); originalSourceSha256[filename] = report[key].sha256;
    }
    const manifest = { schemaVersion: "authored-history-tiger-export.v1", datasetVersion: VERSION, source: "tiger", reportId,
      sha256: body.result.sha256, receivedAt: body.result.receivedAt, origin: "agent-authored", synthetic: true,
      browserObserved: false, realCustomerData: false, humanConversionClaim: false, includeAuthoredSynthetic: true,
      exportEligibility: { defaultRealTrafficTraining: false, explicitAuthoredSyntheticExperiment: true },
      counts: { publicProfiles: 200, trainProfiles: 160, heldoutProfiles: 40, demoPeopleExcludedFromFit: 10 },
      sourceManifest: report.sourceManifest, sourceHashes: report.sourceHashes, originalSourceSha256, files,
      serialization: "UTF8 JSON with recursively sorted object keys; arrays preserved; trailing newline. Original byte hashes retained separately.",
      annotations: "No target annotations included; join separately frozen local train/heldout annotations by profile ID without moving groups." };
    await writeFile(path.join(output, "manifest.json"), encoded(manifest), { flag: "wx", mode: 0o600 });
    return { ...manifest, output, manifestSha256: sha(encoded(manifest)) };
  } catch (error) { await rm(output, { recursive: true, force: true }); throw error; }
}
export async function main(args) {
  const [mode, ...rest] = args, options = {}; let allowed = false, post = false;
  requireValue(["import", "export"].includes(mode), "Use import or export");
  for (let i = 0; i < rest.length; i++) {
    const flag = rest[i];
    if (flag === "--include-authored-synthetic") { requireValue(!allowed, "Repeated synthetic flag"); allowed = true; continue; }
    if (flag === "--post") { requireValue(!post && mode === "import", "--post is import only"); post = true; continue; }
    requireValue(["--seed-dir", "--report-id", "--out"].includes(flag) && rest[i + 1] && !options[flag], "Invalid authored history argument"); options[flag] = rest[++i];
  }
  requireValue(allowed && options["--report-id"] && options["--out"], "Explicit --include-authored-synthetic --report-id ID --out NEW_PATH required");
  if (mode === "export") { requireValue(!options["--seed-dir"], "Export reads Tiger only"); return exportTigerSeed(options["--report-id"], options["--out"]); }
  requireValue(options["--seed-dir"], "Import requires --seed-dir DIR");
  const envelope = await prepareSeed(options["--seed-dir"], options["--report-id"]);
  await writeFile(options["--out"], JSON.stringify(envelope) + "\n", { flag: "wx", mode: 0o600 });
  return post ? postSeed(envelope) : { reportId: envelope.reportId, prepared: true, uploaded: false };
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main(process.argv.slice(2))
  .then((result) => process.stdout.write(JSON.stringify(result) + "\n"))
  .catch((error) => { process.stderr.write(error.message + "\n"); process.exitCode = 1; });
