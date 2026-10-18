import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import { exportTigerSeed, main, prepareSeed } from "./authored-history.mjs";

const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const version = "agent-authored-shopping-history.v1";
async function fixture(directory) {
  const provenance = { kind: "agent-authored-synthetic", browserObserved: false, humanConversionClaim: false };
  const profiles = Array.from({ length: 200 }, (_, i) => ({ profileId: `fixture-${i}`, userGroupId: `group-${Math.floor(i / 2)}`,
    sessionId: `session-${i}`, request: { query: "Protocol fixture only", persona: { background: ["Synthetic test"], colorPreferences: [] }, publicHistory: [] } }));
  const values = { "profiles.public.json": { schemaVersion: version, provenance, profiles },
    "demo-people.json": { schemaVersion: version, provenance, people: Array.from({ length: 10 }, (_, i) => ({ id: `person-${i}` })) },
    "split.json": { trainProfileIds: profiles.slice(0, 160).map((p) => p.profileId), heldoutProfileIds: profiles.slice(160).map((p) => p.profileId) },
    "catalog-binding.json": { schemaVersion: version } };
  const files = {};
  for (const [name, value] of Object.entries(values)) { const bytes = JSON.stringify(value); await writeFile(path.join(directory, name), bytes); files[name] = hash(bytes); }
  const frozen = JSON.stringify({ publicAndSplitPublishedBeforeLabels: true, files: { ...files } });
  await writeFile(path.join(directory, "PUBLIC_SPLIT_FROZEN.json"), frozen); files["PUBLIC_SPLIT_FROZEN.json"] = hash(frozen);
  await writeFile(path.join(directory, "manifest.json"), JSON.stringify({ schemaVersion: version, provenance, files, sourceSha256: { fixture: "a".repeat(64) } }));
  await writeFile(path.join(directory, "labels.heldout.json"), "NOT JSON; private fixture canary must never be opened");
}
test("import opens only frozen public documents; heldout annotations stay outside report", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "authored-seed-test-"));
  try {
    await fixture(dir); const envelope = await prepareSeed(dir, "authored-fixture");
    assert.equal(envelope.report.origin, "agent-authored"); assert.equal(envelope.report.browserObserved, false);
    assert.equal(envelope.report.profiles.data.profiles.length, 200);
    assert(!JSON.stringify(envelope).includes("private fixture canary"));
    await writeFile(path.join(dir, "profiles.public.json"), "{}");
    await assert.rejects(prepareSeed(dir, "authored-fixture"), /changed/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
test("all import/export modes require explicit authored-synthetic opt-in before I/O", async () => {
  await assert.rejects(main(["export", "--report-id", "fixture", "--out", "/not-used"]), /Explicit/);
  await assert.rejects(main(["import", "--report-id", "fixture", "--out", "/not-used"]), /Explicit/);
});
test("fresh Tiger export retains public requests and separate provenance; never falls back locally", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "authored-export-test-"));
  try {
    const seed = path.join(root, "seed"); await import("node:fs/promises").then((fs) => fs.mkdir(seed)); await fixture(seed);
    const envelope = await prepareSeed(seed, "authored-fixture"), output = path.join(root, "export");
    const fetchImpl = async (url, init) => {
      assert.equal(url, "http://127.0.0.1:4311/api/experiment-results/reports/authored-fixture?source=tiger"); assert.equal(init.redirect, "error");
      return new Response(JSON.stringify({ source: "tiger", result: { ...envelope, sha256: "b".repeat(64), receivedAt: "2026-10-18T00:00:00Z" } }));
    };
    const manifest = await exportTigerSeed("authored-fixture", output, fetchImpl);
    assert.equal(manifest.exportEligibility.defaultRealTrafficTraining, false);
    assert.equal((await stat(output)).mode & 0o777, 0o700);
    assert.equal((await stat(path.join(output, "profiles.public.json"))).mode & 0o777, 0o600);
    assert.equal((await readdir(output)).some((name) => name.includes("labels")), false);
    assert.deepEqual(JSON.parse(await readFile(path.join(output, "profiles.public.json"))), envelope.report.profiles.data);
    await assert.rejects(exportTigerSeed("authored-fixture", path.join(root, "no-fallback"), async () => new Response(JSON.stringify({ source: "local", result: { ...envelope, sha256: "b".repeat(64) } }))), /Fresh Tiger/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
