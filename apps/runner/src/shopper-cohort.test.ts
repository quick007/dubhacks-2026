import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { access, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { buildCohortPlan, cohortExecutionOrder, createCohortPlan, dispatchCohort, inspectCohortAttempts,
  shopperCohortMain, verifyCohortPins, type CohortOptions } from "./shopper-cohort.js";
import { loadShopperPacket } from "./shopper-eval-cli.js";
import { executeShopperSession, type SessionDependencies } from "./shopper-runner.js";

const source = path.resolve("experiments/ranker/pretrained/store_shopper_eval_v1");
const sha = (bytes: Buffer | string) => createHash("sha256").update(bytes).digest("hex");
const writeJson = (file: string, value: unknown) => writeFile(file, JSON.stringify(value, null, 2) + "\n");
async function fixture(run: (root: string, options: CohortOptions) => Promise<void>) {
  const root = await mkdtemp(path.join(os.tmpdir(), "shopper-cohort-cpu-fixture-"));
  try {
    const packets: string[] = [];
    for (let index = 0; index < 6; index++) {
      const directory = path.join(root, `packet-${index}`); await mkdir(directory); packets.push(directory);
      const names = ["pair-plan.json", "public-observations.json", "shopper-instructions.json", "catalog-binding.json", "evaluator-gold.json"];
      const hashes: Record<string, string> = {};
      for (const name of names) {
        const value = JSON.parse((await readFile(path.join(source, name), "utf8")).replaceAll("shopper-pair-", `fixture-${index}-pair-`));
        if (name === "catalog-binding.json") value.liveInputSourceHashes = { "AGENTS.md": sha(await readFile("AGENTS.md")) };
        await writeJson(path.join(directory, name), value); hashes[name] = sha(await readFile(path.join(directory, name)));
      }
      await writeJson(path.join(directory, "manifest.json"), { version: "actual16-paired-shopper-packet.v1", synthetic: true,
        scenarioHeldOut: true, sourceReceiptPhase: "frozen-cpu-fixture-only", files: hashes });
    }
    const snapshot = path.join(root, "snapshot.json"); await writeJson(snapshot, { fixture: true, notARealModelSnapshot: true });
    await run(root, { packets, snapshot, out: path.join(root, "output"), seed: 20261018, pilotPairs: 5, deadline: "2026-10-18T13:00:00Z" });
  } finally { await rm(root, { recursive: true, force: true }); }
}
const neverOpen: SessionDependencies = { openBrowser: async () => { throw new Error("CPU test must not open a browser"); },
  openAgent: () => { throw new Error("CPU test must not open an agent"); } };

test("global 60-pair schedule is deterministic, unique and preserves balanced arm order", async () => fixture(async (_root, options) => {
  const packets = await Promise.all(options.packets.map(loadShopperPacket));
  const schedule = cohortExecutionOrder(packets, options.seed, 5);
  assert.deepEqual(schedule, cohortExecutionOrder(packets, options.seed, 5));
  assert.notDeepEqual(schedule, cohortExecutionOrder(packets, 2, 5));
  assert.equal(schedule.length, 120); assert.equal(new Set(schedule.map((s) => s.runId)).size, 120);
  assert.equal(schedule.filter((slot) => slot.pilot).length, 10);
  assert.equal(schedule.filter((slot) => slot.arm === "generic").length, 60);
  assert.equal(schedule.filter((slot, index) => index % 2 === 0 && slot.arm === "generic").length, 30);
  for (let index = 0; index < 120; index += 2) {
    const pair = schedule.slice(index, index + 2), c = packets[pair[0].packetIndex].cases.find((c) => c.caseId === pair[0].caseId)!;
    assert.equal(pair[0].caseId, pair[1].caseId); assert.equal(pair[0].packetIndex, pair[1].packetIndex);
    assert.deepEqual(pair.map((s) => s.arm), c.armOrder);
  }
  assert.throws(() => cohortExecutionOrder(packets, -1, 5));
  assert.throws(() => cohortExecutionOrder(packets, 1, 0));
  assert.throws(() => cohortExecutionOrder([...packets.slice(0, 5), packets[0]], 1, 5), /unique/);
}));

test("CPU plan exclusively creates output and pins every packet file, core and snapshot", async () => fixture(async (_root, options) => {
  const plan = await createCohortPlan(options);
  assert.equal(plan.fixedSessions, 120); assert.equal(plan.settings.concurrency, 1);
  assert.equal(plan.snapshot.sha256, sha(await readFile(options.snapshot)));
  assert.ok(plan.implementationSources["shopper-cohort.ts"]); assert.ok(plan.implementationSources["shopper-chrome.ts"]);
  assert.equal(plan.budget.maxWallSeconds, 120);
  assert.deepEqual((await readdir(options.out)).sort(), [".gitignore", "cohort-plan.json"]);
  await verifyCohortPins(plan);
  await assert.rejects(createCohortPlan(options), /EEXIST/);
  await writeFile(path.join(options.packets[0], "new-unfrozen-file.txt"), "CPU fixture change");
  await assert.rejects(verifyCohortPins(plan), /Frozen packet files changed/);
}));

test("changed snapshot and provisional packet fail before browser dispatch", async () => fixture(async (_root, options) => {
  const plan = await buildCohortPlan(options);
  await writeJson(options.snapshot, { fixture: "changed" });
  await assert.rejects(verifyCohortPins(plan), /snapshot receipt changed/);
  const manifestFile = path.join(options.packets[0], "manifest.json"), manifest = JSON.parse(await readFile(manifestFile, "utf8"));
  manifest.sourceReceiptPhase = "provisional-awaiting-model-owner-ready"; await writeJson(manifestFile, manifest);
  await assert.rejects(buildCohortPlan(options), /provisional/);
  await assert.rejects(access(options.out));
}));

test("distinct packet IDs, fixed catalog and explicit timezone are required", async () => fixture(async (_root, options) => {
  await assert.rejects(buildCohortPlan({ ...options, packets: Array(6).fill(options.packets[0]) }), /distinct/);
  await assert.rejects(buildCohortPlan({ ...options, deadline: "2026-10-18T13:00:00" }), /timezone/);
  const bindingFile = path.join(options.packets[0], "catalog-binding.json"), binding = JSON.parse(await readFile(bindingFile, "utf8"));
  binding.catalogSnapshot.products[0].itemPrice += 1; await writeJson(bindingFile, binding);
  const manifestFile = path.join(options.packets[0], "manifest.json"), manifest = JSON.parse(await readFile(manifestFile, "utf8"));
  manifest.files["catalog-binding.json"] = sha(await readFile(bindingFile)); await writeJson(manifestFile, manifest);
  await assert.rejects(buildCohortPlan(options), /Identical catalog/);
}));

test("expired deadline retains all 120 assigned slots and six complete unattempted aggregates", async () => fixture(async (_root, options) => {
  const plan = await createCohortPlan(options);
  const result = await dispatchCohort(plan, options.out, { dependencies: neverOpen, now: () => new Date("2026-10-18T13:00:00Z") });
  assert.equal(result.stopReason, "deadline"); assert.equal(result.launchedThisInvocation, 0);
  assert.equal(result.recordedReceipts, 0); assert.equal(result.unattempted, 120); assert.equal(result.schedule.length, 120);
  assert.equal(result.aggregateFiles.length, 6);
  for (const artifact of result.aggregateFiles) {
    const report = JSON.parse(await readFile(path.join(options.out, artifact.file), "utf8"));
    assert.equal(report.rows.length, 20); assert.ok(report.rows.every((row: { status: string }) => row.status === "unattempted"));
  }
  await assert.rejects(access(path.join(options.out, "active-run-lock.json")));
}));

test("graceful stop preserves current CPU fixture receipt and does not launch its paired arm", async () => fixture(async (_root, options) => {
  const plan = await createCohortPlan(options); let stopped = false, opened = 0;
  const dependencies: SessionDependencies = {
    openBrowser: async (url, directory) => {
      opened++;
      return { browserVersion: "CPU-fixture-only", contextId: "CPU-fixture-context", tabId: "CPU-fixture-tab",
        observe: async () => ({ url, text: "CPU fixture, not browser evidence", controls: [], screenshotBase64: "",
          observedAtPerformanceMs: 0, cartEmpty: true, cartItemCount: 0, modelStatusText: "", readiness: [],
          viewport: { scrollY: 0, height: 960, documentHeight: 960 } }),
        act: async () => { throw new Error("CPU fixture should abstain without actions"); },
        close: async () => { await writeFile(path.join(directory, "screenshot.png"), "CPU fixture pixels");
          await writeFile(path.join(directory, "trace.zip"), "CPU fixture trace bytes"); },
      };
    },
    openAgent: () => ({ decide: async () => ({ type: "abandon", reason: "CPU fixture stop boundary" }), close: async () => {} }),
  };
  const result = await dispatchCohort(plan, options.out, { dependencies, now: () => new Date("2026-10-18T12:00:00Z"), shouldStop: () => stopped,
    execute: async (...args) => { const result = await executeShopperSession(...args); stopped = true; return result; } });
  assert.equal(opened, 1); assert.equal(result.recordedReceipts, 1); assert.equal(result.unattempted, 119);
  assert.equal(result.stopReason, "requested-stop");
  assert.equal((await inspectCohortAttempts(plan, options.out)).size, 1);
  const logs = (await readFile(path.join(options.out, "dispatch.jsonl"), "utf8")).trim().split("\n").map((line) => JSON.parse(line));
  assert.equal(logs.filter((entry) => entry.type === "start").length, 1);
  assert.equal(logs.filter((entry) => entry.type === "finish").length, 1);
  const preserved = plan.schedule[0], file = path.join(options.out, preserved.packetKey, preserved.caseId, preserved.arm, "session.json");
  const original = await readFile(file);
  const resumed = await dispatchCohort(plan, options.out, { dependencies: neverOpen,
    now: () => new Date("2026-10-18T13:00:00Z") });
  assert.equal(resumed.launchedThisInvocation, 0); assert.equal(resumed.recordedReceipts, 1);
  assert.equal(sha(await readFile(file)), sha(original));
}));

test("SIGTERM after CPU scorer preflight stops future launches and removes only its own lock", async () => fixture(async (_root, options) => {
  const plan = await createCohortPlan(options);
  const { aggregateShopperResults } = await import("./shopper-eval-cli.js");
  let emitted = false;
  const result = await dispatchCohort(plan, options.out, { dependencies: neverOpen,
    now: () => new Date("2026-10-18T12:00:00Z"), aggregate: async (...args) => {
      const report = await aggregateShopperResults(...args);
      if (!emitted) { emitted = true; process.emit("SIGTERM", "SIGTERM"); }
      return report;
    } });
  assert.equal(result.stopReason, "SIGTERM"); assert.equal(result.launchedThisInvocation, 0);
  assert.equal(result.unattempted, 120);
  await assert.rejects(access(path.join(options.out, "active-run-lock.json")));
}));

test("uncertain attempt and existing active dispatcher cannot be silently retried", async () => fixture(async (_root, options) => {
  const plan = await createCohortPlan(options), slot = plan.schedule[0];
  const directory = path.join(options.out, slot.packetKey, slot.caseId, slot.arm); await mkdir(directory, { recursive: true });
  await writeJson(path.join(directory, "attempt-lock.json"), { runId: slot.runId });
  await assert.rejects(inspectCohortAttempts(plan, options.out), /Unfinished or uncertain attempt/);
  await rm(directory, { recursive: true });
  await writeJson(path.join(options.out, "active-run-lock.json"), { fixture: true });
  await assert.rejects(dispatchCohort(plan, options.out, { dependencies: neverOpen }), /EEXIST/);
  assert.deepEqual(JSON.parse(await readFile(path.join(options.out, "active-run-lock.json"), "utf8")), { fixture: true });
}));

test("scorer source freshness blocks the dispatch before opening CPU fixture dependencies", async () => fixture(async (_root, options) => {
  const file = path.join(options.packets[0], "catalog-binding.json"), binding = JSON.parse(await readFile(file, "utf8"));
  binding.liveInputSourceHashes["AGENTS.md"] = "0".repeat(64); await writeJson(file, binding);
  const manifestFile = path.join(options.packets[0], "manifest.json"), manifest = JSON.parse(await readFile(manifestFile, "utf8"));
  manifest.files["catalog-binding.json"] = sha(await readFile(file)); await writeJson(manifestFile, manifest);
  const plan = await createCohortPlan(options);
  await assert.rejects(dispatchCohort(plan, options.out, { dependencies: neverOpen }), /Aggregation blocked.*Catalog source changed/);
  const logs = await readFile(path.join(options.out, "dispatch.jsonl"), "utf8"); assert.ok(!logs.includes('"type":"start"'));
}));

test("CLI plan stays CPU-only; resume rejects changed settings before adapter discovery", async () => fixture(async (_root, options) => {
  const args = ["--packets-json", JSON.stringify(options.packets), "--snapshot", options.snapshot, "--out", options.out,
    "--seed", String(options.seed), "--pilot-pairs", "5", "--deadline", options.deadline];
  assert.equal((await shopperCohortMain(["plan", ...args])).command, "plan");
  const changed = [...args]; changed[changed.indexOf("--seed") + 1] = "19";
  await assert.rejects(shopperCohortMain(["run", ...changed, "--resume"]), /Resume settings/);
  await assert.rejects(shopperCohortMain(["plan", ...args, "--resume"]), /run-only/);
}));
