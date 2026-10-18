// Times hosted LLMs on the same ranking task as the local model, for the deck's speed slide.
//
// Each LLM gets the 16 product fact records and the shopper request the local model read in the
// frozen actual-store run, and returns all 16 products best first. Calls go through the Codex CLI,
// so each one also carries the agent's own system prompt. The timer runs from turn start to turn
// end, which leaves out CLI startup. A trivial "reply ok" turn per model measures the floor that
// any request pays.
//
// Writes apps/web/src/fieldwork/llm-run.json.
//
// Usage: node scripts/benchmark-llm-ranking.mjs [model:effort ...]
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const at = (...parts) => path.join(root, ...parts);
const MODEL_RUN = "experiments/ranker/pretrained/actual_store_function_results/v3";
const OUTPUT = "apps/web/src/fieldwork/llm-run.json";
const FLOOR_RUNS = 3;
const models = (
  process.argv.length > 2 ? process.argv.slice(2) : ["gpt-6.1-sol:medium", "gpt-6-luna:medium"]
).map((spec) => {
  const [slug, effort = "medium"] = spec.split(":");
  return { slug, effort };
});

const readJson = async (file) => JSON.parse(await readFile(at(file), "utf8"));
const packet = await readJson(`${MODEL_RUN}/packet.json`);
const modelRun = await readJson("apps/web/src/fieldwork/model-run.json");

const recordBody = (record) => JSON.parse(record.text.slice(record.text.indexOf("\n") + 1));
const ids = Object.keys(modelRun.products);
const facts = packet.records
  .filter((record) => record.role === "catalog")
  .map((record, index) => ({ id: ids[index], ...recordBody(record) }));
if (facts.some((item) => modelRun.products[item.id].name !== item.product.name))
  throw new Error("The model run's products no longer match the packet catalog.");
const requests = new Map(
  packet.records
    .filter((record) => record.split === "evaluation")
    .map((record) => [record.caseId, recordBody(record)]),
);

const schema = {
  type: "object",
  properties: { order: { type: "array", items: { type: "string", enum: ids } } },
  required: ["order"],
  additionalProperties: false,
};

const rankPrompt = (request) =>
  [
    "You personalize a storefront. Rank all 16 products for this shopper, best fit first.",
    "Answer from the data below only. Do not run commands or read files.",
    'Return JSON: {"order": [16 product ids, best first, each id once]}.',
    "",
    "Products:",
    JSON.stringify(facts),
    "",
    "Shopper request:",
    JSON.stringify(request),
  ].join("\n");

/** One Codex call. Returns the turn time, the final message and token usage. */
function codex({ slug, effort }, prompt, dir, schemaFile) {
  const args = [
    "exec",
    "-m",
    slug,
    "-c",
    `model_reasoning_effort="${effort}"`,
    "--sandbox",
    "read-only",
    "--ephemeral",
    "--skip-git-repo-check",
    "--ignore-rules",
    "-C",
    dir,
    "--json",
    ...(schemaFile ? ["--output-schema", schemaFile] : []),
    prompt,
  ];
  return new Promise((resolve, reject) => {
    const child = spawn("codex", args, { stdio: ["ignore", "pipe", "ignore"] });
    const spawned = performance.now();
    let buffer = "";
    let turnStart;
    const result = { message: undefined, usage: undefined };
    child.stdout.on("data", (chunk) => {
      buffer += chunk;
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        const event = JSON.parse(line);
        if (event.type === "turn.started") turnStart = performance.now();
        if (event.type === "item.completed" && event.item.type === "agent_message")
          result.message = event.item.text;
        if (event.type === "turn.completed") {
          result.turnMs = performance.now() - turnStart;
          result.usage = event.usage;
        }
        if (event.type === "turn.failed" || event.type === "error")
          result.error = JSON.stringify(event);
      }
    });
    child.on("error", reject);
    child.on("close", (code) => {
      result.processMs = performance.now() - spawned;
      if (code !== 0 || result.turnMs === undefined)
        reject(new Error(`codex exited ${code}: ${result.error ?? "no completed turn"}`));
      else resolve(result);
    });
  });
}

const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};
const summary = (values) => ({
  median: median(values),
  min: Math.min(...values),
  max: Math.max(...values),
});

const dir = await mkdtemp(path.join(tmpdir(), "llm-ranking-"));
const schemaFile = path.join(dir, "schema.json");
await writeFile(schemaFile, JSON.stringify(schema));
const work = path.join(dir, "work");
await (await import("node:fs/promises")).mkdir(work);

const results = [];
try {
  for (const model of models) {
    const label = `${model.slug} (${model.effort})`;
    const floor = [];
    for (let i = 0; i < FLOOR_RUNS; i++) {
      const run = await codex(model, "Reply with exactly: ok", work);
      floor.push(run.turnMs);
      console.log(`${label} floor ${i + 1}: ${Math.round(run.turnMs)} ms`);
    }
    const cases = [];
    for (const item of modelRun.requests) {
      const run = await codex(model, rankPrompt(requests.get(item.caseId)), work, schemaFile);
      const order = JSON.parse(run.message).order;
      const valid = order.length === ids.length && new Set(order).size === ids.length;
      const rank = order.indexOf(item.target) + 1;
      cases.push({
        caseId: item.caseId,
        target: item.target,
        rank,
        valid,
        turnMs: run.turnMs,
        processMs: run.processMs,
        usage: run.usage,
        order,
      });
      console.log(`${label} ${item.caseId}: rank ${rank}, ${Math.round(run.turnMs)} ms`);
    }
    results.push({
      model: model.slug,
      effort: model.effort,
      cases: cases.length,
      topOne: cases.filter((item) => item.valid && item.rank === 1).length,
      topSix: cases.filter((item) => item.valid && item.rank >= 1 && item.rank <= 6).length,
      validOrders: cases.filter((item) => item.valid).length,
      turnMs: summary(cases.map((item) => item.turnMs)),
      processMs: summary(cases.map((item) => item.processMs)),
      floorTurnMs: summary(floor),
      requests: cases,
    });
  }
} finally {
  await rm(dir, { recursive: true, force: true });
}

await writeFile(
  at(OUTPUT),
  `${JSON.stringify(
    {
      measuredAt: new Date().toISOString(),
      method:
        "Codex CLI exec, read-only sandbox, ephemeral session, JSON output schema. Turn time runs from turn.started to turn.completed; process time includes CLI startup. Floor is a 'Reply with exactly: ok' turn.",
      source: MODEL_RUN,
      models: results,
    },
    null,
    2,
  )}\n`,
);
console.log(`Wrote ${OUTPUT}`);
