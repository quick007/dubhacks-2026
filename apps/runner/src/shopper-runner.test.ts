import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile, access } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { loadShopperPacket } from "./shopper-eval-cli.js";
import { parseAgentAction, StdioShopperAgent, type AgentAction } from "./shopper-agent.js";
import { executeShopperSession, pairedExecutionOrder, shopperRunnerMain, type SessionDependencies } from "./shopper-runner.js";
import { buildShopperAssignment, verifyShopperArtifactFiles } from "./shopper-contract.js";

const packetDirectory = path.resolve("experiments/ranker/pretrained/store_shopper_eval_v1");
const packet = () => loadShopperPacket(packetDirectory);
async function temporary(run: (root: string) => Promise<void>) {
  const root = await mkdtemp(path.join(os.tmpdir(), "shopper-runner-test-"));
  try { await run(root); } finally { await rm(root, { recursive: true, force: true }); }
}
function fixture(directory: string, actions: AgentAction[], options: { hanging?: boolean; hangingObservation?: boolean; cartEmpty?: boolean;
  observationOverrides?: { url?: string; observedAtPerformanceMs?: number; modelStatusText?: string; readiness?: unknown[] }[] } = {}) {
  let index = 0, browserCloses = 0, agentCloses = 0;
  const observations = { text: "Fixture page, not browser execution", controls: [{ ref: "control-1", role: "button", name: "Fixture" }],
    url: "http://127.0.0.1:4312/", screenshotBase64: "", observedAtPerformanceMs: 0, modelStatusText: "", cartItemCount: 0, cartEmpty: options.cartEmpty ?? true, readiness: [] };
  const selected = { originalName: "Everyday steel bottle", catalogPrice: 36, selectedColor: "Graphite", quantity: 1 as const };
  const dependencies: SessionDependencies = {
    openBrowser: async () => ({ browserVersion: "fixture-only", contextId: "fixture-context", tabId: "fixture-tab",
      observe: async () => options.hangingObservation ? new Promise<typeof observations>(() => {})
        : { ...observations, ...options.observationOverrides?.shift() },
      act: async (action) => {
        if (action.type === "search") return { kind: "search", target: action.query, confirmationVisible: false };
        const count = ++index;
        return { kind: count === 1 ? "add-to-cart" : count === 2 ? "open-cart" : "confirm-demo-order",
          target: count === 1 ? "Add to cart" : count === 2 ? "View cart" : "Place demo order", selected,
          confirmationVisible: count === 3 };
      },
      close: async () => { browserCloses++;
        // These bytes verify plumbing only and never leave this temporary fixture directory.
        await writeFile(path.join(directory, "screenshot.png"), "fixture screenshot bytes");
        await writeFile(path.join(directory, "trace.zip"), "fixture trace bytes");
      },
    }),
    openAgent: (assignment) => {
      assert.equal("catalog" in assignment, false); assert.equal("gold" in assignment, false);
      return { decide: async () => options.hanging ? new Promise<AgentAction>(() => {}) : actions.shift() ?? { type: "abandon", reason: "Fixture ended" },
        close: async () => { agentCloses++; } };
    },
  };
  return { dependencies, counts: () => ({ browserCloses, agentCloses }) };
}

test("seeded twenty-run schedule preserves paired assignment order and unique roster", async () => {
  const p = await packet(), one = pairedExecutionOrder(p, 20261018);
  assert.deepEqual(one, pairedExecutionOrder(p, 20261018)); assert.notDeepEqual(one, pairedExecutionOrder(p, 2));
  assert.equal(new Set(one.map((s) => s.runId)).size, 20);
  for (let i = 0; i < 20; i += 2) {
    assert.equal(one[i].caseId, one[i + 1].caseId);
    assert.deepEqual([one[i].arm, one[i + 1].arm], p.cases.find((c) => c.caseId === one[i].caseId)!.armOrder);
  }
  assert.throws(() => pairedExecutionOrder(p, NaN));
});

test("agent action parser blocks private searches, arbitrary code and stale wire shapes", () => {
  assert.deepEqual(parseAgentAction({ type: "search", query: "public query" }, "public query"), { type: "search", query: "public query" });
  assert.throws(() => parseAgentAction({ type: "search", query: "private need" }, "public query"));
  assert.throws(() => parseAgentAction({ type: "click", selector: "#correct-product" }, "public query"));
  assert.throws(() => parseAgentAction({ type: "evaluate", code: "buy()" }, "public query"));
  assert.throws(() => parseAgentAction({ type: "click", ref: "control-1", productId: "gold" }, "public query"));
});

test("missing installed Codex adapter fails before browser dispatch or output creation", async () => temporary(async (root) => {
  const out = path.join(root, "output");
  await assert.rejects(shopperRunnerMain(["run", "--packet", packetDirectory, "--out", out, "--codex-binary", path.join(root, "missing-codex")]), /No installed Codex CLI/);
  await assert.rejects(access(out));
}));

test("fixture lifecycle binds DOM choice after agent exit and hashes durable evidence", async () => temporary(async (root) => {
  const p = await packet(), c = p.cases.find((c) => c.caseId === "shopper-pair-02")!;
  const f = fixture(root, [{ type: "search", query: c.observableContext.query },
    { type: "click", ref: "control-1" }, { type: "click", ref: "control-1" }, { type: "click", ref: "control-1" }]);
  const session = await executeShopperSession(packetDirectory, p, c, "generic", root, f.dependencies);
  assert.equal(session.status, "completed"); assert.equal(session.fakeOrderConfirmed, true);
  assert.equal(session.choice?.selectedColor, "Graphite"); assert.equal(session.choice?.catalogPrice, 36);
  assert.equal(session.steps.length, 5); assert.equal(session.steps[2].action.type, "add-to-cart");
  assert.equal(session.isolation.sharedStorage, false); assert.equal(session.isolation.storageCleared, true);
  assert.deepEqual(f.counts(), { browserCloses: 1, agentCloses: 1 });
  assert.ok((await verifyShopperArtifactFiles(session.artifacts, root)).every((r) => r.verified));
  await assert.rejects(executeShopperSession(packetDirectory, p, c, "generic", root, f.dependencies), /EEXIST/);
}));

test("startup completion cannot mask assigned-query failure or its observed fallback", async () => temporary(async (root) => {
  const p = await packet(), c = p.cases[1], entryUrl = buildShopperAssignment(c, "local-model", p.defaultBaseUrl).entryUrl;
  const rootComplete = { name: "fieldwork.local-model.complete.after-paint-opportunity", startTime: 20, duration: 0,
    detail: { stage: "complete", requestId: "fixture-browse-request", acceptedFields: 20, withheldFields: 0, requiredFields: 20 } };
  for (const [kind, statusText] of [
    ["service-unavailable", "Local model unavailable: CPU fixture offline. Store catalog shown."],
    ["last-safe-model", "Local model unavailable: CPU fixture offline. Last available result shown."],
  ] as const) {
    const directory = path.join(root, kind);
    const f = fixture(directory, [{ type: "search", query: c.observableContext.query },
      { type: "abandon", reason: "CPU fixture ends after observing model failure" }], {
      observationOverrides: [
        { url: entryUrl, observedAtPerformanceMs: 100, modelStatusText: "Local model ready.", readiness: [rootComplete] },
        { url: `${entryUrl}#collection?${new URLSearchParams({ q: c.observableContext.query })}`,
          observedAtPerformanceMs: 200, modelStatusText: statusText, readiness: [rootComplete] },
      ],
    });
    const session = await executeShopperSession(packetDirectory, p, c, "local-model", directory, f.dependencies);
    assert.equal(session.model.status, "failed"); assert.deepEqual(session.model.readiness, []);
    assert.deepEqual(session.fallback, { used: true, kind, reason: statusText });
    assert.equal(session.status, "abandoned"); assert.equal(session.fakeOrderConfirmed, false);
    const raw = JSON.parse(await readFile(path.join(directory, "browser-observations.json"), "utf8"));
    assert.ok(raw.logs.some((log: { readiness?: unknown[] }) => log.readiness?.length === 1), "Raw startup evidence remains auditable");
    assert.deepEqual(f.counts(), { browserCloses: 1, agentCloses: 1 });
  }
}));

test("assigned-query completion is accepted while earlier startup readiness is excluded", async () => temporary(async (root) => {
  const p = await packet(), c = p.cases[1], entryUrl = buildShopperAssignment(c, "local-model", p.defaultBaseUrl).entryUrl;
  const rootComplete = { name: "fieldwork.local-model.complete.after-paint-opportunity", startTime: 20, duration: 0,
    detail: { stage: "complete", requestId: "fixture-browse-request", acceptedFields: 20, withheldFields: 0, requiredFields: 20 } };
  const queryComplete = { ...rootComplete, startTime: 100,
    detail: { ...rootComplete.detail, requestId: "fixture-assigned-query-request" } };
  const f = fixture(root, [{ type: "search", query: c.observableContext.query },
    { type: "abandon", reason: "CPU fixture ends after observing model completion" }], {
    observationOverrides: [
      { url: entryUrl, observedAtPerformanceMs: 100, readiness: [rootComplete] },
      { url: `${entryUrl}#collection?${new URLSearchParams({ q: c.observableContext.query })}`,
        observedAtPerformanceMs: 200, modelStatusText: "Local model ready.", readiness: [rootComplete, queryComplete] },
    ],
  });
  const session = await executeShopperSession(packetDirectory, p, c, "local-model", root, f.dependencies);
  assert.equal(session.model.status, "complete"); assert.equal(session.model.readiness.length, 1);
  assert.equal(session.model.readiness[0].requestId, "fixture-assigned-query-request");
  assert.equal(session.model.readiness[0].stage, "complete"); assert.equal(session.model.readiness[0].startTimeMs, 100);
  assert.deepEqual(session.fallback, { used: false, kind: "none", reason: null });
  assert.equal(session.fakeOrderConfirmed, false); assert.deepEqual(f.counts(), { browserCloses: 1, agentCloses: 1 });
}));

test("wall deadline cancels hanging agent and preserves timeout receipt plus artifacts", async () => temporary(async (root) => {
  const p = await packet(), c = p.cases[1], f = fixture(root, [], { hanging: true });
  const session = await executeShopperSession(packetDirectory, { ...p, budget: { maxActions: 24, maxWallSeconds: 0.02 } }, c, "generic", root, f.dependencies);
  assert.equal(session.status, "timeout"); assert.match(session.error!, /wall-clock budget/);
  assert.equal(session.choice, null); assert.equal(session.steps.length, 1);
  assert.deepEqual(f.counts(), { browserCloses: 1, agentCloses: 1 });
  assert.equal(session.protocol.passed, false);
}));

test("action budget stops before a twenty-fifth normalized action or extra decision", async () => temporary(async (root) => {
  const p = await packet(), c = p.cases[1];
  const f = fixture(root, []);
  let decisions = 0, actions = 0;
  const open = f.dependencies.openBrowser;
  f.dependencies.openBrowser = async (...args) => ({ ...await open(...args), act: async () => {
    actions++; return { kind: "inspect", target: "Scroll fixture", confirmationVisible: false };
  } });
  f.dependencies.openAgent = () => ({ decide: async () => { decisions++; return { type: "scroll", direction: "down" }; }, close: async () => {} });
  const session = await executeShopperSession(packetDirectory, p, c, "generic", root, f.dependencies);
  assert.equal(session.status, "timeout"); assert.match(session.error!, /action budget/);
  assert.equal(session.steps.length, 24); assert.equal(actions, 23); assert.equal(decisions, 23);
}));

test("deadline during first observation retains unknown cart and a durable timeout receipt", async () => temporary(async (root) => {
  const p = await packet(), c = p.cases[1], f = fixture(root, [], { hangingObservation: true });
  const session = await executeShopperSession(packetDirectory, { ...p, budget: { maxActions: 24, maxWallSeconds: 0.02 } }, c, "generic", root, f.dependencies);
  assert.equal(session.status, "timeout"); assert.equal(session.browser.actualNavigation, true);
  assert.equal(session.isolation.initialCartEmpty, null); assert.equal(session.protocol.passed, false);
  assert.ok(session.protocol.deviations.some((deviation) => deviation.includes("Initial empty cart")));
  assert.equal(session.steps.length, 1); assert.equal(session.choice, null); assert.equal(session.fakeOrderConfirmed, false);
  assert.deepEqual(f.counts(), { browserCloses: 1, agentCloses: 0 });
  assert.equal(JSON.parse(await readFile(path.join(root, "session.json"), "utf8")).status, "timeout");
}));

test("observed nonempty initial cart retains technical failure without starting the shopper", async () => temporary(async (root) => {
  const p = await packet(), c = p.cases[1], f = fixture(root, [], { cartEmpty: false });
  const session = await executeShopperSession(packetDirectory, p, c, "generic", root, f.dependencies);
  assert.equal(session.status, "technical-failure"); assert.match(session.error!, /Initial empty cart/);
  assert.equal(session.isolation.initialCartEmpty, false); assert.equal(session.protocol.passed, false);
  assert.equal(session.steps.length, 1); assert.equal(session.choice, null); assert.equal(session.fakeOrderConfirmed, false);
  assert.deepEqual(f.counts(), { browserCloses: 1, agentCloses: 0 });
  assert.equal(JSON.parse(await readFile(path.join(root, "session.json"), "utf8")).status, "technical-failure");
}));

test("Chrome launch failure records zero invented actions and missing trace", async () => temporary(async (root) => {
  const p = await packet(), c = p.cases[1];
  const session = await executeShopperSession(packetDirectory, p, c, "generic", root, {
    openBrowser: async () => { throw new Error("Fixture Chrome unavailable"); },
    openAgent: () => { throw new Error("Agent must not start before Chrome"); },
  });
  assert.equal(session.status, "technical-failure"); assert.equal(session.browser.actualNavigation, false);
  assert.equal(session.steps.length, 0); assert.equal(session.choice, null);
  assert.equal(session.artifacts.find((a) => a.artifact.kind === "trace")?.status, "missing");
  assert.equal(JSON.parse(await readFile(path.join(root, "session.json"), "utf8")).status, "technical-failure");
}));

test("stdio fixture adapter receives no evaluator input and returns bounded JSON actions", async () => {
  const p = await packet(), assignment = buildShopperAssignment(p.cases[1], "generic");
  const code = `const r=require('node:readline').createInterface({input:process.stdin});r.on('line',l=>{const p=JSON.parse(l);if(p.assignment.gold||p.assignment.catalog)process.exit(9);console.log(JSON.stringify({type:'search',query:p.assignment.recommenderContext.query}));});`;
  const adapter = new StdioShopperAgent(process.execPath, ["-e", code], assignment, () => {});
  try { assert.deepEqual(await adapter.decide({ text: "fixture", controls: [], url: assignment.entryUrl, screenshotBase64: "" }, { actions: 24, wallMs: 1000 }),
    { type: "search", query: assignment.recommenderContext.query }); } finally { await adapter.close(); }
});

test("adapter exit during a large observation write rejects without an unhandled EPIPE", { timeout: 5000 }, async () => {
  const p = await packet(), assignment = buildShopperAssignment(p.cases[1], "generic");
  const adapter = new StdioShopperAgent(process.execPath, ["-e", "process.exit(0)"], assignment, () => {});
  try {
    await assert.rejects(adapter.decide({ text: "CPU pipe failure fixture", controls: [], url: assignment.entryUrl,
      screenshotBase64: "x".repeat(10000000) }, { actions: 24, wallMs: 1000 }), /EPIPE|adapter exited|write/i);
  } finally { await adapter.close(); }
});
