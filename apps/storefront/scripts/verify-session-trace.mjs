// Run after `pnpm --filter @storefront/storefront build` and loopback `vp preview`.
// One isolated, headless, sequential synthetic browser; never calls a model.
import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
const origin = process.env.TRACE_TEST_ORIGIN || "http://127.0.0.1:4329";
assert.match(new URL(origin).hostname, /^(127\.0\.0\.1|localhost)$/);
const output = new URL("../../../data/session-trace/", import.meta.url);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, args: ["--disable-gpu"] });
const context = await browser.newContext({ viewport: { width: 1280, height: 1000 } });
await context.route("**/*", async (route) => {
  if (new URL(route.request().url()).origin === origin) await route.continue();
  else await route.abort();
});
const page = await context.newPage();
const errors = []; page.on("pageerror", (error) => errors.push(error.message));
const requested = []; page.on("request", (request) => requested.push(request.url()));
const wait = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
try {
  await page.goto(origin); await page.getByRole("heading", { level: 1 }).waitFor(); await wait();
  assert.equal(await page.evaluate(() => typeof window.fieldworkTrace), "undefined");
  assert.ok(!requested.some((url) => /TraceControls-|record-|replay-/.test(url)), "disabled path loaded tracing chunk");
  const disabledResources = [...requested];
  await page.goto(`${origin}/?trace=demo`);
  await page.waitForFunction(() => window.fieldworkTrace);
  assert.ok(!requested.some((url) => /\/record-|\/replay-/.test(url)), "config loaded recorder before Start");
  const canary = "private-canary@example.invalid";
  await page.evaluate((secret) => {
    document.title = secret;
    const account = document.createElement("section"); account.id = "account-private"; account.textContent = secret;
    const password = document.createElement("input"); password.type = "password"; password.value = secret;
    const email = document.createElement("input"); email.type = "email"; email.value = secret;
    const text = document.createElement("textarea"); text.value = secret;
    account.append(password, email, text); document.body.append(account);
    document.querySelector("main").setAttribute("data-sensitive-test", secret);
  }, canary);
  await page.getByRole("button", { name: "Session trace", exact: true }).click();
  await page.getByRole("button", { name: "Start recording", exact: true }).click();
  await page.waitForFunction(() => window.fieldworkTrace.status() === "Recording");
  await page.getByRole("button", { name: "Session trace", exact: true }).click();
  await page.locator("#shop-search").fill(canary);
  await page.getByRole("button", { name: "Search", exact: true }).click(); await wait();
  await page.getByRole("button", { name: "Trail mug color: Cobalt", exact: true }).click();
  const mug = page.locator("article.product").filter({ has: page.getByRole("heading", { name: "Trail mug", exact: true }) });
  await mug.getByRole("button", { name: "Add to cart", exact: true }).click(); await wait();
  await page.getByRole("button", { name: "View cart", exact: true }).click();
  await page.getByRole("heading", { name: "Your demo cart" }).waitFor();
  await page.evaluate((secret) => { const el = document.createElement("p"); el.textContent = secret; document.querySelector(".cart-page").append(el); }, canary);
  await page.getByRole("button", { name: "Place demo order", exact: true }).click();
  await page.getByRole("heading", { name: "Demo order confirmed" }).waitFor(); await wait();
  const trace = await page.evaluate(() => { window.fieldworkTrace.stop(); return window.fieldworkTrace.getTrace(); });
  assert.equal(trace.schema, "fieldwork.demo-session.v1");
  assert.ok(!JSON.stringify(trace).includes(canary), "privacy canary leaked");
  assert.ok(!JSON.stringify(trace).includes("q="), "query URL leaked");
  for (const action of ["product-view", "search", "selection", "cart-add", "checkout-success"]) assert.ok(trace.events.some((event) => event.action === action), `missing ${action}`);
  assert.ok(trace.events.some((event) => event.action === "cart-add" && event.data.variantId === "mug-cobalt"));
  assert.ok(trace.replay.events.some((event) => event.type === 2));
  await page.evaluate(() => window.fieldworkTrace.replay());
  await page.waitForFunction(() => document.querySelector("#trace-replay iframe")?.contentDocument?.querySelectorAll("article.product").length > 0);
  const replayProducts = await page.evaluate(() => document.querySelector("#trace-replay iframe").contentDocument.querySelectorAll("article.product").length);
  assert.ok(replayProducts > 0, "replay failed to reconstruct products");
  await page.getByRole("button", { name: "Close replay", exact: true }).click();
  await page.evaluate(() => window.fieldworkTrace.clear());
  assert.equal(await page.evaluate(() => window.fieldworkTrace.getTrace()), null);
  await writeFile(new URL("synthetic-session.v1.json", output), JSON.stringify(trace));
  // Warmed main-thread CPU proxy: identical 1000 DOM mutations+microtask flush;
  // 3 pairs, alternating order, single page/browser; excludes image/network/startup.
  async function mutationBatch(enabled) {
    await page.goto(`${origin}/${enabled ? "?trace=demo" : ""}`);
    await page.getByRole("heading", { level: 1 }).waitFor(); await wait();
    if (enabled) { await page.waitForFunction(() => window.fieldworkTrace); await page.evaluate(() => window.fieldworkTrace.start()); }
    const elapsed = await page.evaluate(async () => {
      const node = document.createElement("span"); document.querySelector("main").append(node);
      const start = performance.now();
      for (let i = 0; i < 1000; i++) { node.textContent = `synthetic ${i}`; if (i % 20 === 0) await Promise.resolve(); }
      await Promise.resolve(); return performance.now() - start;
    });
    if (enabled) await page.evaluate(() => window.fieldworkTrace.clear());
    return elapsed;
  }
  const samples = { disabled: [], enabled: [] };
  for (let i = 0; i < 3; i++) for (const enabled of (i % 2 ? [true, false] : [false, true])) samples[enabled ? "enabled" : "disabled"].push(await mutationBatch(enabled));
  const median = (xs) => [...xs].sort((a, b) => a - b)[1];
  const result = { synthetic: true, browser: await browser.version(), method: "3 alternating sequential pairs; 1000 text mutations with microtask flush every20; warmed page, headless disable-gpu; no model calls; CPU proxy, not user latency", samplesMs: samples,
    medianDisabledMs: median(samples.disabled), medianEnabledMs: median(samples.enabled), deltaMs: median(samples.enabled) - median(samples.disabled),
    replayProducts, shoppingEvents: trace.events.length, replayEvents: trace.replay.events.length, exportBytes: Buffer.byteLength(JSON.stringify(trace)), disabledTracingRequests: disabledResources.filter((url) => /TraceControls-|record-|replay-/.test(url)).length, privacyCanariesPassed: true, pageErrors: errors };
  assert.deepEqual(errors, []);
  await writeFile(new URL("verification.json", output), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
} finally { await browser.close(); }
