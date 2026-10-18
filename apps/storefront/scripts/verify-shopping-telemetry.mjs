// Isolated synthetic build + real backend required. Never calls the model endpoint.
import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, readdir, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
const output = new URL("../../../data/session-trace-auto/", import.meta.url);
const root = path.resolve(fileURLToPath(new URL("../../../data/session-trace-auto-dist/", import.meta.url)));
const origin = "http://127.0.0.1:4329", api = "http://127.0.0.1:4330";
await mkdir(output, { recursive: true });
const server = createServer(async (req, res) => {
  try {
    const name = decodeURIComponent(new URL(req.url, origin).pathname);
    const target = path.resolve(root, `.${name === "/" ? "/index.html" : name}`);
    if (!target.startsWith(`${root}/`)) { res.writeHead(403); res.end(); return; }
    const body = await readFile(target);
    const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp" };
    res.setHeader("Content-Type", types[path.extname(target)] || "application/octet-stream"); res.end(body);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise((resolve, reject) => { server.once("error", reject); server.listen(4329, "127.0.0.1", resolve); });
const browser = await chromium.launch({ headless: true, args: ["--disable-gpu"] });
const receipts = [], receiptJobs = [], errors = [], contexts = [];
async function makePage(preference, dnt = false) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 1000 } }); contexts.push(context);
  await context.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if ([origin, api].includes(url.origin)) return route.continue();
    return route.abort();
  });
  await context.addInitScript(({ preference, dnt }) => {
    if (preference) localStorage.setItem("fieldwork.shopping-privacy.v1", preference);
    if (dnt) Object.defineProperty(navigator, "doNotTrack", { value: "1" });
    document.addEventListener("DOMContentLoaded", () => {
      const account = document.createElement("section"); account.id = "account-private"; account.textContent = "private-canary@example.invalid";
      for (const type of ["password", "email", "text"]) { const input = document.createElement("input"); input.type = type; input.value = "private-canary@example.invalid"; account.append(input); }
      document.body.append(account);
    });
  }, { preference, dnt });
  const page = await context.newPage(); page.on("pageerror", (error) => errors.push(error.stack || error.message));
  page.on("response", (response) => { if (response.url() === `${api}/api/shopping-events` && response.request().method() === "POST") receiptJobs.push((async () => {
    const batch = response.request().postDataJSON(); const receipt = await response.json(); receipts.push({ batch, status: response.status(), receipt });
  })()); });
  return page;
}
try {
  const page = await makePage();
  await page.goto(`${origin}/#collection?q=private-canary%40example.invalid`);
  await page.waitForFunction(() => window.fieldworkShopping, { timeout: 7000 });
  assert.equal(await page.getByRole("button", { name: "Start recording", exact: true }).count(), 0);
  await page.waitForTimeout(350);
  await page.locator("#shop-search").fill("private-canary@example.invalid");
  await page.getByRole("button", { name: "Search", exact: true }).click(); await page.waitForTimeout(350);
  await page.getByRole("button", { name: "Trail mug color: Cobalt", exact: true }).click(); await page.waitForTimeout(350);
  const mug = page.locator("article.product").filter({ has: page.getByRole("heading", { name: "Trail mug", exact: true }) });
  await mug.getByRole("button", { name: "Add to cart", exact: true }).click();
  await page.getByRole("button", { name: "View cart", exact: true }).click();
  await page.getByRole("button", { name: "Place demo order", exact: true }).click();
  await page.getByRole("heading", { name: "Demo order confirmed" }).waitFor();
  await page.evaluate(() => window.fieldworkShopping.flush());
  await page.waitForFunction(() => { const stats = window.fieldworkShopping.stats(); return stats.queued === 0 && !stats.inFlight; });
  await Promise.all(receiptJobs);
  assert.ok(receipts.length > 0);
  assert.ok(receipts.every(({ status, receipt }) => status === 202 && receipt.durableLocal === true), JSON.stringify(receipts.map(({ status, receipt }) => ({ status, receipt }))));
  const events = receipts.flatMap(({ batch }) => batch.events);
  for (const action of ["session-start", "catalog-view", "product-view", "search", "selection", "cart-add", "checkout-success"]) assert.ok(events.some((event) => event.action === action), `missing ${action}`);
  assert.ok(receipts.every(({ batch }) => batch.classification.environment === "test" && batch.classification.traffic === "synthetic"));
  assert.ok(!JSON.stringify(receipts).includes("private-canary"), "sensitive canary in upload");
  assert.ok(events.filter((event) => event.action === "checkout-success").every((event) => event.data.demo === true));
  const replay = receipts.flatMap(({ batch }) => batch.replay?.events ?? []);
  const backendReplay = [];
  for (const { batch } of receipts) if (batch.replay) {
    const stored = JSON.parse(await readFile(new URL(`../../../data/shopping-events/${batch.batchId}.replay`, import.meta.url), "utf8"));
    assert.equal(stored.schema, "fieldwork.sanitized-replay.v1"); backendReplay.push(...stored.events);
  }
  assert.equal(backendReplay.length, replay.length, "server changed replay sequence count");
  await writeFile(new URL("backend-replay-final.json", output), JSON.stringify({ schema: "fieldwork.sanitized-replay.v1", events: backendReplay }));
  assert.ok(replay.some((event) => event.type === 2), "missing replay snapshot");
  // Rebuild the exact uploaded sequence with the same library, scripts sandboxed.
  const replayAsset = (await readdir(path.join(root, "assets"))).find((name) => /^replay-.*\.js$/.test(name));
  assert.ok(replayAsset);
  const rebuiltProducts = await page.evaluate(async ({ events, asset }) => {
    const { Replayer } = await import(`/assets/${asset}`);
    const root = document.createElement("div"); root.setAttribute("data-trace-private", ""); document.body.append(root);
    const player = new Replayer(events, { root, showWarning: false, UNSAFE_replayCanvas: false });
    player.pause(0);
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const doc = root.querySelector("iframe")?.contentDocument;
    const count = doc?.querySelectorAll("article.product").length ?? 0;
    if (!doc?.querySelector("article.product img[src^='/products/']")) throw new Error("server sanitizer lost product image paths");
    player.destroy(); root.remove(); return count;
  }, { events: backendReplay, asset: replayAsset });
  assert.equal(rebuiltProducts, 16, "backend sanitized replay did not rebuild catalog");
  const beforeOptOut = receipts.length;
  await page.getByRole("button", { name: "Anonymous recording on", exact: true }).click();
  await page.getByRole("button", { name: "Stop sharing", exact: true }).click();
  await page.waitForFunction(() => !window.fieldworkShopping);
  await page.getByRole("button", { name: "Shopping privacy", exact: true }).click();
  await page.getByRole("button", { name: "Back to the collection", exact: true }).click();
  await page.getByRole("button", { name: "Search", exact: true }).click(); await page.waitForTimeout(300);
  assert.equal(receipts.length, beforeOptOut, "collection continued after opt-out");
  for (const [preference, dnt] of [["denied", false], [undefined, true]]) {
    const denied = await makePage(preference, dnt); let posts = 0, recorderRequests = 0;
    denied.on("request", (request) => { if (request.url().includes("/api/shopping-events")) posts++; if (/\/record-/.test(request.url())) recorderRequests++; });
    await denied.goto(origin); await denied.getByRole("heading", { level: 1 }).waitFor(); await denied.waitForTimeout(300);
    assert.equal(await denied.evaluate(() => typeof window.fieldworkShopping), "undefined"); assert.equal(posts, 0); assert.equal(recorderRequests, 0);
  }
  assert.deepEqual(errors, []);
  await writeFile(new URL("browser-receipts.json", output), JSON.stringify(receipts));
  const result = { synthetic: true, browser: await browser.version(), automaticWithoutStart: true, sessionIds: [...new Set(receipts.map(({ batch }) => batch.sessionId))],
    batchIds: receipts.map(({ batch }) => batch.batchId), batchCount: receipts.length, events: events.length, replayEvents: replay.length, rebuiltProducts,
    actions: [...new Set(events.map((event) => event.action))], privacyCanariesPassed: true, optOutStopsCollection: true, persistedOptOutNoRecorderOrUpload: true, doNotTrackNoRecorderOrUpload: true,
    backendReceipt: "durableLocal only; Tiger and training export require server-side verification", modelCalls: 0, pageErrors: errors };
  await writeFile(new URL("verification.json", output), JSON.stringify(result, null, 2)); console.log(JSON.stringify(result, null, 2));
} catch (error) { console.error(JSON.stringify({ error: error.message, pageErrors: errors, receipts: receipts.map(({status,receipt}) => ({status,receipt})) }, null, 2)); throw error; } finally { await browser.close(); await new Promise((resolve) => server.close(resolve)); }
