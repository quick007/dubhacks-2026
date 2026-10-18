/** Opt-in infrastructure smoke; this performs no shopper decisions or checkout. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { ChromeShopperBrowser } from "./shopper-chrome.js";

test("actual headed Chrome contexts expose visible controls and persist trace evidence", {
  skip: process.env.SHOPPER_CHROME_SMOKE !== "1", timeout: 60000,
}, async () => {
  const durable = process.env.SHOPPER_CHROME_ARTIFACTS;
  const root = durable ? path.resolve(durable) : await mkdtemp(path.join(os.tmpdir(), "shopper-chrome-smoke-"));
  if (durable) await mkdir(root);
  const receipts = [];
  try {
    for (let i = 0; i < 2; i++) {
      const directory = path.join(root, `context-${i + 1}`);
      const browser = await ChromeShopperBrowser.open("http://127.0.0.1:4312/", directory);
      try {
        const observation = await browser.observe();
        assert.equal(observation.cartEmpty, true);
        assert.ok(observation.controls.some((c) => c.name === "View cart"));
        assert.ok(!observation.controls.some((c) => c.name === "Demo controls" || c.name === "Rain jacket for campus"));
        assert.ok(observation.screenshotBase64.length > 1000);
        receipts.push({ contextId: browser.contextId, tabId: browser.tabId, chromeVersion: browser.browserVersion,
          origin: observation.url, initialCartEmpty: observation.cartEmpty, controls: observation.controls.length });
      } finally { await browser.close(); }
      const screenshot = await readFile(path.join(directory, "screenshot.png"));
      const trace = await readFile(path.join(directory, "trace.zip"));
      assert.equal(screenshot.subarray(1, 4).toString(), "PNG"); assert.equal(trace.subarray(0, 2).toString(), "PK");
      Object.assign(receipts.at(-1)!, { screenshot: { bytes: screenshot.length, sha256: createHash("sha256").update(screenshot).digest("hex") },
        trace: { bytes: trace.length, sha256: createHash("sha256").update(trace).digest("hex") } });
    }
    assert.notEqual(receipts[0].contextId, receipts[1].contextId);
    assert.notEqual(receipts[0].tabId, receipts[1].tabId);
    await writeFile(path.join(root, "infrastructure-smoke.json"), JSON.stringify({ synthetic: true, shopperAgentInvoked: false,
      modelInvoked: false, checkoutAttempted: false, purpose: "Headed Chrome isolation/observation/artifact infrastructure only", receipts }, null, 2) + "\n", { flag: "wx" });
  } finally { if (!durable) await rm(root, { recursive: true, force: true }); }
});
