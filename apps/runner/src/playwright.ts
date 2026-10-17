import { chromium, type Browser, type BrowserContext, type Page } from "playwright";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import type { RunnerInput, RunnerOutput } from "./protocol.js";

const ORIGIN = "http://127.0.0.1:4312";
/** Fixed local demo only. Never accept caller URLs, generated code or real payment actions. */
export async function runBrowser(input: RunnerInput): Promise<RunnerOutput> {
  const output: RunnerOutput = {
    events: [],
    artifacts: [],
    result: {
      taskCompleted: false,
      failureReasons: [],
      notes: ["Local scripted browser task; synthetic, not human conversion evidence."],
    },
  };
  const emit = (kind: RunnerOutput["events"][number]["kind"], message: string) =>
    output.events.push({ kind, message });
  const fail = (error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    output.result.failureReasons.push(message);
    emit("failure", message);
    process.stderr.write(`${message}\n`);
  };
  await mkdir(input.artifactDirectory, { recursive: true });
  let browser: Browser | undefined;
  let context: BrowserContext | undefined;
  let page: Page | undefined;
  let tracing = false;
  try {
    browser = await chromium.launch({ headless: true });
    context = await browser.newContext({ serviceWorkers: "block" });
    await context.route("**/*", (route) => {
      const url = new URL(route.request().url());
      return url.origin === ORIGIN ? route.continue() : route.abort("blockedbyclient");
    });
    await context.routeWebSocket("**/*", (socket) => {
      if (new URL(socket.url()).origin === ORIGIN.replace("http:", "ws:")) socket.connectToServer();
      else void socket.close();
    });
    page = await context.newPage();
    page.setDefaultTimeout(10_000);
    await context.tracing.start({ screenshots: true, snapshots: true });
    tracing = true;
    const query = new URLSearchParams({
      headline: input.run.config.headline,
      showShipping: String(input.run.config.showShipping),
      recommendationOrder: input.run.config.recommendationOrder,
    });
    await page.goto(`${ORIGIN}/?${query}`, { waitUntil: "domcontentloaded" });
    emit("navigation", "Opened the allowlisted local demo storefront");
    await page.getByRole("button", { name: "Add to cart", exact: true }).first().click();
    emit("action", "Added one demo product to cart");
    await page.getByRole("button", { name: "View cart", exact: true }).click();
    emit("action", "Opened the local demo cart");
    await page.getByRole("button", { name: "Place demo order", exact: true }).click();
    await page.getByRole("heading", { name: "Demo order confirmed", exact: true }).waitFor();
    output.result.taskCompleted = true;
    emit(
      "observation",
      "The local fake order-confirmation screen was visible; no real payment or purchase submitted",
    );
  } catch (error) {
    fail(error);
  } finally {
    if (page) {
      try {
        await page.screenshot({
          path: path.join(input.artifactDirectory, "screenshot.png"),
          fullPage: true,
        });
        output.artifacts.push({
          kind: "screenshot",
          fileName: "screenshot.png",
          mimeType: "image/png",
        });
      } catch (error) {
        fail(error);
      }
    }
    if (context && tracing) {
      try {
        await context.tracing.stop({ path: path.join(input.artifactDirectory, "trace.zip") });
        output.artifacts.push({
          kind: "trace",
          fileName: "trace.zip",
          mimeType: "application/zip",
        });
      } catch (error) {
        fail(error);
      }
    }
    if (browser) {
      try {
        await browser.close();
      } catch (error) {
        fail(error);
      }
    }
  }
  await writeFile(
    path.join(input.artifactDirectory, "result.json"),
    JSON.stringify({ synthetic: true, runId: input.run.id, ...output.result }, null, 2),
  );
  output.artifacts.push({ kind: "result", fileName: "result.json", mimeType: "application/json" });
  return output;
}
