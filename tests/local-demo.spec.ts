import { expect, test as base } from "@playwright/test";
import { writeFile, unlink } from "node:fs/promises";
import { LOCAL_ORIGINS, LOCAL_PROXY_PORT } from "../packages/contracts/src/local";

// Browser navigation uses named URLs. Node's macOS resolver can require explicit
// loopback transport; the Host header still exercises the named Portless route.
const test = base.extend({
  request: async ({ playwright }, use) => {
    const request = await playwright.request.newContext({
      baseURL: `http://127.0.0.1:${LOCAL_PROXY_PORT}`,
      extraHTTPHeaders: { host: new URL(LOCAL_ORIGINS.api).host },
    });
    await use(request);
    await request.dispose();
  },
});

test.beforeEach(async ({ request }) => {
  await expect
    .poll(async () => {
      try {
        return (await request.get("/health")).ok();
      } catch {
        return false;
      }
    })
    .toBe(true);
  await expect
    .poll(async () => {
      try {
        return (
          await request.get("/", { headers: { host: new URL(LOCAL_ORIGINS.storefront).host } })
        ).ok();
      } catch {
        return false;
      }
    })
    .toBe(true);
});

test("deck runs shoppers, tells the featured shopper's story and shows the held-out result", async ({
  page,
  request,
}) => {
  await page.goto("/#title");
  await expect(page.getByRole("button", { name: /Run a round/ })).toHaveCount(0);
  await page.getByRole("button", { name: /Slide \d+: All sessions/ }).click();
  await page.getByRole("button", { name: /Run a round/ }).click();
  await page.getByRole("button", { name: "Slide 1: Title", exact: true }).click();
  await expect
    .poll(
      async () => {
        const runs = await (await request.get("/api/runs")).json();
        return runs.filter(
          (run: { adapter: string; status: string }) =>
            run.adapter === "mock" && run.status === "completed",
        ).length;
      },
      { timeout: 30_000 },
    )
    .toBeGreaterThanOrEqual(20);
  await expect(page.getByText("AI shoppers browse your store")).toBeVisible();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("heading", { name: "Two shoppers, one store" })).toBeVisible();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("heading", { name: /^Meet / })).toBeVisible();
  await page.goto("/#sessions");
  await expect(page.getByText("Running at once")).toBeVisible();
  await page.goto("/#results");
  await expect(page.getByText("new shoppers bought something that fits")).toBeVisible({
    timeout: 10_000,
  });
  await page.goto("/#fix");
  const walkthrough = page.getByRole("region", { name: "Bob's page changes" });
  await walkthrough.focus();
  await page.keyboard.press("ArrowDown");
  await expect(walkthrough).toHaveAttribute("data-step", "1");
  await expect(page.getByRole("main", { name: "Personalized page", exact: true })).toBeVisible();
  await page.keyboard.press("ArrowUp");
  await expect(walkthrough).toHaveAttribute("data-step", "0");
  await expect(page.getByRole("main", { name: "Personalized page", exact: true })).toBeVisible();
  await page.goto("/#struggle");
  await expect(page.locator("main img").first()).toBeVisible();
  await page.getByRole("button", { name: "See every step" }).click();
  await expect(page.getByRole("dialog")).toContainText("Task failed");
});

test("local storefront only submits a fake demo order", async ({ page }) => {
  await page.goto(`${LOCAL_ORIGINS.storefront}/?showShipping=true&recommendationOrder=budget`);
  await page.getByRole("button", { name: "Add to cart" }).first().click();
  await page.getByRole("button", { name: "View cart" }).click();
  await page.getByRole("button", { name: "Place demo order" }).click();
  await expect(page.getByRole("heading", { name: "Demo order confirmed" })).toBeVisible();
});

test("API browser adapter stores a trace and screenshot from fake checkout", async ({
  request,
}) => {
  const catalog = await (await request.get("/api/catalog")).json();
  const submitted = await request.post("/api/runs", {
    data: {
      experimentId: catalog.experiment.id,
      variantId: catalog.experiment.baselineVariantId,
      personaId: catalog.personas[0].id,
      adapter: "playwright",
    },
  });
  expect(submitted.status()).toBe(202);
  const run = await submitted.json();
  await expect
    .poll(async () => (await (await request.get(`/api/runs/${run.id}`)).json()).status, {
      timeout: 30_000,
    })
    .toBe("completed");
  const result = await (await request.get(`/api/runs/${run.id}`)).json();
  expect(result.synthetic).toBe(true);
  expect(result.evaluation.taskCompleted).toBe(true);
  expect(result.evaluation.failureReasons).toEqual([]);
  expect(result.artifacts.map((artifact: { kind: string }) => artifact.kind)).toEqual(
    expect.arrayContaining(["trace", "screenshot", "result"]),
  );
  for (const artifact of result.artifacts)
    expect((await request.get(`/api/artifacts/${artifact.id}`)).ok()).toBe(true);
});

test("Vite+ HMR updates CSS through both named Portless routes", async ({ page }) => {
  for (const [app, origin] of [
    ["web", LOCAL_ORIGINS.dashboard],
    ["storefront", LOCAL_ORIGINS.storefront],
  ]) {
    const fileName = `__portless_hmr_${process.pid}.css`;
    const file = new URL(`../apps/${app}/src/${fileName}`, import.meta.url);
    await writeFile(file, ":root { --portless-smoke-probe: before; }", { flag: "wx" });
    try {
      await page.goto(origin);
      await page.evaluate(async (moduleURL) => {
        await import(/* @vite-ignore */ moduleURL);
      }, `/src/${fileName}`);
      const readProbe = () =>
        page.evaluate(() =>
          getComputedStyle(document.documentElement)
            .getPropertyValue("--portless-smoke-probe")
            .trim(),
        );
      await expect.poll(readProbe).toBe("before");
      await page.evaluate(() => {
        (window as Window & { hmrMarker?: string }).hmrMarker = "preserved";
      });
      await writeFile(file, ":root { --portless-smoke-probe: after; }");
      await expect.poll(readProbe).toBe("after");
      expect(await page.evaluate(() => (window as Window & { hmrMarker?: string }).hmrMarker)).toBe(
        "preserved",
      );
    } finally {
      await unlink(file);
    }
  }
});
