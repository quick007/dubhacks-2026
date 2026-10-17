import { expect, test } from "@playwright/test";

test.beforeEach(async ({ request }) => {
  await expect
    .poll(async () => {
      try {
        return (await request.get("http://127.0.0.1:4311/health")).ok();
      } catch {
        return false;
      }
    })
    .toBe(true);
  await expect
    .poll(async () => {
      try {
        return (await request.get("http://127.0.0.1:4312")).ok();
      } catch {
        return false;
      }
    })
    .toBe(true);
});

test("dashboard runs deterministic synthetic comparisons across both persona splits", async ({
  page,
  request,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Run mock comparison" }).click();
  await expect
    .poll(
      async () => {
        const runs = await (await request.get("http://127.0.0.1:4311/api/runs")).json();
        return runs.filter(
          (run: { adapter: string; status: string }) =>
            run.adapter === "mock" && run.status === "completed",
        ).length;
      },
      { timeout: 30_000 },
    )
    .toBeGreaterThanOrEqual(8);
  await expect(page.getByTestId("comparison")).toContainText(/held-out/i);
  await expect(page.getByTestId("comparison")).toContainText(/development/i);
  await expect(page.getByTestId("run-card").first()).toContainText("completed");
});

test("local storefront only submits a fake demo order", async ({ page }) => {
  await page.goto("http://127.0.0.1:4312/?showShipping=true&recommendationOrder=budget");
  await page.getByRole("button", { name: "Add to cart" }).first().click();
  await page.getByRole("button", { name: "View cart" }).click();
  await page.getByRole("button", { name: "Place demo order" }).click();
  await expect(page.getByRole("heading", { name: "Demo order confirmed" })).toBeVisible();
});

test("API browser adapter stores a trace and screenshot from fake checkout", async ({
  request,
}) => {
  const catalog = await (await request.get("http://127.0.0.1:4311/api/catalog")).json();
  const submitted = await request.post("http://127.0.0.1:4311/api/runs", {
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
    .poll(
      async () =>
        (await (await request.get(`http://127.0.0.1:4311/api/runs/${run.id}`)).json()).status,
      { timeout: 30_000 },
    )
    .toBe("completed");
  const result = await (await request.get(`http://127.0.0.1:4311/api/runs/${run.id}`)).json();
  expect(result.synthetic).toBe(true);
  expect(result.evaluation.taskCompleted).toBe(true);
  expect(result.evaluation.failureReasons).toEqual([]);
  expect(result.artifacts.map((artifact: { kind: string }) => artifact.kind)).toEqual(
    expect.arrayContaining(["trace", "screenshot", "result"]),
  );
  for (const artifact of result.artifacts)
    expect((await request.get(`http://127.0.0.1:4311/api/artifacts/${artifact.id}`)).ok()).toBe(
      true,
    );
});
