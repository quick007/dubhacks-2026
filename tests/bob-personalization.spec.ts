import { expect, test } from "@playwright/test";
import { CatalogSchema, RunSchema } from "../packages/contracts/src";
import { readFileSync } from "node:fs";
import { PERSONALIZATION_STEP_HEIGHT } from "../apps/web/src/lib/personalization";

// UI coverage uses local catalog fixtures; it creates no runs in the developer's API.
const catalog = CatalogSchema.parse(
  JSON.parse(readFileSync(new URL("../apps/api/fixtures/catalog.json", import.meta.url), "utf8")),
);
const persona = catalog.personas.find((item) => item.id === "dev-sofa")!;
const runs = catalog.variants.map((variant) =>
  RunSchema.parse({
    id: `bob-${variant.id}`,
    experimentId: catalog.experiment.id,
    baselineVariantId: catalog.experiment.baselineVariantId,
    variantId: variant.id,
    personaId: persona.id,
    personaSplit: persona.split,
    persona,
    config: variant.config,
    seed: 42,
    adapter: "mock",
    synthetic: true,
    contractVersion: 1,
    status: "completed",
    createdAt: "2026-10-18T00:00:00.000Z",
    events: [],
    artifacts: [],
    proposedChanges: [],
    evaluation: {
      runId: `bob-${variant.id}`,
      synthetic: true,
      taskCompleted: !variant.isBaseline,
      score: variant.isBaseline ? 0 : 1,
      failureReasons: [],
      notes: [],
      findability: {
        outcome: variant.isBaseline ? "abandoned" : "purchased",
        requirements: [],
        steps: 3,
        optimalSteps: 3,
        searches: 0,
        emptySearches: 0,
        revisits: 0,
      },
    },
  }),
);

test.beforeEach(async ({ page }) => {
  await page.route("**/api/catalog", (route) => route.fulfill({ json: catalog }));
  await page.route("**/api/runs", (route) => route.fulfill({ json: runs }));
});

test("scroll reveals one change at a time and reverses to the original", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/#fix");
  const story = page.getByRole("region", { name: "Bob's page changes" });
  const caption = page.getByRole("complementary", { name: "Current change" });
  await expect(story).toHaveAttribute("data-step", "0");
  await expect(caption.getByRole("heading")).toHaveText("The original");
  expect(await story.evaluate((element) => getComputedStyle(element).scrollbarWidth)).toBe("none");
  await expect(
    page.getByRole("button", { name: "Previous change", includeHidden: true }),
  ).toBeHidden();
  await story.hover();
  const height = await story.evaluate((element) => element.clientHeight);
  await page.mouse.wheel(0, height * PERSONALIZATION_STEP_HEIGHT);
  await expect(story).toHaveAttribute("data-step", "1");
  await page.mouse.wheel(0, -height * PERSONALIZATION_STEP_HEIGHT);
  await expect(story).toHaveAttribute("data-step", "0");
  for (const [index, title] of [
    "His request",
    "His preferences",
    "His new couch",
    "The color match",
  ].entries()) {
    await story.evaluate(
      (element, { step, spacing }) =>
        element.scrollTo({ top: step * element.clientHeight * spacing, behavior: "instant" }),
      { step: index + 1, spacing: PERSONALIZATION_STEP_HEIGHT },
    );
    await expect(story).toHaveAttribute("data-step", String(index + 1));
    await expect(caption.getByRole("heading")).toHaveText(title);
    await expect(caption.getByRole("heading")).toHaveCount(1);
  }
  await expect(page.getByRole("button", { name: "Next change", includeHidden: true })).toBeHidden();
  await expect
    .poll(() =>
      story.locator('img[src$="match.jpg"]').evaluate((image) => getComputedStyle(image).opacity),
    )
    .toBe("1");
  expect(
    await story
      .locator("img")
      .evaluateAll((images) => images.every((image) => image.complete && image.naturalWidth > 0)),
  ).toBe(true);
  await story.evaluate((element) => element.scrollTo({ top: 0, behavior: "instant" }));
  await expect(story).toHaveAttribute("data-step", "0");
  await expect(caption.getByRole("heading")).toHaveText("The original");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight)).toBe(
    true,
  );
});

test("a short scroll leaves time to read the current change", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/#fix");
  const story = page.getByRole("region", { name: "Bob's page changes" });
  await story.hover();
  const distance = await story.evaluate((element) => element.clientHeight * 0.6);
  await page.mouse.wheel(0, distance);
  await expect.poll(() => story.evaluate((element) => element.scrollTop)).toBeCloseTo(distance, 0);
  await expect(story).toHaveAttribute("data-step", "0");
  await page.mouse.wheel(0, distance);
  await expect(story).toHaveAttribute("data-step", "1");
});

for (const width of [390, 1024, 1440]) {
  test(`captions and arrows stay in place at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/#fix");
    const story = page.getByRole("region", { name: "Bob's page changes" });
    const caption = page.getByRole("complementary", { name: "Current change" });
    await expect(story).toHaveAttribute("data-step", "0");
    await page.evaluate(async () => {
      await document.fonts.ready;
      await Promise.all(document.getAnimations().map((animation) => animation.finished));
    });
    const positions = () =>
      caption.evaluate((element) => {
        const visible = (selector: string) =>
          Array.from(element.querySelectorAll(selector))
            .find((node) => getComputedStyle(node).visibility !== "hidden")!
            .getBoundingClientRect();
        return [visible("h3").y, visible("p").y, visible("nav").y];
      });
    const original = await positions();
    for (const step of [1, 2, 3, 4, 0]) {
      await story.evaluate(
        (element, { step, spacing }) =>
          element.scrollTo({ top: step * element.clientHeight * spacing, behavior: "instant" }),
        { step, spacing: PERSONALIZATION_STEP_HEIGHT },
      );
      await expect(story).toHaveAttribute("data-step", String(step));
      const current = await positions();
      current.forEach((position, index) =>
        expect(Math.abs(position - original[index])).toBeLessThan(1),
      );
      expect(
        await caption
          .getByRole("status")
          .evaluate((element) =>
            Array.from(element.querySelectorAll("*")).every(
              (node) => getComputedStyle(node).transform === "none",
            ),
          ),
      ).toBe(true);
      await expect(caption.getByRole("button")).toHaveCount(step === 0 || step === 4 ? 1 : 2);
    }
    await expect(caption.getByRole("navigation")).not.toContainText(/\d/);
  });
}

test("mobile buttons, native keyboard scrolling, and reduced motion work", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/#fix");
  const story = page.getByRole("region", { name: "Bob's page changes" });
  await page.getByRole("button", { name: "Next change" }).click();
  await expect(story).toHaveAttribute("data-step", "1");
  await page.getByRole("button", { name: "Next change" }).press("Enter");
  await expect(story).toHaveAttribute("data-step", "2");
  await page.getByRole("button", { name: "Previous change" }).click();
  await expect(story).toHaveAttribute("data-step", "1");
  await story.focus();
  await page.keyboard.press("End");
  await expect(story).toHaveAttribute("data-step", "4");
  await expect(page).toHaveURL(/#fix$/);
  expect(
    await story
      .locator("img")
      .first()
      .evaluate((image) => getComputedStyle(image).transitionDuration),
  ).toBe("0s");
  expect(await page.getByRole("status").evaluate((element) => element.getAnimations().length)).toBe(
    0,
  );
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await story.focus();
  await page.keyboard.press("Home");
  await expect(story).toHaveAttribute("data-step", "0");
});

test("up and down arrows change the walkthrough without moving slides", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/#fix");
  const story = page.getByRole("region", { name: "Bob's page changes" });
  const caption = page.getByRole("complementary", { name: "Current change" });
  await expect(caption.getByRole("heading")).toHaveText("The original");
  await page.keyboard.press("ArrowUp");
  await expect(story).toHaveAttribute("data-step", "0");
  for (let step = 1; step <= 4; step++) {
    await page.keyboard.press("ArrowDown");
    await expect(story).toHaveAttribute("data-step", String(step));
  }
  await page.keyboard.press("ArrowDown");
  await expect(story).toHaveAttribute("data-step", "4");
  await page.getByRole("button", { name: "Previous change" }).focus();
  await page.keyboard.press("ArrowUp");
  await expect(caption.getByRole("heading")).toHaveText("His new couch");
  await expect(page).toHaveURL(/#fix$/);
});
