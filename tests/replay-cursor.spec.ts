import { expect, test } from "@playwright/test";
import { LOCAL_ORIGINS, LOCAL_PROXY_PORT } from "../packages/contracts/src/local";

test("default-page replay visibly browses with a blue shopper cursor and pauses", async ({
  page,
  playwright,
}) => {
  const api = await playwright.request.newContext({
    baseURL: `http://127.0.0.1:${LOCAL_PROXY_PORT}`,
    extraHTTPHeaders: { host: new URL(LOCAL_ORIGINS.api).host },
  });
  try {
    const catalog = await (await api.get("/api/catalog")).json();
    const created = await api.post("/api/runs", {
      data: {
        experimentId: catalog.experiment.id,
        personaId: "dev-sofa",
        variantId: catalog.experiment.baselineVariantId,
        adapter: "mock",
      },
    });
    expect(created.ok()).toBe(true);
    const run = await created.json();
    await expect
      .poll(
        async () => {
          const runs = await (await api.get("/api/runs")).json();
          return runs.find((item: { id: string }) => item.id === run.id)?.status;
        },
        { timeout: 30_000 },
      )
      .toBe("completed");

    await page.goto("/#struggle");
    await expect(page.getByRole("heading", { name: "Bob on the default page" })).toBeVisible();
    const cursor = page.locator(".replay-cursor");
    await expect(cursor).toBeVisible();
    const movement = await cursor.evaluate((element) => {
      const animation = element.getAnimations()[0];
      if (!animation) throw new Error("Replay cursor has no browsing motion");
      animation.pause();
      animation.currentTime = 300;
      const first = element.querySelector("svg")!.getBoundingClientRect();
      animation.currentTime = 1700;
      const next = element.querySelector("svg")!.getBoundingClientRect();
      return {
        distance: Math.hypot(next.x - first.x, next.y - first.y),
        fill: getComputedStyle(element.querySelector("svg")!).fill,
      };
    });
    expect(movement.distance).toBeGreaterThan(60);
    expect(movement.fill).toBe("rgb(36, 72, 214)");
    await page.getByRole("button", { name: "Pause", exact: true }).click();
    expect(await cursor.evaluate((element) => element.getAnimations()[0].playState)).toBe("paused");
    expect(await cursor.evaluate((element) => element.getAnimations()[0].currentTime)).toBe(1700);

    // Inspect the first product row, then the row after scrolling. Pixel regions are
    // measured from the 1280×800 fixture photos, independently of the motion code.
    const slider = page.getByRole("slider", { name: "Step", exact: true });
    // The Base UI slider control spans the track; its midpoint selects step 2 of 3.
    const control = page.locator("[data-no-swipe]").first();
    const bounds = (await control.boundingBox())!;
    await page.mouse.click(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
    await expect(page.getByText("Step 2 of 3", { exact: true })).toBeVisible();
    await assertPhotoStops(150, 530, 600, 650);
    await slider.focus();
    await page.keyboard.press("End");
    await expect(page.getByText("Step 3 of 3", { exact: true })).toBeVisible();
    await assertPhotoStops(60, 440, 510, 550);

    async function assertPhotoStops(
      photoTop: number,
      photoBottom: number,
      nameTop: number,
      nameBottom: number,
    ) {
      const stops = await cursor.evaluate((element) => {
        const animation = element.getAnimations()[0];
        animation.pause();
        const frame = element.parentElement!.getBoundingClientRect();
        const pointer = element.querySelector("svg")!;
        return [1080, 1920, 3000, 3840, 4920, 5820].map((time) => {
          animation.currentTime = time;
          const rect = pointer.getBoundingClientRect();
          return {
            x: ((rect.left + 4 - frame.left) / frame.width) * 1280,
            y: ((rect.top + 4 - frame.top) / frame.height) * 800,
          };
        });
      });
      for (const [index, stop] of stops.entries()) {
        const column = Math.floor(index / 2);
        expect(stop.x).toBeGreaterThan([40, 449, 859][column]);
        expect(stop.x).toBeLessThan([421, 830, 1240][column]);
        expect(stop.y).toBeGreaterThan(index % 2 ? nameTop : photoTop);
        expect(stop.y).toBeLessThan(index % 2 ? nameBottom : photoBottom);
      }
    }
    await page.emulateMedia({ reducedMotion: "reduce" });
    await expect(cursor).toBeVisible();
    await expect.poll(() => cursor.evaluate((element) => element.getAnimations().length)).toBe(0);
  } finally {
    await api.dispose();
  }
});
