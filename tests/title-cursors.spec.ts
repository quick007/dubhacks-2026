import { expect, test } from "@playwright/test";

test("title shoppers move visibly and use contrasting colored pointers", async ({ page }) => {
  await page.goto("/#title");
  const cursor = page.locator("[data-shopper]").first();
  await expect(cursor).toBeVisible();
  const result = await cursor.evaluate((element) => {
    const animation = element.getAnimations()[0];
    if (!animation) throw new Error("Shopper has no browsing animation");
    animation.pause();
    animation.currentTime = 1200;
    const before = element.querySelector("svg")!.getBoundingClientRect();
    animation.currentTime = 3000;
    const after = element.querySelector("svg")!.getBoundingClientRect();
    const fill = getComputedStyle(element.querySelector("svg")!).fill;
    const rgb = fill.match(/[\d.]+/g)!.map(Number);
    const linear = rgb.map((value) => {
      const s = value / 255;
      return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    });
    return {
      distance: Math.hypot(after.x - before.x, after.y - before.y),
      saturation: Math.max(...rgb) - Math.min(...rgb),
      whiteContrast: 1.05 / (0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2] + 0.05),
    };
  });
  expect(result.distance).toBeGreaterThan(40);
  expect(result.saturation).toBeGreaterThan(100);
  expect(result.whiteContrast).toBeGreaterThan(4.5);
});

test("mobile keeps the featured pointer and its name within the thumbnail", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#title");
  const cursor = page.locator("[data-shopper]").first();
  await expect(cursor).toBeVisible();
  const contained = await cursor.evaluate((element) => {
    const animation = element.getAnimations()[0];
    animation.pause();
    const frame = element.parentElement!.getBoundingClientRect();
    return [0, 3000, 6000, 8999].every((time) => {
      animation.currentTime = time;
      return [...element.querySelectorAll("svg, span")].every((part) => {
        const rect = part.getBoundingClientRect();
        return rect.left >= frame.left && rect.right <= frame.right &&
          rect.top >= frame.top && rect.bottom <= frame.bottom;
      });
    });
  });
  expect(contained).toBe(true);
});

test("reduced motion leaves the illustrated shoppers stationary", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/#title");
  const cursor = page.locator("[data-shopper]").first();
  await expect(cursor).toBeVisible();
  expect(await cursor.evaluate((element) => element.getAnimations().length)).toBe(0);
});
