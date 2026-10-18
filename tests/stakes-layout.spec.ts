import { expect, test } from "@playwright/test";

for (const viewport of [{ width: 1536, height: 864 }, { width: 1920, height: 900 }]) {
  test(`last slide fits without scrolling at ${viewport.width}×${viewport.height}`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.goto("/#stakes");
    await expect(page.getByRole("heading", { name: "Slow or lost shoppers buy somewhere else" })).toBeVisible();
    await page.evaluate(() => document.fonts.ready);

    const layout = await page.evaluate(() => {
      const main = document.querySelector("main")!;
      const note = main.querySelector("ol + p")!;
      return {
        scrollHeight: document.documentElement.scrollHeight,
        viewportHeight: innerHeight,
        contentBottom: note.getBoundingClientRect().bottom,
        footerTop: document.querySelector("footer")!.getBoundingClientRect().top,
      };
    });
    expect(layout.scrollHeight).toBeLessThanOrEqual(layout.viewportHeight);
    expect(layout.contentBottom).toBeLessThanOrEqual(layout.footerTop);
  });
}
