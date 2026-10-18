import { test, expect } from "@playwright/test";

test("normal personalized shop supports focus, Escape, keyboard switching and rapid A→B→A without stale results", async ({ page }) => {
  const requests: unknown[] = [];
  page.on("request",request => {
    if (request.method() === "POST" && request.url() === "http://127.0.0.1:8767/personalize") requests.push(request.postDataJSON());
  });
  await page.goto("/#collection");
  const trigger = page.getByRole("button",{name:/Shopping as Bob\. Switch person/});
  await expect(trigger).toBeVisible();
  await expect.poll(() => requests.length).toBe(1);
  await expect(page.locator(".demo-banner")).toHaveCount(0);
  await trigger.focus();await page.keyboard.press("Enter");
  const panel = page.getByRole("dialog",{name:"Who’s shopping?"});
  await expect(panel).toBeVisible();
  await expect(panel.locator('button[aria-pressed="true"]')).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(panel).toBeHidden();await expect(trigger).toBeFocused();
  await page.keyboard.press("Enter");
  const maya = panel.getByRole("button",{name:/^Maya/});
  await page.keyboard.press("Tab");await expect(maya).toBeFocused();await page.keyboard.press("Enter");
  await expect(maya).toHaveAttribute("aria-pressed","true");
  await expect.poll(() => requests.length).toBe(2);
  const bob = panel.getByRole("button",{name:/^Bob/});
  await page.keyboard.press("Shift+Tab");await expect(bob).toBeFocused();await page.keyboard.press("Enter");
  await expect(bob).toHaveAttribute("aria-pressed","true");
  await expect.poll(() => requests.length).toBe(3);
  expect(requests[0]).toEqual(requests[2]);expect(requests[0]).not.toEqual(requests[1]);
  await page.keyboard.press("Escape");await expect(trigger).toBeFocused();
  const currentGeneration = await page.evaluate(() => performance.getEntriesByType("mark")
    .filter(mark => mark.name.startsWith("fieldwork.local-model.request.") && mark.name.endsWith(".start"))
    .map(mark => (mark as PerformanceMark).detail.generation).at(-1));
  expect(Number.isInteger(currentGeneration)).toBe(true);
  await page.waitForFunction(generation => {
    const entries = JSON.parse(document.documentElement.getAttribute("data-local-model-readiness") || "[]");
    return entries.some((entry: {detail?:{stage?:string;generation?:number}}) => entry.detail?.stage === "complete" && entry.detail.generation === generation);
  }, currentGeneration);
  const latest = await page.evaluate(() => {
    const entries = JSON.parse(document.documentElement.getAttribute("data-local-model-readiness") || "[]");
    return entries.filter((entry: {detail:{stage:string}}) => entry.detail.stage === "complete").at(-1);
  });
  expect(latest.detail.acceptedFields).toBe(latest.detail.requiredFields);
  expect(latest.detail.copySource.freshPerRequestGeneration).toBe(true);
  const idsBefore = await page.locator(".products h3").allTextContents();
  await page.waitForTimeout(500); // Observe delayed old generation delivery without making another request.
  await expect(trigger).toBeVisible();
  expect(await page.locator(".products h3").allTextContents()).toEqual(idsBefore);
  await page.getByRole("button",{name:"View cart",exact:true}).click();
  await expect(page.getByRole("heading",{name:"Your cart",exact:true})).toBeVisible();
  await page.getByRole("button",{name:"Continue browsing"}).click();
  await expect(trigger).toBeVisible();
});

test("explicit generic baseline has no profile controls and sends no native model POST", async ({ page }) => {
  const posts: string[] = [];
  page.on("request",request => {if(request.method() === "POST" && /127\.0\.0\.1:876[67]/.test(request.url()))posts.push(request.url());});
  await page.goto("/?model=generic#collection?q=work%20bag");
  await expect(page.getByRole("button",{name:/Switch person/})).toHaveCount(0);
  await expect(page.locator("#shop-search")).toHaveValue("work bag");
  await expect(page.locator("#collection .search-status").first()).toContainText("work bag");
  expect(posts).toEqual([]);
});
