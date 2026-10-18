// Designed presentation treatment, not generated model copy. The product order comes from
// the frozen model run; badges and descriptions use only the request and catalog facts.
// Run with the demo storefront running: node scripts/capture-bob-personalization.mjs [shop URL]
import { chromium } from "@playwright/test";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const root = new URL("../", import.meta.url);
const run = JSON.parse(await readFile(new URL("apps/web/src/fieldwork/model-run.json", root)));
const request = run.requests.find((item) => item.personaId === "dev-sofa");
if (request?.target !== "blanket") throw new Error("Bob's sofa-throw request has changed.");
const output = new URL("apps/web/public/fieldwork/bob/", root);
await mkdir(output, { recursive: true });
const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  await page.goto(process.argv[2] ?? "http://shop.dubhacks.localhost:1356/");
  await page.locator("article.product").first().waitFor();
  await page.evaluate(() => document.fonts.ready);
  await page.addStyleTag({ content: "html { scroll-behavior: auto !important; }" });
  const capture = async (name) => {
    await page.locator(".collection").evaluate((element) => {
      window.scrollTo({
        top: element.getBoundingClientRect().top + window.scrollY - 26,
        behavior: "instant",
      });
    });
    await page.waitForFunction(() =>
      [...document.images].every((image) => {
        const box = image.getBoundingClientRect();
        return (
          box.bottom <= 0 || box.top >= innerHeight || (image.complete && image.naturalWidth > 0)
        );
      }),
    );
    await page.screenshot({
      path: fileURLToPath(new URL(`${name}.jpg`, output)),
      type: "jpeg",
      quality: 90,
    });
  };
  await capture("default");
  await page.evaluate(
    ({ request, products }) => {
      const grid = document.querySelector(".products");
      const cards = new Map(
        [...grid.querySelectorAll("article.product")].map((card) => [
          card.querySelector("h3").textContent,
          card,
        ]),
      );
      for (const { id } of request.top) grid.append(cards.get(products[id].name));
      // Move the remaining catalog behind the measured top six, preserving its existing order.
      const top = new Set(request.top.map(({ id }) => products[id].name));
      for (const [name, card] of cards) if (!top.has(name)) grid.append(card);
      const heading = document.querySelector(".collection .section-heading");
      heading.querySelector("h2").textContent = "A throw for your sofa, Bob.";
      heading.querySelector(".search-status").textContent = `You asked: “${request.request}”`;
      const label = document.createElement("span");
      label.className = "bob-personalized";
      label.textContent = "Personalized for Bob";
      heading.append(label);
      const target = cards.get(products.blanket.name);
      target.classList.add("bob-pick");
      const badge = document.createElement("span");
      badge.className = "bob-pick-badge";
      badge.textContent = "Picked for Bob";
      target.querySelector(".product-art").append(badge);
      const details = document.createElement("div");
      details.className = "bob-details";
      const chips = document.createElement("div");
      chips.className = "bob-chips";
      for (const text of ["Cotton", "Throw blanket", "For your sofa"]) {
        const chip = document.createElement("span");
        chip.textContent = text;
        chips.append(chip);
      }
      const why = document.createElement("p");
      const title = document.createElement("strong");
      title.textContent = "Why this fits";
      why.append(
        title,
        document.createTextNode("A cotton throw blanket for the sofa you had in mind."),
      );
      details.append(chips, why);
      target.querySelector(".product-colors").before(details);
    },
    { request, products: run.products },
  );
  await page.addStyleTag({
    content: `
    .collection .section-heading { padding: 22px 0; min-height: 104px; box-sizing: border-box; }
    .collection .section-heading h2 { font-size: 28px; letter-spacing: -.7px; }
    .collection .section-heading .search-status { font-size: 16px; }
    .product-gallery { display: block; }
    .product-thumbnails { display: none; }
    .bob-personalized { padding: 11px 18px; border: 1px solid var(--border); border-radius: 999px; background: var(--soft-blue); font-size: 15px; font-weight: 650; white-space: nowrap; }
    .bob-pick .product-art { outline: 2px solid var(--blue); outline-offset: -2px; }
    .bob-pick-badge { position: absolute; left: 14px; top: 14px; padding: 10px 16px; border-radius: 999px; color: white; background: var(--blue); font-size: 16px; font-weight: 650; }
    .bob-details { margin: 8px 0 12px; }
    .bob-chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .bob-chips span { border: 1px solid var(--border); border-radius: 999px; padding: 5px 9px; font-size: 13px; background: white; }
    .bob-details p { border-left: 3px solid var(--blue); padding: 3px 0 3px 12px; margin: 13px 0 0; font-size: 14px; line-height: 1.5; max-width: 340px; }
    .bob-details strong { display: block; margin-bottom: 3px; }
    .bob-pick .add-button { background: var(--blue); color: white; border-color: var(--blue); }
  `,
  });
  await capture("personalized");
  await writeFile(
    new URL("capture.json", output),
    `${JSON.stringify(
      {
        personaId: request.personaId,
        request: request.request,
        target: request.target,
        treatment:
          "Designed demo. Ranking from the frozen model run; badges and details are authored from catalog facts.",
      },
      null,
      2,
    )}\n`,
  );
  console.log(`Captured Bob's default and personalized pages in ${fileURLToPath(output)}`);
} finally {
  await browser.close();
}
