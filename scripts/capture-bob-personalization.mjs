// Incremental, authored demo screenshots. Ranking uses the frozen model run.
// The sofa purchase and material preference are illustrative, not measured model inputs.
// Run: node scripts/capture-bob-personalization.mjs [shop URL]
import { chromium } from "@playwright/test";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const root = new URL("../", import.meta.url);
const run = JSON.parse(await readFile(new URL("apps/web/src/fieldwork/model-run.json", root)));
const request = run.requests.find((item) => item.personaId === "dev-sofa");
if (request?.target !== "blanket") throw new Error("Bob's sofa-throw request has changed.");
const output = new URL("apps/web/public/fieldwork/bob/", root);
await mkdir(output, { recursive: true });
const sofa = await readFile(new URL("sofa.png", output));
const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  await page.goto(process.argv[2] ?? "http://shop.dubhacks.localhost:1356/");
  await page.locator("article.product").first().waitFor();
  await page.evaluate(() => document.fonts.ready);
  // Reserve the same heading space across every capture so each change stays in place.
  await page.addStyleTag({
    content: `
    html { scroll-behavior: auto !important; }
    .collection .section-heading { height: 94px; box-sizing: border-box; }
    .collection .section-heading h2 { font-size: 22px; letter-spacing: -.3px; }
    .collection .section-heading .search-status { font-size: 14px; }
    .bob-details { height: 66px; box-sizing: border-box; padding-top: 12px; font-size: 15px; line-height: 1.5; }
    .bob-details strong { display: block; font-weight: 600; }
    .bob-details p { margin: 3px 0 0; color: var(--muted-blue); }
    .bob-match { margin: 7px 0 4px; font-size: 15px; line-height: 1.5; color: var(--muted-blue); }
    .bob-match strong { color: var(--ink); font-weight: 600; }
  `,
  });
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
  await capture("original");

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
      const top = new Set(request.top.map(({ id }) => products[id].name));
      for (const [name, card] of cards) if (!top.has(name)) grid.append(card);
      const heading = document.querySelector(".collection .section-heading");
      heading.querySelector("h2").textContent = "For your sofa";
      heading.querySelector(".search-status").textContent = request.request;
      cards.get(products.blanket.name).dataset.bobThrow = "";
    },
    { request, products: run.products },
  );
  await capture("request");

  await page.evaluate(() => {
    const target = document.querySelector("[data-bob-throw]");
    const details = document.createElement("div");
    details.className = "bob-details";
    const material = document.createElement("strong");
    material.textContent = "Woven cotton";
    const warmth = document.createElement("p");
    warmth.textContent = "For cold evenings by the window.";
    details.append(material, warmth);
    target.querySelector(".product-thumbnails").replaceWith(details);
  });
  await capture("details");

  await page.evaluate(
    (image) => {
      const photo = document.querySelector("[data-bob-throw] .product-art .product-photo");
      const replacement = document.createElement("img");
      replacement.src = image;
      replacement.alt = "Rust cotton throw on a sand sofa beside a window.";
      photo.replaceChildren(replacement);
    },
    `data:image/png;base64,${sofa.toString("base64")}`,
  );
  await capture("room");

  await page.evaluate(() => {
    const note = document.createElement("p");
    note.className = "bob-match";
    const reason = document.createElement("strong");
    reason.textContent = "Pairs with your sand sofa.";
    note.append(reason);
    document.querySelector("[data-bob-throw] .product-heading").after(note);
  });
  await capture("match");

  await writeFile(
    new URL("capture.json", output),
    `${JSON.stringify(
      {
        personaId: request.personaId,
        request: request.request,
        target: request.target,
        stages: ["original", "request", "details", "room", "match"],
        treatment:
          "Authored demo. Ranking from the frozen model run. Material preference, sofa purchase, and room styling are illustrative.",
        image: "Sofa scene generated from the existing blanket product photo with imagegen.",
      },
      null,
      2,
    )}\n`,
  );
  console.log(`Captured five incremental stages in ${fileURLToPath(output)}`);
} finally {
  await browser.close();
}
