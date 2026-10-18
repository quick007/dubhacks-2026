// Captures the real fieldwork storefront for the demo deck.
//
// Every shopper in apps/api/fixtures/catalog.json asks one of the 16 frozen evaluation requests
// from the actual-store model run. Each shopper visits the shop twice: once on the default page
// (the store's own keyword search and product order) and once with the products in the order the
// model produced for that request. The mock shopper searches, looks at the first six products and
// buys the one that fits if it is there. Screenshots, click points and outcomes come from the real
// pages; only the shopper's patience is a rule.
//
// Writes:
//   apps/api/fixtures/scenarios/fieldwork.json   scripted paths for the API's mock runner
//   apps/api/fixtures/frames/fieldwork/*.jpg     one screenshot per distinct page state
//   apps/web/src/fieldwork/model-run.json        measured model numbers for the deck
//   apps/web/src/fieldwork/title-stages.json     title and opening slide pictures, overlay positions
//   apps/web/public/fieldwork/                   title pictures and product photos
//
// Usage: node scripts/capture-fieldwork.mjs [shop URL]. Without a URL it starts the storefront.
import { chromium } from "@playwright/test";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const at = (...parts) => path.join(root, ...parts);
const MODEL_RUN = "experiments/ranker/pretrained/actual_store_function_results/v3";
const LABELS = "experiments/ranker/pretrained/store_function_v3/evaluation-labels.json";
const COPY_AUDIT = "experiments/ranker/pretrained/ACTUAL_STORE_COPY_TRANSFER_V1_AUDIT.json";
const FRAMES = "apps/api/fixtures/frames/fieldwork";
const PUBLIC = "apps/web/public/fieldwork";
const VIEWPORT = { width: 1280, height: 800 };
/** Products a mock shopper looks at, two rows of three, before leaving. */
const PATIENCE = 6;
const PORT = 4398;
/** Opens the deck next to the featured shopper: a different need on the same shop. */
const SECOND_SHOPPER = "ho-lunch";

const readJson = async (file) => JSON.parse(await readFile(at(file), "utf8"));
const fail = (message) => {
  throw new Error(message);
};

// Model evidence, read from the frozen run rather than retyped.
const report = await readJson(`${MODEL_RUN}/report.json`);
const packet = await readJson(`${MODEL_RUN}/packet.json`);
const endpoints = await readJson(`${MODEL_RUN}/FRESH_REQUEST_ENDPOINTS.json`);
const labels = await readJson(LABELS);
const copyAudit = await readJson(COPY_AUDIT);
const catalog = await readJson("apps/api/fixtures/catalog.json");

const recordBody = (record) => JSON.parse(record.text.slice(record.text.indexOf("\n") + 1));
const modelNames = packet.records
  .filter((record) => record.role === "catalog")
  .map((record) => recordBody(record).product.name);
const requests = new Map(
  packet.records
    .filter((record) => record.split === "evaluation")
    .map((record) => [record.caseId, recordBody(record).request]),
);
const targets = new Map(labels.cases.map((item) => [item.caseId, item.targetProductIndex]));

// Store products in the store's default order, from the storefront source the model also read.
const source =
  (await readFile(at("apps/storefront/src/config.ts"), "utf8")) +
  (await readFile(at("apps/storefront/src/catalog-expansion.ts"), "utf8"));
const products = [...source.matchAll(/id: "([a-z]+)",\s*name: "([^"]+)",[^}]*?price: (\d+)/g)].map(
  ([, id, name, price]) => ({ id, name, price: Number(price) }),
);
if (products.map((product) => product.name).join("|") !== modelNames.join("|"))
  fail("The storefront catalog no longer matches the model run's catalog.");
const byName = new Map(products.map((product) => [product.name, product]));

function modelCase(persona) {
  const caseId = [...requests].find(
    ([, request]) =>
      request.preferences.length === 1 && request.preferences[0] === persona.task.query,
  )?.[0];
  if (!caseId) fail(`${persona.id} does not ask a frozen evaluation request.`);
  const target = products[targets.get(caseId)];
  if (persona.task.requirements[0]?.value !== target.id)
    fail(
      `${persona.id} needs ${persona.task.requirements[0]?.value}, the label says ${target.id}.`,
    );
  const learned = report.results.evaluation.learned.cases[caseId];
  const order = learned.scores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score || a.index - b.index);
  const topSix = order.slice(0, 6).map(({ index }) => index);
  if (topSix.join() !== learned.topSix.productIndices.join())
    fail(`${caseId}: score order does not reproduce the recorded top six.`);
  return {
    caseId,
    target,
    order: order.map(({ index, score }) => ({ id: products[index].id, score })),
    ms: endpoints[caseId].freshEncodeToSerializedRankMs,
  };
}

async function startShop() {
  const child = spawn("pnpm", ["exec", "vp", "dev"], {
    cwd: at("apps/storefront"),
    env: { ...process.env, PORT: String(PORT) },
    stdio: "ignore",
    detached: true,
  });
  const url = `http://127.0.0.1:${PORT}`;
  for (let i = 0; i < 60; i++) {
    try {
      if ((await fetch(url)).ok) return { url, stop: () => process.kill(-child.pid, "SIGTERM") };
    } catch {
      /* Not listening yet. */
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  process.kill(-child.pid, "SIGTERM");
  fail("The storefront did not start.");
}

const frameFiles = new Map();
async function snap(page) {
  const image = await page.screenshot({ type: "jpeg", quality: 72 });
  const name = `${createHash("sha256").update(image).digest("hex").slice(0, 16)}.jpg`;
  frameFiles.set(name, image);
  return name;
}

async function pointAt(locator) {
  const box = await locator.boundingBox();
  if (!box) fail("Click target is not visible.");
  const round = (value) => Math.round(value * 1000) / 1000;
  return {
    x: round((box.x + box.width / 2) / VIEWPORT.width),
    y: round((box.y + box.height / 2) / VIEWPORT.height),
  };
}

async function open(context, url) {
  const page = await context.newPage();
  await page.goto(`${url}/`);
  // The shop scrolls smoothly; screenshots need the page where it was sent.
  await page.addStyleTag({ content: "html { scroll-behavior: auto !important; }" });
  await settle(page);
  return page;
}

async function settle(page) {
  // Off-screen photos load lazily, so wait only for the ones in view.
  await page.waitForFunction(() =>
    [...document.images].every((image) => {
      const box = image.getBoundingClientRect();
      return image.complete || box.bottom < 0 || box.top > window.innerHeight;
    }),
  );
  await page.waitForTimeout(150);
}

async function scrollTo(page, y) {
  await page.evaluate((top) => window.scrollTo({ top, behavior: "instant" }), y);
  await settle(page);
}

/** Card rectangles as viewport fractions, for overlays drawn on a screenshot. */
async function cardRects(page) {
  return page.$$eval(
    "article.product",
    (cards, viewport) =>
      cards.flatMap((card) => {
        const box = card.getBoundingClientRect();
        if (box.bottom <= 0 || box.top >= viewport.height) return [];
        const round = (value) => Math.round(value * 1000) / 1000;
        return [
          {
            name: card.querySelector("h3").textContent,
            x: round(box.left / viewport.width),
            y: round(box.top / viewport.height),
            width: round(box.width / viewport.width),
            height: round(box.height / viewport.height),
          },
        ];
      }),
    VIEWPORT,
  );
}

/** One shopper on one version of the shop. */
async function shop(browser, url, persona, arm, model) {
  const context = await browser.newContext({ viewport: VIEWPORT });
  const page = await open(context, url);
  const query = persona.task.query;
  const steps = [];
  const record = async (pageInfo, message, extra = {}) => {
    const step = { page: pageInfo, message, ...extra, screenshot: await snap(page) };
    steps.push(step);
    return step;
  };

  if (arm === "ranked") {
    const names = model.order.map(({ id }) => products.find((product) => product.id === id).name);
    await page.evaluate(
      ({ names, query }) => {
        const grid = document.querySelector(".products");
        const cards = new Map(
          [...grid.querySelectorAll("article.product")].map((card) => [
            card.querySelector("h3").textContent,
            card,
          ]),
        );
        for (const name of names) grid.append(cards.get(name));
        document.querySelector(".search-status").textContent = `Personalized for "${query}"`;
      },
      { names, query },
    );
    await settle(page);
  }
  await record({ kind: "home", label: "Home" }, "Opens the shop");

  const shown = (
    await page.$$eval("article.product h3", (headings) =>
      headings.map((heading) => heading.textContent),
    )
  ).map((name) => byName.get(name).id);
  const position = shown.indexOf(model.target.id);
  const rows = PATIENCE / 3;
  let bought = false;
  // Card positions on the last results frame, for overlays on the title slide.
  let rects;
  for (let row = 0; row < rows && !bought; row++) {
    const firstCard = page.locator("article.product").nth(row * 3);
    const top = await firstCard.evaluate(
      (card) => card.getBoundingClientRect().top + window.scrollY,
    );
    // The first row keeps the collection heading and the model's note in view.
    await scrollTo(page, row === 0 ? top - 150 : top - 60);
    const inRow = Math.floor(position / 3) === row;
    const visible = shown.slice(row * 3, row * 3 + 3);
    const opener =
      row > 0 ? "Next 3 products" : arm === "ranked" ? "The personalized page" : "The default page";
    const pageInfo = { kind: "category", label: row === 0 ? "Products" : "Products, further down" };
    rects = await cardRects(page);
    if (!inRow) {
      await record(pageInfo, opener, { products: visible });
      continue;
    }
    const add = page.locator("article.product").nth(position).locator(".add-button");
    await record(pageInfo, `${opener}. Adds the ${model.target.name.toLowerCase()} to the cart`, {
      products: visible,
      click: await pointAt(add),
    });
    await add.click();
    bought = true;
  }

  let reason;
  if (bought) {
    await page.locator(".cart-button").click();
    await scrollTo(page, 0);
    const order = page.getByRole("button", { name: "Place demo order" });
    await record({ kind: "cart", label: "Cart" }, "Opens the cart", {
      products: [model.target.id],
      click: await pointAt(order),
    });
    await order.click();
    await scrollTo(page, 0);
    await record({ kind: "confirmation", label: "Confirmation" }, "Checks out", {
      products: [model.target.id],
    });
  } else {
    reason = `Sees no fit in the first ${PATIENCE} products and leaves`;
  }

  await context.close();
  return {
    path: { steps, ...(bought ? { purchase: model.target.id } : { reason }) },
    position: position + 1,
    rects,
  };
}

/** Where the store's own keyword search puts the product when the shopper types the request. */
async function searchRank(browser, url, persona, target) {
  const context = await browser.newContext({ viewport: VIEWPORT });
  const page = await open(context, url);
  await page.locator("#shop-search").fill(persona.task.query);
  await page.locator("form[role=search] button[type=submit]").click();
  await page.waitForTimeout(300);
  const shown = await page.$$eval("article.product h3", (headings) =>
    headings.map((heading) => heading.textContent),
  );
  await context.close();
  return shown.map((name) => byName.get(name).id).indexOf(target) + 1;
}

async function titleStages(browser, url, featured, cases) {
  const context = await browser.newContext({ viewport: VIEWPORT });
  const page = await open(context, url);
  const top = await page
    .locator("article.product")
    .first()
    .evaluate((card) => card.getBoundingClientRect().top + window.scrollY);
  await scrollTo(page, top - 150);
  const browse = {
    image: await page.screenshot({ type: "jpeg", quality: 72 }),
    cards: await cardRects(page),
  };

  // Each product photo, from the store's own default card.
  const photos = {};
  for (const product of products) {
    const card = page.locator("article.product", { hasText: product.name });
    await card.evaluate((element) => element.scrollIntoView({ block: "center" }));
    await settle(page);
    photos[product.id] = await card
      .locator(".product-art")
      .first()
      .screenshot({ type: "jpeg", quality: 70 });
  }
  await context.close();

  const stuck = cases.get(featured.id).default;
  const fit = cases.get(featured.id).ranked;
  const lastResults = (visit) => visit.path.steps.findLast((step) => step.page.kind === "category");
  const second = cases.get(SECOND_SHOPPER)?.ranked ?? fail(`${SECOND_SHOPPER} is not a shopper.`);
  return {
    photos,
    stages: {
      browse,
      stuck: { image: frameFiles.get(lastResults(stuck).screenshot), cards: stuck.rects },
      fit: { image: frameFiles.get(lastResults(fit).screenshot), cards: fit.rects },
      second: { image: frameFiles.get(lastResults(second).screenshot), cards: second.rects },
    },
  };
}

const given = process.argv[2];
const shopServer = given ? { url: given.replace(/\/$/, ""), stop() {} } : await startShop();
const browser = await chromium.launch();
try {
  const shoppers = catalog.personas.filter((persona) => persona.task);
  const cases = new Map();
  for (const persona of shoppers) {
    const model = modelCase(persona);
    const visits = {};
    for (const arm of ["default", "ranked"])
      visits[arm] = await shop(browser, shopServer.url, persona, arm, model);
    const search = await searchRank(browser, shopServer.url, persona, model.target.id);
    cases.set(persona.id, { model, search, ...visits });
    console.log(
      `${persona.name.padEnd(8)} ${model.target.name.padEnd(30)} default #${String(visits.default.position).padEnd(3)} ${visits.default.path.purchase ? "bought" : "left  "}  model #${visits.ranked.position} ${visits.ranked.path.purchase ? "bought" : "left"}`,
    );
  }

  // Same rule as the deck's featured shopper: first development shopper who left the default page
  // and bought from the ranked one.
  const featured =
    shoppers.find(
      (persona) =>
        persona.split === "development" &&
        !cases.get(persona.id).default.path.purchase &&
        cases.get(persona.id).ranked.path.purchase,
    ) ?? fail("No development shopper left the default page and bought from the ranked one.");
  const title = await titleStages(browser, shopServer.url, featured, cases);

  const scenario = {
    id: "fieldwork",
    products: products.map((product) => ({
      ...product,
      attributes: { product: [product.id] },
    })),
    optimalSteps: Object.fromEntries(shoppers.map((persona) => [persona.id, 5])),
    paths: Object.fromEntries(
      shoppers.map((persona) => [
        persona.id,
        { default: cases.get(persona.id).default.path, ranked: cases.get(persona.id).ranked.path },
      ]),
    ),
  };

  const evaluation = report.results.evaluation;
  const arm = (name) => ({
    topOne: evaluation[name].topOneCorrect,
    topSix: evaluation[name].targetInTopSix,
  });
  const times = Object.values(endpoints).map((item) => item.freshEncodeToSerializedRankMs);
  const modelRun = {
    source: {
      run: MODEL_RUN,
      commit: report.executionCommit,
      checkpointSha256: report.heads.learned.sha256,
      backbone: "Qwen3.5-9B, 4-bit MLX, frozen",
      trainedParameters: packet.config.parameters,
      trainingRequests: packet.records.filter((r) => r.role === "request" && r.split === "train")
        .length,
    },
    cases: Object.keys(evaluation.learned.cases).length,
    arms: {
      learned: arm("learned"),
      prior: arm("prior"),
      random: arm("random"),
      permutedQuery: arm("permuted-query"),
    },
    pairs: {
      bothCorrect: evaluation.learned.pairsBothCorrect,
      total: evaluation.learned.fixedPairs,
    },
    latencyMs: {
      median: report.freshRequestMedianMs,
      min: Math.min(...times),
      max: Math.max(...times),
    },
    modelLoadSeconds: report.modelLoadSeconds,
    peakMemoryBytes: report.memory.mlxPeakBytes,
    copy: { valid: copyAudit.fullSixValid, total: copyAudit.fixedCases },
    // Not a model arm: the store's keyword search, measured here on the same requests.
    keywordSearch: {
      topOne: shoppers.filter((persona) => cases.get(persona.id).search === 1).length,
      topSix: shoppers.filter((persona) => cases.get(persona.id).search <= PATIENCE).length,
    },
    patience: PATIENCE,
    products: Object.fromEntries(
      products.map((product, index) => [
        product.id,
        { name: product.name, price: product.price, defaultRank: index + 1 },
      ]),
    ),
    requests: shoppers.map((persona) => {
      const { model } = cases.get(persona.id);
      return {
        caseId: model.caseId,
        personaId: persona.id,
        request: persona.task.query,
        target: model.target.id,
        modelRank: model.order.findIndex(({ id }) => id === model.target.id) + 1,
        searchRank: cases.get(persona.id).search,
        latencyMs: model.ms,
        top: model.order
          .slice(0, 6)
          .map(({ id, score }) => ({ id, score: Math.round(score * 1000) / 1000 })),
      };
    }),
  };

  await rm(at(FRAMES), { recursive: true, force: true });
  await rm(at(PUBLIC), { recursive: true, force: true });
  await mkdir(at(FRAMES), { recursive: true });
  await mkdir(at(PUBLIC, "products"), { recursive: true });
  await mkdir(at("apps/web/src/fieldwork"), { recursive: true });
  for (const [name, image] of frameFiles) await writeFile(at(FRAMES, name), image);
  for (const [id, image] of Object.entries(title.photos))
    await writeFile(at(PUBLIC, "products", `${id}.jpg`), image);
  const stages = {};
  for (const [stage, { image, cards }] of Object.entries(title.stages)) {
    await writeFile(at(PUBLIC, `${stage}.jpg`), image);
    stages[stage] = {
      src: `/fieldwork/${stage}.jpg`,
      cards: cards.map(({ name, ...box }) => ({ id: byName.get(name).id, ...box })),
    };
  }
  const json = (value) => `${JSON.stringify(value, null, 2)}\n`;
  await writeFile(at("apps/api/fixtures/scenarios/fieldwork.json"), json(scenario));
  await writeFile(at("apps/web/src/fieldwork/model-run.json"), json(modelRun));
  await writeFile(
    at("apps/web/src/fieldwork/title-stages.json"),
    json({
      featured: featured.id,
      target: cases.get(featured.id).model.target.id,
      secondShopper: SECOND_SHOPPER,
      ...stages,
    }),
  );
  console.log(`${frameFiles.size} frames; featured shopper ${featured.name}.`);
} finally {
  await browser.close();
  shopServer.stop();
}
