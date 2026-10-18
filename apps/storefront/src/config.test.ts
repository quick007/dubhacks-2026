import { test } from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_CONFIG,
  PRODUCTS,
  orderedProducts,
  parseVariantConfig,
  serializeVariantConfig,
} from "./config";
import { personalizeStorefront } from "./personalization";
import { PRODUCT_IMAGES, selectProductImage } from "./images";
import { existsSync, readFileSync } from "node:fs";
import { PRODUCT_VARIANTS, getVariantImages } from "./variants";
import { getCartLines } from "./cart";

void test("search intent selects relevant products and environments", () => {
  for (const [query, first, context] of [
    ["HIKING gear", "pack", "outdoors"],
    ["work bag", "pack", "city"],
    ["journaling", "notebook", "studio"],
    ["camping coffee", "flask", "outdoors"],
    ["notebook for work", "notebook", "city"],
    ["rain jacket for campus", "jacket", "campus"],
    ["family park", "pack", "family"],
    ["waterproof hiking jacket", "jacket", "outdoors"],
    ["headphones", "headphones", "studio"],
    ["beach towel", "towel", "studio"],
    ["canvas tote", "tote", "studio"],
  ] as const) {
    const result = personalizeStorefront(query, DEFAULT_CONFIG);
    assert.equal(result.products[0].id, first, query);
    assert.equal(result.context, context, query);
    assert.equal(result.matched, true, query);
    assert.equal(result.products.length, PRODUCTS.length);
  }
});
void test("empty and unknown searches preserve variant defaults without false substring matches", () => {
  const config = {
    ...DEFAULT_CONFIG,
    headline: "Custom headline",
    recommendationOrder: "budget" as const,
  };
  for (const query of ["", "  ", "television", "cupboard", "homework"]) {
    const result = personalizeStorefront(query, config);
    assert.equal(result.matched, false, query);
    assert.equal(result.context, "studio", query);
    assert.equal(result.headline, config.headline);
    assert.deepEqual(result.products, orderedProducts("budget"));
  }
});

void test("accepts the shared structured variant settings", () => {
  assert.deepEqual(
    parseVariantConfig("?headline=Hello+shopper&showShipping=true&recommendationOrder=budget"),
    { headline: "Hello shopper", showShipping: true, recommendationOrder: "budget" },
  );
});
void test("invalid settings fall back to a safe complete default", () => {
  assert.deepEqual(
    parseVariantConfig("?headline=&showShipping=maybe&recommendationOrder=code"),
    DEFAULT_CONFIG,
  );
  assert.deepEqual(parseVariantConfig("?script=alert(1)&url=https://example.com"), DEFAULT_CONFIG);
});
void test("demo controls and the URL describe the same settings", () => {
  for (const config of [
    DEFAULT_CONFIG,
    {
      headline: "Everyday goods, clear delivery",
      showShipping: true,
      recommendationOrder: "budget" as const,
    },
    {
      headline: "Ready for the next chapter.",
      showShipping: true,
      recommendationOrder: "history" as const,
    },
  ]) {
    assert.deepEqual(parseVariantConfig(`?${serializeVariantConfig(config)}`), config);
  }
  assert.equal(serializeVariantConfig(DEFAULT_CONFIG), "");
});
void test("budget ordering is stable, low to high, and does not mutate the catalog", () => {
  const originalIds = PRODUCTS.map((product) => product.id);
  const prices = orderedProducts("budget").map((product) => product.price);
  assert.ok(prices.every((price, index) => index === 0 || prices[index - 1] <= price));
  assert.equal(prices[0], 12);
  assert.equal(prices.at(-1), 128);
  assert.deepEqual(
    orderedProducts("popular").map((product) => product.id),
    originalIds,
  );
});

void test("audience selection uses compatible images and keeps every view accessible", () => {
  for (const id of ["pack", "jacket"] as const) {
    for (const context of ["outdoors", "city", "campus", "family"] as const) {
      const image = selectProductImage(id, context);
      assert.equal(image.context, context);
      assert.ok(PRODUCT_IMAGES[id].includes(image));
    }
  }
  assert.equal(selectProductImage("mug", "campus").context, "everyday");
  assert.equal(selectProductImage("notebook", "family").context, "outdoors");
  for (const product of PRODUCTS) {
    const images = PRODUCT_IMAGES[product.id];
    assert.equal(new Set(images.map((image) => image.id)).size, images.length);
    for (const image of images) {
      assert.ok(existsSync(new URL(`../public${image.src}`, import.meta.url)), image.src);
      assert.ok(image.alt.length > 10);
      assert.ok(image.tags.length > 0);
    }
  }
});

void test("visual adjectives do not prescribe which presentation a shopper prefers", () => {
  assert.deepEqual(
    personalizeStorefront("dark notebook", DEFAULT_CONFIG),
    personalizeStorefront("notebook", DEFAULT_CONFIG),
  );
  assert.deepEqual(
    personalizeStorefront("plain rain jacket for campus", DEFAULT_CONFIG),
    personalizeStorefront("rain jacket for campus", DEFAULT_CONFIG),
  );
});

void test("persona briefs reference real products and compatible image candidates", () => {
  const fixture = JSON.parse(
    readFileSync(new URL("../fixtures/shopper-personas.json", import.meta.url), "utf8"),
  ) as {
    personas: {
      id: string;
      goal: string;
      candidateProductIds: string[];
      candidateImageIds: string[];
    }[];
  };
  const catalogIds = new Set<string>(PRODUCTS.map((product) => product.id));
  assert.equal(catalogIds.size, PRODUCTS.length);
  const imageOwners = new Map(
    Object.entries(PRODUCT_IMAGES).flatMap(([productId, images]) =>
      images.map((image) => [image.id, productId] as const),
    ),
  );
  assert.equal(
    new Set(fixture.personas.map((persona) => persona.id)).size,
    fixture.personas.length,
  );
  for (const persona of fixture.personas) {
    assert.ok(persona.goal.length > 0);
    for (const productId of persona.candidateProductIds)
      assert.ok(catalogIds.has(productId), productId);
    for (const imageId of persona.candidateImageIds) {
      const productId = imageOwners.get(imageId);
      assert.ok(productId, imageId);
      assert.ok(persona.candidateProductIds.includes(productId), `${persona.id}: ${imageId}`);
    }
  }
});

void test("every product color has compatible, existing photography", () => {
  const ids = new Set<string>();
  for (const product of PRODUCTS) {
    const variants = PRODUCT_VARIANTS[product.id];
    assert.ok(variants.length >= 2, product.id);
    for (const variant of variants) {
      assert.equal(variant.productId, product.id);
      assert.ok(!ids.has(variant.id), variant.id);
      ids.add(variant.id);
      const images = getVariantImages(variant);
      assert.ok(images.length > 0);
      for (const image of images) {
        assert.ok(existsSync(new URL(`../public${image.src}`, import.meta.url)), image.src);
      }
      if (variant.image) {
        const originals = PRODUCT_IMAGES[product.id];
        assert.equal(images.length, originals.length);
        for (const [index, image] of images.entries()) {
          assert.equal(image.label, originals[index].label);
          assert.equal(image.context, originals[index].context);
          assert.deepEqual(image.presentation, originals[index].presentation);
          assert.ok(image.alt.includes(variant.colorName.toLowerCase()));
          assert.ok(
            image.tile &&
              image.tile.index >= 0 &&
              image.tile.index < image.tile.columns * image.tile.rows,
          );
          assert.ok(
            image.tile && Number.isFinite(image.tile.sheetAspect) && image.tile.sheetAspect > 0,
          );
        }
      }
    }
  }
});

void test("cart groups the same color and keeps different colors separate", () => {
  const lines = getCartLines([
    { productId: "mug", variantId: "mug-ivory" },
    { productId: "mug", variantId: "mug-cobalt" },
    { productId: "mug", variantId: "mug-ivory" },
    { productId: "pack", variantId: "pack-forest" },
  ]);
  assert.equal(lines.length, 3);
  assert.equal(lines.find((line) => line.variant.id === "mug-ivory")?.quantity, 2);
  assert.equal(lines.find((line) => line.variant.id === "mug-cobalt")?.quantity, 1);
  assert.equal(
    lines.find((line) => line.variant.id === "pack-forest")?.variant.colorName,
    "Forest",
  );
  assert.equal(
    lines.reduce((sum, line) => sum + line.subtotal, 0),
    140,
  );
});
