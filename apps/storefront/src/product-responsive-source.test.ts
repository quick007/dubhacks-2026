import { test } from "node:test";
import assert from "node:assert/strict";
import manifest from "./responsive-product-sizes.json";
import { responsiveProductImage, fullSizeProductImageSource } from "./product-responsive-source";

const wide = "/products/mug-views.png";

void test("wide strip cover crops default to three sheet-width pixels per card pixel", () => {
  const result = responsiveProductImage(wide);
  assert.match(result.src, /w1152\.webp$/);
  assert.equal(result.sizes, "(max-width: 700px) calc(300vw - 120px), (max-width: 1099px) calc(150vw - 162px), calc(100vw - 136px)");
  assert.match(result.srcSet!, / 192w, /);
  assert.match(result.srcSet!, / 1536w, .* 2172w$/);
  // At1280 DPR1 the whole sheet needs1144px, so1152 is available;
  // atDPR2 the largest2172 detail source remains available.
  const widths = result.srcSet!.split(", ").map((entry) => Number(entry.match(/ (\d+)w$/)![1]));
  assert.equal(widths.find((width) => width >= 1280 - 136), 1152);
  assert.equal(widths.at(-1), 2172);
});

void test("single primary and wide thumbnails choose384/192 defaults with correct sizes", () => {
  const square = Object.entries(manifest.assets).find(([, value]) => value.full.width === value.full.height)![0];
  const primary = responsiveProductImage(square);
  assert.match(primary.src, /w384\.webp$/);
  assert.equal(primary.sizes, "(max-width: 700px) calc(100vw - 40px), (max-width: 1099px) calc(50vw - 54px), calc(33.333333vw - 45.333333px)");
  const thumb = responsiveProductImage(wide, undefined, true);
  assert.equal(thumb.sizes, "150px");
  assert.match(thumb.src, /w192\.webp$/);
});

void test("explicit tile scales and full-size detail source preserve exact asset identity", () => {
  const thumb = responsiveProductImage(wide, 2, true);
  assert.equal(thumb.sizes, "100px");
  assert.equal(fullSizeProductImageSource(wide), manifest.assets[wide].full.src);
  for (const [src, asset] of Object.entries(manifest.assets)) {
    const result = responsiveProductImage(src);
    assert.ok(result.srcSet!.endsWith(`${asset.full.src} ${asset.full.width}w`));
    assert.ok(asset.variants.every((variant) => variant.width < asset.full.width));
    assert.equal(new Set([...asset.variants.map((variant) => variant.width), asset.full.width]).size, asset.variants.length + 1);
  }
});

void test("unknown paths retain original without srcset; invalid known scales fail explicitly", () => {
  for (const src of ["/products/new.jpg", "https://example.com/product.jpg", `${wide}?v=2`, "__proto__"]) {
    assert.deepEqual(responsiveProductImage(src), { src });
    assert.equal(fullSizeProductImageSource(src), src);
  }
  for (const scale of [0, -1, Infinity, NaN]) assert.throws(() => responsiveProductImage(wide, scale), RangeError);
});
