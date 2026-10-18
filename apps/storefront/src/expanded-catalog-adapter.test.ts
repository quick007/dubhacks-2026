import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { PRODUCTS, ORIGINAL_PRODUCTS } from "./config";
import { EXPANDED_PUBLIC_CATALOG, parseExpandedCatalog } from "./expanded-catalog-adapter";
import { PRODUCT_VARIANTS, getVariantImages } from "./variants";
import { validExposure } from "./shopping-events";
import { canonicalReplayAsset } from "./session-trace";

test("35-product UI resolves exact authoritative facts, 105 variants and all explicit galleries", () => {
  assert.equal(PRODUCTS.length, 35);
  assert.equal(ORIGINAL_PRODUCTS.length, 16);
  assert.equal(Object.values(PRODUCT_VARIANTS).flat().length, 105);
  for (const product of PRODUCTS) {
    const publicProduct = EXPANDED_PUBLIC_CATALOG.products.find(item => item.productId === product.id)!;
    assert.ok(publicProduct);
    assert.equal(product.name, publicProduct.name); assert.equal(product.price, publicProduct.itemPrice); assert.equal(product.category, publicProduct.category);
    assert.deepEqual(PRODUCT_VARIANTS[product.id].map(variant => variant.id), publicProduct.variants.map(variant => variant.variantId));
    for (const variant of PRODUCT_VARIANTS[product.id]) {
      const publicVariant = publicProduct.variants.find(item => item.variantId === variant.id)!;
      assert.equal(variant.colorName, publicVariant.colorName);
      assert.deepEqual(getVariantImages(variant), publicVariant.images);
      for (const image of getVariantImages(variant)) assert.ok(existsSync(new URL(`../public${image.src}`, import.meta.url)), image.src);
    }
  }
});
test("expanded telemetry uses exact catalog membership and bounded positions; replay permits only known public assets", () => {
  assert.equal(validExposure({productId:"work-briefcase",variantId:PRODUCT_VARIANTS["work-briefcase"][0].id,position:34}),true);
  assert.equal(validExposure({productId:"work-briefcase",variantId:PRODUCT_VARIANTS["work-briefcase"][0].id,position:35}),false);
  assert.equal(validExposure({productId:"unlisted",variantId:"unlisted-red",position:1}),false);
  assert.equal(canonicalReplayAsset("/products/rain-anorak-color-gallery.png"), "/products/rain-anorak-color-gallery.png");
  assert.equal(canonicalReplayAsset("/products/hero-mug-wide.jpg"), "/products/hero-mug-wide.jpg");
  assert.equal(canonicalReplayAsset("/products/secret-unlisted.png"), "");
  assert.equal(canonicalReplayAsset("/products/rain-anorak-color-gallery.png?private=yes"), "");
});
test("public catalog rejects invalid product identity, external images and invalid tile coordinates", () => {
  for (const mutation of [
    (catalog: any) => { catalog.products[1].productId = catalog.products[0].productId; },
    (catalog: any) => { catalog.products[0].variants[0].images[0].src = "https://example.com/image.png"; },
    (catalog: any) => { catalog.products.at(-1).variants[0].images[0].tile.index = 99; },
  ]) {
    const copy = structuredClone(EXPANDED_PUBLIC_CATALOG); mutation(copy); assert.throws(() => parseExpandedCatalog(copy));
  }
});
