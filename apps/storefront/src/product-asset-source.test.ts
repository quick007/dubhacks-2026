import { test } from "node:test";
import assert from "node:assert/strict";
import manifest from "./responsive-product-assets.json";
import { optimizedProductAsset, resolveProductImageSource } from "./product-asset-source";

void test("same-dimension alternatives cover all87 originals and reduce every payload", () => {
  assert.equal(manifest.summary.sourceCount, 87);
  assert.equal(Object.keys(manifest.assets).length, 87);
  assert.equal(manifest.summary.nonSmallerDerivativeCount, 0);
  for (const [src, entry] of Object.entries(manifest.assets)) {
    assert.equal(resolveProductImageSource(src), entry.src);
    assert.deepEqual(optimizedProductAsset(src), entry);
    assert.ok(entry.src.startsWith("/products/optimized/"));
    assert.ok(entry.src.endsWith(".webp"));
    assert.ok(entry.width > 0 && entry.height > 0);
    assert.ok(entry.optimizedBytes < entry.sourceBytes);
  }
});

void test("sprite sheets retain exact source aspect and odd dimensions", () => {
  const all = Object.values(manifest.assets);
  assert.equal(all.filter((asset) => asset.width === 2171 && asset.height === 724).length, 1);
  assert.equal(all.filter((asset) => asset.width === 2172 && asset.height === 724).length, 44);
  assert.equal(all.filter((asset) => asset.width === 1774 && asset.height === 887).length, 16);
  assert.equal(all.filter((asset) => asset.width === 1254 && asset.height === 1254).length, 26);
});

void test("unknown, remote, modified and prototype names retain original URL", () => {
  for (const src of ["/products/new.jpg", "https://example.com/products/mug-views.png", "/products/mug-views.png?v=2", "__proto__", "constructor", ""]) {
    assert.equal(resolveProductImageSource(src), src);
    assert.equal(optimizedProductAsset(src), null);
  }
});
