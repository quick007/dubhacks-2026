import { test } from "node:test";
import assert from "node:assert/strict";
import manifest from "./responsive-product-sizes.json";
import projection from "./product-image-runtime.json";
import * as current from "./product-responsive-source";
import * as compact from "./product-responsive-runtime";

void test("compact runtime is exactly equivalent for all87 sources, scales and thumbnail states", () => {
  for (const src of Object.keys(manifest.assets)) {
    for (const scale of [undefined, 0.5, 1, 2, 3, 4, 2171 / 724]) {
      for (const thumbnail of [false, true]) {
        assert.deepEqual(compact.responsiveProductImage(src, scale, thumbnail), current.responsiveProductImage(src, scale, thumbnail));
      }
    }
    assert.equal(compact.fullSizeProductImageSource(src), current.fullSizeProductImageSource(src));
  }
});

void test("unknown URL and invalid known-scale behavior matches current resolver", () => {
  for (const src of ["", "/products/new.jpg", "/products/mug-views.png?x=1", "/products//mug-views.png", "/products/../mug-views.png", "/products/%6dug-views.png", "https://example.com/products/mug-views.png", "__proto__", "constructor"]) {
    assert.deepEqual(compact.responsiveProductImage(src), current.responsiveProductImage(src));
    assert.equal(compact.fullSizeProductImageSource(src), current.fullSizeProductImageSource(src));
  }
  for (const scale of [0, -1, Infinity, -Infinity, NaN]) {
    assert.throws(() => compact.responsiveProductImage("/products/mug-views.png", scale), RangeError);
    assert.throws(() => current.responsiveProductImage("/products/mug-views.png", scale), RangeError);
  }
});

void test("runtime projection contains geometry only, without audit hash/bytes/timing payload", () => {
  assert.deepEqual(Object.keys(projection).sort(), ["assets", "v", "widths"]);
  assert.equal(Object.keys(projection.assets).length, 87);
  for (const [basename, dimensions] of Object.entries(projection.assets)) {
    assert.match(basename, /^[a-z0-9][a-z0-9._-]*\.(?:jpg|jpeg|png)$/);
    assert.equal(dimensions.length, 2);
    assert.ok(dimensions.every((value) => Number.isInteger(value) && value > 0));
  }
  assert.doesNotMatch(JSON.stringify(projection), /sha256|encodeMs|sourceBytes|optimizedBytes|iccBytes/i);
});
