import { test } from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_CONFIG, orderedProducts, parseVariantConfig } from "./config";

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
void test("budget ordering is stable, low to high, and does not mutate the catalog", () => {
  assert.deepEqual(
    orderedProducts("budget").map((product) => product.price),
    [12, 24, 68],
  );
  assert.deepEqual(
    orderedProducts("popular").map((product) => product.price),
    [24, 68, 12],
  );
});
