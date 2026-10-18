import { test } from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_ROUTE, parseRoute, routeHash } from "./routes";

void test("deep links parse into pages, searches and collections", () => {
  assert.deepEqual(parseRoute(""), DEFAULT_ROUTE);
  assert.deepEqual(parseRoute("#collection"), DEFAULT_ROUTE);
  assert.deepEqual(parseRoute("#collection?q=rain+jacket+for+campus&c=clothing"), {
    page: "shop",
    query: "rain jacket for campus",
    collectionId: "clothing",
  });
  assert.deepEqual(parseRoute("#cart?q=work+bag"), {
    page: "cart",
    query: "work bag",
    collectionId: "all",
  });
  assert.deepEqual(parseRoute("#confirmed"), {
    page: "confirmed",
    query: "",
    collectionId: "all",
  });
});

void test("unknown or oversized hashes fall back to the shop", () => {
  assert.deepEqual(parseRoute("#product/3"), DEFAULT_ROUTE);
  assert.equal(parseRoute(`#collection?q=${"a".repeat(500)}`).query.length, 200);
  assert.deepEqual(parseRoute("#collection?q=+++"), DEFAULT_ROUTE);
});

void test("routes round trip through the hash", () => {
  for (const route of [
    DEFAULT_ROUTE,
    { page: "shop" as const, query: "rain jacket for campus", collectionId: "clothing" },
    { page: "shop" as const, query: "coffee & tea", collectionId: "all" },
    { page: "cart" as const, query: "", collectionId: "all" },
    { page: "confirmed" as const, query: "hiking gear", collectionId: "bags" },
  ]) {
    assert.deepEqual(parseRoute(routeHash(route)), route);
  }
  assert.equal(routeHash(DEFAULT_ROUTE), "#collection");
  assert.equal(routeHash({ page: "cart", query: "", collectionId: "all" }), "#cart");
  assert.equal(
    routeHash({ page: "confirmed", query: "a & b", collectionId: "home" }),
    "#confirmed?q=a+%26+b&c=home",
  );
});
