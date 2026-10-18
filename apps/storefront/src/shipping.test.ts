import { test } from "node:test";
import assert from "node:assert/strict";
import { FLAT_SHIPPING_COST, orderTotal, shippingQuote } from "./shipping";

void test("hidden shipping keeps delivery terms out of the cart", () => {
  assert.equal(shippingQuote(30, false), null);
  assert.equal(shippingQuote(80, false), null);
});

void test("visible shipping follows the stated free-over-$50 terms", () => {
  assert.deepEqual(shippingQuote(12, true), { label: "$5", amount: 5 });
  assert.deepEqual(shippingQuote(50, true), { label: "$5", amount: 5 });
  assert.deepEqual(shippingQuote(51, true), { label: "Free", amount: 0 });
  assert.deepEqual(shippingQuote(128, true), { label: "Free", amount: 0 });
});

void test("the order total only includes shipping when terms are visible", () => {
  assert.equal(orderTotal(30, null), 30);
  assert.equal(orderTotal(30, shippingQuote(30, true)), 30 + FLAT_SHIPPING_COST);
  assert.equal(orderTotal(80, shippingQuote(80, true)), 80);
});
