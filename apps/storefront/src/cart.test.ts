import { test } from "node:test";
import assert from "node:assert/strict";
import {
  addCartItem,
  getCartLines,
  removeCartLine,
  removeOneCartItem,
  type CartItem,
} from "./cart";

const mugSage: CartItem = { productId: "mug", variantId: "mug-sage" };
const mugIvory: CartItem = { productId: "mug", variantId: "mug-ivory" };
const packSand: CartItem = { productId: "pack", variantId: "pack-sand" };

void test("adding keeps insertion order without mutating the input", () => {
  const items: CartItem[] = [mugSage];
  const next = addCartItem(items, packSand);
  assert.deepEqual(next, [mugSage, packSand]);
  assert.deepEqual(items, [mugSage]);
});

void test("removing one unit targets the most recent matching line", () => {
  const items = [mugSage, mugIvory, mugSage, packSand];
  const next = removeOneCartItem(items, "mug-sage");
  assert.deepEqual(next, [mugSage, mugIvory, packSand]);
  assert.deepEqual(getCartLines(next).find((line) => line.variant.id === "mug-sage")?.quantity, 1);
  assert.equal(getCartLines(next).length, 3);
});

void test("removing from an unknown variant leaves the cart untouched", () => {
  const items = [mugSage, packSand];
  assert.deepEqual(removeOneCartItem(items, "nope"), items);
});

void test("removing a line drops every unit of that color only", () => {
  const items = [mugSage, mugIvory, mugSage, packSand, mugSage];
  const next = removeCartLine(items, "mug-sage");
  assert.deepEqual(next, [mugIvory, packSand]);
  const lines = getCartLines(next);
  assert.equal(lines.length, 2);
  assert.equal(
    lines.reduce((sum, line) => sum + line.subtotal, 0),
    24 + 68,
  );
});
