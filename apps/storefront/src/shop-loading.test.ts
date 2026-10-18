import { test } from "node:test";
import assert from "node:assert/strict";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ShopView } from "./ShopView";
import { DEFAULT_CONFIG, PRODUCTS } from "./config";
import { personalizeStorefront } from "./personalization";
import { PRODUCT_VARIANTS } from "./variants";
import { INITIAL_LOCAL_MODEL_STATE, beginLocalModelRequest, applyLocalModelEvent, type LocalCopyEvent, type LocalRankEvent, type LocalModelState } from "./local-model";

const six = PRODUCTS.slice(0, 6);
const rank: LocalRankEvent = { type: "rank", requestId: "current", productIds: six.map(p => p.id),
  selectedProductIds: six.map(p => p.id), variants: Object.fromEntries(six.map(p => [p.id, PRODUCT_VARIANTS[p.id][0].id])),
  shortfall: 0, timing: { rankMs: 20 }, provenance: { ranking: "learned", images: "catalog" } };
const copy = (type: "firstThree" | "complete"): LocalCopyEvent => {
  const products = type === "firstThree" ? six.slice(0, 3) : six;
  return { type, requestId: "current", sections: Array.from({ length: products.length / 3 }, (_, i) => ({ heading: "Chosen for this request",
    cards: products.slice(i * 3, i * 3 + 3).map(p => ({ productId: p.id, variantId: PRODUCT_VARIANTS[p.id][0].id,
      originalName: p.name, itemPrice: p.price, selectedColor: PRODUCT_VARIANTS[p.id][0].colorName,
      title: `Fresh ${p.name}`, badge: null, bodyCopy: null })) })), acceptedFields: products.length + products.length / 3,
    withheldFields: products.length * 2, requiredFields: products.length * 3 + products.length / 3,
    timing: { readyMs: 40, copyMs: 20 } };
};
function render(localModel?: LocalModelState) {
  return renderToStaticMarkup(React.createElement(ShopView, {
    config: DEFAULT_CONFIG, experience: personalizeStorefront("", DEFAULT_CONFIG), collectionId: "all", query: "", searchInput: "",
    visibleProducts: six, selectedVariants: {}, added: "", localModel, personName: "Maya", onSearchInput() {}, onSearchSubmit() {},
    onExampleSearch() {}, onClearSearch() {}, onCollectionChange() {}, onVariantChange() {}, onAdd() {},
  }));
}
const count = (html: string, fragment: string) => html.split(fragment).length - 1;

test("new persona/query requests show six placeholders, not supplied fallback products", () => {
  const html = render(beginLocalModelRequest(INITIAL_LOCAL_MODEL_STATE, "new", "person:new", true));
  assert.equal(count(html, 'class="product product-skeleton"'), 6);
  assert.equal(count(html, '<article class="product"'), 0);
  assert.doesNotMatch(html, /Add to cart|Trail mug/);
  assert.match(html, /aria-busy="true"/);
  assert.doesNotMatch(html, /disabled=/);
});
test("rank-only updates keep the skeleton until fresh copy arrives", () => {
  const state = applyLocalModelEvent(beginLocalModelRequest(INITIAL_LOCAL_MODEL_STATE, "new"), rank, "new");
  const html = render(state);
  assert.equal(count(html, 'class="product product-skeleton"'), 6);
  assert.equal(count(html, '<article class="product"'), 0);
});
test("firstThree reveals three current cards and keeps three pending cards empty", () => {
  let state = applyLocalModelEvent(beginLocalModelRequest(INITIAL_LOCAL_MODEL_STATE, "new"), rank, "new");
  state = applyLocalModelEvent(state, copy("firstThree"), "new");
  const html = render(state);
  assert.equal(count(html, '<article class="product"'), 3);
  assert.equal(count(html, 'class="product product-skeleton"'), 3);
  assert.match(html, /Fresh Trail mug/);
});
test("completion reveals all six and errors hide every old or fallback card", () => {
  let state = applyLocalModelEvent(beginLocalModelRequest(INITIAL_LOCAL_MODEL_STATE, "new"), rank, "new");
  state = applyLocalModelEvent(state, copy("complete"), "new");
  assert.equal(count(render(state), '<article class="product"'), 6);
  const html = render({ ...state, phase: "error", error: "offline" });
  assert.equal(count(html, '<article class="product"'), 0);
  assert.equal(count(html, 'class="product product-skeleton"'), 0);
  assert.match(html, /couldn’t refresh/);
});
test("baseline storefront remains immediately browsable", () => {
  const html = render();
  assert.equal(count(html, '<article class="product"'), 6);
  assert.equal(count(html, 'class="product product-skeleton"'), 0);
});
