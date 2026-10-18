import { test } from "node:test";
import assert from "node:assert/strict";
import { PRODUCTS } from "./config";
import { PRODUCT_VARIANTS } from "./variants";
import { ProductCard } from "./ProductCard";
import { applyLocalModelEvent, beginLocalModelRequest, INITIAL_LOCAL_MODEL_STATE, localModelEnabled,
  localModelStatus, parseLocalModelEvent, preserveLocalModelSearch, readLocalModelStream,
  type LocalCopyEvent, type LocalRankEvent } from "./local-model";
import { subscribeLocalModel } from "./use-local-model";

function rank(requestId = "current"): LocalRankEvent {
  return { type: "rank", requestId, productIds: PRODUCTS.map((p) => p.id),
    selectedProductIds: PRODUCTS.slice(0, 6).map((p) => p.id),
    variants: Object.fromEntries(PRODUCTS.slice(0, 6).map((p) => [p.id, PRODUCT_VARIANTS[p.id][0].id])),
    shortfall: 0, timing: { rankMs: 150 }, provenance: { ranking: "Learned product order", images: "Existing storefront gallery" } };
}
function copy(type: "firstThree" | "complete" = "complete", requestId = "current"): LocalCopyEvent {
  const products = PRODUCTS.slice(0, type === "firstThree" ? 3 : 6);
  const sections = Array.from({ length: products.length / 3 }, (_, section) => ({
    heading: "Your current collection",
    cards: products.slice(section * 3, section * 3 + 3).map((p) => ({ productId: p.id,
      variantId: PRODUCT_VARIANTS[p.id][0].id, title: `${p.name} in ${PRODUCT_VARIANTS[p.id][0].colorName}`,
      badge: `In ${PRODUCT_VARIANTS[p.id][0].colorName}`, bodyCopy: `This ${p.name} is shown at $${p.price}.`,
      originalName: p.name, itemPrice: p.price, selectedColor: PRODUCT_VARIANTS[p.id][0].colorName })),
  }));
  return { type, requestId, sections, timing: { readyMs: type === "firstThree" ? 1000 : 2000, copyMs: 850 },
    acceptedFields: type === "firstThree" ? 10 : 20, withheldFields: 0, requiredFields: type === "firstThree" ? 10 : 20 };
}
const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 5));

void test("local model is an exact opt-in and survives demo settings changes", () => {
  assert.equal(localModelEnabled("?model=local"), true);
  for (const search of ["", "?model=remote", "?model=LOCAL", "?other=local"]) assert.equal(localModelEnabled(search), false);
  assert.equal(preserveLocalModelSearch("headline=Hello", "?model=local"), "headline=Hello&model=local");
  assert.equal(preserveLocalModelSearch("", "?model=local"), "model=local");
  assert.equal(preserveLocalModelSearch("headline=Hello", ""), "headline=Hello");
});

void test("rank validation rejects unknown IDs, duplicates, bad variants and padded shortfalls", () => {
  assert.deepEqual(parseLocalModelEvent(rank()), rank());
  const changes = [
    { productIds: ["not-a-product"] }, { productIds: ["mug", "mug"] },
    { variants: { mug: "pack-sand" } }, { selectedProductIds: ["pack", "mug"] },
    { shortfall: 2 }, { timing: { rankMs: Infinity } },
  ];
  for (const change of changes) assert.throws(() => parseLocalModelEvent({ ...rank(), ...change }));
});

void test("copy validation checks actual product identity, exact variant, color and price", () => {
  assert.deepEqual(parseLocalModelEvent(copy()), copy());
  for (const change of [{ productId: "unknown" }, { variantId: "mug-unknown" }, { itemPrice: 23 },
    { originalName: "Another mug" }, { selectedColor: "Cobalt" }, { title: "<script>bad</script>" }]) {
    const event = copy(); Object.assign(event.sections[0].cards[0], change);
    assert.throws(() => parseLocalModelEvent(event));
  }
  const event = copy(); event.sections[1].cards[0] = { ...event.sections[0].cards[0] };
  assert.throws(() => parseLocalModelEvent(event), /repeated card/);
});

void test("nullable withheld fields stay null and counts must reflect actual content", () => {
  const event = copy(); event.sections[0].cards[0].title = null;
  event.acceptedFields -= 1; event.withheldFields += 1;
  const parsed = parseLocalModelEvent(event) as LocalCopyEvent;
  assert.equal(parsed.sections[0].cards[0].title, null);
  assert.equal(parsed.acceptedFields, 19);
  assert.throws(() => parseLocalModelEvent({ ...event, acceptedFields: 20 }));
  assert.throws(() => parseLocalModelEvent({ ...event, acceptedFields: 18, withheldFields: 2 }));
});

void test("hard guard shortfalls and zero eligible results never get padded", () => {
  const event = { ...rank(), productIds: [] as LocalRankEvent["productIds"], selectedProductIds: [] as LocalRankEvent["selectedProductIds"], variants: {}, shortfall: 6 };
  let state = beginLocalModelRequest(INITIAL_LOCAL_MODEL_STATE, "size M");
  state = applyLocalModelEvent(state, parseLocalModelEvent(event), "size M");
  assert.deepEqual(state.rank?.productIds, []);
  state = applyLocalModelEvent(state, parseLocalModelEvent({ type: "complete", requestId: "current", sections: [],
    timing: { readyMs: 160, copyMs: 0 }, acceptedFields: 0, withheldFields: 0, requiredFields: 0 }), "size M");
  assert.equal(state.phase, "complete");
  assert.match(localModelStatus(state), /0 eligible products/);
});

void test("stale query and stale request updates cannot replace current results", () => {
  let state = beginLocalModelRequest(INITIAL_LOCAL_MODEL_STATE, "new query");
  assert.equal(applyLocalModelEvent(state, rank("old"), "old query"), state);
  state = applyLocalModelEvent(state, rank(), "new query");
  assert.equal(applyLocalModelEvent(state, copy("complete", "old"), "new query"), state);
  state = applyLocalModelEvent(state, copy("firstThree"), "new query");
  state = applyLocalModelEvent(state, copy(), "new query");
  assert.equal(state.phase, "complete");
  assert.equal(state.firstThreeReadyMs, 1000);
  assert.equal(applyLocalModelEvent(state, copy("firstThree"), "new query"), state);
  assert.equal(applyLocalModelEvent(state, rank("late duplicate"), "new query"), state);
  assert.match(localModelStatus(state), /First cards 1.00 s.*Complete copy 2.00 s.*20\/20 fresh fields/);
});

void test("copy cannot change selected product order or the bound variant", () => {
  const state = applyLocalModelEvent(beginLocalModelRequest(INITIAL_LOCAL_MODEL_STATE, "mug"), rank(), "mug");
  const swapped = copy(); swapped.sections[0].cards.reverse();
  assert.throws(() => applyLocalModelEvent(state, swapped, "mug"), /selected products/);
  const color = copy(); color.sections[0].cards[0].variantId = PRODUCT_VARIANTS.mug[1].id;
  assert.throws(() => applyLocalModelEvent(state, color, "mug"), /selected products/);
});

void test("failure retains last safe display with an explicit error", () => {
  let state = beginLocalModelRequest(INITIAL_LOCAL_MODEL_STATE, "mug");
  state = applyLocalModelEvent(state, rank(), "mug"); state = applyLocalModelEvent(state, copy(), "mug");
  const next = beginLocalModelRequest(state, "bag");
  const failed = applyLocalModelEvent(next, { type: "error", message: "Service disconnected." }, "bag");
  assert.equal(failed.rank, state.rank); assert.equal(failed.copy, state.copy);
  assert.equal(failed.displayQuery, "mug");
  assert.match(localModelStatus(failed), /unavailable.*Last available result/);
});

void test("NDJSON decoder supports split UTF-8 chunks and streamed first/full updates", async () => {
  const encoded = new TextEncoder().encode([rank(), copy("firstThree"), copy()].map((e) => JSON.stringify(e)).join("\n"));
  const updates: string[] = [];
  const response = new Response(new ReadableStream({ start(controller) {
    for (let index = 0; index < encoded.length; index += 37) controller.enqueue(encoded.slice(index, index + 37));
    controller.close();
  } }));
  await readLocalModelStream(response, (event) => updates.push(event.type));
  assert.deepEqual(updates, ["rank", "firstThree", "complete"]);
  await assert.rejects(readLocalModelStream(new Response(JSON.stringify(rank()) + "\n"), () => {}), /stopped before finishing/);
  await assert.rejects(readLocalModelStream(new Response("unavailable", { status: 503 }), () => {}), /503/);
});

void test("StrictMode unsubscribe/resubscribe shares one fetch and ignores detached listeners", async () => {
  const originalFetch = globalThis.fetch;
  let calls = 0, signal: AbortSignal | null | undefined;
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const stream = new ReadableStream<Uint8Array>({ start(value) { controller = value; } });
  globalThis.fetch = (async (_url, options) => { calls += 1; signal = options?.signal; return new Response(stream); }) as typeof fetch;
  const detached: string[] = [], current: string[] = [];
  try {
    const unsubscribe = subscribeLocalModel({ query: "strict-mode-fixture" }, { event: (e) => detached.push(e.type), error: () => {} });
    unsubscribe();
    const cleanup = subscribeLocalModel({ query: "strict-mode-fixture" }, { event: (e) => current.push(e.type), error: () => {} });
    await tick();
    controller.enqueue(new TextEncoder().encode(JSON.stringify(rank()) + "\n")); await tick();
    assert.equal(calls, 1); assert.deepEqual(detached, []); assert.deepEqual(current, ["rank"]);
    cleanup(); await tick(); assert.equal(signal?.aborted, true);
    controller.close();
  } finally { globalThis.fetch = originalFetch; }
});

void test("product copy disappears on a shopper variant change; price and cart action stay catalog based", () => {
  const card = copy().sections[0].cards[0];
  const props = { product: PRODUCTS[0], context: "studio" as const, localCopy: card,
    onVariantChange: () => {}, onAdd: (_variant: string, _color: string) => {} };
  const initial = ProductCard(props);
  const changed = ProductCard({ ...props, variantId: PRODUCT_VARIANTS.mug[1].id });
  const stringify = (value: unknown): string => JSON.stringify(value, (_key, v: unknown) => typeof v === "symbol" ? String(v) : v);
  assert.match(stringify(initial), /Trail mug in Sage/);
  assert.doesNotMatch(stringify(changed), /Trail mug in Sage/);
  assert.match(stringify(changed), /Trail mug/);
  const children = changed.props.children;
  const add = children[children.length - 1];
  let selected = "";
  const actionable = ProductCard({ ...props, variantId: PRODUCT_VARIANTS.mug[1].id, onAdd: (id) => { selected = id; } });
  actionable.props.children.at(-1).props.onClick();
  assert.equal(selected, PRODUCT_VARIANTS.mug[1].id);
  assert.equal(add.props.className, "add-button");
});
