import assert from "node:assert/strict";
import test from "node:test";
import { confirmDemoCheckout, validateDemoCheckoutReceipt } from "./demo-checkout-client";

const checkoutId = "5d2c9d6a-4e28-42ab-8f2f-cc52804be913";
const cart = [{ productId: "bottle" as const, variantId: "bottle-graphite" }];
const receipt = () => ({ schema: "fieldwork.demo-checkout-receipt.v1", checkoutId,
  receiptId: "demo-" + "a".repeat(32), createdAt: "2026-10-18T08:00:00.000Z", realPurchase: false,
  items: [{ productId: "bottle", variantId: "bottle-graphite", quantity: 1, unitPrice: 36, lineTotal: 36 }],
  subtotal: 36, shipping: { status: "hidden", cost: null }, total: 36 });

test("demo checkout posts no client prices and validates the authoritative receipt", async () => {
  let request: unknown;
  const fetcher = (async (url, options) => {
    assert.equal(url, "/api/demo-checkout"); request = JSON.parse(String(options?.body));
    return new Response(JSON.stringify({ receipt: receipt(), duplicate: false }), { status: 202 });
  }) as typeof fetch;
  const confirmed = await confirmDemoCheckout(cart, false, checkoutId, fetcher);
  assert.equal(confirmed.total, 36);
  assert.deepEqual(request, { schema: "fieldwork.demo-checkout.v1", checkoutId, shippingVisible: false,
    items: [{ productId: "bottle", variantId: "bottle-graphite", quantity: 1 }] });
});

test("unrelated, substituted, repriced or real-payment receipts never confirm checkout", () => {
  for (const change of [
    { checkoutId: "89c501f4-7acd-469f-b384-97dd89f40f61" }, { realPurchase: true }, { total: 999 },
    { items: [{ ...receipt().items[0], variantId: "bottle-cream" }] },
    { items: [{ ...receipt().items[0], unitPrice: 1, lineTotal: 1 }], subtotal: 1, total: 1 },
    { items: [receipt().items[0], receipt().items[0]] }, { shipping: { status: "hidden", cost: 0 } },
  ]) assert.throws(() => validateDemoCheckoutReceipt({ ...receipt(), ...change }, checkoutId, cart, false));
});

test("failed responses remain unconfirmed and retries retain the caller checkout identity", async () => {
  const ids: string[] = [];
  const fetcher = (async (_url, options) => {
    ids.push(JSON.parse(String(options?.body)).checkoutId);
    return ids.length === 1 ? new Response("unavailable", { status: 503 })
      : new Response(JSON.stringify({ receipt: receipt(), duplicate: true }), { status: 202 });
  }) as typeof fetch;
  await assert.rejects(confirmDemoCheckout(cart, false, checkoutId, fetcher), /not confirmed/);
  assert.equal((await confirmDemoCheckout(cart, false, checkoutId, fetcher)).checkoutId, checkoutId);
  assert.deepEqual(ids, [checkoutId, checkoutId]);
});

test("visible shipping is verified separately from merchandise revenue", () => {
  assert.equal(validateDemoCheckoutReceipt({ ...receipt(), shipping: { status: "flat", cost: 5 }, total: 41 }, checkoutId, cart, true).total, 41);
  assert.throws(() => validateDemoCheckoutReceipt(receipt(), checkoutId, cart, true));
});
