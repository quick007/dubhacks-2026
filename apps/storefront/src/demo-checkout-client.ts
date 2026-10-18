/** Local simulated checkout. Sends identities/quantities only; prices come from the server. */
import { getCartLines, type CartItem } from "./cart";

export type DemoCheckoutReceipt = {
  schema: "fieldwork.demo-checkout-receipt.v1"; checkoutId: string; receiptId: string; createdAt: string;
  items: { productId: string; variantId: string; quantity: number; unitPrice: number; lineTotal: number }[];
  subtotal: number; shipping: { status: "hidden" | "free" | "flat"; cost: number | null };
  total: number; realPurchase: false;
};
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const cents = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value)
  && value >= 0 && value <= 1_000_000 && Math.abs(value * 100 - Math.round(value * 100)) < 1e-7;
const object = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("The demo order could not be verified. Please try again.");
  return value as Record<string, unknown>;
};

export function validateDemoCheckoutReceipt(value: unknown, checkoutId: string, cart: CartItem[], shippingVisible: boolean): DemoCheckoutReceipt {
  const receipt = object(value), shipping = object(receipt.shipping), lines = getCartLines(cart);
  const fail = () => { throw new Error("The demo order could not be verified. Please try again."); };
  if (!uuid.test(checkoutId) || receipt.schema !== "fieldwork.demo-checkout-receipt.v1" || receipt.checkoutId !== checkoutId
    || typeof receipt.receiptId !== "string" || !/^demo-[a-f0-9]{32}$/.test(receipt.receiptId)
    || typeof receipt.createdAt !== "string" || !Number.isFinite(Date.parse(receipt.createdAt)) || receipt.realPurchase !== false
    || !Array.isArray(receipt.items) || !lines.length || receipt.items.length !== lines.length) fail();
  const seen = new Set<string>();
  for (const raw of receipt.items as unknown[]) {
    const item = object(raw), key = `${item.productId}/${item.variantId}`;
    const line = lines.find((line) => line.product.id === item.productId && line.variant.id === item.variantId);
    if (!line || seen.has(key) || item.quantity !== line.quantity || !cents(item.unitPrice) || item.unitPrice !== line.product.price
      || !cents(item.lineTotal) || item.lineTotal !== line.subtotal) fail();
    seen.add(key);
  }
  const subtotal = lines.reduce((sum, line) => sum + line.subtotal, 0);
  const status = !shippingVisible ? "hidden" : subtotal > 50 ? "free" : "flat";
  const cost = !shippingVisible ? null : subtotal > 50 ? 0 : 5;
  if (!cents(receipt.subtotal) || receipt.subtotal !== subtotal || !cents(receipt.total)
    || shipping.status !== status || shipping.cost !== cost || receipt.total !== subtotal + (cost ?? 0)) fail();
  return receipt as unknown as DemoCheckoutReceipt;
}

export async function confirmDemoCheckout(cart: CartItem[], shippingVisible: boolean, checkoutId: string,
  fetcher: typeof fetch = fetch): Promise<DemoCheckoutReceipt> {
  if (!uuid.test(checkoutId) || !cart.length || cart.length > 1000) throw new Error("Add an item before placing a demo order.");
  const response = await fetcher("/api/demo-checkout", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ schema: "fieldwork.demo-checkout.v1", checkoutId, shippingVisible,
      items: getCartLines(cart).map((line) => ({ productId: line.product.id, variantId: line.variant.id, quantity: line.quantity })) }),
    signal: AbortSignal.timeout(10000), credentials: "same-origin" });
  if (response.status !== 202) throw new Error("The demo order was not confirmed. Please try again.");
  const text = await response.text();
  if (text.length > 64 * 1024) throw new Error("The demo order could not be verified. Please try again.");
  const body = object(JSON.parse(text));
  if (typeof body.duplicate !== "boolean") throw new Error("The demo order could not be verified. Please try again.");
  return validateDemoCheckoutReceipt(body.receipt, checkoutId, cart, shippingVisible);
}
