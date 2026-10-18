export const FREE_SHIPPING_OVER = 50;
export const FLAT_SHIPPING_COST = 5;

export type ShippingQuote = { label: string; amount: number };

/**
 * Fixture delivery terms, the same fact the hero states: free over $50, $5 flat
 * rate otherwise. Hidden configurations deliberately keep terms out of the cart
 * so the shipping-visible variant has a measurable difference to show.
 */
export function shippingQuote(subtotal: number, showShipping: boolean): ShippingQuote | null {
  if (!showShipping) return null;
  const free = subtotal > FREE_SHIPPING_OVER;
  return {
    label: free ? "Free" : `$${FLAT_SHIPPING_COST}`,
    amount: free ? 0 : FLAT_SHIPPING_COST,
  };
}

export function orderTotal(subtotal: number, quote: ShippingQuote | null): number {
  return subtotal + (quote?.amount ?? 0);
}
