import { PRODUCTS, type Product } from "./config";
import { getProductVariant, type ProductVariant } from "./variants";

export type CartItem = { productId: Product["id"]; variantId: string };
export type CartLine = {
  product: Product;
  variant: ProductVariant;
  quantity: number;
  subtotal: number;
};

export function addCartItem(items: CartItem[], item: CartItem): CartItem[] {
  return [...items, item];
}

/** Removes a single unit, preferring the most recently added matching item. */
export function removeOneCartItem(items: CartItem[], variantId: string): CartItem[] {
  for (let index = items.length - 1; index >= 0; index -= 1) {
    if (items[index].variantId === variantId) {
      return [...items.slice(0, index), ...items.slice(index + 1)];
    }
  }
  return items;
}

export function removeCartLine(items: CartItem[], variantId: string): CartItem[] {
  return items.filter((item) => item.variantId !== variantId);
}

export function getCartLines(items: CartItem[]): CartLine[] {
  const lines = new Map<string, CartLine>();
  for (const item of items) {
    const product = PRODUCTS.find((candidate) => candidate.id === item.productId)!;
    const variant = getProductVariant(item.productId, item.variantId);
    const existing = lines.get(variant.id);
    if (existing) {
      existing.quantity += 1;
      existing.subtotal += product.price;
    } else {
      lines.set(variant.id, { product, variant, quantity: 1, subtotal: product.price });
    }
  }
  return [...lines.values()];
}
