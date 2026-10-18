import type { CartItem } from "./cart";

/** A person's current basket survives a switch; order authority never does. */
export function switchShoppingPerson<T>(state: { personId: string; cart: CartItem[]; carts: Record<string, CartItem[]>; checkoutReceipt: T | null }, nextPersonId: string) {
  if (state.personId === nextPersonId) return state;
  const carts = { ...state.carts, [state.personId]: state.cart };
  return { personId: nextPersonId, cart: carts[nextPersonId] ?? [], carts, checkoutReceipt: null };
}

export function checkoutStillCurrent(startedId: string, activeId: string): boolean {
  return startedId === activeId;
}

/** Every asynchronous outcome checks the same person/cart checkout authority. */
export async function runCurrentCheckout<T>(submit: () => Promise<T>, isCurrent: () => boolean,
  handlers: { confirmed: (receipt: T) => void; failed: () => void; settled: () => void }): Promise<void> {
  try {
    const receipt = await submit();
    if (isCurrent()) handlers.confirmed(receipt);
  } catch {
    if (isCurrent()) handlers.failed();
  } finally {
    if (isCurrent()) handlers.settled();
  }
}
