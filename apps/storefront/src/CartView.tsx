import type { CartLine } from "./cart";
import type { ShippingQuote } from "./shipping";

export function CartView({
  lines,
  subtotal,
  quote,
  total,
  onPlaceOrder,
  onContinue,
}: {
  lines: CartLine[];
  subtotal: number;
  quote: ShippingQuote | null;
  total: number;
  onPlaceOrder: () => void;
  onContinue: () => void;
}) {
  return (
    <section className="cart-page">
      <h1>Your demo cart</h1>
      {lines.length === 0 ? (
        <p>Your cart is empty.</p>
      ) : (
        <>
          <div className="cart-items">
            {lines.map(({ product, variant, quantity, subtotal: lineSubtotal }) => (
              <div key={variant.id}>
                <span>
                  {product.name}{" "}
                  <small>
                    {variant.colorName} · × {quantity}
                  </small>
                </span>
                <span>${lineSubtotal}</span>
              </div>
            ))}
          </div>
          <div className="cart-summary">
            <div>
              <span>Demo subtotal</span>
              <span>${subtotal}</span>
            </div>
            {quote && (
              <div>
                <span>Demo shipping</span>
                <span>{quote.label}</span>
              </div>
            )}
            {quote && (
              <div className="cart-grand-total">
                <span>Demo total</span>
                <strong>${total}</strong>
              </div>
            )}
          </div>
          <p className="demo-note">No payment required. Nothing will be shipped.</p>
          <button className="primary-button" onClick={onPlaceOrder}>
            Place demo order
          </button>
        </>
      )}
      <button className="text-button" onClick={onContinue}>
        Continue browsing
      </button>
    </section>
  );
}
