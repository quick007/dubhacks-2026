import { HugeiconsIcon } from "@hugeicons/react";
import { Add01Icon, MinusIcon } from "@hugeicons/core-free-icons";
import type { CartLine } from "./cart";
import type { ShippingQuote } from "./shipping";

export function CartView({
  lines,
  subtotal,
  quote,
  total,
  onQuantity,
  onRemove,
  onPlaceOrder,
  onContinue,
}: {
  lines: CartLine[];
  subtotal: number;
  quote: ShippingQuote | null;
  total: number;
  onQuantity: (variantId: string, change: 1 | -1) => void;
  onRemove: (variantId: string) => void;
  onPlaceOrder: () => void;
  onContinue: () => void;
}) {
  return (
    <section className="cart-page">
      <h1 tabIndex={-1}>Your demo cart</h1>
      {lines.length === 0 ? (
        <p>Your cart is empty.</p>
      ) : (
        <>
          <div className="cart-items">
            {lines.map(({ product, variant, quantity, subtotal: lineSubtotal }) => (
              <div className="cart-line" key={variant.id}>
                <div className="cart-line-info">
                  <span>
                    {product.name} <small>{variant.colorName}</small>
                  </span>
                  <div className="cart-line-controls">
                    <button
                      type="button"
                      className="stepper-button"
                      aria-label={`Decrease ${product.name}, ${variant.colorName} quantity`}
                      disabled={quantity === 1}
                      onClick={() => onQuantity(variant.id, -1)}
                    >
                      <HugeiconsIcon icon={MinusIcon} size={16} aria-hidden="true" />
                    </button>
                    <span className="cart-line-quantity">{quantity}</span>
                    <button
                      type="button"
                      className="stepper-button"
                      aria-label={`Increase ${product.name}, ${variant.colorName} quantity`}
                      onClick={() => onQuantity(variant.id, 1)}
                    >
                      <HugeiconsIcon icon={Add01Icon} size={16} aria-hidden="true" />
                    </button>
                    <button
                      type="button"
                      className="cart-line-remove"
                      onClick={() => onRemove(variant.id)}
                    >
                      Remove
                    </button>
                  </div>
                </div>
                <span className="cart-line-price">${lineSubtotal}</span>
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
