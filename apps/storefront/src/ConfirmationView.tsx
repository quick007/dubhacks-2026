import { HugeiconsIcon } from "@hugeicons/react";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import type { ShippingQuote } from "./shipping";

export function ConfirmationView({
  itemCount,
  subtotal,
  quote,
  total,
  onBackToCollection,
}: {
  itemCount: number;
  subtotal: number;
  quote: ShippingQuote | null;
  total: number;
  onBackToCollection: () => void;
}) {
  const shipping =
    quote && (quote.amount === 0 ? " · free demo shipping" : ` · $${quote.amount} demo shipping`);
  return (
    <section className="confirmation">
      <span className="checkmark" aria-hidden="true">
        <HugeiconsIcon icon={Tick02Icon} size={36} />
      </span>
      <h1 tabIndex={-1}>Demo order confirmed</h1>
      <p>No payment was processed or real order placed.</p>
      <div className="receipt">
        {itemCount} demo item{itemCount === 1 ? "" : "s"} · ${subtotal} demo subtotal
        {shipping}
        {quote && ` · $${total} demo total`}
      </div>
      <button className="primary-button" onClick={onBackToCollection}>
        Back to the collection
      </button>
    </section>
  );
}
