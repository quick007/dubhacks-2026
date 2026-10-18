/** Empty product-shaped placeholders; never substitute catalog recommendations. */
export function ProductSkeleton() {
  return (
    <div className="product product-skeleton" aria-hidden="true">
      <div className="skeleton-surface skeleton-photo" />
      <div className="skeleton-surface skeleton-title" />
      <div className="skeleton-surface skeleton-detail" />
      <div className="skeleton-surface skeleton-detail skeleton-short" />
      <div className="skeleton-surface skeleton-action" />
    </div>
  );
}
