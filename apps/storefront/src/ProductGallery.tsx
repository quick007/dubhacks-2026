import { useState } from "react";
import type { Product } from "./config";
import { selectProductImage, type ImageContext } from "./images";
import { getVariantImages, type ProductVariant } from "./variants";
import { ProductPhoto } from "./ProductPhoto";

export function ProductGallery({
  product,
  preferredContext,
  variant,
}: {
  product: Product;
  preferredContext: ImageContext;
  variant: ProductVariant;
}) {
  const views = getVariantImages(variant);
  const [selectedLabel, setSelectedLabel] = useState(() => {
    if (preferredContext === "studio") {
      return (
        views.find((view) => view.label === "Still life") ??
        views.find(
          (view) =>
            view.presentation.subject === "product" && view.presentation.background !== "plain",
        ) ??
        views.find((view) => view.presentation.subject === "product") ??
        views[0]
      ).label;
    }
    const preferred = selectProductImage(product.id, preferredContext);
    return preferred.label;
  });
  const selectedIndex = Math.max(
    0,
    views.findIndex((image) => image.label === selectedLabel),
  );
  const currentView = views[selectedIndex];
  return (
    <div className="product-gallery">
      <div className={`product-art ${product.color}`}>
        <ProductPhoto image={currentView} id={`${product.id}-photo`} />
      </div>
      {views.length > 1 && (
        <div className="product-thumbnails" role="group" aria-label={`${product.name} photos`}>
          {views.map((view) => (
            <button
              key={view.id}
              type="button"
              aria-label={`Show ${product.name}: ${view.label}`}
              aria-pressed={selectedLabel === view.label}
              aria-controls={`${product.id}-photo`}
              onClick={() => setSelectedLabel(view.label)}
            >
              <ProductPhoto image={view} alt="" />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
