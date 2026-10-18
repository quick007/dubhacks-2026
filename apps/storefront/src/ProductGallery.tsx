import { useRef, useState, type KeyboardEvent } from "react";
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
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const selectedIndex = Math.max(
    0,
    views.findIndex((image) => image.label === selectedLabel),
  );
  const currentView = views[selectedIndex];

  // Roving tabindex: the photo strip is one tab stop and arrow keys move within it.
  function moveSelection(index: number) {
    const clamped = Math.max(0, Math.min(index, views.length - 1));
    setSelectedLabel(views[clamped].label);
    buttons.current[clamped]?.focus();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    const moves: Partial<Record<string, number>> = {
      ArrowRight: selectedIndex + 1,
      ArrowLeft: selectedIndex - 1,
      Home: 0,
      End: views.length - 1,
    };
    const next = moves[event.key];
    if (next === undefined) return;
    event.preventDefault();
    moveSelection(next);
  }

  return (
    <div className="product-gallery">
      <div className={`product-art ${product.color}`}>
        <ProductPhoto image={currentView} id={`${product.id}-photo`} />
      </div>
      {views.length > 1 && (
        <div className="product-thumbnails" role="group" aria-label={`${product.name} photos`}>
          {views.map((view, index) => (
            <button
              key={view.id}
              type="button"
              ref={(element) => {
                buttons.current[index] = element;
              }}
              tabIndex={index === selectedIndex ? 0 : -1}
              aria-label={`Show ${product.name}: ${view.label}`}
              aria-pressed={index === selectedIndex}
              aria-controls={`${product.id}-photo`}
              onClick={() => setSelectedLabel(view.label)}
              onKeyDown={handleKeyDown}
            >
              <ProductPhoto image={view} alt="" />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
