import { useRef, useState, type KeyboardEvent } from "react";
import type { Product } from "./config";
import { getVariantImages, type ProductVariant } from "./variants";
import { ProductPhoto } from "./ProductPhoto";
import { selectShowcaseImage, type ImageContext } from "./images";

export function ProductGallery({
  product,
  variant,
  context,
}: {
  product: Product;
  variant: ProductVariant;
  context: ImageContext;
}) {
  const views = getVariantImages(variant);
  const [selectedLabel, setSelectedLabel] = useState(
    () => selectShowcaseImage(product.id, context, views).label,
  );
  const [expanded, setExpanded] = useState(false);
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const selectedIndex = Math.max(
    0,
    views.findIndex((image) => image.label === selectedLabel),
  );
  const currentView = views[selectedIndex];

  // Roving tabindex: the photo strip is one tab stop and arrow keys move within it.
  function moveSelection(index: number) {
    const clamped = Math.max(0, Math.min(index, views.length - 1));
    setExpanded(true);
    setSelectedLabel(views[clamped].label);
    requestAnimationFrame(() => buttons.current[clamped]?.focus());
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
          {views
            .map((view, index) => ({ view, index }))
            .filter(({ index }) => expanded || index < 2 || index === Math.max(2, selectedIndex))
            .map(({ view, index }) => (
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
          {views.length > 3 && (
            <button
              type="button"
              className="gallery-more"
              aria-label={
                expanded
                  ? `Show fewer ${product.name} photos`
                  : `Show ${views.length - 3} more ${product.name} photos`
              }
              aria-expanded={expanded}
              onClick={() => setExpanded((value) => !value)}
            >
              {expanded ? "Less" : `+${views.length - 3}`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
