import { HugeiconsIcon } from "@hugeicons/react";
import { Add01Icon } from "@hugeicons/core-free-icons";
import type { Product } from "./config";
import type { ImageContext } from "./images";
import { ProductGallery } from "./ProductGallery";
import { PRODUCT_VARIANTS, getProductVariant } from "./variants";

export function ProductCard({
  product,
  context,
  variantId,
  onVariantChange,
  onAdd,
}: {
  product: Product;
  context: ImageContext;
  variantId?: string;
  onVariantChange: (variantId: string) => void;
  onAdd: (variantId: string, colorName: string) => void;
}) {
  const variant = getProductVariant(product.id, variantId);
  return (
    <article className="product">
      <ProductGallery key={`${product.id}:${context}`} product={product} variant={variant} />
      <div className="product-heading">
        <h3>{product.name}</h3>
        <span>${product.price}</span>
      </div>
      <div className="product-colors" role="group" aria-label={`${product.name} colors`}>
        {PRODUCT_VARIANTS[product.id].map((option) => (
          <button
            type="button"
            key={option.id}
            aria-pressed={option.id === variant.id}
            aria-label={`${product.name} color: ${option.colorName}`}
            title={option.colorName}
            onClick={() => onVariantChange(option.id)}
          >
            <span
              className="color-swatch"
              style={{ background: option.swatch }}
              aria-hidden="true"
            />
          </button>
        ))}
        <span className="selected-color" aria-live="polite">
          {variant.colorName}
        </span>
      </div>
      <button className="add-button" onClick={() => onAdd(variant.id, variant.colorName)}>
        <HugeiconsIcon icon={Add01Icon} size={18} aria-hidden="true" />
        Add to cart
      </button>
    </article>
  );
}
