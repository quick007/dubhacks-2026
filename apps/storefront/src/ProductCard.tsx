import { HugeiconsIcon } from "@hugeicons/react";
import { Add01Icon } from "@hugeicons/core-free-icons";
import type { Product } from "./config";
import type { ImageContext } from "./images";
import { ProductGallery } from "./ProductGallery";
import { PRODUCT_VARIANTS, getProductVariant } from "./variants";
import type { LocalCopyCard } from "./local-model";

export function ProductCard({
  product,
  context,
  variantId,
  localCopy,
  onVariantChange,
  onAdd,
}: {
  product: Product;
  context: ImageContext;
  variantId?: string;
  localCopy?: LocalCopyCard;
  onVariantChange: (variantId: string) => void;
  onAdd: (variantId: string, colorName: string) => void;
}) {
  const variant = getProductVariant(product.id, variantId);
  const freshCopy = localCopy?.variantId === variant.id ? localCopy : undefined;
  return (
    <article className="product">
      <ProductGallery
        key={`${product.id}:${context}`}
        product={product}
        preferredContext={context}
        variant={variant}
      />
      <div className="product-heading">
        <h3>{freshCopy?.title ?? product.name}</h3>
        <span>${product.price}</span>
      </div>
      {freshCopy?.title && freshCopy.title !== product.name && (
        <p className="search-status">{product.name}</p>
      )}
      {freshCopy?.badge && <p className="search-status">{freshCopy.badge}</p>}
      {freshCopy?.bodyCopy && <p className="search-status">{freshCopy.bodyCopy}</p>}
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
