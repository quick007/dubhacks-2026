import source from "../../../packages/contracts/data/expanded-store-catalog-v1.json";
import type { ProductImage } from "./images";

type PublicVariant = { variantId: string; colorName: string; colorHex: string; images: ProductImage[] };
type PublicProduct = { productId: string; name: string; category: string; itemPrice: number; kind: string;
  searchTerms: string[]; intentTags: string[]; parentFamily: string; variants: PublicVariant[] };
type PublicCatalog = { version: "expanded-store-public-catalog.v1"; products: PublicProduct[] };
const contexts = new Set(["studio", "outdoors", "city", "campus", "family", "everyday"]);
const text = (value: unknown): value is string => typeof value === "string" && value.length > 0 && value.length <= 500;
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value);

/** Public fixture metadata only. Activation separately requires the complete 35-product packet. */
export function parseExpandedCatalog(value: unknown): PublicCatalog {
  if (!record(value) || value.version !== "expanded-store-public-catalog.v1" || !Array.isArray(value.products)) throw new Error("Invalid expanded catalog version.");
  const ids = new Set<string>(), variants = new Set<string>();
  for (const product of value.products) {
    if (!record(product) || !text(product.productId) || !/^[a-z0-9-]+$/.test(product.productId) || ids.has(product.productId)
      || !text(product.name) || !text(product.category) || !text(product.kind) || !Number.isFinite(product.itemPrice) || Number(product.itemPrice) <= 0
      || !Array.isArray(product.variants) || !product.variants.length) throw new Error("Invalid or duplicate public product.");
    ids.add(product.productId);
    for (const variant of product.variants) {
      if (!record(variant) || !text(variant.variantId) || variants.has(variant.variantId) || !variant.variantId.startsWith(`${product.productId}-`)
        || !text(variant.colorName) || !text(variant.colorHex) || !/^#[a-fA-F0-9]{6}$/.test(variant.colorHex)
        || !Array.isArray(variant.images) || !variant.images.length) throw new Error("Invalid public variant.");
      variants.add(variant.variantId);
      for (const image of variant.images) {
        if (!record(image) || !text(image.id) || !text(image.src) || !/^\/products\/[a-z0-9-]+\.(?:jpg|png|webp)$/.test(image.src)
          || !text(image.label) || !text(image.alt) || !contexts.has(String(image.context)) || !text(image.position)
          || !record(image.presentation) || !Array.isArray(image.tags) || image.tags.some(tag => typeof tag !== "string")) throw new Error("Invalid public fixture image.");
        if (image.tile !== undefined) {
          const tile = image.tile;
          if (!record(tile) || ![tile.columns, tile.rows, tile.index].every(Number.isInteger) || Number(tile.columns) < 1 || Number(tile.rows) < 1
            || Number(tile.index) < 0 || Number(tile.index) >= Number(tile.columns) * Number(tile.rows) || !Number.isFinite(tile.sheetAspect) || Number(tile.sheetAspect) <= 0) throw new Error("Invalid fixture image tile.");
        }
      }
    }
  }
  return value as unknown as PublicCatalog;
}

export const EXPANDED_PUBLIC_CATALOG = parseExpandedCatalog(source);
export function additionalPublicProducts(originalIds: readonly string[], expectedTotal = 35) {
  if (EXPANDED_PUBLIC_CATALOG.products.length !== expectedTotal) throw new Error(`Expanded model catalog requires ${expectedTotal} products before activation.`);
  const originals = new Set(originalIds);
  if ([...originals].some(id => !EXPANDED_PUBLIC_CATALOG.products.some(product => product.productId === id))) throw new Error("Expanded catalog omitted an original product.");
  return EXPANDED_PUBLIC_CATALOG.products.filter(product => !originals.has(product.productId)).map(product => ({
    id: product.productId, name: product.name, category: product.category, price: product.itemPrice,
    color: "sage", kind: product.kind,
  }));
}
export function explicitPublicVariants(productId: string) {
  const product = EXPANDED_PUBLIC_CATALOG.products.find(product => product.productId === productId);
  if (!product) throw new Error("Unknown expanded product.");
  return product.variants.map(variant => ({ id: variant.variantId, productId, colorName: variant.colorName,
    swatch: variant.colorHex, image: variant.images[0] }));
}
export function explicitPublicImages(productId: string, variantId?: string): ProductImage[] {
  const product = EXPANDED_PUBLIC_CATALOG.products.find(product => product.productId === productId);
  const variant = variantId ? product?.variants.find(variant => variant.variantId === variantId) : product?.variants[0];
  if (!variant) throw new Error("Unknown expanded product variant.");
  return variant.images;
}
