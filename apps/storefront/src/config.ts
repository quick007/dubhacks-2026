import { VariantConfigSchema, type VariantConfig } from "@storefront/contracts";
import { ADDITIONAL_PRODUCTS } from "./catalog-expansion";
export type { VariantConfig } from "@storefront/contracts";

export const DEFAULT_CONFIG: VariantConfig = {
  headline: "Good things. Wherever life takes you.",
  showShipping: false,
  recommendationOrder: "popular",
};

/** Keeps the demo-control URL in step with the same structured settings the app reads. */
export function serializeVariantConfig(config: VariantConfig): string {
  const params = new URLSearchParams();
  if (config.headline !== DEFAULT_CONFIG.headline) params.set("headline", config.headline);
  if (config.showShipping !== DEFAULT_CONFIG.showShipping)
    params.set("showShipping", String(config.showShipping));
  if (config.recommendationOrder !== DEFAULT_CONFIG.recommendationOrder)
    params.set("recommendationOrder", config.recommendationOrder);
  return params.toString();
}

/** Only these structured query settings affect the demo. No executable code or remote URLs. */
export function parseVariantConfig(search: string): VariantConfig {
  const params = new URLSearchParams(search);
  const shipping = params.get("showShipping");
  const candidate = {
    headline: params.get("headline") ?? DEFAULT_CONFIG.headline,
    showShipping:
      shipping === null
        ? DEFAULT_CONFIG.showShipping
        : shipping === "true"
          ? true
          : shipping === "false"
            ? false
            : shipping,
    recommendationOrder: params.get("recommendationOrder") ?? DEFAULT_CONFIG.recommendationOrder,
  };
  const parsed = VariantConfigSchema.safeParse(candidate);
  return parsed.success ? parsed.data : { ...DEFAULT_CONFIG };
}

export const PRODUCTS = [
  {
    id: "mug",
    name: "Trail mug",
    category: "For the slow mornings",
    price: 24,
    color: "sage",
    kind: "mug",
  },
  {
    id: "pack",
    name: "Everyday daypack",
    category: "Pack a smaller plan",
    price: 68,
    color: "sand",
    kind: "pack",
  },
  {
    id: "notebook",
    name: "Pocket field notes",
    category: "Keep a good thought",
    price: 12,
    color: "clay",
    kind: "notebook",
  },
  {
    id: "jacket",
    name: "Everywhere rain jacket",
    category: "Waterproof hooded shell",
    price: 128,
    color: "sage",
    kind: "jacket",
  },
  ...ADDITIONAL_PRODUCTS,
] as const;
export type Product = (typeof PRODUCTS)[number];

export function orderedProducts(order: VariantConfig["recommendationOrder"]): Product[] {
  if (order === "budget") return [...PRODUCTS].sort((a, b) => a.price - b.price);
  // No personal history is collected in this shop. A server adapter owns supplied history.
  if (order === "history") return [...PRODUCTS].reverse();
  return [...PRODUCTS];
}
