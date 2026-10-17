import { VariantConfigSchema, type VariantConfig } from "@storefront/contracts";
export type { VariantConfig } from "@storefront/contracts";

export const DEFAULT_CONFIG: VariantConfig = {
  headline: "Everyday goods for a little more outside.",
  showShipping: false,
  recommendationOrder: "popular",
};

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
] as const;
export type Product = (typeof PRODUCTS)[number];

export function orderedProducts(order: VariantConfig["recommendationOrder"]): Product[] {
  if (order === "budget") return [...PRODUCTS].sort((a, b) => a.price - b.price);
  // No personal history is collected in this shop. A server adapter owns supplied history.
  if (order === "history") return [...PRODUCTS].reverse();
  return [...PRODUCTS];
}
