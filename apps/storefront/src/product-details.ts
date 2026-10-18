import type { Product } from "./config";

const details: Partial<Record<Product["id"], string>> = {
  mug: "Contrasting rim · Rounded handle",
  pack: "Canvas · Front zip pocket",
  notebook: "Cloth cover · Elastic closure",
  jacket: "Hooded shell · Zip pockets",
};

export function productDetail(product: Product): string {
  return details[product.id] ?? product.category;
}
