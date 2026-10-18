import { PRODUCTS, type Product } from "./config";
import { PRODUCT_IMAGES, type ProductImage } from "./images";
import { NEW_PRODUCTS } from "./catalog-new-products";
import galleryLayouts from "./color-gallery-layouts.json";

type ColorOption = readonly [name: string, swatch: string];
const COLOR_OPTIONS: Record<Product["id"], readonly ColorOption[]> = {
  umbrella: NEW_PRODUCTS[0].colorOptions,
  cap: NEW_PRODUCTS[1].colorOptions,
  speaker: NEW_PRODUCTS[2].colorOptions,
  stand: NEW_PRODUCTS[3].colorOptions,
  cubes: NEW_PRODUCTS[4].colorOptions,
  planter: NEW_PRODUCTS[5].colorOptions,
  mug: [
    ["Sage", "#819374"],
    ["Ivory", "#ded7c7"],
    ["Cobalt", "#315fbc"],
  ],
  pack: [
    ["Sand", "#c4af8c"],
    ["Slate blue", "#607a91"],
    ["Forest", "#405c48"],
  ],
  notebook: [
    ["Terracotta", "#a86348"],
    ["Navy", "#263e62"],
    ["Olive", "#6a734e"],
  ],
  jacket: [
    ["Forest", "#3b503d"],
    ["Navy", "#283f65"],
    ["Ochre", "#bd8d36"],
  ],
  bottle: [
    ["Graphite", "#53575a"],
    ["Cream", "#ded9cc"],
    ["Steel blue", "#587a91"],
  ],
  tote: [
    ["Olive", "#73734f"],
    ["Natural", "#d7cbb0"],
    ["Navy", "#304564"],
  ],
  blanket: [
    ["Rust", "#a55d43"],
    ["Ocean blue", "#4d7693"],
    ["Moss", "#667456"],
  ],
  lamp: [
    ["Ivory", "#ece5d4"],
    ["Slate blue", "#6c8499"],
    ["Terracotta", "#b46c52"],
  ],
  lunchbox: [
    ["Olive band", "#73805a"],
    ["Navy band", "#30496c"],
    ["Rust band", "#a95c3f"],
  ],
  headphones: [
    ["Navy", "#2b405c"],
    ["Sand", "#cbbda2"],
    ["Graphite", "#444b52"],
  ],
  sneakers: [
    ["Off-white", "#dedacb"],
    ["Charcoal", "#515657"],
    ["Olive", "#6a7457"],
  ],
  sweatshirt: [
    ["Oatmeal", "#d9d2c2"],
    ["Slate blue", "#6b8396"],
    ["Forest", "#46634e"],
  ],
  duffel: [
    ["Charcoal", "#575858"],
    ["Sand", "#bfac8c"],
    ["Olive", "#677154"],
  ],
  sunglasses: [
    ["Tortoise", "#805731"],
    ["Black", "#24272b"],
    ["Amber", "#b27a36"],
  ],
  organizer: [
    ["Ochre", "#bf9337"],
    ["Navy", "#2d4667"],
    ["Sage", "#88977a"],
  ],
  towel: [
    ["Sage stripe", "#96a38c"],
    ["Blue stripe", "#5c83a2"],
    ["Clay stripe", "#b8775e"],
  ],
};

export type ProductVariant = {
  id: string;
  productId: Product["id"];
  colorName: string;
  swatch: string;
  image?: ProductImage;
};

export const PRODUCT_VARIANTS = Object.fromEntries(
  PRODUCTS.map((product) => [
    product.id,
    COLOR_OPTIONS[product.id].map(([colorName, swatch], index): ProductVariant => {
      const id = `${product.id}-${colorName.toLowerCase().replace(/\s+/g, "-")}`;
      return {
        id,
        productId: product.id,
        colorName,
        swatch,
        ...(index === 0
          ? {}
          : {
              image: {
                id: `${id}-studio`,
                src: PRODUCT_IMAGES[product.id][0].src.endsWith("-color-gallery.jpg")
                  ? PRODUCT_IMAGES[product.id][0].src
                  : `/products/${product.id}-colors.jpg`,
                label: colorName,
                alt: `${product.name} — ${colorName.toLowerCase()} color, studio view`,
                context: "studio",
                position: index === 1 ? "0% center" : "100% center",
                tags: ["product-only", "color-variant", colorName.toLowerCase()],
                presentation: {
                  subject: "product",
                  background: "plain",
                  tone: "light",
                  composition: "full-product",
                },
              },
            }),
      };
    }),
  ]),
) as Record<Product["id"], ProductVariant[]>;

export function getProductVariant(productId: Product["id"], variantId?: string): ProductVariant {
  const variants = PRODUCT_VARIANTS[productId];
  return variants.find((variant) => variant.id === variantId) ?? variants[0];
}

export function getVariantImages(variant: ProductVariant): ProductImage[] {
  const originals = PRODUCT_IMAGES[variant.productId];
  if (!variant.image) return originals;
  const product = PRODUCTS.find((item) => item.id === variant.productId)!;
  const layout = galleryLayouts[variant.id as keyof typeof galleryLayouts];
  return originals.map((image, index) => {
    const tabletop =
      image.id === `${variant.productId}-tabletop` || image.id === "notebook-outdoors";
    const sharedColors = image.src.endsWith("-color-gallery.jpg");
    const colorIndex = PRODUCT_VARIANTS[variant.productId].findIndex(
      (item) => item.id === variant.id,
    );
    return {
      ...image,
      id: `${variant.id}-${image.id}`,
      src: tabletop || sharedColors ? image.src : `/products/${variant.id}-gallery.jpg`,
      alt: `${product.name} in ${variant.colorName.toLowerCase()} — ${image.label.toLowerCase()}`,
      tags: [...image.tags, "color-variant", variant.colorName.toLowerCase()],
      position: "center",
      tile: sharedColors
        ? { ...image.tile!, index: image.tile!.index + colorIndex * 3 }
        : tabletop
          ? {
              ...image.tile!,
              index: PRODUCT_VARIANTS[variant.productId].findIndex(
                (item) => item.id === variant.id,
              ),
            }
          : { ...layout, index },
    };
  });
}
