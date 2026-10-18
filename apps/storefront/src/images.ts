import type { Product } from "./config";
import { ADDITIONAL_PRODUCTS } from "./catalog-expansion";

export type ImageContext = "studio" | "outdoors" | "city" | "campus" | "family" | "everyday";
type Presentation = {
  subject: "product" | "in-use";
  background: "plain" | "textured" | "wood" | "location";
  tone: "light" | "dark" | "warm" | "natural";
  composition: "full-product" | "detail" | "flat-lay" | "lifestyle";
};
export type ProductImage = {
  id: string;
  src: string;
  label: string;
  alt: string;
  context: ImageContext;
  presentation: Presentation;
  tags: string[];
  position: string;
  tile?: { columns: number; rows: number; index: number; sheetAspect: number };
};

function studioImages(productId: Product["id"], name: string): ProductImage[] {
  return ["Front", "Back", "Detail"].map((label, index) => ({
    id: `${productId}-${label.toLowerCase()}`,
    src: `/products/${productId}-views.png`,
    label,
    alt: `${name} — ${label.toLowerCase()} view`,
    context: "studio",
    presentation: {
      subject: "product",
      background: "plain",
      tone: "light",
      composition: index === 2 ? "detail" : "full-product",
    },
    tags: ["product", label.toLowerCase()],
    position: `${index * 50}% center`,
  }));
}

function scene(
  productId: Product["id"],
  file: string,
  context: ImageContext,
  label: string,
  alt: string,
  tags: string[],
): ProductImage {
  return {
    id: `${productId}-${file}`,
    src: `/products/${productId}-${file}.jpg`,
    context,
    label,
    alt,
    tags,
    position: "center",
    presentation: {
      subject: "in-use",
      background: "location",
      tone: "natural",
      composition: "lifestyle",
    },
  };
}

function stillLife(
  productId: Product["id"],
  style: string,
  label: string,
  alt: string,
  presentation: Presentation,
): ProductImage {
  return {
    id: `${productId}-${style}`,
    src: `/products/${productId}-${style}.jpg`,
    context: "studio",
    label,
    alt,
    tags: ["product-only", style],
    position: "center",
    presentation,
  };
}

const originalImages = {
  mug: [
    ...studioImages("mug", "Trail mug"),
    scene(
      "mug",
      "outdoors",
      "outdoors",
      "At camp",
      "A camper drinking from the sage mug beside a wooded lake",
      ["camping", "coffee", "lake"],
    ),
    scene(
      "mug",
      "everyday",
      "everyday",
      "At home",
      "A person drinking coffee from the sage mug in a sunlit kitchen",
      ["home", "breakfast", "coffee"],
    ),
  ],
  pack: [
    ...studioImages("pack", "Everyday daypack"),
    scene(
      "pack",
      "outdoors",
      "outdoors",
      "On the trail",
      "A hiker wearing the tan canvas daypack on a forest trail",
      ["hiking", "trail", "weekend"],
    ),
    scene(
      "pack",
      "everyday",
      "city",
      "In the city",
      "A commuter wearing the same tan daypack on a city sidewalk",
      ["commute", "city", "work"],
    ),
    scene(
      "pack",
      "campus",
      "campus",
      "On campus",
      "A university student wearing the tan daypack while carrying books across campus",
      ["college", "study", "campus", "books"],
    ),
    scene(
      "pack",
      "family",
      "family",
      "Family days",
      "A parent wearing the tan daypack while walking with their child in a park",
      ["parenting", "park", "family", "day trip"],
    ),
  ],
  notebook: [
    ...studioImages("notebook", "Pocket field notes"),
    scene(
      "notebook",
      "outdoors",
      "outdoors",
      "Outside",
      "A person writing in the terracotta notebook at a lakeside picnic table",
      ["travel", "journaling", "outdoors"],
    ),
    scene(
      "notebook",
      "everyday",
      "everyday",
      "At your desk",
      "A person writing in the terracotta notebook at a home office desk",
      ["work", "study", "journaling", "home"],
    ),
  ],
  jacket: [
    ...studioImages("jacket", "Everywhere rain jacket"),
    scene(
      "jacket",
      "outdoors",
      "outdoors",
      "On the trail",
      "A hiker wearing the forest green rain jacket on a misty forest trail",
      ["rain", "hiking", "forest", "outdoors"],
    ),
    scene(
      "jacket",
      "city",
      "city",
      "In the city",
      "A commuter wearing the same forest green rain jacket on a rainy city street",
      ["rain", "commute", "city", "work"],
    ),
    scene(
      "jacket",
      "campus",
      "campus",
      "On campus",
      "A student wearing the forest green rain jacket while carrying a textbook across a rainy campus",
      ["rain", "college", "campus", "books"],
    ),
    scene(
      "jacket",
      "family",
      "family",
      "Family days",
      "A parent wearing the forest green rain jacket beside a rainy neighborhood soccer field",
      ["rain", "family", "soccer", "weekend"],
    ),
    stillLife(
      "jacket",
      "plain",
      "Plain flat lay",
      "The forest green jacket laid flat on a plain ivory background",
      { subject: "product", background: "plain", tone: "light", composition: "flat-lay" },
    ),
    stillLife(
      "jacket",
      "dark",
      "Dark flat lay",
      "The same forest green jacket laid flat on dark scratched slate",
      { subject: "product", background: "textured", tone: "dark", composition: "flat-lay" },
    ),
  ],
};

const stillLifePresentations: Record<(typeof ADDITIONAL_PRODUCTS)[number]["id"], Presentation> = {
  bottle: {
    subject: "product",
    background: "textured",
    tone: "light",
    composition: "full-product",
  },
  tote: { subject: "product", background: "wood", tone: "natural", composition: "full-product" },
  blanket: {
    subject: "product",
    background: "location",
    tone: "warm",
    composition: "full-product",
  },
  lamp: { subject: "product", background: "wood", tone: "dark", composition: "full-product" },
  lunchbox: {
    subject: "product",
    background: "location",
    tone: "natural",
    composition: "full-product",
  },
  headphones: {
    subject: "product",
    background: "textured",
    tone: "dark",
    composition: "full-product",
  },
  sneakers: { subject: "product", background: "wood", tone: "warm", composition: "full-product" },
  sweatshirt: {
    subject: "product",
    background: "textured",
    tone: "warm",
    composition: "full-product",
  },
  duffel: {
    subject: "product",
    background: "textured",
    tone: "natural",
    composition: "full-product",
  },
  sunglasses: {
    subject: "product",
    background: "textured",
    tone: "warm",
    composition: "full-product",
  },
  organizer: {
    subject: "product",
    background: "plain",
    tone: "light",
    composition: "full-product",
  },
  towel: { subject: "product", background: "wood", tone: "light", composition: "full-product" },
};

const additionalImages: [string, ProductImage[]][] = ADDITIONAL_PRODUCTS.map((product) => [
  product.id,
  [
    {
      id: `${product.id}-studio`,
      src: `/products/${product.id}-gallery.jpg`,
      label: "Studio",
      alt: `${product.name} on a plain ivory studio background`,
      context: "studio",
      position: "0% center",
      tags: ["product-only", "studio", ...product.searchTerms],
      presentation: {
        subject: "product",
        background: "plain",
        tone: "light",
        composition: "full-product",
      },
    },
    {
      id: `${product.id}-still-life`,
      src: `/products/${product.id}-gallery.jpg`,
      label: "Still life",
      alt: product.stillLife,
      context: "studio",
      position: "50% center",
      tags: ["product-only", "still-life", ...product.searchTerms],
      presentation: stillLifePresentations[product.id],
    },
    {
      id: `${product.id}-in-use`,
      src: `/products/${product.id}-gallery.jpg`,
      label: "In use",
      alt: product.lifestyle,
      context: product.context,
      position: "100% center",
      tags: ["in-use", product.scenario, ...product.searchTerms],
      presentation: {
        subject: "in-use",
        background: "location",
        tone: "natural",
        composition: "lifestyle",
      },
    },
  ],
]);

const tabletopImages: Partial<Record<Product["id"], ProductImage>> = Object.fromEntries(
  [
    ["mug", "Trail mug", "tabletop", "oatmeal linen and pale oak in soft window light"],
    [
      "pack",
      "Everyday daypack",
      "tabletop-v3",
      "a wooden entryway bench beside the front door",
    ],
    [
      "notebook",
      "Pocket field notes",
      "tabletop-v2",
      "a walnut writing desk in warm afternoon light",
    ],
  ].map(([id, name, file, setting]) => [
    id,
    {
      id: `${id}-tabletop`,
      src: `/products/${id}-${file}.jpg`,
      label: "Still life",
      alt: `${name} resting on ${setting}`,
      context: "studio",
      tags: [
        "product-only",
        "tabletop",
        ...(id === "mug"
          ? ["linen", "warm"]
          : id === "pack"
            ? ["entryway", "bench", "wood"]
            : ["walnut", "warm"]),
      ],
      presentation: {
        subject: "product",
        background: id === "notebook" ? "wood" : "location",
        tone: id === "pack" ? "natural" : "warm",
        composition: "full-product",
      },
      position: "0% center",
      tile: { columns: 3, rows: 1, index: 0, sheetAspect: 3 },
    },
  ]),
);

export const PRODUCT_IMAGES = Object.fromEntries(
  [...Object.entries(originalImages), ...additionalImages].map(([id, images]) => [
    id,
    tabletopImages[id as Product["id"]]
      ? [...images, tabletopImages[id as Product["id"]]!]
      : images,
  ]),
) as Record<Product["id"], ProductImage[]>;

export function selectProductImage(productId: Product["id"], context: ImageContext): ProductImage {
  const images = PRODUCT_IMAGES[productId];
  const fallback: ImageContext =
    context === "city" || context === "campus"
      ? "everyday"
      : context === "family"
        ? "outdoors"
        : "studio";
  return (
    images.find((image) => image.context === context) ??
    images.find((image) => image.context === fallback) ??
    images[0]
  );
}
