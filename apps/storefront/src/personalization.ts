import { orderedProducts, type Product, type VariantConfig } from "./config";
import type { ImageContext } from "./images";
import { ADDITIONAL_PRODUCTS } from "./catalog-expansion";

const productTerms: Record<string, readonly string[]> = {
  ...Object.fromEntries(ADDITIONAL_PRODUCTS.map((product) => [product.id, product.searchTerms])),
  mug: ["mug", "mugs", "cup", "cups", "coffee", "tea", "drink", "drinks"],
  pack: ["bag", "bags", "backpack", "backpacks", "daypack", "pack", "commute", "commuting"],
  notebook: [
    "notebook",
    "notebooks",
    "journal",
    "journals",
    "journaling",
    "writing",
    "write",
    "notes",
    "sketch",
    "sketching",
  ],
  jacket: [
    "jacket",
    "jackets",
    "rain",
    "rainy",
    "raincoat",
    "waterproof",
    "shell",
    "rainwear",
    "coat",
  ],
};

const contexts: {
  id: ImageContext;
  terms: string[];
  headline: string;
  collection: string;
  /** Shown next to the search status so a viewer can see why the order changed. */
  note: string;
  boosts: Partial<Record<Product["id"], number>>;
}[] = [
  {
    id: "campus",
    terms: [
      "campus",
      "college",
      "university",
      "student",
      "students",
      "school",
      "study",
      "studying",
      "class",
      "classes",
    ],
    headline: "Ready for the next chapter",
    collection: "For life on campus",
    note: "campus days",
    boosts: { pack: 3, notebook: 2, jacket: 1 },
  },
  {
    id: "family",
    terms: [
      "family",
      "parent",
      "parents",
      "parenting",
      "kids",
      "children",
      "playground",
      "park",
      "soccer",
    ],
    headline: "For the days you spend together.",
    collection: "For family days out",
    note: "family days out",
    boosts: { pack: 3, jacket: 2 },
  },
  {
    id: "city",
    terms: ["commute", "commuting", "city", "urban", "work", "office", "train", "bus"],
    headline: "Made for your daily rhythm.",
    collection: "For the daily commute",
    note: "the daily commute",
    boosts: { pack: 3, jacket: 2, notebook: 1 },
  },
  {
    id: "outdoors",
    terms: [
      "hike",
      "hikes",
      "hiking",
      "trail",
      "trails",
      "camp",
      "camping",
      "outdoor",
      "outdoors",
      "nature",
      "mountain",
      "mountains",
      "rain",
      "rainy",
    ],
    headline: "Take the long way out.",
    collection: "For a little more outside",
    note: "time outside",
    boosts: { pack: 3, jacket: 2 },
  },
  {
    id: "everyday",
    terms: ["home", "desk", "breakfast", "kitchen"],
    headline: "A little better, every day.",
    collection: "For your everyday",
    note: "everyday routines",
    boosts: { mug: 2, notebook: 1 },
  },
];

// Local demo policy. A future ranker can select from the same catalog and image IDs.
export function personalizeStorefront(query: string, config: VariantConfig) {
  const words: string[] = query.toLowerCase().match(/[a-z]+/g) ?? [];
  const count = (terms: readonly string[]) => terms.filter((term) => words.includes(term)).length;
  const interest = contexts
    .map((context) => ({ context, score: count(context.terms) }))
    .sort((a, b) => b.score - a.score)
    .find(({ score }) => score > 0)?.context;
  const ranked = orderedProducts(config.recommendationOrder)
    .map((product) => ({
      product,
      score: count(productTerms[product.id]) * 10 + (interest?.boosts[product.id] ?? 0),
    }))
    .sort((a, b) => b.score - a.score);
  const matched = ranked.some(({ score }) => score > 0);
  const headlines: Partial<Record<Product["id"], string>> = {
    mug: "Your next coffee, wherever you are.",
    pack: "A bag for wherever the day takes you.",
    notebook: "Room for your next idea.",
    jacket: "A good day. Whatever the forecast.",
  };
  return {
    products: ranked.map(({ product }) => product),
    context: interest?.id ?? ("studio" as ImageContext),
    headline: matched
      ? (interest?.headline ?? headlines[ranked[0].product.id] ?? config.headline)
      : config.headline,
    collection: matched ? (interest?.collection ?? "Picked for your search") : "Collection",
    reason: matched ? (interest?.note ?? null) : null,
    matched,
  };
}
