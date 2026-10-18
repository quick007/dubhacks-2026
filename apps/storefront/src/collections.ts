import type { Product } from "./config";

export const COLLECTIONS: { id: string; label: string; productIds: readonly Product["id"][] }[] = [
  { id: "all", label: "All products", productIds: [] },
  {
    id: "bags",
    label: "Bags & travel",
    productIds: ["pack", "tote", "duffel", "organizer", "cubes"],
  },
  { id: "clothing", label: "Clothing", productIds: ["jacket", "sneakers", "sweatshirt", "cap"] },
  {
    id: "home",
    label: "Home & desk",
    productIds: ["notebook", "lamp", "blanket", "stand", "planter"],
  },
  { id: "food", label: "Food & drink", productIds: ["mug", "bottle", "lunchbox"] },
  {
    id: "accessories",
    label: "Accessories",
    productIds: ["sunglasses", "towel", "headphones", "umbrella", "speaker"],
  },
];
