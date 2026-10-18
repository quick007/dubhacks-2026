import type { Product } from "./config";

export const HERO_BANNERS: Partial<Record<Product["id"], readonly [string, string]>> = {
  pack: [
    "The sand canvas daypack on an entryway bench in warm daylight",
    "A commuter wearing the sand daypack on a leafy city sidewalk",
  ],
  jacket: [
    "A woman wearing the forest rain jacket in a rainy university courtyard",
    "The forest rain jacket draped over a chair on a covered cabin porch",
  ],
  mug: [
    "The sage Trail Mug on linen at a sunlit breakfast table",
    "The sage Trail Mug on a picnic table overlooking a lake",
  ],
  notebook: [
    "The terracotta notebook on a walnut writing desk",
    "A person writing on the cream pages of the notebook at a cafe",
  ],
  sunglasses: [
    "Round tortoiseshell sunglasses on a sunlit stone terrace",
    "A woman wearing round tortoiseshell sunglasses at an outdoor cafe",
  ],
  lunchbox: [
    "The open steel lunch box and bamboo lid on natural linen",
    "The steel lunch box with its bamboo lid and olive band on a picnic table",
  ],
};
