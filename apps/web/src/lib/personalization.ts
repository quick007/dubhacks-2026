/** Authored presentation stages. Purchase history and room styling are illustrative demo context. */
export const personalizationSteps = [
  {
    image: "original",
    title: "The original",
    detail: "Same collection for everyone.",
    alt: "The original store collection, starting with a mug, daypack, and notebook.",
  },
  {
    image: "request",
    title: "His request",
    detail: "The sofa throw comes first.",
    alt: "The woven throw moves to the first product position.",
  },
  {
    image: "details",
    title: "His preferences",
    detail: "Woven cotton for a drafty room.",
    alt: "The throw card adds cotton and warmth details below the product photo.",
  },
  {
    image: "room",
    title: "His new couch",
    detail: "Show the throw with the couch he just bought.",
    alt: "The product photo shows the rust throw on a sand-colored sofa.",
  },
  {
    image: "match",
    title: "The color match",
    detail: "Rust goes with his new sand sofa.",
    alt: "The card explains that the rust color pairs with Bob's recently purchased sand sofa.",
  },
] as const;

export const PERSONALIZATION_STEP_HEIGHT = 2;
