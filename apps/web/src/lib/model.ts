import run from "../fieldwork/model-run.json";

/**
 * Measured results of the frozen actual-store ranking run, copied from the experiment by
 * scripts/capture-fieldwork.mjs. Every number on the model slides comes from here.
 */
export const modelRun = run;
export type ModelRequest = (typeof run.requests)[number];

const products: Record<string, { name: string; price: number; defaultRank: number }> = run.products;

export function product(id: string) {
  return products[id];
}
/** The product's photo, captured from its card on the real shop. */
export function photo(id: string) {
  return `/fieldwork/products/${id}.jpg`;
}
export function modelRequest(personaId: string) {
  return run.requests.find((request) => request.personaId === personaId);
}
/** 1st, 2nd, 3rd, 4th, with 11th to 13th as exceptions. */
export function ordinal(rank: number) {
  const teen = rank % 100 >= 11 && rank % 100 <= 13;
  return `${rank}${teen ? "th" : (["th", "st", "nd", "rd"][rank % 10] ?? "th")}`;
}
export function ms(value: number) {
  return `${Math.round(value)} ms`;
}
