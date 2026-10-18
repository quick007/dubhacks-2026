import projection from "./product-image-runtime.json";

const assets: Readonly<Record<string, readonly number[]>> = projection.assets;
export type ResponsiveProductImage = Readonly<{ src: string; srcSet?: string; sizes?: string }>;

function geometry(source: string): readonly number[] | undefined {
  if (typeof source !== "string" || !source.startsWith("/products/")) return undefined;
  const basename = source.slice("/products/".length);
  return Object.hasOwn(assets, basename) ? assets[basename] : undefined;
}

function number(value: number): string {
  return String(Number(value.toFixed(6)));
}

/** Equivalent gallery sizing, using only compact public source geometry. */
export function responsiveProductImage(originalSrc: string, sheetScale?: number, thumbnail = false): ResponsiveProductImage {
  const dimensions = geometry(originalSrc);
  if (!dimensions) return { src: originalSrc };
  const [width, height] = dimensions;
  const scale = sheetScale ?? Math.max(1, width / height);
  if (!Number.isFinite(scale) || scale <= 0) throw new RangeError("Positive finite sheet scale required");
  const basename = originalSrc.slice("/products/".length);
  const widths = [...projection.widths.filter((candidate) => candidate < width), width];
  const url = (candidate: number) => candidate === width ? `/products/optimized/${basename}.webp` :
    `/products/optimized/responsive/${basename}.w${candidate}.webp`;
  const preferred = (thumbnail ? 50 : 384) * scale;
  const selected = widths.find((candidate) => candidate >= preferred) ?? width;
  const sizes = thumbnail ? `${number(50 * scale)}px` :
    `(max-width: 700px) calc(${number(100 * scale)}vw - ${number(40 * scale)}px), ` +
    `(max-width: 1099px) calc(${number(50 * scale)}vw - ${number(54 * scale)}px), ` +
    `calc(${number(100 * scale / 3)}vw - ${number(136 * scale / 3)}px)`;
  return { src: url(selected), srcSet: widths.map((candidate) => `${url(candidate)} ${candidate}w`).join(", "), sizes };
}

export function fullSizeProductImageSource(originalSrc: string): string {
  return geometry(originalSrc) ? `/products/optimized/${originalSrc.slice("/products/".length)}.webp` : originalSrc;
}
