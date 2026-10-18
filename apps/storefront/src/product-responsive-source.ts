import manifest from "./responsive-product-sizes.json";

type Variant = Readonly<{ src: string; width: number; height: number }>;
type Asset = Readonly<{ full: Variant; variants: readonly Variant[] }>;
const assets: Readonly<Record<string, Asset>> = manifest.assets;

export type ResponsiveProductImage = Readonly<{ src: string; srcSet?: string; sizes?: string }>;

function number(value: number): string {
  return String(Number(value.toFixed(6)));
}

/** Whole-sheet rendered width, including cover crops inside square cards. */
function displaySizes(scale: number, thumbnail: boolean): string {
  if (thumbnail) return `${number(50 * scale)}px`;
  return `(max-width: 700px) calc(${number(100 * scale)}vw - ${number(40 * scale)}px), ` +
    `(max-width: 1099px) calc(${number(50 * scale)}vw - ${number(54 * scale)}px), ` +
    `calc(${number(100 * scale / 3)}vw - ${number(136 * scale / 3)}px)`;
}

/**
 * Every descriptor is the ENTIRE sheet width. The implicit cover-crop scale is
 * derived from the original aspect. Explicit tile layout may supply its scale.
 * Source/crop identity and original fallback URLs remain the caller's metadata.
 */
export function responsiveProductImage(originalSrc: string, sheetScale?: number, thumbnail = false): ResponsiveProductImage {
  if (!Object.hasOwn(assets, originalSrc)) return { src: originalSrc };
  const asset = assets[originalSrc];
  const scale = sheetScale ?? Math.max(1, asset.full.width / asset.full.height);
  if (!Number.isFinite(scale) || scale <= 0) throw new RangeError("Positive finite sheet scale required");
  const variants = [...asset.variants, asset.full];
  const preferredWidth = (thumbnail ? 50 : 384) * scale;
  const fallback = variants.find((variant) => variant.width >= preferredWidth) ?? asset.full;
  return {
    src: fallback.src,
    srcSet: variants.map((variant) => `${variant.src} ${variant.width}w`).join(", "),
    sizes: displaySizes(scale, thumbnail),
  };
}

/** Largest retained same-dimension WebP for an optional detail presentation. */
export function fullSizeProductImageSource(originalSrc: string): string {
  return Object.hasOwn(assets, originalSrc) ? assets[originalSrc].full.src : originalSrc;
}
