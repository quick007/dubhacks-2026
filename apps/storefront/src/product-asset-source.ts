import manifest from "./responsive-product-assets.json";

export type OptimizedProductAsset = Readonly<{
  src: string;
  width: number;
  height: number;
  sourceBytes: number;
  optimizedBytes: number;
  sourceSha256: string;
  optimizedSha256: string;
  iccSha256: string | null;
}>;

const assets: Readonly<Record<string, OptimizedProductAsset>> = manifest.assets;

/** Exact source lookup; no crop, variant, dimensions or remote URL rewriting. */
export function optimizedProductAsset(originalSrc: string): OptimizedProductAsset | null {
  if (!Object.hasOwn(assets, originalSrc)) return null;
  const candidate = assets[originalSrc];
  return candidate.optimizedBytes < candidate.sourceBytes ? candidate : null;
}

/** Unwired format alternative. Unknown/non-smaller sources retain their URL. */
export function resolveProductImageSource(originalSrc: string): string {
  return optimizedProductAsset(originalSrc)?.src ?? originalSrc;
}
