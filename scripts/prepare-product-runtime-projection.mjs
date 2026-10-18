/** Compact runtime geometry only; full audit manifests stay separate and unchanged. */
import { readFile, writeFile, access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { verifyResponsiveProducts } from "./prepare-product-responsive-webp.mjs";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sourceRelative = "apps/storefront/src/responsive-product-sizes.json";
const targetRelative = "apps/storefront/src/product-image-runtime.json";

export function runtimeProjection(manifest) {
  if (manifest.version !== "responsive-product-webp.v1" || !Array.isArray(manifest.widths) ||
      manifest.widths.some((width, index) => !Number.isInteger(width) || width <= 0 || index > 0 && width <= manifest.widths[index - 1])) throw new Error("Reviewed sorted responsive width contract required");
  const assets = {};
  for (const [source, asset] of Object.entries(manifest.assets)) {
    const basename = source.slice("/products/".length);
    if (!source.startsWith("/products/") || !/^[a-z0-9][a-z0-9._-]*\.(?:jpg|jpeg|png)$/.test(basename) ||
        source !== `/products/${basename}` || !Number.isInteger(asset.full.width) || !Number.isInteger(asset.full.height) || asset.full.width <= 0 || asset.full.height <= 0 ||
        asset.full.src !== `/products/optimized/${basename}.webp`) throw new Error(`Unsafe/noncanonical full-size mapping: ${source}`);
    const expected = manifest.widths.filter((width) => width < asset.full.width);
    if (asset.variants.length !== expected.length) throw new Error(`Noncanonical variant count: ${source}`);
    asset.variants.forEach((variant, index) => {
      if (variant.width !== expected[index] || variant.src !== `/products/optimized/responsive/${basename}.w${variant.width}.webp` ||
          !Number.isInteger(variant.height) || variant.height <= 0 ||
          Math.abs(variant.height - asset.full.height * variant.width / asset.full.width) > 1) throw new Error(`Unsafe URL/geometry variant: ${source}`);
    });
    assets[basename] = [asset.full.width, asset.full.height];
  }
  // Basename is the public source key. The already-verified fixed local naming
  // rule replaces repeated URLs; full geometry supplies cover-crop scaling.
  return { v: 1, widths: manifest.widths, assets };
}

export async function prepareRuntimeProjection(root = projectRoot, check = false) {
  await verifyResponsiveProducts(root);
  const manifest = JSON.parse(await readFile(path.join(root, sourceRelative), "utf8"));
  const projected = runtimeProjection(manifest);
  const serialized = JSON.stringify(projected) + "\n";
  const destination = path.join(root, targetRelative);
  if (check) {
    if (await readFile(destination, "utf8") !== serialized) throw new Error("Runtime projection differs from verified full manifest");
  } else {
    try { await access(destination); throw new Error("Existing projection preserved; use --check"); }
    catch (error) { if (error.code !== "ENOENT") throw error; }
    await writeFile(destination, serialized, { flag: "wx" });
  }
  return { status: check ? "verified" : "prepared", assets: Object.keys(projected.assets).length,
    auditManifestBytes: Buffer.byteLength(await readFile(path.join(root, sourceRelative), "utf8")),
    runtimeProjectionBytes: Buffer.byteLength(serialized),
    sourceOrDerivativeFilesChanged: false, fullAuditsRemainSeparate: true };
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  const args = process.argv.slice(2);
  if (args.length !== 1 || !["--prepare", "--check"].includes(args[0])) throw new Error("Use --prepare once or --check");
  process.stdout.write(JSON.stringify(await prepareRuntimeProjection(projectRoot, args[0] === "--check")) + "\n");
}
