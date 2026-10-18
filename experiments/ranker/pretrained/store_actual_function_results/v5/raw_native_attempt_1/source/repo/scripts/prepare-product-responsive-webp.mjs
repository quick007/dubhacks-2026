/** Lower-resolution alternatives only. Original and full-size v1 bytes are immutable. */
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { readFile, writeFile, mkdir, access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { hash, imageProperties, verifyManifest } from "./prepare-product-webp.mjs";

const execute = promisify(execFile);
const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const originalManifest = "apps/storefront/src/responsive-product-assets.json";
const outputManifest = "apps/storefront/src/responsive-product-sizes.json";
export const WIDTHS = Object.freeze([192, 384, 768, 1152, 1536]);

export function checkResponsiveGeometry(original, derivative, expectedWidth) {
  if (expectedWidth >= original.width || derivative.width !== expectedWidth ||
      Math.abs(derivative.height - original.height * derivative.width / original.width) > 1 ||
      derivative.iccSha256 !== original.iccSha256) throw new Error("Responsive dimensions/aspect/ICC invariant failed");
}

async function absent(file) {
  try { await access(file); throw new Error(`Existing artifact preserved: ${file}`); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
}

export async function verifyResponsiveProducts(root = projectRoot) {
  await verifyManifest(root);
  const v1Bytes = await readFile(path.join(root, originalManifest));
  const v1 = JSON.parse(v1Bytes), current = JSON.parse(await readFile(path.join(root, outputManifest), "utf8"));
  if (hash(v1Bytes) !== current.fullSizeManifestSha256 ||
      JSON.stringify(Object.keys(v1.assets).sort()) !== JSON.stringify(Object.keys(current.assets).sort())) throw new Error("Full-size v1 manifest/source membership changed");
  const publicRoot = path.join(root, "apps/storefront/public");
  let variantCount = 0, variantBytes = 0;
  for (const [src, item] of Object.entries(current.assets)) {
    const original = await readFile(path.join(publicRoot, src));
    const properties = imageProperties(original), full = v1.assets[src];
    if (hash(original) !== item.sourceSha256 || item.sourceSha256 !== full.sourceSha256 ||
        JSON.stringify(item.full) !== JSON.stringify(full) || item.sourceBytes !== original.length) throw new Error(`Original/full-size changed: ${src}`);
    const expected = WIDTHS.filter((width) => width < full.width);
    if (JSON.stringify(item.variants.map((variant) => variant.width)) !== JSON.stringify(expected)) throw new Error(`Responsive roster changed: ${src}`);
    for (const variant of item.variants) {
      const bytes = await readFile(path.join(publicRoot, variant.src));
      if (hash(bytes) !== variant.sha256 || bytes.length !== variant.bytes) throw new Error(`Responsive bytes changed: ${variant.src}`);
      checkResponsiveGeometry(properties, imageProperties(bytes), variant.width);
      variantCount++; variantBytes += bytes.length;
    }
  }
  if (variantCount !== current.summary.variantCount || variantBytes !== current.summary.variantBytes) throw new Error("Responsive aggregate changed");
  return { status: "verified", ...current.summary };
}

export async function prepareResponsiveProducts(root = projectRoot, encoder = "/opt/homebrew/bin/cwebp") {
  await absent(path.join(root, outputManifest));
  await verifyManifest(root);
  const bytes = await readFile(path.join(root, originalManifest)), manifest = JSON.parse(bytes);
  const version = (await execute(encoder, ["-version"])).stdout.trim();
  const publicRoot = path.join(root, "apps/storefront/public"), assets = {};
  const totals = Object.fromEntries(WIDTHS.map((width) => [width, { count: 0, bytes: 0 }]));
  let completed = 0;
  const total = Object.values(manifest.assets).reduce((n, full) => n + WIDTHS.filter((w) => w < full.width).length, 0);
  for (const [src, full] of Object.entries(manifest.assets)) {
    const sourcePath = path.join(publicRoot, src), source = await readFile(sourcePath), original = imageProperties(source);
    if (hash(source) !== full.sourceSha256) throw new Error(`Original changed before encoding: ${src}`);
    const variants = [];
    for (const width of WIDTHS.filter((width) => width < original.width)) {
      const relative = src.slice("/products/".length);
      const optimizedSrc = `/products/optimized/responsive/${relative}.w${width}.webp`;
      const target = path.join(publicRoot, optimizedSrc);
      await absent(target); await mkdir(path.dirname(target), { recursive: true });
      const started = performance.now();
      // Whole-image resample from the untouched original, never a tile crop or
      // recompression of the full-size lossy WebP. No -mt or concurrent jobs.
      await execute(encoder, ["-q", "82", "-m", "4", "-metadata", "icc", "-resize", String(width), "0", "-quiet", sourcePath, "-o", target]);
      const output = await readFile(target), properties = imageProperties(output);
      checkResponsiveGeometry(original, properties, width);
      const variant = { src: optimizedSrc, width: properties.width, height: properties.height,
        bytes: output.length, sha256: hash(output), iccSha256: properties.iccSha256, encodeMs: performance.now() - started };
      variants.push(variant); totals[width].count++; totals[width].bytes += output.length;
      process.stdout.write(JSON.stringify({ completed: ++completed, total, src, width, bytes: output.length }) + "\n");
    }
    if (hash(await readFile(sourcePath)) !== full.sourceSha256) throw new Error(`Original changed during encoding: ${src}`);
    assets[src] = { sourceSha256: full.sourceSha256, sourceBytes: full.sourceBytes, full, variants };
  }
  const result = { version: "responsive-product-webp.v1", fullSizeManifestSha256: hash(bytes),
    encoder: { version, quality: 82, method: 4, metadata: "icc", encoderJobs: 1, multithreadFlag: false,
      resize: "requested width with automatic proportional height; no upscaling", crop: false },
    widths: WIDTHS, summary: { sourceCount: Object.keys(assets).length, sourceBytes: manifest.summary.sourceBytes,
      fullSizeWebpBytes: manifest.summary.optimizedBytes, variantCount: completed,
      variantBytes: Object.values(totals).reduce((n, row) => n + row.bytes, 0), perWidth: totals },
    scope: "Every width describes the entire original sheet; gallery/crop identity remains unchanged; full-size v1 detail alternatives retained byte-identically",
    assets };
  await writeFile(path.join(root, outputManifest), JSON.stringify(result, null, 2) + "\n", { flag: "wx" });
  await verifyResponsiveProducts(root);
  return result;
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  const args = process.argv.slice(2);
  if (args.length !== 1 || !["--prepare", "--check"].includes(args[0])) throw new Error("Use --prepare once or --check; existing artifacts are preserved");
  const result = args[0] === "--check" ? await verifyResponsiveProducts() : await prepareResponsiveProducts();
  process.stdout.write(JSON.stringify(result.summary ?? result) + "\n");
}
