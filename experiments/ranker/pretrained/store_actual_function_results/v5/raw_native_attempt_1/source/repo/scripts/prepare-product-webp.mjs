/** Same-dimension WebP alternatives. Originals are read-only; one encoder at a time. */
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { readFile, writeFile, mkdir, readdir, access } from "node:fs/promises";
import { createHash } from "node:crypto";
import { inflateSync } from "node:zlib";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const execute = promisify(execFile);
const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const defaultEncoder = "/opt/homebrew/bin/cwebp";
const imageExtensions = new Set([".jpg", ".jpeg", ".png"]);
const manifestRelative = "apps/storefront/src/responsive-product-assets.json";
export const ENCODING = Object.freeze({ quality: 82, method: 4, metadata: "icc", resize: false, crop: false, encoderJobs: 1, multithreadFlag: false });
export const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");

function orientation(tiff) {
  if (tiff.length < 8) throw new Error("Truncated EXIF TIFF");
  const little = tiff.toString("ascii", 0, 2) === "II";
  if (!little && tiff.toString("ascii", 0, 2) !== "MM") throw new Error("Unknown EXIF byte order");
  const u16 = (offset) => little ? tiff.readUInt16LE(offset) : tiff.readUInt16BE(offset);
  const u32 = (offset) => little ? tiff.readUInt32LE(offset) : tiff.readUInt32BE(offset);
  if (u16(2) !== 42) throw new Error("Invalid EXIF TIFF");
  const start = u32(4), count = u16(start);
  for (let i = 0; i < count; i++) {
    const offset = start + 2 + i * 12;
    if (u16(offset) === 0x112) {
      if (u16(offset + 2) !== 3 || u32(offset + 4) !== 1) throw new Error("Unsupported orientation field");
      return u16(offset + 8);
    }
  }
  return null;
}

/** Decode only container dimensions/ICC metadata; never alters image pixels. */
export function imageProperties(bytes) {
  let width, height, format, icc = null, exifOrientation = null;
  if (bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))) {
    format = "png";
    for (let offset = 8; offset + 12 <= bytes.length;) {
      const length = bytes.readUInt32BE(offset), name = bytes.toString("ascii", offset + 4, offset + 8);
      if (offset + 12 + length > bytes.length) throw new Error("Truncated PNG chunk");
      const data = bytes.subarray(offset + 8, offset + 8 + length);
      if (name === "IHDR") { width = data.readUInt32BE(0); height = data.readUInt32BE(4); }
      if (name === "iCCP") {
        const end = data.indexOf(0);
        if (end < 0 || data[end + 1] !== 0) throw new Error("Unsupported PNG ICC compression");
        icc = inflateSync(data.subarray(end + 2));
      }
      if (name === "eXIf") exifOrientation = orientation(data);
      offset += length + 12;
    }
  } else if (bytes[0] === 0xff && bytes[1] === 0xd8) {
    format = "jpeg"; const profiles = new Map(); let expectedParts = null;
    for (let offset = 2; offset < bytes.length;) {
      if (bytes[offset++] !== 0xff) throw new Error("Malformed JPEG marker");
      while (bytes[offset] === 0xff) offset++;
      const marker = bytes[offset++];
      if (marker === 0xda || marker === 0xd9) break;
      if (marker === 0x01 || marker >= 0xd0 && marker <= 0xd7) continue;
      const length = bytes.readUInt16BE(offset);
      if (length < 2 || offset + length > bytes.length) throw new Error("Truncated JPEG segment");
      const data = bytes.subarray(offset + 2, offset + length);
      if ([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf].includes(marker)) {
        height = data.readUInt16BE(1); width = data.readUInt16BE(3);
      }
      if (marker === 0xe2 && data.subarray(0, 12).equals(Buffer.from("ICC_PROFILE\0", "ascii"))) {
        if (expectedParts !== null && expectedParts !== data[13] || profiles.has(data[12])) throw new Error("Conflicting JPEG ICC chunks");
        expectedParts = data[13]; profiles.set(data[12], data.subarray(14));
      }
      if (marker === 0xe1 && data.subarray(0, 6).equals(Buffer.from("Exif\0\0", "ascii"))) exifOrientation = orientation(data.subarray(6));
      offset += length;
    }
    if (expectedParts !== null) {
      if (!expectedParts || profiles.size !== expectedParts || [...profiles.keys()].some((key) => key < 1 || key > expectedParts)) throw new Error("Incomplete JPEG ICC profile");
      icc = Buffer.concat(Array.from({ length: expectedParts }, (_, i) => profiles.get(i + 1)));
    }
  } else if (bytes.toString("ascii", 0, 4) === "RIFF" && bytes.toString("ascii", 8, 12) === "WEBP") {
    format = "webp";
    if (bytes.readUInt32LE(4) + 8 !== bytes.length) throw new Error("WebP RIFF length mismatch");
    for (let offset = 12; offset + 8 <= bytes.length;) {
      const name = bytes.toString("ascii", offset, offset + 4), length = bytes.readUInt32LE(offset + 4);
      if (offset + 8 + length > bytes.length) throw new Error("Truncated WebP chunk");
      const data = bytes.subarray(offset + 8, offset + 8 + length);
      if (name === "VP8X") { width = 1 + data.readUIntLE(4, 3); height = 1 + data.readUIntLE(7, 3); }
      if (name === "VP8 " && width === undefined) {
        if (!data.subarray(3, 6).equals(Buffer.from([0x9d, 1, 0x2a]))) throw new Error("Invalid WebP VP8 frame");
        width = data.readUInt16LE(6) & 0x3fff; height = data.readUInt16LE(8) & 0x3fff;
      }
      if (name === "VP8L" && width === undefined) {
        if (data[0] !== 0x2f) throw new Error("Invalid lossless WebP frame");
        const bits = data.readUInt32LE(1); width = 1 + (bits & 0x3fff); height = 1 + (bits >>> 14 & 0x3fff);
      }
      if (name === "ICCP") icc = data;
      if (name === "ANIM") throw new Error("Animated assets require a separate reviewed path");
      offset += 8 + length + (length & 1);
    }
  } else throw new Error("Unsupported image container");
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) throw new Error("Missing valid raster dimensions");
  return { format, width, height, iccSha256: icc === null ? null : hash(icc), iccBytes: icc?.length ?? 0, exifOrientation };
}

async function originals(directory) {
  const result = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (entry.name === "optimized") continue;
    const full = path.join(directory, entry.name);
    if (entry.isDirectory()) result.push(...await originals(full));
    else if (entry.isFile() && imageExtensions.has(path.extname(entry.name).toLowerCase())) result.push(full);
  }
  return result.sort();
}

export async function verifyManifest(root = projectRoot) {
  const manifest = JSON.parse(await readFile(path.join(root, manifestRelative), "utf8"));
  const directory = path.join(root, "apps/storefront/public");
  const found = await originals(path.join(directory, "products"));
  const urls = found.map((file) => "/" + path.relative(directory, file).split(path.sep).join("/")).sort();
  if (found.length !== manifest.summary.sourceCount || JSON.stringify(Object.keys(manifest.assets).sort()) !== JSON.stringify(urls)) throw new Error("Original asset membership changed");
  let totalOriginal = 0, totalOptimized = 0;
  for (const [source, item] of Object.entries(manifest.assets)) {
    const sourceBytes = await readFile(path.join(directory, source));
    const destination = await readFile(path.join(directory, item.src));
    const before = imageProperties(sourceBytes), after = imageProperties(destination);
    if (hash(sourceBytes) !== item.sourceSha256 || hash(destination) !== item.optimizedSha256 ||
        sourceBytes.length !== item.sourceBytes || destination.length !== item.optimizedBytes ||
        before.width !== after.width || before.height !== after.height || before.width !== item.width || before.height !== item.height ||
        before.iccSha256 !== after.iccSha256 || before.iccSha256 !== item.iccSha256) throw new Error(`Asset invariant changed: ${source}`);
    totalOriginal += sourceBytes.length; totalOptimized += destination.length;
  }
  if (totalOriginal !== manifest.summary.sourceBytes || totalOptimized !== manifest.summary.optimizedBytes) throw new Error("Aggregate byte counts changed");
  return { status: "verified", ...manifest.summary };
}

export async function prepareProductWebp(root = projectRoot, encoder = defaultEncoder) {
  const destinationManifest = path.join(root, manifestRelative);
  try { await access(destinationManifest); throw new Error("Manifest already exists; use --check without replacing artifacts"); }
  catch (error) { if (error.code !== "ENOENT") throw error; }
  const version = (await execute(encoder, ["-version"])).stdout.trim();
  const publicRoot = path.join(root, "apps/storefront/public"), productRoot = path.join(publicRoot, "products");
  const paths = await originals(productRoot), sourceInventory = [];
  for (const sourcePath of paths) {
    const bytes = await readFile(sourcePath), properties = imageProperties(bytes);
    if (properties.exifOrientation !== null && properties.exifOrientation !== 1) throw new Error(`Rotated EXIF needs separate review: ${sourcePath}`);
    sourceInventory.push({ sourcePath, sourceBytes: bytes.length, sourceSha256: hash(bytes), properties });
  }
  const assets = {};
  for (const source of sourceInventory) {
    const relative = path.relative(productRoot, source.sourcePath).split(path.sep).join("/");
    const src = `/products/optimized/${relative}.webp`, destination = path.join(publicRoot, src);
    try { await access(destination); throw new Error(`Existing derivative preserved: ${destination}`); }
    catch (error) { if (error.code !== "ENOENT") throw error; }
    await mkdir(path.dirname(destination), { recursive: true });
    const started = performance.now();
    // No -mt, resize, crop, quality search or original-file write is permitted.
    await execute(encoder, ["-q", "82", "-m", "4", "-metadata", "icc", "-quiet", source.sourcePath, "-o", destination]);
    const encoded = await readFile(destination), after = imageProperties(encoded), before = source.properties;
    const sourceNow = await readFile(source.sourcePath);
    if (hash(sourceNow) !== source.sourceSha256 || after.width !== before.width || after.height !== before.height || after.iccSha256 !== before.iccSha256) throw new Error(`Encoding changed source/dimension/ICC invariant: ${relative}`);
    const sourceUrl = `/products/${relative}`;
    assets[sourceUrl] = { src, width: before.width, height: before.height, sourceBytes: source.sourceBytes,
      optimizedBytes: encoded.length, sourceSha256: source.sourceSha256, optimizedSha256: hash(encoded),
      iccSha256: before.iccSha256, iccBytes: before.iccBytes, sourceExifOrientation: before.exifOrientation,
      encodeMs: performance.now() - started };
    process.stdout.write(JSON.stringify({ completed: Object.keys(assets).length, total: paths.length, source: sourceUrl, sourceBytes: source.sourceBytes, optimizedBytes: encoded.length }) + "\n");
  }
  const sourceBytes = sourceInventory.reduce((sum, value) => sum + value.sourceBytes, 0);
  const optimizedBytes = Object.values(assets).reduce((sum, value) => sum + value.optimizedBytes, 0);
  const manifest = { version: "same-dimension-product-webp.v1", encoder: { version, ...ENCODING },
    scope: "Lossy q82 alternative encoding; exact raster dimensions/aspect and ICC retained; all source files byte-identical; no rescale, crop or generated imagery",
    summary: { sourceCount: paths.length, sourceBytes, optimizedBytes, savedBytes: sourceBytes - optimizedBytes,
      optimizedToOriginalRatio: optimizedBytes / sourceBytes, bytesSavedPercent: 100 * (1 - optimizedBytes / sourceBytes),
      nonSmallerDerivativeCount: Object.values(assets).filter((item) => item.optimizedBytes >= item.sourceBytes).length }, assets };
  await writeFile(destinationManifest, JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
  await verifyManifest(root);
  return manifest;
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  const args = process.argv.slice(2);
  if (args.length !== 1 || !["--prepare", "--check"].includes(args[0])) throw new Error("Use --prepare once or --check to verify retained originals/derivatives");
  const result = args[0] === "--check" ? await verifyManifest() : await prepareProductWebp();
  process.stdout.write(JSON.stringify(result.summary ?? result) + "\n");
}
