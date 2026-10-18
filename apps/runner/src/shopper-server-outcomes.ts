/** Post-run verification of local demo receipts. Never creates orders or dispatches agents. */
import { createHash } from "node:crypto";
import { lstat, readFile, realpath, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { SHOPPER_ALLOWED_ORIGINS, type ShopperArm } from "./shopper-contract.js";

type Obj = Record<string, unknown>;
type Artifact = { path: string; sha256: string; bytes: number };
export type ServerOutcomeRow = { runId: string; caseId: string; arm: ShopperArm; serverConfirmed: boolean; visibleCheckoutConfirmed: boolean;
  receipt: Obj | null; artifact: Artifact | null; readback: { httpStatus: 200; identical: true } | null; errors: string[] };
export type ServerOutcomesIndex = { schema: "shopper-server-outcomes.v1"; synthetic: true; humanConversionClaim: false;
  reports: { path: string; sha256: string }[]; rows: ServerOutcomeRow[] };
export type ServerOutcomesOptions = { fetch?: typeof fetch; timeoutMs?: number };
const HASH = /^[a-f0-9]{64}$/;
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
const LIMIT = 16 * 1024 * 1024;
const sha = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
function requireValue(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }
function object(value: unknown, label: string): Obj {
  requireValue(value && typeof value === "object" && !Array.isArray(value), `${label}: object required`); return value as Obj;
}
function exact(value: unknown, keys: string[], label: string): Obj {
  const result = object(value, label);
  requireValue(Object.keys(result).length === keys.length && keys.every((key) => Object.hasOwn(result, key)), `${label}: unexpected or missing fields`); return result;
}
function text(value: unknown, label: string): string { requireValue(typeof value === "string" && value.length > 0 && value.length <= 4000, `${label}: bounded text required`); return value; }
function array(value: unknown, label: string): unknown[] { requireValue(Array.isArray(value), `${label}: array required`); return value; }
function cents(value: unknown, label: string): number {
  requireValue(typeof value === "number" && Number.isFinite(value) && value >= 0, `${label}: nonnegative money required`);
  const amount = Math.round(value * 100);
  requireValue(Number.isSafeInteger(amount) && Math.abs(value * 100 - amount) < 1e-7, `${label}: whole cents required`); return amount;
}
async function jsonFile(file: string) {
  requireValue((await stat(file)).size <= LIMIT, "JSON input exceeds 16 MiB");
  const bytes = await readFile(file); return { bytes, value: JSON.parse(bytes.toString("utf8")) as unknown };
}
async function containedFile(root: string, relative: string) {
  requireValue(!path.isAbsolute(relative), "Evidence path must be relative");
  const target = path.resolve(root, relative);
  requireValue(target.startsWith(path.resolve(root) + path.sep), "Evidence path escapes root");
  const [actualRoot, actualTarget] = await Promise.all([realpath(root), realpath(target)]);
  requireValue(actualTarget.startsWith(actualRoot + path.sep), "Evidence symlink escapes root"); return actualTarget;
}
function endpoint(value: unknown, origin: string, pathname: string): string {
  const url = new URL(text(value, "checkout URL"));
  requireValue(url.origin === origin && url.pathname === pathname && !url.username && !url.password && !url.search && !url.hash,
    "Checkout URL must match assigned storefront and exact demo endpoint"); return url.href;
}
function verifyReceipt(value: unknown, choice: Obj): Obj {
  const receipt = exact(value, ["schema", "checkoutId", "receiptId", "createdAt", "items", "subtotal", "shipping", "total", "realPurchase"], "receipt");
  requireValue(receipt.schema === "fieldwork.demo-checkout-receipt.v1" && receipt.realPurchase === false, "Only synthetic demo receipts accepted");
  requireValue(UUID.test(text(receipt.checkoutId, "checkoutId")) && /^demo-[a-f0-9]{32}$/.test(text(receipt.receiptId, "receiptId")), "Invalid receipt identity");
  const timestamp = text(receipt.createdAt, "createdAt");
  requireValue(/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z$/.test(timestamp) && Number.isFinite(Date.parse(timestamp)), "Invalid UTC receipt timestamp");
  const items = array(receipt.items, "receipt items"); requireValue(items.length === 1, "Exactly one purchased item required");
  const item = exact(items[0], ["productId", "variantId", "quantity", "unitPrice", "lineTotal"], "receipt item");
  requireValue(choice.observationSource === "independent-dom" && choice.quantity === 1 && item.quantity === 1
    && item.productId === choice.productId && item.variantId === choice.variantId, "Receipt identity/quantity differs from catalog-bound DOM choice");
  const unit = cents(item.unitPrice, "unitPrice"), subtotal = cents(receipt.subtotal, "subtotal");
  requireValue(unit === cents(choice.catalogPrice, "catalog price") && cents(item.lineTotal, "lineTotal") === unit && subtotal === unit,
    "Receipt price/subtotal differs from catalog-bound choice");
  const shipping = exact(receipt.shipping, ["status", "cost"], "shipping");
  let cost = 0;
  if (shipping.status === "hidden") requireValue(shipping.cost === null, "Hidden shipping must remain unknown/null");
  else {
    const free = subtotal > 5000;
    requireValue(shipping.status === (free ? "free" : "flat"), "Shipping policy differs from authoritative threshold");
    cost = cents(shipping.cost, "shipping cost"); requireValue(cost === (free ? 0 : 500), "Shipping cost differs from authoritative policy");
  }
  requireValue(cents(receipt.total, "total") === subtotal + cost, "Receipt total differs from authoritative subtotal and shipping"); return receipt;
}

async function verifyRow(input: Obj, root: string, catalogSha: unknown, options: ServerOutcomesOptions): Promise<ServerOutcomeRow> {
  const row: ServerOutcomeRow = { runId: text(input.expectedRunId, "expectedRunId"), caseId: text(input.caseId, "caseId"), arm: input.arm as ShopperArm,
    serverConfirmed: false, visibleCheckoutConfirmed: false, receipt: null, artifact: null, readback: null, errors: [] };
  requireValue(["generic", "local-model"].includes(row.arm), "Unknown assigned arm");
  requireValue(["recorded", "unattempted", "invalid-sidecar"].includes(String(input.status)), "Unknown assignment status");
  if (input.status !== "recorded") { if (input.status === "invalid-sidecar") row.errors.push("Assigned session sidecar is invalid"); return row; }
  try {
    const sidecarPath = await containedFile(root, text(input.path, "session path")), sidecar = await jsonFile(sidecarPath);
    requireValue(typeof input.sidecarSha256 === "string" && HASH.test(input.sidecarSha256) && sha(sidecar.bytes) === input.sidecarSha256, "Session sidecar SHA-256 mismatch");
    requireValue(isDeepStrictEqual(sidecar.value, input.session), "Report embedded session differs from hashed sidecar");
    const session = object(sidecar.value, "session"), run = object(session.run, "session run");
    requireValue(session.caseId === row.caseId && session.arm === row.arm && run.id === row.runId && session.realPurchase === false, "Session identity or synthetic scope mismatch");
    const verification = object(session.verification, "session verification");
    requireValue(typeof catalogSha === "string" && HASH.test(catalogSha) && verification.catalogSha256 === catalogSha
      && verification.independentOfShopperAgent === true, "Independent aggregate/catalog binding required");
    const matches = array(session.artifacts, "session artifacts").map((item) => object(item, "artifact evidence"))
      .filter((item) => object(item.artifact, "artifact").path === "checkout-receipt.json");
    requireValue(matches.length === 1, "Exactly one checkout-receipt.json contract artifact required");
    const captured = matches[0], artifact = object(captured.artifact, "artifact");
    requireValue(captured.status === "captured" && captured.error === null && artifact.kind === "result" && artifact.runId === row.runId
      && typeof captured.sha256 === "string" && HASH.test(captured.sha256), "Captured checkout result artifact required");
    const artifactPath = await containedFile(path.dirname(sidecarPath), "checkout-receipt.json"), data = await jsonFile(artifactPath);
    requireValue(data.bytes.length === captured.bytes && sha(data.bytes) === captured.sha256, "Checkout artifact bytes or SHA-256 mismatch");
    row.artifact = { path: artifactPath, sha256: captured.sha256, bytes: data.bytes.length };
    const evidence = exact(data.value, ["schema", "synthetic", "post", "get", "confirmation", "errors"], "checkout evidence");
    requireValue(evidence.schema === "shopper-demo-checkout-evidence.v1" && evidence.synthetic === true, "Versioned synthetic checkout evidence required");
    const errors = array(evidence.errors, "checkout errors"); requireValue(errors.every((item) => typeof item === "string"), "Checkout evidence errors must be strings");
    row.errors.push(...errors as string[]);
    if (evidence.post === null && evidence.get === null) {
      const confirmation = exact(evidence.confirmation, ["visible", "checkoutId", "receiptId"], "confirmation");
      requireValue(confirmation.visible === false && confirmation.checkoutId === null && confirmation.receiptId === null,
        "No-order evidence cannot claim a visible receipt");
      return row;
    }
    const choice = object(session.choice, "catalog-bound choice");
    const post = exact(evidence.post, ["url", "httpStatus", "request", "body"], "POST evidence"), get = exact(evidence.get, ["url", "httpStatus", "body"], "GET evidence");
    requireValue(post.httpStatus === 202 && get.httpStatus === 200, "Successful recorded POST202/GET200 required");
    const postBody = exact(post.body, ["receipt", "duplicate"], "POST body"), getBody = exact(get.body, ["receipt"], "GET body");
    requireValue(typeof postBody.duplicate === "boolean", "POST duplicate flag required");
    const receipt = verifyReceipt(postBody.receipt, choice);
    requireValue(isDeepStrictEqual(receipt, getBody.receipt), "Recorded POST/GET receipts differ");
    const request = exact(post.request, ["schema", "checkoutId", "items", "shippingVisible"], "checkout request");
    requireValue(request.schema === "fieldwork.demo-checkout.v1" && request.checkoutId === receipt.checkoutId && typeof request.shippingVisible === "boolean", "Checkout request schema/identity invalid");
    const requestItems = array(request.items, "request items"), receiptItem = object((receipt.items as unknown[])[0], "receipt item");
    requireValue(requestItems.length === 1, "Exactly one request item required");
    const requestItem = exact(requestItems[0], ["productId", "variantId", "quantity"], "request item");
    requireValue(isDeepStrictEqual(requestItem, { productId: receiptItem.productId, variantId: receiptItem.variantId, quantity: receiptItem.quantity }), "Request/receipt items differ");
    requireValue(request.shippingVisible === (object(receipt.shipping, "shipping").status !== "hidden"), "Request/receipt shipping visibility differs");
    let visibleCheckoutConfirmed = false;
    try {
      const confirmation = exact(evidence.confirmation, ["visible", "checkoutId", "receiptId"], "confirmation");
      requireValue(typeof confirmation.visible === "boolean", "Visible confirmation state must be boolean");
      const matchingIds = confirmation.checkoutId === receipt.checkoutId && confirmation.receiptId === receipt.receiptId;
      if (confirmation.visible) requireValue(matchingIds, "Visible confirmation IDs differ from receipt");
      visibleCheckoutConfirmed = confirmation.visible && matchingIds && errors.length === 0;
    } catch (error) { row.errors.push(error instanceof Error ? error.message : String(error)); }
    const entry = new URL(text(session.entryUrl, "entryUrl"));
    requireValue(SHOPPER_ALLOWED_ORIGINS.some((origin) => origin === entry.origin) && !entry.username && !entry.password, "Assigned storefront origin is not allowlisted");
    endpoint(post.url, entry.origin, "/api/demo-checkout");
    const readbackUrl = endpoint(get.url, entry.origin, `/api/demo-checkout/${receipt.checkoutId}`);
    const response = await (options.fetch ?? fetch)(readbackUrl, { method: "GET", redirect: "error", cache: "no-store", signal: AbortSignal.timeout(options.timeoutMs ?? 10000) });
    requireValue(response.status === 200, `Durable GET readback failed: HTTP ${response.status}`);
    const body = await response.text(); requireValue(Buffer.byteLength(body) <= LIMIT, "Durable receipt response exceeds size limit");
    const readback = exact(JSON.parse(body), ["receipt"], "durable GET body");
    requireValue(isDeepStrictEqual(readback.receipt, receipt), "Durable GET receipt differs from captured receipt");
    row.serverConfirmed = true; row.visibleCheckoutConfirmed = visibleCheckoutConfirmed; row.receipt = receipt; row.readback = { httpStatus: 200, identical: true };
  } catch (error) { row.errors.push(error instanceof Error ? error.message : String(error)); }
  return row;
}

/** Report paths must be at their session root, with row.path relative to that root. */
export async function generateServerOutcomes(reportPaths: string[], outputFile: string, options: ServerOutcomesOptions = {}): Promise<ServerOutcomesIndex> {
  requireValue(reportPaths.length > 0, "At least one report required");
  const paths = reportPaths.map((file) => path.resolve(file)), output = path.resolve(outputFile);
  requireValue(new Set(paths).size === paths.length && !paths.includes(output), "Duplicate report path or source overwrite");
  requireValue(options.timeoutMs === undefined || Number.isInteger(options.timeoutMs) && options.timeoutMs >= 1 && options.timeoutMs <= 60000, "Invalid GET timeout");
  try { await lstat(output); throw new Error("Output already exists (EEXIST)"); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
  const result: ServerOutcomesIndex = { schema: "shopper-server-outcomes.v1", synthetic: true, humanConversionClaim: false, reports: [], rows: [] };
  const assignments: { row: Obj; root: string; catalogSha: unknown }[] = [];
  for (const file of paths) {
    const input = await jsonFile(file), report = object(input.value, "aggregate");
    requireValue(report.version === "shopper-evidence-aggregate.v1" && report.synthetic === true && report.humanConversionClaim === false, "Versioned synthetic aggregate required");
    const rows = array(report.rows, "aggregate rows"); requireValue(rows.length === report.fixedSessions && rows.length > 0, "All assigned aggregate rows required");
    result.reports.push({ path: file, sha256: sha(input.bytes) });
    for (const value of rows) assignments.push({ row: object(value, "assignment"), root: path.dirname(file), catalogSha: report.catalogSnapshotSha256 });
  }
  const runIds = assignments.map(({ row }) => text(row.expectedRunId, "expectedRunId"));
  requireValue(new Set(runIds).size === runIds.length, "Duplicate assigned run ID across reports");
  for (const assigned of assignments) result.rows.push(await verifyRow(assigned.row, assigned.root, assigned.catalogSha, options));
  await writeFile(output, `${JSON.stringify(result, null, 2)}\n`, { flag: "wx" }); return result;
}

export async function runServerOutcomesCli(args: string[], options: ServerOutcomesOptions = {}) {
  const usage = "Usage: shopper-server-outcomes.ts --report AGGREGATE.json [--report OTHER.json ...] --out NEW.json";
  const reports: string[] = []; let output: string | undefined;
  for (let i = 0; i < args.length; i += 2) {
    requireValue(["--report", "--out"].includes(args[i]) && !!args[i + 1], usage);
    if (args[i] === "--report") reports.push(args[i + 1]);
    else { requireValue(output === undefined, usage); output = args[i + 1]; }
  }
  requireValue(output && reports.length, usage); return generateServerOutcomes(reports, output, options);
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runServerOutcomesCli(process.argv.slice(2)).then((index) => console.log(JSON.stringify({ rows: index.rows.length,
    serverConfirmed: index.rows.filter((row) => row.serverConfirmed).length }))).catch((error) => { console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1; });
}
