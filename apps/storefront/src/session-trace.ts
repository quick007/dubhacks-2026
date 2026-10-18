import { PRODUCTS } from "./config";
import { PRODUCT_VARIANTS } from "./variants";
import { PRODUCT_IMAGES } from "./images";
import type { TraceAction, TraceContext, TraceFields } from "./session-trace-bridge";

export const TRACE_LIMITS = { durationMs: 5 * 60_000, retentionMs: 15 * 60_000, bytes: 4 * 1024 * 1024, replayEvents: 2000, shoppingEvents: 500 } as const;
export const TRACE_BLOCK_SELECTOR = "input,textarea,select,[contenteditable],iframe,video,audio,.cart-page,.confirmation,.confirmation-page,[data-trace-private],.trace-controls,#trace-replay,[autocomplete], [id*=checkout i],[class*=checkout i],[id*=account i],[class*=account i],[id*=payment i],[class*=payment i]";
export const TRACE_RECORD_OPTIONS = {
  maskAllInputs: true, maskInputFn: () => "[masked]", maskTextSelector: "body,title",
  maskTextFn: (text: string) => text.replace(/\S/g, "*"), blockSelector: TRACE_BLOCK_SELECTOR,
  recordCanvas: false, recordCrossOriginIframes: false, inlineImages: false, collectFonts: false,
  sampling: { mousemove: false, scroll: 200, input: "last" as const, media: 500 },
  plugins: [], slimDOMOptions: { script: true, comment: true, headMetaDescKeywords: true, headMetaSocial: true, headMetaRobots: true, headMetaHttpEquiv: true, headMetaAuthorship: true, headMetaVerification: true },
};
type ReplayEvent = { type: number; timestamp: number; data: Record<string, unknown> };
type ShoppingEvent = { at: number; elapsedMs: number; action: TraceAction | "product-view" | "model-readiness"; data: Record<string, unknown>; context: TraceContext & { modelVersion: null } };
export type DemoSessionTrace = {
  schema: "fieldwork.demo-session.v1"; sessionId: string; startedAt: number; stoppedAt: number | null;
  stopReason: string | null; expiresAt: number; synthetic: true;
  privacy: { text: "masked"; inputs: "blocked"; network: false; persistence: "memory-only"; upload: false };
  events: ShoppingEvent[]; replay: { package: "@rrweb/record"; version: "2.1.7"; events: ReplayEvent[] };
};
const ids = new Set<string>(PRODUCTS.map((product) => product.id));
const variants = new Map<string, Set<string>>(Object.entries(PRODUCT_VARIANTS).map(([id, items]) => [id, new Set(items.map((item) => item.id))]));
const actions = new Set(["search", "selection", "cart-add", "cart-clear", "checkout-success"]);
const mask = (text: string) => text.replace(/\S/g, "*");
const replayAssets = new Set<string>([
  ...Object.values(PRODUCT_IMAGES).flat().map((image) => image.src),
  ...Object.values(PRODUCT_VARIANTS).flat().flatMap((variant) => variant.image ? [variant.image.src] : []),
]);
/** Replay references the known original fixture, independent of delivery derivatives. */
export function canonicalReplayAsset(value: string) {
  try {
    const url = new URL(value, "http://local.invalid");
    if (url.search || url.hash) return "";
    if (replayAssets.has(url.pathname)) return url.pathname;
    const derivative = /^\/products\/optimized\/(?:responsive\/)?([a-z0-9-]+\.(?:jpg|png))(?:\.w(?:192|384|768|1152|1536))?\.webp$/.exec(url.pathname);
    const original = derivative ? `/products/${derivative[1]}` : "";
    return replayAssets.has(original) ? original : "";
  } catch { return ""; }
}
/** Defense in depth: strip all arbitrary attributes/URLs, including URL hashes,
 * document titles, aria text, data-* mirrors, and any query reflected into DOM.
 * Replay retains layout/images; readable shopping identity lives in typed events. */
function attributes(value: unknown) {
  const result: Record<string, string | number> = {};
  if (!value || typeof value !== "object") return result;
  for (const [key, item] of Object.entries(value)) {
    if (typeof item !== "string" && typeof item !== "number") continue;
    if (["class", "width", "height", "rr_width", "rr_height", "rr_scrollLeft", "rr_scrollTop"].includes(key)) result[key] = item;
    if (key === "style") result[key] = String(item).replace(/url\([^)]*\)/gi, "none");
    if (key === "src") { const src = canonicalReplayAsset(String(item)); if (src) result[key] = src; }
    if (key === "_cssText") result[key] = String(item).replace(/url\([^)]*\)/gi, "none");
  }
  return result;
}
function node(value: unknown, parentTag = ""): unknown {
  if (!value || typeof value !== "object") return null;
  const source = value as Record<string, unknown>;
  const tag = typeof source.tagName === "string" ? source.tagName : "";
  const result: Record<string, unknown> = {};
  for (const key of ["type", "id", "tagName", "isSVG", "isShadow", "isShadowHost", "compatMode"]) if (source[key] !== undefined) result[key] = source[key];
  if (source.attributes) result.attributes = attributes(source.attributes);
  if (typeof source.textContent === "string") result.textContent = parentTag === "style" ? source.textContent.replace(/url\([^)]*\)/gi, "none") : mask(source.textContent);
  if (Array.isArray(source.childNodes)) result.childNodes = source.childNodes.map((child) => node(child, tag));
  return result;
}
export function sanitizeReplay(value: unknown): ReplayEvent | null {
  if (!value || typeof value !== "object") return null;
  const event = value as ReplayEvent;
  if (!Number.isFinite(event.timestamp) || !event.data || typeof event.data !== "object") return null;
  if (event.type === 4) return { type: 4, timestamp: event.timestamp, data: { href: "http://localhost/", width: event.data.width, height: event.data.height } };
  if (event.type === 2) return { type: 2, timestamp: event.timestamp, data: { node: node(event.data.node), initialOffset: event.data.initialOffset } };
  if (event.type === 0 || event.type === 1) return { type: event.type, timestamp: event.timestamp, data: {} };
  if (event.type !== 3) return null; // No custom/plugin/network/console events.
  const data = event.data;
  if (data.source === 0) return { type: 3, timestamp: event.timestamp, data: { source: 0,
    texts: Array.isArray(data.texts) ? data.texts.map((item) => ({ id: item.id, value: mask(String(item.value)) })) : [],
    attributes: Array.isArray(data.attributes) ? data.attributes.map((item) => ({ id: item.id, attributes: attributes(item.attributes) })) : [],
    removes: data.removes,
    adds: Array.isArray(data.adds) ? data.adds.map((item) => ({ parentId: item.parentId, nextId: item.nextId, node: node(item.node) })) : [] } };
  if (![2, 3, 4].includes(Number(data.source))) return null; // input, media, font, canvas, logs excluded
  const safe: Record<string, unknown> = {};
  for (const key of ["source", "type", "id", "x", "y", "width", "height"]) if (typeof data[key] === "number") safe[key] = data[key];
  return { type: 3, timestamp: event.timestamp, data: safe };
}
export function createTraceBuffer(now: () => number = Date.now, uuid: () => string = () => crypto.randomUUID()) {
  const startedAt = now();
  const trace: DemoSessionTrace = { schema: "fieldwork.demo-session.v1", sessionId: uuid(), startedAt, stoppedAt: null, stopReason: null,
    expiresAt: startedAt + TRACE_LIMITS.retentionMs, synthetic: true,
    privacy: { text: "masked", inputs: "blocked", network: false, persistence: "memory-only", upload: false },
    events: [], replay: { package: "@rrweb/record", version: "2.1.7", events: [] } };
  let bytes = 0;
  let context: TraceContext = { mode: "baseline", order: "popular", shipping: false };
  const seenProducts = new Set<string>();
  function stop(reason = "manual") { if (trace.stoppedAt === null) { trace.stoppedAt = now(); trace.stopReason = reason; } }
  function live() { if (now() - startedAt >= TRACE_LIMITS.durationMs) stop("duration-limit"); return trace.stoppedAt === null; }
  function append(items: unknown[], value: unknown, max: number) {
    if (!live()) return false;
    const size = new TextEncoder().encode(JSON.stringify(value)).length;
    if (items.length >= max || bytes + size > TRACE_LIMITS.bytes) { stop("buffer-limit"); return false; }
    items.push(value); bytes += size; return true;
  }
  function event(action: ShoppingEvent["action"], data: Record<string, unknown>) {
    return append(trace.events, { at: now(), elapsedMs: now() - startedAt, action, data, context: { ...context, modelVersion: null } }, TRACE_LIMITS.shoppingEvents);
  }
  return {
    stop, live, bytes: () => bytes,
    context(value: TraceContext) { context = { mode: value.mode === "local-model" ? "local-model" : "baseline", order: ["popular", "budget", "history"].includes(value.order) ? value.order : "popular", shipping: value.shipping === true }; },
    event(action: TraceAction, fields: TraceFields = {}) {
      if (!actions.has(action)) return;
      const data: Record<string, unknown> = {};
      if (action === "search") { if (["submit", "example", "clear"].includes(fields.source ?? "")) data.source = fields.source; }
      if (action === "selection" || action === "cart-add") {
        if (!fields.productId || !ids.has(fields.productId) || !variants.get(fields.productId)?.has(fields.variantId ?? "")) return;
        data.productId = fields.productId; data.variantId = fields.variantId;
      }
      if (["cart-add", "cart-clear", "checkout-success"].includes(action) && Number.isInteger(fields.count) && fields.count! >= 0 && fields.count! <= 1000) data.count = fields.count;
      event(action, data);
    },
    product(id: string) { if (ids.has(id) && !seenProducts.has(id) && live()) { seenProducts.add(id); event("product-view", { productId: id, visibility: "intersection-at-least-half" }); } },
    readiness(entry: { name: string; duration: number; detail?: unknown }) {
      const stage = /^fieldwork\.local-model\.(rank|firstThree|complete)\.after-paint-opportunity$/.exec(entry.name)?.[1];
      if (!stage || !Number.isFinite(entry.duration)) return;
      const detail = entry.detail as { copySource?: { kind?: string } } | undefined;
      const kind = detail?.copySource?.kind;
      event("model-readiness", { stage, durationMs: entry.duration, scope: "after-paint-opportunity", copySource: ["fresh-native", "fresh-parallel-head", "generated-earlier", "mixed-native-cache", "no-generation"].includes(kind ?? "") ? kind : null });
    },
    replay(value: unknown) { const safe = sanitizeReplay(value); if (safe) append(trace.replay.events, safe, TRACE_LIMITS.replayEvents); },
    export(): DemoSessionTrace | null { return now() >= trace.expiresAt ? null : structuredClone(trace); },
  };
}
