import { connectTrace, type TraceAction, type TraceFields, type TraceContext } from "./session-trace-bridge";
import { PRODUCT_VARIANTS } from "./variants";
import { sanitizeReplay, TRACE_RECORD_OPTIONS } from "./session-trace";
import { SHOPPING_LIMITS, validExposure, type ProductExposure, type ShoppingContext, type ShoppingAction, type ShoppingData, type ShoppingEvent } from "./shopping-events";
import { createShoppingTransport } from "./shopping-transport";
import type { ShoppingTelemetryConfig } from "./shopping-telemetry-config";

/** One page session, created only after deployment+privacy gates. No stable user ID. */
export function startShoppingTelemetry(config: ShoppingTelemetryConfig, consent: "notice" | "granted") {
  const startedAt = Date.now(), startedClock = performance.now(), sessionId = crypto.randomUUID();
  const transport = createShoppingTransport(config, sessionId, startedAt);
  let sequence = 0, stopped = false, sessionStarted = false, contextDirty = false, viewId = crypto.randomUUID(), currentProducts: string[] = [];
  let context: ShoppingContext = { mode: "baseline", order: "popular", shipping: false, modelVersion: "deterministic-v1" };
  let stopReplay: (() => void) | undefined, replayBytes = 0, replayCount = 0;
  let observer: IntersectionObserver | undefined, performanceObserver: PerformanceObserver | undefined;
  const visibilityTimers = new Map<Element, ReturnType<typeof setTimeout>>();
  const viewSeen = new Set<string>();
  function emit<K extends ShoppingAction>(action: K, data: ShoppingData[K]) {
    if (stopped || sequence >= 1_000_000) return;
    const elapsedMs = Math.min(SHOPPING_LIMITS.sessionMs, Math.max(0, Math.floor(performance.now() - startedClock)));
    transport.event({ eventId: crypto.randomUUID(), sequence: sequence++, at: startedAt + elapsedMs, elapsedMs, action, context: { ...context }, data } as ShoppingEvent);
  }
  function exposures(ids: string[]): { element: Element; product: ProductExposure }[] {
    return [...document.querySelectorAll("article.product")].flatMap((element, position) => {
      const productId = ids[position];
      const variants = PRODUCT_VARIANTS[productId as keyof typeof PRODUCT_VARIANTS];
      const chosen = [...element.querySelectorAll(".product-colors button")].findIndex((button) => button.getAttribute("aria-pressed") === "true");
      const product = { productId, variantId: variants?.[Math.max(0, chosen)]?.id ?? "", position };
      return validExposure(product) ? [{ element, product }] : [];
    });
  }
  function clearVisibility() { observer?.disconnect(); for (const timer of visibilityTimers.values()) clearTimeout(timer); visibilityTimers.clear(); }
  function observeProducts(ids: string[], force = false) {
    if (stopped) return;
    const rows = exposures(ids);
    const fingerprint = rows.map(({ product }) => `${product.productId}:${product.variantId}`).join(",");
    if (!force && fingerprint === currentProducts.join(",")) return;
    currentProducts = fingerprint ? fingerprint.split(",") : [];
    clearVisibility(); viewSeen.clear(); viewId = crypto.randomUUID();
    if (!rows.length) return;
    emit("catalog-view", { viewId, products: rows.map(({ product }) => product) });
    if (!globalThis.IntersectionObserver) return;
    const products = new Map(rows.map((row) => [row.element, row.product]));
    const thisView = viewId;
    observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const existing = visibilityTimers.get(entry.target); if (existing) clearTimeout(existing); visibilityTimers.delete(entry.target);
        const product = products.get(entry.target);
        if (!product || !entry.isIntersecting || entry.intersectionRatio < 0.5 || document.visibilityState !== "visible" || viewSeen.has(product.productId)) continue;
        const timer = setTimeout(() => {
          visibilityTimers.delete(entry.target);
          if (stopped || thisView !== viewId || document.visibilityState !== "visible") return;
          viewSeen.add(product.productId);
          emit("product-view", { ...product, viewId, visibility: "intersection-at-least-half", visibleMs: 250 });
        }, 250);
        visibilityTimers.set(entry.target, timer);
      }
    }, { threshold: 0.5 });
    rows.forEach(({ element }) => observer!.observe(element));
  }
  let latestIds: string[] = [];
  function event(action: TraceAction, fields: TraceFields = {}) {
    const identity = { productId: fields.productId ?? "", variantId: fields.variantId ?? "", position: 0 };
    if (action === "search" && ["submit", "example", "clear"].includes(fields.source ?? "")) {
      emit("search", { source: fields.source! });
      requestAnimationFrame(() => requestAnimationFrame(() => { if (!stopped) observeProducts(latestIds, true); }));
    }
    if (action === "selection" && validExposure(identity)) {
      emit("selection", { viewId, productId: identity.productId, variantId: identity.variantId });
      // React updates the pressed variant after the synchronous action callback.
      queueMicrotask(() => { if (!stopped) requestAnimationFrame(() => observeProducts(latestIds)); });
    }
    if (action === "cart-add" && validExposure(identity) && Number.isInteger(fields.count) && fields.count! > 0 && fields.count! <= 1000) emit("cart-add", { viewId, productId: identity.productId, variantId: identity.variantId, count: fields.count! });
    if (action === "cart-clear") emit("cart-clear", { count: 0 });
    if (action === "checkout-success" && Number.isInteger(fields.count) && fields.count! > 0 && fields.count! <= 1000) emit("checkout-success", { count: fields.count!, demo: true });
  }
  const disconnect = connectTrace({ event, products: (ids) => { latestIds = ids; observeProducts(ids, contextDirty); contextDirty = false; }, context(value: TraceContext) {
    const previous = JSON.stringify(context);
    context = { mode: value.mode === "local-model" ? "local-model" : "baseline", order: ["popular", "budget", "history"].includes(value.order) ? value.order : "popular",
      shipping: value.shipping === true, modelVersion: value.mode === "local-model" ? config.modelVersion : "deterministic-v1" };
    contextDirty = previous !== JSON.stringify(context);
    if (!sessionStarted) { sessionStarted = true; emit("session-start", { consent, replay: true }); }
  } });
  // No user-entered query/request IDs/raw native detail are copied from measures.
  if (globalThis.PerformanceObserver) {
    performanceObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const stage = /^fieldwork\.local-model\.(rank|firstThree|complete)\.after-paint-opportunity$/.exec(entry.name)?.[1] as "rank" | "firstThree" | "complete" | undefined;
        const kind = (entry as PerformanceMeasure).detail?.copySource?.kind;
        if (stage && entry.duration >= 0 && entry.duration <= SHOPPING_LIMITS.sessionMs) emit("model-readiness", { stage, durationMs: entry.duration, scope: "after-paint-opportunity",
          copySource: ["fresh-native", "fresh-parallel-head", "generated-earlier", "mixed-native-cache", "no-generation"].includes(kind) ? kind : null });
      }
    });
    performanceObserver.observe({ type: "measure" });
  }
  void import("@rrweb/record").then(({ record }) => {
    if (stopped) return;
    stopReplay = record({ ...TRACE_RECORD_OPTIONS, emit(value) {
      if (stopped || replayCount >= SHOPPING_LIMITS.replayEvents) { queueMicrotask(() => stopReplay?.()); return; }
      const safe = sanitizeReplay(value); if (!safe) return;
      const bytes = new TextEncoder().encode(JSON.stringify(safe)).length;
      if (replayBytes + bytes > SHOPPING_LIMITS.replayTotalBytes || !transport.replay(safe)) { queueMicrotask(() => stopReplay?.()); return; }
      replayCount++; replayBytes += bytes;
    } });
  }).catch(() => { /* Shopping actions still work when optional replay cannot load. */ });
  const replayTimer = setTimeout(() => stopReplay?.(), SHOPPING_LIMITS.replayMs);
  const sessionTimer = setTimeout(() => { emit("session-end", { reason: "duration-limit" }); stop(false); }, SHOPPING_LIMITS.sessionMs);
  function hidden() { if (document.visibilityState === "hidden") { clearVisibility(); void transport.flush(true); } else observeProducts(latestIds, true); }
  function pagehide() { emit("session-end", { reason: "pagehide" }); stop(false, true); }
  document.addEventListener("visibilitychange", hidden); window.addEventListener("pagehide", pagehide);
  function stop(discard = true, leavingPage = false) {
    if (stopped) return; stopped = true; disconnect(); clearVisibility(); performanceObserver?.disconnect(); stopReplay?.();
    clearTimeout(replayTimer); clearTimeout(sessionTimer); document.removeEventListener("visibilitychange", hidden); window.removeEventListener("pagehide", pagehide);
    if (discard) transport.stop(); else void transport.flush(leavingPage);
  }
  return { stop, flush: () => transport.flush(), stats: () => ({ sessionId, classification: config.classification, ...transport.stats(), replayEvents: replayCount }) };
}
