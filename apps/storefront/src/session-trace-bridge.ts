/** No replay imports, observers, storage or timers on the default store path. */
export const demoTraceEnabled = typeof window !== "undefined"
  && new URLSearchParams(window.location.search).get("trace") === "demo";
export type TraceAction = "search" | "selection" | "cart-add" | "cart-clear" | "checkout-success";
export type TraceFields = { productId?: string; variantId?: string; count?: number; source?: "submit" | "example" | "clear" };
export type TraceContext = { mode: "baseline" | "local-model"; order: "popular" | "budget" | "history"; shipping: boolean };
type Sink = { event: (action: TraceAction, fields?: TraceFields) => void; products: (ids: string[]) => void; context: (value: TraceContext) => void };
let sink: Sink | undefined;
let products: string[] = [];
let context: TraceContext = { mode: "baseline", order: "popular", shipping: false };
export function traceEvent(action: TraceAction, fields?: TraceFields) { sink?.event(action, fields); }
export function traceProducts(ids: string[]) { if (demoTraceEnabled) { products = ids; sink?.products(ids); } }
export function traceContext(value: TraceContext) { if (demoTraceEnabled) { context = value; sink?.context(value); } }
export function connectTrace(value: Sink) {
  sink = value; value.context(context); value.products(products);
  return () => { if (sink === value) sink = undefined; };
}
