import { PRODUCTS } from "./config";
import { PRODUCT_VARIANTS } from "./variants";
import type { TraceContext } from "./session-trace-bridge";
import type { ShoppingTelemetryConfig } from "./shopping-telemetry-config";
export type ShoppingContext = TraceContext & { modelVersion: string | null };
export type ProductExposure = { productId: string; variantId: string; position: number };
export type ShoppingData = {
  "session-start": { consent: "notice" | "granted"; replay: boolean };
  "catalog-view": { viewId: string; products: ProductExposure[] };
  "product-view": ProductExposure & { viewId: string; visibility: "intersection-at-least-half"; visibleMs: 250 };
  search: { source: "submit" | "example" | "clear" };
  selection: { viewId: string; productId: string; variantId: string };
  "cart-add": { viewId: string; productId: string; variantId: string; count: number };
  "cart-clear": { count: 0 };
  "checkout-success": { count: number; demo: true };
  "model-readiness": { stage: "rank" | "firstThree" | "complete"; durationMs: number; scope: "after-paint-opportunity"; copySource: "fresh-native" | "fresh-parallel-head" | "generated-earlier" | "mixed-native-cache" | "no-generation" | null };
  "session-end": { reason: "pagehide" | "duration-limit" };
};
export type ShoppingAction = keyof ShoppingData;
export type ShoppingEvent = { [K in ShoppingAction]: { eventId: string; sequence: number; at: number; elapsedMs: number; action: K; context: ShoppingContext; data: ShoppingData[K] } }[ShoppingAction];
export type SanitizedReplayEvent = { type: number; timestamp: number; data: Record<string, unknown> };
export type ShoppingBatch = {
  schema: "fieldwork.shopping-batch.v1"; batchId: string; sessionId: string;
  classification: ShoppingTelemetryConfig["classification"]; startedAt: number; events: ShoppingEvent[];
  replay?: { schema: "fieldwork.sanitized-replay.v1"; events: SanitizedReplayEvent[] };
};
const ids = new Set<string>(PRODUCTS.map((product) => product.id));
export function validExposure(product: ProductExposure): boolean {
  return ids.has(product.productId) && Number.isInteger(product.position) && product.position >= 0 && product.position < 16
    && !!PRODUCT_VARIANTS[product.productId as keyof typeof PRODUCT_VARIANTS]?.some((variant) => variant.id === product.variantId);
}
export const SHOPPING_LIMITS = { queueBytes: 1024 * 1024, queueItems: 1000, batchBytes: 384 * 1024, replayBytes: 320 * 1024,
  batchEvents: 100, batchReplay: 100, retries: 3, sessionMs: 8 * 60 * 60_000, replayMs: 5 * 60_000, replayEvents: 2000, replayTotalBytes: 4 * 1024 * 1024 } as const;
