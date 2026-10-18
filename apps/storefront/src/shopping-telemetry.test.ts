import assert from "node:assert/strict";
import { test } from "node:test";
import { shoppingTelemetryConfig, privacyChoice, type ShoppingTelemetryConfig } from "./shopping-telemetry-config";
import { createShoppingTransport } from "./shopping-transport";
import { SHOPPING_LIMITS, type ShoppingEvent } from "./shopping-events";
const production = { hostname: "shop.example.test", origin: "https://shop.example.test", protocol: "https:" };
const local = { hostname: "127.0.0.1", origin: "http://127.0.0.1:4329", protocol: "http:" };
const settings = { PROD: true, VITE_SHOP_TELEMETRY: "production", VITE_SHOP_TELEMETRY_HOSTS: "shop.example.test", VITE_SHOP_TELEMETRY_API_ORIGIN: "https://api.example.test" };
const config: ShoppingTelemetryConfig = { endpoint: "http://127.0.0.1:4330/api/shopping-events", classification: { environment: "test", traffic: "synthetic" }, consent: "notice", modelVersion: null };
function event(sequence = 0): ShoppingEvent { return { eventId: `00000000-0000-4000-8000-${String(sequence).padStart(12, "0")}`, sequence, at: 1000, elapsedMs: 0, action: "search", context: { mode: "baseline", order: "popular", shipping: false, modelVersion: "deterministic-v1" }, data: { source: "submit" } }; }
function harness(responses: number[]) {
  const requests: RequestInit[] = [], timers = new Map<number, () => void>(); let timerId = 0, uuidId = 0;
  const transport = createShoppingTransport(config, "00000000-0000-4000-8000-000000000000", 1000, {
    fetch: (async (_url, init) => { requests.push(init!); const batch = JSON.parse(init!.body as string); const status = responses.shift() ?? 202; if (status === 0) throw new Error("offline"); return new Response(JSON.stringify({ batchId: batch.batchId, durableLocal: true }), { status }); }) as typeof fetch,
    now: () => 1000, uuid: () => `00000000-0000-4000-8000-${String(++uuidId).padStart(12, "0")}`,
    setTimer: (fn) => { timers.set(++timerId, fn); return timerId as unknown as ReturnType<typeof setTimeout>; }, clearTimer: (id) => { timers.delete(id as unknown as number); },
  });
  return { transport, requests, timers };
}
test("automatic collection cannot be promoted by development, unrelated host, plaintext or URL params", () => {
  assert.equal(shoppingTelemetryConfig({}, production), null);
  assert.equal(shoppingTelemetryConfig({ ...settings, PROD: false, DEV: true }, production), null);
  assert.equal(shoppingTelemetryConfig(settings, { ...production, hostname: "other.example.test" }), null);
  assert.equal(shoppingTelemetryConfig({ ...settings, VITE_SHOP_TELEMETRY_HOSTS: "127.0.0.1" }, local), null);
  assert.equal(shoppingTelemetryConfig(settings, { ...production, protocol: "http:" }), null);
  assert.equal(shoppingTelemetryConfig({ ...settings, VITE_SHOP_TELEMETRY_API_ORIGIN: "https://user:password@api.example.test" }, production), null);
  assert.equal(shoppingTelemetryConfig({ ...settings, VITE_SHOP_TELEMETRY_API_ORIGIN: "https://api.example.test/?secret=x" }, production), null);
  assert.equal(shoppingTelemetryConfig(settings, production)?.classification.traffic, "real");
  const synthetic = shoppingTelemetryConfig({ VITE_SHOP_TELEMETRY: "synthetic", VITE_SHOP_TELEMETRY_HOSTS: "127.0.0.1" }, local)!;
  assert.deepEqual(synthetic.classification, { environment: "test", traffic: "synthetic" });
  assert.equal(shoppingTelemetryConfig({ VITE_SHOP_TELEMETRY: "synthetic", VITE_SHOP_TELEMETRY_HOSTS: "shop.example.test" }, production), null);
});
test("consent defaults gated, approved notice auto starts, opt-out and browser signals win", () => {
  const required = shoppingTelemetryConfig(settings, production)!;
  assert.equal(privacyChoice(required, null, {}), "pending");
  assert.equal(privacyChoice(config, null, {}), "notice");
  assert.equal(privacyChoice(config, "denied", {}), "denied");
  assert.equal(privacyChoice(required, "granted", {}), "granted");
  assert.equal(privacyChoice(config, "granted", { globalPrivacyControl: true }), "denied");
  assert.equal(privacyChoice(config, "granted", { doNotTrack: "1" }), "denied");
});
test("offline retries preserve exact batch ID/body and omit cookies/referrer; stop discards", async () => {
  const { transport, requests } = harness([0, 503, 202]);
  transport.event(event()); await transport.flush(); await transport.flush(); await transport.flush();
  assert.equal(requests.length, 3);
  assert.ok(requests.every((request) => request.body === requests[0].body));
  assert.equal(requests[0].credentials, "omit"); assert.equal(requests[0].referrerPolicy, "no-referrer");
  assert.equal(transport.stats().delivered, 1);
  transport.event(event(1)); transport.stop(); await transport.flush();
  assert.equal(requests.length, 3); assert.equal(transport.stats().queued, 0);
});
test("permanent rejection never retries/mutates a batch and broken replay is halted", async () => {
  const { transport, requests } = harness([409]);
  transport.replay({ type: 2, timestamp: 1000, data: {} }); await transport.flush(); await transport.flush();
  assert.equal(requests.length, 1); assert.equal(transport.stats().failedBatches, 1);
  assert.equal(transport.replay({ type: 3, timestamp: 1001, data: {} }), false);
  assert.equal(transport.stats().replayIncomplete, true);
});
test("queue/event/replay batch limits and retry cap bound network/memory work", async () => {
  const { transport, requests } = harness([503, 503, 503]);
  for (let i = 0; i < SHOPPING_LIMITS.queueItems + 5; i++) transport.event(event(i));
  assert.equal(transport.stats().queued, SHOPPING_LIMITS.queueItems);
  assert.equal(transport.stats().dropped, 5);
  assert.ok(transport.stats().queuedBytes <= SHOPPING_LIMITS.queueBytes);
  await transport.flush(); await transport.flush(); await transport.flush();
  assert.equal(requests.length, SHOPPING_LIMITS.retries);
  const batch = JSON.parse(requests[0].body as string);
  assert.equal(batch.events.length, SHOPPING_LIMITS.batchEvents);
  assert.deepEqual(batch.classification, { environment: "test", traffic: "synthetic" });
  transport.stop();
});

test("default browser timers/fetch are bound to the global receiver", async () => {
  const originalTimer = globalThis.setTimeout, originalClear = globalThis.clearTimeout, originalFetch = globalThis.fetch;
  try {
    globalThis.setTimeout = function (this: unknown, ...args: Parameters<typeof setTimeout>) {
      assert.equal(this, globalThis); return originalTimer(...args);
    } as typeof setTimeout;
    globalThis.clearTimeout = function (this: unknown, ...args: Parameters<typeof clearTimeout>) {
      assert.equal(this, globalThis); return originalClear(...args);
    } as typeof clearTimeout;
    globalThis.fetch = async function (this: unknown, _url, init) {
      assert.equal(this, globalThis); const batch = JSON.parse(init!.body as string);
      return new Response(JSON.stringify({ batchId: batch.batchId, durableLocal: true }), { status: 202 });
    };
    const transport = createShoppingTransport(config, crypto.randomUUID(), 1000);
    transport.event(event()); await transport.flush(); transport.stop();
    assert.equal(transport.stats().delivered, 1);
  } finally { globalThis.setTimeout = originalTimer; globalThis.clearTimeout = originalClear; globalThis.fetch = originalFetch; }
});
