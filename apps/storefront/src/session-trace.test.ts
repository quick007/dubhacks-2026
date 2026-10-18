import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import { createTraceBuffer, sanitizeReplay, canonicalReplayAsset, TRACE_LIMITS, TRACE_RECORD_OPTIONS } from "./session-trace";
import { demoTraceEnabled, traceEvent, traceProducts } from "./session-trace-bridge";

test("disabled bridge is inert and never imports recorder, storage or observers", () => {
  assert.equal(demoTraceEnabled, false);
  for (let i = 0; i < 10000; i++) traceEvent("search", { source: "submit" });
  traceProducts(["mug"]);
  const bridge = readFileSync(new URL("./session-trace-bridge.ts", import.meta.url), "utf8");
  assert.doesNotMatch(bridge, /import\(|@rrweb|localStorage|sessionStorage|MutationObserver|PerformanceObserver|fetch\(/);
});
test("mask all inputs/text; block form fields, checkout, account, iframe and private areas", () => {
  assert.equal(TRACE_RECORD_OPTIONS.maskAllInputs, true);
  assert.equal(TRACE_RECORD_OPTIONS.maskInputFn(), "[masked]");
  assert.equal(TRACE_RECORD_OPTIONS.maskTextFn("private@example.com"), "*******************");
  for (const required of ["input", "textarea", "contenteditable", "checkout", "account", "payment", "iframe", "cart-page", "confirmation-page", "data-trace-private"]) assert.ok(TRACE_RECORD_OPTIONS.blockSelector.includes(required));
  assert.deepEqual(TRACE_RECORD_OPTIONS.plugins, []);
  assert.equal(TRACE_RECORD_OPTIONS.recordCrossOriginIframes, false);
});
test("replay sanitizer drops sensitive URL, title, attrs, input, custom/network payloads", () => {
  const secret = "sensitive-canary@example.com";
  const safe = sanitizeReplay({ type: 2, timestamp: 1, data: { node: { type: 0, id: 1, childNodes: [
    { type: 2, id: 2, tagName: "div", attributes: { title: secret, "aria-label": secret, "data-local-model-readiness": secret, value: secret, href: `/?q=${secret}`, class: "hero" }, childNodes: [{ type: 3, id: 3, textContent: secret }] },
    { type: 2, id: 4, tagName: "img", attributes: { src: `https://tracker.example/x?q=${secret}` }, childNodes: [] },
  ] } } });
  assert.ok(safe);
  assert.ok(!JSON.stringify(safe).includes(secret));
  assert.deepEqual(sanitizeReplay({ type: 4, timestamp: 1, data: { href: `https://foo/?email=${secret}#q=${secret}`, width: 100, height: 100 } })?.data, { href: "http://localhost/", width: 100, height: 100 });
  assert.equal(sanitizeReplay({ type: 3, timestamp: 1, data: { source: 5, text: secret } }), null);
  assert.equal(sanitizeReplay({ type: 6, timestamp: 1, data: { headers: secret, body: secret } }), null);
  const mutation = sanitizeReplay({ type: 3, timestamp: 1, data: { source: 0, texts: [{ id: 3, value: secret }], attributes: [{ id: 2, attributes: { value: secret, title: secret } }], adds: [], removes: [] } });
  assert.ok(!JSON.stringify(mutation).includes(secret));
});
test("event schema only accepts fixture identities and structured action metadata", () => {
  let now = 1000;
  const buffer = createTraceBuffer(() => now, () => "anonymous-test-id");
  buffer.context({ mode: "local-model", order: "budget", shipping: true });
  buffer.event("search", { source: "submit", query: "secret" } as never);
  buffer.event("selection", { productId: "mug", variantId: "mug-cobalt", email: "secret" } as never);
  buffer.event("cart-add", { productId: "mug", variantId: "mug-cobalt", count: 1 });
  buffer.event("selection", { productId: "secret", variantId: "secret" });
  buffer.event("checkout-success", { count: 1 });
  buffer.product("mug"); buffer.product("mug"); buffer.product("secret");
  buffer.readiness({ name: "fieldwork.local-model.complete.after-paint-opportunity", duration: 10, detail: { query: "secret", requestId: "secret", copySource: { kind: "fresh-parallel-head" } } });
  const trace = buffer.export()!;
  assert.equal(trace.events.length, 6);
  assert.ok(!JSON.stringify(trace).includes("secret"));
  assert.equal(trace.events[1]?.data.variantId, "mug-cobalt");
  assert.equal(trace.events[0]?.context.modelVersion, null);
  assert.equal(trace.events.at(-1)?.data.scope, "after-paint-opportunity");
  assert.equal(trace.events.at(-1)?.data.copySource, "fresh-parallel-head");
  now += 10; buffer.stop(); buffer.event("search"); assert.equal(buffer.export()!.events.length, 6);
});
test("bounds stop a complete replay prefix and expiry prevents later export", () => {
  let now = 0;
  const buffer = createTraceBuffer(() => now, () => "test");
  for (let i = 0; i < TRACE_LIMITS.replayEvents + 3; i++) buffer.replay({ type: 1, timestamp: i, data: {} });
  assert.equal(buffer.export()?.replay.events.length, TRACE_LIMITS.replayEvents);
  assert.equal(buffer.export()?.stopReason, "buffer-limit");
  now = TRACE_LIMITS.retentionMs;
  assert.equal(buffer.export(), null);
  const duration = createTraceBuffer(() => now, () => "test2");
  now += TRACE_LIMITS.durationMs; duration.event("search");
  assert.equal(duration.export()?.stopReason, "duration-limit");
  const size = createTraceBuffer(() => 0, () => "size");
  size.replay({ type: 2, timestamp: 0, data: { node: { type: 3, id: 1, textContent: "x".repeat(TRACE_LIMITS.bytes + 1) } } });
  assert.equal(size.export()?.replay.events.length, 0);
  assert.equal(size.export()?.stopReason, "buffer-limit");
});

test("readiness recognizes every existing copy-source kind while excluding arbitrary provenance", () => {
  const buffer = createTraceBuffer(() => 1, () => "test");
  for (const kind of ["fresh-native", "fresh-parallel-head", "generated-earlier", "mixed-native-cache", "no-generation", "private@example.invalid"]) {
    buffer.readiness({ name: "fieldwork.local-model.complete.after-paint-opportunity", duration: 10, detail: { copySource: { kind } } });
  }
  assert.deepEqual(buffer.export()?.events.map((event) => event.data.copySource), ["fresh-native", "fresh-parallel-head", "generated-earlier", "mixed-native-cache", "no-generation", null]);
});

test("responsive replay images map only to known original fixtures", () => {
  assert.equal(canonicalReplayAsset("/products/optimized/responsive/mug-colors.jpg.w384.webp"), "/products/mug-colors.jpg");
  assert.equal(canonicalReplayAsset("http://localhost/products/optimized/mug-colors.jpg.webp"), "/products/mug-colors.jpg");
  assert.equal(canonicalReplayAsset("/products/optimized/responsive/unknown-private.jpg.w384.webp"), "");
  assert.equal(canonicalReplayAsset("/products/optimized/responsive/mug-colors.jpg.w384.webp?email=private"), "");
  assert.equal(canonicalReplayAsset("/products/optimized/responsive/mug-colors.jpg.w123.webp"), "");
});
