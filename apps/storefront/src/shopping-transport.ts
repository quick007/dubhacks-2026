import { SHOPPING_LIMITS, type ShoppingBatch, type ShoppingEvent, type SanitizedReplayEvent } from "./shopping-events";
import type { ShoppingTelemetryConfig } from "./shopping-telemetry-config";
type Item = { kind: "event"; value: ShoppingEvent } | { kind: "replay"; value: SanitizedReplayEvent };
type Environment = { fetch: typeof fetch; now: () => number; uuid: () => string; setTimer: (fn: () => void, ms: number) => ReturnType<typeof setTimeout>; clearTimer: (timer: ReturnType<typeof setTimeout>) => void };
const byteSize = (value: unknown) => new TextEncoder().encode(JSON.stringify(value)).length;
/** Volatile bounded queue. Stable in-flight batch IDs make ambiguous retries idempotent.
 * No outbox storage, cookies, referrer, credentials, synchronous requests or beacon. */
export function createShoppingTransport(config: ShoppingTelemetryConfig, sessionId: string, startedAt: number, supplied?: Environment) {
  const env: Environment = supplied ?? { fetch: fetch.bind(globalThis), now: Date.now, uuid: () => crypto.randomUUID(), setTimer: setTimeout.bind(globalThis), clearTimer: clearTimeout.bind(globalThis) };
  let queue: Item[] = [], bytes = 0, stopped = false, pending: { batch: ShoppingBatch; body: string; bodyBytes: number; attempt: number } | null = null;
  let replayBlocked = false;
  let timer: ReturnType<typeof setTimeout> | undefined, sending = false, abort: AbortController | undefined;
  const counters = { accepted: 0, delivered: 0, dropped: 0, failedBatches: 0 };
  function schedule(ms = 2000) {
    if (stopped || timer !== undefined) return;
    timer = env.setTimer(() => { timer = undefined; void flush(); }, ms);
  }
  function disableReplay() {
    replayBlocked = true; queue = queue.filter((item) => item.kind !== "replay");
    bytes = queue.reduce((sum, item) => sum + byteSize(item), 0);
  }
  function enqueue(item: Item) {
    if (stopped || (item.kind === "replay" && replayBlocked)) return false;
    const size = byteSize(item);
    const total = bytes + (pending?.bodyBytes ?? 0);
    if (size > SHOPPING_LIMITS.replayBytes || total + size > SHOPPING_LIMITS.queueBytes || queue.length >= SHOPPING_LIMITS.queueItems) { counters.dropped++; if (item.kind === "replay") disableReplay(); return false; }
    queue.push(item); bytes += size; counters.accepted++; schedule(); return true;
  }
  function makeBatch() {
    const batch: ShoppingBatch = { schema: "fieldwork.shopping-batch.v1", batchId: env.uuid(), sessionId,
      classification: config.classification, startedAt, events: [] };
    let count = 0, replayBytes = 0;
    for (const item of queue) {
      if (item.kind === "event" && batch.events.length >= SHOPPING_LIMITS.batchEvents) break;
      if (item.kind === "replay" && ((batch.replay?.events.length ?? 0) >= SHOPPING_LIMITS.batchReplay || replayBytes + byteSize(item.value) > SHOPPING_LIMITS.replayBytes)) break;
      if (item.kind === "event") batch.events.push(item.value);
      else { batch.replay ??= { schema: "fieldwork.sanitized-replay.v1", events: [] }; batch.replay.events.push(item.value); }
      if (byteSize(batch) > SHOPPING_LIMITS.batchBytes) {
        if (item.kind === "event") batch.events.pop();
        else { batch.replay!.events.pop(); if (!batch.replay!.events.length) delete batch.replay; }
        break;
      }
      count++; if (item.kind === "replay") replayBytes += byteSize(item.value);
    }
    if (!count) return null;
    queue = queue.slice(count); bytes = queue.reduce((sum, item) => sum + byteSize(item), 0);
    return { batch, body: JSON.stringify(batch), bodyBytes: byteSize(batch), attempt: 0 };
  }
  async function flush(pagehide = false) {
    if (stopped || sending) return;
    if (timer !== undefined) { env.clearTimer(timer); timer = undefined; }
    pending ??= makeBatch();
    if (!pending) return;
    const current = pending; sending = true; abort = new AbortController();
    const timeout = env.setTimer(() => abort?.abort(), 8000);
    let retry = false;
    try {
      const response = await env.fetch(config.endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: current.body,
        credentials: "omit", referrerPolicy: "no-referrer", mode: "cors", keepalive: pagehide && current.bodyBytes < 60_000, signal: abort.signal });
      if (response.status === 202) {
        const receipt = await response.json() as { batchId?: string; durableLocal?: boolean };
        if (receipt.batchId !== current.batch.batchId || receipt.durableLocal !== true) retry = true;
        else { counters.delivered += current.batch.events.length + (current.batch.replay?.events.length ?? 0); pending = null; }
      } else if (response.status === 429 || response.status >= 500) retry = true;
      else { counters.failedBatches++; if (current.batch.replay) disableReplay(); pending = null; }
    } catch { retry = true; }
    finally { env.clearTimer(timeout); sending = false; abort = undefined; }
    if (stopped) return;
    if (retry) {
      current.attempt++;
      if (current.attempt >= SHOPPING_LIMITS.retries) { counters.failedBatches++; if (current.batch.replay) disableReplay(); pending = null; }
    }
    if (pending || queue.length) schedule(pending ? 1000 * 2 ** current.attempt : 10);
  }
  return { event: (value: ShoppingEvent) => enqueue({ kind: "event", value }), replay: (value: SanitizedReplayEvent) => enqueue({ kind: "replay", value }), flush,
    stop() { stopped = true; if (timer !== undefined) env.clearTimer(timer); abort?.abort(); queue = []; pending = null; bytes = 0; },
    stats: () => ({ ...counters, queued: queue.length, queuedBytes: bytes, inFlight: !!pending, replayIncomplete: replayBlocked }),
  };
}
