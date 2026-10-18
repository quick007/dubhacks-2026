import { test } from "node:test";
import assert from "node:assert/strict";
import { beginLocalModelPerformance, LOCAL_MODEL_MEASURES, LOCAL_MODEL_READINESS_ATTRIBUTE } from "./local-model-performance";
import { applyLocalModelEvent, beginLocalModelRequest, INITIAL_LOCAL_MODEL_STATE,
  type LocalCopyEvent, type LocalRankEvent } from "./local-model";

function fixture() {
  let now = 10, id = 0;
  const frames = new Map<number, FrameRequestCallback>();
  const marks = new Map<string, { time: number; detail: unknown }>();
  const measures: { name: string; duration: number; detail: unknown }[] = [];
  const environment = {
    performance: {
      now: () => now,
      mark(name: string, options?: PerformanceMarkOptions) { marks.set(name, { time: options?.startTime ?? now, detail: options?.detail }); },
      measure(name: string, options: PerformanceMeasureOptions) {
        const start = marks.get(options.start as string)!, end = marks.get(options.end as string)!;
        const entry = { name, startTime: start.time, duration: end.time - start.time, detail: options.detail };
        measures.push(entry);
        return entry;
      },
      clearMarks(name: string) { marks.delete(name); },
    },
    requestFrame(callback: FrameRequestCallback) { frames.set(++id, callback); return id; },
    cancelFrame(handle: number) { frames.delete(handle); },
  };
  const frame = () => {
    now += 16;
    const pending = [...frames.values()]; frames.clear();
    pending.forEach((callback) => callback(now));
  };
  return { environment, frames, marks, measures, frame };
}
const rank: LocalRankEvent = { type: "rank", requestId: "measured-request", productIds: [], selectedProductIds: [],
  variants: {}, shortfall: 6, timing: { rankMs: 120 }, provenance: { ranking: "Learned", images: "Existing gallery" } };
const first: LocalCopyEvent = { type: "firstThree", requestId: rank.requestId, sections: [],
  acceptedFields: 8, withheldFields: 2, requiredFields: 10, timing: { readyMs: 1000, copyMs: 880 } };
const complete: LocalCopyEvent = { ...first, type: "complete", acceptedFields: 17,
  withheldFields: 3, requiredFields: 20, timing: { readyMs: 2000, copyMs: 1880 } };
function rankedState() {
  return applyLocalModelEvent(beginLocalModelRequest(INITIAL_LOCAL_MODEL_STATE, "test"), rank, "test");
}

void test("no measure before React commit and two separate animation frame callbacks", () => {
  const f = fixture(); const metrics = beginLocalModelPerformance(7, () => true, 3, f.environment)!;
  metrics.recordEvent(rank);
  f.frame(); f.frame(); assert.equal(f.measures.length, 0);
  metrics.afterCommit(rankedState()); assert.equal(f.frames.size, 1);
  f.frame(); assert.equal(f.measures.length, 0); assert.equal(f.frames.size, 1);
  f.frame(); assert.equal(f.measures.length, 1);
  assert.equal(f.measures[0].name, LOCAL_MODEL_MEASURES.rank);
  assert.equal(f.measures[0].duration, 71);
  const detail = f.measures[0].detail as Record<string, unknown>;
  assert.equal(detail.requestId, rank.requestId); assert.equal(detail.generation, 7);
  assert.equal(detail.imagesReadyMeasured, false); assert.equal(detail.physicalPaintMeasured, false);
  assert.equal(detail.endpointUpload, false);
  assert.match(detail.requestStartScope as string, /hook effect entry/);
  assert.match(detail.readinessScope as string, /opportunity only/);
});

void test("firstThree and complete retain their own field counts even when React batches them", () => {
  const f = fixture(); const metrics = beginLocalModelPerformance(1, () => true, 0, f.environment)!;
  metrics.recordEvent(rank); metrics.recordEvent(first); metrics.recordEvent(complete);
  const state = { ...rankedState(), phase: "complete" as const, copy: complete, firstThreeReadyMs: 1000 };
  metrics.afterCommit(state); metrics.afterCommit(state);
  f.frame(); f.frame();
  assert.deepEqual(f.measures.map((measure) => measure.name), Object.values(LOCAL_MODEL_MEASURES));
  assert.deepEqual(f.measures.map((measure) => {
    const detail = measure.detail as Record<string, unknown>;
    return [detail.acceptedFields, detail.withheldFields, detail.requiredFields, detail.committedPhase];
  }), [[null, null, null, "complete"], [8, 2, 10, "complete"], [17, 3, 20, "complete"]]);
  metrics.afterCommit(state); f.frame(); f.frame(); assert.equal(f.measures.length, 3);
});

void test("stale generation, request mismatch and pending state cannot create measures", () => {
  const f = fixture(); let current = true;
  const metrics = beginLocalModelPerformance(2, () => current, 0, f.environment)!;
  metrics.recordEvent(rank);
  metrics.afterCommit(beginLocalModelRequest(rankedState(), "next"));
  metrics.afterCommit({ ...rankedState(), rank: { ...rank, requestId: "stale" } });
  assert.equal(f.frames.size, 0);
  metrics.afterCommit(rankedState()); current = false;
  f.frame(); f.frame(); assert.equal(f.measures.length, 0);
  metrics.recordEvent(complete); metrics.afterCommit(rankedState());
  assert.equal(f.frames.size, 0);
});

void test("cleanup cancels queued frames, clears marks and preserves completed measures", () => {
  const f = fixture(); const metrics = beginLocalModelPerformance(3, () => true, 0, f.environment)!;
  metrics.recordEvent(rank); metrics.afterCommit(rankedState()); f.frame(); f.frame();
  metrics.recordEvent(first);
  metrics.afterCommit({ ...rankedState(), phase: "partial", copy: first, firstThreeReadyMs: 1000 });
  assert.equal(f.frames.size, 1);
  metrics.dispose(); assert.equal(f.frames.size, 0); assert.equal(f.marks.size, 0);
  f.frame(); f.frame(); assert.equal(f.measures.length, 1);
});

void test("unavailable timing APIs do not affect normal model results", () => {
  assert.equal(beginLocalModelPerformance(1, () => true), null);
  const f = fixture(); f.environment.performance.mark = () => { throw new Error("unsupported timing"); };
  assert.equal(beginLocalModelPerformance(1, () => true, 0, f.environment), null);
});

void test("DOM mirror matches completed measures, keeps only12, and never writes stale readiness", () => {
  const f = fixture();
  const attributes = new Map<string, string>(); let writes = 0;
  const environment = { ...f.environment, documentElement: {
    getAttribute: (name: string) => attributes.get(name) ?? null,
    setAttribute: (name: string, value: string) => { writes += 1; attributes.set(name, value); },
  } };
  for (let generation = 1; generation <= 13; generation += 1) {
    const metrics = beginLocalModelPerformance(generation, () => true, 0, environment)!;
    metrics.recordEvent(rank); metrics.afterCommit(rankedState()); f.frame(); f.frame(); metrics.dispose();
  }
  const serialized = attributes.get(LOCAL_MODEL_READINESS_ATTRIBUTE)!;
  const mirrored = JSON.parse(serialized) as { name: string; duration: number; detail: { generation: number } }[];
  assert.equal(mirrored.length, 12); assert.equal(mirrored[0].detail.generation, 2);
  assert.equal(mirrored.at(-1)?.name, f.measures.at(-1)?.name);
  assert.equal(mirrored.at(-1)?.duration, f.measures.at(-1)?.duration);
  assert.deepEqual(mirrored.at(-1)?.detail, f.measures.at(-1)?.detail);
  let current = true;
  const stale = beginLocalModelPerformance(14, () => current, 0, environment)!;
  stale.recordEvent(rank); stale.afterCommit(rankedState()); f.frame(); current = false; f.frame();
  assert.equal(writes, 13); assert.equal(attributes.get(LOCAL_MODEL_READINESS_ATTRIBUTE), serialized);
  stale.dispose();
});
