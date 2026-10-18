import { connectTrace, type TraceContext } from "./session-trace-bridge";
import { createTraceBuffer, TRACE_LIMITS, TRACE_RECORD_OPTIONS } from "./session-trace";

type Buffer = ReturnType<typeof createTraceBuffer>;
let buffer: Buffer | null = null;
let status = "Idle";
let generation = 0;
let stopRecording: (() => void) | undefined;
let disconnect: (() => void) | undefined;
let observer: IntersectionObserver | undefined;
let performanceObserver: PerformanceObserver | undefined;
let stopTimer: ReturnType<typeof setTimeout> | undefined;
let clearTimer: ReturnType<typeof setTimeout> | undefined;
let destroyReplay: (() => void) | undefined;
const notify = (value: string) => { status = value; window.dispatchEvent(new Event("fieldwork-trace-status")); };
export const traceStatus = () => status;
export function stopTrace(reason = "manual") {
  generation++;
  stopRecording?.(); stopRecording = undefined;
  disconnect?.(); disconnect = undefined;
  observer?.disconnect(); observer = undefined;
  performanceObserver?.disconnect(); performanceObserver = undefined;
  clearTimeout(stopTimer);
  buffer?.stop(reason);
  notify(buffer ? "Stopped" : "Idle");
}
export function clearTrace() {
  stopTrace(); clearTimeout(clearTimer); destroyReplay?.(); destroyReplay = undefined;
  buffer = null; notify("Idle");
}
export function getTrace() { const value = buffer?.export() ?? null; if (!value && buffer) clearTrace(); return value; }
function observeProducts(ids: string[]) {
  observer?.disconnect();
  if (!buffer?.live() || !globalThis.IntersectionObserver) return;
  const identity = new Map<Element, string>();
  observer = new IntersectionObserver((entries) => {
    for (const entry of entries) if (entry.isIntersecting && entry.intersectionRatio >= 0.5) {
      const id = identity.get(entry.target); if (id) buffer?.product(id);
    }
  }, { threshold: 0.5 });
  document.querySelectorAll("article.product").forEach((element, index) => {
    if (ids[index]) { identity.set(element, ids[index]); observer!.observe(element); }
  });
}
export async function startTrace() {
  if (status === "Recording" || status === "Loading") return;
  // This spike is deliberately restricted to the local demo host.
  if (!/^(localhost|127\.0\.0\.1|\[::1\]|[a-z0-9.-]+\.localhost)$/.test(location.hostname)) { notify("Local demo host required"); return; }
  clearTrace(); notify("Loading");
  const current = ++generation;
  try {
    const { record } = await import("@rrweb/record");
    if (current !== generation) return;
    buffer = createTraceBuffer();
    const active = buffer;
    stopRecording = record({ ...TRACE_RECORD_OPTIONS, emit: (event) => {
      if (current !== generation) return;
      active.replay(event);
      if (!active.live()) queueMicrotask(() => { if (current === generation) stopTrace("buffer-limit"); });
    } });
    if (!stopRecording) throw new Error("Recorder unavailable");
    disconnect = connectTrace({
      context: (value: TraceContext) => active.context(value), products: observeProducts,
      event: (action, fields) => { active.event(action, fields); if (!active.live()) stopTrace("buffer-limit"); },
    });
    if (globalThis.PerformanceObserver) {
      performanceObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) active.readiness(entry as PerformanceMeasure);
        if (!active.live()) stopTrace("buffer-limit");
      });
      performanceObserver.observe({ type: "measure" });
    }
    stopTimer = setTimeout(() => stopTrace("duration-limit"), TRACE_LIMITS.durationMs);
    clearTimer = setTimeout(clearTrace, TRACE_LIMITS.retentionMs);
    notify("Recording");
  } catch { if (current === generation) { stopTrace("recorder-error"); notify("Recorder unavailable"); } }
}
export function exportTrace() {
  const trace = getTrace(); if (!trace) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(trace)], { type: "application/json" }));
  const link = document.createElement("a"); link.href = url; link.download = `demo-session-${trace.sessionId}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
export async function replayTrace() {
  stopTrace(); const trace = getTrace(); if (!trace || trace.replay.events.length < 2) return;
  const token = generation;
  const { Replayer } = await import("@rrweb/replay");
  await import("@rrweb/replay/dist/style.css");
  if (token !== generation || !getTrace()) return;
  destroyReplay?.();
  const container = document.createElement("div"); container.id = "trace-replay"; container.setAttribute("data-trace-private", "");
  container.style.cssText = "position:fixed;inset:16px;z-index:1000;background:white;overflow:auto;padding:16px;box-shadow:0 4px 32px #0004";
  const close = document.createElement("button"); close.textContent = "Close replay"; close.className = "demo-button";
  const root = document.createElement("div"); container.append(close, root); document.body.append(container);
  const player = new Replayer(trace.replay.events as ConstructorParameters<typeof Replayer>[0], { root, speed: 2, showWarning: false, mouseTail: false, UNSAFE_replayCanvas: false });
  destroyReplay = () => { player.destroy(); container.remove(); };
  close.onclick = () => { destroyReplay?.(); destroyReplay = undefined; };
  player.play();
}
// Small local integration surface for synthetic validation and a future dashboard.
// Never installed on the disabled path. No fetch, beacon, storage or credentials.
export const demoTrace = { start: startTrace, stop: stopTrace, clear: clearTrace, getTrace, replay: replayTrace, status: traceStatus };
declare global { interface Window { fieldworkTrace?: typeof demoTrace } }
window.fieldworkTrace = demoTrace;
