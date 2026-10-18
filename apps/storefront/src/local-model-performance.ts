import type { LocalModelEvent, LocalModelState } from "./local-model";

type Stage = "rank" | "firstThree" | "complete";
type PerformanceEnvironment = {
  performance: { now: () => number; mark: (name: string, options?: PerformanceMarkOptions) => unknown;
    measure: (name: string, options: PerformanceMeasureOptions) => unknown; clearMarks: (name: string) => void };
  requestFrame: (callback: FrameRequestCallback) => number;
  cancelFrame: (handle: number) => void;
  documentElement?: Pick<Element, "getAttribute" | "setAttribute"> | null;
};
let sequence = 0;
export const LOCAL_MODEL_READINESS_ATTRIBUTE = "data-local-model-readiness";
export const LOCAL_MODEL_MEASURES = {
  rank: "fieldwork.local-model.rank.after-paint-opportunity",
  firstThree: "fieldwork.local-model.firstThree.after-paint-opportunity",
  complete: "fieldwork.local-model.complete.after-paint-opportunity",
} as const;

/** Browser-local only: DOM commit plus two frames is a paint opportunity,
 * not proof of physical paint, image download/decode, or viewport visibility. */
export function beginLocalModelPerformance(generation: number, isCurrent: () => boolean,
  startedAt?: number, environment?: PerformanceEnvironment) {
  const env: PerformanceEnvironment | null = environment ?? (typeof performance !== "undefined" && typeof requestAnimationFrame === "function"
    ? { performance, requestFrame: requestAnimationFrame.bind(globalThis), cancelFrame: cancelAnimationFrame.bind(globalThis),
        documentElement: typeof document === "undefined" ? null : document.documentElement } : null);
  if (!env) return null;
  const prefix = `fieldwork.local-model.request.${++sequence}`;
  const start = `${prefix}.start`;
  const marks = new Set([start]), frames = new Set<number>(), scheduled = new Set<Stage>();
  const received = new Map<Stage, Exclude<LocalModelEvent, { type: "error" }>>();
  let disposed = false;
  const startScope = "hook effect entry before query normalization and subscribe/fetch; includes network and service queue; excludes click-to-effect, image transfer/decode and physical paint";
  try { env.performance.mark(start, { startTime: startedAt ?? env.performance.now(), detail: { generation, scope: startScope } }); }
  catch { return null; }
  const live = () => !disposed && isCurrent();
  return {
    recordEvent(event: LocalModelEvent) {
      if (live() && event.type !== "error") received.set(event.type, event);
    },
    afterCommit(state: LocalModelState) {
      if (!live() || state.phase === "ranking" || !state.rank) return;
      const rank = received.get("rank");
      if (!rank || rank.requestId !== state.rank.requestId) return;
      const stages: Stage[] = ["rank"];
      if (state.copy?.requestId === rank.requestId) {
        if (received.has("firstThree") && state.firstThreeReadyMs !== null) stages.push("firstThree");
        if (state.copy.type === "complete" && received.has("complete")) stages.push("complete");
      }
      for (const stage of stages) {
        if (scheduled.has(stage)) continue;
        scheduled.add(stage);
        const event = received.get(stage)!;
        const detail = { generation, requestId: rank.requestId, stage, committedPhase: state.phase,
          acceptedFields: event.type === "rank" ? null : event.acceptedFields,
          withheldFields: event.type === "rank" ? null : event.withheldFields,
          requiredFields: event.type === "rank" ? null : event.requiredFields,
          copySource: event.type === "rank" ? null : event.copySource ?? { kind: "fresh-native", generatedEarlier: false, freshPerRequestGeneration: true },
          fieldCountScope: "server accepted/withheld fields; shopper variant changes and viewport visibility are not measured",
          nativeTiming: event.timing, requestStartScope: startScope,
          readinessScope: "React DOM commit followed by two requestAnimationFrame callbacks; after-paint opportunity only",
          imagesReadyMeasured: false, physicalPaintMeasured: false, endpointUpload: false };
        const commit = `${prefix}.${stage}.react-commit`;
        const end = `${prefix}.${stage}.after-paint-opportunity`;
        marks.add(commit); marks.add(end);
        try {
          env.performance.mark(commit, { detail });
          const first = env.requestFrame(() => {
            frames.delete(first);
            if (!live()) return;
            const second = env.requestFrame(() => {
              frames.delete(second);
              if (!live()) return;
              try {
                env.performance.mark(end, { detail });
                const measured = env.performance.measure(LOCAL_MODEL_MEASURES[stage], { start, end, detail });
                if (env.documentElement && measured && typeof measured === "object") {
                  const entry = measured as PerformanceMeasure;
                  if (typeof entry.name === "string" && Number.isFinite(entry.startTime) && Number.isFinite(entry.duration)) {
                    let previous: unknown[] = [];
                    try {
                      const stored: unknown = JSON.parse(env.documentElement.getAttribute(LOCAL_MODEL_READINESS_ATTRIBUTE) ?? "[]");
                      if (Array.isArray(stored)) previous = stored.slice(-11);
                    } catch { /* A malformed mirror never affects native timing. */ }
                    // Standard completed measures, mirrored as inspectable DOM data.
                    // This helper is created only by the opt-in local-model hook.
                    env.documentElement.setAttribute(LOCAL_MODEL_READINESS_ATTRIBUTE, JSON.stringify([...previous,
                      { name: entry.name, startTime: entry.startTime, duration: entry.duration, detail }]));
                  }
                }
              } catch { /* Measurement support must never interrupt the demo. */ }
            });
            frames.add(second);
          });
          frames.add(first);
        } catch { /* Unsupported User Timing or rAF does not affect results. */ }
      }
    },
    dispose() {
      disposed = true;
      for (const frame of frames) env.cancelFrame(frame);
      for (const mark of marks) env.performance.clearMarks(mark);
      frames.clear();
      // Keep completed PerformanceMeasure entries available for read-only capture.
    },
  };
}
