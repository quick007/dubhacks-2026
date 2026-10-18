import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { applyLocalModelEvent, beginLocalModelRequest, buildLocalModelRequest, INITIAL_LOCAL_MODEL_STATE,
  LOCAL_MODEL_ENDPOINT, type LocalModelRequest, type PersonalizationContext } from "./local-model";
import { beginLocalModelPerformance } from "./local-model-performance";
import { subscribeLocalModel } from "./local-model-client";
export { subscribeLocalModel } from "./local-model-client";

export function useLocalModel(enabled: boolean, query: string, context: PersonalizationContext = {}, endpoint = LOCAL_MODEL_ENDPOINT) {
  const requestKey = JSON.stringify({ endpoint, request: buildLocalModelRequest(query, context) });
  const [state, setState] = useState(INITIAL_LOCAL_MODEL_STATE);
  const generation = useRef(0);
  const measurement = useRef<ReturnType<typeof beginLocalModelPerformance>>(null);
  useEffect(() => {
    const startedAt = typeof performance === "undefined" ? undefined : performance.now();
    const current = ++generation.current;
    if (!enabled) { setState(INITIAL_LOCAL_MODEL_STATE); return; }
    const metrics = beginLocalModelPerformance(current, () => generation.current === current, startedAt);
    measurement.current = metrics;
    const request = (JSON.parse(requestKey) as { request: LocalModelRequest }).request;
    setState(previous => beginLocalModelRequest(previous, request.query, requestKey, true));
    const unsubscribe = subscribeLocalModel(request, {
      event: event => {
        if (generation.current !== current) return;
        metrics?.recordEvent(event);
        setState(previous => {
          if (generation.current !== current) return previous;
          try { return applyLocalModelEvent(previous, event, request.query, requestKey); }
          catch (error) { return { ...previous, phase: "error", error: error instanceof Error ? error.message : "Invalid local model response." }; }
        });
      },
      error: error => {
        if (generation.current === current) setState(previous => generation.current === current
          ? { ...previous, phase: "error", error: error.message } : previous);
      },
    }, endpoint);
    return () => { generation.current += 1; metrics?.dispose(); measurement.current = null; unsubscribe(); };
  }, [enabled, requestKey, endpoint]);
  // Never render the prior person's copy while waiting for the new effect to run.
  const visibleState = !enabled ? INITIAL_LOCAL_MODEL_STATE : state.requestKey === requestKey ? state
    : beginLocalModelRequest(INITIAL_LOCAL_MODEL_STATE, query.trim() || "Browse the store", requestKey, true);
  useLayoutEffect(() => { measurement.current?.afterCommit(visibleState); }, [visibleState]);
  return visibleState;
}
