import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { applyLocalModelEvent, beginLocalModelRequest, INITIAL_LOCAL_MODEL_STATE,
  LOCAL_MODEL_ENDPOINT, readLocalModelStream, type LocalModelEvent, type LocalModelRequest } from "./local-model";
import { beginLocalModelPerformance } from "./local-model-performance";

type Listener = { event: (event: LocalModelEvent) => void; error: (error: Error) => void };
type SharedRequest = { controller: AbortController; listeners: Set<Listener>; events: LocalModelEvent[];
  error?: Error; cleanup?: ReturnType<typeof setTimeout> };
const requests = new Map<string, SharedRequest>();

/** One fetch survives React StrictMode's immediate effect cleanup/re-subscribe. */
export function subscribeLocalModel(request: LocalModelRequest, listener: Listener): () => void {
  const key = JSON.stringify(request);
  let shared = requests.get(key);
  if (!shared) {
    shared = { controller: new AbortController(), listeners: new Set(), events: [] };
    requests.set(key, shared);
    const entry = shared;
    void Promise.resolve().then(async () => {
      if (entry.controller.signal.aborted) return;
      try {
        const response = await fetch(LOCAL_MODEL_ENDPOINT, { method: "POST", headers: { "Content-Type": "application/json" },
          body: key, signal: entry.controller.signal, credentials: "omit" });
        await readLocalModelStream(response, (event) => {
          if (entry.controller.signal.aborted) return;
          entry.events.push(event);
          for (const active of entry.listeners) active.event(event);
        });
      } catch (value) {
        if (entry.controller.signal.aborted) return;
        entry.error = value instanceof Error ? value : new Error("Local model request failed.");
        for (const active of entry.listeners) active.error(entry.error);
      }
    });
  }
  if (shared.cleanup) clearTimeout(shared.cleanup);
  shared.listeners.add(listener);
  for (const event of shared.events) listener.event(event);
  if (shared.error) listener.error(shared.error);
  const entry = shared;
  return () => {
    entry.listeners.delete(listener);
    if (!entry.listeners.size) entry.cleanup = setTimeout(() => {
      if (!entry.listeners.size) { entry.controller.abort(); requests.delete(key); }
    }, 0);
  };
}

export function useLocalModel(enabled: boolean, query: string) {
  const [state, setState] = useState(INITIAL_LOCAL_MODEL_STATE);
  const generation = useRef(0);
  const measurement = useRef<ReturnType<typeof beginLocalModelPerformance>>(null);
  useEffect(() => {
    const startedAt = typeof performance === "undefined" ? undefined : performance.now();
    const current = ++generation.current;
    if (!enabled) { setState(INITIAL_LOCAL_MODEL_STATE); return; }
    const metrics = beginLocalModelPerformance(current, () => generation.current === current, startedAt);
    measurement.current = metrics;
    const actualQuery = query.trim() || "Browse the store";
    setState((previous) => beginLocalModelRequest(previous, actualQuery));
    const unsubscribe = subscribeLocalModel({ query: actualQuery }, {
      event: (event) => {
        if (generation.current !== current) return;
        metrics?.recordEvent(event);
        setState((previous) => {
          try { return applyLocalModelEvent(previous, event, actualQuery); }
          catch (error) { return { ...previous, phase: "error", error: error instanceof Error ? error.message : "Invalid local model response." }; }
        });
      },
      error: (error) => {
        if (generation.current === current) setState((previous) => ({ ...previous, phase: "error", error: error.message }));
      },
    });
    return () => { generation.current += 1; metrics?.dispose(); measurement.current = null; unsubscribe(); };
  }, [enabled, query]);
  useLayoutEffect(() => { measurement.current?.afterCommit(state); }, [state]);
  return state;
}
