import { LOCAL_MODEL_ENDPOINT, readLocalModelStream, type LocalModelEvent, type LocalModelRequest } from "./local-model";

type Listener = { event: (event: LocalModelEvent) => void; error: (error: Error) => void };
type SharedRequest = { controller: AbortController; listeners: Set<Listener>; events: LocalModelEvent[];
  error?: Error; cleanup?: ReturnType<typeof setTimeout> };

export function createLocalModelClient(fetchRequest: typeof fetch = (...args) => fetch(...args)) {
  const requests = new Map<string, SharedRequest>();
  function cancel(key: string, entry: SharedRequest) {
    if (entry.cleanup) clearTimeout(entry.cleanup);
    entry.controller.abort();
    if (requests.get(key) === entry) requests.delete(key);
  }
  return {
    /** Same-turn StrictMode re-subscription shares its fetch; a different context cancels detached work immediately. */
    subscribe(request: LocalModelRequest, listener: Listener, endpoint = LOCAL_MODEL_ENDPOINT): () => void {
      const body = JSON.stringify(request), key = `${endpoint}\n${body}`;
      for (const [otherKey, entry] of requests) if (otherKey !== key && !entry.listeners.size) cancel(otherKey, entry);
      let shared = requests.get(key);
      if (!shared) {
        shared = { controller: new AbortController(), listeners: new Set(), events: [] };
        requests.set(key, shared);
        const entry = shared;
        void Promise.resolve().then(async () => {
          if (entry.controller.signal.aborted) return;
          try {
            const response = await fetchRequest(endpoint, { method: "POST", headers: { "Content-Type": "application/json" },
              body, signal: entry.controller.signal, credentials: "omit" });
            await readLocalModelStream(response, event => {
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
          if (!entry.listeners.size) cancel(key, entry);
        }, 0);
      };
    },
  };
}

export const subscribeLocalModel = createLocalModelClient().subscribe;
