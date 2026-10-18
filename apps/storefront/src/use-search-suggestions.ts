import { useCallback, useEffect, useRef, useState } from "react";
import { parseValidatedSearchSuggestions } from "./validated-search-suggestions";

/** Only the service owner's verified public list; never fabricated browsing history. */
export function useSearchSuggestions() {
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const current = useRef<AbortController | null>(null);
  const refresh = useCallback(() => {
    current.current?.abort();
    const controller = new AbortController();
    current.current = controller;
    fetch("/search-suggestions.json", { credentials: "omit", cache: "no-store", signal: controller.signal })
      .then(response => response.ok ? response.json() : null)
      .then(value => { if (!controller.signal.aborted) setSuggestions(parseValidatedSearchSuggestions(value)); })
      .catch(() => { if (!controller.signal.aborted) setSuggestions([]); });
  }, []);
  useEffect(() => { refresh(); return () => current.current?.abort(); }, [refresh]);
  return { suggestions, refresh };
}
