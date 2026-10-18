import { useCallback, useEffect, useState } from "react";
import { parseRoute, routeHash, type StorefrontRoute } from "./routes";

/**
 * Keeps page state in the hash so views are shareable, browser Back/Forward
 * step through the shop, and captured runs record a meaningful URL.
 */
export function useRoute() {
  const [route, setRoute] = useState<StorefrontRoute>(() => parseRoute(window.location.hash));

  useEffect(() => {
    const sync = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);

  // Canonicalize the first load (for example /#collection?q=…) without adding history.
  useEffect(() => {
    const canonical = routeHash(parseRoute(window.location.hash));
    if (window.location.hash !== canonical) {
      window.history.replaceState(null, "", canonical);
    }
  }, []);

  // A committed search should reveal its results, including repeat submissions
  // and history navigation. Wait for the shop section to exist after render.
  useEffect(() => {
    if (route.page === "shop" && route.query) {
      document.getElementById("collection")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [route]);

  const navigate = useCallback((next: StorefrontRoute, options?: { replace?: boolean }) => {
    const hash = routeHash(next);
    if (window.location.hash !== hash) {
      if (options?.replace) window.history.replaceState(null, "", hash);
      else window.history.pushState(null, "", hash);
    }
    setRoute(next);
  }, []);

  return { route, navigate };
}
