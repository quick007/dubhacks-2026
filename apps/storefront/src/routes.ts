export type StorefrontPage = "shop" | "cart" | "confirmed";
export type StorefrontRoute = {
  page: StorefrontPage;
  query: string;
  collectionId: string;
};

export const DEFAULT_ROUTE: StorefrontRoute = { page: "shop", query: "", collectionId: "all" };
/** Matches the search input's maxLength so a shared link cannot inject an unbounded query. */
const MAX_QUERY_LENGTH = 200;

/** The hash is the shareable page state: #collection?q=…&c=…, #cart or #confirmed. */
export function parseRoute(hash: string): StorefrontRoute {
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  const [name = "", search = ""] = raw.split("?", 2);
  const params = new URLSearchParams(search);
  return {
    page: name === "cart" || name === "confirmed" ? name : DEFAULT_ROUTE.page,
    query: (params.get("q") ?? "").trim().slice(0, MAX_QUERY_LENGTH),
    collectionId: params.get("c") || DEFAULT_ROUTE.collectionId,
  };
}

export function routeHash(route: StorefrontRoute): string {
  const params = new URLSearchParams();
  if (route.query) params.set("q", route.query);
  if (route.collectionId !== DEFAULT_ROUTE.collectionId) params.set("c", route.collectionId);
  const search = params.toString();
  const name = route.page === "shop" ? "collection" : route.page;
  return `#${name}${search ? `?${search}` : ""}`;
}
