export function parseValidatedSearchSuggestions(value: unknown): string[] {
  if (!value || typeof value !== "object") return [];
  const data = value as { version?: unknown; queries?: unknown };
  if (data.version !== "fieldwork.validated-search-suggestions.v1" || !Array.isArray(data.queries) || data.queries.length > 32) return [];
  if (data.queries.some(query => typeof query !== "string" || query.length === 0 || query.length > 200 || query.trim() !== query)) return [];
  return [...new Set(data.queries as string[])];
}
