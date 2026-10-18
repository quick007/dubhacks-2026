import type { RunRequest } from "@storefront/contracts";

export async function api(path: string, body?: RunRequest): Promise<unknown> {
  const response = await fetch(
    `/api${path}`,
    body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : undefined,
  );
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    // The dev proxy answers with an HTML page while the API is still starting.
    throw new Error(`The local API isn't answering yet (HTTP ${response.status}). Retrying.`);
  }
  if (!response.ok)
    throw new Error(
      typeof data === "object" && data !== null && "error" in data
        ? String(data.error)
        : `Request failed (${response.status})`,
    );
  return data;
}

export function artifactUrl(id: string) {
  return `/api/artifacts/${encodeURIComponent(id)}`;
}
