import type { Run } from "@storefront/contracts";

export type Adapter = Run["adapter"];
export type Outcome = "passed" | "failed" | "error" | "pending";

export function cellKey(adapter: Adapter, personaId: string, variantId: string) {
  return `${adapter}:${personaId}:${variantId}`;
}

export function outcomeOf(run: Run): Outcome {
  if (run.status === "queued" || run.status === "running") return "pending";
  if (run.status === "failed") return "error";
  return run.evaluation?.taskCompleted ? "passed" : "failed";
}

/** Groups runs by adapter, persona and variant, newest first within each cell. */
export function groupRuns(runs: Run[]) {
  const cells = new Map<string, Run[]>();
  for (const run of runs) {
    const key = cellKey(run.adapter, run.personaId, run.variantId);
    cells.set(key, [...(cells.get(key) ?? []), run]);
  }
  for (const list of cells.values()) list.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  return cells;
}

/** Server snapshot plus optimistic queued runs it doesn't know yet (its request started before they were submitted). */
export function mergeRuns(server: Run[], local: Run[]) {
  const known = new Set(server.map((run) => run.id));
  return [...local.filter((run) => run.status === "queued" && !known.has(run.id)), ...server];
}
