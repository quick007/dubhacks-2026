import assert from "node:assert/strict";
import test from "node:test";
import type { Persona, Run } from "@storefront/contracts";
import { cellKey, groupRuns, mergeRuns, outcomeOf } from "./runs.ts";

const persona = (id: string): Persona => ({
  id,
  name: id,
  goal: "",
  split: "development",
  traits: [],
  history: [],
});

function run(
  id: string,
  personaId: string,
  variantId: string,
  score: number,
  createdAt: string,
): Run {
  return {
    id,
    experimentId: "e",
    baselineVariantId: "base",
    variantId,
    personaId,
    personaSplit: "development",
    seed: 1,
    config: { headline: "h", showShipping: false, recommendationOrder: "popular" },
    persona: persona(personaId),
    adapter: "mock",
    synthetic: true,
    contractVersion: 1,
    status: "completed",
    createdAt,
    events: [],
    artifacts: [],
    evaluation: {
      runId: id,
      synthetic: true,
      taskCompleted: score === 1,
      score,
      failureReasons: [],
      notes: [],
    },
    proposedChanges: [],
  };
}

void test("groups runs newest first per cell", () => {
  const cells = groupRuns([
    run("old", "a", "base", 0, "2026-01-01T00:00:00Z"),
    run("new", "a", "base", 1, "2026-01-02T00:00:00Z"),
  ]);
  assert.deepEqual(
    cells.get(cellKey("mock", "a", "base"))?.map((r) => r.id),
    ["new", "old"],
  );
});

void test("classifies outcomes", () => {
  const base = run("r", "a", "base", 0, "2026-01-01T00:00:00Z");
  assert.equal(outcomeOf(base), "failed");
  assert.equal(outcomeOf({ ...base, status: "running" }), "pending");
  assert.equal(outcomeOf({ ...base, status: "failed" }), "error");
});

void test("a poll that started before a submission keeps the optimistic queued runs", () => {
  const done = run("done", "a", "base", 1, "2026-01-01T00:00:00Z");
  const queued = { ...run("new", "a", "v", 0, "2026-01-02T00:00:00Z"), status: "queued" as const };
  const merged = mergeRuns([done], [queued, done]);
  assert.deepEqual(
    merged.map((r) => r.id),
    ["new", "done"],
  );
  assert.deepEqual(
    mergeRuns([{ ...queued, status: "completed" }, done], [queued, done]).map((r) => r.status),
    ["completed", "completed"],
  );
});
