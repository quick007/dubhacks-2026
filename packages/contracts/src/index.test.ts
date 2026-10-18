import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ExperimentSchema,
  VariantConfigSchema,
  EvaluationResultSchema,
  RunRequestSchema,
  RunSchema,
  StepEventSchema,
} from "./index.js";
void test("structured variants reject arbitrary code and unknown fields", () => {
  assert.equal(
    VariantConfigSchema.safeParse({
      headline: "Shop",
      showShipping: true,
      recommendationOrder: "budget",
      code: "eval(1)",
    }).success,
    false,
  );
});
void test("split is disjoint and baseline is recorded", () => {
  const experiment = {
    id: "e",
    name: "Test",
    baselineVariantId: "b",
    variantIds: ["b", "v"],
    developmentPersonaIds: ["p"],
    heldOutPersonaIds: ["p"],
    seed: 42,
    createdAt: new Date().toISOString(),
    contractVersion: 1,
  };
  assert.equal(ExperimentSchema.safeParse(experiment).success, false);
  assert.equal(
    ExperimentSchema.safeParse({ ...experiment, heldOutPersonaIds: ["h"] }).success,
    true,
  );
});
void test("evaluations must explicitly be synthetic and adapters are restricted", () => {
  assert.equal(
    EvaluationResultSchema.safeParse({
      runId: "r",
      synthetic: false,
      taskCompleted: true,
      score: 1,
      failureReasons: [],
      notes: [],
    }).success,
    false,
  );
  assert.equal(
    RunRequestSchema.safeParse({
      experimentId: "e",
      variantId: "v",
      personaId: "p",
      adapter: "remote",
    }).success,
    false,
  );
});

void test("findability fields round-trip and older runs still parse", () => {
  const at = new Date().toISOString();
  const step = {
    id: "s",
    runId: "r",
    sequence: 0,
    at,
    kind: "action",
    message: "Searched for navy jacket",
    page: { kind: "search", label: "Search: navy jacket" },
    click: { x: 0.5, y: 0.2 },
    screenshotArtifactId: "a",
    search: { query: "navy jacket", resultCount: 0 },
  };
  const run = {
    id: "r",
    experimentId: "e",
    baselineVariantId: "b",
    variantId: "b",
    personaId: "p",
    personaSplit: "development",
    seed: 1,
    config: {
      headline: "h",
      showShipping: false,
      recommendationOrder: "popular",
      showAttributes: true,
    },
    persona: {
      id: "p",
      name: "P",
      goal: "g",
      split: "development",
      traits: [],
      history: [],
      bio: "Walks to class through Seattle rain.",
      task: {
        query: "navy jacket",
        budget: 150,
        requirements: [
          { id: "color", label: "Navy", kind: "hard", attribute: "color", value: "navy" },
        ],
      },
    },
    adapter: "mock",
    synthetic: true,
    contractVersion: 1,
    status: "completed",
    createdAt: at,
    events: [step],
    artifacts: [],
    evaluation: {
      runId: "r",
      synthetic: true,
      taskCompleted: false,
      score: 0,
      failureReasons: [],
      notes: [],
      findability: {
        outcome: "abandoned",
        requirements: [{ id: "color", met: false }],
        steps: 1,
        optimalSteps: 6,
        searches: 1,
        emptySearches: 1,
        revisits: 0,
        abandonedAt: "s",
        orderValue: 0,
      },
    },
    proposedChanges: [],
  };
  assert.deepEqual(RunSchema.parse(run), run);
  const { task: _task, bio: _bio, ...oldPersona } = run.persona;
  const { findability: _findability, ...oldEvaluation } = run.evaluation;
  assert.equal(
    RunSchema.safeParse({
      ...run,
      persona: oldPersona,
      evaluation: oldEvaluation,
      events: [{ id: "s", runId: "r", sequence: 0, at, kind: "status", message: "running" }],
    }).success,
    true,
  );
  assert.equal(StepEventSchema.safeParse({ ...step, click: { x: 2, y: 0 } }).success, false);
});
