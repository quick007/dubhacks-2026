import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ExperimentSchema,
  VariantConfigSchema,
  EvaluationResultSchema,
  RunRequestSchema,
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
