import assert from "node:assert/strict";
import { test } from "node:test";
import assumptions from "../fieldwork/illustrative-scenario.json";
import { illustrativeScenario } from "./illustrative-scenario";

test("illustration subtracts the displayed rounded assumptions, not a measured net effect", () => {
  assert.deepEqual(illustrativeScenario(assumptions.reportId, assumptions.reportSha256, 4235, 4395), {
    observedPercent: 3.78, assumedLossPercent: 0.52, remainderPercentagePoints: 3.26,
    assumedLatencyMs: 86, measuredNetRevenue: false,
  });
  assert.equal(assumptions.latencyInputMeasuredAddedPageDelay, false);
  assert.equal(assumptions.kind, "cross-experiment-assumption");
});

test("a different report or undefined baseline cannot inherit the fixed cohort assumptions", () => {
  assert.equal(illustrativeScenario("other", assumptions.reportSha256, 4235, 4395), null);
  assert.equal(illustrativeScenario(assumptions.reportId, "0".repeat(64), 4235, 4395), null);
  assert.equal(illustrativeScenario(assumptions.reportId, assumptions.reportSha256, 0, 4395), null);
  assert.equal(illustrativeScenario(assumptions.reportId, assumptions.reportSha256, NaN, 4395), null);
});
