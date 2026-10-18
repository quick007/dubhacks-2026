import assert from "node:assert/strict";
import { test } from "node:test";
import { latencySeries, llmRun, median, medianInterval } from "./latency";

void test("median of odd and even samples", () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 3, 2]), 2.5);
});

void test("the median's interval holds the median and stays inside the sample", () => {
  const values = [5, 1, 9, 3, 7, 2, 8, 4, 6];
  const [low, high] = medianInterval(values);
  assert.ok(low <= median(values) && median(values) <= high);
  assert.ok(low >= 1 && high <= 9);
  assert.deepEqual(medianInterval(values), [low, high]);
});

void test("every series times the same requests, and LLM counts match their ranks", () => {
  for (const series of latencySeries()) assert.equal(series.ms.length, series.cases);
  for (const model of llmRun.models) {
    assert.equal(
      model.requests.filter((request) => request.valid && request.rank === 1).length,
      model.topOne,
    );
  }
});
