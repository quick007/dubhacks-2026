import assert from "node:assert/strict";
import { test } from "node:test";
import { latencySeries, linearAxisTicks, linearScale, llmRun, median, medianInterval } from "./latency";

void test("scatter axis includes every raw request, interval endpoint and client floor", () => {
  const series = latencySeries();
  const displayed = series.flatMap((item) => [...item.ms, ...medianInterval(item.ms),
    ...(item.floorMs === undefined ? [] : [item.floorMs])]);
  const ticks = linearAxisTicks(series.flatMap((item) => [...item.ms,
    ...(item.floorMs === undefined ? [] : [item.floorMs])]));
  assert.equal(Math.max(...displayed), 8438.18690999999);
  assert.deepEqual(ticks, [0, 2000, 4000, 6000, 8000, 9000]);
  for (const value of displayed) assert.ok(value >= 0 && value < ticks.at(-1)!);
});

void test("study axis includes span endpoints and all displayed comparison medians", () => {
  const displayed = [100, 250, 1000, 3000, ...latencySeries().map((item) => median(item.ms))];
  const ticks = linearAxisTicks(displayed);
  assert.deepEqual(ticks, [0, 2000, 4000, 6000, 7000]);
  for (const value of displayed) assert.ok(value >= 0 && value < ticks.at(-1)!);
});

void test("axis bounds follow outliers rather than their median and reject missing evidence", () => {
  assert.equal(linearAxisTicks([1, 2, 25_000]).at(-1), 26_000);
  assert.throws(() => linearAxisTicks([]));
  assert.throws(() => linearAxisTicks([NaN]));
});

void test("linear timing axis starts at zero and preserves proportional distances", () => {
  const x = linearScale(0, 20_000);
  assert.equal(x(0), 0);
  assert.equal(x(200), 1);
  assert.equal(x(4000), 20);
  assert.equal(x(10_000), 50);
  assert.equal(x(20_000), 100);
  assert.equal(x(-1), 0);
  assert.equal(x(25_000), 100);
});

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
