import assert from "node:assert/strict";
import { test } from "node:test";
import { modelRequest, modelRun, ordinal, product } from "./model";

void test("summary counts match the per-request ranks they summarize", () => {
  const { requests, arms, cases, patience } = modelRun;
  assert.equal(requests.length, cases);
  assert.equal(requests.filter((request) => request.modelRank === 1).length, arms.learned.topOne);
  assert.equal(requests.filter((request) => request.modelRank <= 6).length, arms.learned.topSix);
  const defaultRank = (request: (typeof requests)[number]) => product(request.target).defaultRank;
  assert.equal(requests.filter((request) => defaultRank(request) === 1).length, arms.prior.topOne);
  assert.equal(
    requests.filter((request) => defaultRank(request) <= patience).length,
    arms.prior.topSix,
  );
});

void test("each request's top six starts with the model's pick and holds known products", () => {
  for (const request of modelRun.requests) {
    assert.equal(request.top.length, 6);
    assert.equal(request.top[0].id === request.target, request.modelRank === 1);
    for (const item of request.top) assert.ok(product(item.id));
  }
  assert.equal(modelRequest("nobody"), undefined);
});

void test("ordinals", () => {
  assert.deepEqual([1, 2, 3, 4, 11, 12, 13, 16, 21].map(ordinal), [
    "1st",
    "2nd",
    "3rd",
    "4th",
    "11th",
    "12th",
    "13th",
    "16th",
    "21st",
  ]);
});
