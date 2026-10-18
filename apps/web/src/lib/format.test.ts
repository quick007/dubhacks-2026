import { test } from "node:test";
import assert from "node:assert/strict";
import { describePatch, percent, splitLabel } from "./format";
void test("comparison percentages and split labels remain explicit", () => {
  assert.equal(percent(0.815), "82%");
  assert.equal(percent(0), "0%");
  assert.equal(splitLabel("held-out"), "Held-out");
  assert.equal(splitLabel("development"), "Development");
});

void test("describes a config patch as before to after", () => {
  assert.deepEqual(
    describePatch(
      { headline: "h", showShipping: true, recommendationOrder: "popular" },
      { showAttributes: true },
    ),
    ["Details on cards: Hidden to Shown"],
  );
});
