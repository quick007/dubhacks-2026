import assert from "node:assert/strict";
import test from "node:test";
import { allSlides, slideFromHash, stepFor, visibleSlides } from "./deck.ts";

void test("slides without data are skipped, order is kept", () => {
  const ids = visibleSlides({ runs: false, proposal: false, result: false });
  // The session grid is where the first round starts, so it shows before any run exists.
  assert.deepEqual(ids, ["title", "meet", "sessions"]);
  assert.deepEqual(visibleSlides({ runs: true, proposal: true, result: true }), allSlides);
});

void test("hash picks the slide, unknown hashes start at the beginning", () => {
  const ids = visibleSlides({ runs: true, proposal: false, result: true });
  assert.equal(slideFromHash("#retest", ids), ids.indexOf("retest"));
  assert.equal(slideFromHash("#fix", ids), 0);
  assert.equal(slideFromHash("", ids), 0);
});

void test("keys move one slide and stop at the ends", () => {
  assert.equal(stepFor("ArrowRight", 0, 5), 1);
  assert.equal(stepFor(" ", 4, 5), undefined);
  assert.equal(stepFor("ArrowLeft", 0, 5), undefined);
  assert.equal(stepFor("End", 1, 5), 4);
  assert.equal(stepFor("Home", 3, 5), 0);
  assert.equal(stepFor("a", 2, 5), undefined);
});
