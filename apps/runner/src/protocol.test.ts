import assert from "node:assert/strict";
import { test } from "node:test";
import { parseRunnerInput, protocolFailure } from "./protocol.js";

const run = {
  id: "browser-test",
  experimentId: "demo-experiment",
  baselineVariantId: "baseline",
  variantId: "baseline",
  personaId: "shopper",
  personaSplit: "development",
  seed: 42,
  config: { headline: "A demo shop", showShipping: false, recommendationOrder: "popular" },
  persona: {
    id: "shopper",
    name: "Shopper",
    goal: "Complete a fake order",
    split: "development",
    traits: [],
    history: [],
  },
  adapter: "playwright",
  synthetic: true,
  contractVersion: 1,
  status: "running",
  createdAt: "2026-10-17T00:00:00.000Z",
  events: [],
  artifacts: [],
  proposedChanges: [],
};
void test("parses a frozen Run snapshot and absolute artifact directory", () => {
  const parsed = parseRunnerInput({ run, artifactDirectory: "/tmp/storefront-runner/test" });
  assert.equal(parsed.run.id, "browser-test");
  assert.equal(parsed.artifactDirectory, "/tmp/storefront-runner/test");
});
void test("rejects invalid snapshots, non-browser runs and nonabsolute paths", () => {
  assert.throws(() => parseRunnerInput({ run, artifactDirectory: "../artifacts" }), /absolute/);
  assert.throws(() => parseRunnerInput({ run, artifactDirectory: "/" }), /filesystem root/);
  assert.throws(
    () => parseRunnerInput({ run: { ...run, adapter: "mock" }, artifactDirectory: "/tmp/run" }),
    /playwright/,
  );
  assert.throws(() =>
    parseRunnerInput({ run: { ...run, synthetic: false }, artifactDirectory: "/tmp/run" }),
  );
  assert.throws(
    () => parseRunnerInput({ run, artifactDirectory: "/tmp/run", url: "https://example.com" }),
    /only run/,
  );
});
void test("fatal protocol failures retain the complete output shape", () => {
  const output = protocolFailure(new Error("Invalid request"));
  assert.equal(output.error, "Invalid request");
  assert.equal(output.result.taskCompleted, false);
  assert.deepEqual(output.artifacts, []);
  assert.deepEqual(output.events, [{ kind: "failure", message: "Invalid request" }]);
});
