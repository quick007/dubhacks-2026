import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { choiceExamples } from "./shopping-training.mjs";
function fixture() {
  const sessionId = "00000000-0000-4000-8000-000000000002";
  const bucket = createHash("sha256").update(`fieldwork.session-split.v1/${sessionId}`).digest()[0] % 10;
  const split = bucket === 8 ? "validation" : bucket === 9 ? "test" : "train";
  return [
    { eventId: "view", sequence: 2, action: "catalog-view", data: { viewId: "catalog", products: [{ productId: "mug", variantId: "mug-sage", position: 0 }, { productId: "pack", variantId: "pack-sand", position: 1 }] } },
    { eventId: "selection", sequence: 4, action: "selection", data: { viewId: "catalog", productId: "mug", variantId: "mug-sage" } },
    { eventId: "impression", sequence: 3, action: "product-view", data: { viewId: "catalog", productId: "pack", variantId: "pack-sand" } },
  ].map((observation) => ({ schema: "fieldwork.training-observation.v1", split,
    provenance: { splitVersion: "fieldwork.session-split.v1", sessionId, eventId: observation.eventId, batchId: "batch", sourceSha256: "digest", classification: { environment: "production", traffic: "real" } }, observation }));
}
test("joins out-of-order batches and leaves every nonclick unlabeled", () => {
  const rows = fixture(); const examples = choiceExamples([rows[1], rows[2], rows[0]]);
  assert.equal(examples.length, 1); assert.deepEqual(examples[0].negativeLabels, []);
  assert.equal(examples[0].chosen.productId, "mug"); assert.equal(examples[0].candidates.length, 2);
  assert.equal(examples[0].realPurchaseObserved, false);
});
test("synthetic requires explicit second-stage flag", () => {
  const rows = fixture(); rows.forEach((r) => { r.provenance.classification = { environment: "test", traffic: "synthetic" }; });
  assert.equal(choiceExamples(rows).length, 0); assert.equal(choiceExamples(rows, { allowSynthetic: true }).length, 1);
});
test("variant selection preserves rendered and newly chosen variant identities", () => {
  const rows=fixture();rows[1].observation.data.variantId="mug-cobalt";
  const examples=choiceExamples(rows);assert.equal(examples.length,1);
  assert.equal(examples[0].candidates[0].variantId,"mug-sage");
  assert.equal(examples[0].chosen.variantId,"mug-cobalt");
  assert.equal(examples[0].choiceScope,"product-with-observed-variant-change");
});
test("duplicate identities and cross-session split mismatch are rejected", () => {
  const rows = fixture(); assert.throws(() => choiceExamples([...rows, rows[0]]), /Duplicate/);
  rows[1].split = rows[1].split === "test" ? "train" : "test"; assert.throws(() => choiceExamples(rows), /split/);
});
test("missing, later, conflicting and unrelated catalog exposures create no labels", () => {
  assert.equal(choiceExamples(fixture().slice(1)).length, 0);
  const rows = fixture(); rows[0].observation.sequence = 9; assert.equal(choiceExamples(rows).length, 0);
  const original = fixture(); const conflict = structuredClone(original[0]); conflict.observation.eventId = "conflict"; conflict.provenance.eventId = "conflict";
  conflict.observation.data.products = []; assert.equal(choiceExamples([...original, conflict]).length, 0);
  const unmatched = fixture(); unmatched[1].observation.data.productId = "unshown"; assert.equal(choiceExamples(unmatched).length, 0);
});
