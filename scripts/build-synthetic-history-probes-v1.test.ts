import test from "node:test";
import assert from "node:assert/strict";
import { buildHistoryOnlyProbes } from "./build-synthetic-history-probes-v1.ts";
import { buildPublicDataset } from "./build-synthetic-history-v1.ts";
test("ten matched pairs change only one observed purchase, with all other public input fixed", () => {
  const { public: probes, labels } = buildHistoryOnlyProbes(); const training = buildPublicDataset();
  assert.equal(probes.pairs.length, 10); assert.equal(labels.labels.length, 10);
  for (const pair of probes.pairs) {
    const a = pair.profileA.request; const b = pair.profileB.request;
    assert.equal(a.query, b.query); assert.equal(a.itemBudget, b.itemBudget); assert.deepEqual(a.persona, b.persona);
    assert.deepEqual(a.hardRequirements, b.hardRequirements); assert.deepEqual(a.publicHistory.slice(1), b.publicHistory.slice(1));
    assert.equal(a.publicHistory[0]!.eventId, b.publicHistory[0]!.eventId); assert.equal(a.publicHistory[0]!.occurredAt, b.publicHistory[0]!.occurredAt);
    assert.equal(a.publicHistory[0]!.eventType, "purchase"); assert.equal(b.publicHistory[0]!.eventType, "purchase");
    assert.equal(a.publicHistory[0]!.external, true); assert.equal(b.publicHistory[0]!.external, undefined);
    assert(!training.split.trainProfileIds.includes(pair.profileA.profileId)); assert(!training.split.heldoutProfileIds.includes(pair.profileB.profileId));
    const l = labels.labels.find(l => l.pairId === pair.pairId)!;
    assert.equal(l.profileA.relevanceGrades[l.changedPurchaseProductId], 3); assert.equal(l.profileB.relevanceGrades[l.changedPurchaseProductId], 0);
    assert(l.profileB.alreadyOwnedProductIds.includes(l.changedPurchaseProductId)); assert(Object.values(l.profileB.relevanceGrades).includes(3));
  }
});
