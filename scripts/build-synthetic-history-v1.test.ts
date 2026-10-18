import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { createHash } from "node:crypto";
import { buildPublicDataset, buildLabels, auditDataset, loadCatalog, validateCatalogHistory, writeDataset, OUT, ROOT } from "./build-synthetic-history-v1.ts";
import { validatePublicHistoryRequest } from "../packages/contracts/src/synthetic-history-v1.ts";
const dataset = buildPublicDataset(); const labels = buildLabels(dataset);
const read = (name: string) => JSON.parse(readFileSync(resolve(OUT, name), "utf8"));

test("frozen200public profiles and10demo people reproduce byte-equivalent data", () => {
  assert.deepEqual(read("profiles.public.json"), dataset.publicDataset);
  assert.deepEqual(read("demo-people.json").people, dataset.people);
  assert.equal(dataset.profiles.length, 200); assert.equal(dataset.people.length, 10);
  assert.equal(new Set(dataset.profiles.map(p => p.profileId)).size, 200);
  assert.equal(new Set(dataset.people.map(p => p.displayName)).size, 10);
});
test("session groups, family templates and demos never cross split", () => {
  const { split } = dataset;
  assert.equal(split.trainGroupIds.length, 80); assert.equal(split.heldoutGroupIds.length, 20);
  assert.equal(split.trainProfileIds.length, 160); assert.equal(split.heldoutProfileIds.length, 40);
  assert.equal(split.trainFamilyIds.length, 20); assert.equal(split.heldoutFamilyIds.length, 5);
  for (const id of split.trainFamilyIds) assert(!split.heldoutFamilyIds.includes(id));
  for (const id of split.trainGroupIds) assert(!split.heldoutGroupIds.includes(id));
  for (const group of split.groups) {
    assert.equal(group.profileIds.length, 2);
    for (const id of group.profileIds) assert((group.split === "train" ? split.trainProfileIds : split.heldoutProfileIds).includes(id));
  }
  for (const id of split.demoGroupIds) assert(![...split.trainGroupIds, ...split.heldoutGroupIds].includes(id));
});
test("all authored histories are ordered4–12 events and use realcatalog bindings", () => {
  const catalog = loadCatalog();
  for (const request of [...dataset.profiles.map(p => p.request), ...dataset.people.map(p => p.request)]) {
    validateCatalogHistory(request, catalog);
    assert(request.publicHistory.length >= 4 && request.publicHistory.length <= 12);
    assert(request.publicHistory.every(e => e.source === "demo-seed"));
    const externals = request.publicHistory.filter(e => e.external);
    assert.equal(externals.length, 1); assert.equal(externals[0]!.eventType, "purchase");
    assert.equal(externals[0]!.productId, undefined); assert.equal(externals[0]!.variantId, undefined);
  }
  const bob = dataset.people.find(p => p.id === "bob")!;
  assert.match(bob.request.publicHistory[0]!.itemName, /sofa/);
});
test("private outcome fields and fake external SKUs are rejected from public input", () => {
  const request = dataset.profiles[0]!.request;
  assert.throws(() => validatePublicHistoryRequest({ ...request, relevanceGrades: { blanket: 3 } }));
  assert.throws(() => validatePublicHistoryRequest({ ...request, persona: { ...request.persona, desiredProductId: "blanket" } }));
  const injected = structuredClone(request); (injected.publicHistory[0] as unknown as Record<string, unknown>).chosenProductId = "blanket";
  assert.throws(() => validatePublicHistoryRequest(injected));
  const fake = structuredClone(request); fake.publicHistory[0]!.productId = "sofa";
  assert.throws(() => validatePublicHistoryRequest(fake));
  const stale = structuredClone(request); stale.publicHistory[1]!.selectedColor = "Invented purple";
  assert.throws(() => validateCatalogHistory(stale, dataset.catalog));
});
test("all labels are separate, multi-positive, budget-compatible and suppress owned duplicates", () => {
  assert.deepEqual(read("labels.train.json").labels, labels.train);
  assert.deepEqual(read("labels.heldout.json").labels, labels.heldout);
  assert.equal(labels.train.length, 160); assert.equal(labels.heldout.length, 40);
  const byId = new Map(dataset.profiles.map(p => [p.profileId, p]));
  for (const l of [...labels.train, ...labels.heldout]) {
    const request = byId.get(l.profileId)!.request;
    assert.equal(Object.keys(l.relevanceGrades).length, 16);
    assert(Object.values(l.relevanceGrades).filter(x => x > 0).length >= 2);
    for (const p of dataset.catalog) {
      if (l.relevanceGrades[p.id]! > 0) assert(p.price <= request.itemBudget!);
      if (l.alreadyOwnedProductIds.includes(p.id)) assert.equal(l.relevanceGrades[p.id], 0);
    }
    for (const id of l.preferredVariantIds) assert(l.acceptableVariantIds.includes(id));
    for (const id of l.acceptableVariantIds) assert(dataset.catalog.some(p => l.relevanceGrades[p.id]! > 0 && p.variants.some(v => v.id === id)));
  }
  const audit = auditDataset(dataset, labels);
  assert(Object.values(audit.trainTopGradeByProduct).every(x => x >= 8));
  assert.equal(audit.globalPriorTop1UpperBound, .1); assert.equal(audit.humanConversionClaim, false);
});
test("same broad queries coexist with distinct public people and no hidden target keys", () => {
  assert.equal(new Set(dataset.people.map(p => p.request.query)).size, 1);
  assert.equal(new Set(dataset.people.map(p => JSON.stringify(p.request.persona.background))).size, 10);
  for (const p of dataset.profiles) {
    assert.deepEqual(Object.keys(p.request).sort(), ["hardRequirements", "itemBudget", "persona", "publicHistory", "query"]);
    assert(!Object.keys(p.request.persona).some(k => /target|desired|gold|score/i.test(k)));
  }
  const constrained = { ...dataset.profiles[0]!.request, hardRequirements: [{ attribute: "color", value: "green" }] };
  assert.deepEqual(validatePublicHistoryRequest(constrained).hardRequirements, constrained.hardRequirements);
});
test("manifest hashes bind public-before-label freeze, source and every published dataset artifact", () => {
  const hash = (p: string) => createHash("sha256").update(readFileSync(p)).digest("hex");
  const manifest = read("manifest.json"); const frozen = read("PUBLIC_SPLIT_FROZEN.json");
  for (const [name, expected] of Object.entries(manifest.files)) assert.equal(hash(resolve(OUT, name)), expected);
  for (const [name, expected] of Object.entries(manifest.sourceSha256)) assert.equal(hash(resolve(ROOT, name)), expected);
  for (const [name, expected] of Object.entries(frozen.files)) assert.equal(hash(resolve(OUT, name)), expected);
  assert.equal(frozen.publicAndSplitPublishedBeforeLabels, true);
});
test("builder reproduces all hashes to unused output and refuses overwriting frozen artifacts", () => {
  const dir = mkdtempSync(resolve(tmpdir(), "synthetic-history-v1-"));
  try {
    const result = writeDataset(dir); assert.deepEqual(result.hashes, read("manifest.json").files);
    assert.throws(() => writeDataset(dir), /Refuse overwriting/);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
