import test from "node:test";
import assert from "node:assert/strict";
import fs, { readFileSync, mkdtempSync, rmSync } from "node:fs";
import { syncBuiltinESMExports } from "node:module";
import { resolve } from "node:path";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";
import { buildContrasts, writeContrasts, makeContrast, eligibleVariants, INPUTS, ROOT, type Product } from "./build-synthetic-history-train-contrasts-v2.ts";
const hash = (path: string) => createHash("sha256").update(readFileSync(path)).digest("hex");
const data = buildContrasts();
const catalog: Product[] = JSON.parse(readFileSync(resolve(ROOT, INPUTS.catalog), "utf8")).products;

test("160 source sessions become160 paired contrasts in the same80TRAIN groups", () => {
  assert.equal(data.profiles.length, 160); assert.equal(data.contrasts.length, 160); assert.equal(data.pairs.length, 160);
  assert.equal(new Set(data.profiles.map(p => p.userGroupId)).size, 80);
  assert.deepEqual(new Set(data.contrasts.map(p => p.userGroupId)), new Set(data.profiles.map(p => p.userGroupId)));
  assert.equal(new Set([...data.profiles, ...data.contrasts].map(p => p.profileId)).size, 320);
  assert.equal(new Set([...data.profiles, ...data.contrasts].map(p => p.sessionId)).size, 320);
  assert.equal(Object.keys(data.audit.changedPurchaseByProduct).length, 16);
  assert(Object.values(data.audit.changedPurchaseByProduct).every(n => n === 8 || n === 16));
});
test("every contrast preserves full publiccontext and changes exactly one factual purchase", () => {
  for (const [n, pair] of data.pairs.entries()) {
    const a = data.profiles[n]!; const b = data.contrasts[n]!;
    assert.equal(pair.sourceProfileId, a.profileId); assert.equal(pair.contrastProfileId, b.profileId);
    assert.deepEqual({ ...a.request, publicHistory: null }, { ...b.request, publicHistory: null });
    assert.equal(a.userGroupId, b.userGroupId);
    assert.equal(a.request.publicHistory.filter((e, i) => JSON.stringify(e) !== JSON.stringify(b.request.publicHistory[i])).length, 1);
    const event = b.request.publicHistory[pair.replacedEventIndex]!; const old = a.request.publicHistory[pair.replacedEventIndex]!;
    assert.equal(old.external, true); assert.equal(old.eventType, "purchase"); assert.equal(event.eventType, "purchase");
    assert.equal(event.occurredAt, old.occurredAt); assert.notEqual(event.eventId, old.eventId); assert.equal(event.external, undefined);
    const product = catalog.find(p => p.id === event.productId)!; const variant = product.variants.find(v => v.id === event.variantId)!;
    assert.equal(event.itemName, product.name); assert.equal(event.category, product.category); assert.equal(event.selectedColor, variant.colorName);
    assert(product.price <= b.request.itemBudget!);
  }
});
test("newlyownedtarget becomeszero and only originaleligibleunownedpositivealternatives gain grade", () => {
  for (const [n, pair] of data.pairs.entries()) {
    const a = data.labels[n]!; const b = data.contrastLabels[n]!;
    assert.equal(a.relevanceGrades[pair.newlyOwnedProductId], 3); assert.equal(b.relevanceGrades[pair.newlyOwnedProductId], 0);
    assert(b.alreadyOwnedProductIds.includes(pair.newlyOwnedProductId));
    assert.equal(Object.values(b.relevanceGrades).filter(n => n > 0).length, 2);
    for (const p of catalog) {
      if (b.relevanceGrades[p.id]! > 0) { assert(a.relevanceGrades[p.id]! > 0); assert(!b.alreadyOwnedProductIds.includes(p.id)); assert(eligibleVariants(p, data.contrasts[n]!.request).length); }
      if (pair.promotedExistingPositiveProductIds.includes(p.id)) assert.equal(b.relevanceGrades[p.id], 3);
    }
    for (const id of b.preferredVariantIds) { assert(a.preferredVariantIds.includes(id)); assert(b.acceptableVariantIds.includes(id)); }
    for (const id of b.acceptableVariantIds) assert(catalog.some(p => b.relevanceGrades[p.id]! > 0 && eligibleVariants(p, data.contrasts[n]!.request).some(v => v.id === id)));
  }
});
test("unknownconstraints, ineligibleprimary, inventedcolor and labelidentityconflicts fail", () => {
  const p = data.profiles[0]!; const l = data.labels[0]!;
  const constraint = structuredClone(p); constraint.request.hardRequirements = [{ attribute: "unknown-constraint", value: "keep me" }];
  assert.throws(() => makeContrast(constraint, l, catalog), /Unknown hard requirement/);
  const budget = structuredClone(p); budget.request.itemBudget = 1; assert.throws(() => makeContrast(budget, l, catalog), /eligible/);
  const fake = structuredClone(p); fake.request.publicHistory[1]!.selectedColor = "Invented purple"; assert.throws(() => makeContrast(fake, l, catalog), /catalog factual/);
  assert.throws(() => makeContrast(p, { ...l, profileId: "wrong" }, catalog), /identity differ/);
  const noalternative = structuredClone(l); for (const id of Object.keys(noalternative.relevanceGrades)) if (noalternative.relevanceGrades[id] !== 3) noalternative.relevanceGrades[id] = 0;
  assert.throws(() => makeContrast(p, noalternative, catalog), /previously-positive/);
});
test("futurecolorhardrequirementfiltersvariantswithoutdroppingit", () => {
  const p = data.profiles[0]!.request; const product = catalog[0]!;
  const constrained = { ...p, hardRequirements: [{ attribute: "color", value: product.variants[1]!.colorName }] };
  assert.deepEqual(eligibleVariants(product, constrained), [product.variants[1]]);
  assert.equal(eligibleVariants(product, { ...p, itemBudget: 1 }).length, 0);
});
test("runtimefile-readguard permits only four TRAIN/provenance/catalog inputs", () => {
  const allowed = new Set(Object.values(INPUTS).map(p => resolve(ROOT, p))); const touched: string[] = [];
  const original = fs.readFileSync;
  fs.readFileSync = ((path: Parameters<typeof fs.readFileSync>[0], ...args: unknown[]) => {
    assert.equal(typeof path, "string"); const absolute = resolve(path as string);
    assert(allowed.has(absolute), "Forbidden nonTRAIN data read: " + absolute); touched.push(absolute);
    return Reflect.apply(original, fs, [path, ...args]);
  }) as typeof fs.readFileSync;
  syncBuiltinESMExports();
  try { const guarded = buildContrasts(); assert.equal(guarded.contrasts.length, 160); assert.equal(touched.length, 4); }
  finally { fs.readFileSync = original; syncBuiltinESMExports(); }
});
test("unusedoutputreproduceshashesandcannotoverwrite; allsourceTRAINbytesstayfixed", () => {
  const before = Object.fromEntries(Object.values(INPUTS).map(p => [p, hash(resolve(ROOT, p))]));
  const a = mkdtempSync(resolve(tmpdir(), "history-contrast-a-")); const b = mkdtempSync(resolve(tmpdir(), "history-contrast-b-"));
  try {
    const first = writeContrasts(a); const second = writeContrasts(b);
    assert.deepEqual(first.files, second.files); assert.equal(first.manifestSha256, second.manifestSha256);
    const frozen = JSON.parse(readFileSync(resolve(a, "TRAIN_PUBLIC_FROZEN.json"), "utf8"));
    assert.equal(frozen.trainingLabelsWereReadToAuthorCounterfactualPurchases, true);
    for (const [name, expected] of Object.entries(first.files)) assert.equal(hash(resolve(a, name)), expected);
    assert.throws(() => writeContrasts(a), /Refuse overwriting/);
  } finally { rmSync(a, { recursive: true, force: true }); rmSync(b, { recursive: true, force: true }); }
  for (const [p, expected] of Object.entries(before)) assert.equal(hash(resolve(ROOT, p)), expected);
});
