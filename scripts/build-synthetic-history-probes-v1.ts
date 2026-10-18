/** Diagnostic-only matched purchases. Never imported by training or public UI. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { OUT, ROOT, VERSION, buildPublicDataset, validateCatalogHistory } from "./build-synthetic-history-v1.ts";

export function buildHistoryOnlyProbes() {
  const data = buildPublicDataset();
  const pairs = data.people.map((person, index) => {
    const family = data.families[index]!;
    const primary = data.catalog.find(p => family.scores[p.id] === 3)!;
    const pairId = `history-only-${String(index + 1).padStart(2, "0")}`;
    const a = structuredClone(person.request); const b = structuredClone(a);
    const event = b.publicHistory[0]!; const variant = primary.variants[0]!;
    b.publicHistory[0] = { eventId: event.eventId, eventType: "purchase", itemName: primary.name, category: primary.category, selectedColor: variant.colorName, occurredAt: event.occurredAt, source: "demo-seed", productId: primary.id, variantId: variant.id };
    validateCatalogHistory(a, data.catalog); validateCatalogHistory(b, data.catalog);
    if (JSON.stringify({ ...a, publicHistory: undefined }) !== JSON.stringify({ ...b, publicHistory: undefined })) throw new Error("Matched public context differs");
    if (a.publicHistory.filter((e, i) => JSON.stringify(e) !== JSON.stringify(b.publicHistory[i])).length !== 1) throw new Error("Probe changes more than one history event");
    return { pairId, profileA: { profileId: `${pairId}-a`, request: a }, profileB: { profileId: `${pairId}-b`, request: b } };
  });
  const labels = pairs.map((pair, index) => {
    const family = data.families[index]!;
    const purchasedProductId = pair.profileB.request.publicHistory[0]!.productId!;
    const grades = (arm: "profileA" | "profileB") => {
      const request = pair[arm].request;
      const owned = request.publicHistory.filter(e => e.eventType === "purchase" && !e.external).map(e => e.productId!);
      const base = Object.fromEntries(data.catalog.map(p => [p.id, owned.includes(p.id) || p.price > request.itemBudget! ? 0 : (family.scores[p.id] ?? 0)]));
      const best = Math.max(...Object.values(base));
      if (best > 0 && best < 3) for (const id of Object.keys(base)) if (base[id] === best) base[id] = 3;
      return { profileId: pair[arm].profileId, relevanceGrades: base, alreadyOwnedProductIds: owned };
    };
    return { pairId: pair.pairId, changedPurchaseProductId: purchasedProductId, expectedHistoryEffect: "The newly purchased item should lose recommendation priority; plausible unowned complements retain priority. No request asks for a duplicate or replacement.", profileA: grades("profileA"), profileB: grades("profileB") };
  });
  return { public: { schemaVersion: VERSION, scope: "Ten diagnostic matched pairs, excluded from training/model selection. Background/query/color/budget/other events identical; only one observed purchase changes. Authored synthetic, not browser evidence.", pairs }, labels: { schemaVersion: VERSION, scope: "Private diagnostic judgments, read only after prediction freeze; not training labels.", labels } };
}

export function writeHistoryOnlyProbes() {
  const names = ["history-only-probes.public.json", "history-only-probes.labels.json", "history-only-probes.manifest.json"];
  if (names.some(n => existsSync(resolve(OUT, n)))) throw new Error("Refuse overwriting diagnostic probes");
  const value = buildHistoryOnlyProbes(); const dump = (name: string, data: unknown) => writeFileSync(resolve(OUT, name), JSON.stringify(data, null, 2) + "\n");
  dump(names[0]!, value.public); dump(names[1]!, value.labels);
  const sha = (p: string) => createHash("sha256").update(readFileSync(p)).digest("hex");
  const manifest = { schemaVersion: VERSION, trainAllowed: false, modelSelectionAllowed: false, pairs: 10, predictionFreezeRequiredBeforePrivateLabels: true, files: Object.fromEntries(names.slice(0, 2).map(n => [n, sha(resolve(OUT, n))])), sourceSha256: { "scripts/build-synthetic-history-probes-v1.ts": sha(resolve(ROOT, "scripts/build-synthetic-history-probes-v1.ts")) }, frozenBaseManifestSha256: sha(resolve(OUT, "manifest.json")) };
  dump(names[2]!, manifest); return manifest;
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) console.log(JSON.stringify(writeHistoryOnlyProbes(), null, 2));
