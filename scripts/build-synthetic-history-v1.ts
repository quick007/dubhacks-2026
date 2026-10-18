/** Reproducible agent-authored synthetic records, never browser-observed behavior. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { validatePublicHistoryRequest, type PublicHistoryRequest, type SyntheticPublicProfile, type SyntheticDemoPerson, type PublicShoppingEvent } from "../packages/contracts/src/synthetic-history-v1.ts";

export const VERSION = "agent-authored-shopping-history.v1";
export const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
export const OUT = resolve(ROOT, "experiments/ranker/pretrained/synthetic_history_v1");
const CATALOG = "experiments/ranker/pretrained/store_catalog_v1/runtime_export.json";
const RECIPE = "scripts/synthetic-history-seed-v1.json";
const sha = (value: string) => createHash("sha256").update(value).digest("hex");
const read = (path: string) => JSON.parse(readFileSync(resolve(ROOT, path), "utf8"));
const json = (value: unknown) => JSON.stringify(value, null, 2) + "\n";
const sourcePins = [CATALOG, RECIPE, "apps/storefront/src/config.ts", "apps/storefront/src/catalog-expansion.ts", "apps/storefront/src/variants.ts", "packages/contracts/src/synthetic-history-v1.ts", "scripts/build-synthetic-history-v1.ts"];
const provenance = { kind: "agent-authored-synthetic", authoredAt: "2026-10-18T00:00:00.000Z", browserObserved: false, humanConversionClaim: false, purpose: "Illustrative public histories and synthetic relevance supervision; no real people or measured preferences." };
type Family = { id: string; split: "train" | "heldout"; background: string[]; anchor: [string, string]; history: string[]; owned: string; scores: Record<string, number>; budget: number; reason: string };
type Variant = { id: string; colorName: string };
type Product = { id: string; name: string; category: string; price: number; variants: Variant[] };
export type PrivateHistoryLabel = { profileId: string; relevanceGrades: Record<string, number>; preferredVariantIds: string[]; acceptableVariantIds: string[]; alreadyOwnedProductIds: string[]; rationale: string; judgmentSource: "agent-authored-synthetic-not-observed-choice" };

export function loadCatalog(): Product[] {
  const captured = read(CATALOG);
  const products = captured.products.map((p: Omit<Product, "variants">) => ({ id: p.id, name: p.name, category: p.category, price: p.price,
    variants: captured.variants.find((v: { productId: string }) => v.productId === p.id).colorVariants.map((v: { variant: Variant }) => ({ id: v.variant.id, colorName: v.variant.colorName })) }));
  if (products.length !== 16 || products.reduce((n: number, p: Product) => n + p.variants.length, 0) !== 48) throw new Error("Expected actual16products/48variants");
  return products;
}
export function validateCatalogHistory(request: PublicHistoryRequest, catalog: Product[]): void {
  validatePublicHistoryRequest(request);
  for (const event of request.publicHistory) {
    if (event.external) continue;
    const product = catalog.find(p => p.id === event.productId);
    const variant = product?.variants.find(v => v.id === event.variantId);
    if (!product || !variant || product.name !== event.itemName || product.category !== event.category || variant.colorName !== event.selectedColor)
      throw new Error("History item/variant/name/category differs from actual catalog");
  }
}
const palette = [
  { label: "warm earth tones", match: /Rust|Terracotta|Ochre|Amber|Tortoise|Clay|Oatmeal|Sand|Natural/ },
  { label: "navy and blue", match: /Navy|blue|Blue|Cobalt/ },
  { label: "green and natural shades", match: /Sage|Forest|Olive|Moss|Natural/ },
  { label: "quiet neutrals", match: /Graphite|Charcoal|Black|Ivory|Off-white|Cream|Oatmeal|Sand/ },
];
const settings = ["Lives in a compact apartment and usually shops for one practical addition at a time.", "Shares a home and likes belongings that are easy to use in more than one room.", "Keeps a small set of everyday things and prefers to avoid buying duplicates.", "Has recently reorganized a weekly routine and is filling in a few practical gaps."];

function makeRequest(family: Family, group: number, session: number, identity: string, familyIndex: number, demo = false): PublicHistoryRequest {
  const catalog = loadCatalog();
  const preference = palette[group % palette.length];
  const count = 4 + (familyIndex * 3 + group * 2 + session) % 9;
  const end = Date.parse("2026-10-17T20:00:00.000Z") - familyIndex * 3600_000 - group * 900_000 + session * 1800_000;
  const events: PublicShoppingEvent[] = [];
  for (let n = 0; n < count; n++) {
    const occurredAt = new Date(end - (count - 1 - n) * 18 * 3600_000).toISOString();
    const eventId = `${identity}-event-${n + 1}`;
    if (n === 0) {
      events.push({ eventId, eventType: "purchase", itemName: family.anchor[0], category: family.anchor[1], occurredAt, source: "demo-seed", external: true });
      continue;
    }
    const available = family.history.filter(id => id !== family.owned);
    const productId = n === 1 ? family.owned : available[(n + session - 2) % available.length]!;
    const product = catalog.find(p => p.id === productId);
    if (!product) throw new Error("Recipe uses invented store product");
    const variant = product.variants[(familyIndex + group + (n === 1 ? 0 : n + session)) % product.variants.length]!;
    events.push({ eventId, eventType: n === 1 ? "purchase" : n % 4 === 0 ? "cart" : "view", itemName: product.name, category: product.category, selectedColor: variant.colorName, occurredAt, source: "demo-seed", productId: product.id, variantId: variant.id });
  }
  const request: PublicHistoryRequest = { query: session === 0 ? "Show me a few useful things for the next couple of weeks." : "I am browsing for a useful addition to my routine.", itemBudget: family.budget + group * 5, hardRequirements: [], persona: { background: [...family.background, settings[group % settings.length]!], colorPreferences: [preference.label] }, publicHistory: events };
  validateCatalogHistory(request, catalog);
  return request;
}

function label(profile: SyntheticPublicProfile, family: Family, group: number, catalog: Product[]): PrivateHistoryLabel {
  const owned = [...new Set(profile.request.publicHistory.filter(e => e.eventType === "purchase" && !e.external).map(e => e.productId!))];
  const relevanceGrades = Object.fromEntries(catalog.map(p => [p.id, owned.includes(p.id) || p.price > profile.request.itemBudget! ? 0 : (family.scores[p.id] ?? 0)]));
  const eligible = catalog.filter(p => relevanceGrades[p.id]! > 0);
  const preferredVariantIds = eligible.map(p => (p.variants.find(v => palette[group % palette.length]!.match.test(v.colorName)) ?? p.variants[0])!.id);
  return { profileId: profile.profileId, relevanceGrades, preferredVariantIds, acceptableVariantIds: eligible.flatMap(p => p.variants.map(v => v.id)), alreadyOwnedProductIds: owned, rationale: family.reason, judgmentSource: "agent-authored-synthetic-not-observed-choice" };
}

export function buildPublicDataset() {
  const catalog = loadCatalog(); const families: Family[] = read(RECIPE).families;
  if (families.length !== 25 || families.filter(f => f.split === "train").length !== 20) throw new Error("Frozen25family split differs");
  const profiles: SyntheticPublicProfile[] = []; const groupMetadata: { userGroupId: string; familyId: string; split: "train" | "heldout"; profileIds: string[] }[] = [];
  for (const [index, family] of families.entries()) for (let group = 0; group < 4; group++) {
    const userGroupId = `history-u-${sha(family.id + ":" + group).slice(0, 12)}`; const profileIds: string[] = [];
    for (let session = 0; session < 2; session++) {
      const profileId = `${userGroupId}-s${session + 1}`; profileIds.push(profileId);
      profiles.push({ profileId, userGroupId, sessionId: `session-${sha(profileId).slice(0, 16)}`, request: makeRequest(family, group, session, profileId, index) });
    }
    groupMetadata.push({ userGroupId, familyId: family.id, split: family.split, profileIds });
  }
  const demos = [
    ["bob", "Bob", "Settling into a first apartment", "A new sofa, evening reading and a room that is still coming together.", 0, 2],
    ["maya", "Maya", "Evening reader", "Making a dim reading corner feel ready for the next chapter.", 1, 0],
    ["priya", "Priya", "Campus notebook keeper", "Morning lectures, loose pens and handwritten ideas between classes.", 2, 1],
    ["lena", "Lena", "All-weather commuter", "A walk from the train, changing forecasts and a bag for the workday.", 3, 2],
    ["june", "June", "Train-ride listener", "Albums on the train and focused work at a shared desk.", 4, 3],
    ["grace", "Grace", "Meal-prep regular", "Home-cooked lunches and a proper break away from the desk.", 5, 1],
    ["sam", "Sam", "Light weekend traveler", "Short train trips with a change of clothes and a few small essentials.", 6, 3],
    ["ana", "Ana", "After-hours sketcher", "Drawing classes, new pencils and supplies that need a home.", 7, 0],
    ["rosa", "Rosa", "Before-work swimmer", "Early laps and a bag packed the night before.", 8, 1],
    ["felix", "Felix", "Neighborhood wanderer", "Bakery stops, park loops and more errands on foot.", 9, 2],
  ] as const;
  const people: SyntheticDemoPerson[] = demos.map(([id, displayName, role, summary, familyIndex, group]) => ({ id, displayName, role, summary, request: makeRequest(families[familyIndex]!, group, 0, `demo-${id}`, familyIndex, true) }));
  const split = { schemaVersion: VERSION, policy: "20 authored scenario families TRAIN,5 families heldout; both sessions for a user group stay together. Named demos excluded. Semantic vocabulary is shared; this is synthetic family/group interpolation, not measured human generalization.", trainFamilyIds: families.filter(f => f.split === "train").map(f => f.id), heldoutFamilyIds: families.filter(f => f.split === "heldout").map(f => f.id), trainGroupIds: groupMetadata.filter(g => g.split === "train").map(g => g.userGroupId), heldoutGroupIds: groupMetadata.filter(g => g.split === "heldout").map(g => g.userGroupId), trainProfileIds: groupMetadata.filter(g => g.split === "train").flatMap(g => g.profileIds), heldoutProfileIds: groupMetadata.filter(g => g.split === "heldout").flatMap(g => g.profileIds), demoGroupIds: people.map(p => `demo-${p.id}`), groups: groupMetadata };
  const publicDataset = { schemaVersion: VERSION, provenance, profiles };
  return { catalog, families, profiles, people, split, publicDataset };
}

export function buildLabels(dataset: ReturnType<typeof buildPublicDataset>) {
  const byProfile = new Map(dataset.split.groups.flatMap(g => g.profileIds.map(id => [id, g] as const)));
  const rows = dataset.profiles.map(profile => {
    const metadata = byProfile.get(profile.profileId)!; const family = dataset.families.find(f => f.id === metadata.familyId)!;
    const group = dataset.split.groups.filter(g => g.familyId === family.id).findIndex(g => g.userGroupId === profile.userGroupId);
    return { split: metadata.split, label: label(profile, family, group, dataset.catalog) };
  });
  return { train: rows.filter(r => r.split === "train").map(r => r.label), heldout: rows.filter(r => r.split === "heldout").map(r => r.label) };
}

export function auditDataset(dataset: ReturnType<typeof buildPublicDataset>, labels: ReturnType<typeof buildLabels>) {
  const all = [...labels.train, ...labels.heldout];
  const historyVariants = new Set<string>(); const counts: Record<string, number> = {};
  for (const p of dataset.profiles) { validateCatalogHistory(p.request, dataset.catalog); counts[p.request.publicHistory.length] = (counts[p.request.publicHistory.length] ?? 0) + 1; p.request.publicHistory.forEach(e => { if (e.variantId) historyVariants.add(e.variantId); }); }
  for (const row of all) {
    if (Object.keys(row.relevanceGrades).length !== 16 || !Object.values(row.relevanceGrades).every(x => Number.isInteger(x) && x >= 0 && x <= 3) || !Object.values(row.relevanceGrades).includes(3)) throw new Error("Invalid relevance labels");
    if (row.alreadyOwnedProductIds.some(id => row.relevanceGrades[id] !== 0)) throw new Error("Already-owned item preferred over complement");
    if (Object.values(row.relevanceGrades).filter(x => x > 0).length < 2) throw new Error("Expected multi-positive labels");
  }
  const topCounts = Object.fromEntries(dataset.catalog.map(p => [p.id, labels.train.filter(l => l.relevanceGrades[p.id] === 3).length]));
  if (Object.values(topCounts).some(x => x === 0)) throw new Error("TRAIN top-label class missing");
  return { authoredSyntheticOnly: true, browserObserved: false, humanConversionClaim: false, anonymousProfiles: dataset.profiles.length, trainProfiles: labels.train.length, heldoutProfiles: labels.heldout.length, trainGroups: dataset.split.trainGroupIds.length, heldoutGroups: dataset.split.heldoutGroupIds.length, demoPeopleExcludedFromFit: dataset.people.length, actualCatalogProducts: dataset.catalog.length, actualCatalogVariants: 48, observedStoreVariants: historyVariants.size, eventCountHistogram: counts, totalAuthoredAnonymousEvents: dataset.profiles.reduce((n, p) => n + p.request.publicHistory.length, 0), externalEvents: dataset.profiles.flatMap(p => p.request.publicHistory).filter(e => e.external).length, trainTopGradeByProduct: topCounts, globalPriorTop1UpperBound: Math.max(...Object.values(topCounts)) / labels.train.length, ownedDuplicateNegativeRows: all.length, multiPositiveRows: all.length, publicRequestsContainOutcomeFields: false, holdoutCaveat: "No exact scenario family or user group crosses split; shared author and semantic building blocks limit external validity. Freeze model schedules on TRAIN only." };
}

export function writeDataset(out = OUT) {
  const files = ["profiles.public.json", "demo-people.json", "split.json", "PUBLIC_SPLIT_FROZEN.json", "labels.train.json", "labels.heldout.json", "catalog-binding.json", "audit.json", "manifest.json"];
  if (files.some(f => existsSync(resolve(out, f)))) throw new Error("Refuse overwriting frozen dataset");
  mkdirSync(out, { recursive: true }); const dataset = buildPublicDataset();
  const publish = (name: string, value: unknown) => writeFileSync(resolve(out, name), json(value));
  publish("profiles.public.json", dataset.publicDataset);
  publish("demo-people.json", { schemaVersion: VERSION, provenance, people: dataset.people });
  publish("split.json", dataset.split);
  const frozenFiles = Object.fromEntries(["profiles.public.json", "demo-people.json", "split.json"].map(name => [name, sha(readFileSync(resolve(out, name), "utf8"))]));
  publish("PUBLIC_SPLIT_FROZEN.json", { schemaVersion: VERSION, publicAndSplitPublishedBeforeLabels: true, files: frozenFiles });
  const labels = buildLabels(dataset);
  publish("labels.train.json", { schemaVersion: VERSION, labelScope: "TRAIN-only agent-authored relevance judgments", labels: labels.train });
  publish("labels.heldout.json", { schemaVersion: VERSION, labelScope: "Sealed heldout judgments: read only after prediction freeze", labels: labels.heldout });
  publish("catalog-binding.json", { schemaVersion: VERSION, source: CATALOG, sourceSha256: sha(readFileSync(resolve(ROOT, CATALOG), "utf8")), products: dataset.catalog });
  const audit = auditDataset(dataset, labels); publish("audit.json", audit);
  const hashes = Object.fromEntries(files.filter(f => f !== "manifest.json").map(f => [f, sha(readFileSync(resolve(out, f), "utf8"))]));
  publish("manifest.json", { schemaVersion: VERSION, provenance, files: hashes, sourceSha256: Object.fromEntries(sourcePins.map(p => [p, sha(readFileSync(resolve(ROOT, p), "utf8"))])), catalogScope: "16 real storefront product IDs/48 real variant IDs, captured runtime export; external history items have no store SKU", trainingSource: "Backend must persist immutable seed reports and TRAIN from the verified readback export; this seed is not proof of that round trip.", labelIsolation: "Inference imports only demo/public requests. Trainer reads labels.train only before both candidate checkpoints/predictions freeze; labels.heldout is evaluation only.", counts: audit });
  return { out: relative(ROOT, out), hashes, audit };
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) console.log(json(writeDataset(process.argv[2] ? resolve(process.argv[2]) : OUT)));
