/** TRAIN-only authored purchase counterfactuals from verified Tiger readback. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { validatePublicHistoryRequest, type PublicHistoryRequest, type SyntheticPublicProfile } from "../packages/contracts/src/synthetic-history-v1.ts";

export const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
export const OUT = resolve(ROOT, "experiments/ranker/pretrained/synthetic_history_train_contrasts_v2");
export const VERSION = "authored-tiger-train-purchase-contrasts.v2";
export const INPUTS = {
  public: "data/authored-history-tiger-train-v1/profiles.train.public.json",
  receipt: "data/authored-history-tiger-train-v1/receipt.json",
  labels: "experiments/ranker/pretrained/synthetic_history_v1/labels.train.json",
  catalog: "experiments/ranker/pretrained/synthetic_history_v1/catalog-binding.json",
} as const;
const EXPECTED = {
  public: "32588aa231532443f9d954f17c2e2cb3c1ff01859ac55666ba332d453513b973",
  labels: "1b1ae769cb162bcb0deee76ea938ac619fbd7a94ee3ef39d7544d5d2491e8ccb",
  catalog: "bdd7e2f5f5bc41c147830b5ecd03b550794a237ad633d4e4233688ec223b8450",
};
const schemaPath = "packages/contracts/src/synthetic-history-v1.ts";
const selfPath = "scripts/build-synthetic-history-train-contrasts-v2.ts";
const digest = (bytes: string | Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const json = (value: unknown) => JSON.stringify(value, null, 2) + "\n";
export type Product = { id: string; name: string; category: string; price: number; variants: { id: string; colorName: string }[] };
export type Label = { profileId: string; relevanceGrades: Record<string, number>; preferredVariantIds: string[]; acceptableVariantIds: string[]; alreadyOwnedProductIds: string[]; rationale: string; judgmentSource: string };
export type Pair = { sourceProfileId: string; contrastProfileId: string; userGroupId: string; sourceSessionId: string; contrastSessionId: string; replacedEventIndex: number; replacedExternalItemName: string; newlyOwnedProductId: string; newlyOwnedVariantId: string; promotedExistingPositiveProductIds: string[]; originalPrimaryGrade: number; counterfactualPrimaryGrade: number };

export function eligibleVariants(product: Product, request: PublicHistoryRequest) {
  if (request.itemBudget !== undefined && product.price > request.itemBudget) return [];
  let variants = product.variants;
  for (const requirement of request.hardRequirements ?? []) {
    const value = requirement.value.toLowerCase();
    if (["product", "productId"].includes(requirement.attribute)) { if (product.id.toLowerCase() !== value) return []; }
    else if (requirement.attribute === "name") { if (product.name.toLowerCase() !== value) return []; }
    else if (requirement.attribute === "category") { if (product.category.toLowerCase() !== value) return []; }
    else if (["color", "selectedColor"].includes(requirement.attribute)) variants = variants.filter(v => v.colorName.toLowerCase() === value);
    else throw new Error("Unknown hard requirement: do not silently drop it");
  }
  return variants;
}
export function validateBinding(request: PublicHistoryRequest, catalog: Product[]) {
  validatePublicHistoryRequest(request);
  for (const event of request.publicHistory) {
    if (event.external) continue;
    const product = catalog.find(p => p.id === event.productId);
    const variant = product?.variants.find(v => v.id === event.variantId);
    if (!product || !variant || product.name !== event.itemName || product.category !== event.category || variant.colorName !== event.selectedColor)
      throw new Error("Observed store product/name/category/color/variant is not catalog factual");
  }
}
export function makeContrast(source: SyntheticPublicProfile, original: Label, catalog: Product[]) {
  validateBinding(source.request, catalog);
  if (source.profileId !== original.profileId) throw new Error("Profile and TRAIN label identity differ");
  const ids = catalog.map(p => p.id).sort();
  if (JSON.stringify(Object.keys(original.relevanceGrades).sort()) !== JSON.stringify(ids) || !Object.values(original.relevanceGrades).every(n => Number.isInteger(n) && n >= 0 && n <= 3)) throw new Error("Exact catalog relevance grades required");
  const primaries = catalog.filter(p => original.relevanceGrades[p.id] === 3);
  if (primaries.length !== 1) throw new Error("Require one source primary; no label-selected tie breaking");
  const primary = primaries[0]!;
  const owned = [...new Set(source.request.publicHistory.filter(e => e.eventType === "purchase" && !e.external).map(e => e.productId!))];
  if (owned.includes(primary.id) || original.alreadyOwnedProductIds.some(id => original.relevanceGrades[id] !== 0) || JSON.stringify([...original.alreadyOwnedProductIds].sort()) !== JSON.stringify([...owned].sort())) throw new Error("Original ownership/label conflict");
  const eligible = eligibleVariants(primary, source.request);
  const selected = eligible.filter(v => original.preferredVariantIds.includes(v.id));
  if (selected.length !== 1) throw new Error("Unique source-preferred factual eligible primary variant required");
  const external = source.request.publicHistory.flatMap((e, index) => e.external && e.eventType === "purchase" ? [index] : []);
  if (external.length !== 1) throw new Error("Require exactly one external purchase anchor; no selection based on evaluation");
  const replacedIndex = external[0]!; const anchor = source.request.publicHistory[replacedIndex]!;
  const profile: SyntheticPublicProfile = structuredClone(source);
  profile.profileId = source.profileId + "-purchase-contrast-v2";
  profile.sessionId = source.sessionId + "-purchase-contrast-v2";
  const variant = selected[0]!;
  profile.request.publicHistory[replacedIndex] = { eventId: anchor.eventId + "-counterfactual-v2", eventType: "purchase", itemName: primary.name, category: primary.category, selectedColor: variant.colorName, occurredAt: anchor.occurredAt, source: "demo-seed", productId: primary.id, variantId: variant.id };
  validateBinding(profile.request, catalog);
  const nextOwned = [...new Set([...owned, primary.id])];
  const grades = Object.fromEntries(catalog.map(p => [p.id, nextOwned.includes(p.id) || eligibleVariants(p, profile.request).length === 0 ? 0 : original.relevanceGrades[p.id]!]));
  const best = Math.max(...Object.values(grades));
  if (best < 1 || best > 2) throw new Error("Need previously-positive unowned alternative; never manufacture new utility target");
  const promoted = catalog.filter(p => grades[p.id] === best).map(p => p.id);
  for (const id of promoted) grades[id] = 3;
  const positives = catalog.filter(p => grades[p.id]! > 0);
  const positiveVariants = new Set(positives.flatMap(p => eligibleVariants(p, profile.request).map(v => v.id)));
  const preferred = original.preferredVariantIds.filter(id => positiveVariants.has(id));
  if (positives.some(p => !p.variants.some(v => preferred.includes(v.id)))) throw new Error("Alternative lacks source-preferred factual variant");
  const label: Label = { profileId: profile.profileId, relevanceGrades: grades, preferredVariantIds: preferred, acceptableVariantIds: original.acceptableVariantIds.filter(id => positiveVariants.has(id)), alreadyOwnedProductIds: nextOwned,
    rationale: `Authored TRAIN counterfactual: the history now records purchase of ${primary.name}. The request seeks an addition without replacement intent, so that newly owned item and previously owned items receive zero. Existing budget-eligible unowned positives retain their original factual justification; the strongest remaining alternatives are promoted. This is synthetic supervision, not an observed choice.`, judgmentSource: "agent-authored-synthetic-training-transform-not-observed-choice" };
  const pair: Pair = { sourceProfileId: source.profileId, contrastProfileId: profile.profileId, userGroupId: source.userGroupId, sourceSessionId: source.sessionId, contrastSessionId: profile.sessionId, replacedEventIndex: replacedIndex, replacedExternalItemName: anchor.itemName, newlyOwnedProductId: primary.id, newlyOwnedVariantId: variant.id, promotedExistingPositiveProductIds: promoted, originalPrimaryGrade: 3, counterfactualPrimaryGrade: 0 };
  if (JSON.stringify({ ...source.request, publicHistory: undefined }) !== JSON.stringify({ ...profile.request, publicHistory: undefined })) throw new Error("Counterfactual changed public background/query/colors/budget/constraints");
  if (source.request.publicHistory.filter((e, n) => JSON.stringify(e) !== JSON.stringify(profile.request.publicHistory[n])).length !== 1) throw new Error("Counterfactual changed more than one event");
  return { profile, label, pair };
}

export function buildContrasts() {
  const accessLog: { path: string; sha256: string; purpose: string }[] = [];
  const readAllowed = (key: keyof typeof INPUTS, purpose: string) => {
    const path = INPUTS[key]; const bytes = readFileSync(resolve(ROOT, path)); const hash = digest(bytes);
    if (key !== "receipt" && hash !== EXPECTED[key]) throw new Error("Frozen TRAIN source hash differs: " + key);
    accessLog.push({ path, sha256: hash, purpose }); return JSON.parse(bytes.toString());
  };
  const receipt = readAllowed("receipt", "Verify actual Tiger readback and exact TRAIN-only projection provenance");
  if (receipt.source !== "tiger" || receipt.rows !== 160 || receipt.groups !== 80 || receipt.projectionSha256 !== EXPECTED.public || receipt.publicExportSha256 !== "c36a54ebf54021389486f594275258852fbe460cb0933f8f9701f93c939c25f9" || receipt.reportId !== "authored-public-history-v1-200profiles") throw new Error("Wrong verified Tiger TRAIN projection");
  const publicSource = readAllowed("public", "Only exact160TRAIN public rows, not full200 export");
  const privateSource = readAllowed("labels", "Only original160TRAIN judgments");
  const catalogSource = readAllowed("catalog", "Only public16products/48variants factual bindings");
  const profiles: SyntheticPublicProfile[] = publicSource.profiles; const labels: Label[] = privateSource.labels; const catalog: Product[] = catalogSource.products;
  if (profiles.length !== 160 || labels.length !== 160 || catalog.length !== 16 || catalog.reduce((n, p) => n + p.variants.length, 0) !== 48 || new Set(profiles.map(p => p.userGroupId)).size !== 80 || new Set(profiles.map(p => p.profileId)).size !== 160) throw new Error("Expected160TRAIN rows/80groups/16products/48variants");
  const map = new Map(labels.map(l => [l.profileId, l])); if (map.size !== 160) throw new Error("Duplicate labels");
  const transformed = profiles.map(p => { const l = map.get(p.profileId); if (!l) throw new Error("Missing TRAIN label"); return makeContrast(p, l, catalog); });
  const contrasts = transformed.map(t => t.profile); const contrastLabels = transformed.map(t => t.label); const pairs = transformed.map(t => t.pair);
  const counter = (values: string[]) => Object.fromEntries([...new Set(values)].sort().map(x => [x, values.filter(v => v === x).length]));
  const changedByProduct = counter(pairs.map(p => p.newlyOwnedProductId));
  if (Object.keys(changedByProduct).length !== 16 || Math.min(...Object.values(changedByProduct)) !== 8 || Math.max(...Object.values(changedByProduct)) !== 16) throw new Error("Changed-purchase balance across all16candidates differs");
  const provenance = { kind: "agent-authored-synthetic-training-transform", parentDatasetVersion: publicSource.schemaVersion, source: "verified Tiger TRAIN-only projection plus original TRAIN labels", browserObserved: false, humanConversionClaim: false, realCustomerData: false, outcomeClaim: "Counterfactual relevance supervision, not an observed click/cart/purchase outcome", selectionPolicy: "One deterministic counterpart for each original TRAIN session, no sampling or evaluation-dependent selection" };
  const audit = { originalTrainRows: 160, authoredContrastRows: 160, combinedTrainRows: 320, originalTrainGroups: 80, combinedTrainGroups: 80, addedHeldoutGroups: 0, changedPurchaseByProduct: changedByProduct, promotedExistingPositiveByProduct: counter(pairs.flatMap(p => p.promotedExistingPositiveProductIds)), twoOrMorePositiveContrastRows: contrastLabels.filter(l => Object.values(l.relevanceGrades).filter(v => v > 0).length >= 2).length, constraintsDropped: 0, exactNonHistoryContextPreservedRows: 160, exactlyOnePurchaseEventChangedRows: 160, trainOnlyInputFilesRead: accessLog.map(x => x.path), heldoutOrProbeFilesReadByGenerator: false, sourceLabelConditionedPurchaseSelection: true, originalProfilesAndLabelsUnmodified: true, scope: "Learns an authored ownership novelty intervention already represented in TRAIN; no new human, behavior-category or heldout generalization claim" };
  return { receipt, publicSource, privateSource, profiles, labels, contrasts, contrastLabels, pairs, provenance, audit, accessLog };
}
export function writeContrasts(out = OUT) {
  const filenames = ["profiles.contrasts.train.public.json", "profiles.train.public.json", "labels.contrasts.train.json", "labels.train.json", "pair-links.json", "split.train.json", "TRAIN_PUBLIC_FROZEN.json", "SOURCE_ACCESS_LOG.json", "TIGER_TRAIN_PROJECTION_RECEIPT.json", "audit.json", "manifest.json"];
  if (filenames.some(f => existsSync(resolve(out, f)))) throw new Error("Refuse overwriting frozen TRAIN transform");
  const data = buildContrasts(); mkdirSync(out, { recursive: true }); const publish = (f: string, x: unknown) => writeFileSync(resolve(out, f), json(x));
  publish("profiles.contrasts.train.public.json", { schemaVersion: VERSION, provenance: data.provenance, profiles: data.contrasts });
  publish("profiles.train.public.json", { schemaVersion: VERSION, provenance: data.provenance, profiles: [...data.profiles, ...data.contrasts] });
  publish("pair-links.json", { schemaVersion: VERSION, pairs: data.pairs });
  publish("split.train.json", { schemaVersion: VERSION, trainProfileIds: [...data.profiles, ...data.contrasts].map(p => p.profileId), trainGroupIds: [...new Set(data.profiles.map(p => p.userGroupId))], newHeldoutGroups: [], groupPolicy: "Both source sessions and both authored counterfactual sessions stay within the original TRAIN user group" });
  publish("TRAIN_PUBLIC_FROZEN.json", { schemaVersion: VERSION, trainingPublicAndPairLinksPublishedBeforeLabelOutputs: true, trainingLabelsWereReadToAuthorCounterfactualPurchases: true, files: Object.fromEntries(["profiles.contrasts.train.public.json", "profiles.train.public.json", "pair-links.json", "split.train.json"].map(f => [f, digest(readFileSync(resolve(out, f)))])) });
  publish("labels.contrasts.train.json", { schemaVersion: VERSION, labelScope: "TRAIN-only authored counterfactual judgments", labels: data.contrastLabels });
  publish("labels.train.json", { schemaVersion: VERSION, labelScope: "Original160TRAIN judgments plus160authored contrasts; no evaluation annotations", labels: [...data.labels, ...data.contrastLabels] });
  publish("SOURCE_ACCESS_LOG.json", { schemaVersion: VERSION, dataFilesRead: data.accessLog, excludedReads: ["full200Tiger public export", "40heldout public profiles/labels", "new32evaluation", "private/history-only diagnostic probes", "authored scenario recipe"] });
  publish("TIGER_TRAIN_PROJECTION_RECEIPT.json", data.receipt); publish("audit.json", data.audit);
  const files = Object.fromEntries(filenames.filter(f => f !== "manifest.json").map(f => [f, digest(readFileSync(resolve(out, f)))]));
  const manifest = { schemaVersion: VERSION, provenance: data.provenance, files, sourceInputs: Object.fromEntries(data.accessLog.map(x => [x.path, x.sha256])), sourceSha256: Object.fromEntries([selfPath, schemaPath].map(p => [p, digest(readFileSync(resolve(ROOT, p)))])), originalFrozenV1ManifestSha256: "f717829925121e35470635f08a4baf508ac1f46c287f38a0657b93debb10b27d", nativeFeaturePolicy: "New public contexts require their own verified Qwen feature keys. Never relabel/reuse an old context vector for a changed purchase. Original catalog feature cache is unchanged.", persistencePolicy: "Explicit training transform of verified Tiger-exported TRAIN inputs. Counterfactual rows themselves have not been persisted/read back as a new Tiger report.", audit: data.audit };
  publish("manifest.json", manifest); return { out: relative(ROOT, out), manifestSha256: digest(readFileSync(resolve(out, "manifest.json"))), files, audit: data.audit };
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) console.log(json(writeContrasts(process.argv[2] ? resolve(process.argv[2]) : OUT)));
