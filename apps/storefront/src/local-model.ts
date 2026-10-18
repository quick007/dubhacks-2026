import { PRODUCTS, type Product } from "./config";
import { PRODUCT_VARIANTS } from "./variants";

export type PublicHistoryEvent = {
  eventId: string;
  eventType: "view" | "cart" | "purchase";
  itemName: string;
  category: string;
  selectedColor?: string;
  occurredAt: string;
  source: "demo-seed" | "storefront-action";
  productId?: string;
  variantId?: string;
  external?: true;
};
export type PersonalizationContext = {
  persona?: { background: string[]; colorPreferences: string[] };
  publicHistory?: PublicHistoryEvent[];
  itemBudget?: number | null;
  hardRequirements?: { attribute: string; value: string }[];
};
export type LocalModelRequest = PersonalizationContext & {
  query: string;
  itemBudget?: number | null;
  hardRequirements?: { attribute: string; value: string }[];
};
export type LocalCopyCard = {
  productId: Product["id"];
  variantId: string;
  title: string | null;
  badge: string | null;
  bodyCopy: string | null;
  originalName: string;
  itemPrice: number;
  selectedColor: string;
};
export type LocalCopySection = { heading: string | null; cards: LocalCopyCard[] };
export type LocalCopySource = {
  kind: "fresh-native" | "fresh-parallel-head" | "generated-earlier" | "mixed-native-cache" | "no-generation";
  generatedEarlier: boolean;
  freshPerRequestGeneration: boolean;
  maxAgeMs?: number;
  noEligibleProducts?: boolean;
};
export type LocalRankEvent = {
  type: "rank";
  requestId: string;
  productIds: Product["id"][];
  selectedProductIds: Product["id"][];
  variants: Partial<Record<Product["id"], string>>;
  shortfall: number;
  timing: { rankMs: number };
  provenance: { ranking: string; images: string };
};
export type LocalCopyEvent = {
  type: "firstThree" | "complete";
  requestId: string;
  sections: LocalCopySection[];
  timing: { readyMs: number; copyMs: number };
  acceptedFields: number;
  withheldFields: number;
  requiredFields: number;
  copySource?: LocalCopySource;
};
export type LocalErrorEvent = { type: "error"; requestId?: string; message: string };
export type LocalModelEvent = LocalRankEvent | LocalCopyEvent | LocalErrorEvent;
export type LocalModelState = {
  phase: "idle" | "ranking" | "copying" | "partial" | "complete" | "error";
  rank: LocalRankEvent | null;
  copy: LocalCopyEvent | null;
  displayQuery: string;
  pendingQuery: string;
  error: string | null;
  firstThreeReadyMs: number | null;
  requestKey: string;
};

export const INITIAL_LOCAL_MODEL_STATE: LocalModelState = {
  phase: "idle", rank: null, copy: null, displayQuery: "", pendingQuery: "", error: null, firstThreeReadyMs: null, requestKey: "",
};
export const LOCAL_MODEL_ENDPOINT = "http://127.0.0.1:8766/personalize";

/** Fixed local candidates only; no arbitrary remote endpoint in shared links. */
export function localModelEndpoint(search: string, personalizedDefault = false): string {
  return new URLSearchParams(search).get("candidate") === "persona" || personalizedDefault
    ? "http://127.0.0.1:8767/personalize" : LOCAL_MODEL_ENDPOINT;
}

/** Project inference-visible fields explicitly; display identity and labels never travel. */
export function buildLocalModelRequest(query: string, context: PersonalizationContext = {}): LocalModelRequest {
  return {
    query: query.trim() || "Browse the store",
    ...(context.itemBudget === undefined ? {} : { itemBudget: context.itemBudget }),
    ...(context.hardRequirements ? { hardRequirements: context.hardRequirements.map(({ attribute, value }) => ({ attribute, value })) } : {}),
    ...(context.persona ? { persona: { background: [...context.persona.background], colorPreferences: [...context.persona.colorPreferences] } } : {}),
    ...(context.publicHistory ? { publicHistory: context.publicHistory.map(event => ({
      eventId: event.eventId, eventType: event.eventType, itemName: event.itemName, category: event.category,
      occurredAt: event.occurredAt, source: event.source,
      ...(event.selectedColor === undefined ? {} : { selectedColor: event.selectedColor }),
      ...(event.productId === undefined ? {} : { productId: event.productId }),
      ...(event.variantId === undefined ? {} : { variantId: event.variantId }),
      ...(event.external === true ? { external: true as const } : {}),
    })) } : {}),
  };
}

export function localModelEnabled(search: string, personalizedDefault = false): boolean {
  const mode = new URLSearchParams(search).get("model");
  return mode === "generic" ? false : mode === "local" || personalizedDefault;
}

export function preserveLocalModelSearch(serializedConfig: string, currentSearch: string): string {
  const params = new URLSearchParams(serializedConfig);
  if (localModelEnabled(currentSearch)) params.set("model", "local");
  if (new URLSearchParams(currentSearch).get("candidate") === "persona") params.set("candidate", "persona");
  return params.toString();
}

function object(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("Invalid local model response.");
  return value as Record<string, unknown>;
}
function text(value: unknown, limit = 500): string {
  if (typeof value !== "string" || !value.trim() || value.length > limit || /[\r\n\t<>]/.test(value)) {
    throw new Error("Invalid local model text.");
  }
  return value;
}
function number(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) throw new Error("Invalid local model timing or count.");
  return value;
}
function count(value: unknown): number {
  const parsed = number(value);
  if (!Number.isInteger(parsed)) throw new Error("Invalid local model field count.");
  return parsed;
}
function productIds(value: unknown): Product["id"][] {
  if (!Array.isArray(value) || value.length > PRODUCTS.length || value.some((id) => !PRODUCTS.some((p) => p.id === id)) || new Set(value).size !== value.length) {
    throw new Error("Local model returned unknown or repeated products.");
  }
  return value as Product["id"][];
}
function nullableText(value: unknown, limit: number): string | null {
  return value === null ? null : text(value, limit);
}

function copySource(value: unknown): LocalCopySource {
  const source = object(value);
  const flags = { "fresh-native": [false, true], "fresh-parallel-head": [false, true],
    "generated-earlier": [true, false], "mixed-native-cache": [true, true], "no-generation": [false, false] } as const;
  if (typeof source.kind !== "string" || !Object.hasOwn(flags, source.kind)) throw new Error("Unknown local model copy source.");
  const kind = source.kind as LocalCopySource["kind"];
  const [generatedEarlier, freshPerRequestGeneration] = flags[kind];
  if (source.generatedEarlier !== generatedEarlier || source.freshPerRequestGeneration !== freshPerRequestGeneration) {
    throw new Error("Inconsistent local model generation provenance.");
  }
  return { kind, generatedEarlier, freshPerRequestGeneration,
    ...(kind === "no-generation" ? { noEligibleProducts: source.noEligibleProducts === true } : {}),
    ...(source.maxAgeMs === undefined ? {} : { maxAgeMs: number(source.maxAgeMs) }) };
}

export function parseLocalModelEvent(value: unknown): LocalModelEvent {
  const event = object(value);
  if (event.type === "error") return { type: "error", ...(event.requestId === undefined ? {} : { requestId: text(event.requestId, 200) }), message: text(event.message) };
  const requestId = text(event.requestId, 200);
  const timing = object(event.timing);
  if (event.type === "rank") {
    const ids = productIds(event.productIds);
    const selected = productIds(event.selectedProductIds);
    if (selected.length > 6 || selected.some((id, i) => ids[i] !== id)) throw new Error("Local model selection differs from its product order.");
    const supplied = object(event.variants);
    const variants: LocalRankEvent["variants"] = {};
    for (const [id, variant] of Object.entries(supplied)) {
      const product = PRODUCTS.find((p) => p.id === id);
      if (!product || !PRODUCT_VARIANTS[product.id].some((v) => v.id === variant)) throw new Error("Local model returned an unknown product color.");
      variants[product.id] = variant as string;
    }
    if (selected.some((id) => !variants[id])) throw new Error("Local model omitted a selected product color.");
    const provenance = object(event.provenance);
    const shortfall = count(event.shortfall);
    if (shortfall !== 6 - selected.length) throw new Error("Local model product shortfall differs from its selection.");
    return { type: "rank", requestId, productIds: ids, selectedProductIds: selected, variants, shortfall,
      timing: { rankMs: number(timing.rankMs) }, provenance: { ranking: text(provenance.ranking), images: text(provenance.images) } };
  }
  if (event.type !== "firstThree" && event.type !== "complete") throw new Error("Unknown local model update.");
  if (!Array.isArray(event.sections) || event.sections.length > (event.type === "firstThree" ? 1 : 2)) throw new Error("Invalid local model sections.");
  const seen = new Set<string>();
  const sections = event.sections.map((value): LocalCopySection => {
    const section = object(value);
    if (!Array.isArray(section.cards) || section.cards.length < 1 || section.cards.length > 3) throw new Error("Invalid local model card count.");
    return { heading: nullableText(section.heading, 100), cards: section.cards.map((value): LocalCopyCard => {
      const card = object(value);
      const product = PRODUCTS.find((p) => p.id === card.productId);
      if (!product || seen.has(product.id)) throw new Error("Local model returned an unknown or repeated card.");
      seen.add(product.id);
      const variant = PRODUCT_VARIANTS[product.id].find((v) => v.id === card.variantId);
      if (!variant || card.originalName !== product.name || card.itemPrice !== product.price || card.selectedColor !== variant.colorName) {
        throw new Error("Local model card facts differ from the store catalog.");
      }
      return { productId: product.id, variantId: variant.id, originalName: product.name, itemPrice: product.price,
        selectedColor: variant.colorName, title: nullableText(card.title, 140), badge: nullableText(card.badge, 60), bodyCopy: nullableText(card.bodyCopy, 240) };
    }) };
  });
  const acceptedFields = count(event.acceptedFields), withheldFields = count(event.withheldFields), requiredFields = count(event.requiredFields);
  if (requiredFields !== sections.length + seen.size * 3 || acceptedFields + withheldFields !== requiredFields) throw new Error("Local model field totals differ from its payload.");
  const actualAccepted = sections.reduce((n, s) => n + Number(s.heading !== null) + s.cards.reduce((m, c) => m + Number(c.title !== null) + Number(c.badge !== null) + Number(c.bodyCopy !== null), 0), 0);
  if (actualAccepted !== acceptedFields) throw new Error("Local model fresh field count differs from its payload.");
  if (event.copySource !== undefined && object(event.copySource).kind === "no-generation"
    && (requiredFields !== 0 || sections.length !== 0 || object(event.copySource).noEligibleProducts !== true)) {
    throw new Error("No-generation copy must be an empty eligible result.");
  }
  return { type: event.type, requestId, sections, acceptedFields, withheldFields, requiredFields,
    ...(event.copySource === undefined ? {} : { copySource: copySource(event.copySource) }),
    timing: { readyMs: number(timing.readyMs), copyMs: number(timing.copyMs) } };
}

export function beginLocalModelRequest(state: LocalModelState, query: string, requestKey = query, clearPrevious = false): LocalModelState {
  return { ...(clearPrevious ? INITIAL_LOCAL_MODEL_STATE : state), phase: "ranking", pendingQuery: query, requestKey, error: null };
}

export function applyLocalModelEvent(state: LocalModelState, event: LocalModelEvent, query: string, requestKey = query): LocalModelState {
  if (requestKey !== state.requestKey || query !== state.pendingQuery) return state;
  if (event.type === "error") {
    if (event.requestId && state.rank?.requestId && event.requestId !== state.rank.requestId && state.phase !== "ranking") return state;
    return { ...state, phase: "error", error: event.message };
  }
  if (event.type === "rank") return state.phase === "ranking"
    ? { ...state, phase: "copying", rank: event, copy: null, displayQuery: query, error: null, firstThreeReadyMs: null }
    : state;
  if (!state.rank || state.rank.requestId !== event.requestId || state.phase === "ranking" || state.phase === "error" || state.phase === "complete") return state;
  const cards = event.sections.flatMap((s) => s.cards);
  const expected = event.type === "firstThree" ? state.rank.selectedProductIds.slice(0, 3) : state.rank.selectedProductIds;
  if (cards.length !== expected.length || cards.some((card, i) => card.productId !== expected[i] || card.variantId !== state.rank?.variants[card.productId])) {
    throw new Error("Local model copy differs from its selected products or colors.");
  }
  return { ...state, phase: event.type === "complete" ? "complete" : "partial", copy: event,
    firstThreeReadyMs: event.type === "firstThree" ? event.timing.readyMs : state.firstThreeReadyMs };
}

export async function readLocalModelStream(response: Response, onEvent: (event: LocalModelEvent) => void): Promise<void> {
  if (!response.ok) throw new Error(`Local model service returned ${response.status}.`);
  if (!response.body) throw new Error("Local model service returned no stream.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", finished = false;
  const line = (value: string) => {
    if (!value.trim()) return;
    const event = parseLocalModelEvent(JSON.parse(value));
    if (event.type === "complete" || event.type === "error") finished = true;
    onEvent(event);
  };
  try {
    while (true) {
      const part = await reader.read();
      buffer += decoder.decode(part.value, { stream: !part.done });
      if (buffer.length > 1_000_000) throw new Error("Local model response exceeded its expected size.");
      let boundary: number;
      while ((boundary = buffer.indexOf("\n")) !== -1) { line(buffer.slice(0, boundary)); buffer = buffer.slice(boundary + 1); }
      if (part.done) break;
    }
    line(buffer);
    if (!finished) throw new Error("Local model stopped before finishing. The last available result is still shown.");
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally { reader.releaseLock(); }
}

export function localModelStatus(state: LocalModelState): string {
  if (state.phase === "idle") return "Local model ready.";
  if (state.phase === "error") return `Local model unavailable: ${state.error ?? "Request failed."}${state.rank ? " Last available result shown." : " Store catalog shown."}`;
  if (state.phase === "ranking") return `Local model is ordering products${state.rank ? "; last available result shown" : ""}…`;
  const rank = state.rank;
  if (!rank) return "Local model is starting…";
  const timing = `Order ${(rank.timing.rankMs / 1000).toFixed(2)} s`;
  const provenance = "Local model · Learned product order · Existing storefront gallery";
  if (!state.copy) return `${provenance} · ${timing} · Preparing model copy…${rank.shortfall ? ` ${rank.selectedProductIds.length} eligible products.` : ""}`;
  const endpoint = state.copy.type === "firstThree" ? "First cards" : "Complete copy";
  const firstTiming = state.copy.type === "complete" && state.firstThreeReadyMs !== null
    ? ` · First cards ${(state.firstThreeReadyMs / 1000).toFixed(2)} s` : "";
  const fieldLabel = state.copy.copySource?.kind === "no-generation" ? "fields · No eligible products; no text generated"
    : state.copy.copySource?.kind === "generated-earlier" ? "fields generated earlier"
    : state.copy.copySource?.kind === "mixed-native-cache" ? "cached and new model fields"
    : state.copy.copySource?.kind === "fresh-parallel-head" ? "fresh learned fields · Catalog-trained text" : "fresh fields";
  return `${provenance} · ${timing}${firstTiming} · ${endpoint} ${(state.copy.timing.readyMs / 1000).toFixed(2)} s · ${state.copy.acceptedFields}/${state.copy.requiredFields} ${fieldLabel}${state.copy.withheldFields ? ` · ${state.copy.withheldFields} withheld` : ""}${rank.shortfall ? ` · ${rank.selectedProductIds.length} eligible products` : ""}`;
}
