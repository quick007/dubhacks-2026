/** Public, authored synthetic histories. Outcome labels are never part of this contract. */
export type PublicShoppingEvent = {
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
export type PublicHistoryRequest = {
  query: string;
  itemBudget?: number;
  hardRequirements?: { attribute: string; value: string }[];
  persona: { background: string[]; colorPreferences: string[] };
  publicHistory: PublicShoppingEvent[];
};
export type SyntheticPublicProfile = {
  profileId: string;
  userGroupId: string;
  sessionId: string;
  request: PublicHistoryRequest;
};
export type SyntheticDemoPerson = {
  id: string;
  displayName: string;
  role: string;
  summary: string;
  request: PublicHistoryRequest;
};

function object(value: unknown, required: string[], optional: string[] = []): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Expected object");
  const out = value as Record<string, unknown>;
  if (required.some(key => !(key in out)) || Object.keys(out).some(key => ![...required, ...optional].includes(key)))
    throw new Error("Missing or unexpected public field");
  return out;
}
function text(value: unknown): asserts value is string {
  if (typeof value !== "string" || !value.trim() || value.length > 600) throw new Error("Expected bounded public text");
}
function strings(value: unknown, min: number, max: number): asserts value is string[] {
  if (!Array.isArray(value) || value.length < min || value.length > max) throw new Error("Invalid public text array");
  value.forEach(text);
}

/** Reject hidden labels, extra context keys and external items masquerading as store SKUs. */
export function validatePublicHistoryRequest(value: unknown): PublicHistoryRequest {
  const request = object(value, ["query", "persona", "publicHistory"], ["itemBudget", "hardRequirements"]);
  text(request.query);
  if (request.itemBudget !== undefined && (typeof request.itemBudget !== "number" || !Number.isFinite(request.itemBudget) || request.itemBudget <= 0))
    throw new Error("Invalid item budget");
  if (request.hardRequirements !== undefined) {
    if (!Array.isArray(request.hardRequirements) || request.hardRequirements.length > 12) throw new Error("Invalid hard requirements");
    for (const entry of request.hardRequirements) {
      const constraint = object(entry, ["attribute", "value"]); text(constraint.attribute); text(constraint.value);
    }
  }
  const persona = object(request.persona, ["background", "colorPreferences"]);
  strings(persona.background, 1, 6); strings(persona.colorPreferences, 0, 4);
  if (!Array.isArray(request.publicHistory) || request.publicHistory.length < 4 || request.publicHistory.length > 12)
    throw new Error("Expected4–12 public history events");
  const seen = new Set<string>(); let previous = -Infinity;
  for (const entry of request.publicHistory) {
    const event = object(entry, ["eventId", "eventType", "itemName", "category", "occurredAt", "source"], ["productId", "variantId", "selectedColor", "external"]);
    for (const key of ["eventId", "itemName", "category", "occurredAt"]) text(event[key]);
    if (!["view", "cart", "purchase"].includes(event.eventType as string) || !["demo-seed", "storefront-action"].includes(event.source as string))
      throw new Error("Unknown history event type/source");
    const time = Date.parse(event.occurredAt as string);
    if (!Number.isFinite(time) || !/^\d{4}-\d{2}-\d{2}T/.test(event.occurredAt as string) || time < previous)
      throw new Error("History must have ordered ISO times");
    previous = time;
    if (seen.has(event.eventId as string)) throw new Error("Duplicate event ID");
    seen.add(event.eventId as string);
    if (event.selectedColor !== undefined) text(event.selectedColor);
    if (event.external !== undefined && event.external !== true) throw new Error("External flag must be true when present");
    if (event.external === true) {
      if (event.productId !== undefined || event.variantId !== undefined) throw new Error("External item cannot claim store SKU");
    } else {
      text(event.productId); text(event.variantId); text(event.selectedColor);
    }
  }
  return structuredClone(value) as PublicHistoryRequest;
}
