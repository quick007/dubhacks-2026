import seed from "../../../experiments/ranker/pretrained/synthetic_history_v1/demo-people.json";
import { buildLocalModelRequest, type LocalModelRequest, type PublicHistoryEvent } from "./local-model";

export type ShopperPersona = {
  id: string;
  displayName: string;
  role: string;
  summary: string;
  request: LocalModelRequest & { publicHistory: PublicHistoryEvent[] };
};

export const SHOPPER_PERSONAS = seed.people as ShopperPersona[];
export const INITIAL_PERSONA_ID = SHOPPER_PERSONAS[0].id;

export function personaRequest(person: ShopperPersona, query: string) {
  return buildLocalModelRequest(query, person.request);
}

export function recentHistory(person: ShopperPersona, count = 3): PublicHistoryEvent[] {
  return [...person.request.publicHistory].sort((a,b) => b.occurredAt.localeCompare(a.occurredAt)).slice(0,count);
}

export function personInitials(name: string): string {
  return name.split(/\s+/).slice(0,2).map(part => part[0]).join("");
}
