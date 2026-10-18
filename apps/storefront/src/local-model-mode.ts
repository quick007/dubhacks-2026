import { localModelEnabled, localModelEndpoint, type PersonalizationContext } from "./local-model";

/** The legacy service accepts query/guards only; public people belong to8767. */
export function storefrontModelMode(search: string, personalizedDefault: boolean,
  person: { displayName: string; request: PersonalizationContext }) {
  const enabled = localModelEnabled(search, personalizedDefault);
  const endpoint = localModelEndpoint(search, personalizedDefault);
  const personaEnabled = enabled && endpoint === "http://127.0.0.1:8767/personalize";
  return { enabled, endpoint, personaEnabled,
    context: personaEnabled ? person.request : {},
    personName: personaEnabled ? person.displayName : undefined };
}
