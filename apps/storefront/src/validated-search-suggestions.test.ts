import { test } from "node:test";
import assert from "node:assert/strict";
import { parseValidatedSearchSuggestions } from "./validated-search-suggestions";

test("suggestions expose only exact strings from the owner's validated public metadata", () => {
  assert.deepEqual(parseValidatedSearchSuggestions({ version: "fieldwork.validated-search-suggestions.v1", queries: ["Rain jackets", "Work bags", "Rain jackets"] }), ["Rain jackets", "Work bags"]);
  for (const value of [null, {}, { queries: ["History"] }, { version: "unverified", queries: ["Rain jackets"] },
    { version: "fieldwork.validated-search-suggestions.v1", queries: [" Rain jackets"] },
    { version: "fieldwork.validated-search-suggestions.v1", queries: ["Valid", 5] }]) {
    assert.deepEqual(parseValidatedSearchSuggestions(value), []);
  }
});
