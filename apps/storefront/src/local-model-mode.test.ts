import { test } from "node:test";
import assert from "node:assert/strict";
import { storefrontModelMode } from "./local-model-mode";
import { buildLocalModelRequest } from "./local-model";
import { SHOPPER_PERSONAS } from "./personas";

const person = SHOPPER_PERSONAS[0];

test("legacy8766 keeps the original query-only DTO and hides person menu/badge", () => {
  const mode = storefrontModelMode("?model=local", false, person);
  assert.equal(mode.enabled, true);
  assert.equal(mode.endpoint, "http://127.0.0.1:8766/personalize");
  assert.equal(mode.personaEnabled, false);
  assert.equal(mode.personName, undefined);
  assert.deepEqual(buildLocalModelRequest(" work bag ", mode.context), { query: "work bag" });
  assert.deepEqual(buildLocalModelRequest("rain jacket under $130 in Navy", mode.context),
    { query: "rain jacket under $130 in Navy" });
});

test("explicit8767 preserves every public context field and exposes the selected person", () => {
  for (const selected of SHOPPER_PERSONAS) {
    const mode = storefrontModelMode("?model=local&candidate=persona", false, selected);
    assert.equal(mode.enabled, true);
    assert.equal(mode.endpoint, "http://127.0.0.1:8767/personalize");
    assert.equal(mode.personaEnabled, true);
    assert.equal(mode.personName, selected.displayName);
    assert.equal(mode.context, selected.request);
    assert.deepEqual(buildLocalModelRequest("same query", mode.context),
      buildLocalModelRequest("same query", selected.request));
  }
});

test("trusted personalized default uses8767 unchanged while generic override removes person UI/context", () => {
  const trusted = storefrontModelMode("", true, person);
  assert.equal(trusted.enabled, true);
  assert.equal(trusted.personaEnabled, true);
  assert.equal(trusted.personName, person.displayName);
  assert.deepEqual(buildLocalModelRequest("", trusted.context), buildLocalModelRequest("", person.request));
  for (const search of ["?model=generic", "?model=generic&candidate=persona"]) {
    const generic = storefrontModelMode(search, true, person);
    assert.equal(generic.enabled, false);
    assert.equal(generic.personaEnabled, false);
    assert.equal(generic.personName, undefined);
    assert.deepEqual(generic.context, {});
  }
});

test("candidate flag alone cannot enable a person menu or send history on a disabled model", () => {
  const mode = storefrontModelMode("?candidate=persona", false, person);
  assert.equal(mode.enabled, false);
  assert.equal(mode.personaEnabled, false);
  assert.equal(mode.personName, undefined);
  assert.deepEqual(mode.context, {});
});
