import test from 'node:test';
import assert from 'node:assert/strict';
import { fixedSchedule, navigateAttempt, verifyRequestProtocol, selectReadiness, validFieldReceipt, summarize, matchDomToCopy, parsedSourceFieldsMatch, TIMEOUT_MS } from './store_candidate_browser_probe_v1.mjs';

const mirror = (stage, duration = 123, startTime = 20) => ({ startTime, duration,
  detail: { stage, requiredFields: stage === 'firstThree' ? 10 : 20,
    acceptedFields: stage === 'firstThree' ? 10 : 20, withheldFields: 0 } });
test('fixed eight paired requests preserve query and fresh-before-repeat order', () => {
  const rows = fixedSchedule(); assert.equal(rows.length, 8); assert.equal(new Set(rows.map(r => r.query)).size, 4);
  for (let i = 0; i < rows.length; i += 2) {
    assert.equal(rows[i].query, rows[i + 1].query); assert.equal(rows[i].condition, 'fresh-context');
    assert.equal(rows[i + 1].condition, 'same-context-repeat');
  }
  assert.equal(TIMEOUT_MS, 30_000);
});
test('same-context repeat explicitly reloads while fresh attempt navigates', async () => {
  const calls = [];
  const page = { goto: async (...args) => calls.push(['goto', ...args]), reload: async (...args) => calls.push(['reload', ...args]) };
  const options = { waitUntil: 'domcontentloaded', timeout: TIMEOUT_MS };
  await navigateAttempt(page, fixedSchedule()[0], 'http://127.0.0.1:4312/?model=local#collection?q=test', options);
  await navigateAttempt(page, fixedSchedule()[1], 'http://127.0.0.1:4312/?model=local#collection?q=test', options);
  assert.deepEqual(calls, [['goto', 'http://127.0.0.1:4312/?model=local#collection?q=test', options], ['reload', options]]);
});
test('request protocol rejects stale mirrors, zero POST, mismatched IDs and reused documents', () => {
  const row = { query: 'test', actualPostRequests: [{ query: 'test' }],
    modelResponses: [{ events: ['rank', 'firstThree', 'complete'].map(type => ({ type, requestId: 'new-id' })) }],
    browser: { documentId: 'new-document', stages: Object.fromEntries(['rank', 'firstThree', 'complete'].map(stage =>
      [stage, { mirror: { detail: { requestId: 'new-id' } } }])) } };
  assert.equal(verifyRequestProtocol(row).passed, true);
  assert.equal(verifyRequestProtocol({ ...row, actualPostRequests: [] }).passed, false);
  assert.equal(verifyRequestProtocol(row, new Set(['new-id'])).passed, false);
  assert.equal(verifyRequestProtocol(row, new Set(), new Set(['new-document'])).passed, false);
  assert.equal(verifyRequestProtocol({ ...row, actualPostRequests: [{ query: 'wrong' }] }).passed, false);
  assert.equal(verifyRequestProtocol({ ...row, modelResponses: [{ events: [{ type: 'complete', requestId: 'different' }] }] }).passed, false);
  const bad = structuredClone(row); bad.browser.stages.complete.mirror.detail.requestId = 'old-id';
  assert.equal(verifyRequestProtocol(bad).passed, false);
});
test('readiness filtering excludes malformed timings and unrelated stages', () => {
  assert.equal(selectReadiness([mirror('complete', 400, 5), mirror('firstThree'), mirror('complete', 40, 50),
    { startTime: 100, duration: NaN, detail: { stage: 'complete' } }], 'complete').duration, 40);
  assert.equal(selectReadiness([], 'complete'), null);
});
test('field gates require every field, no withheld fields and expected six-card contract', () => {
  assert.equal(validFieldReceipt(mirror('firstThree'), 'firstThree'), true);
  assert.equal(validFieldReceipt(mirror('firstThree'), 'complete'), false);
  const bad = mirror('complete'); bad.detail.withheldFields = 1;
  assert.equal(validFieldReceipt(bad, 'complete'), false);
  assert.equal(validFieldReceipt(null, 'complete'), false);
});
test('parsed provenance permits extra raw checkpoint facts but requires canonical kind and flags', () => {
  const mirrorSource = { kind: 'fresh-parallel-head', generatedEarlier: false, freshPerRequestGeneration: true };
  const raw = { ...mirrorSource, checkpointSha256: 'checkpoint', featuresSha256: 'features', trainingUpdates: 320 };
  assert.equal(parsedSourceFieldsMatch(mirrorSource, raw), true);
  assert.equal(parsedSourceFieldsMatch(mirrorSource, { ...raw, kind: 'generated-earlier' }), false);
  assert.equal(parsedSourceFieldsMatch(mirrorSource, { ...raw, freshPerRequestGeneration: false }), false);
  assert.equal(parsedSourceFieldsMatch({ ...mirrorSource, generatedEarlier: true }, { ...raw, generatedEarlier: true }), false);
  assert.equal(parsedSourceFieldsMatch(mirrorSource, { kind: raw.kind, generatedEarlier: false }), false);
  assert.equal(parsedSourceFieldsMatch({ ...mirrorSource, maxAgeMs: 25 }, { ...raw, maxAgeMs: 25 }), true);
  assert.equal(parsedSourceFieldsMatch({ ...mirrorSource, maxAgeMs: 25 }, { ...raw, maxAgeMs: 26 }), false);
  assert.equal(parsedSourceFieldsMatch(null, raw), false);
});
test('summary uses parsed source match rather than raw exact JSON equality', () => {
  const summary = summarize([{ ...fixedSchedule()[0], protocol: { passed: true }, domCopy: {
    complete: { parsedSourceFieldsMatch: true, sourceExactlyMatches: false } } }]);
  assert.equal(summary.byCondition['fresh-context'].endpoints.complete.responseParsedSourceFieldsMatchMirror, 1);
  assert.equal(Object.hasOwn(summary.byCondition['fresh-context'].endpoints.complete, 'responseProvenanceExactlyMatchesMirror'), false);
});
test('summary keeps fixed denominators, errors, null image misses and unattempted runs', () => {
  const summary = summarize([{ ...fixedSchedule()[0], protocol: { passed: true }, error: 'timeout', browser: {
    stages: { firstThree: { mirror: mirror('firstThree') } }, imageEndpoints: {} } }]);
  assert.equal(summary.fixedDenominator, 8); assert.equal(summary.unattempted.length, 7);
  assert.equal(summary.byCondition['fresh-context'].fixedDenominator, 4);
  assert.deepEqual(summary.byCondition['fresh-context'].endpoints.firstThree.reactAfterTwoRafMs, [123]);
  assert.deepEqual(summary.byCondition['fresh-context'].endpoints.firstThree.copyAndImagesDecodedFromEffectMs, [null]);
  assert.equal(summary.byCondition['fresh-context'].errors.length, 1);
});
test('failed protocol never contributes duplicate old durations even if raw mirrors remain', () => {
  const summary = summarize([{ ...fixedSchedule()[1], protocol: { passed: false, deviations: ['No POST'] }, browser: {
    stages: { complete: { mirror: mirror('complete', 99) } }, imageEndpoints: { complete: { fromEffectMs: 105 } } } }]);
  const arm = summary.byCondition['same-context-repeat'];
  assert.deepEqual(arm.endpoints.complete.reactAfterTwoRafMs, [null]);
  assert.deepEqual(arm.endpoints.complete.copyAndImagesDecodedFromEffectMs, [null]);
  assert.equal(arm.protocolPassed, 0); assert.equal(arm.protocolFailures.length, 1);
});
test('DOM agreement checks actual expected titles, bodies, badges, names and selected colors', () => {
  const cards = Array.from({ length: 6 }, (_, i) => ({ productId: `p${i}`, variantId: `v${i}`,
    originalName: `Name${i}`, title: `Fresh ${i}`, bodyCopy: `Useful ${i}`, badge: `Blue ${i}`, selectedColor: 'Blue' }));
  const snapshot = { cards: cards.map(c => ({ originalName: c.originalName, title: c.title,
    selectedColor: c.selectedColor, statusParagraphs: [{ text: c.bodyCopy }, { text: c.badge }] })) };
  const event = { type: 'complete', sections: [{ cards: cards.slice(0, 3) }, { cards: cards.slice(3) }] };
  assert.equal(matchDomToCopy(snapshot, event).passed, true);
  snapshot.cards[5].statusParagraphs = [{ text: 'original fallback' }];
  assert.equal(matchDomToCopy(snapshot, event).passed, false);
  assert.equal(matchDomToCopy(snapshot, { ...event, sections: [{ cards: cards.slice(0, 2) }] }).passed, false);
  assert.equal(matchDomToCopy(null, event).passed, false);
});
