import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { visibleThreeReady, installVisibleObserver, collectionProtocol, collectionSummary, installFetchCapture,
  parseResponseBytes, selectResponseEvidence, settleBeforeNextQuery, canDispatchNext } from './store_candidate_collection_probe_v1.mjs';

const rect = { fullyWithinViewport: true };
const snapshot = () => ({ cards: Array.from({ length: 6 }, () => ({ rect, titleRect: rect,
  statusParagraphs: [{ text: 'Badge', rect }, { text: 'Body', rect }],
  image: { clippedPrimaryRect: rect, complete: true, naturalWidth: 100, decodeStatus: 'decoded' } })) });
test('visible first-three gate requires full card, title/copy geometry and successful image decode', () => {
  const s = snapshot(); assert.equal(visibleThreeReady(s), true);
  s.cards[0].rect = { fullyWithinViewport: false }; assert.equal(visibleThreeReady(s), false);
  s.cards[0].rect = rect; s.cards[2].image.complete = false; assert.equal(visibleThreeReady(s), false);
  s.cards[2].image.complete = true; s.cards[1].statusParagraphs[1].rect = { fullyWithinViewport: false };
  assert.equal(visibleThreeReady(s), false);
});
test('serialized browser visibility observer records same bounded predicate and effect-relative clock', () => {
  const queue = [], window = { __candidateProbe: { last: snapshot(), stages: { firstThree: {
    mirror: { startTime: 100, detail: { requestId: 'new', requiredFields: 10, acceptedFields: 10, withheldFields: 0 } } } } } };
  vm.runInNewContext(`(${installVisibleObserver.toString()})()`, { window, requestAnimationFrame: cb => queue.push(cb), performance: { now: () => 150 } });
  queue.shift()(); assert.equal(window.__collectionVisibility.endpoint.fromEffectMs, 50);
  assert.equal(window.__collectionVisibility.endpoint.requestId, 'new');
  assert.equal(visibleThreeReady(window.__collectionVisibility.endpoint.snapshot), true);
});
test('collection protocol requires genuine new matching POST and post-dispatch effect', () => {
  const row = { query: 'test', actualPostRequests: [{ query: 'test' }], reset: { dispatch: { dispatchedAt: 20 } },
    bodySettlement: { settled: true }, cloneCaptureSettlement: { settled: true }, responseEvidence: { selectedSource: 'in-page-actual-response-clone', pathsAgree: null },
    events: ['rank', 'firstThree', 'complete'].map(type => ({ type, requestId: 'id' })),
    browser: { stages: Object.fromEntries(['firstThree', 'complete'].map(stage => [stage,
      { mirror: { startTime: 30, detail: { requestId: 'id' } } }])) } };
  assert.equal(collectionProtocol(row, new Set()).passed, true);
  assert.equal(collectionProtocol(row, new Set(['id'])).passed, false);
  assert.equal(collectionProtocol({ ...row, actualPostRequests: [] }, new Set()).passed, false);
  assert.equal(collectionProtocol({ ...row, reset: { dispatch: { dispatchedAt: 40 } } }, new Set()).passed, false);
});
test('passive fetch clone sends one original request, returns original Response and captures actual bytes', async () => {
  const raw = ['rank', 'firstThree', 'complete'].map(type => JSON.stringify({ type, requestId: 'id' })).join('\n') + '\n';
  let calls = 0, receivedArgs;
  const response = new Response(raw, { status: 200 });
  const window = { fetch: (...args) => { calls++; receivedArgs = args; return Promise.resolve(response); } };
  vm.runInNewContext(`(${installFetchCapture.toString()})()`, { window, location: { href: 'http://127.0.0.1:4312/' },
    performance: { now: () => 10 }, URL });
  const init = { method: 'POST', body: JSON.stringify({ query: 'test' }) };
  const actual = await window.fetch('http://127.0.0.1:8766/personalize', init);
  assert.equal(actual, response); assert.equal(calls, 1); assert.equal(receivedArgs[1], init);
  assert.equal(await actual.text(), raw);
  for (let i = 0; i < 20 && window.__nativeFetchCapture.requests[0].status === 'pending'; i++) await new Promise(r => setTimeout(r, 5));
  const captured = window.__nativeFetchCapture.requests[0];
  assert.equal(captured.status, 'complete');
  const parsed = parseResponseBytes(captured.bodyBytes);
  assert.equal(parsed.raw, raw); assert.equal(parsed.events.length, 3);
  assert.equal(selectResponseEvidence([{ error: 'CDP body unavailable' }], [parsed], 'id').selectedSource, 'in-page-actual-response-clone');
});
test('actual byte paths must agree; incomplete or malformed bytes cannot replace missing evidence', () => {
  const raw = ['rank', 'firstThree', 'complete'].map(type => JSON.stringify({ type, requestId: 'id' })).join('\n');
  const first = parseResponseBytes(Buffer.from(raw));
  const second = parseResponseBytes(Buffer.from(raw + '\n'));
  assert.equal(selectResponseEvidence([first], [first], 'id').pathsAgree, true);
  const mismatch = selectResponseEvidence([first], [second], 'id');
  assert.equal(mismatch.pathsAgree, false); assert.equal(mismatch.selectedSource, null);
  assert.equal(selectResponseEvidence([], [parseResponseBytes(Buffer.from('{bad}\n'))], 'id').selectedSource, null);
});
test('unsettled body barrier blocks next dispatch rather than allowing stream contamination', async () => {
  const never = new Promise(() => {});
  const bodySettlement = await settleBeforeNextQuery([never], 5);
  assert.equal(bodySettlement.settled, false);
  const base = { protocol: { passed: true }, bodySettlement, cloneCaptureSettlement: { settled: true },
    responseEvidence: { selectedSource: 'in-page-actual-response-clone', pathsAgree: null } };
  assert.equal(canDispatchNext(base), false);
  assert.equal(canDispatchNext({ ...base, bodySettlement: await settleBeforeNextQuery([Promise.resolve()], 5) }), true);
  assert.equal(canDispatchNext({ ...base, bodySettlement: { settled: true }, cloneCaptureSettlement: { settled: false } }), false);
});
test('collection summary preserves fixed four and nulls geometry/protocol/DOM failures', () => {
  const base = { query: 'a light for my bedside table', protocol: { passed: true }, endpointDomMatch: { passed: true },
    endpointParsedSourceFieldsMatch: true, visible: { endpoint: { snapshot: snapshot(), fromEffectMs: 300 } } };
  const valid = collectionSummary([base]); assert.equal(valid.fixedDenominator, 4); assert.equal(valid.unattempted.length, 3);
  assert.deepEqual(valid.effectToFullyVisibleThreeCopyAndDecodedImagesMs, [300]);
  const bad = collectionSummary([{ ...base, endpointDomMatch: { passed: false } }, { ...base, protocol: { passed: false } }]);
  assert.deepEqual(bad.effectToFullyVisibleThreeCopyAndDecodedImagesMs, [null, null]);
});
test('residual page errors and failed resources remain visible even with passing request protocol', () => {
  const summary = collectionSummary([{ query: 'test', protocol: { passed: true },
    pageErrors: [{ message: 'WebSocket closed without opened.', stack: 'observed stack' }],
    requestFailures: [{ url: 'http://127.0.0.1:4312/image.webp', resourceType: 'image', error: { errorText: 'failed' } }] }]);
  assert.equal(summary.protocolPassed, 1);
  assert.equal(summary.pageErrors[0].error.stack, 'observed stack');
  assert.equal(summary.requestFailures[0].failure.resourceType, 'image');
  assert.equal(summary.fullyVisibleThreeValidated, 0);
});
