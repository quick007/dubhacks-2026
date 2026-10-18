import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { fixedSchedule, parsedSourceFieldsMatch } from './store_candidate_browser_probe_v1.mjs';
import { installFetchCapture, parseResponseBytes, selectResponseEvidence, canDispatchNext } from './store_candidate_collection_probe_v1.mjs';
import { CONTEXT_OPTIONS, INTEGRATION_MANIFEST, sourceInventory, verifyNavigationProtocol,
  settleNavigationCaptures, navigationSummary, domEvidence, run } from './store_candidate_browser_probe_v2.mjs';

const source = { kind: 'fresh-parallel-head', generatedEarlier: false, freshPerRequestGeneration: true };
const events = id => ['rank', 'firstThree', 'complete'].map(type => ({ type, requestId: id,
  copySource: { ...source, checkpointSha256: 'actual-head' } }));
const raw = id => Buffer.from(events(id).map(e => JSON.stringify(e)).join('\n') + '\n');
function row(index = 0) {
  const id = `request-${index}`;
  return { ...fixedSchedule()[index], actualPostRequests: [{ query: fixedSchedule()[index].query }],
    bodySettlement: { settled: true }, cloneCaptureSettlement: { settled: true },
    modelResponses: [{ error: 'CDP ERR_ABORTED retained independently' }],
    responseEvidence: selectResponseEvidence([], [parseResponseBytes(raw(id))], id),
    browser: { documentId: `document-${index}`, devicePixelRatio: 1,
      stages: Object.fromEntries(['rank', 'firstThree', 'complete'].map(stage => [stage,
        { mirror: { startTime: 10, duration: 20, detail: { requestId: id, stage, copySource: source,
          requiredFields: stage === 'firstThree' ? 10 : 20, acceptedFields: stage === 'firstThree' ? 10 : 20, withheldFields: 0 } } }])) } };
}

test('v2 retains fixed eight goto/reload schedule, explicit DPR1 and v10 source inventory', () => {
  assert.equal(fixedSchedule().length, 8); assert.equal(CONTEXT_OPTIONS.deviceScaleFactor, 1);
  assert.deepEqual(CONTEXT_OPTIONS.viewport, { width: 1280, height: 960 });
  const files = sourceInventory({ frontendSha256: { 'apps/storefront/src/product-image-runtime.json': 'sha',
    'apps/storefront/src/ProductPhoto.tsx': 'sha' }, verificationSourcesSha256: { 'scripts/prepare-product-runtime-projection.mjs': 'sha' } });
  for (const file of [INTEGRATION_MANIFEST, 'apps/storefront/src/product-image-runtime.json',
    'apps/storefront/src/ProductPhoto.tsx', 'scripts/prepare-product-runtime-projection.mjs',
    'experiments/ranker/pretrained/store_candidate_browser_probe_v1.mjs',
    'experiments/ranker/pretrained/store_candidate_collection_probe_v1.mjs']) assert.ok(files.includes(file));
  assert.equal(files.length, new Set(files).size);
});

test('passive clone preserves one application POST, original arguments/Response and actual bytes', async () => {
  let posts = 0; const response = new Response(raw('id')); let originalInit;
  const window = { fetch: (...args) => { posts++; originalInit = args[1]; return Promise.resolve(response); } };
  vm.runInNewContext(`(${installFetchCapture.toString()})()`, { window, URL,
    location: { href: 'http://127.0.0.1:4312/' }, performance: { now: () => 10 } });
  const init = { method: 'POST', body: JSON.stringify({ query: 'assigned' }) };
  const original = await window.fetch('http://127.0.0.1:8766/personalize', init);
  assert.equal(original, response); assert.equal(originalInit, init); assert.equal(posts, 1);
  assert.equal(await original.text(), raw('id').toString());
  for (let i = 0; i < 30 && window.__nativeFetchCapture.requests[0].status === 'pending'; i++) await new Promise(r => setTimeout(r, 2));
  const captured = window.__nativeFetchCapture.requests[0];
  assert.equal(captured.status, 'complete');
  assert.equal(parseResponseBytes(captured.bodyBytes).bodySha256, parseResponseBytes(raw('id')).bodySha256);
});

test('settled actual clone validates protocol despite independently retained CDP error', () => {
  const result = row(); result.protocol = verifyNavigationProtocol(result);
  assert.equal(result.protocol.passed, true); assert.equal(canDispatchNext(result), true);
  const summary = navigationSummary([result]);
  assert.equal(summary.responseCapture[0].cdpErrors.length, 1);
  assert.equal(summary.responseCapture[0].selectedSource, 'in-page-actual-response-clone');
  assert.equal(summary.fixedDenominator, 8); assert.equal(summary.unattempted.length, 7);
});

test('raw byte disagreement, missing stages, wrong POST and wrong DPR never validate', () => {
  const result = row();
  for (const mutate of [r => { r.actualPostRequests = []; }, r => { r.actualPostRequests[0].query = 'wrong'; },
    r => { r.browser.devicePixelRatio = 2; }, r => { r.responseEvidence.events = [{ type: 'complete', requestId: 'request-0' }]; },
    r => { r.responseEvidence = selectResponseEvidence([parseResponseBytes(raw('request-0'))],
      [parseResponseBytes(Buffer.concat([raw('request-0'), Buffer.from('\n')]))], 'request-0'); }]) {
    const bad = structuredClone(result); mutate(bad);
    assert.equal(verifyNavigationProtocol(bad).passed, false);
  }
  assert.equal(verifyNavigationProtocol(result, new Set(['request-0'])).passed, false);
  assert.equal(verifyNavigationProtocol(result, new Set(), new Set(['document-0'])).passed, false);
});

test('unsettled CDP or clone capture blocks further navigation with fixed remaining denominator', async () => {
  const never = new Promise(() => {});
  const unsettled = await settleNavigationCaptures([never], async () => {}, 5);
  assert.equal(unsettled.bodySettlement.settled, false);
  let result = { ...row(), ...unsettled }; result.protocol = verifyNavigationProtocol(result);
  assert.equal(canDispatchNext(result), false);
  const cloneFailed = await settleNavigationCaptures([Promise.resolve()], async () => { throw new Error('clone pending at deadline'); }, 5);
  assert.equal(cloneFailed.bodySettlement.settled, true); assert.equal(cloneFailed.cloneCaptureSettlement.settled, false);
  result = { ...row(), ...cloneFailed }; result.protocol = verifyNavigationProtocol(result);
  assert.equal(canDispatchNext(result), false);
  assert.equal(navigationSummary([result]).unattempted.length, 7);
});

test('a new pending CDP capture appearing inside the barrier cannot slip through', async () => {
  const pending = [];
  const settled = await settleNavigationCaptures(pending, async () => { pending.push(new Promise(() => {})); }, 5);
  assert.equal(settled.bodySettlement.settled, false);
  const result = { ...row(), ...settled }; result.protocol = verifyNavigationProtocol(result);
  assert.equal(canDispatchNext(result), false);
});

test('late body capture is rejected even when helper started-count observes mutated array', async () => {
  let finish;
  const pending = [new Promise(resolve => { finish = resolve; })];
  const settled = await settleNavigationCaptures(pending, async () => {
    pending.push(new Promise(() => {})); finish();
  }, 5);
  assert.equal(settled.bodySettlement.started, 2);
  assert.equal(settled.bodySettlement.outcomes.length, 1);
  assert.equal(settled.bodySettlement.settled, false);
});

test('both real byte paths agree by SHA while frontend parsed-source fields allow raw checkpoint provenance', () => {
  const parsed = parseResponseBytes(raw('request-0'));
  assert.equal(selectResponseEvidence([parsed], [parsed], 'request-0').pathsAgree, true);
  assert.equal(parsedSourceFieldsMatch(source, events('request-0')[1].copySource), true);
  const result = row(); const evidence = domEvidence(result.browser, result.responseEvidence.events);
  assert.equal(evidence.firstThree.parsedSourceFieldsMatch, true);
  assert.equal(evidence.firstThree.sourceExactlyMatches, false);
  assert.equal(evidence.complete.parsedSourceFieldsMatch, true);
  const corrupted = structuredClone(result.responseEvidence.events); corrupted[2].copySource.freshPerRequestGeneration = false;
  assert.equal(domEvidence(result.browser, corrupted).complete.parsedSourceFieldsMatch, false);
});

test('missing protocol evidence produces null timings, retained failures and fixed eight', () => {
  const bad = row(); bad.protocol = { passed: false }; bad.requestFailures = [{ url: 'native', error: 'aborted' }];
  const summary = navigationSummary([bad]);
  assert.equal(summary.fixedDenominator, 8); assert.equal(summary.protocolPassed, 0);
  assert.deepEqual(summary.byCondition['fresh-context'].endpoints.complete.reactAfterTwoRafMs, [null]);
  assert.equal(summary.requestFailures.length, 1);
});

test('unsupported origin is rejected before creating outputs or any browser', async () => {
  await assert.rejects(() => run('must-not-create-this-output', 'http://127.0.0.1:5173'), /4312/);
});
