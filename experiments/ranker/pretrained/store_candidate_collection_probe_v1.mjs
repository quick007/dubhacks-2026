import { createRequire } from 'node:module';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { QUERIES, TIMEOUT_MS, installObserver, validFieldReceipt, parsedSourceFieldsMatch, matchDomToCopy } from './store_candidate_browser_probe_v1.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const save = (path, value) => writeFile(path, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
export const BODY_SETTLEMENT_MS = 5000;

/** Passive tee of only the application's actual native fetch. No extra fetch,
 * no response replacement, no request/body mutation, no generated-text injection.
 * Clone work runs concurrently and is part of this instrumented workload. */
export function installFetchCapture() {
  if (window.__nativeFetchCapture) return;
  const original = window.fetch;
  const capture = window.__nativeFetchCapture = { version: 'passive-response-clone.v1', requests: [] };
  window.fetch = function (...args) {
    const [input, init] = args;
    const url = typeof input === 'string' ? new URL(input, location.href).href : input?.url ?? String(input);
    const method = String(init?.method ?? input?.method ?? 'GET').toUpperCase();
    if (url !== 'http://127.0.0.1:8766/personalize' || method !== 'POST') return Reflect.apply(original, this, args);
    const entry = { index: capture.requests.length, url, method, requestBody: typeof init?.body === 'string' ? init.body : null,
      startedAt: performance.now(), status: 'pending', responseStatus: null, bodyBytes: null, finishedAt: null, error: null };
    capture.requests.push(entry);
    return Reflect.apply(original, this, args).then(response => {
      entry.headersAt = performance.now(); entry.responseStatus = response.status;
      try {
        response.clone().arrayBuffer().then(buffer => {
          entry.bodyBytes = Array.from(new Uint8Array(buffer)); entry.finishedAt = performance.now(); entry.status = 'complete';
        }, error => { entry.error = String(error); entry.finishedAt = performance.now(); entry.status = 'error'; });
      } catch (error) { entry.error = String(error); entry.finishedAt = performance.now(); entry.status = 'error'; }
      return response;
    }, error => { entry.error = String(error); entry.finishedAt = performance.now(); entry.status = 'error'; throw error; });
  };
}
export function parseResponseBytes(bytes, metadata = {}) {
  const body = Buffer.from(bytes), receipt = { ...metadata, bodySha256: sha(body), bytes: body.length,
    raw: body.toString('utf8'), events: [], parseErrors: [] };
  for (const [lineIndex, line] of receipt.raw.split('\n').entries()) {
    if (!line.trim()) continue;
    try { receipt.events.push(JSON.parse(line)); } catch (e) { receipt.parseErrors.push({ lineIndex, error: String(e) }); }
  }
  return receipt;
}
export function selectResponseEvidence(cdp, clones, requestId) {
  const complete = r => typeof requestId === 'string' && requestId.length > 0 && r.parseErrors?.length === 0 && ['rank', 'firstThree', 'complete'].every(stage =>
    r.events?.some(e => e.type === stage && e.requestId === requestId));
  const cdpMatch = cdp.find(complete), cloneMatch = clones.find(complete);
  const pathsAgree = cdpMatch && cloneMatch ? cdpMatch.bodySha256 === cloneMatch.bodySha256 : null;
  const chosen = pathsAgree === false ? null : cdpMatch ?? cloneMatch;
  return { selectedSource: chosen ? (chosen === cdpMatch ? 'playwright-response-body' : 'in-page-actual-response-clone') : null,
    pathsAgree, selectedBodySha256: chosen?.bodySha256 ?? null, events: chosen?.events ?? [],
    scope: 'Only bytes from the actual POST response; raw CDP and passive in-page clone paths retained independently' };
}
export async function settleBeforeNextQuery(pending, timeoutMs = BODY_SETTLEMENT_MS) {
  let timer;
  try {
    return await Promise.race([
      Promise.allSettled(pending).then(results => ({ settled: true, started: pending.length, outcomes: results.map(r => r.status) })),
      new Promise(resolve => { timer = setTimeout(() => resolve({ settled: false, started: pending.length,
        reason: 'Native CDP response finished/body capture did not settle before next-query barrier' }), timeoutMs); }),
    ]);
  } finally { clearTimeout(timer); }
}
export function canDispatchNext(row) {
  return !row.error && row.protocol?.passed === true && row.bodySettlement?.settled === true
    && row.cloneCaptureSettlement?.settled === true && Boolean(row.responseEvidence?.selectedSource)
    && row.responseEvidence.pathsAgree !== false;
}

/** Entire first three cards, their titles/copy and clipped primary images must
 * fit the viewport. This is geometry plus decoding, never physical paint. */
export function visibleThreeReady(snapshot) {
  const cards = snapshot?.cards?.slice(0, 3) ?? [];
  return cards.length === 3 && cards.every(c => c.rect?.fullyWithinViewport
    && c.titleRect?.fullyWithinViewport && c.statusParagraphs?.length >= 2
    && c.statusParagraphs.every(p => p.rect?.fullyWithinViewport)
    && c.image?.clippedPrimaryRect?.fullyWithinViewport && c.image.complete
    && c.image.naturalWidth > 0 && c.image.decodeStatus === 'decoded');
}
// Passed as a self-contained browser function; same predicate as tested above.
export function installVisibleObserver() {
  const receipt = window.__collectionVisibility = { endpoint: null, stopped: false };
  const poll = () => {
    if (receipt.stopped) return;
    const state = window.__candidateProbe, entry = state?.stages?.firstThree;
    const cards = state?.last?.cards?.slice(0, 3) ?? [];
    if (!receipt.endpoint && entry && entry.mirror.detail.requiredFields === 10
      && entry.mirror.detail.acceptedFields === 10 && entry.mirror.detail.withheldFields === 0
      && cards.length === 3 && cards.every(c => c.rect?.fullyWithinViewport
        && c.titleRect?.fullyWithinViewport && c.statusParagraphs?.length >= 2
        && c.statusParagraphs.every(p => p.rect?.fullyWithinViewport)
        && c.image?.clippedPrimaryRect?.fullyWithinViewport && c.image.complete
        && c.image.naturalWidth > 0 && c.image.decodeStatus === 'decoded')) {
      const observedAt = performance.now();
      receipt.endpoint = { observedAt, fromEffectMs: observedAt - entry.mirror.startTime,
        requestId: entry.mirror.detail.requestId, mirror: entry.mirror, snapshot: state.last,
        scope: 'After application two-rAF readiness, all first-three entire card/title/copy/primary-image rectangles inside viewport, primary images complete and successfully decoded; physical paint unmeasured' };
    }
    requestAnimationFrame(poll);
  };
  requestAnimationFrame(poll);
}
export async function beginScoredQuery(page, query) {
  const baseline = await page.evaluate(() => {
    const state = window.__candidateProbe;
    state.stopped = true;
    const raw = document.documentElement.getAttribute('data-local-model-readiness');
    const documentId = state.documentId;
    document.documentElement.removeAttribute('data-local-model-readiness');
    return { raw, documentId, previousHash: location.hash, beforeResetAt: performance.now() };
  });
  await page.evaluate(installObserver);
  await page.evaluate(documentId => {
    window.__candidateProbe.observerId = window.__candidateProbe.documentId;
    window.__candidateProbe.documentId = documentId;
  }, baseline.documentId);
  await page.evaluate(installVisibleObserver);
  const dispatch = await page.evaluate(queryValue => {
    const dispatchedAt = performance.now();
    location.hash = `#collection?q=${encodeURIComponent(queryValue)}`;
    return { dispatchedAt, targetHash: location.hash };
  }, query);
  return { baseline, dispatch };
}
export function collectionProtocol(row, seenIds) {
  const deviations = [];
  if (row.actualPostRequests?.length !== 1 || row.actualPostRequests[0]?.query !== row.query) deviations.push('Expected one actual POST with assigned new query');
  const first = row.browser?.stages?.firstThree?.mirror, complete = row.browser?.stages?.complete?.mirror;
  const id = first?.detail?.requestId;
  if (!id || complete?.detail?.requestId !== id || seenIds.has(id)) deviations.push('Missing, mismatched or reused scored request ID');
  for (const stage of ['rank', 'firstThree', 'complete']) {
    if (!row.events?.some(e => e.type === stage && e.requestId === id)) deviations.push(`No actual matching ${stage} response event`);
  }
  if (!Number.isFinite(first?.startTime) || !Number.isFinite(row.reset?.dispatch?.dispatchedAt)
    || first.startTime < row.reset.dispatch.dispatchedAt) deviations.push('Missing fresh effect clock or mirror begins before query dispatch');
  if (row.bodySettlement?.settled !== true || row.cloneCaptureSettlement?.settled !== true) deviations.push('Response capture did not settle before next-query barrier');
  if (!row.responseEvidence?.selectedSource || row.responseEvidence.pathsAgree === false) deviations.push('No complete actual response bytes or capture paths disagree');
  return { passed: deviations.length === 0, deviations, requestId: id ?? null };
}
export function collectionSummary(rows) {
  return { fixedDenominator: 4, recorded: rows.length, unattempted: QUERIES.filter(q => !rows.some(r => r.query === q)),
    protocolPassed: rows.filter(r => r.protocol?.passed).length,
    fullyVisibleThreeValidated: rows.filter(r => r.protocol?.passed && r.endpointDomMatch?.passed && r.endpointParsedSourceFieldsMatch
      && visibleThreeReady(r.visible?.endpoint?.snapshot)).length,
    effectToFullyVisibleThreeCopyAndDecodedImagesMs: rows.map(r => r.protocol?.passed && r.endpointDomMatch?.passed
      && r.endpointParsedSourceFieldsMatch && visibleThreeReady(r.visible?.endpoint?.snapshot) ? r.visible.endpoint.fromEffectMs : null),
    failures: rows.filter(r => r.error || !r.protocol?.passed).map(r => ({ query: r.query, error: r.error ?? null, protocol: r.protocol })),
    pageErrors: rows.flatMap(r => (r.pageErrors ?? []).map(error => ({ query: r.query, error }))),
    requestFailures: rows.flatMap(r => (r.requestFailures ?? []).map(failure => ({ query: r.query, failure }))),
    responseCapture: rows.map(r => ({ query: r.query, selectedSource: r.responseEvidence?.selectedSource ?? null,
      pathsAgree: r.responseEvidence?.pathsAgree ?? null, bodySettlement: r.bodySettlement ?? null, cloneCaptureSettlement: r.cloneCaptureSettlement ?? null,
      cdpErrors: (r.responses ?? []).filter(p => p.error), cloneErrors: (r.cloneResponses ?? []).filter(p => p.error) })),
    residualErrorScope: 'Page errors and all request failures retained separately from request protocol; no origin/cause attribution without evidence',
    scope: 'Loaded collection-view workload after unscored bootstrap and pre-scroll; fixed order, browser/native caches uncontrolled; physical paint not measured' };
}

export async function runCollection(output) {
  await mkdir(output);
  const files = ['experiments/ranker/pretrained/store_candidate_collection_probe_v1.mjs', 'experiments/ranker/pretrained/store_candidate_browser_probe_v1.mjs',
    'apps/storefront/src/local-model-performance.ts', 'apps/storefront/src/ShopView.tsx', 'apps/storefront/src/ProductCard.tsx'];
  const report = { version: 'store-candidate-collection-probe.v1', startedAt: new Date().toISOString(), queries: QUERIES,
    sourceHashes: Object.fromEntries(await Promise.all(files.map(async p => [p, sha(await readFile(join(ROOT, p)))]))),
    viewport: { width: 1280, height: 960 }, headed: true, timeoutMs: TIMEOUT_MS, attempts: [], pageErrors: [],
    workload: 'ONE unscored Browse the store bootstrap, then four sequential scored query hash changes on same loaded page. Before each query scroll collection top without smooth behavior; reset observer and mirror baseline. No form click or post-dispatch scroll. This is separate from cold navigation and never pooled with it.',
    totalNativeRequestsPlanned: 5, physicalPaintMeasured: false,
    responseCaptureScope: 'Passive actual-fetch Response.clone arrayBuffer tee installed before bootstrap; no extra POST or body mutation. Wrapper/clone overhead is included in measured effect-to-readiness. Both CDP and clone raw bytes retained. No next hash dispatch before bounded capture settlement; stop on pending evidence.',
    bodySettlementTimeoutMs: BODY_SETTLEMENT_MS };
  await save(join(output, 'plan.json'), report);
  let browser;
  try {
    const require = createRequire(join(ROOT, 'apps/runner/package.json'));
    const { chromium } = require('playwright');
    browser = await chromium.launch({ channel: 'chrome', headless: false });
    report.browserVersion = browser.version();
    const context = await browser.newContext({ viewport: report.viewport, serviceWorkers: 'block', acceptDownloads: false });
    await context.addInitScript(installFetchCapture);
    await context.addInitScript(installObserver);
    const page = await context.newPage();
    page.on('pageerror', error => report.pageErrors.push({ atNodeMs: performance.now(), name: error.name,
      message: error.message, stack: error.stack ?? null }));
    await page.goto('http://127.0.0.1:4312/?model=local#collection?q=Browse%20the%20store', { waitUntil: 'domcontentloaded', timeout: TIMEOUT_MS });
    await page.waitForFunction(() => Boolean(window.__candidateProbe?.stages?.complete), null, { timeout: TIMEOUT_MS });
    await page.waitForFunction(() => window.__nativeFetchCapture?.requests?.length > 0
      && window.__nativeFetchCapture.requests.every(r => r.status !== 'pending'), null, { timeout: BODY_SETTLEMENT_MS });
    report.unscoredBootstrapClone = await page.evaluate(() => window.__nativeFetchCapture);
    report.unscoredBootstrap = await page.evaluate(() => window.__candidateProbe);
    report.unscoredBootstrapPageErrors = [...report.pageErrors];
    const seenIds = new Set(Object.values(report.unscoredBootstrap.stages).map(s => s.mirror.detail.requestId));
    for (const [index, query] of QUERIES.entries()) {
      const row = { index, query, actualPostRequests: [], responses: [], events: [], requestFailures: [] };
      const pageErrorStart = report.pageErrors.length;
      const pending = []; let open = true;
      const request = req => { if (req.url() !== 'http://127.0.0.1:8766/personalize' || req.method() !== 'POST') return;
        let q = null; try { q = JSON.parse(req.postData() ?? '{}').query ?? null; } catch {}
        row.actualPostRequests.push({ query: q, raw: req.postData() }); };
      const failed = req => row.requestFailures.push({ url: req.url(), resourceType: req.resourceType(),
        method: req.method(), error: req.failure(), observedAtNodeMs: performance.now() });
      const response = res => { if (res.url() !== 'http://127.0.0.1:8766/personalize' || res.request().method() !== 'POST') return;
        pending.push((async () => { try {
          const finishedError = await res.finished();
          if (finishedError) throw finishedError;
          const receipt = parseResponseBytes(await res.body(), { status: res.status(), source: 'playwright-response-body' });
          if (open) row.responses.push(receipt);
        } catch (e) { if (open) row.responses.push({ error: String(e) }); } })()); };
      await page.evaluate(() => document.getElementById('collection').scrollIntoView({ block: 'start', behavior: 'instant' }));
      await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
      page.on('request', request); page.on('response', response); page.on('requestfailed', failed);
      const nodeStart = performance.now();
      try {
        row.reset = await beginScoredQuery(page, query);
        await page.waitForFunction(() => Boolean(window.__collectionVisibility?.endpoint && window.__candidateProbe?.stages?.complete), null,
          { timeout: Math.max(1, TIMEOUT_MS - (performance.now() - nodeStart)) });
      } catch (e) { row.error = String(e); }
      row.nodeDispatchAndObservationMs = performance.now() - nodeStart;
      try {
        await page.waitForFunction(({ query, dispatchedAt }) => {
          const rows = window.__nativeFetchCapture?.requests?.filter(r => {
            try { return JSON.parse(r.requestBody ?? '{}').query === query && r.startedAt >= dispatchedAt; } catch { return false; }
          }) ?? [];
          return rows.length > 0 && rows.every(r => r.status !== 'pending');
        }, { query, dispatchedAt: row.reset?.dispatch?.dispatchedAt ?? Infinity }, { timeout: BODY_SETTLEMENT_MS });
        row.cloneCaptureSettlement = { settled: true };
      } catch (e) { row.cloneCaptureSettlement = { settled: false, error: String(e) }; }
      row.bodySettlement = await settleBeforeNextQuery(pending);
      open = false;
      try {
        const captures = await page.evaluate(({ query, dispatchedAt }) => window.__nativeFetchCapture.requests.filter(r => {
          try { return JSON.parse(r.requestBody ?? '{}').query === query && r.startedAt >= dispatchedAt; } catch { return false; }
        }), { query, dispatchedAt: row.reset?.dispatch?.dispatchedAt ?? Infinity });
        row.cloneResponses = captures.map(c => c.status === 'complete' ? parseResponseBytes(c.bodyBytes,
          { ...c, source: 'in-page-actual-response-clone' }) : c);
      } catch (e) { row.cloneResponses = [{ error: String(e) }]; }
      try {
        const captured = await page.evaluate(() => { window.__candidateProbe.stopped = true; window.__collectionVisibility.stopped = true;
          return { browser: window.__candidateProbe, visible: window.__collectionVisibility }; });
        Object.assign(row, captured);
        const path = join(output, `query-${index + 1}.png`), bytes = await page.screenshot({ path, fullPage: false, timeout: 5000 });
        row.screenshot = { path, sha256: sha(bytes), bytes: bytes.length };
      } catch (e) { row.captureError = String(e); }
      if (row.error) await page.evaluate(() => window.stop()).catch(() => {});
      row.pageErrors = report.pageErrors.slice(pageErrorStart);
      page.off('request', request); page.off('response', response); page.off('requestfailed', failed);
      row.responseEvidence = selectResponseEvidence(row.responses, row.cloneResponses, row.browser?.stages?.firstThree?.mirror?.detail?.requestId);
      row.events = row.responseEvidence.events;
      row.protocol = collectionProtocol(row, seenIds); if (row.protocol.requestId) seenIds.add(row.protocol.requestId);
      const firstEvent = row.events.find(e => e.type === 'firstThree' && e.requestId === row.protocol.requestId);
      row.endpointDomMatch = matchDomToCopy(row.visible?.endpoint?.snapshot, firstEvent);
      row.endpointParsedSourceFieldsMatch = parsedSourceFieldsMatch(row.visible?.endpoint?.mirror?.detail?.copySource, firstEvent?.copySource);
      row.completeFieldReceiptValid = validFieldReceipt(row.browser?.stages?.complete?.mirror, 'complete');
      const completeEvent = row.events.find(e => e.type === 'complete' && e.requestId === row.protocol.requestId);
      row.completeDomMatch = matchDomToCopy(row.browser?.stages?.complete?.snapshot, completeEvent);
      report.attempts.push(row); await save(join(output, `query-${index + 1}.json`), row);
      // This barrier runs before the loop can call beginScoredQuery again.
      if (!canDispatchNext(row)) {
        report.stoppedBeforeNextQuery = { afterIndex: index, reason: 'Timeout, failed protocol or unsettled/missing/inconsistent actual response evidence' };
        break;
      }
    }
  } catch (e) { report.error = String(e); }
  finally { if (browser) await browser.close(); report.finishedAt = new Date().toISOString(); report.summary = collectionSummary(report.attempts);
    await save(join(output, 'report.json'), report); }
  return report;
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error('Usage: node store_candidate_collection_probe_v1.mjs NEW_OUTPUT_DIRECTORY');
  const report = await runCollection(resolve(process.argv[2])); console.log(JSON.stringify(report.summary, null, 2));
  if (report.error || report.summary.fullyVisibleThreeValidated !== 4) process.exitCode = 1;
}
