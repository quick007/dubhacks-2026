import { createRequire } from 'node:module';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { QUERIES, TIMEOUT_MS, fixedSchedule, navigateAttempt, verifyRequestProtocol,
  installObserver, validFieldReceipt, parsedSourceFieldsMatch, matchDomToCopy, summarize } from './store_candidate_browser_probe_v1.mjs';
import { BODY_SETTLEMENT_MS, installFetchCapture, parseResponseBytes,
  selectResponseEvidence, settleBeforeNextQuery, canDispatchNext } from './store_candidate_collection_probe_v1.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
export const INTEGRATION_MANIFEST = 'experiments/ranker/pretrained/store_demo_integration_v10/manifest.json';
export const CONTEXT_OPTIONS = Object.freeze({ viewport: Object.freeze({ width: 1280, height: 960 }),
  deviceScaleFactor: 1, serviceWorkers: 'block', acceptDownloads: false });
const NATIVE = 'http://127.0.0.1:8766/personalize';
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const save = (path, value) => writeFile(path, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });

export function sourceInventory(manifest) {
  return [...new Set(['experiments/ranker/pretrained/store_candidate_browser_probe_v2.mjs',
    'experiments/ranker/pretrained/test_store_candidate_browser_probe_v2.mjs',
    'experiments/ranker/pretrained/store_candidate_browser_probe_v1.mjs',
    'experiments/ranker/pretrained/store_candidate_collection_probe_v1.mjs',
    'experiments/ranker/pretrained/store_candidate_image_service_v1.py',
    'experiments/ranker/pretrained/store_candidate_engine_v1.py',
    'experiments/ranker/pretrained/store_demo_engine.py',
    'experiments/ranker/pretrained/store_demo_service.py',
    'experiments/ranker/pretrained/store_rank_session.py',
    'experiments/ranker/pretrained/store_rank_session_v5.py', INTEGRATION_MANIFEST,
    ...Object.keys(manifest.frontendSha256 ?? {}), ...Object.keys(manifest.verificationSourcesSha256 ?? {}),
    ...Object.keys(manifest.rendererOnlyExceptions ?? {}), ...Object.keys(manifest.auxiliarySourceExceptions ?? {})])].sort();
}

export function verifyNavigationProtocol(row, seenIds = new Set(), seenDocuments = new Set()) {
  const base = verifyRequestProtocol({ ...row, modelResponses: [{ events: row.responseEvidence?.events ?? [] }] }, seenIds, seenDocuments);
  const deviations = [...base.deviations];
  if (row.bodySettlement?.settled !== true || row.cloneCaptureSettlement?.settled !== true)
    deviations.push('Actual response capture did not settle before next-navigation barrier');
  if (!row.responseEvidence?.selectedSource || row.responseEvidence.pathsAgree === false)
    deviations.push('No complete actual response bytes or actual byte paths disagree');
  if (row.browser?.devicePixelRatio !== 1) deviations.push('Observed document DPR differs from declared DPR1');
  return { ...base, passed: deviations.length === 0, deviations,
    scope: `${base.scope}; bounded settled actual-response bytes and observed DPR1 required` };
}

export async function settleNavigationCaptures(pending, waitForClone, timeoutMs = BODY_SETTLEMENT_MS) {
  const [bodySettlement, cloneCaptureSettlement] = await Promise.all([
    settleBeforeNextQuery(pending, timeoutMs),
    Promise.resolve().then(() => waitForClone(timeoutMs)).then(() => ({ settled: true }),
      error => ({ settled: false, error: String(error) })),
  ]);
  if (bodySettlement.started !== pending.length) {
    bodySettlement.settled = false;
    bodySettlement.reason = 'A CDP body capture began during the bounded barrier; do not navigate again';
    bodySettlement.startedAtBarrierEnd = pending.length;
  }
  return { bodySettlement, cloneCaptureSettlement };
}

export function domEvidence(browser, events) {
  return Object.fromEntries(['firstThree', 'complete'].map(stage => {
    const entry = browser?.stages?.[stage];
    const event = events.find(e => e.type === stage && e.requestId === entry?.mirror?.detail?.requestId);
    return [stage, { fieldReceiptValid: validFieldReceipt(entry?.mirror, stage),
      copySourceFromMirror: entry?.mirror?.detail?.copySource ?? null,
      copySourceFromActualResponse: event?.copySource ?? null,
      sourceExactlyMatches: Boolean(event && JSON.stringify(event.copySource) === JSON.stringify(entry?.mirror?.detail?.copySource)),
      parsedSourceFieldsMatch: parsedSourceFieldsMatch(entry?.mirror?.detail?.copySource, event?.copySource),
      sourceComparisonScope: 'Compare fields retained by frontend parser; raw actual-response provenance is preserved separately',
      matchAtStageObservation: matchDomToCopy(entry?.snapshot, event),
      matchAtFinalObservation: matchDomToCopy(browser?.last, event) }];
  }));
}

export function navigationSummary(rows) {
  return { ...summarize(rows), protocolPassed: rows.filter(r => r.protocol?.passed).length,
    responseCapture: rows.map(r => ({ index: r.index, selectedSource: r.responseEvidence?.selectedSource ?? null,
      pathsAgree: r.responseEvidence?.pathsAgree ?? null, selectedBodySha256: r.responseEvidence?.selectedBodySha256 ?? null,
      bodySettlement: r.bodySettlement ?? null, cloneCaptureSettlement: r.cloneCaptureSettlement ?? null,
      cdpErrors: (r.modelResponses ?? []).filter(p => p.error), cloneErrors: (r.cloneResponses ?? []).filter(p => p.error) })),
    requestFailures: rows.flatMap(r => (r.requestFailures ?? []).map(failure => ({ index: r.index, failure }))),
    pageErrors: rows.flatMap(r => (r.consoleErrors ?? []).map(error => ({ index: r.index, error }))),
    captureScope: 'Passive actual fetch clone, original Response unchanged, one assigned application POST; CDP and clone raw bytes/error receipts retained independently; no fabricated fallback' };
}

async function health() {
  const response = await fetch('http://127.0.0.1:8766/health', { signal: AbortSignal.timeout(5000) });
  const raw = await response.text();
  return { status: response.status, observedAt: new Date().toISOString(), bodySha256: sha(Buffer.from(raw)), body: JSON.parse(raw) };
}

export async function run(output, origin = 'http://127.0.0.1:4312') {
  if (origin !== 'http://127.0.0.1:4312') throw new Error('Only declared local storefront4312 is supported');
  await mkdir(output);
  const manifest = JSON.parse(await readFile(join(ROOT, INTEGRATION_MANIFEST), 'utf8'));
  const files = sourceInventory(manifest);
  const sourceHashes = Object.fromEntries(await Promise.all(files.map(async p => [p, sha(await readFile(join(ROOT, p)))])));
  const report = { version: 'store-candidate-browser-probe.v2', startedAt: new Date().toISOString(),
    sourceHashes, integrationManifest: { path: INTEGRATION_MANIFEST, sha256: sourceHashes[INTEGRATION_MANIFEST],
      version: manifest.version, imageDelivery: manifest.imageDelivery }, schedule: fixedSchedule(),
    timeoutMs: TIMEOUT_MS, bodySettlementTimeoutMs: BODY_SETTLEMENT_MS, viewport: CONTEXT_OPTIONS.viewport,
    deviceScaleFactor: 1, headed: true, attempts: [],
    workload: 'Unchanged v1 fixed eight: four fresh contexts, each goto then same-context reload; no scroll or click. Actual one POST/new matching request/document required. Geometry/parsed fields use v1 helpers unchanged.',
    timingScope: 'Original app effect-to-React/two-rAF and image complete/decode observation; navigation separately measured. Passive Response.clone overhead is part of this instrumented workload. No physical paint or useful-above-fold claim.',
    cacheScope: 'Fresh isolated context per pair, then same-context explicit reload; HTTP cache eligible, OS/native caches uncontrolled; no eager-image changes.',
    responseCaptureScope: 'Original application fetch arguments and Response unchanged; passive clone and CDP finished/body bytes collected concurrently with bounded five-second barrier; no next navigation after unsettled/invalid evidence.' };
  await save(join(output, 'plan.json'), report);
  let browser;
  const seenRequestIds = new Set(), seenDocumentIds = new Set();
  try {
    report.serviceHealthBefore = await health();
    if (report.serviceHealthBefore.body?.ready !== true) throw new Error('Native service is not ready before navigation');
    const require = createRequire(join(ROOT, 'apps/runner/package.json'));
    const { chromium } = require('playwright');
    browser = await chromium.launch({ channel: 'chrome', headless: false });
    report.browserVersion = browser.version();
    pairs: for (let pair = 0; pair < QUERIES.length; pair++) {
      const context = await browser.newContext({ ...CONTEXT_OPTIONS, viewport: { ...CONTEXT_OPTIONS.viewport } });
      try {
        await context.addInitScript(installFetchCapture);
        await context.addInitScript(installObserver);
        const page = await context.newPage();
        for (const item of report.schedule.filter(s => s.pair === pair)) {
          const row = { ...item, startedAt: new Date().toISOString(), requestFailures: [], consoleErrors: [],
            actualPostRequests: [], modelResponses: [], cloneResponses: [], navigationAction: item.condition === 'same-context-repeat' ? 'reload' : 'goto' };
          const pending = []; let open = true;
          const failed = req => row.requestFailures.push({ url: req.url(), resourceType: req.resourceType(), method: req.method(), error: req.failure() });
          const pageError = error => row.consoleErrors.push({ name: error.name, message: error.message, stack: error.stack ?? null });
          const request = req => {
            if (req.url() !== NATIVE || req.method() !== 'POST') return;
            let query = null; try { query = JSON.parse(req.postData() ?? '{}').query ?? null; } catch {}
            row.actualPostRequests.push({ url: req.url(), method: req.method(), postData: req.postData(), query, observedAtNodeMs: performance.now() });
          };
          const response = res => {
            if (res.url() !== NATIVE || res.request().method() !== 'POST') return;
            const receipt = { url: res.url(), status: res.status(), method: 'POST', request: res.request().postData(), captureStatus: 'pending' };
            row.modelResponses.push(receipt);
            pending.push((async () => {
              try {
                const finishedError = await res.finished(); if (finishedError) throw finishedError;
                const parsed = parseResponseBytes(await res.body(), { source: 'playwright-response-body' });
                if (open) Object.assign(receipt, parsed, { captureStatus: 'complete' });
              } catch (error) { if (open) Object.assign(receipt, { captureStatus: 'error', error: String(error) }); }
            })());
          };
          page.on('requestfailed', failed); page.on('pageerror', pageError); page.on('request', request); page.on('response', response);
          const started = performance.now();
          try {
            await navigateAttempt(page, item, `${origin}/?model=local#collection?q=${encodeURIComponent(item.query)}`,
              { waitUntil: 'domcontentloaded', timeout: TIMEOUT_MS });
            row.navigationDomContentLoadedMs = performance.now() - started;
            await page.waitForFunction(() => Boolean(window.__candidateProbe?.imageEndpoints?.complete), null,
              { timeout: Math.max(1, TIMEOUT_MS - (performance.now() - started)) });
          } catch (error) { row.error = String(error); }
          row.navigationToProbeEndMs = performance.now() - started;
          Object.assign(row, await settleNavigationCaptures(pending, timeout => page.waitForFunction(() => {
            const entries = window.__nativeFetchCapture?.requests ?? [];
            return entries.length > 0 && entries.every(r => r.status !== 'pending');
          }, null, { timeout })));
          open = false;
          try {
            const captures = await page.evaluate(() => window.__nativeFetchCapture?.requests ?? []);
            row.cloneResponses = captures.map(c => c.status === 'complete'
              ? parseResponseBytes(c.bodyBytes, { ...c, source: 'in-page-actual-response-clone' }) : c);
            row.browser = await page.evaluate(() => {
              window.__candidateProbe.stopped = true;
              return { ...window.__candidateProbe, devicePixelRatio: window.devicePixelRatio,
                navigation: performance.getEntriesByType('navigation').map(e => e.toJSON()),
                resources: performance.getEntriesByType('resource').map(e => e.toJSON()),
                readinessRaw: document.documentElement.getAttribute('data-local-model-readiness') };
            });
            const path = join(output, `attempt-${item.index + 1}.png`), bytes = await page.screenshot({ path, fullPage: false, timeout: 5000 });
            row.screenshot = { path, sha256: sha(bytes), bytes: bytes.length, status: 'captured' };
          } catch (error) { row.captureError = String(error); }
          if (row.error) await page.evaluate(() => window.stop()).catch(() => {});
          page.off('requestfailed', failed); page.off('pageerror', pageError); page.off('request', request); page.off('response', response);
          const requestId = row.browser?.stages?.firstThree?.mirror?.detail?.requestId;
          row.responseEvidence = selectResponseEvidence(row.modelResponses, row.cloneResponses, requestId);
          row.domCopy = domEvidence(row.browser, row.responseEvidence.events);
          row.protocol = verifyNavigationProtocol(row, seenRequestIds, seenDocumentIds);
          if (row.protocol.requestId) seenRequestIds.add(row.protocol.requestId);
          if (row.protocol.documentId) seenDocumentIds.add(row.protocol.documentId);
          report.attempts.push(row); await save(join(output, `attempt-${item.index + 1}.json`), row);
          if (!canDispatchNext(row)) {
            report.stoppedBeforeNextNavigation = { afterIndex: item.index, reason: 'Unsettled, failed or inconsistent actual response evidence; fixed remaining attempts unscored' };
            break pairs;
          }
        }
      } finally { await context.close(); }
    }
    report.serviceHealthAfter = await health();
    report.serviceStartupMetadataUnchanged = JSON.stringify(report.serviceHealthBefore.body?.startup)
      === JSON.stringify(report.serviceHealthAfter.body?.startup);
    const after = Object.fromEntries(await Promise.all(files.map(async p => [p, sha(await readFile(join(ROOT, p)))])));
    report.sourceHashesAfter = after;
    report.sourcesUnchanged = Object.entries(sourceHashes).every(([p, hash]) => after[p] === hash);
    if (!report.sourcesUnchanged) report.error = 'Probe/image/renderer/source bytes changed during fixed navigation workload';
    if (!report.serviceStartupMetadataUnchanged) report.error = 'Native service startup/checkpoint metadata changed during fixed navigation workload';
  } catch (error) { report.error = String(error); }
  finally {
    if (browser) await browser.close();
    report.finishedAt = new Date().toISOString(); report.summary = navigationSummary(report.attempts);
    await save(join(output, 'report.json'), report);
  }
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error('Usage: node store_candidate_browser_probe_v2.mjs NEW_OUTPUT_DIRECTORY');
  const report = await run(resolve(process.argv[2])); console.log(JSON.stringify(report.summary, null, 2));
  if (report.error || report.summary.protocolPassed !== 8 || report.attempts.some(r => r.error || r.captureError)) process.exitCode = 1;
}
