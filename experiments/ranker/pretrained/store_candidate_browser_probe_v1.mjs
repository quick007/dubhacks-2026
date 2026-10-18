import { createRequire } from 'node:module';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '../../..');
export const QUERIES = Object.freeze([
  'a light for my bedside table', 'a bottle for drinking water',
  'a notebook for quick notes', 'a bag for carrying my things',
]);
export const TIMEOUT_MS = 30_000;
export function fixedSchedule() {
  return QUERIES.flatMap((query, pair) => ['fresh-context', 'same-context-repeat'].map((condition, repeat) =>
    ({ index: pair * 2 + repeat, pair, query, condition })));
}
export async function navigateAttempt(page, item, url, options) {
  if (item.condition === 'same-context-repeat') return page.reload(options);
  return page.goto(url, options);
}
export function verifyRequestProtocol(result, seenRequestIds = new Set(), seenDocumentIds = new Set()) {
  const deviations = [];
  if (result.actualPostRequests?.length !== 1) deviations.push('Expected exactly one actual native POST for this scored navigation');
  if (result.actualPostRequests?.some(p => p.query !== result.query)) deviations.push('Actual native POST query differs from assigned query');
  const events = result.modelResponses?.flatMap(r => r.events ?? []) ?? [];
  const ids = ['rank', 'firstThree', 'complete'].map(stage => result.browser?.stages?.[stage]?.mirror?.detail?.requestId);
  const requestId = ids[0];
  if (typeof requestId !== 'string' || !requestId || ids.some(id => id !== requestId)) deviations.push('Missing or inconsistent current rank/firstThree/complete mirror request ID');
  if (typeof requestId === 'string' && seenRequestIds.has(requestId)) deviations.push('Request ID was already observed in a previous scored attempt');
  for (const stage of ['rank', 'firstThree', 'complete']) {
    if (!events.some(e => e.type === stage && e.requestId === requestId)) deviations.push(`Actual POST response lacks matching ${stage} request ID`);
  }
  const documentId = result.browser?.documentId;
  if (typeof documentId !== 'string' || !documentId || seenDocumentIds.has(documentId)) deviations.push('Scored navigation did not produce a new observed document');
  return { passed: deviations.length === 0, deviations, requestId: requestId ?? null, documentId: documentId ?? null,
    scope: 'Exactly one matching actual POST, all three response/mirror stages share a unique request ID, and a new document per navigation' };
}
export function selectReadiness(entries, stage) {
  return entries.filter(e => e?.detail?.stage === stage && Number.isFinite(e.startTime) && Number.isFinite(e.duration))
    .sort((a, b) => b.startTime - a.startTime)[0] ?? null;
}
export function validFieldReceipt(entry, stage) {
  const required = stage === 'firstThree' ? 10 : 20;
  return entry?.detail?.requiredFields === required && entry.detail.acceptedFields === required
    && entry.detail.withheldFields === 0;
}
/** Compare the frontend's retained fields, while allowing raw native checkpoint/
 * feature provenance that its parser deliberately omits. Raw evidence is saved. */
export function parsedSourceFieldsMatch(mirror, raw) {
  if (!mirror || !raw || typeof mirror !== 'object' || typeof raw !== 'object'
    || Array.isArray(mirror) || Array.isArray(raw)) return false;
  const flags = { 'fresh-native': [false, true], 'fresh-parallel-head': [false, true],
    'generated-earlier': [true, false], 'mixed-native-cache': [true, true], 'no-generation': [false, false] };
  if (!Object.hasOwn(flags, mirror.kind) || raw.kind !== mirror.kind) return false;
  const [generatedEarlier, freshPerRequestGeneration] = flags[mirror.kind];
  for (const source of [mirror, raw]) {
    if (source.generatedEarlier !== generatedEarlier || source.freshPerRequestGeneration !== freshPerRequestGeneration) return false;
  }
  return Object.entries(mirror).every(([key, value]) => Object.hasOwn(raw, key) && Object.is(raw[key], value));
}
export function summarize(attempts) {
  const byCondition = {};
  for (const condition of ['fresh-context', 'same-context-repeat']) {
    const rows = attempts.filter(a => a.condition === condition);
    byCondition[condition] = { fixedDenominator: 4, attempted: rows.length,
      protocolPassed: rows.filter(r => r.protocol?.passed === true).length,
      protocolFailures: rows.filter(r => r.protocol?.passed !== true).map(r => ({ index: r.index, protocol: r.protocol ?? null })),
      errors: rows.filter(r => r.error).map(r => ({ index: r.index, error: r.error })),
      endpoints: Object.fromEntries(['firstThree', 'complete'].map(stage => [stage, {
        mirrorAllFieldsAccepted: rows.filter(r => r.protocol?.passed === true && validFieldReceipt(r.browser?.stages?.[stage]?.mirror, stage)).length,
        responseCopyMatchesDomAtStage: rows.filter(r => r.protocol?.passed === true && r.domCopy?.[stage]?.matchAtStageObservation?.passed === true).length,
        responseParsedSourceFieldsMatchMirror: rows.filter(r => r.protocol?.passed === true && r.domCopy?.[stage]?.parsedSourceFieldsMatch === true).length,
        reactAfterTwoRafMs: rows.map(r => r.protocol?.passed === true ? r.browser?.stages?.[stage]?.mirror?.duration ?? null : null),
        copyAndImagesDecodedFromEffectMs: rows.map(r => r.protocol?.passed === true ? r.browser?.imageEndpoints?.[stage]?.fromEffectMs ?? null : null),
        missingOrUnscoredImageEndpoint: rows.filter(r => r.protocol?.passed !== true || !r.browser?.imageEndpoints?.[stage]).map(r => r.index),
      }])) };
  }
  return { fixedDenominator: 8, recorded: attempts.length, unattempted: fixedSchedule().filter(s => !attempts.some(a => a.index === s.index)),
    byCondition, aggregation: 'fixed-order protocol-verified timings; failed/missing protocol yields null, raw evidence retained per attempt; no successful-only latency average',
    physicalPaintMeasured: false };
}

// Installed before navigation. Observes application DOM and existing mirror only;
// never fetches model output, scrolls, changes lazy loading, or decodes unloaded images.
export function installObserver() {
  const state = window.__candidateProbe = { documentId: crypto.randomUUID(), stages: {}, imageEndpoints: {}, last: null, stopped: false, decodeErrors: [] };
  const decodeStates = new WeakMap();
  const visible = r => r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth;
  const rect = el => { if (!el) return null; const r = el.getBoundingClientRect();
    return { x: r.x, y: r.y, width: r.width, height: r.height, top: r.top, bottom: r.bottom,
      intersectsViewport: visible(r), fullyWithinViewport: visible(r) && r.top >= 0 && r.left >= 0 && r.bottom <= innerHeight && r.right <= innerWidth }; };
  const snapshot = () => {
    const cards = [...document.querySelectorAll('#collection article.product')].slice(0, 6).map((el, index) => {
      const img = el.querySelector('.product-art img');
      let decoded = img && decodeStates.get(img);
      const src = img?.currentSrc || img?.src || '';
      if (img && img.complete && img.naturalWidth > 0 && (!decoded || decoded.src !== src)) {
        decoded = { src, status: 'pending', at: null }; decodeStates.set(img, decoded);
        img.decode().then(() => { decoded.status = 'decoded'; decoded.at = performance.now(); }, error => {
          decoded.status = 'error'; decoded.error = String(error); state.decodeErrors.push({ src, error: String(error) });
        });
      }
      return { index, originalName: el.querySelector('.product-colors')?.getAttribute('aria-label')?.replace(/ colors$/, '') ?? null,
        title: el.querySelector('h3')?.textContent?.trim() ?? null,
        statusParagraphs: [...el.querySelectorAll(':scope > p.search-status')].map(p => ({ text: p.textContent?.trim(), rect: rect(p) })),
        selectedColor: el.querySelector('.selected-color')?.textContent?.trim() ?? null,
        rect: rect(el), titleRect: rect(el.querySelector('h3')),
        image: img ? { src, loading: img.loading, complete: img.complete, naturalWidth: img.naturalWidth,
          naturalHeight: img.naturalHeight, imageRect: rect(img), clippedPrimaryRect: rect(el.querySelector('.product-art')),
          decodeStatus: decoded?.status ?? 'not-decoded-unloaded', decodedAt: decoded?.at ?? null,
          decodeError: decoded?.error ?? null } : null };
    });
    return { observedAt: performance.now(), scrollY, viewport: { width: innerWidth, height: innerHeight },
      collectionRect: rect(document.querySelector('#collection')), heading: document.querySelector('#collection h2')?.textContent?.trim() ?? null,
      modelStatusText: document.querySelector('#collection [role=status]')?.textContent?.trim() ?? null,
      domCardCount: document.querySelectorAll('#collection article.product').length, cards,
      actualIntersectingFirstSixCards: cards.filter(c => c.rect?.intersectsViewport).length,
      actualFullyVisibleFirstSixCards: cards.filter(c => c.rect?.fullyWithinViewport).length,
      actualIntersectingPrimaryImages: cards.filter(c => c.image?.clippedPrimaryRect?.intersectsViewport).length };
  };
  const poll = () => {
    if (state.stopped) return;
    let mirrors = [];
    try { mirrors = JSON.parse(document.documentElement.getAttribute('data-local-model-readiness') || '[]'); } catch {}
    const snap = snapshot(); state.last = snap;
    for (const stage of ['rank', 'firstThree', 'complete']) {
      const mirror = mirrors.filter(e => e?.detail?.stage === stage).sort((a, b) => b.startTime - a.startTime)[0];
      if (!mirror) continue;
      if (!state.stages[stage]) state.stages[stage] = { mirror, snapshot: snap,
        navigationToReactAfterTwoRafMs: mirror.startTime + mirror.duration,
        clickToEffectMs: null, clickTimingReason: 'Direct hash-query navigation; no search click performed' };
      if (stage === 'rank' || state.imageEndpoints[stage]) continue;
      const n = stage === 'firstThree' ? 3 : 6;
      const cards = snap.cards.slice(0, n);
      if (cards.length === n && cards.every(c => c.image?.complete && c.image.naturalWidth > 0 && c.image.decodeStatus === 'decoded')) {
        // Endpoint includes copy mirror, image complete+successful decode and this observation.
        state.imageEndpoints[stage] = { observedAt: snap.observedAt, fromEffectMs: snap.observedAt - mirror.startTime,
          navigationToImageEndpointMs: snap.observedAt,
          requestId: mirror.detail.requestId, requiredImages: n, snapshot: snap };
      }
    }
    requestAnimationFrame(poll);
  };
  requestAnimationFrame(poll);
}

export function matchDomToCopy(snapshot, event) {
  if (!snapshot || !event) return { passed: false, reason: 'missing DOM snapshot or actual response event' };
  const expected = event.sections?.flatMap(s => s.cards) ?? [];
  const checks = expected.map((card, i) => {
    const dom = snapshot.cards[i]; const paragraphs = dom?.statusParagraphs?.map(p => p.text) ?? [];
    return { productId: card.productId, variantId: card.variantId, originalName: card.originalName,
      title: dom?.title === card.title, originalNameMatches: dom?.originalName === card.originalName,
      selectedColor: dom?.selectedColor === card.selectedColor,
      badge: card.badge !== null && paragraphs.includes(card.badge),
      body: card.bodyCopy !== null && paragraphs.includes(card.bodyCopy), expectedBody: card.bodyCopy,
      expectedTitle: card.title, domTitle: dom?.title ?? null, domParagraphs: paragraphs };
  });
  return { passed: checks.length === (event.type === 'firstThree' ? 3 : 6)
    && checks.every(c => c.title && c.originalNameMatches && c.selectedColor && c.badge && c.body), checks };
}
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
async function save(path, value) { await writeFile(path, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' }); }

export async function run(output, origin = 'http://127.0.0.1:4312') {
  if (origin !== 'http://127.0.0.1:4312') throw new Error('Only declared local storefront4312 is supported');
  await mkdir(output); // Must be new; never overwrite an earlier attempt.
  const sourceFiles = ['experiments/ranker/pretrained/store_candidate_browser_probe_v1.mjs',
    'apps/storefront/src/local-model-performance.ts', 'apps/storefront/src/ShopView.tsx', 'apps/storefront/src/ProductCard.tsx',
    'apps/storefront/src/ProductPhoto.tsx', 'apps/storefront/src/use-local-model.ts'];
  const sourceHashes = Object.fromEntries(await Promise.all(sourceFiles.map(async p => [p, sha(await readFile(join(ROOT, p)))])));
  const report = { version: 'store-candidate-browser-probe.v1', startedAt: new Date().toISOString(), sourceHashes,
    schedule: fixedSchedule(), timeoutMs: TIMEOUT_MS, viewport: { width: 1280, height: 960 }, headed: true,
    workload: 'First goto directly to ?model=local#collection?q=… then explicit page.reload for same-context repeat; no scroll or search click. Actual native POST and new matched request ID required per scored navigation. All six DOM cards versus first-three intended image readiness are separate. This workload does not establish useful above-fold copy: viewport geometry is reported.',
    cacheScope: 'Each pair begins with new isolated browser context, then repeats same URL in same context via explicit reload. Browser HTTP cache is eligible but actual resource timings govern cache interpretation; OS/CDN/native model caches uncontrolled. No forced eager image loading.',
    timingScope: 'Existing app mirror: effect before fetch through React commit and two rAF. Probe image endpoint: same effect clock through mirror plus complete+decode and observation. Navigation separately measured. No click measurement, physical paint claim, or trace claim.',
    attempts: [] };
  await save(join(output, 'plan.json'), report);
  let browser;
  const seenRequestIds = new Set(), seenDocumentIds = new Set();
  try {
    const require = createRequire(join(ROOT, 'apps/runner/package.json'));
    const { chromium } = require('playwright');
    browser = await chromium.launch({ channel: 'chrome', headless: false });
    report.browserVersion = browser.version();
    for (let pair = 0; pair < QUERIES.length; pair++) {
      const context = await browser.newContext({ viewport: report.viewport, serviceWorkers: 'block', acceptDownloads: false });
      try {
        await context.addInitScript(installObserver);
        const page = await context.newPage();
        for (const item of report.schedule.filter(s => s.pair === pair)) {
          const result = { ...item, startedAt: new Date().toISOString(), requestFailures: [], consoleErrors: [], modelResponses: [], actualPostRequests: [],
            navigationAction: item.condition === 'same-context-repeat' ? 'reload' : 'goto' };
          const pending = []; let captureOpen = true, settledBodies = 0;
          const fail = req => result.requestFailures.push({ url: req.url(), resourceType: req.resourceType(), error: req.failure() });
          const err = error => result.consoleErrors.push(String(error));
          const request = req => {
            if (req.url() !== 'http://127.0.0.1:8766/personalize' || req.method() !== 'POST') return;
            let query = null; try { query = JSON.parse(req.postData() ?? '{}').query ?? null; } catch {}
            result.actualPostRequests.push({ url: req.url(), method: req.method(), postData: req.postData(), query, observedAtNodeMs: performance.now() });
          };
          const response = res => { if (res.url() !== 'http://127.0.0.1:8766/personalize' || res.request().method() !== 'POST') return;
            pending.push((async () => { try { const body = await res.body();
              const receipt = { url: res.url(), status: res.status(), method: res.request().method(),
                request: res.request().postData(), bodySha256: sha(body), bodyBytes: body.length,
                rawBody: body.toString('utf8'), events: [], parseErrors: [] };
              for (const [index, line] of receipt.rawBody.split('\n').entries()) {
                if (!line.trim()) continue;
                try { receipt.events.push(JSON.parse(line)); } catch (e) { receipt.parseErrors.push({ index, error: String(e) }); }
              }
              if (captureOpen) result.modelResponses.push(receipt);
            } catch (e) { if (captureOpen) result.modelResponses.push({ url: res.url(), status: res.status(), error: String(e) }); }
            finally { settledBodies++; } })()); };
          page.on('requestfailed', fail); page.on('pageerror', err); page.on('response', response); page.on('request', request);
          const start = performance.now();
          try {
            await navigateAttempt(page, item, `${origin}/?model=local#collection?q=${encodeURIComponent(item.query)}`, { waitUntil: 'domcontentloaded', timeout: TIMEOUT_MS });
            result.navigationDomContentLoadedMs = performance.now() - start;
            const remaining = Math.max(1, TIMEOUT_MS - (performance.now() - start));
            await page.waitForFunction(() => Boolean(window.__candidateProbe?.imageEndpoints?.complete), null, { timeout: remaining });
          } catch (e) { result.error = String(e); }
          result.navigationToProbeEndMs = performance.now() - start;
          try {
            result.browser = await page.evaluate(() => { window.__candidateProbe.stopped = true;
              return { ...window.__candidateProbe, navigation: performance.getEntriesByType('navigation').map(e => e.toJSON()),
                resources: performance.getEntriesByType('resource').map(e => e.toJSON()),
                readinessRaw: document.documentElement.getAttribute('data-local-model-readiness') }; });
            const screenshotPath = join(output, `attempt-${item.index + 1}.png`);
            const bytes = await page.screenshot({ path: screenshotPath, fullPage: false, timeout: 5000 });
            result.screenshot = { path: screenshotPath, sha256: sha(bytes), bytes: bytes.length, status: 'captured' };
          } catch (e) { result.captureError = String(e); }
          // Responses already complete by complete DOM receipt; timeout paths may be aborted below.
          if (result.error) await page.evaluate(() => window.stop()).catch(() => {});
          await Promise.race([Promise.allSettled(pending), new Promise(r => setTimeout(r, 1000))]);
          captureOpen = false;
          result.responseBodyCapture = { started: pending.length, settled: settledBodies, pendingAtCaptureEnd: pending.length - settledBodies };
          const events = result.modelResponses.flatMap(r => r.events ?? []);
          result.domCopy = Object.fromEntries(['firstThree', 'complete'].map(stage => {
            const entry = result.browser?.stages?.[stage];
            const event = events.find(e => e.type === stage && e.requestId === entry?.mirror?.detail?.requestId);
            return [stage, { fieldReceiptValid: validFieldReceipt(entry?.mirror, stage),
              copySourceFromMirror: entry?.mirror?.detail?.copySource ?? null, copySourceFromActualResponse: event?.copySource ?? null,
              sourceExactlyMatches: Boolean(event && JSON.stringify(event.copySource) === JSON.stringify(entry?.mirror?.detail?.copySource)),
              parsedSourceFieldsMatch: parsedSourceFieldsMatch(entry?.mirror?.detail?.copySource, event?.copySource),
              sourceComparisonScope: 'Summary compares fields retained by frontend parser; sourceExactlyMatches preserves raw JSON equality separately. Extra raw checkpoint/feature provenance remains in response receipt.',
              matchAtStageObservation: matchDomToCopy(entry?.snapshot, event),
              matchAtFinalObservation: matchDomToCopy(result.browser?.last, event) }];
          }));
          result.protocol = verifyRequestProtocol(result, seenRequestIds, seenDocumentIds);
          if (result.protocol.requestId) seenRequestIds.add(result.protocol.requestId);
          if (result.protocol.documentId) seenDocumentIds.add(result.protocol.documentId);
          page.off('requestfailed', fail); page.off('pageerror', err); page.off('response', response); page.off('request', request);
          report.attempts.push(result);
          await save(join(output, `attempt-${item.index + 1}.json`), result);
        }
      } finally { await context.close(); }
    }
  } catch (e) { report.error = String(e); }
  finally { if (browser) await browser.close(); report.finishedAt = new Date().toISOString();
    report.summary = summarize(report.attempts); await save(join(output, 'report.json'), report); }
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error('Usage: node store_candidate_browser_probe_v1.mjs NEW_OUTPUT_DIRECTORY');
  const report = await run(resolve(process.argv[2]));
  console.log(JSON.stringify(report.summary, null, 2));
  if (report.error || report.attempts.some(a => a.error || a.captureError || a.protocol?.passed !== true)) process.exitCode = 1;
}
