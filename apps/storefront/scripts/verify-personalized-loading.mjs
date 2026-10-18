// Synthetic browser UI protocol test. Every model response is intercepted; no native calls.
import { chromium } from '@playwright/test';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import assert from 'node:assert/strict';
import { PRODUCTS } from '../src/config.ts';
import { PRODUCT_VARIANTS } from '../src/variants.ts';
const out = resolve(process.argv[2] || 'data/storefront-followups-loading-v2');
const url = process.argv[3] || 'http://shop.dubhacks.localhost:1356/';
await mkdir(out, { recursive: true });
const endpoint = 'http://127.0.0.1:8767/personalize';
const checks = [], errors = [];
const six = PRODUCTS.slice(0, 6);
const rank = { type: 'rank', requestId: 'synthetic-ui-only', productIds: PRODUCTS.map(p => p.id), selectedProductIds: six.map(p => p.id),
  variants: Object.fromEntries(six.map(p => [p.id, PRODUCT_VARIANTS[p.id][0].id])), shortfall: 0,
  timing: { rankMs: 10 }, provenance: { ranking: 'Synthetic UI fixture, not measured inference', images: 'catalog' } };
const copy = { type: 'complete', requestId: rank.requestId, sections: [0, 1].map(i => ({ heading: 'UI fixture collection',
  cards: six.slice(i * 3, i * 3 + 3).map(p => ({ productId: p.id, variantId: PRODUCT_VARIANTS[p.id][0].id, originalName: p.name,
    itemPrice: p.price, selectedColor: PRODUCT_VARIANTS[p.id][0].colorName, title: `UI fixture ${p.name}`, badge: null, bodyCopy: null })) })),
  acceptedFields: 8, withheldFields: 12, requiredFields: 20, timing: { readyMs: 20, copyMs: 10 } };
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const context = await browser.newContext({ viewport: { width: 1280, height: 960 } });
const page = await context.newPage();
page.on('pageerror', error => errors.push(String(error)));
await page.route('**/search-suggestions.json', route => route.fulfill({ contentType: 'application/json', body: JSON.stringify({ version: 'fieldwork.validated-search-suggestions.v1', queries: ['Fixture search one', 'Fixture search two'] }) }));
await page.route(endpoint, route => route.abort()); // Safety net: never reaches native service.
await page.addInitScript(endpoint => {
  const originalFetch = window.fetch.bind(window);
  window.__loadingStreams = [];
  window.fetch = (input, init) => {
    if (String(input) !== endpoint) return originalFetch(input, init);
    let controller;
    const stream = new ReadableStream({ start(value) { controller = value; } });
    window.__loadingStreams.push({ body: JSON.parse(init.body), controller });
    return Promise.resolve(new Response(stream, { headers: { 'content-type': 'application/x-ndjson' } }));
  };
}, endpoint);
const waitRequest = async count => page.waitForFunction(count => window.__loadingStreams.length >= count, count);
async function checkEmpty(label, skeletons = 6) {
  await page.waitForFunction(n => document.querySelectorAll('.product-skeleton').length === n, skeletons);
  assert.equal(await page.locator('article.product').count(), 0);
  assert.equal(await page.getByRole('button', { name: 'Add to cart', exact: true }).count(), 0);
  assert.equal(await page.locator('.hero img').count(), 0);
  assert.equal(await page.locator('#shop-search').isEnabled(), true);
  checks.push(label);
}
async function emit(index, event, close = false) {
  await page.evaluate(({index,event,close}) => {
    const stream = window.__loadingStreams[index];
    stream.controller.enqueue(new TextEncoder().encode(JSON.stringify(event)+'\n'));
    if (close) stream.controller.close();
  }, {index,event,close});
}
async function finish(index, status = 200) {
  if (status !== 200) return page.evaluate(index => window.__loadingStreams[index].controller.error(new Error('Synthetic unavailable response')), index);
  await emit(index, rank); await emit(index, copy, true);
}
const report = { version: 'storefront-followups-loading-browser.v2', synthetic: true, modelCalls: 0,
  scope: 'Intercepted synthetic UI protocol only; no quality or inference latency claim', catalogProducts: PRODUCTS.length, url, checks, errors };
try {
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await waitRequest(1); await checkEmpty('initial pending request has six placeholders and no generic cards/hero');
  await page.screenshot({ path: `${out}/initial-loading.png`, fullPage: true });
  await emit(0, rank);
  await page.waitForFunction(() => document.querySelector('.search-status[role="status"]')?.textContent.includes('Finding'));
  await checkEmpty('rank-only stream update still shows six placeholders and zero purchasable cards');
  const firstThree = { ...copy, type: 'firstThree', sections: copy.sections.slice(0,1), acceptedFields: 4, withheldFields: 6, requiredFields: 10 };
  await emit(0, firstThree);
  await page.waitForFunction(count => document.querySelectorAll('article.product').length === count - 3 && document.querySelectorAll('.product-skeleton').length === 3, PRODUCTS.length);
  checks.push('firstThree stream update reveals current generated cards and keeps three selected pending slots empty');
  await page.screenshot({ path: `${out}/first-three.png`, fullPage: false });
  await emit(0, copy, true); await page.waitForFunction(count => document.querySelectorAll('article.product').length === count, PRODUCTS.length);
  assert.ok(await page.locator('.hero img').count() > 0);
  assert.ok(await page.locator('.category-photo-row button').count() > 0);
  assert.equal(await page.locator('.header-search #shop-search').count(), 1);
  assert.equal(await page.locator('#shop-search').count(), 1);
  const heroSource = await page.locator('.hero img').getAttribute('src');
  await page.waitForTimeout(6200);
  assert.equal(await page.locator('.hero img').getAttribute('src'), heroSource);
  assert.equal(await page.locator('.hero img').count(), 1);
  checks.push('hero has one stable image, with no timer change after six seconds');
  assert.equal(await page.evaluate(() => document.documentElement.className.includes('darkreader')), false);
  checks.push('teammate hero/category photography and top-right header search render without DarkReader');
  await page.screenshot({ path: `${out}/teammate-design.png`, fullPage: true });
  await page.screenshot({ path: `${out}/teammate-design-viewport.png`, fullPage: false });
  await page.getByRole('button', { name: /Shopping as.*Switch person/ }).click();
  await page.locator('.person-options button[aria-pressed="false"]').first().click();
  await waitRequest(2); await page.keyboard.press('Escape'); await checkEmpty('persona switch removes prior cards and hero immediately');
  await page.screenshot({ path: `${out}/persona-loading.png`, fullPage: true });
  await finish(1); await page.waitForFunction(count => document.querySelectorAll('article.product').length === count, PRODUCTS.length);
  await page.locator('#shop-search').focus();
  await page.getByText('Suggested searches', { exact: true }).waitFor();
  await page.getByRole('option', { name: 'Fixture search one', exact: true }).click();
  await waitRequest(3); await checkEmpty('suggested search submits the exact synthetic metadata string without fake history');
  assert.equal(await page.locator('.hero').isVisible(), false);
  await finish(2); await page.waitForFunction(count => document.querySelectorAll('article.product').length === count, PRODUCTS.length);
  report.inputAfterSuggestion = await page.locator('#shop-search').inputValue();
  await page.locator('#shop-search').fill('desk organizer');
  report.inputAfterTyping = await page.locator('#shop-search').inputValue();
  await page.locator('.header-search button[type="submit"]').click();
  await waitRequest(4); assert.equal(await page.locator('.hero').isVisible(), false); checks.push('active search hides the entire hero'); await checkEmpty('new search removes previous cards and keeps controls usable');
  await finish(3, 503); await checkEmpty('failed request shows no fallback/old cards or infinite skeleton', 0);
  await page.getByRole('status').filter({hasText:'couldn’t refresh'}).waitFor();
  await page.screenshot({ path: `${out}/request-error.png`, fullPage: true });
  await page.getByRole('button', { name: 'Clear search', exact: true }).click();
  await waitRequest(5); await finish(4);
  await page.waitForFunction(count => document.querySelectorAll('article.product').length === count, PRODUCTS.length);
  assert.equal(await page.locator('.hero').isVisible(), true);
  assert.equal(await page.locator('.hero img').count(), 1);
  checks.push('clearing search restores the stable persona hero');
  assert.deepEqual(errors, []);
  report.passed = true;
} catch (error) { report.error = String(error); report.passed = false; process.exitCode = 1; }
finally {
  report.interceptedModelRequests = await page.evaluate(() => window.__loadingStreams?.length ?? 0);
  report.requestQueries = await page.evaluate(() => window.__loadingStreams?.map(item => item.body.query) ?? []);
  report.sourceHashes = {};
  for (const file of ['main.tsx','ShopView.tsx','styles.css','HeroCarousel.tsx','CategoryShortcuts.tsx','ProductSkeleton.tsx','SearchField.tsx','use-search-suggestions.ts','validated-search-suggestions.ts']) {
    report.sourceHashes[file] = createHash('sha256').update(await readFile(new URL(`../src/${file}`, import.meta.url))).digest('hex');
  }
  await context.close(); await browser.close(); report.browserClosed = true;
  await writeFile(`${out}/report.json`, JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify(report));
}
