/** Export compact README figures from the existing presentation sources.
 * Run after `pnpm --filter @storefront/web build`:
 * TSX_TSCONFIG_PATH=apps/web/tsconfig.json node --import tsx scripts/export-readme-figures.mjs [output-directory]
 */
import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import { createHash } from 'node:crypto';

// When reviewed outside scripts/, REPO_ROOT selects the same source checkout.
const root = process.env.REPO_ROOT || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const webRequire = createRequire(path.join(root, 'apps/web/package.json'));
const rootRequire = createRequire(path.join(root, 'package.json'));
const React = webRequire('react');
const { renderToStaticMarkup } = webRequire('react-dom/server');
const { chromium } = rootRequire('@playwright/test');
const { LatencyChart } = await import(pathToFileURL(path.join(root, 'apps/web/src/slides/LatencyChart.tsx')));
const { latencySeries, median, medianInterval } = await import(pathToFileURL(path.join(root, 'apps/web/src/lib/latency.ts')));
const out = path.resolve(process.argv[2] || path.join(root, 'docs/assets'));
await mkdir(out, { recursive: true });
const assetDir = path.join(root, 'apps/web/dist/assets');
const cssFile = (await readdir(assetDir)).find(name => name.endsWith('.css'));
if (!cssFile) throw new Error('Build apps/web before exporting figures.');
const css = await readFile(path.join(assetDir, cssFile), 'utf8');
const series = latencySeries().map(item => item.ours ? { ...item, label: 'Current local · warm backend' } : item);
const chart = renderToStaticMarkup(React.createElement(LatencyChart, { series }));
const sha = async rel => createHash('sha256').update(await readFile(path.join(root, rel))).digest('hex');
const dataUri = async rel => `data:image/jpeg;base64,${(await readFile(path.join(root, rel))).toString('base64')}`;
const sequence = `<div class="sequence">${(await Promise.all(['browse', 'stuck', 'fit'].map(async (stage, i) => `<div><img src="${await dataUri(`apps/web/public/fieldwork/${stage}.jpg`)}" alt=""><span>(${String.fromCharCode(97+i)})</span></div>`))).join('')}</div>`;
const labels = ['Frozen Qwen', 'Learned selection', 'Selected fact rows', 'Parallel copy'];
const modelSource = await readFile(path.join(root, 'apps/web/src/slides/ModelSlide.tsx'), 'utf8');
for (const label of labels) if (!modelSource.includes(label)) throw new Error(`Architecture source changed: ${label}`);
const architecture = `<svg xmlns="http://www.w3.org/2000/svg" width="1360" height="144" viewBox="0 0 1360 144" role="img" aria-label="Frozen Qwen feeds learned selection, selected fact rows, then parallel copy"><rect width="1360" height="144" fill="white"/><defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="none" stroke="#475569" stroke-width="1.2"/></marker></defs>${labels.map((label, i) => `<rect x="${24+i*336}" y="40" width="304" height="64" fill="#fff" stroke="#94a3b8" stroke-width="1.2"/><text x="${176+i*336}" y="79" text-anchor="middle" fill="#17241d" font-family="Arial, sans-serif" font-size="23">${label}</text>${i<3 ? `<path d="M${328+i*336} 72H${354+i*336}" fill="none" stroke="#475569" stroke-width="1.4" marker-end="url(#arrow)"/>` : ''}`).join('')}</svg>`;
await writeFile(path.join(out, 'architecture-figure.svg'), architecture+'\n');
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1360, height: 700 }, deviceScaleFactor: 2 });
  const style = `:root{--color-fog:#fff;--color-ink:#17241d;--color-muted:#475569;--color-line:#e2e8f0;--color-graphite:#64716c}html,body{margin:0;background:white}main{width:1360px;padding:24px;box-sizing:border-box}main>svg{width:100%;height:auto;display:block}figure{margin:0}.text-sm{font-size:17px;line-height:1.4}.text-lg{font-size:21px}.h-24{height:108px}.sequence{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}.sequence img{width:100%;height:auto;border:1px solid #d5dad3}.sequence span{display:block;font-size:18px;margin-top:8px;color:#475569}.sequence>div{text-align:center}`;
  const render = async (content, name) => {
    await page.setContent(`<!doctype html><html><head><meta charset="utf-8"><base href="${pathToFileURL(assetDir+'/').href}"><style>${css}</style><style>${style}</style></head><body><main>${content}</main></body></html>`);
    await page.evaluate(() => document.fonts.ready);
    await page.locator('img').evaluateAll(images => Promise.all(images.map(img => img.decode())));
    if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw new Error('Figure overflow');
    await page.locator('main').screenshot({ path: path.join(out, name), animations: 'disabled' });
  };
  await render(chart, 'latency-figure.png');
  if (await page.locator('figure button').count() !== series.reduce((n,item)=>n+item.ms.length,0)) throw new Error('Request dot count differs');
  await render(sequence, 'shopper-sequence.png');
  if (process.env.FIGURE_PREVIEW) await render(architecture, 'architecture-figure.png');
} finally { await browser.close(); }
const sources = ['apps/web/src/slides/LatencyChart.tsx','apps/web/src/slides/LatencyRow.tsx','apps/web/src/lib/latency.ts','apps/web/src/fieldwork/current-warm-speed.json','apps/web/src/fieldwork/llm-run.json','apps/web/src/slides/ModelSlide.tsx','apps/web/public/fieldwork/browse.jpg','apps/web/public/fieldwork/stuck.jpg','apps/web/public/fieldwork/fit.jpg'];
const receipt = { renderer: 'scripts/export-readme-figures.mjs', sourceHashes: Object.fromEntries(await Promise.all(sources.map(async rel => [rel,await sha(rel)]))), series: series.map(item => ({id:item.id,label:item.label,detail:item.detail,count:item.ms.length,medianMs:median(item.ms),medianInterval95Ms:medianInterval(item.ms),trivialCliReplyMs:item.floorMs ?? null})), files: await Promise.all(['latency-figure.png','shopper-sequence.png','architecture-figure.svg'].map(async asset=>({asset,sha256:createHash('sha256').update(await readFile(path.join(out,asset))).digest('hex')}))) };
await writeFile(path.join(out,'figure-export.json'),JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify(receipt.series,null,2));
console.log('Figure exports:',out);
