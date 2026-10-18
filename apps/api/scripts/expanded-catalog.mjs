import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = new URL('../../../', import.meta.url);
export const catalogPath = 'packages/contracts/data/expanded-store-catalog-v1.json';
export function deriveCatalogs(raw) {
  const data = JSON.parse(raw);
  if (data.version !== 'expanded-store-public-catalog.v1' || !Array.isArray(data.products)) throw Error('Unsupported public catalog');
  const products = {}, telemetry = {}, skus = new Set();
  for (const p of data.products) {
    if (!/^[a-z][a-z0-9-]*$/.test(p.productId) || products[p.productId] || !p.name || !Number.isFinite(p.itemPrice) || p.itemPrice < 0 || !Number.isSafeInteger(p.itemPrice * 100) || !Array.isArray(p.variants) || !p.variants.length) throw Error('Invalid product');
    const variants = p.variants.map(v => {
      if (!v.variantId.startsWith(p.productId + '-') || skus.has(v.variantId) || !v.colorName || !/^#[a-f0-9]{6}$/i.test(v.colorHex) || !Array.isArray(v.images) || !v.images.length) throw Error('Invalid unique product variant');
      skus.add(v.variantId);
      for (const image of v.images) if (!image.src?.startsWith('/products/') || image.src.includes('..') || !image.id || !image.alt) throw Error('Invalid public image');
      return v.variantId;
    });
    products[p.productId] = { unitCents: p.itemPrice * 100, variants };
    telemetry[p.productId] = variants;
  }
  return { checkout: { schema: 'fieldwork.demo-checkout-catalog.v1', products, freeShippingOverCents: 5000, flatShippingCents: 500, sourceSha256: { [catalogPath]: createHash('sha256').update(raw).digest('hex') } }, telemetry };
}
export function refreshCatalogs({ check = false } = {}) {
  const derived = deriveCatalogs(readFileSync(new URL(catalogPath, root), 'utf8'));
  for (const [kind, path] of [['checkout', 'apps/api/internal/democheckout/catalog.json'], ['telemetry', 'apps/api/internal/shoppingevents/catalog.json']]) {
    const text = JSON.stringify(derived[kind], null, 2) + '\n';
    const target = new URL(path, root);
    if (check) { if (readFileSync(target, 'utf8') !== text) throw Error(`Stale generated catalog: ${path}`); }
    else writeFileSync(target, text);
  }
  return derived;
}
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const result = refreshCatalogs({ check: process.argv.includes('--check') });
  console.log(JSON.stringify({ products: Object.keys(result.telemetry).length, variants: Object.values(result.telemetry).flat().length, checked: process.argv.includes('--check') }));
}
