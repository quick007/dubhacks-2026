import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { deriveCatalogs, refreshCatalogs } from './expanded-catalog.mjs';
const root = new URL('../../../', import.meta.url);
const read = path => JSON.parse(readFileSync(new URL(path, root), 'utf8'));
const catalog = read('packages/contracts/data/expanded-store-catalog-v1.json');
const frozen = read('experiments/ranker/pretrained/store_catalog_v1/runtime_export.json');
const { NEW_PRODUCTS } = await import(new URL('apps/storefront/src/catalog-new-products.ts', root));
test('shared expanded source has 35 products / 105 unique variants and exact generated API catalogs', () => {
  assert.equal(catalog.products.length, 35);
  const derived = refreshCatalogs({ check: true });
  assert.equal(Object.values(derived.telemetry).flat().length, 105);
  assert.equal(new Set(Object.values(derived.telemetry).flat()).size, 105);
});
test('all original 16 names, IDs, prices, SKU colors and image galleries remain exact', () => {
  for (const old of frozen.products) {
    const current = catalog.products.find(p => p.productId === old.id);
    assert.equal(current.name, old.name); assert.equal(current.itemPrice, old.price);
    assert.equal(current.parentFamily, old.id);
    const variants = frozen.variants.find(p => p.productId === old.id).colorVariants;
    assert.deepEqual(current.variants, variants.map(v => ({ variantId: v.variant.id, colorName: v.variant.colorName, colorHex: v.variant.swatch, images: v.gallery })));
  }
});
test('only genuine rainwear and dedicated work bags have strict category intent', () => {
  assert.deepEqual(catalog.products.filter(p => p.intentTags.includes('rain-jacket')).map(p => p.productId), ['jacket', 'rain-anorak', 'commuter-raincoat']);
  assert.deepEqual(catalog.products.filter(p => p.intentTags.includes('work-bag')).map(p => p.productId), ['work-briefcase', 'work-tote', 'commuter-pack']);
});
test('invalid duplicate SKU or price is rejected before regeneration', () => {
  const bad = structuredClone(catalog); bad.products[1].variants[0].variantId = bad.products[0].variants[0].variantId;
  assert.throws(() => deriveCatalogs(JSON.stringify(bad)));
  const price = structuredClone(catalog); price.products[1].itemPrice = 0.001; assert.throws(() => deriveCatalogs(JSON.stringify(price)));
  const image = structuredClone(catalog); image.products[0].variants[0].images[0].src = 'https://private.invalid/image'; assert.throws(() => deriveCatalogs(JSON.stringify(image)));
});

test('all fourteen teammate additions preserve their actual names, prices and color SKU facts', () => {
  assert.equal(NEW_PRODUCTS.length, 14);
  for (const product of NEW_PRODUCTS) {
    const actual = catalog.products.find(p => p.productId === product.id);
    assert.equal(actual.name, product.name); assert.equal(actual.category, product.category); assert.equal(actual.itemPrice, product.price);
    assert.deepEqual(actual.variants.map(v => [v.colorName, v.colorHex]), product.colorOptions);
    for (const [i, variant] of actual.variants.entries()) for (const [j, image] of variant.images.entries()) {
      assert.equal(image.src, `/products/${product.id}-color-gallery.jpg`); assert.equal(image.tile.index, i * 3 + j);
      assert.ok(readFileSync(new URL(`apps/storefront/public${image.src}`, root)).length > 0);
    }
  }
});
test('five authored fixtures bind real square PNG assets, prices and all nine gallery tiles', () => {
  for (const [id, price] of [['rain-anorak',88],['commuter-raincoat',148],['work-briefcase',96],['work-tote',74],['commuter-pack',112]]) {
    const product = catalog.products.find(p => p.productId === id);
    assert.equal(product.itemPrice, price);
    const image = readFileSync(new URL(`apps/storefront/public/products/${id}-color-gallery.png`, root));
    assert.equal(image.subarray(0,8).toString('hex'), '89504e470d0a1a0a');
    assert.equal(image.readUInt32BE(16),1254); assert.equal(image.readUInt32BE(20),1254);
    assert.deepEqual(product.variants.flatMap(v => v.images.map(i => i.tile.index)), [0,1,2,3,4,5,6,7,8]);
    assert.ok(product.variants.every(v => v.images.every(i => i.src === `/products/${id}-color-gallery.png`)));
  }
});
