import { PRODUCTS } from '../../storefront/src/config.ts';
import { PRODUCT_VARIANTS, getVariantImages } from '../../storefront/src/variants.ts';
import { readFileSync } from 'node:fs';

const shared = JSON.parse(readFileSync(new URL('../../../packages/contracts/data/expanded-store-catalog-v1.json', import.meta.url), 'utf8'));
const source = new Map(shared.products.map((p: any) => [p.productId, p]));
const products = PRODUCTS.map(product => {
  const expected: any = source.get(product.id);
  if (!expected) throw new Error(`Active UI product absent from public catalog: ${product.id}`);
  return {
    productId: product.id, name: product.name, category: product.category,
    itemPrice: product.price, kind: product.kind,
    searchTerms: 'searchTerms' in product ? [...product.searchTerms] : expected.searchTerms,
    intentTags: expected.intentTags, parentFamily: expected.parentFamily,
    variants: PRODUCT_VARIANTS[product.id].map(variant => ({
      variantId: variant.id, colorName: variant.colorName, colorHex: variant.swatch,
      images: getVariantImages(variant),
    })),
  };
});
process.stdout.write(JSON.stringify({version:'expanded-store-public-catalog.v1', products}) + '\n');
