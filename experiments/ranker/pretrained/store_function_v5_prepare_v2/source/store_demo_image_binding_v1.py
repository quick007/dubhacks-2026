"""Explicit render-only WebP binding; original catalog rules remain unchanged."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from .store_demo_catalog import ROOT, FRONTEND, runtime_export
from .store_catalog_v1 import SOURCES, RAW_EXPORT_SHA256, digest, load_snapshot, StoreCatalogError

VERSION = 'actual-store-render-image-delivery.v1'
ALLOWED = {'apps/storefront/src/ProductCard.tsx', 'apps/storefront/src/ProductPhoto.tsx',
           'apps/storefront/src/ProductGallery.tsx'}
EXTRA = ('apps/storefront/src/ProductPhoto.tsx', 'apps/storefront/src/ProductGallery.tsx',
         'apps/storefront/src/product-asset-source.ts', 'apps/storefront/src/responsive-product-assets.json',
         'apps/storefront/src/product-responsive-source.ts', 'apps/storefront/src/responsive-product-sizes.json')
FULL_MANIFEST = 'apps/storefront/src/responsive-product-assets.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_assets():
    result = json.loads(subprocess.check_output(['node', 'scripts/prepare-product-webp.mjs', '--check'], cwd=ROOT, text=True))
    if result.get('status') != 'verified':
        raise ValueError('Verified original and derivative asset hashes required')
    responsive = json.loads(subprocess.check_output(['node', 'scripts/prepare-product-responsive-webp.mjs', '--check'], cwd=ROOT, text=True))
    if responsive.get('status') != 'verified':
        raise ValueError('Verified responsive derivative hashes required')
    return json.loads((ROOT / FULL_MANIFEST).read_text())


def freeze_image_delivery(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve prior integration receipts')
    assets = checked_assets(); snapshot = load_snapshot(); current = runtime_export()
    if current != snapshot['rawExport'] or digest(current) != RAW_EXPORT_SHA256:
        raise StoreCatalogError('Changed catalog cannot reuse this learned checkpoint')
    expected = {r['path']: r['sha256'] for r in snapshot['receipt']['sources']}
    changed = {name: sha(ROOT / name) for name in SOURCES if sha(ROOT / name) != expected[name]}
    if set(changed) - ALLOWED:
        raise StoreCatalogError('Unreviewed input changes: ' + ','.join(sorted(changed)))
    value = {'version': VERSION, 'rawExportSha256': digest(current),
        'modelInputSha256': snapshot['receipt']['outputs']['model_inputs.json']['canonicalSha256'],
        'rendererOnlyExceptions': changed,
        'frontendSha256': {name: sha(ROOT / name) for name in (*FRONTEND, *EXTRA)},
        'sourceCommit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'reason': 'Same public catalog and original assets; reviewed same-dimension WebP delivery with original fallback.',
        'learnedImageSelection': False,
        'imageDelivery': {'version': assets['version'], 'encoder': assets['encoder'], 'summary': assets['summary'],
            'manifestFile': FULL_MANIFEST, 'manifestSha256': sha(ROOT / FULL_MANIFEST),
            'responsiveManifestFile': EXTRA[-1], 'responsiveManifestSha256': sha(ROOT / EXTRA[-1]),
            'responsiveSummary': json.loads((ROOT / EXTRA[-1]).read_text())['summary'],
            'delivery': 'responsive whole-sheet WebP srcset/sizes; small lazy thumbnails; original-format fallback; full-size alternative retained',
            'originalAssetBytesUnchanged': True, 'catalogFactsUnchanged': True, 'modelInputsUnchanged': True,
            'learnedImageFeatures': False, 'selection': 'existing color/gallery metadata; alternate format only',
            'derivativeHashesVerifiedAtFreeze': True}}
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2) + '\n')
    return value


class ImageDeliveryCatalogBinding:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path); self.manifest = json.loads(self.path.read_text()); self.manifest_sha = sha(self.path)
        if self.manifest.get('version') != VERSION or set(self.manifest['rendererOnlyExceptions']) - ALLOWED:
            raise StoreCatalogError('Explicit reviewed image-delivery receipt required')
        if not set((*FRONTEND, *EXTRA)) <= set(self.manifest['frontendSha256']):
            raise StoreCatalogError('Full image renderer/source manifest pins required')
        if digest(runtime_export()) != RAW_EXPORT_SHA256 or self.manifest['rawExportSha256'] != RAW_EXPORT_SHA256:
            raise StoreCatalogError('Actual public catalog differs from trained catalog')
        snapshot = load_snapshot()
        if self.manifest['modelInputSha256'] != snapshot['receipt']['outputs']['model_inputs.json']['canonicalSha256']:
            raise StoreCatalogError('Trained model facts differ')
        self.assert_current_sources(snapshot['receipt']); assets = checked_assets()
        if assets['summary'] != self.manifest['imageDelivery']['summary']:
            raise StoreCatalogError('Derivative image inventory changed')

    def assert_current_sources(self, receipt):
        if sha(self.path) != self.manifest_sha:
            raise StoreCatalogError('Image-delivery receipt changed after startup')
        if [r['path'] for r in receipt['sources']] != list(SOURCES):
            raise StoreCatalogError('Full original source inventory required')
        expected = {r['path']: self.manifest['rendererOnlyExceptions'].get(r['path'], r['sha256']) for r in receipt['sources']}
        expected.update(self.manifest['frontendSha256'])
        for name, checksum in expected.items():
            file = ROOT / name
            if file.is_symlink() or not file.is_file() or sha(file) != checksum:
                raise StoreCatalogError('Active image-delivery source changed: ' + name)
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); result = freeze_image_delivery(args.out)
    print(json.dumps({'status': 'same-catalog-render-image-delivery-frozen', 'imageDelivery': result['imageDelivery']}))
