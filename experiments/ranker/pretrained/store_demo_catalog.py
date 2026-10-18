"""Versioned demo renderer binding; catalog facts stay byte/execution verified.

The old import/experiment freshness rules remain unchanged. An opt-in renderer
can add copy props without changing the public catalog; this new receipt allows
only its reviewed exact ProductCard bytes, plus an identical runtime export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .store_catalog_v1 import (ROOT, SOURCES, RAW_EXPORT_SHA256, _EXPORT_JS,
    StoreCatalogError, digest, load_snapshot, validate_export)

DEFAULT_MANIFEST=Path(__file__).resolve().parent/'store_demo_integration_v3/manifest.json'
RENDERER='apps/storefront/src/ProductCard.tsx'
FRONTEND=('apps/storefront/src/main.tsx','apps/storefront/src/ShopView.tsx',RENDERER,
    'apps/storefront/src/local-model.ts','apps/storefront/src/use-local-model.ts',
    'apps/storefront/src/local-model-performance.ts')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_export():
    return validate_export(json.loads(subprocess.check_output(
        ['node','--import','tsx','--input-type=module','-e',_EXPORT_JS],cwd=ROOT,text=True)))


def freeze(path=DEFAULT_MANIFEST):
    path=Path(path)
    if path.exists():raise FileExistsError('Preserve prior integration receipt')
    snapshot=load_snapshot();current=runtime_export()
    if digest(current)!=RAW_EXPORT_SHA256 or current!=snapshot['rawExport']:
        raise StoreCatalogError('Current runtime catalog differs; do not reuse old learned catalog')
    expected={r['path']:r['sha256'] for r in snapshot['receipt']['sources']}
    changed={name:sha(ROOT/name) for name in SOURCES if sha(ROOT/name)!=expected[name]}
    if set(changed)-{RENDERER}:raise StoreCatalogError('Unreviewed input changes: '+','.join(sorted(changed)))
    value={'version':'actual-store-opt-in-demo-integration.v1','rawExportSha256':digest(current),
        'modelInputSha256':snapshot['receipt']['outputs']['model_inputs.json']['canonicalSha256'],
        'rendererOnlyExceptions':changed,'frontendSha256':{name:sha(ROOT/name) for name in FRONTEND},
        'reason':'Reviewed optional generated-copy props; runtime public products/prices/colors/images/shipping exactly equal pinned export.',
        'learnedImageSelection':False,'sourceCommit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
    return value


class DemoCatalogBinding:
    def __init__(self, manifest_path=DEFAULT_MANIFEST):
        self.path=Path(manifest_path);self.manifest=json.loads(self.path.read_text());self.manifest_sha=sha(self.path)
        if self.manifest['version']!='actual-store-opt-in-demo-integration.v1' or set(self.manifest['rendererOnlyExceptions'])-{RENDERER}:
            raise StoreCatalogError('Exact reviewed demo integration receipt required')
        if digest(runtime_export())!=self.manifest['rawExportSha256'] or self.manifest['rawExportSha256']!=RAW_EXPORT_SHA256:
            raise StoreCatalogError('Runtime catalog no longer matches trained catalog')
        self.assert_current_sources(load_snapshot()['receipt'])

    def assert_current_sources(self, receipt):
        if sha(self.path)!=self.manifest_sha:raise StoreCatalogError('Demo integration receipt changed after startup')
        if [r['path'] for r in receipt['sources']]!=list(SOURCES):raise StoreCatalogError('Full source inventory required')
        for row in receipt['sources']:
            expected=self.manifest['rendererOnlyExceptions'].get(row['path'],row['sha256'])
            path=ROOT/row['path']
            if path.is_symlink() or not path.is_file() or sha(path)!=expected:
                raise StoreCatalogError('Active demo source changed: '+row['path'])
        for name,expected in self.manifest['frontendSha256'].items():
            if sha(ROOT/name)!=expected:raise StoreCatalogError('Active demo client changed: '+name)
        return True

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,default=DEFAULT_MANIFEST);a=p.parse_args()
    z=freeze(a.out);print(json.dumps({'status':'runtime-export-identical','rendererOnlyExceptions':z['rendererOnlyExceptions']}))
