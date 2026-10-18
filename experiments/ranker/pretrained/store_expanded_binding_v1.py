"""Explicit expanded public catalog binding; frozen original16 rules untouched."""
from copy import deepcopy
import json
import subprocess
from pathlib import Path
from .store_persona_adapter_contract_v2 import sha, exact_sha
from . import store_qwen_lexical_choice_inputs_v2 as family_formatter

VERSION='expanded-store-active-binding.v1'
ROOT=Path(__file__).resolve().parents[3]


def normalized_products(value):
    if value.get('version')!='expanded-store-public-catalog.v1' or value.get('synthetic') is not True:
        raise ValueError('Explicit synthetic expanded public catalog required')
    result=[]
    for item in value['products']:
        p=deepcopy(item);p['title']=p['name']
        p['colorVariants']=[{'colorName':v['colorName'],'images':deepcopy(v['images'])} for v in p['variants']]
        result.append(p)
    return result


def check_original_projection(products):
    source=family_formatter.load_public_catalog();byid={p['productId']:p for p in products}
    if len(byid)!=len(products): raise ValueError('Unique expanded source identities required')
    for old in source:
        new=byid.get(old['id'])
        if new is None or (new['name'],new['category'],new['itemPrice'])!=(old['name'],old['category'],old['price']):
            raise ValueError('Original16 identity/category/price must remain exact')
        if [(v['variantId'],v['colorName']) for v in new['variants']] != [(v['id'],v['colorName']) for v in old['variants']]:
            raise ValueError('Original16 variant/color order must remain exact')
    # Original exact images come from the authenticated original action snapshot.
    from .store_catalog_v1 import load_snapshot
    snapshot=load_snapshot()
    for product in snapshot['rawExport']['variants']:
        actual=byid[product['productId']]
        expected=[(v['variant']['id'],v['gallery']) for v in product['colorVariants']]
        if [(v['variantId'],v['images']) for v in actual['variants']]!=expected:
            raise ValueError('All original16 source image metadata must remain exact')
    return True


class ExpandedBinding:
    def __init__(self,manifest_path):
        self.path=Path(manifest_path).resolve();self.manifest_sha=sha(self.path)
        self.manifest=json.loads(self.path.read_text())
        if self.manifest.get('version')!=VERSION: raise ValueError('Explicit expanded binding manifest required')
        ref=self.manifest['catalog'];self.catalog_path=Path(ref['file'])
        if not self.catalog_path.is_absolute(): self.catalog_path=ROOT/self.catalog_path
        if self.catalog_path.is_symlink() or sha(self.catalog_path)!=exact_sha(ref['sha256']): raise ValueError('Actual expanded catalog checksum changed')
        self.catalog_sha=ref['sha256'];self.catalog=json.loads(self.catalog_path.read_text())
        self.products=normalized_products(self.catalog);check_original_projection(self.products)
        if len(self.products)!=35: raise ValueError('Final reviewed35 products required; no pending asset placeholders')
        self.sources=self.manifest.get('sourceSha256',{})
        if not self.sources or 'packages/contracts/data/expanded-store-catalog-v1.json' not in self.sources:
            raise ValueError('Full actual frontend/catalog source SHA inventory required')
        self.original_assets=self.manifest.get('originalAssetsSha256',{})
        from .store_catalog_v1 import load_snapshot
        original_expected={'apps/storefront/public'+r['src']:r['sha256'] for r in load_snapshot()['receipt']['assets']}
        if self.original_assets!=original_expected:
            raise ValueError('Exact complete authenticated original source asset byte pins required')
        all_declared={'apps/storefront/public'+image['src'] for p in self.products for v in p['variants'] for image in v['images']}
        self.expanded_assets=self.manifest.get('expandedAssetsSha256',{})
        if set(self.expanded_assets)!=all_declared-set(self.original_assets):
            raise ValueError('Every new declared expanded gallery source asset must be pinned')
        if len(set(self.expanded_assets.values()))!=len(self.expanded_assets) or set(self.expanded_assets.values())&set(self.original_assets.values()):
            raise ValueError('New actual gallery assets must not duplicate existing image bytes')
        self.all_assets={**self.original_assets,**self.expanded_assets}
        self._asset_stats={}
        for relative,expected in self.all_assets.items():
            path=ROOT/relative
            if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink() or not path.is_file() or sha(path)!=exact_sha(expected): raise ValueError('Original source asset changed: '+relative)
            self._asset_stats[relative]=self._stat(path)
        self.assert_current_sources(None)
        self.assert_runtime_export()

    def assert_current_sources(self,receipt):
        # receipt authenticates the historical16 feature session, not expanded
        # active membership. This explicit manifest checks all current35 facts.
        if sha(self.path)!=self.manifest_sha or sha(self.catalog_path)!=self.catalog_sha:
            raise ValueError('Expanded binding/catalog changed')
        for relative,expected in self.sources.items():
            path=ROOT/relative
            if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink() or not path.is_file() or sha(path)!=exact_sha(expected):
                raise ValueError('Expanded source/asset changed: '+relative)
        for relative,before in self._asset_stats.items():
            path=ROOT/relative
            if path.is_symlink() or self._stat(path)!=before:raise ValueError('Original asset changed since startup: '+relative)
        return True

    @staticmethod
    def _stat(path):
        info=path.stat();return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)

    def assert_runtime_export(self):
        command=['node','--import','tsx','apps/api/scripts/expanded-store-runtime-export.mts']
        if 'apps/api/scripts/expanded-store-runtime-export.mts' not in self.sources:
            raise ValueError('Actual UI runtime exporter source pin required')
        value=json.loads(subprocess.check_output(command,cwd=ROOT,text=True,timeout=30))
        if value.get('version')!=self.catalog['version'] or value.get('products')!=self.catalog['products']:
            raise ValueError('Actual active UI products/variants/gallery facts differ from expanded public catalog')
        return True

    def assert_full_integrity(self):
        self.assert_current_sources(None)
        for relative,expected in self.all_assets.items():
            if sha(ROOT/relative)!=expected:raise ValueError('Original asset bytes changed: '+relative)
        return True


def freeze(path):
    """Root's CPU-only active35 freeze; never replaces an older manifest."""
    path=Path(path)
    if path.exists():raise FileExistsError('Preserve prior binding; choose a new output')
    catalog_path=ROOT/'packages/contracts/data/expanded-store-catalog-v1.json'
    catalog=json.loads(catalog_path.read_text());products=normalized_products(catalog);check_original_projection(products)
    if len(products)!=35 or sum(len(p['variants'])for p in products)!=105:raise ValueError('Complete real35/105 catalog required')
    actual=json.loads(subprocess.check_output(['node','--import','tsx','apps/api/scripts/expanded-store-runtime-export.mts'],cwd=ROOT,text=True,timeout=30))
    if actual.get('version')!=catalog['version'] or actual.get('products')!=catalog['products']:raise ValueError('Actual active35 UI differs from shared catalog')
    from .store_catalog_v1 import load_snapshot
    original={'apps/storefront/public'+r['src']:r['sha256']for r in load_snapshot()['receipt']['assets']}
    declared={'apps/storefront/public'+image['src']for p in products for v in p['variants']for image in v['images']}
    source_files={p.relative_to(ROOT).as_posix()for p in (ROOT/'apps/storefront/src').rglob('*') if p.is_file()and p.suffix in('.ts','.tsx','.css','.json')}
    source_files.update({'packages/contracts/data/expanded-store-catalog-v1.json','apps/api/scripts/expanded-store-runtime-export.mts',
        'experiments/ranker/pretrained/store_expanded_binding_v1.py','experiments/ranker/pretrained/store_expanded_engine_v1.py',
        'experiments/ranker/pretrained/store_expanded_rank_v1.py','experiments/ranker/pretrained/store_expanded_parallel_copy_v1.py',
        'experiments/ranker/pretrained/store_persona_portless_http_v1.py'})
    value={'version':VERSION,'catalog':{'file':catalog_path.relative_to(ROOT).as_posix(),'sha256':sha(catalog_path)},
        'sourceSha256':{r:sha(ROOT/r)for r in sorted(source_files)},'originalAssetsSha256':original,
        'expandedAssetsSha256':{r:sha(ROOT/r)for r in sorted(declared-set(original))},
        'actualUIRuntimeExportMatchesSharedCatalog':True,'original16FamilyCheckpointCatalogUnchanged':True,
        'scope':'expanded35 known-public synthetic store; new SKUs use declared family priors plus genuine native catalog features; no model promotion'}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as stream:stream.write(json.dumps(value,indent=2)+'\n')
    ExpandedBinding(path)
    return {'file':str(path.resolve()),'sha256':sha(path),'products':35,'variants':105,'sourceFiles':len(source_files),'newGalleryAssets':len(value['expandedAssetsSha256'])}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--freeze',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(freeze(args.freeze),indent=2))
