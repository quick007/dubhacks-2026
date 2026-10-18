"""CPU fail-stale integration checks; fixture bytes are not model evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import store_demo_catalog as demo

class DemoCatalogTests(unittest.TestCase):
    def test_only_recorded_renderer_bytes_can_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for name in demo.SOURCES:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original:'+name)
                rows.append({'path':name,'sha256':demo.sha(p)})
            renderer=root/demo.RENDERER;renderer.write_text('reviewed optional generated-copy props')
            frontend={}
            for name in demo.FRONTEND:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
                if not p.exists():p.write_text('reviewed frontend:'+name)
                frontend[name]=demo.sha(p)
            manifest={'rendererOnlyExceptions':{demo.RENDERER:demo.sha(renderer)},'frontendSha256':frontend}
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            binding=demo.DemoCatalogBinding.__new__(demo.DemoCatalogBinding)
            binding.path=path;binding.manifest=manifest;binding.manifest_sha=demo.sha(path)
            with patch.object(demo,'ROOT',root):
                self.assertTrue(binding.assert_current_sources({'sources':rows}))
                catalog=root/'apps/storefront/src/config.ts';catalog.write_text('changed item price')
                with self.assertRaisesRegex(demo.StoreCatalogError,'Active demo source changed'):
                    binding.assert_current_sources({'sources':rows})
    def test_renderer_and_new_client_changes_invalidate_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for name in demo.SOURCES:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name);rows.append({'path':name,'sha256':demo.sha(p)})
            for name in demo.FRONTEND:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
                if not p.exists():p.write_text(name)
            manifest={'rendererOnlyExceptions':{},'frontendSha256':{n:demo.sha(root/n) for n in demo.FRONTEND}}
            path=root/'manifest.json';path.write_text(json.dumps(manifest));binding=demo.DemoCatalogBinding.__new__(demo.DemoCatalogBinding)
            binding.path=path;binding.manifest=manifest;binding.manifest_sha=demo.sha(path)
            with patch.object(demo,'ROOT',root):
                (root/'apps/storefront/src/use-local-model.ts').write_text('unreviewed')
                with self.assertRaisesRegex(demo.StoreCatalogError,'Active demo client changed'):
                    binding.assert_current_sources({'sources':rows})
                path.write_text(json.dumps({**manifest,'unexpected':True}))
                with self.assertRaisesRegex(demo.StoreCatalogError,'receipt changed'):
                    binding.assert_current_sources({'sources':rows})
    def test_no_unrecorded_source_inventory_is_accepted(self):
        binding=demo.DemoCatalogBinding.__new__(demo.DemoCatalogBinding)
        with tempfile.TemporaryDirectory() as tmp:
            binding.path=Path(tmp)/'manifest.json';binding.path.write_text('{}');binding.manifest_sha=demo.sha(binding.path)
            binding.manifest={'rendererOnlyExceptions':{},'frontendSha256':{}}
            with self.assertRaisesRegex(demo.StoreCatalogError,'Full source inventory'):
                binding.assert_current_sources({'sources':[]})
if __name__=='__main__':unittest.main()
