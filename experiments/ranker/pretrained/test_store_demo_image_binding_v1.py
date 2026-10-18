import json
from pathlib import Path
import tempfile
import unittest
from .store_demo_image_binding_v1 import freeze_image_delivery, ImageDeliveryCatalogBinding
from .store_demo_catalog import DemoCatalogBinding
from .store_catalog_v1 import load_snapshot, StoreCatalogError


class ImageBindingTests(unittest.TestCase):
    def test_actual_catalog_equivalence_and_original_asset_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            value = freeze_image_delivery(path)
            self.assertTrue(value['imageDelivery']['originalAssetBytesUnchanged'])
            self.assertEqual(value['imageDelivery']['summary']['sourceCount'], 87)
            self.assertTrue(ImageDeliveryCatalogBinding(path).assert_current_sources(load_snapshot()['receipt']))
            with self.assertRaises(FileExistsError):
                freeze_image_delivery(path)
            with self.assertRaises(StoreCatalogError):
                DemoCatalogBinding(path)

    def test_unreviewed_renderer_and_changed_model_facts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            value = freeze_image_delivery(path)
            value['rendererOnlyExceptions']['apps/storefront/src/config.ts'] = '0' * 64
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(StoreCatalogError, 'reviewed'):
                ImageDeliveryCatalogBinding(path)
            del value['rendererOnlyExceptions']['apps/storefront/src/config.ts']
            value['modelInputSha256'] = '0' * 64
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(StoreCatalogError, 'model facts'):
                ImageDeliveryCatalogBinding(path)

    def test_photo_source_pin_and_post_startup_manifest_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'; value = freeze_image_delivery(path)
            binding = ImageDeliveryCatalogBinding(path)
            value['frontendSha256']['apps/storefront/src/ProductPhoto.tsx'] = '0' * 64
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(StoreCatalogError, 'receipt changed'):
                binding.assert_current_sources(load_snapshot()['receipt'])
            with self.assertRaisesRegex(StoreCatalogError, 'source changed'):
                ImageDeliveryCatalogBinding(path)


if __name__ == '__main__':
    unittest.main()
