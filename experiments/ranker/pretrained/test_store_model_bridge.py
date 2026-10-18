"""Active-source freshness and complete real gallery action-binding checks."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from .store_catalog_v1 import ROOT, SOURCES, StoreCatalogError, load_snapshot
from .store_model_bridge import assert_current_sources, prepare_current_catalog


class StoreBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.snapshot=load_snapshot()

    def test_complete195_encodings_bind_original_product_variant_image(self):
        result=prepare_current_catalog()
        self.assertEqual(len(result['rows']),195)
        self.assertFalse(result['learnedRankingOrImageSelectionEstablished'])
        self.assertEqual(result['selectedActions'],[])
        for row in result['rows']:
            pi,ci,ii=row['member'];route=self.snapshot['externalMapping']['products'][pi]
            self.assertEqual(row['action']['productId'],route['productId'])
            self.assertEqual(row['action']['variantId'],route['colorVariants'][ci]['variantId'])
            self.assertEqual(row['action']['image'],route['colorVariants'][ci]['images'][ii])
            self.assertEqual(row['textSha256'],hashlib.sha256(row['text'].encode()).hexdigest())

    def test_changed_product_source_fails_without_rewriting_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in SOURCES:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
            self.assertTrue(assert_current_sources(self.snapshot['receipt'],root))
            path=root/'apps/storefront/src/config.ts';path.write_bytes(path.read_bytes()+b'\n// changed catalog\n')
            with self.assertRaisesRegex(StoreCatalogError,'source differs'):
                assert_current_sources(self.snapshot['receipt'],root)

    def test_missing_source_and_partial_inventory_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(StoreCatalogError):assert_current_sources(self.snapshot['receipt'],directory)
        receipt=dict(self.snapshot['receipt']);receipt['sources']=receipt['sources'][:-1]
        with self.assertRaises(StoreCatalogError):assert_current_sources(receipt)


if __name__ == '__main__':unittest.main()
