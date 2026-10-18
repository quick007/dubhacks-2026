from copy import deepcopy
import json
from pathlib import Path
import unittest
from . import store_expanded_binding_v1 as binding


class Tests(unittest.TestCase):
    def products(self):
        return binding.normalized_products(json.loads(Path('/Users/lukas/Documents/Projects/dubhacks-2026/packages/contracts/data/expanded-store-catalog-v1.json').read_text()))
    def test_real35_preserves_complete_original16_snapshot_gallery_metadata(self):
        products=self.products();self.assertEqual(len(products),35);self.assertTrue(binding.check_original_projection(products))
        bad=deepcopy(products);bad[0]['variants'][0]['images'][0]['alt']='changed real metadata'
        with self.assertRaisesRegex(ValueError,'image metadata'):binding.check_original_projection(bad)
    def test_original_identity_price_and_variant_color_cannot_be_substituted(self):
        for mutate in (lambda p:p[0].update(itemPrice=25),lambda p:p[0].update(name='Different name'),lambda p:p[0]['variants'][0].update(colorName='Other')):
            products=self.products();mutate(products)
            with self.assertRaises(ValueError):binding.check_original_projection(products)

if __name__=='__main__':unittest.main()
