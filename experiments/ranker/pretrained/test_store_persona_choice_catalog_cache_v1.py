"""CPU exact-token-boundary tests, no native cache evidence."""
import unittest
from .store_persona_choice_catalog_cache_v1 import checked_prefix

class CatalogPrefixTests(unittest.TestCase):
    def test_exact_literal_prefix(self):
        self.assertEqual(checked_prefix({'tokenIds':list(range(24)),'catalogPrefixTokenIds':[0,1]}),[0,1])
    def test_prefix_mismatch_or_dynamic_final_token_rejected(self):
        for value in ([],[0,3],list(range(8)),list(range(24))):
            with self.assertRaises(ValueError):checked_prefix({'tokenIds':list(range(24)),'catalogPrefixTokenIds':value})

if __name__=='__main__':unittest.main()
