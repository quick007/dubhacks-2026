"""Small CPU fixtures only; no model load, timing claims or source validation subprocess."""
from pathlib import Path
import tempfile
import unittest

import torch

from .store_rank_catalog_cache_v1 import VerifiedCatalogCache, sha


class FakeStore:
    def __init__(self, directory):
        self.root = Path(directory); self.loads = 0
        for member in range(16):
            for path in self.paths('catalog', member):
                path.write_bytes(f'original public catalog {member} {path.suffix}'.encode())

    def paths(self, role, member):
        return self.root / f'{member}.pt', self.root / f'{member}.json'

    def load(self, row):
        self.loads += 1
        return torch.tensor([[row['member'] + .5, 1.0]], dtype=torch.float32), {'member': row['member']}


class FakeHead:
    def __init__(self):
        self.calls = []

    def probabilities(self, role, values):
        self.calls.append(role)
        return torch.softmax(values, dim=-1)


class CatalogCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.store = FakeStore(self.temp.name); self.head = FakeHead()
        self.rows = [{'role': 'catalog', 'member': i, 'text': f'public {i}'} for i in range(16)]
        self.pins = {str(path): sha(path) for row in self.rows for path in self.store.paths('catalog', row['member'])}
        self.cache = VerifiedCatalogCache(self.store, self.head, self.rows, self.pins)

    def test_reuse_exact_probabilities_and_fresh_query_projection(self):
        expected = torch.softmax(torch.cat([self.store.load(row)[0] for row in self.rows]), dim=-1)
        loads = self.store.loads
        for query in (torch.tensor([[1., 3.]]), torch.tensor([[4., 2.]])):
            with self.cache.request_scope():
                actual = torch.cat([self.cache.load(row)[0] for row in self.rows])
                self.assertTrue(torch.equal(self.cache.probabilities('catalog', actual), expected))
                self.assertTrue(torch.equal(self.cache.probabilities('query', query), torch.softmax(query, -1)))
        self.assertEqual(self.store.loads, loads)
        self.assertEqual(self.head.calls, ['catalog', 'query', 'query'])

    def test_all_original32_files_rechecked_before_and_after_each_request(self):
        for path in self.pins:
            original = Path(path).read_bytes(); Path(path).write_bytes(b'mutated')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                with self.cache.request_scope():
                    self.fail('changed source entered request')
            Path(path).write_bytes(original)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            with self.cache.request_scope():
                Path(next(iter(self.pins))).write_bytes(b'mutated during request')
        self.assertFalse(self.cache._active)

    def test_unknown_rows_inactive_access_and_changed_catalog_reject(self):
        with self.assertRaises(ValueError):
            self.cache.load(self.rows[0])
        with self.cache.request_scope():
            with self.assertRaises(ValueError):
                self.cache.load({**self.rows[0], 'text': 'different'})
            with self.assertRaises(ValueError):
                self.cache.probabilities('catalog', torch.zeros(16, 2))
            with self.assertRaises(RuntimeError):
                with self.cache.request_scope():
                    pass

    def test_owned_tensor_receipt_and_projection_cannot_mutate_cache(self):
        with self.cache.request_scope():
            value, receipt = self.cache.load(self.rows[0]); value.zero_(); receipt['member'] = 99
            catalog = torch.cat([self.cache.load(row)[0] for row in self.rows])
            self.cache.probabilities('catalog', catalog).zero_()
            self.assertEqual(self.cache.load(self.rows[0])[1]['member'], 0)
            self.assertEqual(self.cache.load(self.rows[0])[0][0, 0].item(), .5)
        self.assertTrue(self.cache.assert_integrity())

    def test_request_error_restores_scope_and_private_mutation_detected(self):
        with self.assertRaisesRegex(RuntimeError, 'original error'):
            with self.cache.request_scope():
                raise RuntimeError('original error')
        self.assertFalse(self.cache._active)
        self.cache._probabilities.zero_()
        with self.assertRaisesRegex(ValueError, 'Private cached'):
            self.cache.assert_integrity()


if __name__ == '__main__':
    unittest.main()
