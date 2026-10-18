"""Unpromoted CPU candidate: verified catalog tensors/projections, fresh queries.

Explicit factory only. Original32 tensor/receipt files are fully SHA256 checked
at both request boundaries. No generated words, responses or query features are
cached. Existing v5/base source, head, constraints and native controls still run.
The ~4ms observed deserialize opportunity is small and not a claimed speedup.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path
import time

import torch

from .store_catalog_v1 import canonical
from .store_model_bridge import assert_current_sources
from .store_rank_session_v5 import V5RankSession, read_export, sha

VERSION = 'verified-store-catalog-cache.cpu-candidate.v1'


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class VerifiedCatalogCache:
    """Private immutable catalog only; loading unknown rows fails closed."""
    def __init__(self, store, head, rows, file_pins):
        self.store = store
        self.head = head
        self.rows = deepcopy(rows)
        self.file_pins = {Path(name).resolve(): expected for name, expected in file_pins.items()}
        required = {path.resolve() for row in rows for path in store.paths(row['role'], row['member'])}
        if len(rows) != 16 or len(self.file_pins) != 32 or set(self.file_pins) != required:
            raise ValueError('Exactly16 catalog rows and32 original tensor/receipt file pins required')
        if any(row['role'] != 'catalog' for row in rows):
            raise ValueError('Only public catalog features may be cached')
        self._verify_files()
        self._values = {}
        for row in rows:
            value, receipt = store.load(row)
            key = canonical(row)
            if key in self._values:
                raise ValueError('Distinct exact catalog rows required')
            self._values[key] = (value.detach().clone(), deepcopy(receipt))
        self._catalog = torch.cat([self._values[canonical(row)][0] for row in rows])
        with torch.no_grad():
            self._probabilities = head.probabilities('catalog', self._catalog).detach().clone()
        self._fingerprints = self._memory_fingerprints()
        self._active = False
        self.completed_requests = 0
        self._verify_files()

    def _verify_files(self):
        for path, expected in self.file_pins.items():
            if path.is_symlink() or sha(path) != expected:
                raise ValueError('Original catalog tensor/receipt checksum changed')

    def _memory_fingerprints(self):
        return [tensor_sha(value) for value, _ in self._values.values()] + [
            tensor_sha(self._catalog), tensor_sha(self._probabilities)]

    def assert_integrity(self):
        self._verify_files()
        if self._fingerprints != self._memory_fingerprints():
            raise ValueError('Private cached catalog tensor/projection changed')
        return True

    @contextmanager
    def request_scope(self):
        if self._active:
            raise RuntimeError('Catalog cache requires the existing serial worker')
        self.assert_integrity()
        self._active = True
        try:
            yield
            self.completed_requests += 1
        finally:
            self._active = False
            self.assert_integrity()

    def load(self, row):
        if not self._active or canonical(row) not in self._values:
            raise ValueError('Verified active request and exact public catalog row required')
        value, receipt = self._values[canonical(row)]
        # Callers receive owned values. No mutation can alter later requests.
        return value.clone(), deepcopy(receipt)

    def probabilities(self, role, values):
        if role == 'catalog':
            if not self._active or not torch.equal(values, self._catalog):
                raise ValueError('Exact complete original16 catalog tensor required')
            return self._probabilities.clone()
        # Every query projection executes on the fresh native input.
        return self.head.probabilities(role, values)

    def provenance(self):
        return {'version': VERSION, 'catalogVectorsReused': 16, 'catalogProjectionReused': True,
            'originalTensorAndReceiptFilesFullySha256VerifiedPerRequest': 32,
            'verificationBoundaries': ['before request', 'after request'],
            'queryFeaturesCached': False, 'responseWordsCached': False,
            'numericPolicy': 'same original16 batch shape and unchanged head; no new precision or approximation',
            'scope': 'explicit unpromoted candidate; latency benefit unmeasured'}


class _CatalogStoreProxy:
    def __init__(self, cache):
        self.cache = cache

    def load(self, row):
        return self.cache.load(row)

    def __getattr__(self, name):
        return getattr(self.cache.store, name)


class _CatalogHeadProxy:
    def __init__(self, cache):
        self.cache = cache

    def probabilities(self, role, values):
        return self.cache.probabilities(role, values)

    def __getattr__(self, name):
        return getattr(self.cache.head, name)


class CachedCatalogV5RankSession(V5RankSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        export = read_export(self.head_export_path, self.head_export_sha)
        self.catalog_cache = VerifiedCatalogCache(self.store, self.head, self.catalog_rows,
            export['originalCatalogFeatureFilesSha256'])
        self.store = _CatalogStoreProxy(self.catalog_cache)
        self.head = _CatalogHeadProxy(self.catalog_cache)
        self.startup_provenance['catalogCache'] = self.catalog_cache.provenance()

    def integrity_receipt(self):
        result = super().integrity_receipt()
        if hasattr(self, 'catalog_cache'):
            self.catalog_cache.assert_integrity()
            result['catalogCacheUnchanged'] = True
        return result

    def rank_request(self, request, *, copy_adapter=None):
        started = time.perf_counter()
        with self.catalog_cache.request_scope():
            result = super().rank_request(request, copy_adapter=copy_adapter)
        payload = {key: value for key, value in result.items() if key not in (
            'payloadSha256', 'payloadBytes', 'freshRankToSerializedActionsMs')}
        payload['catalogCache'] = self.catalog_cache.provenance()
        encoded = canonical(payload).encode()
        return {**payload, 'payloadSha256': hashlib.sha256(encoded).hexdigest(), 'payloadBytes': len(encoded),
            'freshRankToSerializedActionsMs': (time.perf_counter() - started) * 1000}


def make_cached_catalog_v5_factory(head_export_path, head_export_sha256):
    def factory(backbone, run_directory, *, source_validator=assert_current_sources):
        return CachedCatalogV5RankSession(backbone, run_directory, head_export_path=head_export_path,
            head_export_sha256=head_export_sha256, source_validator=source_validator)
    return factory
