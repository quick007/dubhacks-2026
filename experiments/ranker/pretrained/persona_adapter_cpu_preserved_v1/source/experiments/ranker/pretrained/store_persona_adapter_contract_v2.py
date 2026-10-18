"""Target-free public adapter serving contracts; CPU-only imports.

New endpoint bounds are explicit intersections with the learned color runtime.
No TRAIN target, person/case identifier, teacher response or private label is
accepted by the cache/suffix APIs. Existing training/gate sources are untouched.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading

VERSION = 'persona-choice-adapter-runtime.v2'
TARGET = 'language_model.model.layers.31.mlp.down_proj'
INDEX_FIELDS = {'version', 'entries', 'teacherOrTargetsIncluded', 'key', 'modelRevision',
    'formatterSha256', 'baseModelFiles', 'scoreSourceSha256', 'featureSourceSha256'}
ENTRY_FIELDS = {'archive', 'archiveSha256', 'dtypes', 'routeTokenIds', 'promptTokenIdsSha256',
    'promptTokens', 'prefixPath', 'nativeInputSuffixTokens'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def exact_sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Exact lowercase SHA256 required')
    return value


def local_file(root, name):
    root = Path(root).resolve()
    if type(name) is not str or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Explicit relative co-located artifact required')
    result = root / name
    if result.is_symlink() or not result.resolve().is_relative_to(root):
        raise ValueError('Co-located regular artifact required; no original-path fallback')
    return result


def common_request(value, formatter_catalog, legacy_catalog):
    """Reject excess input; never silently trim observations or substitute IDs."""
    def bounded(item):
        if isinstance(item, str) and len(item) > 500:
            raise ValueError('Public adapter endpoint requires all text values at most500 characters; no clipping')
        if isinstance(item, dict):
            for child in item.values(): bounded(child)
        if isinstance(item, list):
            for child in item: bounded(child)
    bounded(value)
    from . import store_qwen_lexical_choice_inputs_v2 as formatter
    from .store_persona_context_v1 import checked_context_request
    formatter.checked_request(value, formatter_catalog)  # history0..12/background<=6/colors<=4
    return checked_context_request(value, legacy_catalog)  # original strict budget/color/hardguard parsing


def suffix_contract(row):
    allowed = {'prompt', 'tokenIds', 'tokenIdsSha256', 'routeTokenIds', 'catalogPrefixTokenIds'}
    if set(row) != allowed:
        raise ValueError('Exact target-free public suffix field roster required')
    ids = row['tokenIds']; routes = row['routeTokenIds']; prefix = row['catalogPrefixTokenIds']
    if (type(ids) is not list or not 738 <= len(ids) <= 2048 or
            any(type(t) is not int or not 0 <= t < 248320 for t in ids) or token_hash(ids) != row['tokenIdsSha256']):
        raise ValueError('Complete actual public prompt tokens required, without answer/EOS append or truncation')
    if (type(routes) is not list or len(routes) != 16 or len(set(routes)) != 16 or
            any(type(t) is not int or not 0 <= t < 248320 for t in routes)):
        raise ValueError('Exactly16 product-bound lexical route token IDs required')
    if type(prefix) is not list or len(prefix) != 721 or ids[:721] != prefix or len(prefix) >= len(ids) - 16:
        raise ValueError('Exact721-token literal catalog prefix plus nonempty actual request prefix required')
    if type(row['prompt']) is not str or not row['prompt']:
        raise ValueError('Actual public prompt required')
    return {'prefixTokenIds': ids[:-16], 'suffixTokenIds': ids[-16:],
        'catalogPrefixTokenIds': list(prefix), 'routeTokenIds': list(routes), 'promptTokenIdsSha256': token_hash(ids)}


class ProjectionOwnership:
    """One worker, three distinct identities sharing exactly one pristine linear.

    Synchronization is caller-supplied so CPU fixtures never initialize MLX.
    Copy is the idle/default projection. Rank/copy adapters are never nested,
    fused, summed, trained or loaded into each other's A/B tensors.
    """
    def __init__(self, model, copy_adapter, rank_adapter, synchronize):
        self.layer = model.layers[-1].mlp
        self.copy = copy_adapter; self.rank = rank_adapter
        self.base = copy_adapter.linear
        self.sync = synchronize; self.thread = threading.get_ident(); self.active = None
        if len(model.layers) != 32 or self.copy is self.rank or self.rank.linear is not self.base:
            raise ValueError('Separate copy/rank adapters on the same pristine final31 base required')
        if hasattr(self.base, 'lora_a') or hasattr(self.base, 'lora_b'):
            raise ValueError('Nested rank/copy adapter is forbidden')
        self.assert_idle()

    def assert_idle(self):
        if threading.get_ident() != self.thread or self.active is not None or self.layer.down_proj is not self.copy:
            raise RuntimeError('Serial original-copy projection ownership required')
        return True

    @contextmanager
    def select(self, mode):
        if mode not in ('base', 'copy', 'rank'):
            raise ValueError('Explicit base/copy/rank identity required')
        self.assert_idle(); selected = getattr(self, mode)
        self.sync(); self.active = mode; self.layer.down_proj = selected
        try:
            yield selected
            if self.layer.down_proj is not selected:
                raise RuntimeError('Projection changed unexpectedly inside owned operation')
        finally:
            try:
                self.sync()
            finally:
                self.layer.down_proj = self.copy; self.active = None
        self.assert_idle()


class PublicFeatureCache:
    def __init__(self, filename, expected_sha, *, routes, revision, formatter_sha, score_sha, feature_sha):
        self.path = Path(filename); self.expected_sha = exact_sha(expected_sha)
        if sha(self.path) != self.expected_sha:
            raise ValueError('Public feature index checksum changed')
        value = json.loads(self.path.read_text())
        if (set(value) != INDEX_FIELDS or value['version'] != 'persona-choice-public-feature-cache.v1' or
                value['teacherOrTargetsIncluded'] is not False or value['modelRevision'] != revision or
                value['formatterSha256'] != formatter_sha or value['scoreSourceSha256'] != score_sha or
                value['featureSourceSha256'] != feature_sha or
                type(value['entries']) is not dict or len(value['entries']) != 20):
            raise ValueError('Exact20 pure public target-free feature index required')
        self.entries = deepcopy(value['entries']); self.routes = list(routes)
        self.paths = {}
        for key, entry in self.entries.items():
            exact_sha(key)
            if (set(entry) != ENTRY_FIELDS or entry['promptTokenIdsSha256'] != key or entry['routeTokenIds'] != self.routes or
                    entry['nativeInputSuffixTokens'] != 16 or type(entry['promptTokens']) is not int or not 738 <= entry['promptTokens'] <= 2048):
                raise ValueError('Target/identifier-free exact public prompt feature metadata required')
            filename = local_file(self.path.parent, entry['archive'])
            if filename.name != key + '.npz':
                raise ValueError('Public feature archive must use actual prompt token SHA filename')
            self.paths[key] = filename
        self.assert_current(full=True)

    def assert_current(self, *, full=False):
        if sha(self.path) != self.expected_sha:
            raise ValueError('Public feature index changed')
        if full:
            for key, filename in self.paths.items():
                if filename.is_symlink() or sha(filename) != exact_sha(self.entries[key]['archiveSha256']):
                    raise ValueError('Public feature archive changed')
        return True

    def get(self, row):
        contract = suffix_contract(row); key = contract['promptTokenIdsSha256']
        self.assert_current()
        if key not in self.entries:
            return None
        entry = self.entries[key]; filename = self.paths[key]
        if (entry['promptTokens'] != len(row['tokenIds']) or entry['routeTokenIds'] != row['routeTokenIds'] or
                filename.is_symlink() or sha(filename) != entry['archiveSha256']):
            raise ValueError('Matched public source feature identity changed')
        import numpy as np
        from .store_persona_choice_scoring_v1 import validate
        with np.load(filename, allow_pickle=False) as arrays:
            if set(arrays.files) != {'x', 'h', 'y'}:
                raise ValueError('Public archives may contain only source X/H/Y, never targets or labels')
            item = {key: arrays[key].copy() for key in ('x', 'h', 'y')}
        item.update(dtypes=deepcopy(entry['dtypes']), routeTokenIds=list(entry['routeTokenIds']))
        validate(item)
        if any(not np.isfinite(item[key]).all() for key in ('x', 'h', 'y')):
            raise ValueError('Finite native public source features required')
        return item
