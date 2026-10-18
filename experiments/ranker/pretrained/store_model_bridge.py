"""Fail-stale model inputs for the current storefront; no serving or inference."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from .store_catalog_v1 import (COMMIT, DEFAULT_OUT, ROOT, SOURCES, StoreCatalogError,
                               canonical, catalog_text, load_snapshot)

VERSION = 'actual-store-model-candidates.v1'


def assert_current_sources(receipt, source_root=ROOT):
    """Historical authenticity alone is insufficient for active model requests."""
    root = Path(source_root).resolve()
    rows = receipt.get('sources')
    if not isinstance(rows, list) or [r.get('path') for r in rows] != list(SOURCES):
        raise StoreCatalogError('Complete pinned source inventory required for freshness checks')
    for row in rows:
        path = root / row['path']
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise StoreCatalogError('Current storefront source differs from pinned model input: ' + row['path'])
    return True


def prepare_current_catalog(snapshot_directory=DEFAULT_OUT):
    """Keep stable action bindings outside predictive text; reject changed sources.

    The archive loader authenticates pinned source and image bytes. This active
    preparation additionally checks current source bytes before and after all
    encodings, preventing changed products/prices/variant logic from silently
    reusing this historical snapshot. A changed catalog requires a new version.
    """
    started = time.perf_counter()
    snapshot = load_snapshot(snapshot_directory)
    assert_current_sources(snapshot['receipt'])
    inputs, mapping = snapshot['inputs'], snapshot['externalMapping']
    rows = []
    for pi, product in enumerate(inputs['products']):
        external = mapping['products'][pi]
        for ci, color in enumerate(product['colorVariants']):
            routes = external['colorVariants'][ci]
            for ii, _ in enumerate(color['images']):
                text = catalog_text(inputs, pi, ci, ii)
                rows.append({'member': [pi,ci,ii], 'text': text,
                    'textSha256': hashlib.sha256(text.encode()).hexdigest(),
                    'action': {'productId': external['productId'], 'variantId': routes['variantId'],
                               'image': deepcopy(routes['images'][ii])}})
    assert_current_sources(snapshot['receipt'])
    if len(rows) != 195 or len({tuple(r['member']) for r in rows}) != 195:
        raise StoreCatalogError('Current candidate membership differs from exact frozen roster')
    # Raw/projected caches must also bind actual model/tokenizer/runtime and head
    # hashes. This receipt deliberately does not invent those unexecuted keys.
    return {'schemaVersion': VERSION, 'sourceCommit': COMMIT,
        'modelInputSha256': snapshot['receipt']['outputs']['model_inputs.json']['canonicalSha256'],
        'bridgeSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sourceReceipts': deepcopy(snapshot['receipt']['sources']),
        'currentSourcesMatchPinnedInputs': True, 'rows': rows,
        'counts': deepcopy(snapshot['receipt']['counts']),
        'capabilities': deepcopy(inputs['capabilities']),
        'shippingTerms': deepcopy(inputs['shippingTerms']),
        'taskCompatibility': inputs['taskProvenance'],
        'nativeForwards': 0, 'optimizerUpdates': 0, 'selectedActions': [],
        'learnedRankingOrImageSelectionEstablished': False,
        'cpuPreparationMsExcludingImportAndIO': (time.perf_counter()-started)*1000}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve prior results; choose a new directory')
    result=prepare_current_catalog();args.out.mkdir(parents=True,exist_ok=False)
    with (args.out/'report.json').open('x') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    print(canonical({'status':'prepared; no inference','counts':result['counts'],
                     'currentSourcesMatchPinnedInputs':True}))
