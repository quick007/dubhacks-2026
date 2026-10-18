"""Stdlib-only independent receipt audit; optional explicit lossless archive.

No model imports, worker launch, source mutation or original receipt rewrite.
All twenty planned pairs stay in the readout, including negative observer deltas.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUNS = [ROOT / f'data/ranker/pretrained/store-fast-path-profile-v1-attempt-{n}' for n in (1, 2)]
TENSORS = {'.pt', '.pth', '.safetensors', '.npy', '.npz', '.bin', '.gguf'}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def stats(values):
    ordered = sorted(values)
    return {'n': len(ordered), 'p50': statistics.median(ordered),
            'p95NearestRank': ordered[math.ceil(.95 * len(ordered)) - 1],
            'min': ordered[0], 'max': ordered[-1]}


def audit():
    inventory = []
    failures = []
    results = []
    for directory in RUNS:
        result = json.loads((directory / 'results.json').read_text())
        preflight = json.loads((directory / 'preflight.json').read_text())
        assert sha(directory / 'preflight.json') == result['preflightSha256']
        for name, expected in preflight['sourceSha256'].items():
            assert sha(directory / 'source' / name) == expected, name
        assert all(result['shutdownIntegrity'][key] is True for key in (
            'frozen927HashesUnchanged', 'copyCheckpointUnchanged', 'candidateAssetsUnchanged',
            'imageServingSourcesUnchanged', 'v5ServingExportUnchanged', 'v5HeadCheckpointUnchanged'))
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                assert not path.is_symlink() and path.suffix not in TENSORS
                inventory.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
        results.append((result, preflight))
    failed, first_preflight = results[0]
    good, preflight = results[1]
    assert failed['status'] == 'failed-profile' and len(failed['warmups']) == 2
    assert not failed['records'] and failed['summary']['unattemptedArms'] == 40
    assert 'TokenizerWrapper' in str(failed['error']) and 'encode' in str(failed['error'])
    failures.append({'attempt': 1, 'error': failed['error'], 'failureStage': 'excluded B warmup restoration',
        'timedArmsAttempted': 0, 'timedArmsUnattempted': 40, 'shutdownIntegrity': failed['shutdownIntegrity']})
    assert good['status'] == 'completed-profile-not-optimized' and len(good['warmups']) == 2
    records = good['records']
    assert len(records) == 40 and len(preflight['cases']) == 20
    paired = defaultdict(dict)
    phases = defaultdict(list)
    accounting_errors = []
    for record in records:
        assert record['status'] == 'completed'
        signature = record['parity']
        receipt = json.loads((ROOT / record['receiptPath']).read_text())
        generated = receipt['parallelGeneration']
        reconstructed = {'scores16': receipt['ranking']['scores16'], 'finalOrder': receipt['finalOrder'],
            'cards': receipt['cards'], 'runtimeRows': receipt['runtimeRows'],
            'copyWordFields': [{key: field.get(key) for key in ('text', 'slotIds', 'finishReason', 'firstEosSlot',
                'rejections', 'accepted')} for field in generated['fields']],
            'endpoints': [{key: event.get(key) for key in ('type', 'productIds', 'selectedProductIds', 'variants',
                'sections', 'acceptedFields', 'withheldFields', 'requiredFields', 'validation', 'provenance',
                'copySource')} for event in receipt['events']]}
        assert receipt['status'] == 'completed' and reconstructed == signature['logical']
        assert generated['feedforwardCalls'] == 1 and generated['fieldCount'] == 20
        assert generated['teacherStringsRead'] == 0
        actual = sha_bytes(json.dumps(signature['logical'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        assert actual == signature['sha256']
        assert (signature['cardCount'], signature['copyFieldCount'], signature['freshFeedforwardCalls'],
                signature['freshCopySource']) == (6, 20, 1, 'fresh-parallel-head')
        assert record['quality']['firstThree'] == {'acceptedFields': 10, 'withheldFields': 0, 'requiredFields': 10}
        assert record['quality']['complete'] == {'acceptedFields': 20, 'withheldFields': 0, 'requiredFields': 20}
        assert not record['warmupExcluded'] and record['arm'] not in paired[record['caseId']]
        paired[record['caseId']][record['arm']] = record
        if record['arm'] == 'B-profile':
            profile = record['profile']['throughFullSixWire']
            timeline = profile['timeline']
            by_id = {span['id']: span for span in timeline}
            for span in timeline:
                children = [child for child in timeline if child['parent'] == span['id']]
                exclusive = span['endMs'] - span['startMs'] - sum(child['endMs'] - child['startMs'] for child in children)
                assert abs(exclusive - span['exclusiveMs']) < 1e-7
                if span['parent'] is not None:
                    parent = by_id[span['parent']]
                    assert parent['startMs'] <= span['startMs'] <= span['endMs'] <= parent['endMs'] + 1e-8
            total = sum(span['exclusiveMs'] for span in timeline)
            accounting_errors.append(total - profile['wallMs'])
            assert abs(total - profile['wallMs']) < 1e-7
            for name, value in profile['phases'].items():
                phases[name].append(value)
    assert set(paired) == {case['caseId'] for case in preflight['cases']}
    for case in preflight['cases']:
        pair = paired[case['caseId']]
        assert list(pair) == case['armOrder']
        assert pair['A-original']['input'] == pair['B-profile']['input'] == case['request']
        assert pair['A-original']['parity']['logical'] == pair['B-profile']['parity']['logical']
    timings = {arm: {key: stats([record['timing'][key] for record in records if record['arm'] == arm])
        for key in ('firstThreeWireReadyMs', 'fullSixWireReadyMs')} for arm in ('A-original', 'B-profile')}
    report = {'version': 'fast-path-profile-independent-audit.v1', 'status': 'PASS',
        'scope': 'same fast model original versus instrumented server path; no generic UI baseline or browser paint',
        'selectionPolicy': 'retain all original20 planned public-query pairs; no outlier or negative-delta removal',
        'distinctPublicQueries': preflight['distinctPublicQueries'], 'completePairs': len(paired),
        'exactLogicalParityPairs': len(paired), 'timedArms': len(records), 'cardsPerArm': 6, 'copyFieldsPerArm': 20,
        'rawRequestReceiptsIndependentlyReconstructed': len(records),
        'timingMs': timings, 'pairedObserverFullSixDeltaMs': stats([
            pair['B-profile']['timing']['fullSixWireReadyMs'] - pair['A-original']['timing']['fullSixWireReadyMs'] for pair in paired.values()]),
        'phaseMs': {name: {key: stats([value[key] for value in values]) for key in ('inclusiveMs', 'exclusiveMs')}
            for name, values in sorted(phases.items())},
        'maxAbsoluteIndependentPartitionErrorMs': max(map(abs, accounting_errors)),
        'accountingLimits': ['inclusive intervals overlap', 'exclusive partition checked independently per request',
            'phase medians do not add to median total', 'observed B cost is retained; no overhead subtraction',
            'native encode already synchronizes; wrapper adds no GPU synchronization',
            '170ms encode includes native completion/pooling/transfer; no isolated kernel attribution'],
        'excludedWarmupFailurePreserved': failures[0], 'successfulShutdownIntegrity': good['shutdownIntegrity'],
        'originalResultsSha256': {str(path.relative_to(ROOT)): sha(path / 'results.json') for path in RUNS},
        'originalPreflightSha256': {str(path.relative_to(ROOT)): sha(path / 'preflight.json') for path in RUNS},
        'sourcePinCountPerAttempt': [len(value['sourceSha256']) for _, value in results],
        'archiveInventory': inventory, 'archiveFiles': len(inventory), 'archiveUncompressedBytes': sum(row['bytes'] for row in inventory),
        'tensorFilesIncluded': 0, 'archiveStatus': 'inventory ready; compression explicitly deferred during browser measurements'}
    return report


def write_readout(report, output):
    output = Path(output)
    if output.exists() or output.with_suffix('.md').exists():
        raise FileExistsError('New unused report paths required')
    output.write_text(json.dumps(report, indent=2) + '\n')
    a = report['timingMs']['A-original']; b = report['timingMs']['B-profile']
    delta = report['pairedObserverFullSixDeltaMs']; phases = report['phaseMs']
    output.with_suffix('.md').write_text(f'''All20 planned pairs pass exact rank, generated-word/slot-ID, action, validation and provenance parity. Every timed arm returned6 cards and20 accepted fresh parallel fields. All40 timed arms are retained, with two startup warmups excluded.

| Server event | Original p50 / p95 ms | Profiled p50 / p95 ms |
| --- | ---: | ---: |
| First3 serialized | {a['firstThreeWireReadyMs']['p50']:.4f} / {a['firstThreeWireReadyMs']['p95NearestRank']:.4f} | {b['firstThreeWireReadyMs']['p50']:.4f} / {b['firstThreeWireReadyMs']['p95NearestRank']:.4f} |
| Full6 serialized | {a['fullSixWireReadyMs']['p50']:.4f} / {a['fullSixWireReadyMs']['p95NearestRank']:.4f} | {b['fullSixWireReadyMs']['p50']:.4f} / {b['fullSixWireReadyMs']['p95NearestRank']:.4f} |

Paired B−A full6 delta: median {delta['p50']:.4f}ms, nearest-rank p95 {delta['p95NearestRank']:.4f}ms, range {delta['min']:.4f} to {delta['max']:.4f}ms. Difference of global medians is a different statistic. This measures observer cost on the same fast model; it is not a generic UI latency comparison, browser paint or a speedup experiment.

Profiled exclusive median fresh backbone encode is {phases['freshBackboneEncode']['exclusiveMs']['p50']:.4f}ms; catalog loads {phases['catalogFeatureLoad']['exclusiveMs']['p50']:.4f}ms; parallel text head forward {phases['parallelHeadForward']['exclusiveMs']['p50']:.5f}ms. Whole parallel generation inclusive median is {phases['parallelTextGeneration']['inclusiveMs']['p50']:.4f}ms. The tiny head-forward number does not describe full request latency. Native encode includes its existing completion/pooling/transfer boundary; there is no isolated GPU-kernel timing.

The exclusive partition was independently recomputed from every raw parent/child interval. Inclusive times overlap. Phase medians cannot be added to obtain the median total. No observer cost is subtracted from reported request times.

Attempt1 remains a failed excluded B warmup: TokenizerWrapper encode restoration raised AttributeError, zero timed arms ran and all40 were unattempted. Its original sources, error and full927/adapter/head/export shutdown proofs remain in the inventory. Attempt2 reports all corresponding integrity checks true. Neither run is overwritten.

The lossless inventory covers {report['archiveFiles']} original JSON/source files ({report['archiveUncompressedBytes']:,} bytes), including all raw request receipts and both preflights. It includes no tensors. Compression is deferred until the browser timing window is released. The companion audit script can produce and verify a new archive with an explicit --archive path.
''')


def archive(report, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Unused archive path required')
    for row in report['archiveInventory']:
        assert sha(ROOT / row['path']) == row['sha256']
    # New archive only, no symlink dereference or modification of originals.
    with tarfile.open(destination, 'x:gz', compresslevel=1) as target:
        manifest = json.dumps(report, indent=2).encode() + b'\n'
        entry = tarfile.TarInfo('_archive-manifest.json'); entry.size = len(manifest)
        target.addfile(entry, io.BytesIO(manifest))
        for row in report['archiveInventory']:
            target.add(ROOT / row['path'], arcname=row['path'], recursive=False)
    with tarfile.open(destination, 'r:gz') as source:
        assert source.getnames() == ['_archive-manifest.json'] + [row['path'] for row in report['archiveInventory']]
        assert source.extractfile('_archive-manifest.json').read() == manifest
        for row in report['archiveInventory']:
            assert sha_bytes(source.extractfile(row['path']).read()) == row['sha256']
    return {'archive': str(destination), 'sha256': sha(destination), 'bytes': destination.stat().st_size,
            'originalMembers': report['archiveFiles'], 'manifestMembers': 1, 'allMemberBytesVerified': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report', type=Path)
    parser.add_argument('--archive', type=Path, help='Explicit compression after browser timing window release')
    args = parser.parse_args()
    packet = audit()
    if args.write_report:
        write_readout(packet, args.write_report)
    if args.archive:
        print(json.dumps(archive(packet, args.archive), indent=2))
    else:
        print(json.dumps({key: packet[key] for key in ('status', 'completePairs', 'exactLogicalParityPairs',
            'pairedObserverFullSixDeltaMs', 'archiveFiles', 'archiveUncompressedBytes')}, indent=2))
