"""CPU-only, lossless small measurement preservation package. Never rewrites inputs."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'experiments/ranker/pretrained'
NAMES = (
    'store_candidate_browser_probe_results_v3_compact_oldrank_attempt_1',
    'store_candidate_browser_probe_results_v3_compact_oldrank_attempt_2',
    'store_candidate_browser_probe_results_v4_compact_v5rank_attempt_1',
    'store_candidate_collection_probe_results_v4_compact_v5rank_attempt_1',
    'store_paired_incremental_probe_results_v1_attempt_1',
)
IMAGES = {'.png', '.jpg', '.jpeg', '.webp', '.avif'}
FORBIDDEN = {'.pt', '.pth', '.safetensors', '.npz', '.npy', '.bin', '.gguf'}
PROFILE = ROOT / 'data/ranker/pretrained/store-fast-path-profile-attempts-1-2-audit-v1.tar.gz'
PROFILE_SHA = 'fbf1d342af67bc343430f09218e609c44201078dd00702101edff09eed4d9a9d'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return digest(path.read_bytes())


def relative(path):
    return str(path.relative_to(ROOT))


def exact_historical_source(logical, expected):
    current = ROOT / logical
    if current.is_file() and sha(current) == expected:
        return current.read_bytes(), {'kind': 'current exact checksum', 'path': relative(current)}
    if logical == 'apps/storefront/src/main.tsx':
        source = ROOT / 'data/ranker/pretrained/store-candidate-fast-v5-v10-attempt-1/source' / logical
        if sha(source) == expected:
            return source.read_bytes(), {'kind': 'existing historical source snapshot', 'path': relative(source)}
    if logical == 'experiments/ranker/pretrained/store_candidate_browser_probe_v2.mjs':
        source = BASE / 'store_candidate_browser_probe_v2_historical_fd0614/store_candidate_browser_probe_v2.mjs'
        if sha(source) == expected:
            return source.read_bytes(), {'kind': 'existing historical source snapshot', 'path': relative(source)}
    if logical == 'experiments/ranker/pretrained/test_store_candidate_browser_probe_v2.mjs':
        text = current.read_text()
        start = text.index("test('late body capture is rejected")
        end = text.index("test('both real byte paths agree", start)
        recovered = (text[:start] + text[end:]).encode()
        if digest(recovered) == expected:
            return recovered, {'kind': 'checksum-exact historical reconstruction', 'path': relative(current),
                'currentSha256': sha(current), 'recipe': 'Remove complete later-added late-body-capture regression test block, from its test declaration up to next both-real-byte-paths test. Original recorded SHA256 matches exactly.',
                'historicalOriginalWasNotOverwritten': True}
    raise ValueError(f'Exact historical source bytes unavailable: {logical} {expected}')


def main():
    for name in ('measurement-evidence.tar.xz', 'inventory.json', 'receipt.json', 'README.md'):
        if (OUT / name).exists():
            raise FileExistsError('Package outputs must be new and unused')
    inventory = []
    includes = {}
    groups = []
    references = defaultdict(list)
    for name in NAMES:
        directory = BASE / name
        plan = json.loads((directory / 'plan.json').read_text())
        report = json.loads((directory / 'report.json').read_text())
        for logical, expected in plan['sourceHashes'].items():
            references[logical, expected].append(relative(directory / 'plan.json'))
        chosen = {'pair-1-generic.png', 'pair-1-local.png'} if name.startswith('store_paired') else {
            'query-1.png' if 'collection' in name else 'attempt-1.png'}
        files = sorted(path for path in directory.rglob('*') if path.is_file())
        group = {'directory': relative(directory), 'originalFiles': len(files),
            'originalBytes': sum(path.stat().st_size for path in files),
            'representativeScreenshotPolicy': 'fixed first original ordinal; both arms for paired run; independent of timing or success',
            'representativeScreenshots': sorted(chosen), 'reportSummaryPreservedWithoutReclassification': report.get('summary')}
        for path in files:
            assert not path.is_symlink()
            included = path.suffix.lower() not in FORBIDDEN and (path.suffix.lower() not in IMAGES or path.name in chosen)
            entry = {'kind': 'browser-original', 'path': relative(path), 'bytes': path.stat().st_size,
                'sha256': sha(path), 'included': included,
                'inclusion': 'lossless original payload' if included else 'image retained locally; checksum inventoried'}
            if included:
                entry['archiveMember'] = relative(path)
                includes[relative(path)] = path
            inventory.append(entry)
        groups.append(group)
    assert sha(PROFILE) == PROFILE_SHA
    profile_member = 'profiles/store-fast-path-profile-attempts-1-2-audit-v1.tar.gz'
    includes[profile_member] = PROFILE
    with tarfile.open(PROFILE, 'r:gz') as source:
        profile_manifest = json.loads(source.extractfile('_archive-manifest.json').read())
        assert len(profile_manifest['archiveInventory']) == 96
        for original in profile_manifest['archiveInventory']:
            payload = source.extractfile(original['path']).read()
            assert digest(payload) == original['sha256'] and sha(ROOT / original['path']) == original['sha256']
            inventory.append({'kind': 'profile-original', **original, 'included': True,
                'archiveMember': profile_member, 'nestedArchiveMember': original['path'],
                'inclusion': 'lossless original payload inside unchanged prior archive'})
    source_versions = []
    for (logical, expected), reference_paths in sorted(references.items()):
        payload, recovery = exact_historical_source(logical, expected)
        assert digest(payload) == expected and Path(logical).suffix.lower() not in FORBIDDEN | IMAGES
        member = 'historical-sources/' + expected + '/' + logical
        includes[member] = payload
        source_versions.append({'logicalPath': logical, 'sha256': expected, 'bytes': len(payload),
            'referencedBy': reference_paths, 'archiveMember': member, 'recovery': recovery})
    metadata = {
        'version': 'store-fast-measurement-results.v1',
        'scope': 'five fixed browser runs, including partial old-rank navigation attempt1, plus both native profile attempts via unchanged prior archive',
        'originalsModified': False, 'imageSelectionPolicy': 'fixed first ordinal per run; first generic/local pair; no reencoding or result-based selection',
        'originalFileInventory': inventory, 'historicalSourceVersions': source_versions, 'groups': groups,
        'priorProfileArchive': {'path': relative(PROFILE), 'sha256': PROFILE_SHA, 'bytes': PROFILE.stat().st_size,
            'archiveMember': profile_member, 'originalFiles': 96, 'failureAndTimedRawReceiptsPreserved': True},
        'limits': ['Excluded screenshots remain at original paths and are all SHA256-inventoried.',
            'No weights, feature tensors, NPZ or large product image assets are included.',
            'Source hash inventories and referenced manifests retain their original asset pins; public asset binaries are not duplicated.',
            'Raw timing definitions, denominators and partial/failure receipts are preserved; this package does not merge or rescore workloads.',
            'Historical test source reconstruction is only accepted when its exact original recorded SHA256 matches.'],
        'builderSha256': sha(Path(__file__)),
    }
    manifest_payload = json.dumps(metadata, indent=2).encode() + b'\n'
    archive_path = OUT / 'measurement-evidence.tar.xz'
    with tarfile.open(archive_path, 'x:xz', preset=3) as target:
        entry = tarfile.TarInfo('_preservation-inventory.json'); entry.size = len(manifest_payload)
        target.addfile(entry, io.BytesIO(manifest_payload))
        for name, source in includes.items():
            if isinstance(source, Path):
                target.add(source, arcname=name, recursive=False)
            else:
                entry = tarfile.TarInfo(name); entry.size = len(source)
                target.addfile(entry, io.BytesIO(source))
    with tarfile.open(archive_path, 'r:xz') as source:
        assert source.getnames() == ['_preservation-inventory.json'] + list(includes)
        assert source.extractfile('_preservation-inventory.json').read() == manifest_payload
        for name, original in includes.items():
            payload = original.read_bytes() if isinstance(original, Path) else original
            assert source.extractfile(name).read() == payload
    # Rehash every original after packaging, including images excluded from tar.
    for entry in inventory:
        assert sha(ROOT / entry['path']) == entry['sha256']
    assert sha(PROFILE) == PROFILE_SHA
    (OUT / 'inventory.json').write_bytes(manifest_payload)
    receipt = {'version': metadata['version'], 'status': 'completed-and-all-bytes-verified',
        'archive': relative(archive_path), 'archiveBytes': archive_path.stat().st_size, 'archiveSha256': sha(archive_path),
        'allOriginalFilesInventoried': len(inventory), 'originalBrowserFiles': sum(v['kind'] == 'browser-original' for v in inventory),
        'originalProfileFiles': sum(v['kind'] == 'profile-original' for v in inventory),
        'originalImagesInventoried': sum(Path(v['path']).suffix.lower() in IMAGES for v in inventory),
        'representativeImagesIncluded': sum(Path(v['path']).suffix.lower() in IMAGES and v['included'] for v in inventory),
        'excludedOriginalsStillLocal': sum(not v['included'] for v in inventory),
        'historicalSourceVersions': len(source_versions), 'everyIncludedByteMatchesOriginal': True,
        'everyOriginalRehashedUnchangedAfterPackaging': True, 'priorProfileArchiveUnchanged': True,
        'tensorsAndWeightsIncluded': 0, 'inventorySha256': digest(manifest_payload),
        'builderSha256': metadata['builderSha256'], 'countsByArchiveMemberSuffix': dict(Counter(Path(name).suffix for name in includes)),
        'nativeBrowserOrGitExecution': False}
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (OUT / 'README.md').write_text(f'''This lossless evidence package preserves all raw JSON, plans and referenced historical source versions for the five requested browser runs. The unchanged prior profile archive carries both native profile attempts, including the excluded-warmup failure and all40 timed arms. Existing originals remain untouched.

`measurement-evidence.tar.xz` is {receipt['archiveBytes']:,} bytes, SHA256 `{receipt['archiveSha256']}`. Every included byte was checked against its original. All {len(inventory)} original browser/profile files were SHA256-inventoried and rehashed unchanged after packaging. `inventory.json` also maps every historical source version to its exact recorded checksum and recovery source. `receipt.json` records verification and archive pins.

Six representative screenshots are included: first navigation image from each navigation run, first collection image, and first generic/local pair. This fixed ordinal policy does not select by performance or success. All {receipt['originalImagesInventoried']} original screenshots, including the {receipt['excludedOriginalsStillLocal']} excluded from the archive, remain local with hashes and paths recorded. Images are preserved as original bytes, never reencoded. No weights, feature tensors, NPZ or product asset binaries are included.

The original partial v3 navigation attempt1 remains separate from v3 attempt2, v4 navigation, v4 collection and the loaded-collection20-pair incremental comparison. The package does not merge timing populations or replace failure/outlier records. Browser probe endpoints retain their original instrumentation limits; no physical paint claim is introduced.

There are {len(source_versions)} distinct referenced historical source versions. The old main entry and navigation script come from already preserved snapshots. The historical navigation test was reconstructed by removing the later-added regression-test block, then accepted only because its complete bytes exactly match the original recorded SHA256. Current live files were not edited. The detailed recipe and current-source checksum are in the inventory.

Use `tar -tJf measurement-evidence.tar.xz` to list files. The internal `_preservation-inventory.json` matches the loose inventory exactly. The nested profile archive remains byte-identical to its previous archive; its own embedded inventory includes all96 original profile files. The build helper is CPU-only, refuses existing package outputs and imports no model or browser runtime.
''')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
