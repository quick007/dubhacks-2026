"""Lossless, compact browser evidence archive. Writes only beside this file.

Build once: python build_archive.py --build
Read-only verification: python build_archive.py --verify
No browser, model, GPU, service, Git, or original-file mutations.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = HERE.parents[4]
RUNS = (
    ('store_candidate_browser_probe_results_v1_attempt_1', 'navigation-v1-attempt1-failed-protocol'),
    ('store_candidate_browser_probe_results_v1_attempt_2', 'navigation-v1-attempt2-baseline'),
    ('store_candidate_collection_probe_results_v1_attempt_1', 'collection-v1-attempt1-failed-capture'),
    ('store_candidate_collection_probe_results_v1_attempt_2', 'collection-v1-attempt2-fixed-capture'),
    ('store_candidate_browser_probe_results_v2_responsive_attempt_1', 'navigation-v2-responsive-attempt1'),
)
SELECTED = {
    ('store_candidate_browser_probe_results_v1_attempt_2', 'attempt-1.png'),
    ('store_candidate_browser_probe_results_v2_responsive_attempt_1', 'attempt-1.png'),
    ('store_candidate_collection_probe_results_v1_attempt_2', 'query-1.png'),
}
TEXT_SUFFIXES = {'.json', '.jsonl', '.txt', '.md', '.mjs', '.js', '.ts', '.tsx', '.py'}

def sha(value): return hashlib.sha256(value).hexdigest()
def json_bytes(value): return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
def rel(path): return str(path.relative_to(ROOT))
def file_row(path):
    if path.is_symlink(): raise ValueError('Do not follow evidence symlinks: ' + str(path))
    data = path.read_bytes()
    return {'path': rel(path), 'bytes': len(data), 'sha256': sha(data)}
def write_new(name, value):
    path = HERE / name; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle: handle.write(value)

def packed_members(members):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', filename='', mtime=0, compresslevel=9) as compressed:
        with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for name, data in sorted(members.items()):
                header = tarfile.TarInfo(name); header.size = len(data); header.mode = 0o644; header.mtime = 0
                tar.addfile(header, io.BytesIO(data))
    return buffer.getvalue()

def replace_archive_file(name, data):
    temporary = HERE / (name + '.new')
    with temporary.open('xb') as handle: handle.write(data)
    temporary.replace(HERE / name)

def refresh_checksums():
    value = ''.join(sha(p.read_bytes()) + '  ' + str(p.relative_to(HERE)) + '\n'
        for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'CHECKSUMS.sha256' and '__pycache__' not in p.parts)
    replace_archive_file('CHECKSUMS.sha256', value.encode())

def enrich_proven_sources():
    """Add only independently prepared blobs that match an original plan SHA.
    Originals/v1 are read only; all replacement files are inside this v2 archive.
    """
    verify(check_checksums=False)
    proof = json.loads((HERE / 'source-recovery.json').read_bytes())
    manifest = json.loads((HERE / 'manifest.json').read_bytes())
    with tarfile.open(HERE / 'artifacts-text.tar.gz', 'r:gz') as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    resolved = 0
    for recovery in proof['blobs']:
        data = (HERE / recovery['recoveredPath']).read_bytes()
        if sha(data) != recovery['expectedSha256']: raise AssertionError('Recovery candidate does not match historical expected SHA')
        member = 'verified-source/' + recovery['expectedSha256'] + '/ProductPhoto.tsx'
        for entry in manifest['sourceAvailability']:
            if entry['expectedSha256'] == recovery['expectedSha256'] and entry['archiveMember'] is None:
                entry.update(status='exact-recovered-bytes-match-original-plan-sha256', archiveMember=member,
                    originPath=str((HERE / recovery['recoveredPath']).relative_to(ROOT)), recoveryProof='source-recovery.json')
                resolved += 1
        members[member] = data
    archive = packed_members(members)
    manifest['totals'].update(textTarMembersIncludingVerifiedSources=len(members),
        textTarUncompressedBytes=sum(map(len, members.values())), textTarCompressedBytes=len(archive))
    manifest['artifacts']['artifacts-text.tar.gz'] = {'bytes': len(archive), 'sha256': sha(archive)}
    proof_paths = ['source-recovery.json'] + [r['recoveredPath'] for r in proof['blobs']] + proof.get('inputSnapshots', [])
    for name in proof_paths: manifest['artifacts'][name] = file_row(HERE / name)
    manifest['tarMembers'] = [{'name': n, 'bytes': len(d), 'sha256': sha(d)} for n, d in sorted(members.items())]
    manifest['sourceRecovery'] = {'proof': 'source-recovery.json', 'resolvedPlanReferences': resolved,
        'recoveredBlobs': len(proof['blobs']), 'method': 'Exact original plan SHA match, with archived Git blob or frozen-input single-import transformation provenance',
        'transitiveSourceGraphCompletenessClaim': False}
    replace_archive_file('artifacts-text.tar.gz', archive)
    replace_archive_file('manifest.json', json_bytes(manifest))
    replace_archive_file('verification.json', json_bytes(verify(check_checksums=False)))
    refresh_checksums()
    print(json.dumps(verify(), indent=2))

def original_inventory():
    runs = []
    for name, label in RUNS:
        folder = BASE / name
        if not folder.is_dir(): raise FileNotFoundError(folder)
        rows = []
        for path in sorted(folder.rglob('*')):
            if path.is_symlink(): raise ValueError('Evidence contains a symlink')
            if not path.is_file(): continue
            row = {**file_row(path), 'relativePath': str(path.relative_to(folder))}
            if path.suffix.lower() == '.png':
                row['disposition'] = 'representative-exact-png' if (name, row['relativePath']) in SELECTED else 'excluded-original-screenshot-retained-locally'
                if row['disposition'] == 'representative-exact-png': row['representativePath'] = 'screenshots/' + name + '/' + row['relativePath']
            elif path.suffix in TEXT_SUFFIXES:
                path.read_bytes().decode('utf8')
                row.update(disposition='lossless-text-tar-member', archiveMember=name + '/' + row['relativePath'])
            else: raise ValueError('Unclassified original evidence; preserve and declare explicitly: ' + str(path))
            rows.append(row)
        runs.append({'runId': name, 'label': label, 'originalDirectory': rel(folder), 'files': rows})
    v1 = BASE / 'store_candidate_browser_results/v1'
    return {'version': 'browser-evidence-inventory.v2', 'runs': runs,
        'preexistingV1Archive': [file_row(p) for p in sorted(v1.rglob('*')) if p.is_file()]}

def build():
    if (HERE / 'manifest.json').exists() or (HERE / 'artifacts-text.tar.gz').exists():
        raise FileExistsError('Preserve completed/partial v2 archive; verify it instead')
    inventory = original_inventory()
    members = {}
    for run in inventory['runs']:
        for row in run['files']:
            data = (ROOT / row['path']).read_bytes()
            if 'archiveMember' in row: members[row['archiveMember']] = data
            if 'representativePath' in row: write_new(row['representativePath'], data)
    # Include only historical source bytes whose SHA can actually be verified.
    # This does not assert closure over imports/dependencies or unseen sources.
    source_availability = []
    for run in inventory['runs']:
        plan = json.loads((BASE / run['runId'] / 'plan.json').read_bytes())
        existing = [r for r in run['files'] if r['relativePath'].startswith('source/')]
        for reference, expected in plan.get('sourceHashes', {}).items():
            entry = {'runId': run['runId'], 'planReference': reference, 'expectedSha256': expected}
            old = next((r for r in existing if r['sha256'] == expected), None)
            if old:
                entry.update(status='exact-bytes-already-in-original-run', archiveMember=old['archiveMember'], originPath=old['path'])
            else:
                path = ROOT / reference
                if not path.resolve().is_relative_to(ROOT): raise ValueError('Source reference escapes repository')
                if path.is_file() and not path.is_symlink() and sha(path.read_bytes()) == expected:
                    member = 'verified-source/' + expected + '/' + path.name
                    members[member] = path.read_bytes()
                    entry.update(status='exact-current-bytes-match-original-plan-sha256', archiveMember=member, originPath=reference)
                else: entry.update(status='exact-historical-bytes-not-available', archiveMember=None)
            source_availability.append(entry)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', filename='', mtime=0, compresslevel=9) as compressed:
        with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for name, data in sorted(members.items()):
                header = tarfile.TarInfo(name); header.size = len(data); header.mode = 0o644; header.mtime = 0
                tar.addfile(header, io.BytesIO(data))
    write_new('artifacts-text.tar.gz', buffer.getvalue())
    write_new('inventory.json', json_bytes(inventory))
    original_rows = [r for run in inventory['runs'] for r in run['files']]
    screenshots = [r for r in original_rows if r['disposition'] == 'representative-exact-png']
    excluded = [r for r in original_rows if r['disposition'] == 'excluded-original-screenshot-retained-locally']
    manifest = {'version': 'store-candidate-browser-results.v2',
        'scope': 'Five original browser runs, including both failed attempts. Exact all JSON/text/original saved-source bytes retained in gzip tar. Three exact representative PNGs included; every omitted screenshot has original path/bytes/SHA in inventory. No weights/features.',
        'preservation': 'All original directories and preexisting v1 archive remain untouched at every path; this archive is an additional copy.',
        'sourceScope': 'Plan-referenced exact source bytes included when available by SHA. Source availability is explicit; no complete transitive source graph, environment closure, or reproduction claim.',
        'sourceAvailability': source_availability,
        'originalRuns': [{'runId': r['runId'], 'label': r['label'], 'originalDirectory': r['originalDirectory']} for r in inventory['runs']],
        'totals': {'originalFiles': len(original_rows), 'originalBytes': sum(r['bytes'] for r in original_rows),
            'originalTextSourceFiles': sum('archiveMember' in r for r in original_rows),
            'originalTextSourceBytes': sum(r['bytes'] for r in original_rows if 'archiveMember' in r),
            'textTarMembersIncludingVerifiedSources': len(members), 'textTarUncompressedBytes': sum(map(len, members.values())),
            'textTarCompressedBytes': len(buffer.getvalue()), 'representativeScreenshots': len(screenshots),
            'representativeScreenshotBytes': sum(r['bytes'] for r in screenshots),
            'excludedScreenshots': len(excluded), 'excludedScreenshotBytes': sum(r['bytes'] for r in excluded)},
        'artifacts': {'artifacts-text.tar.gz': {'bytes': len(buffer.getvalue()), 'sha256': sha(buffer.getvalue())},
            'inventory.json': file_row(HERE / 'inventory.json'),
            **{r['representativePath']: {'bytes': r['bytes'], 'sha256': r['sha256']} for r in screenshots}},
        'tarMembers': [{'name': n, 'bytes': len(d), 'sha256': sha(d)} for n, d in sorted(members.items())],
        'interpretation': 'Original reports/audits retain their own statuses and limits. Compression does not repair missing capture/protocol evidence or establish physical paint, unseen generalization, or speedup.'}
    write_new('manifest.json', json_bytes(manifest))
    receipt = verify(check_checksums=False)
    write_new('verification.json', json_bytes(receipt))
    checksums = ''.join(sha(p.read_bytes()) + '  ' + str(p.relative_to(HERE)) + '\n'
        for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'CHECKSUMS.sha256' and '__pycache__' not in p.parts)
    write_new('CHECKSUMS.sha256', checksums.encode())
    print(json.dumps(verify(), indent=2))

def verify(*, check_checksums=True):
    manifest = json.loads((HERE / 'manifest.json').read_bytes())
    inventory = json.loads((HERE / 'inventory.json').read_bytes())
    if original_inventory() != inventory: raise AssertionError('An original browser run or preexisting v1 archive changed')
    for name, expected in manifest['artifacts'].items():
        data = (HERE / name).read_bytes()
        if len(data) != expected['bytes'] or sha(data) != expected['sha256']: raise AssertionError('Archive artifact checksum differs: ' + name)
    with tarfile.open(HERE / 'artifacts-text.tar.gz', 'r:gz') as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    if set(members) != {r['name'] for r in manifest['tarMembers']}: raise AssertionError('Archive member inventory differs')
    for row in manifest['tarMembers']:
        data = members[row['name']]
        if len(data) != row['bytes'] or sha(data) != row['sha256']: raise AssertionError('Decompressed checksum differs')
    for run in inventory['runs']:
        for row in run['files']:
            original = (ROOT / row['path']).read_bytes()
            if 'archiveMember' in row and members[row['archiveMember']] != original: raise AssertionError('Exact text/source byte equality failed')
            if 'representativePath' in row and (HERE / row['representativePath']).read_bytes() != original: raise AssertionError('Representative PNG bytes differ')
    for row in manifest['sourceAvailability']:
        if row['archiveMember'] is not None and sha(members[row['archiveMember']]) != row['expectedSha256']: raise AssertionError('Source plan SHA differs')
    if manifest.get('sourceRecovery'):
        proof = json.loads((HERE / manifest['sourceRecovery']['proof']).read_bytes())
        for row in proof['blobs']:
            recovered = (HERE / row['recoveredPath']).read_bytes()
            if sha(recovered) != row['expectedSha256']: raise AssertionError('Recovered source hash differs')
            if row['method'] == 'single-import-byte-substitution':
                original = (HERE / row['inputSnapshot']).read_bytes()
                old, new = row['replaceFrom'].encode(), row['replaceTo'].encode()
                if sha(original) != row['inputSha256'] or original.count(old) != 1 or original.replace(old, new, 1) != recovered:
                    raise AssertionError('Recovered optimized source does not equal declared frozen-input transformation')
    if check_checksums:
        for line in (HERE / 'CHECKSUMS.sha256').read_text().splitlines():
            expected, name = line.split('  ', 1)
            if sha((HERE / name).read_bytes()) != expected: raise AssertionError('Deliverable checksum differs: ' + name)
    return {'status': 'verified', 'exactOriginalTextSourceBytesAfterDecompression': True,
        'representativePngBytesExact': True, 'allOriginalFilesStillPresentAtSamePathsAndHashes': True,
        'preexistingV1ArchiveUnchanged': True, 'tarMemberChecksumsVerified': len(members),
        'originalFilesVerified': sum(len(r['files']) for r in inventory['runs']),
        'manifestSha256': sha((HERE / 'manifest.json').read_bytes()),
        'archiveSha256': sha((HERE / 'artifacts-text.tar.gz').read_bytes()),
        'sourceGraphCompletenessClaim': False, 'totals': manifest['totals']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group(required=True); mode.add_argument('--build', action='store_true'); mode.add_argument('--verify', action='store_true'); mode.add_argument('--enrich-proven-sources', action='store_true')
    args = parser.parse_args()
    if args.build: build()
    elif args.enrich_proven_sources: enrich_proven_sources()
    else: print(json.dumps(verify(), indent=2))
