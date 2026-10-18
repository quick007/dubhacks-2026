"""Lossless small evidence archive; never copies tensors or changes originals."""
import hashlib
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
RUN = Path('data/ranker/pretrained/actual-store-function-v5-attempt-1')
DEST = Path('experiments/ranker/pretrained/store_actual_function_results/v5')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    original = json.loads((HERE / 'ORIGINAL_RUN_FILES_SHA256.json').read_text())['files']
    current = {str(p.relative_to(RUN)): sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()}
    if current != original:
        raise ValueError('Original310-file inventory changed')
    DEST.mkdir(parents=True, exist_ok=False)
    entries = []; excluded = []

    def copy(source, relative, kind):
        if source.suffix not in ('.json', '.py', '.mjs', '.md'):
            raise ValueError('Only reviewed text/source/JSON permitted: ' + str(source))
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        checksum = sha(source)
        if sha(target) != checksum:
            raise ValueError('Copy mismatch: ' + str(source))
        entries.append({'path': str(relative), 'originalPath': str(source),
                        'sha256': checksum, 'bytes': target.stat().st_size, 'kind': kind})

    for relative, checksum in original.items():
        source = RUN / relative
        if source.suffix == '.pt':
            excluded.append({'originalPath': str(source), 'sha256': checksum,
                             'bytes': source.stat().st_size, 'reason': 'local checkpoint or feature tensor; deliberately excluded'})
        else:
            copy(source, Path('raw_native_attempt_1') / relative, 'immutable original native evidence')
    for source in sorted(HERE.iterdir()):
        if source.is_file():
            copy(source, Path('independent_audit') / source.name, 'independent diagnostic correction or audit')
    base = HERE.parent
    for source in sorted((base / 'store_function_v5').rglob('*')):
        if source.is_file():
            copy(source, Path('frozen_dataset') / source.relative_to(base / 'store_function_v5'), 'frozen TRAIN/evaluation dataset; labels offline only')
    for relative in ('store_function_v5_prepare_v3/preflight.json', 'store_function_v5_eval_seed_v1.json',
                     'store_function_adaptation_v5.py', 'test_store_function_v5.py'):
        copy(base / relative, Path('support') / relative, 'preflight, independent seed, or future corrected source')
    after = {str(p.relative_to(RUN)): sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()}
    if after != original:
        raise ValueError('Original inventory changed during package')
    manifest = {'version': 'actual-function-v5-lossless-small-evidence.v1',
                'originalRun': str(RUN), 'originalFilesUnchanged': len(original),
                'originalNativeTextSourceJsonFiles': sum(r['kind'] == 'immutable original native evidence' for r in entries),
                'files': entries, 'excludedLocalTensorFiles': excluded,
                'totalCopiedFiles': len(entries), 'totalCopiedBytes': sum(e['bytes'] for e in entries),
                'scope': 'Original text/source/JSON lossless copies plus corrected independent receipts; all weights and feature tensors stay local, all hashes retained. No promotion or native rerun.'}
    dump(DEST / 'MANIFEST.json', manifest)
    (DEST / 'README.md').write_text(
        '# Actual function v5 evidence\n\n'
        'Raw native attempt1, frozen data, archived source, and independent corrected diagnostics. '
        'All originals remain unchanged. MANIFEST.json records byte hashes for every copied file '
        'and every omitted local checkpoint/feature tensor. No weights or feature tensor files are included.\n\n'
        'See independent_audit/INDEPENDENT_AUDIT.md for measured results and limitations. '
        'The raw miss-rank diagnostics are historically incorrect; independent_audit/CORRECTED_RESULTS.json '
        'corrects only that diagnostic. The original report and prediction hashes remain explicit.\n\n'
        'Support source is the future corrected driver/test; raw_native_attempt_1/source is the exact native source. '
        'The CPU replay script uses the original local run and tensors; this small archive deliberately cannot '
        'recreate tensor inference by itself. Evaluation labels are offline evidence, never serving inputs.\n')
    for entry in entries:
        if sha(DEST / entry['path']) != entry['sha256']:
            raise ValueError('Final package hash mismatch')
    if any(p.suffix in ('.pt', '.safetensors', '.npz', '.npy') for p in DEST.rglob('*')):
        raise ValueError('Forbidden tensor in package')
    print(json.dumps({'path':str(DEST),'manifestSha256':sha(DEST / 'MANIFEST.json'),
                      'files':len(entries),'copiedBytes':manifest['totalCopiedBytes'],'excludedTensors':len(excluded)}))


if __name__ == '__main__':
    main()
