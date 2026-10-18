"""Independent CPU receipt audit; never imports MLX or executes a model.

Run as a module. Feature archives are optional and loaded one row at a time.
Incomplete snapshots retain fixed denominators and never imply completed work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct

ROOT = Path(__file__).resolve().parent
MODES = ('frozen', 'failed-daypack', 'new-store')
EXPECTED_RUNNER = '7f63b9ac18f9142910694499b18c1a5427d4b2da13f0dc3be20860fb44b89968'
EXPECTED_PREFLIGHT = 'cd8ead6445ac7785c90dfcfb0c15f6999e35a2da4c54530a59106db87cbb1cbd'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(8 * 2**20):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def audit(out, *, check_features=False, decode=False, require_complete=False):
    from . import store_copy_data_v2 as data
    out = Path(out)
    receipt_path = out / 'results.json'
    receipt_bytes = receipt_path.read_bytes()
    run = json.loads(receipt_bytes)
    checks, issues = 0, []

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            issues.append(label)

    frozen = run['preflight']
    check(run['preflightSha256'] == EXPECTED_PREFLIGHT, 'exact reviewed preflight checksum')
    preflight_path = ROOT / 'STORE_COPY_FIT_V2_PREFLIGHT_V1.json'
    check(sha(preflight_path) == EXPECTED_PREFLIGHT and read(preflight_path) == frozen, 'embedded preflight equals original frozen file')
    check(frozen['sources']['store_copy_fit_v2.py'] == EXPECTED_RUNNER, 'exact reviewed runner checksum')
    for name, expected in frozen['sources'].items():
        check(sha(ROOT / name) == expected, 'current producer source: ' + name)
    manifest, groups, fields = data.load_dataset()
    check(frozen['payloadSha256'] == manifest['fileSha256'], 'frozen dataset payload membership')
    check(sha(data.DATA / 'manifest.json') == frozen['datasetManifestSha256'], 'dataset manifest checksum')
    check(run['schemaVersion'] == 'actual-store-copy-fit.v2', 'schema version')
    check(run['fixedArms'] == 24, 'fixed24 denominator')
    check(run['config']['updates'] == 320 and run['config']['sampledCapPerCase'] == 416,
          'fixed320 updates and416 sampled cap')
    check(len(run.get('frozenBaseParameterHashes', {})) == 927, '927 frozen base tensor receipts')
    initial, control = run['initialAdapter'], run['numericalProbeControl']
    check(initial['lora_b']['sha256'] == hashlib.sha256(bytes(4 * 4096 * 4)).hexdigest(), 'actual initial B bytes exactly zero')
    check(initial['lora_a'] == control['lora_a'], 'temporary control preserves original A')
    control_bytes = struct.pack('<f', 1e-5) * (4 * 4096)
    check(control['lora_b']['sha256'] == hashlib.sha256(control_bytes).hexdigest(), 'actual fixed probe B bytes exactly1e-5FP32')
    train_plan = {(r['caseId'], r['rowId']): r for r in frozen['splits']['train']['rows']}
    train_order = [(c['caseId'], r['rowId']) for c in groups['train']
                   for r in fields['train'] if r['caseId'] == c['caseId']]
    cases = {c['caseId']: c for c in groups['eval']}
    expected_arms = {(case_id, mode, 6) for case_id in cases for mode in MODES}
    check(len(run['arms']) == 24 and {(a['caseId'], a['mode'], a['width']) for a in run['arms']} == expected_arms,
          'full unique24-arm roster')
    check(frozen['gateWork']['evaluationRowsExtracted'] == 0, 'no evaluation feature extraction declared')
    extracted = run['extraction']
    check(len(extracted) <= 320, 'extraction maximum320')
    check([(r['caseId'], r['rowId']) for r in extracted] == train_order[:len(extracted)],
          'TRAIN-only extraction and complete expected row order')
    gate_adapter = None
    artifact_bytes = 0
    for index, entry in enumerate(extracted):
        label = f'gate{index:04d}'
        plan = train_plan[(entry['caseId'], entry['rowId'])]
        gate_path = out / 'gates' / f'{index:04d}.json'
        gate, metadata = read(gate_path), read(out / entry['metadata'])
        check(entry['index'] == index and entry['gateExact'] is True and entry['trueZeroBRestored'] is True,
              label + ' index/exact/reset')
        check(sha(gate_path) == entry['gateSha256'], label + ' gate checksum')
        for key in ('archive', 'metadata'):
            path = out / entry[key]
            artifact_bytes += path.stat().st_size
            check(sha(path) == entry[key + 'Sha256'], label + ' ' + key + ' checksum')
        check(gate['sourceSha256'] == frozen['sources']['semantic_copy_gradient_features.py'], label + ' gate source')
        check(gate['status'] == 'exact-gate-passed' and gate['nativeUpdates'] == 0 and gate['nonzeroB'] is True,
              label + ' exact zero-update numerical probe')
        check(gate['adapterPreserved'] is True and gate['adapterBefore'] == gate['adapterAfter'], label + ' adapter preserved')
        if gate_adapter is None:
            gate_adapter = gate['adapterBefore']
        check(gate['adapterBefore'] == gate_adapter, label + ' fixed gate adapter')
        check(gate['adapterBefore']['lora_b']['sha256'] == hashlib.sha256(
              b'mlx.core.float32' + str((4, 4096)).encode() + control_bytes).hexdigest(), label + ' actual nonzero probe B tensor bytes')
        check(entry['fixedNonzeroBControl'] == run['numericalProbeControl'], label + ' control receipt')
        check(gate['sourceStatePreserved'] is True and gate['sourceStateHashBefore'] == gate['sourceStateHashAfter'] == entry['sourceStateHash'],
              label + ' pristine source cache')
        check(gate['arms']['direct'] == gate['arms']['auxiliaryNative'] == gate['arms']['frozenFeatures'],
              label + ' actual loss/gradient receipt equality')
        check(set(gate['comparisons']) == {'nativeVsAuxiliaryNative', 'auxiliaryNativeVsFrozenFeatures', 'nativeVsFrozenFeatures'},
              label + ' three comparisons')
        for comparison in gate['comparisons'].values():
            check(comparison['exact'] is True, label + ' exact comparison')
            for component in (comparison['loss'], *comparison['gradients'].values()):
                check(all(component[k] is True for k in ('shapeExact', 'dtypeExact', 'arrayExact'))
                      and component['maxAbsDifference'] == 0, label + ' zero numerical difference')
        for key in ('modelInputSha256', 'sourceTokenSha256', 'promptSha256', 'predictionStartInBranch'):
            check(gate['contract'][key] == plan[key], label + ' token contract ' + key)
        contract = gate['featureContract']
        check(contract['suffixTokens'] == plan['suffixTokens'] and contract['targetTokens'] == plan['targetTokens'], label + ' full suffix shape')
        check(entry['featureTensors'] == gate['featureTensors'], label + ' native tensor receipts')
        check(entry['hostFeatureBytes'] == plan['projectedHostFeatureBytes'], label + ' complete feature bytes')
        check(metadata['contract'] == gate['contract'] and metadata['caseId'] == entry['caseId']
              and metadata['rowId'] == entry['rowId'] and metadata['gradientFeatureGatePassed'] is True,
              label + ' feature metadata binding')
        check(len(metadata['targets']) == plan['targetTokens'] and metadata['targets'][-1] == gate['contract']['stopTokenId'],
              label + ' complete EOS supervision')
        check(gate['memory']['mlxPeakBytes'] <= 7 * 2**30, label + ' declared native memory guard')
        if check_features:
            import numpy as np
            # The archive is closed and arrays released before the next row.
            with np.load(out / entry['archive'], allow_pickle=False) as archive:
                check(set(archive.files) == {'x', 'h', 'y'}, label + ' archive tensor membership')
                total = 0
                for name in ('x', 'h', 'y'):
                    value = archive[name]
                    tensor = entry['featureTensors'][name]
                    total += value.nbytes
                    check(value.dtype == np.float32 and bool(np.isfinite(value).all()), label + ' finite exact host ' + name)
                    check(list(value.shape) == tensor['shape'], label + ' original shape ' + name)
                    checksum = hashlib.sha256(metadata['dtypes'][name].encode() + str(tuple(value.shape)).encode() + value.tobytes()).hexdigest()
                    check(checksum == tensor['sha256'] and metadata['dtypes'][name] == tensor['dtype'], label + ' original tensor bytes ' + name)
                    del value
                check(total == entry['hostFeatureBytes'], label + ' archive byte sum')
    check(artifact_bytes == run['featureArtifactBytes'], 'reported cumulative feature artifact bytes')
    check(artifact_bytes <= 5 * 2**30, 'five-GiB artifact budget')
    steps = run['steps']
    check(len(steps) <= 320 and run['optimizerUpdateCalls'] <= 320 and run['nativeUpdates'] <= 320, 'bounded actual optimizer counts')
    completed_steps = [s for s in steps if s['status'] == 'completed']
    check(len(completed_steps) <= run['nativeUpdates'] <= run['optimizerUpdateCalls'] <= len(steps), 'optimizer progress receipt ordering')
    if steps:
        check(len(extracted) == 320 and run['controls']['fitStartedFromExactFrozenBaseZeroB'] is True,
              'all320 gates and zero-B start precede optimizer')
        order = list(range(320)); random.Random(19079).shuffle(order)
        check(run['permutation'] == order, 'exact seeded permutation')
        for step, row in enumerate(steps):
            check(row['step'] == step and row['featureIndex'] == order[step], f'update{step} sequence')
            original = extracted[row['featureIndex']]
            check((row['caseId'], row['rowId'], row['kind']) == (original['caseId'], original['rowId'], original['kind']), f'update{step} TRAIN identity')
            check(math.isfinite(row['loss']) and math.isfinite(row['gradientL2']), f'update{step} finite loss/gradient')
            if row['status'] == 'completed':
                check(row['updateEvaluated'] is True and row['ms'] > 0, f'update{step} evaluated')
    if 'checkpoint' in run:
        checkpoint = out / 'adapter.safetensors'
        record = run['checkpoint']
        check(sha(checkpoint) == record['sha256'] and record['exact'] is True, 'checkpoint checksum')
        check(record['parametersBeforeReload'] == record['parametersAfterReload']
              and record['lossBeforeReload'] == record['lossAfterReload'], 'exact checkpoint reload producer receipts')
        from safetensors.numpy import load_file
        import numpy as np
        parameters = load_file(checkpoint)
        check(set(parameters) == {'lora_a', 'lora_b'}, 'checkpoint only A/B')
        for key, shape in (('lora_a', (12288, 4)), ('lora_b', (4, 4096))):
            array = parameters[key]
            expected = record['parametersAfterReload'][key]
            check(array.shape == shape and array.dtype == np.float32 and bool(np.isfinite(array).all()), 'checkpoint shape/dtype/finite ' + key)
            check(hashlib.sha256(array.tobytes()).hexdigest() == expected['sha256'], 'actual checkpoint parameter bytes ' + key)
        del array, parameters
        check(all(v > 0 for v in run['parameterAbsoluteChanges'].values()), 'both A and B actually changed')
    if 'learnedOriginalArchivedNativeParityExact' in run['controls']:
        learned = read(out / 'learned-probe.json')
        check(learned['status'] == 'exact-gate-passed' and run['controls']['learnedOriginalArchivedNativeParityExact'] is True,
              'learned native parity producer evidence')
        check(learned['featureTensors'] == extracted[0]['featureTensors'], 'learned original archive tensor receipts')
        check(learned['arms']['direct'] == learned['arms']['auxiliaryNative'] == learned['arms']['frozenFeatures'],
              'learned direct/auxiliary/feature loss and gradient receipt equality')
        check(all(c['exact'] is True for c in learned['comparisons'].values()), 'all learned strict comparisons exact')
        check(run['controls']['learnedOriginalArchiveParityScope'] == {'trainFeatureIndices': [0], 'notAll320LearnedReplays': True, 'all320PassedBeforeFit': True},
              'honest one-row learned replay scope')
    for mode in MODES:
        if mode not in run['controls']:
            continue
        control = run['controls'][mode]
        check(read(out / ('controls-' + mode) / 'results.json') == control, mode + ' saved complete control report')
        check(control['status'] == 'exact-controls-passed' and control['width'] == 6, mode + ' native control status')
        control_arms = control['arms']
        check([a['label'] for a in control_arms] == ['fork', 'independent-rebuild', 'fork-repeat'], mode + ' all3 control paths')
        check(all(a['rows'] == control_arms[0]['rows'] for a in control_arms), mode + ' raw token/EOS/distribution identity')
        for arm in control_arms:
            check(read(out / ('controls-' + mode) / (arm['label'] + '.json')) == arm, mode + ' saved control arm')
            check(arm['maximumRowsEmittingTogether'] == 6 and arm['sourceAfter'] == control['sourceSha256'], mode + ' native width/cache control')
    tokenizer = None
    if decode:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(frozen['modelFiles']['modelPath'], local_files_only=True, trust_remote_code=False)
    metric = {mode: {'fixedCases': 8, 'completed': 0, 'failed': 0, 'unattempted': 0, 'inProgress': 0,
                     'validFirstThree': 0, 'validFullSix': 0, 'acceptedFields': 0,
                     'acceptedByKind': {kind: 0 for kind in ('heading', 'title', 'badge', 'copy')},
                     'rejectionCategories': {},
                     'attemptFirstThreeMs': [], 'attemptFullSixMs': [], 'validFirstThreeMs': [], 'validFullSixMs': []} for mode in MODES}
    examples = []
    for arm in run['arms']:
        label = arm['caseId'] + '/' + arm['mode']
        counts = metric[arm['mode']]
        status = arm['status']
        check(status in ('completed-attempt', 'failed', 'running', 'not-attempted'), label + ' status')
        counts[{'completed-attempt': 'completed', 'failed': 'failed', 'running': 'inProgress', 'not-attempted': 'unattempted'}[status]] += 1
        if status != 'completed-attempt':
            continue
        case = cases[arm['caseId']]
        rows = arm['rawRows']
        common, jobs = data.layout(case['bound'])
        check(len(rows) == 20 and arm['width'] == 6 and arm['sampledCap'] == 416, label + ' complete20 width6 original caps')
        check(arm['sourceCacheBefore'] == arm['sourceCacheAfter'], label + ' pristine inference cache')
        for row, job in zip(rows, jobs, strict=True):
            check(all(row[k] == job[k] for k in ('rowId', 'key', 'fields', 'kind', 'maxTokens', 'messages')), label + ' serving job identity')
            check(not any(k in row for k in ('targetValue', 'teacherConstruction', 'teacherOnlySyntheticStop')), label + ' no teacher in inference')
            raw, tokens = row['rawSampledTokenIds'], row['tokenIds']
            check(all(type(t) is int and t >= 0 for t in raw + tokens) and len(raw) == row['arSteps'] <= job['maxTokens'], label + ' sampled token/cap receipt')
            if row['finishReason'] == 'stop':
                check(raw[-1] == row['stopTokenId'] and tokens == raw[:-1], label + ' stop token retained separately')
            elif row['finishReason'] == 'length':
                check(len(raw) == job['maxTokens'] and raw == tokens and row['stopTokenId'] is None, label + ' honest length failure')
            else:
                check(False, label + ' unrecognized terminal reason')
            if tokenizer is not None:
                check(tokenizer.decode(tokens) == row['text'], label + ' independent actual sampled text decode')
        for key, n, endpoint in (('firstThree', 3, 'firstThreeReceipt'), ('fullSix', 6, 'fullSixReceipt')):
            assembled = data.assemble(case['bound'], rows[:10] if n == 3 else rows, cards=n, require_native=True)
            check(assembled == arm[key], label + ' independent validator/payload reassembly ' + key)
            check(assembled['payloadSha256'] == arm[endpoint]['payloadSha256'] and assembled['payloadBytes'] == arm[endpoint]['payloadBytes'], label + ' actual serialized endpoint bytes ' + key)
            valid = assembled['validation']['valid']
            ms = arm[endpoint]['validatedAndSerializedAtMs']
            counts['attempt' + key[0].upper() + key[1:] + 'Ms'].append(ms)
            if valid:
                counts['valid' + key[0].upper() + key[1:]] += 1
                counts['valid' + key[0].upper() + key[1:] + 'Ms'].append(ms)
        counts['acceptedFields'] += arm['fullSix']['validation']['acceptedFields']
        for row, validation in zip(rows, arm['fullSix']['validation']['fields'], strict=True):
            counts['acceptedByKind'][row['kind']] += validation['accepted']
            for rejection in validation['rejections']:
                counts['rejectionCategories'][rejection] = counts['rejectionCategories'].get(rejection, 0) + 1
        if arm['caseId'] == next(iter(cases)):
            examples.append({'caseId': arm['caseId'], 'mode': arm['mode'],
                             'firstThreeValid': arm['firstThree']['validation']['valid'],
                             'fullSixValid': arm['fullSix']['validation']['valid'],
                             'rawFields': [{'field': r['fields'][0], 'text': r['text'], 'finishReason': r['finishReason']}
                                           for r in rows if r['fields'][0] in ('S1', 'T1', 'B1', 'C1', 'C2', 'C3')],
                             'firstThreeRejections': arm['firstThree']['validation']['rejections']})
        check(0 <= arm['metadataReadyAtMs'] <= arm['copyAdmittedAtMs'] <= arm['firstThreeReceipt']['validatedAndSerializedAtMs'] <= arm['fullSixReceipt']['validatedAndSerializedAtMs'], label + ' contiguous endpoint timing order')
        check(arm['actualOutputTokens'] == sum(len(r['tokenIds']) for r in rows)
              and arm['actualArSteps'] == sum(r['arSteps'] for r in rows), label + ' actual sampled totals')
    for mode, m in metric.items():
        producer = run['summary'][mode]
        for ours, theirs in (('completed', 'completed'), ('failed', 'failed'), ('unattempted', 'unattempted'), ('inProgress', 'inProgress'), ('validFirstThree', 'validThreeCardResults'), ('validFullSix', 'validSixCardResults')):
            check(m[ours] == producer[theirs], mode + ' independent summary ' + ours)
        check(sum(m[k] for k in ('completed', 'failed', 'unattempted', 'inProgress')) == 8, mode + ' fixed8 accounting')
        check(producer['validThreeCardRateFixedDenominator'] == m['validFirstThree'] / 8
              and producer['validSixCardRateFixedDenominator'] == m['validFullSix'] / 8, mode + ' fixed8 rates')
        for key in ('attemptFirstThreeMs', 'attemptFullSixMs', 'validFirstThreeMs', 'validFullSixMs'):
            m[key.replace('Ms', 'MedianMs')] = statistics.median(m[key]) if m[key] else None
    complete = run['status'] == 'completed'
    if require_complete:
        check(complete, 'required completed native run')
    if complete:
        check(len(extracted) == len(steps) == len(completed_steps) == run['nativeUpdates'] == run['optimizerUpdateCalls'] == 320, 'completed all320 gates/updates')
        check(all(a['status'] == 'completed-attempt' for a in run['arms']), 'completed all24 inference arms')
        for key in ('postExtractionFrozen927Unchanged', 'postFitFrozen927Unchanged', 'postReplayFrozen927Unchanged', 'postEvaluationFrozen927Unchanged', 'sourceFilesUnchanged', 'datasetManifestUnchanged', 'originalPriorUnchanged'):
            check(run['controls'].get(key) is True, 'final producer proof ' + key)
    return {'schemaVersion': 'actual-store-copy-fit-v2-independent-audit.v1', 'status': 'passed' if not issues else 'issues-found',
            'nativeRunStatus': run['status'], 'nativePhase': run['phase'], 'nativeRunCompleted': complete,
            'checks': checks, 'issues': issues, 'auditedReceiptSha256': hashlib.sha256(receipt_bytes).hexdigest(),
            'auditSourceSha256': sha(__file__), 'completedGates': len(extracted), 'completedUpdates': len(completed_steps),
            'optimizerUpdateCalls': run['optimizerUpdateCalls'], 'featureArtifactBytes': artifact_bytes,
            'featureArchiveContentsChecked': check_features, 'independentTokenDecodePerformed': decode,
            'summary': metric, 'examples': examples, 'wallMs': run['wallMs'], 'memory': run.get('memory'), 'blocker': run.get('blocker'),
            'limitations': ['CPU verifies producer native evidence and artifact bytes; does not rerun native tensor/cache/gradient work.',
                            'No new model quality, unseen generalization, browser paint, or full storefront latency claim.']}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--check-features', action='store_true')
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('Preserve earlier audit reports; choose a new report path')
    result = audit(args.out, check_features=args.check_features, decode=args.decode, require_complete=args.require_complete)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'nativeRunStatus', 'nativePhase', 'checks', 'issues', 'completedGates', 'completedUpdates')}))
    if result['issues']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
