"""Explicit TRAIN-ALL catalog demo head; CPU only, no heldout quality claim.

This adaptation is prepared after the frozen v1 heldout result. All48 public
product-color bindings now supply supervised targets. Runtime still predicts
word slots from Qwen features and public conditions, never teacher lookup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from . import store_parallel_text_data_v1 as data
from . import store_parallel_text_head_v1 as model
from .store_parallel_text_run_v1 import load_packet, runtime_rows, endpoint, sha, dump

VERSION = 'parallel-word-train-all-catalog-demo.v1'
CONFIG = {'steps': 160, 'seed': 7211, 'hiddenWidth': 128, 'learningRate': .004,
          'cpuThreads': 1, 'publicCatalogBindings': 48, 'heldoutBindings': 0,
          'vocabularyPolicy': 'frozen v1 vocabulary; no new teacher words',
          'selection': 'single predeclared final checkpoint; no optimization against heldout outcomes'}


def demo_labels(packet):
    """Offline only: every actual public binding is declared TRAIN here."""
    rows = []
    for index, record in enumerate(packet['featureRecords']):
        for style in data.STYLES:
            for kind in data.KINDS[1:]:
                rows.append({'rowId': f'train-all-binding-{index:02d}-{style}-{kind}',
                    'input': data.serving_row(kind, style, record['facts']),
                    'featureKeys': [record['featureKey']], 'targetText': data.teacher(kind, style, record['facts'])})
    # Existing public TRAIN section means provide the generic heading labels.
    # They are source-only supervised training rows, never runtime templates.
    rows.extend(r for r in packet['labels']['train'] if r['input']['kind'] == 'section')
    if len(rows) != 528:
        raise ValueError('Exactly432 card-field +96 section TRAIN rows required')
    model.encode_labels(rows, packet['vocabulary'])  # refuse new/OOV tokens
    return rows


def load_head(path):
    path = Path(path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['schemaVersion'] != VERSION or saved['scope'] != 'TRAIN-ALL public catalog demo; no heldout evaluation':
        raise ValueError('Explicit TRAIN-ALL demo checkpoint required')
    head = model.ParallelTextHead(saved['vocabulary'], width=saved['width'], seed=7211)
    head.load_state_dict(saved['state_dict'], strict=True); head.eval()
    return head, saved


def all_binding_replay(head, rows, feature_map):
    predicted = model.generate(head, model.inference_rows(rows), feature_map)
    strata = {kind: {'fields': 0, 'acceptedFields': 0, 'exactTeacherMatch': 0} for kind in data.KINDS}
    receipts = []
    for row, result in zip(rows, predicted['fields'], strict=True):
        group = strata[row['input']['kind']]
        group['fields'] += 1; group['acceptedFields'] += result['accepted']
        group['exactTeacherMatch'] += result['text'] == row['targetText']
        receipts.append({'rowId': row['rowId'], 'publicInput': row['input'], 'prediction': result,
                         'offlineTrainingReference': row['targetText']})
    return {'scope': 'TRAIN replay only; all48bindings and every3style are supervised, no heldout quality claim',
        'strata': strata, 'receipts': receipts, 'nativeCalls': 0, 'runtimeTemplateOrTeacherLookup': False}


def run_demo(source_run, out, *, steps=160, prior_training_run=None):
    source_run, out = Path(source_run), Path(out)
    if out.exists():
        raise FileExistsError('Unused output required; preserve all failed candidates')
    if type(steps) is not int or steps not in (160, 320):
        raise ValueError('Only predeclared160 or TRAIN-loss-justified320 steps allowed')
    justification = None
    if steps == 320:
        if prior_training_run is None:
            raise ValueError('320-step followup requires a preserved160-step TRAIN-only receipt')
        previous = json.loads((Path(prior_training_run) / 'results.json').read_text())
        if previous['status'] != 'completed-train-all-demo-not-promoted' or previous['config']['steps'] != 160 or previous['scope'] != 'TRAIN-ALL public catalog demo; no heldout evaluation':
            raise ValueError('Prior completed TRAIN-ALL160 receipt required')
        if not any(v['acceptedFields'] < v['fields'] for v in previous['trainReplayStrata'].values()):
            raise ValueError('No unresolved TRAIN fields justify more steps')
        justification = {'priorTrainingResultSha256': sha(Path(prior_training_run) / 'results.json'),
            'trigger': 'two failed supervised TRAIN bodies at fixed160, loss0.104264; no heldout checkpoint selection'}
    prior = json.loads((source_run / 'results.json').read_text())
    if prior['status'] != 'completed-candidate-not-promoted' or not all(prior['shutdownIntegrity'][k] for k in ('frozen927HashesUnchanged', 'copyCheckpointUnchanged')):
        raise ValueError('Preserved completed v1 with frozen integrity required before TRAIN-ALL adaptation')
    bundle_path = source_run / prior['featureBundle']['file']
    if sha(bundle_path) != prior['featureBundle']['sha256']:
        raise ValueError('Native feature archive checksum mismatch')
    packet = load_packet(); bundle = torch.load(bundle_path, map_location='cpu', weights_only=True)
    feature_map = model.validate_bundle(bundle, packet); labels = demo_labels(packet)
    out.mkdir(parents=True)
    config = {**CONFIG, 'steps': steps}
    report = {'version': VERSION, 'status': 'started-train-all-demo', 'config': config,
        'nativeExecution': False, 'sourceRun': str(source_run), 'sourceResultSha256': sha(source_run / 'results.json'),
        'featureBundleSha256': sha(bundle_path), 'sourceSha256': sha(Path(__file__)),
        'scope': 'TRAIN-ALL public catalog demo; no heldout evaluation',
        'priorHeldoutV1Unchanged': True, 'all48BindingsDeclaredTrain': True,
        'freshQueryBenchmarkMeasuredHere': False, 'humanConversionClaim': False,
        'followupTrainingOnlyJustification': justification}
    dump(out / 'preflight.json', report); dump(out / 'TRAIN-ALL-labels.json', labels)
    (out / 'source.py').write_text(Path(__file__).read_text())
    try:
        with model.bounded_cpu_threads(1):
            head = model.ParallelTextHead(packet['vocabulary'], width=128, seed=7211)
            features, conditions = model.serving_tensors(model.inference_rows(labels), feature_map)
            target = model.encode_labels(labels, packet['vocabulary'])
            optimizer = torch.optim.AdamW(head.parameters(), lr=.004, weight_decay=.0001)
            started = time.perf_counter(); losses = []
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head(features, conditions).flatten(0, 1), target.flatten(), ignore_index=-100)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite TRAIN-ALL loss')
                loss.backward(); optimizer.step()
                if step == 0 or step == steps - 1:
                    losses.append({'step': step + 1, 'loss': float(loss.detach())})
            report.update(fitSeconds=time.perf_counter() - started, lossSnapshots=losses)
            head.eval()
            torch.save({'schemaVersion': VERSION, 'state_dict': head.state_dict(), 'vocabulary': list(head.vocabulary),
                'width': head.width, 'config': config, 'scope': report['scope'],
                'featureBundleSha256': report['featureBundleSha256']}, out / 'head.pt')
            report['checkpoint'] = {'file': 'head.pt', 'sha256': sha(out / 'head.pt')}
            restored, _ = load_head(out / 'head.pt')
            replay = all_binding_replay(head, labels, feature_map)
            restored_replay = all_binding_replay(restored, labels, feature_map)
            ids = lambda value: [r['prediction']['slotIds'] for r in value['receipts']]
            if ids(replay) != ids(restored_replay):
                raise ValueError('TRAIN-ALL head safe reload slot parity failed')
            dump(out / 'TRAIN-ALL-replay.json', replay)
            report['trainReplayStrata'] = replay['strata']; report['reloadExactSlotParity'] = True
            public_contract = {'schemaVersion': 'parallel-word-public-runtime-features.v1',
                'records': packet['featureRecords'], 'featureBundle': str(bundle_path),
                'featureBundleSha256': sha(bundle_path), 'teacherOrLabelContent': False}
            dump(out / 'public-runtime-features.json', public_contract)
            dump(out / 'serving-manifest.json', {'version': VERSION, 'scope': report['scope'],
                'headFile': str(out / 'head.pt'), 'headSha256': report['checkpoint']['sha256'],
                'publicFeatureContractFile': str(out / 'public-runtime-features.json'),
                'publicFeatureContractSha256': sha(out / 'public-runtime-features.json'),
                'featureBundleFile': str(bundle_path), 'featureBundleSha256': sha(bundle_path),
                'featureTensorSha256': bundle['featureTensorSha256'],
                'modelRevision': bundle['provenance']['modelRevision'],
                'copyAdapterEnabledForFeatureExtraction': False,
                'nativeSourceRunResultSha256': report['sourceResultSha256'],
                'sourceSha256': report['sourceSha256'], 'runtimeTeacherLookup': False,
                'slotsPerField': 14, 'vocabularySize': len(head.vocabulary),
                'publicStyles': list(data.STYLES), 'all48BindingsSupervised': True,
                'heldoutQualityClaim': False, 'freshQueryTimingClaim': False})
            # These are offline CPU replays of previously chosen public cards.
            # They do not inherit or sum any prior native query latency.
            views = []
            for index, request in enumerate(prior['requests']):
                previous = json.loads((source_run / request['file']).read_text())
                variants = []
                for style in data.STYLES:
                    rows = runtime_rows(previous['cards'], style)
                    generated = model.generate(restored, model.inference_rows(rows), feature_map)
                    full = endpoint(previous['cards'], rows, generated, card_limit=6)
                    variants.append({'style': style, 'generated': generated, 'fullSix': full})
                receipt = {'archiveCaseId': request['caseId'], 'sourceRequestSha256': sha(source_run / request['file']),
                    'scope': 'CPU-only same-public-card replay; allbindings TRAIN, no fresh query encode/browser latency',
                    'variants': variants}
                dump(out / f'replay-{index:03d}.json', receipt)
                views.append({'caseId': request['caseId'], 'allThreeStylesFull20Pass': all(v['fullSix']['validation']['allFieldsAccepted'] for v in variants),
                    'full20PassByStyle': {v['style']: v['fullSix']['validation']['allFieldsAccepted'] for v in variants}})
            report['knownCardReplay'] = {'requests': len(views), 'stylesPerRequest': 3,
                'allThreeStylesFull20Passes': sum(v['allThreeStylesFull20Pass'] for v in views), 'views': views,
                'freshQwenQueryMeasured': False, 'scope': 'supervised catalog replay; no generalization or end-to-end speed claim'}
        report['status'] = 'completed-train-all-demo-not-promoted'
    except BaseException as error:
        report.update(status='failed-train-all-demo', error={'type': type(error).__name__, 'message': str(error)})
    dump(out / 'results.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-cpu', action='store_true', required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, choices=(160, 320), default=160)
    parser.add_argument('--prior-training-run', type=Path)
    args = parser.parse_args()
    with model.bounded_cpu_threads(1):
        result = run_demo(args.source_run, args.out, steps=args.steps, prior_training_run=args.prior_training_run)
    print(json.dumps({'status': result['status'], 'strata': result.get('trainReplayStrata'), 'knownCards': result.get('knownCardReplay', {}).get('allThreeStylesFull20Passes')}))
    if result['status'] != 'completed-train-all-demo-not-promoted':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
