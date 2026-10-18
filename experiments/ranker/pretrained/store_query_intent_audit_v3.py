"""Independent CPU audit/replay of frozen TRAIN query gate; never reads quality gold.

Source preparation is separate from execution. Replay establishes reproducibility of
250 declared updates, not direct observation of the earlier optimizer process.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from . import store_query_intent_gate_v3 as gate
from .store_query_intent_diagnostics_v2 import summarize

VERSION = 'query-intent-independent-TRAIN-audit.v3'
EXPECTED_ROWS = 1272
EXPECTED_FRESH = 1176
REPLAY_STEPS = 250


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def independent_initial_head():
    with torch.random.fork_rng():
        torch.manual_seed(20261019)
        return nn.Linear(4096, 1)


def independent_loss(logits, labels):
    if logits.shape != labels.shape or set(labels.tolist()) != {0., 1.}:
        raise ValueError('Complete two-class TRAIN logits/labels required')
    elementwise = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    return (elementwise[labels == 0].mean() + elementwise[labels == 1].mean()) / 2


def state_delta(initial, final):
    if set(initial) != set(final):
        raise ValueError('Parameter identity mismatch')
    result = {}
    for name in sorted(initial):
        before, after = initial[name], final[name]
        if before.shape != after.shape or not torch.isfinite(before).all() or not torch.isfinite(after).all():
            raise ValueError('Finite matching parameter shapes required')
        difference = after - before
        result[name] = {'initialSha256': tensor_sha(before), 'finalSha256': tensor_sha(after),
                        'changed': not torch.equal(before, after),
                        'changedElements': int((difference != 0).sum()),
                        'maxAbsoluteDelta': float(difference.abs().max()),
                        'l2Delta': float(difference.norm())}
    return result


def replay_updates(x, labels):
    """Independent fixed procedure, not the original fit() implementation."""
    head = independent_initial_head()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.03, weight_decay=.001)
    count, snapshots = 0, []
    for index in range(REPLAY_STEPS):
        optimizer.zero_grad()
        loss = independent_loss(head(x).squeeze(-1), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Independent replay loss is nonfinite')
        loss.backward()
        optimizer.step()
        count += 1
        if index in (0, 49, 99, 249):
            snapshots.append({'step': count, 'loss': float(loss.detach())})
    return head, count, snapshots


def verify_saved_predictions(rows, probabilities, saved):
    values = saved['rows']
    if len(values) != len(rows) or len(probabilities) != len(rows):
        raise ValueError('Saved prediction denominator differs')
    for row, probability, result in zip(rows, probabilities, values, strict=True):
        if result['caseId'] != row['caseId'] or result['labelSpecific'] != row['labelSpecific']:
            raise ValueError('Saved prediction identity/label mismatch')
        if result['pSpecific'] != probability:
            raise ValueError('Reload prediction differs from saved TRAIN probability')
    return {'rows': len(rows), 'exactSavedProbabilityParity': True}


def _load_inputs(feature_run, fit_run, expected_plan_sha):
    paths = {'plan': feature_run / 'plan.json', 'nativeReport': feature_run / 'report.json',
             'features': feature_run / 'query-features.pt', 'head': fit_run / 'head.pt',
             'headFreeze': fit_run / 'HEAD_FROZEN.json', 'trainResults': fit_run / 'train-results.json'}
    if gate.sha(paths['plan']) != expected_plan_sha:
        raise ValueError('Explicit expected frozen TRAIN manifest differs')
    plan = json.loads(paths['plan'].read_text())
    native = json.loads(paths['nativeReport'].read_text())
    freeze = json.loads(paths['headFreeze'].read_text())
    if plan['version'] != gate.VERSION or plan['recipe'] != gate.RECIPE:
        raise ValueError('Exact v3 frozen recipe required')
    gate.pincheck(plan['sha256'])
    if native['version'] != gate.VERSION or native['sourceManifestSha256'] != expected_plan_sha:
        raise ValueError('Native manifest/version differs')
    counters = ('attemptedNativeEncodes', 'physicalCompletedNativeForwards',
                'savedNativeFeatureReceipts', 'completedNativeForwards')
    if (native['status'] != 'completed-native-features' or
        any(native.get(k) != EXPECTED_FRESH for k in counters) or
        native['reusedVectors'] != 96 or native['frozen927Unchanged'] is not True or
        native['shutdownCompleted'] is not True or
        len(native['roster']) != EXPECTED_FRESH or
        any(item['status'] != 'completed' for item in native['roster'])):
        raise ValueError('Complete verified1176forward/saved receipts +96reuse required')
    expected_roster = [r['caseId'] for r in plan['rows'] if r['origin'] == 'fresh-native-required']
    if [r['caseId'] for r in native['roster']] != expected_roster:
        raise ValueError('Native fixed roster identity differs')
    if (len(native['beforeParameterSha256']) != 927 or
        native['beforeParameterSha256'] != native['afterParameterSha256']):
        raise ValueError('All927 actual tensor hashes must match')
    if gate.sha(paths['features']) != native['queryFeaturesSha256']:
        raise ValueError('Native feature checksum differs')
    if (freeze['version'] != gate.VERSION or freeze['optimizerUpdates'] != 250 or
        freeze['sourceManifestSha256'] != expected_plan_sha or
        freeze['qualityEvaluationLabelsRead'] is not False):
        raise ValueError('Fixed final250update TRAIN checkpoint declaration required')
    bundle = torch.load(paths['features'], map_location='cpu', weights_only=True)
    rows = plan['rows']
    if (bundle['version'] != gate.VERSION or bundle['caseIds'] != [r['caseId'] for r in rows] or
        len(rows) != EXPECTED_ROWS or len(set(bundle['caseIds'])) != EXPECTED_ROWS or
        bundle['sourceManifestSha256'] != expected_plan_sha or
        bundle['features'].shape != (EXPECTED_ROWS, 4096)):
        raise ValueError('Exact1272row TRAIN feature identity/shape required')
    x = bundle['features'].float()
    for vector in x:
        gate._feature(vector)
    if (sum(r['labelSpecific'] == 0 for r in rows), sum(r['labelSpecific'] == 1 for r in rows)) != (504, 768):
        raise ValueError('Exact504broad/768specific classes required')
    labels = torch.tensor([r['labelSpecific'] for r in rows], dtype=torch.float32)
    head, metadata = gate.load_intent_head(paths['head'], freeze['headSha256'],
                                         expected_source_manifest_sha256=expected_plan_sha)
    if (metadata['featureBundleSha256'] != native['queryFeaturesSha256'] or
        metadata['sourceSha256'] != gate.sha(gate.__file__) or
        metadata['trainingOnly'] is not True or metadata['qualityEvaluationLabelsRead'] is not False):
        raise ValueError('Checkpoint source/features/TRAIN scope differ')
    saved = json.loads(paths['trainResults'].read_text())
    if saved['version'] != gate.VERSION:
        raise ValueError('Saved TRAIN prediction version differs')
    return paths, rows, x, labels, head, metadata, saved


def audit(feature_run, fit_run, out, expected_plan_sha):
    feature_run, fit_run, out = Path(feature_run), Path(fit_run), Path(out)
    if out.exists():
        raise FileExistsError('Unique independent audit output required')
    out.mkdir(parents=True)
    start, old_threads = time.perf_counter(), torch.get_num_threads()
    result = {'version': VERSION, 'status': 'starting', 'privateQualityLabelsRead': False,
              'sourceManifestSha256': expected_plan_sha, 'auditSourceSha256': gate.sha(__file__),
              'scope': 'TRAIN reproducibility/finite updates only; no quality or generalization claim',
              'historicalUpdateProofLimit': 'Replay matches declared250update procedure; not direct instrumentation of original optimizer calls'}
    try:
        torch.set_num_threads(1)
        paths, rows, x, labels, head, metadata, saved = _load_inputs(feature_run, fit_run, expected_plan_sha)
        result['inputSha256'] = {str(path): gate.sha(path) for path in paths.values()}
        checkpoint_state = {k.removeprefix('linear.'): v for k, v in head.state_dict().items()}
        initial = independent_initial_head()
        deltas = state_delta(initial.state_dict(), checkpoint_state)
        result['parameters'] = deltas
        if not all(row['changed'] for row in deltas.values()):
            raise ValueError('Both linear weight and bias must have actually changed')
        with torch.inference_mode():
            logits = head(x)
            probabilities = logits.sigmoid().tolist()
            final_loss = float(independent_loss(logits, labels))
        result['reloadSavedParity'] = verify_saved_predictions(rows, probabilities, saved)
        second, _ = gate.load_intent_head(paths['head'], gate.sha(paths['head']),
                                          expected_source_manifest_sha256=expected_plan_sha)
        with torch.inference_mode():
            if not torch.equal(head(x), second(x)):
                raise ValueError('Two independently reloaded checkpoints disagree')
        result['twoReloadsExactLogitParity'] = True
        result['independentFinalBalancedBCE'] = final_loss
        replay, steps, snapshots = replay_updates(x, labels)
        differences = state_delta(checkpoint_state, replay.state_dict())
        result['replay'] = {'observedReplayOptimizerSteps': steps,
                            'everyFinalParameterBitwiseMatches': all(not d['changed'] for d in differences.values()),
                            'parameterDifferences': differences, 'lossSnapshots': snapshots}
        if steps != 250 or not result['replay']['everyFinalParameterBitwiseMatches']:
            raise ValueError('Independent250update replay differs from frozen checkpoint')
        if snapshots != saved['lossSnapshots']:
            raise ValueError('Independent replay loss snapshots differ')
        result['diagnostics'] = summarize(rows, probabilities)
        result['headSha256'] = gate.sha(paths['head'])
        result['status'] = 'passed-independent-TRAIN-replay'
    except BaseException as error:
        result.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        torch.set_num_threads(old_threads)
        result['elapsedSeconds'] = time.perf_counter() - start
        gate.dump(out / 'audit.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args(argv)
    result = audit(args.features, args.fit, args.out, args.expected_plan_sha256)
    print(json.dumps({k: result[k] for k in ('status', 'headSha256', 'independentFinalBalancedBCE')}, indent=2))


if __name__ == '__main__':
    main()
