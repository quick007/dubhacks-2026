"""Separate CPU-only diagnostics and shared-slot text-head candidates.

Reuses the immutable native48 fact features and v1 TRAIN labels. No Qwen/MLX
execution, new teacher data, heldout normalization, serving mutation or runtime
template. All declared final checkpoints are retained, including failures.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from . import store_parallel_text_data_v1 as data
from . import store_parallel_text_head_v1 as base
from .store_parallel_text_run_v1 import load_packet, sha, dump

VERSION = 'parallel-text-cpu-followup.v2'
ARMS = (
    {'name': 'train-normalized-independent160', 'sharedSlots': False, 'normalize': True, 'steps': 160},
    {'name': 'shared-slots-raw800', 'sharedSlots': True, 'normalize': False, 'steps': 800},
    {'name': 'shared-slots-train-normalized800', 'sharedSlots': True, 'normalize': True, 'steps': 800},
)


def train_feature_keys(packet):
    return sorted({key for row in packet['labels']['train'] for key in row['featureKeys']})


def normalization(packet, feature_map, *, enabled):
    keys = train_feature_keys(packet)
    train = torch.stack([feature_map[key] for key in keys])
    if enabled:
        mean = train.mean(0)
        std = train.std(0, unbiased=False)
        floor = max(float(std.median()) * .25, 1e-6)
        inverse = std.clamp_min(floor).reciprocal()
    else:
        mean, inverse, floor = torch.zeros(base.FEATURE_WIDTH), torch.ones(base.FEATURE_WIDTH), None
    return mean, inverse, {'enabled': enabled, 'fitSource': 'unique TRAIN source bindings only',
        'fitBindings': len(keys), 'fitFeatureKeysSha256': data.digest(keys), 'stdFloor': floor,
        'meanTensorSha256': base.tensor_sha(mean), 'inverseStdTensorSha256': base.tensor_sha(inverse),
        'evalFeatureKeysUsedForFit': False}


class ParallelTextV2(nn.Module):
    """Either normalized-v1 control or position-conditioned shared word logits."""
    def __init__(self, vocabulary, mean, inverse_std, *, shared_slots=True, width=128, seed=7211):
        super().__init__()
        if mean.shape != (4096,) or inverse_std.shape != (4096,) or not torch.isfinite(mean).all() or not torch.isfinite(inverse_std).all() or not (inverse_std > 0).all():
            raise ValueError('Finite TRAIN-fitted complete normalization buffers required')
        self.vocabulary, self.width, self.shared_slots = tuple(vocabulary), width, shared_slots
        self.register_buffer('feature_mean', mean.clone())
        self.register_buffer('feature_inverse_std', inverse_std.clone())
        if not shared_slots:
            self.independent = base.ParallelTextHead(vocabulary, width=width, seed=seed)
        else:
            # One vocabulary classifier is trained by ALL positions. Position
            # conditioning is learned but does not own a separate word class.
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.source = nn.Linear(4096, width)
                self.condition = nn.Linear(len(data.KINDS) + len(data.STYLES) + 2, width, bias=False)
                self.position = nn.Parameter(torch.randn(data.SLOTS, width) * .03)
                self.interaction = nn.Linear(width, width)
                self.words = nn.Linear(width, len(vocabulary))

    def forward(self, features, conditions):
        if features.device.type != 'cpu' or conditions.device.type != 'cpu':
            raise ValueError('CPU only; no GPU timing interference')
        if features.ndim != 2 or features.shape[1] != 4096 or conditions.shape != (len(features), 9):
            raise ValueError('Exact complete feature/condition input shape required')
        centered = (features - self.feature_mean) * self.feature_inverse_std
        if not self.shared_slots:
            return self.independent(centered, conditions)
        state = self.source(centered) + self.condition(conditions)
        slots = F.gelu(state[:, None, :] + self.position[None, :, :])
        return self.words(F.gelu(self.interaction(slots)))


def feature_geometry(packet, bundle):
    feature_map = base.validate_bundle(bundle, packet)
    values = bundle['features']; similarities = values @ values.T
    offdiag = similarities[~torch.eye(len(values), dtype=torch.bool)]
    train = torch.stack([feature_map[key] for key in train_feature_keys(packet)])
    color_cosines = [float(similarities[i, j]) for p in range(16) for i in range(p * 3, p * 3 + 3) for j in range(i + 1, p * 3 + 3)]
    within_deltas = [float(torch.linalg.vector_norm(values[i] - values[j])) for p in range(16) for i in range(p * 3, p * 3 + 3) for j in range(i + 1, p * 3 + 3)]
    return {'allPairCosineMean': float(offdiag.mean()), 'allPairCosineMin': float(offdiag.min()),
        'withinProductColorCosineMean': sum(color_cosines) / len(color_cosines),
        'withinProductColorCosineMin': min(color_cosines),
        'withinProductColorL2Mean': sum(within_deltas) / len(within_deltas),
        'trainCoordinateStdMean': float(train.std(0, unbiased=False).mean()),
        'trainCoordinateStdMin': float(train.std(0, unbiased=False).min()),
        'trainCoordinateStdMax': float(train.std(0, unbiased=False).max()),
        'diagnosticScope': 'descriptive all48 geometry only; actual fitted normalization uses unique34TRAIN only'}


def position_coverage(packet):
    slot_counts, conditioned_counts = Counter(), Counter()
    for row in packet['labels']['train']:
        inp = row['input']
        for pos, word in enumerate(data.tokens(row['targetText']) + [data.EOS]):
            slot_counts[pos, word] += 1
            conditioned_counts[inp['kind'], inp['style'], pos, word] += 1
    missing, conditioned, affected = set(), set(), []
    for row in packet['labels']['eval']:
        inp = row['input']; absent = []
        for pos, word in enumerate(data.tokens(row['targetText']) + [data.EOS]):
            if not slot_counts[pos, word]:
                missing.add((pos, word)); absent.append({'slot': pos, 'word': word})
            if not conditioned_counts[inp['kind'], inp['style'], pos, word]:
                conditioned.add((inp['kind'], inp['style'], pos, word))
        if absent:
            affected.append({'rowId': row['rowId'], 'kind': inp['kind'], 'missingPositiveSlotWords': absent})
    return {'zeroPositiveTrainSlotWordPairs': sorted(missing),
        'zeroPositiveTrainKindStyleSlotWordPairs': sorted(conditioned), 'affectedFields': affected,
        'interpretation': 'v1 independent word classifiers have no positive gradient for missing position-word classes; shared classifier removes this parameter-level restriction, not the need to learn word order'}


def factual_metrics(row, decoded):
    text, inp = decoded.get('text'), row['input']
    if inp['kind'] == 'section':
        return {'completeIdentity': None, 'selectedColor': None, 'exactItemPrice': None}
    facts = inp['source']['facts']; text = text or ''
    def span(value):
        return bool(re.search(r'(?<![A-Za-z])' + re.escape(value) + r'(?![A-Za-z])', text))
    from decimal import Decimal
    prices = re.findall(r'\$(\d+(?:\.\d+)?)', text)
    return {'completeIdentity': span(facts['name']) if inp['kind'] == 'title' else None,
        'selectedColor': span(facts['selectedColor']),
        'exactItemPrice': (len(prices) == 1 and Decimal(prices[0]) == Decimal(str(facts['itemPrice']))) if inp['kind'] == 'copy' else None}


def evaluate_fields(head, packet, feature_map, split):
    labels = packet['labels'][split]
    generated = base.generate(head, base.inference_rows(labels), feature_map)
    strata = {kind: {'fields': 0, 'accepted': 0, 'exactTeacher': 0, 'literal': defaultdict(lambda: [0, 0])} for kind in data.KINDS}
    receipts = []
    for row, out in zip(labels, generated['fields'], strict=True):
        kind = row['input']['kind']; metrics = factual_metrics(row, out); group = strata[kind]
        group['fields'] += 1; group['accepted'] += out['accepted']; group['exactTeacher'] += out['text'] == row['targetText']
        for metric, result in metrics.items():
            if result is not None:
                group['literal'][metric][0] += result; group['literal'][metric][1] += 1
        receipts.append({'rowId': row['rowId'], 'kind': kind, 'publicInput': row['input'],
                         'predicted': out, 'literalMetrics': metrics,
                         'offlineReference': row['targetText']})
    # Full references are retained offline for audit, never returned to model.
    return {'strata': strata, 'receipts': receipts,
        'scope': 'literal span checks are separate from screen grammar and independent human usefulness; source equality alone is not language quality'}


def fit_arm(packet, bundle, arm, *, width=128):
    feature_map = base.validate_bundle(bundle, packet)
    mean, inverse, normalization_receipt = normalization(packet, feature_map, enabled=arm['normalize'])
    head = ParallelTextV2(packet['vocabulary'], mean, inverse, shared_slots=arm['sharedSlots'], width=width)
    train = packet['labels']['train']
    features, conditions = base.serving_tensors(base.inference_rows(train), feature_map)
    target = base.encode_labels(train, packet['vocabulary'])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.004, weight_decay=.0001)
    losses = []; started = time.perf_counter()
    for step in range(arm['steps']):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, conditions)
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=-100)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite TRAIN loss')
        loss.backward(); optimizer.step()
        if step == 0 or step == arm['steps'] - 1:
            losses.append({'step': step + 1, 'loss': float(loss.detach())})
    # Exactly one evaluation after each declared final checkpoint. All arms
    # were declared before reading heldout outcomes and all are saved.
    head.eval()
    return head, {'arm': arm, 'normalization': normalization_receipt,
        'trainingBindings': 34, 'trainingFieldRows': len(train), 'lossSnapshots': losses,
        'fitSeconds': time.perf_counter() - started, 'width': width,
        'parameters': sum(p.numel() for p in head.parameters()), 'backboneUpdates': 0,
        'train': evaluate_fields(head, packet, feature_map, 'train'),
        'eval': evaluate_fields(head, packet, feature_map, 'eval')}


def run_cpu(source_run, out):
    source_run, out = Path(source_run), Path(out)
    if out.exists():
        raise FileExistsError('Unused output required; all failed candidates preserved')
    original = json.loads((source_run / 'results.json').read_text())
    if original['status'] != 'completed-candidate-not-promoted' or not all(original['shutdownIntegrity'].get(k) is True for k in ('frozen927HashesUnchanged', 'copyCheckpointUnchanged')):
        raise ValueError('Completed preserved v1 run with frozen integrity required')
    bundle_path = source_run / original['featureBundle']['file']
    if sha(bundle_path) != original['featureBundle']['sha256']:
        raise ValueError('Preserved native feature archive changed')
    packet = load_packet(); bundle = torch.load(bundle_path, map_location='cpu', weights_only=True)
    feature_map = base.validate_bundle(bundle, packet)
    out.mkdir(parents=True)
    report = {'version': VERSION, 'status': 'cpu-followup-started', 'nativeExecution': False,
        'sourceNativeRun': str(source_run), 'sourceResultSha256': sha(source_run / 'results.json'),
        'sourceFeatureBundleSha256': sha(bundle_path), 'sourceFrozenPreflightSha256': sha(source_run / 'preflight.json'),
        'declaredArms': ARMS, 'cpuThreads': 1, 'arms': [],
        'sourceSha256': sha(Path(__file__)),
        'scope': 'same curated known catalog;14heldout source combinations; three fixed TRAIN-only candidates; no live promotion, new native feature or browser latency claim'}
    dump(out / 'preflight.json', report)
    with base.bounded_cpu_threads(1):
        report['featureGeometry'] = feature_geometry(packet, bundle)
        report['positionCoverage'] = position_coverage(packet)
        for arm in ARMS:
            arm_out = out / arm['name']; arm_out.mkdir()
            try:
                head, receipt = fit_arm(packet, bundle, arm)
                torch.save({'state_dict': head.state_dict(), 'vocabulary': list(head.vocabulary),
                    'width': head.width, 'sharedSlots': head.shared_slots, 'arm': arm,
                    'sourceFeatureBundleSha256': sha(bundle_path)}, arm_out / 'head.pt')
                dump(arm_out / 'fields.json', receipt)
                views = base.evaluate_cases(head, packet['cases']['eval'], feature_map)
                dump(arm_out / 'heldout-views.json', views)
                summary = {'name': arm['name'], 'status': 'completed-unpromoted', 'checkpointSha256': sha(arm_out / 'head.pt'),
                    'normalization': receipt['normalization'], 'lossSnapshots': receipt['lossSnapshots'],
                    'train': receipt['train']['strata'], 'eval': receipt['eval']['strata'],
                    'heldoutViews': {k: v for k, v in views.items() if k != 'receipts'}}
                report['arms'].append(summary)
                dump(out / 'results.json', report)
                print(json.dumps({'arm': arm['name'], 'status': summary['status'], 'eval': summary['eval']}), flush=True)
            except BaseException as error:
                failed = {'name': arm['name'], 'status': 'failed', 'error': {'type': type(error).__name__, 'message': str(error)}}
                dump(arm_out / 'failure.json', failed); report['arms'].append(failed)
                dump(out / 'results.json', report)
                raise
    report['status'] = 'completed-cpu-candidates-not-promoted'
    dump(out / 'results.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-cpu', action='store_true', required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with base.bounded_cpu_threads(1):
        result = run_cpu(args.source_run, args.out)
    print(json.dumps({'status': result['status'], 'nativeExecution': False}))


if __name__ == '__main__':
    main()
