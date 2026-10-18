"""CPU research: learned parallel pointer/vocabulary words from public facts.

The original34 TRAIN/14 heldout source bindings and teachers are unchanged.
No runtime teacher strings, field format rules, output repair or GPU execution.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from . import store_parallel_text_data_v1 as data
from . import store_parallel_text_head_v1 as base
from .store_parallel_text_run_v1 import PACKET, sha, dump
from .store_catalog_v1 import load_snapshot
from .store_parallel_text_cpu_v2 import factual_metrics

VERSION = 'parallel-public-fact-pointer.v3'
SOURCE_CAP = 8
ARMS = ({'name': 'typed-role-position-pointer800', 'lexicalKeys': False, 'steps': 800},
        {'name': 'typed-role-position-word-pointer800', 'lexicalKeys': True, 'steps': 800})


def load_research_packet():
    """Read TRAIN teachers/public cases; do not decode heldout label text."""
    manifest = json.loads((PACKET / 'manifest.json').read_text())
    if manifest['schemaVersion'] != data.VERSION or manifest['sourceSha256'] != sha(Path(data.__file__)):
        raise ValueError('Original frozen data source/schema differs')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or sha(PACKET / name) != expected:
            raise ValueError('Original frozen packet bytes changed')
    records = json.loads((PACKET / 'public-feature-contract.json').read_text())['records']
    expected_records = [data.public_record({'name': product['name'], 'category': product['category'],
        'itemPrice': product['itemPrice'], 'selectedColor': variant['colorName']})
        for product in load_snapshot()['inputs']['products'] for variant in product['colorVariants']]
    if records != expected_records:
        raise ValueError('Actual48public catalog fact inputs differ')
    train = json.loads((PACKET / 'labels.train.json').read_text())
    cases = json.loads((PACKET / 'cases.eval.json').read_text())
    keys = {key for row in train if row['input']['kind'] != 'section' for key in row['featureKeys']}
    heldout = {key for case in cases for key in case['sourceBindingKeys']}
    if len(train) != 402 or len(keys) != 34 or len(heldout) != 14 or keys & heldout:
        raise ValueError('Original34TRAIN/14heldout partition differs')
    for row in train:
        if not set(row['featureKeys']) <= keys:
            raise ValueError('TRAIN section pools heldout features')
        inp = row['input']
        if data.field_rejections(inp['kind'], row['targetText'], inp.get('source', {}).get('facts')):
            raise ValueError('Frozen TRAIN teacher fails original factual screen')
    return {'featureRecords': records, 'vocabulary': json.loads((PACKET / 'vocabulary.json').read_text()),
        'labels': {'train': train}, 'cases': {'eval': cases},
        'evalLabelSha256': manifest['files']['labels.eval.json'], 'heldoutLabelsDecoded': False}


def inference_tensors(rows, feature_map, vocabulary):
    """Only public source facts, public conditions and verified Qwen vectors."""
    features, conditions = base.serving_tensors(rows, feature_map)
    lookup = {word: i for i, word in enumerate(vocabulary)}
    source = torch.ones(len(rows), SOURCE_CAP, dtype=torch.long)
    roles = torch.zeros_like(source); positions = torch.zeros_like(source)
    mask = torch.zeros(len(rows), SOURCE_CAP, dtype=torch.bool)
    lengths = torch.zeros(len(rows), 3)
    for index, row in enumerate(rows):
        # Sections legitimately contain no product facts; source mask remains
        # empty. This is data extraction, not an output formatting rule.
        evidence = row['input'].get('source')
        if evidence is None:
            continue
        facts = evidence['facts']
        groups = (data.tokens(facts['name']), data.tokens(facts['selectedColor']),
                  ['$' + data.money(facts['itemPrice'])])
        words = [(word, role + 1, position) for role, group in enumerate(groups)
                 for position, word in enumerate(group)]
        if len(words) > SOURCE_CAP:
            raise ValueError('Whole public fact words exceed8; no truncation')
        for offset, (word, role, position) in enumerate(words):
            if word not in lookup:
                raise ValueError('Public source word is outside declared vocabulary; no target/UNK repair')
            source[index, offset] = lookup[word]; roles[index, offset] = role
            positions[index, offset] = position; mask[index, offset] = True
        lengths[index] = torch.tensor([len(group) / 4 for group in groups])
    return {'features': features, 'conditions': conditions, 'sourceIds': source,
            'sourceRoles': roles, 'sourcePositions': positions, 'sourceMask': mask,
            'factLengths': lengths}


class PointerWordHead(nn.Module):
    def __init__(self, vocabulary, *, lexical_keys=False, width=64, seed=7211):
        super().__init__()
        if vocabulary[:2] != [data.EOS, data.PAD] or len(set(vocabulary)) != len(vocabulary):
            raise ValueError('Frozen EOS/PAD vocabulary required')
        self.vocabulary = tuple(vocabulary); self.width = width; self.lexical_keys = lexical_keys
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.source = nn.Linear(4096, width)
            self.condition = nn.Linear(12, width, bias=False)
            self.slot = nn.Parameter(torch.randn(data.SLOTS, width) * .03)
            self.interaction = nn.Linear(width, width)
            self.words = nn.Linear(width, len(vocabulary))
            self.pointer_query = nn.Linear(width, 32)
            self.role_keys = nn.Embedding(4, 32)
            self.position_keys = nn.Embedding(SOURCE_CAP, 32)
            self.word_keys = nn.Embedding(len(vocabulary), 32) if lexical_keys else None
            self.copy_gate = nn.Linear(width, 1)

    def forward(self, batch):
        if any(value.device.type != 'cpu' for value in batch.values()):
            raise ValueError('CPU only')
        features, conditions = batch['features'], batch['conditions']
        if features.ndim != 2 or features.shape[1] != 4096 or conditions.shape != (len(features), 9):
            raise ValueError('Exact Qwen/public condition dimensions required')
        state = self.source(features) + self.condition(torch.cat((conditions, batch['factLengths']), dim=1))
        slots = F.gelu(self.interaction(F.gelu(state[:, None, :] + self.slot[None, :, :])))
        vocabulary = self.words(slots).softmax(-1)
        keys = self.role_keys(batch['sourceRoles']) + self.position_keys(batch['sourcePositions'])
        if self.word_keys is not None:
            keys = keys + self.word_keys(batch['sourceIds'])
        scores = torch.einsum('nsd,nkd->nsk', self.pointer_query(slots), keys) / math.sqrt(32)
        source_mask = batch['sourceMask'][:, None, :]
        attention = scores.masked_fill(~source_mask, -1e4).softmax(-1) * source_mask
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-9)
        indices = batch['sourceIds'][:, None, :].expand(-1, data.SLOTS, -1)
        copied = torch.zeros_like(vocabulary).scatter_add(-1, indices, attention)
        copy_probability = self.copy_gate(slots).sigmoid()
        copy_probability = copy_probability * batch['sourceMask'].any(-1)[:, None, None]
        probabilities = vocabulary * (1 - copy_probability) + copied * copy_probability
        # Empty public sources contribute zero pointer mass. Renormalization
        # makes this an ordinary vocabulary distribution without a field rule.
        probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-9)
        return {'probabilities': probabilities, 'attention': attention, 'copyProbability': copy_probability}


def generate(head, rows, feature_map):
    started = time.perf_counter()
    batch = inference_tensors(rows, feature_map, list(head.vocabulary))
    head.eval()
    with torch.inference_mode():
        predicted = head(batch)
        ids = predicted['probabilities'].argmax(-1).tolist()
        gates = predicted['copyProbability'].squeeze(-1).tolist()
        attention = predicted['attention'].tolist()
    results = []
    for index, (row, words) in enumerate(zip(rows, ids, strict=True)):
        output = base.decode_ids(words, head.vocabulary)
        if output['text'] is not None:
            inp = row['input']
            output['rejections'] = data.field_rejections(inp['kind'], output['text'], inp.get('source', {}).get('facts'))
        output.update(accepted=not output['rejections'], copyProbabilityBySlot=gates[index],
                      sourceAttentionBySlot=attention[index], sourceWordIds=batch['sourceIds'][index].tolist())
        results.append(output)
    return {'version': VERSION, 'fields': results, 'fieldCount': len(results), 'feedforwardCalls': 1,
        'autoregressiveSteps': 0, 'teacherStringsRead': False, 'outputFormatsOrRepairsUsed': False,
        'cpuHeadAndValidationMs': (time.perf_counter() - started) * 1000,
        'scope': 'CPU neural pointer+vocabulary head only; no fresh query, ranking, serialization, network or browser readiness claim'}


def evaluate(head, packet, feature_map, split):
    rows = packet['labels'][split]
    result = generate(head, base.inference_rows(rows), feature_map)
    strata = {kind: {'fields': 0, 'accepted': 0, 'exactTeacher': 0, 'literal': defaultdict(lambda: [0, 0])} for kind in data.KINDS}
    receipts = []
    for row, predicted in zip(rows, result['fields'], strict=True):
        group = strata[row['input']['kind']]; metrics = factual_metrics(row, predicted)
        group['fields'] += 1; group['accepted'] += predicted['accepted']; group['exactTeacher'] += predicted['text'] == row['targetText']
        for name, value in metrics.items():
            if value is not None:
                group['literal'][name][0] += value; group['literal'][name][1] += 1
        receipts.append({'rowId': row['rowId'], 'publicInput': row['input'], 'prediction': predicted,
                         'literalMetrics': metrics, 'offlineReference': row['targetText']})
    return {'strata': strata, 'receipts': receipts,
        'scope': 'same frozen34TRAIN/14heldout source bindings; literal metrics/screen/exact-reference scores separate; no independent human claim'}


def fit_arm(packet, feature_map, arm, *, width=64):
    head = PointerWordHead(packet['vocabulary'], lexical_keys=arm['lexicalKeys'], width=width)
    rows = packet['labels']['train']
    batch = inference_tensors(base.inference_rows(rows), feature_map, packet['vocabulary'])
    target = base.encode_labels(rows, packet['vocabulary'])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.004, weight_decay=.0001)
    losses = []; started = time.perf_counter()
    for step in range(arm['steps']):
        optimizer.zero_grad(set_to_none=True)
        prediction = head(batch)['probabilities'].clamp_min(1e-9).log()
        loss = F.nll_loss(prediction.flatten(0, 1), target.flatten(), ignore_index=-100)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite TRAIN-only pointer loss')
        loss.backward(); optimizer.step()
        if step == 0 or step == arm['steps'] - 1:
            losses.append({'step': step + 1, 'loss': float(loss.detach())})
    return head, {'arm': arm, 'lossSnapshots': losses, 'fitSeconds': time.perf_counter() - started,
        'trainingRows': len(rows), 'trainBindings': 34, 'heldoutBindings': 14,
        'parameters': sum(p.numel() for p in head.parameters()),
        'publicSourceRoles': {'1': 'name', '2': 'selectedColor', '3': 'itemPrice'},
        'teacherWordsOrFormatsAtInference': False, 'inputAugmentation': False,
        'train': evaluate(head, packet, feature_map, 'train')}


def run_cpu(source_run, out):
    source_run, out = Path(source_run), Path(out)
    if out.exists():
        raise FileExistsError('Unused output required; preserve all failures')
    source = json.loads((source_run / 'results.json').read_text())
    if source['status'] != 'completed-candidate-not-promoted' or not all(source['shutdownIntegrity'][k] for k in ('frozen927HashesUnchanged', 'copyCheckpointUnchanged')):
        raise ValueError('Preserved completed original v1 required')
    feature_path = source_run / source['featureBundle']['file']
    if sha(feature_path) != source['featureBundle']['sha256']:
        raise ValueError('Native48 feature bytes changed')
    packet = load_research_packet(); bundle = torch.load(feature_path, map_location='cpu', weights_only=True)
    feature_map = base.validate_bundle(bundle, packet)
    out.mkdir(parents=True)
    report = {'version': VERSION, 'status': 'started-cpu-pointer-research', 'nativeExecution': False,
        'sourceFeatureBundleSha256': sha(feature_path), 'sourceResultSha256': sha(source_run / 'results.json'),
        'sourceSha256': sha(Path(__file__)), 'declaredArms': ARMS, 'cpuThreads': 1, 'arms': [],
        'trainingBindings': 34, 'heldoutBindings': 14, 'trainingRows': 402,
        'inputAugmentation': False, 'hiddenPersonaInputs': False,
        'selection': 'both fixed800finalcheckpoints retained; no heldout selection or iterative adjustment',
        'inferenceInputs': 'verified frozenQwen4096features; publickind/style/section; publicname/color/pricewords withrole+within-factposition+length only',
        'scope': 'known-catalog compositional research, no live promotion/end-to-end latency/unseen catalog claim'}
    dump(out / 'preflight.json', report); (out / 'source.py').write_text(Path(__file__).read_text())
    with base.bounded_cpu_threads(1):
        fitted = []
        for arm in ARMS:
            arm_out = out / arm['name']; arm_out.mkdir()
            try:
                head, result = fit_arm(packet, feature_map, arm)
                torch.save({'state_dict': head.state_dict(), 'vocabulary': list(head.vocabulary),
                    'width': head.width, 'lexicalKeys': head.lexical_keys, 'version': VERSION}, arm_out / 'head.pt')
                dump(arm_out / 'HEAD_FROZEN.json', {'checkpointSha256': sha(arm_out / 'head.pt'),
                    'sourceSha256': report['sourceSha256'], 'steps': arm['steps'],
                    'heldoutLabelTextDecodedBeforeFreeze': False,
                    'heldoutLabelsAbsentFromTrainingPacket': True})
                dump(arm_out / 'TRAIN-fit.json', result)
                fitted.append((arm, head, result, arm_out))
            except BaseException as error:
                failed = {'name': arm['name'], 'status': 'failed-fit', 'error': {'type': type(error).__name__, 'message': str(error)}}
                dump(arm_out / 'failure.json', failed); report['arms'].append(failed); dump(out / 'results.json', report); raise
        dump(out / 'ALL_HEADS_FROZEN.json', {'sourceSha256': report['sourceSha256'],
            'checkpoints': {arm['name']: sha(arm_out / 'head.pt') for arm, _, _, arm_out in fitted},
            'allTrainingCompleteBeforeHeldoutLabelTextDecode': True})
        report['allHeadsFrozenBeforeAnyHeldoutTargetDecode'] = True
        for arm, head, result, arm_out in fitted:
            try:
                # BOTH final checkpoints have been hashed before any heldout
                # target is decoded. No subsequent training occurs.
                if sha(PACKET / 'labels.eval.json') != packet['evalLabelSha256']:
                    raise ValueError('Frozen heldout label bytes changed')
                eval_packet = {**packet, 'labels': {**packet['labels'],
                    'eval': json.loads((PACKET / 'labels.eval.json').read_text())}}
                result['eval'] = evaluate(head, eval_packet, feature_map, 'eval')
                dump(arm_out / 'fields.json', result)
                saved = torch.load(arm_out / 'head.pt', map_location='cpu', weights_only=True)
                restored = PointerWordHead(saved['vocabulary'], lexical_keys=saved['lexicalKeys'], width=saved['width'])
                restored.load_state_dict(saved['state_dict'], strict=True)
                views = []
                for case in packet['cases']['eval']:
                    rows = base.inference_rows(case['rows'])
                    generated = generate(restored, rows, feature_map)
                    before = generate(head, rows, feature_map)
                    if [f['slotIds'] for f in before['fields']] != [f['slotIds'] for f in generated['fields']]:
                        raise ValueError('Reloaded pointer slot parity failed')
                    # Source/action metadata are deliberately omitted here;
                    # this research view only audits20 learned field outputs.
                    views.append({'caseId': case['caseId'], 'fields': generated,
                        'firstThreeAll10Accepted': all(f['accepted'] for f in generated['fields'][:10]),
                        'fullSixAll20Accepted': all(f['accepted'] for f in generated['fields'])})
                dump(arm_out / 'heldout-views.json', views)
                summary = {'name': arm['name'], 'status': 'completed-research-not-promoted',
                    'checkpointSha256': sha(arm_out / 'head.pt'), 'lossSnapshots': result['lossSnapshots'],
                    'fitSeconds': result['fitSeconds'], 'train': result['train']['strata'], 'eval': result['eval']['strata'],
                    'views': {'correlatedCases': 42, 'independentHeldoutBindings': 14,
                        'firstThreeAll10Accepted': sum(v['firstThreeAll10Accepted'] for v in views),
                        'fullSixAll20Accepted': sum(v['fullSixAll20Accepted'] for v in views)}, 'reloadExactSlotParity': True}
                report['arms'].append(summary); dump(out / 'results.json', report)
                print(json.dumps(summary), flush=True)
            except BaseException as error:
                failed = {'name': arm['name'], 'status': 'failed', 'error': {'type': type(error).__name__, 'message': str(error)}}
                dump(arm_out / 'failure.json', failed); report['arms'].append(failed); dump(out / 'results.json', report); raise
    report['status'] = 'completed-cpu-pointer-research-not-promoted'; dump(out / 'results.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-cpu', required=True, action='store_true')
    parser.add_argument('--source-run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    with base.bounded_cpu_threads(1):
        result = run_cpu(args.source_run, args.out)
    print(json.dumps({'status': result['status'], 'nativeExecution': False}))


if __name__ == '__main__':
    main()
