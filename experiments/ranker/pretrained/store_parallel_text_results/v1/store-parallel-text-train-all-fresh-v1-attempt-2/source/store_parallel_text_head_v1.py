"""CPU fixed-slot generative decoder on frozen public Qwen fact features.

Every output slot is predicted in one feedforward pass. The model receives no
previous generated/teacher words and serving never calls a teacher constructor.
This isolated candidate has not been fitted on native features or promoted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from statistics import median
import time

import torch
from torch import nn
from torch.nn import functional as F

from .store_catalog_v1 import canonical, digest
from .store_parallel_text_data_v1 import (EOS, PAD, KINDS, STYLES, SLOTS, VERSION,
    detokenize, tokens, field_rejections, public_record)

MODEL_REVISION = '8b2b98c00a6b4d291155e4890773ca8f769aee53'
FEATURE_WIDTH = 4096
HEAD_VERSION = 'qwen-fact-parallel-word-head.v1'


@contextmanager
def bounded_cpu_threads(count=1):
    old = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def tensor_sha(values):
    return hashlib.sha256(values.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_feature_bundle(records, values, provenance):
    """Root-owned extractor adapter. No native import or extraction happens here.

    `provenance` must describe the actual extraction, not an inherited fixture.
    Record order exactly matches the fresh 48 native B1 forward outputs.
    """
    if (type(provenance) is not dict or provenance.get('evidenceSource') != 'native-frozen-qwen' or
            provenance.get('modelRevision') != MODEL_REVISION or provenance.get('copyAdapterEnabled') is not False or
            provenance.get('nativeForwards') != 48 or provenance.get('truncatedRows') != 0):
        raise ValueError('Exact 48-forward frozen-base native extraction provenance required')
    values = values.detach().cpu().to(torch.float32).clone()
    validate_vectors(values, len(records))
    return {'schemaVersion': HEAD_VERSION, 'featureKeys': [r['featureKey'] for r in records],
            'textSha256': [r['textSha256'] for r in records], 'features': values,
            'featureTensorSha256': tensor_sha(values), 'provenance': provenance}


def validate_vectors(values, count):
    if (not isinstance(values, torch.Tensor) or values.device.type != 'cpu' or
            values.dtype != torch.float32 or list(values.shape) != [count, FEATURE_WIDTH] or
            not torch.isfinite(values).all() or
            not torch.allclose(torch.linalg.vector_norm(values, dim=1), torch.ones(count), atol=1e-5, rtol=1e-5)):
        raise ValueError('Finite CPU float32 unit-L2 4096-wide complete feature matrix required')


def validate_bundle(bundle, packet, *, fixture=False):
    records = packet['featureRecords']
    if bundle.get('schemaVersion') != HEAD_VERSION:
        raise ValueError('Feature bundle schema differs')
    if (bundle.get('featureKeys') != [r['featureKey'] for r in records] or
            bundle.get('textSha256') != [r['textSha256'] for r in records]):
        raise ValueError('Feature text/key order differs from public packet')
    values = bundle.get('features')
    validate_vectors(values, len(records))
    if tensor_sha(values) != bundle.get('featureTensorSha256'):
        raise ValueError('Feature matrix content differs from checksum')
    provenance = bundle.get('provenance', {})
    if fixture:
        if provenance != {'evidenceSource': 'engineered-cpu-fixture', 'nativeForwards': 0}:
            raise ValueError('Fixture must be explicitly non-native')
    else:
        # Repeat provenance checking, without rebuilding or mutating features.
        make_feature_bundle(records, values, provenance)
    return {key: values[i] for i, key in enumerate(bundle['featureKeys'])}


class ParallelTextHead(nn.Module):
    def __init__(self, vocabulary, *, width=128, seed=7211):
        super().__init__()
        if (type(vocabulary) is not list or vocabulary[:2] != [EOS, PAD] or
                len(set(vocabulary)) != len(vocabulary) or any(type(w) is not str for w in vocabulary)):
            raise ValueError('Unique lexical vocabulary with EOS/PAD at 0/1 required')
        if type(width) is not int or not 8 <= width <= 256:
            raise ValueError('Bounded hidden width required')
        self.vocabulary = tuple(vocabulary)
        self.width = width
        # Local seed scope preserves surrounding caller randomness.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.source = nn.Linear(FEATURE_WIDTH, width)
            self.condition = nn.Linear(len(KINDS) + len(STYLES) + 2, width, bias=False)
            self.output = nn.Linear(width, SLOTS * len(vocabulary))

    def forward(self, features, conditions):
        if features.device.type != 'cpu' or conditions.device.type != 'cpu':
            raise ValueError('This candidate is a CPU head; no GPU timing interference')
        if (features.ndim != 2 or features.shape[1] != FEATURE_WIDTH or
                conditions.shape != (len(features), len(KINDS) + len(STYLES) + 2)):
            raise ValueError('Exact public-feature/condition batch shape required')
        hidden = F.gelu(self.source(features) + self.condition(conditions))
        return self.output(hidden).reshape(len(features), SLOTS, len(self.vocabulary))


def serving_tensors(rows, feature_map):
    """Pure public row adapter: no labels, case IDs or target word sequences."""
    features, conditions = [], []
    for row in rows:
        if type(row) is not dict or set(row) != {'input', 'featureKeys'}:
            raise ValueError('Exact inference row shape; teacher/identity extras rejected')
        inp, keys = row['input'], row['featureKeys']
        kind, style = inp.get('kind'), inp.get('style')
        if kind not in KINDS or style not in STYLES:
            raise ValueError('Declared kind/style required')
        expected = {'kind', 'style', 'sectionIndex'} if kind == 'section' else {'kind', 'style', 'source'}
        if set(inp) != expected or type(keys) is not list or not keys or len(keys) != len(set(keys)):
            raise ValueError('Exact public inference input shape required')
        condition = torch.zeros(len(KINDS) + len(STYLES) + 2)
        condition[KINDS.index(kind)] = 1
        condition[len(KINDS) + STYLES.index(style)] = 1
        if kind == 'section':
            if len(keys) != 3 or type(inp['sectionIndex']) is not int or inp['sectionIndex'] not in (0, 1):
                raise ValueError('Section must pool its three selected public products')
            condition[len(KINDS) + len(STYLES) + inp['sectionIndex']] = 1
        else:
            if len(keys) != 1 or public_record(inp['source']['facts']) != inp['source'] or keys != [inp['source']['featureKey']]:
                raise ValueError('Card fact text/key/feature binding mismatch')
        try:
            vector = torch.stack([feature_map[key] for key in keys]).mean(0)
        except KeyError as exc:
            raise ValueError('Missing source feature; no fallback synthesis') from exc
        if vector.shape != (FEATURE_WIDTH,) or not torch.isfinite(vector).all():
            raise ValueError('Incomplete/nonfinite source feature')
        features.append(vector)
        conditions.append(condition)
    if not rows:
        raise ValueError('Nonempty generation batch required')
    return torch.stack(features), torch.stack(conditions)


def encode_labels(rows, vocabulary):
    lookup = {word: index for index, word in enumerate(vocabulary)}
    target = torch.full((len(rows), SLOTS), -100, dtype=torch.long)
    for i, row in enumerate(rows):
        values = tokens(row['targetText']) + [EOS]
        if len(values) > SLOTS:
            raise ValueError('Teacher exceeds bounded slots; refusing truncation')
        try:
            target[i, :len(values)] = torch.tensor([lookup[word] for word in values])
        except KeyError as exc:
            raise ValueError('OOV teacher token; no unknown-token target repair') from exc
    return target


def inference_rows(rows):
    return [{key: row[key] for key in ('input', 'featureKeys')} for row in rows]


def decode_ids(ids, vocabulary):
    if not isinstance(ids, list) or len(ids) != SLOTS or any(type(i) is not int or not 0 <= i < len(vocabulary) for i in ids):
        raise ValueError('Exact independent slot prediction receipt required')
    if 0 not in ids:
        return {'text': None, 'finishReason': 'length', 'slotIds': ids, 'rejections': ['missing-learned-eos']}
    stop = ids.index(0)
    if not stop or 1 in ids[:stop]:
        return {'text': None, 'finishReason': 'invalid', 'slotIds': ids, 'rejections': ['empty-or-pad-before-eos']}
    return {'text': detokenize([vocabulary[i] for i in ids[:stop]]), 'finishReason': 'eos',
            'slotIds': ids, 'firstEosSlot': stop, 'rejections': []}


def generate(head, rows, feature_map):
    started = time.perf_counter()
    features, conditions = serving_tensors(rows, feature_map)
    prepare_ms = (time.perf_counter() - started) * 1000
    head.eval()
    with torch.inference_mode():
        native_start = time.perf_counter()
        predicted = head(features, conditions).argmax(-1).tolist()
        forward_ms = (time.perf_counter() - native_start) * 1000
    outputs = []
    for row, ids in zip(rows, predicted, strict=True):
        decoded = decode_ids(ids, head.vocabulary)
        if decoded['text'] is not None:
            inp = row['input']
            decoded['rejections'] = field_rejections(inp['kind'], decoded['text'], inp.get('source', {}).get('facts'))
        decoded['accepted'] = not decoded['rejections']
        outputs.append(decoded)
    return {'schemaVersion': HEAD_VERSION, 'fields': outputs, 'fieldCount': len(rows),
            'autoregressiveSteps': 0, 'feedforwardCalls': 1, 'teacherStringsRead': False,
            'timing': {'prepareMs': prepare_ms, 'headForwardArgmaxMs': forward_ms,
                       'generationAndValidationMs': (time.perf_counter() - started) * 1000},
            'scope': 'CPU head only; excludes fresh query rank, catalog I/O, network, React and images'}


def evaluate_cases(head, cases, feature_map):
    """All 20 fields per six-card view, with first-three and full-six counters.

    These are correlated recombinations of the declared heldout bindings,
    not an independent sample of new customers/products or browser timing.
    """
    receipts = []
    expected_fields = ['S1'] + [f'{p}{i}' for i in range(1, 4) for p in 'TBC'] + ['S2'] + [f'{p}{i}' for i in range(4, 7) for p in 'TBC']
    for case in cases:
        if ([r.get('field') for r in case['rows']] != expected_fields or
                len(case['sourceBindingKeys']) != 6 or len(set(case['sourceBindingKeys'])) != 6):
            raise ValueError('Exact six-card/twenty-field case roster required')
        for row in case['rows']:
            prefix, position = row['field'][0], int(row['field'][1:]) - 1
            keys = (case['sourceBindingKeys'][position * 3:position * 3 + 3] if prefix == 'S'
                    else [case['sourceBindingKeys'][position]])
            kind = {'S': 'section', 'T': 'title', 'B': 'badge', 'C': 'copy'}[prefix]
            if row['featureKeys'] != keys or row['input']['kind'] != kind:
                raise ValueError('Case field/source-card binding mismatch')
        result = generate(head, inference_rows(case['rows']), feature_map)
        receipts.append({'caseId': case['caseId'], 'firstThreeAllFieldsAccepted': all(r['accepted'] for r in result['fields'][:10]),
            'fullSixAllFieldsAccepted': all(r['accepted'] for r in result['fields']), 'result': result})
    if not receipts:
        raise ValueError('Nonempty six-card evaluation cases required')
    return {'caseCount': len(receipts),
        'firstThreeAllFieldsAccepted': sum(r['firstThreeAllFieldsAccepted'] for r in receipts),
        'fullSixAllFieldsAccepted': sum(r['fullSixAllFieldsAccepted'] for r in receipts),
        'medianHeadForwardArgmaxMs': median(r['result']['timing']['headForwardArgmaxMs'] for r in receipts),
        'medianHeadPreparationGenerationValidationMs': median(r['result']['timing']['generationAndValidationMs'] for r in receipts),
        'timingScope': 'CPU head only; fresh Qwen query, ranking, catalog verification, network, React/images excluded',
        'receipts': receipts}


def fit(packet, bundle, *, steps=160, learning_rate=.004, width=128, fixture=False):
    if type(steps) is not int or not 1 <= steps <= 1000 or not 0 < learning_rate <= .1:
        raise ValueError('Bounded fixed CPU training configuration required')
    feature_map = validate_bundle(bundle, packet, fixture=fixture)
    train = packet['labels']['train']
    features, conditions = serving_tensors(inference_rows(train), feature_map)
    target = encode_labels(train, packet['vocabulary'])
    head = ParallelTextHead(packet['vocabulary'], width=width)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=.0001)
    start = time.perf_counter()
    losses = []
    with bounded_cpu_threads():
        head.train()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = head(features, conditions)
            loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=-100)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            if step == 0 or step == steps - 1:
                losses.append({'step': step + 1, 'loss': loss.item()})
        # Evaluation occurs once after the predeclared final step. No heldout
        # loss/checkpoint selection, early stopping or target-conditioned input.
        evaluations = {}
        for split in ('train', 'eval'):
            rows = packet['labels'][split]
            result = generate(head, inference_rows(rows), feature_map)
            exact = sum(out['text'] == row['targetText'] for out, row in zip(result['fields'], rows, strict=True))
            evaluations[split] = {'fields': len(rows), 'exactTeacherMatch': exact,
                'acceptedFields': sum(r['accepted'] for r in result['fields']),
                'missingEos': sum(r['finishReason'] == 'length' for r in result['fields'])}
    return head, {'status': 'engineered-fixture-only' if fixture else 'native-feature-candidate-not-promoted',
        'modelVersion': HEAD_VERSION, 'sourceEvidence': bundle['provenance'], 'seed': 7211,
        'steps': steps, 'trainingRows': len(train), 'parameterCount': sum(p.numel() for p in head.parameters()),
        'hiddenWidth': width, 'slotsPerField': SLOTS, 'vocabularySize': len(packet['vocabulary']),
        'lossSnapshots': losses, 'fitAndFinalEvaluationSeconds': time.perf_counter() - start,
        'evaluation': evaluations, 'backboneUpdates': 0, 'nativeCalls': 0,
        'heldoutUse': 'one final report only, no checkpoint selection',
        'limitations': ['Independent slots can disagree, repeat, omit or lack EOS.',
            'Bounded catalog words and short factual grammar, not open-vocabulary prose.',
            'Heldout product-color combinations; all product identities seen in TRAIN.',
            'Fresh Qwen query encoding and browser readiness are not measured by CPU head timing.',
            'No live frontend integration or template fallback.']}


def engineered_fixture(packet):
    """Arithmetic plumbing fixture only: deterministic random fact-word sums."""
    generator = torch.Generator().manual_seed(971)
    words = sorted({w for record in packet['featureRecords'] for w in tokens(record['facts']['name']) + tokens(record['facts']['selectedColor'])})
    word_features = torch.randn(len(words), FEATURE_WIDTH, generator=generator)
    lookup = {w: word_features[i] for i, w in enumerate(words)}
    vectors = []
    for record in packet['featureRecords']:
        vector = sum((lookup[w] for w in tokens(record['facts']['name']) + tokens(record['facts']['selectedColor'])))
        vectors.append(F.normalize(vector, dim=0))
    values = torch.stack(vectors)
    return {'schemaVersion': HEAD_VERSION, 'featureKeys': [r['featureKey'] for r in packet['featureRecords']],
        'textSha256': [r['textSha256'] for r in packet['featureRecords']], 'features': values,
        'featureTensorSha256': tensor_sha(values),
        'provenance': {'evidenceSource': 'engineered-cpu-fixture', 'nativeForwards': 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-smoke', action='store_true', required=True,
                        help='Engineered CPU fixture only; does not load Qwen/MLX or train native candidate')
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from .store_parallel_text_data_v1 import build
    from .store_catalog_v1 import load_snapshot
    packet = build(load_snapshot())
    bundle = engineered_fixture(packet)
    _, report = fit(packet, bundle, steps=args.steps, width=32, fixture=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
