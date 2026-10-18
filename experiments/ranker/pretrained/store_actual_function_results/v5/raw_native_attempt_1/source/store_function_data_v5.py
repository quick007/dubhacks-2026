"""Known16-store paraphrase augmentation; CPU only, no hidden profile inference.

Keep the exact v3/live generic-query plus whole-utterance preference encoding.
All sixteen identities are supervised. Evaluation wording is independently
written and frozen before extraction; no new-catalog/human generalization claim.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .store_catalog_v1 import canonical, load_snapshot
from .store_demo_engine import checked_request, guarded_order, ranking_input
from .store_function_adaptation import request_text, validate_request, product_text

HERE = Path(__file__).resolve().parent
VERSION = 'actual-store-function-data.v5'
DATA = HERE / 'store_function_v5'
EVAL = HERE / 'store_function_v5_eval_seed_v1.json'
PRIOR = Path('data/ranker/pretrained/actual-store-function-v3-attempt-1')
# Four distinct utterances/function, selected without consulting fresh eval.
# Generic functional synonyms are training labels, never catalog specification
# additions: notebook=field notes, daily backpack=daypack, towel vs blanket etc.
PHRASES = (
 ('I need a cup-shaped mug for a drink.', 'A mug that I can sip my drink from.',
  'Looking for a mug to use at breakfast.', 'A reusable mug for sipping a beverage.'),
 ('I need a backpack for everyday errands.', 'A bag worn on my back for my daily belongings.',
  'Show a day bag with backpack straps.', 'Looking for a small daily backpack rather than trip luggage.'),
 ('I need a notebook to jot things down.', 'A paper notebook for handwritten reminders.',
  'Some pocket stationery to record my thoughts.', 'Looking for a small writing pad for notes.'),
 ('I need a hooded jacket for rainy weather.', 'A waterproof outer shell to wear in rain.',
  'Show a raincoat with a hood.', 'Looking for rainwear, specifically a waterproof jacket.'),
 ('I need a water container I can drink from.', 'A steel drinking bottle for water.',
  'Show a refillable bottle for my drinking water.', 'Looking for a stainless bottle, not a lunch container.'),
 ('I need an open carry bag for groceries.', 'A shopping tote for carrying market purchases.',
  'Show a canvas grocery carryall.', 'Looking for a tote bag to take to the shops.'),
 ('I need a blanket to drape over my sofa.', 'A cotton throw for a cozy seat.',
  'Show a woven blanket for sitting on at a picnic.', 'Looking for a throw blanket rather than a towel.'),
 ('I need a lamp that sits on a desk.', 'A cordless light for a bedside table.',
  'Show portable lighting for reading at a table.', 'Looking for a table lamp without a cord.'),
 ('I need a container for bringing my lunch.', 'A lunchbox for a packed meal.',
  'Show a food box with the stated bamboo lid.', 'Looking for a steel meal container rather than a water bottle.'),
 ('I need something to wear over my ears for music.', 'Headphones for listening to songs.',
  'Show cushioned over-ear audio headphones.', 'Looking for ear-covering headphones to listen with.'),
 ('I need canvas footwear for walking.', 'Low-cut sneakers to wear on my feet.',
  'Show a pair of everyday canvas shoes.', 'Looking for low-top trainers rather than a sweatshirt.'),
 ('I need a sweatshirt with a round crew neck.', 'A cotton sweater-style layer for my torso.',
  'Show a crewneck top made from cotton.', 'Looking for a pullover sweatshirt, not footwear.'),
 ('I need a luggage bag for a short trip.', 'A duffel to pack for a weekend away.',
  'Show a canvas holdall for travel.', 'Looking for a weekender travel bag rather than a daily backpack.'),
 ('I need round eyewear for sunny days.', 'A pair of sunglasses with acetate frames.',
  'Show circular-shaped sunglasses.', 'Looking for sun glasses to wear on my face.'),
 ('I need a pouch to organize charging cables.', 'A zippered organizer for small loose essentials.',
  'Show a small canvas case for my cords.', 'Looking for a zipped cable pouch rather than paper stationery.'),
 ('I need a towel to dry off after a swim.', 'A cotton towel to bring to the beach.',
  'Show a striped drying towel.', 'Looking for a swimming towel rather than a cozy throw blanket.'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def canonical_request(utterance):
    request, _ = checked_request({'query': utterance})
    return ranking_input(request)


def build(eval_seed=EVAL, prior=PRIOR):
    snapshot = load_snapshot(); products = snapshot['inputs']['products']; prior = Path(prior)
    old = json.loads((prior / 'report.json').read_text())
    source_cases = HERE / 'store_function_v3/public-cases.json'
    source_labels = HERE / 'store_function_v3/train-labels.json'
    if sha(source_cases) != old['publicCasesSha256']:
        raise ValueError('Original v3 public query archive changed')
    original = [r for r in json.loads(source_cases.read_text()) if r['split'] == 'train']
    old_labels = json.loads(source_labels.read_text())
    target_by_id = {r['caseId']: r['targetProductIndex'] for r in old_labels}
    old_targets_sha = hashlib.sha256(canonical([target_by_id[r['caseId']] for r in original]).encode()).hexdigest()
    if old_targets_sha != old['heads']['learned']['metadata']['trainTargetsSha256']:
        raise ValueError('Original32 TRAIN targets differ from checkpoint provenance')
    if any(r['publicSupport'] != {k: products[r['targetProductIndex']][k] for k in ('name', 'category', 'searchTerms')} for r in old_labels):
        raise ValueError('Original public-support labels differ from current catalog')
    if len(original) != 32 or len(old_labels) != 32:
        raise ValueError('Exactly32 original TRAIN queries required')
    cases = deepcopy(original); labels = deepcopy(old_labels)
    for index, phrases in enumerate(PHRASES):
        for offset, phrase in enumerate(phrases):
            rid = f'v5-train-{index:02d}-{offset}'
            cases.append({'caseId': rid, 'split': 'train', 'request': canonical_request(phrase)})
            labels.append({'caseId': rid, 'targetProductIndex': index,
                'publicSupport': {k: deepcopy(products[index][k]) for k in ('name', 'category', 'searchTerms')},
                'labelScope': 'known public product function; synonym interpretation, no invented specifications or hidden persona'})
    evaluation = json.loads(Path(eval_seed).read_text())
    if type(evaluation) is not list or len(evaluation) != 32:
        raise ValueError('Independent32-query frozen evaluation seed required')
    eval_labels = []
    for row in evaluation:
        validate_request(row['request']); index = row['productIndex']
        if type(index) is not int or not 0 <= index < 16:
            raise ValueError('Actual catalog evaluation target required')
        support = {k: products[index][k] for k in ('name', 'category', 'searchTerms')}
        if row['publicSupport'] != support or row['request']['query'] != 'Choose a product for my stated use.':
            raise ValueError('Evaluation must use actual public facts and live query format')
        if len(row['request']['preferences']) != 1 or row['request']['itemBudget'] is not None or row['request']['hardRequirements']:
            raise ValueError('Unconstrained whole-utterance eval only; guards are separate')
        cases.append({'caseId': row['caseId'], 'split': 'evaluation', 'request': deepcopy(row['request'])})
        eval_labels.append({'caseId': row['caseId'], 'targetProductIndex': index,
            'publicSupport': support, 'rationale': row['rationale']})
    if (len(cases) != 128 or len({r['caseId'] for r in cases}) != 128 or
            len({request_text(r['request']) for r in cases}) != 128 or
            Counter(r['targetProductIndex'] for r in labels) != Counter({i: 6 for i in range(16)}) or
            Counter(r['targetProductIndex'] for r in eval_labels) != Counter({i: 2 for i in range(16)})):
        raise ValueError('Balanced96 TRAIN/32 disjoint evaluation roster required')
    for row in cases:
        validate_request(row['request'])
        if set(row['request']) != {'query', 'preferences', 'itemBudget', 'hardRequirements'}:
            raise ValueError('Only live inference-visible v3 request fields allowed')
    return {'version': VERSION, 'products': products, 'publicCases': cases, 'trainLabels': labels,
        'evaluationLabels': eval_labels, 'reuseTrainIds': [r['caseId'] for r in original],
        'evalSeedSha256': sha(eval_seed), 'v3ReportSha256': sha(prior / 'report.json'),
        'v3TrainLabelsSha256': sha(source_labels), 'v3PublicCasesSha256': sha(source_cases)}


def guard_probes(products):
    # Guard/identity policy is deterministic, separately reported from rank quality.
    probes = []
    for i in (0, 2, 4, 8):
        name, price = products[i]['name'], products[i]['itemPrice']
        probes.extend([
            {'caseId': f'inclusive-{i}', 'request': {'query': name, 'itemBudget': price}, 'expectedFirst': i},
            {'caseId': f'exclusive-{i}', 'request': {'query': name + f' under ${price}'}, 'expectedEmpty': True}])
    probes.extend([
        {'caseId': 'stock-unknown', 'request': {'query': 'a notebook in stock'}, 'expectedEmpty': True},
        {'caseId': 'size-unknown', 'request': {'query': 'sneakers', 'hardRequirements': [{'attribute': 'size', 'value': 'M'}]}, 'expectedEmpty': True},
        {'caseId': 'unknown-color', 'request': {'query': 'browse', 'hardRequirements': [{'attribute': 'color', 'value': 'Unlisted violet'}]}, 'expectedEmpty': True},
        {'caseId': 'contradictory-colors', 'request': {'query': 'browse', 'hardRequirements': [{'attribute': 'color', 'value': 'Sage'}, {'attribute': 'color', 'value': 'Ivory'}]}, 'expectedEmpty': True},
        {'caseId': 'price-one', 'request': {'query': 'browse', 'itemBudget': 1}, 'expectedEmpty': True},
        {'caseId': 'known-color', 'request': {'query': 'Trail mug', 'hardRequirements': [{'attribute': 'color', 'value': 'Cobalt'}]}, 'expectedFirst': 0, 'expectedColor': 'Cobalt'},
        {'caseId': 'delivery-budget-unsupported', 'request': {'query': 'a mug delivered under $30'}, 'expectedRequestError': True},
        {'caseId': 'exact-notebook', 'request': {'query': products[2]['name']}, 'expectedFirst': 2}])
    return probes


def freeze(directory=DATA, eval_seed=EVAL, prior=PRIOR):
    packet = build(eval_seed, prior); directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    files = {'catalog.json': packet['products'], 'public-cases.json': packet['publicCases'],
        'train-labels.json': packet['trainLabels'], 'evaluation-labels.json': packet['evaluationLabels'],
        'guard-probes.json': guard_probes(packet['products']),
        'known-language-gaps.json': [{'caseId': 'including-delivery-alias', 'request': {'query': 'a mug under $30 including delivery'},
            'expectedRequestError': True, 'scope': 'documented existing parser limitation, not primary16 supported guard controls'}]}
    for name, value in files.items():
        dump(directory / name, value)
    receipt = {'version': VERSION, 'files': {name: sha(directory / name) for name in files},
        'builderSha256': sha(__file__), 'counts': {'products': 16, 'trainQueries': 96, 'reusedTrainQueries': 32,
            'newTrainQueries': 64, 'evaluationQueries': 32, 'reusedCatalogFeatures': 16, 'newNativeForwards': 96,
            'totalFeatureVectors': 144, 'guardProbes': 16}, 'reuseTrainIds': packet['reuseTrainIds'],
        **{k: packet[k] for k in ('evalSeedSha256', 'v3ReportSha256', 'v3TrainLabelsSha256', 'v3PublicCasesSha256')},
        'format': 'exact live v3 generic query plus whole shopper utterance in preferences; no serialization change',
        'selection': 'one fixed80-step TRAIN-only fit; evaluation labels opened only after predictions freeze',
        'minimumReviewGoals': {'freshTopOneAtLeast': '28/32', 'freshTargetInTopSix': '32/32',
            'perFunctionTopOneAtLeast': '1/2 for every function', 'guardChecks': '16/16',
            'controlGap': 'learned top1 exceeds prior/random/permuted-query; compare exact same new eval with frozen v3',
            'latency': 'fresh rank/action payload median <=300ms on same M5; excludes copy/browser; no guarantee'},
        'limitations': ['All16 product identities and functions are TRAIN seen; synthetic known-catalog paraphrases.',
            'Eval wording written separately before extraction; author knows public catalog, not a sealed human sample.',
            'Single-function softmax only; no learned multi-need, profile history, negative preference or OOD abstention claim.',
            'Deterministic item-price/color/unknown-stock/size guards separately tested. Known parser gap: including delivery is not recognized, unlike delivered/including shipping; separately reported, no general delivered-cost NLP claim.',
            'Observed notebook/browser failures motivate data breadth but are not new unseen evaluation.'],
        'nativeExecuted': False, 'trainingExecuted': False}
    dump(directory / 'FREEZE.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, default=DATA)
    parser.add_argument('--eval-seed', type=Path, default=EVAL); parser.add_argument('--prior', type=Path, default=PRIOR)
    args = parser.parse_args(); print(canonical(freeze(args.out, args.eval_seed, args.prior)['counts']))
