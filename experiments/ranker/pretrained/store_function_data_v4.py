"""CPU-only known-store ranking data with explicit multi-need and empty targets.

No native inference, fitting, runtime or storefront changes. Synthetic observed
profiles contain only inference-visible searches and declared preferences.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .store_catalog_v1 import canonical, load_snapshot
from .store_model_bridge import prepare_current_catalog

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'store_function_v4'
VERSION = 'actual-store-ranking-data.v4'
SOURCE_FILES = ('store_function_data_v4.py', 'test_store_function_data_v4.py',
                'store_catalog_v1.py', 'store_model_bridge.py')

# Generic functions supported by public names/categories/search terms. None of
# these assert performance, availability, demographic traits or image quality.
USES = (
    'a mug to drink from', 'a daypack for my everyday things',
    'field notes to write a thought down', 'a waterproof shell with a hood',
    'a steel bottle for drinking water', 'a canvas tote for grocery shopping',
    'a cotton throw blanket', 'a cordless lamp for a table',
    'a steel lunch box for a meal', 'over-ear headphones to listen to music',
    'canvas low-top sneakers', 'a cotton crewneck sweatshirt',
    'a canvas duffel for travel', 'round sunglasses with acetate frames',
    'a zipped canvas organizer for cables', 'a striped cotton towel for swimming')
SEARCHES = (
    'mug', 'everyday daypack', 'field notes for writing', 'hooded waterproof shell',
    'water bottle', 'grocery tote', 'cotton blanket', 'cordless table lamp',
    'lunch box', 'over-ear headphones', 'low-top canvas shoes', 'cotton sweatshirt',
    'travel duffel', 'round acetate sunglasses', 'cable organizer pouch',
    'cotton swimming towel')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request(query, preferences=(), avoid=(), searches=(), profile_preferences=(),
            profile_avoid=(), budget=None, requirements=()):
    return {'query': query, 'observedProfile': {'recentSearches': list(searches),
        'statedPreferences': list(profile_preferences), 'statedAvoidances': list(profile_avoid)},
        'preferences': list(preferences), 'negativePreferences': list(avoid),
        'itemBudget': budget, 'hardRequirements': deepcopy(list(requirements))}


def validate_request(value):
    keys = {'query', 'observedProfile', 'preferences', 'negativePreferences',
            'itemBudget', 'hardRequirements'}
    if type(value) is not dict or set(value) != keys:
        raise ValueError('Exact public v4 request schema required')
    if type(value['query']) is not str or not value['query'].strip():
        raise ValueError('Nonempty public query required')
    profile = value['observedProfile']
    if type(profile) is not dict or set(profile) != {'recentSearches', 'statedPreferences', 'statedAvoidances'}:
        raise ValueError('Only observed searches and declared preferences are allowed')
    for strings in [value['preferences'], value['negativePreferences'], *profile.values()]:
        if type(strings) is not list or any(type(x) is not str or not x.strip() for x in strings):
            raise ValueError('Public preference/search fields must be string lists')
    budget = value['itemBudget']
    if budget is not None and (type(budget) not in (int, float) or not math.isfinite(budget) or budget < 0):
        raise ValueError('Finite nonnegative item budget required')
    if type(value['hardRequirements']) is not list:
        raise ValueError('Hard requirements must be explicit')
    for r in value['hardRequirements']:
        if (type(r) is not dict or set(r) != {'attribute', 'value'}
                or r['attribute'] not in ('category', 'color', 'stock', 'size')
                or type(r['value']) is not str or not r['value'].strip()):
            raise ValueError('Unsupported hard requirement')
    return value


def request_text(value):
    validate_request(value)
    return ('Encode this public shopping request and observed searches as data.\n'
            + canonical({'schemaVersion': VERSION + '/query', 'request': value,
                         'budgetScope': 'inclusive item price; delivery excluded',
                         'precedence': 'explicit current query/preferences override older searches'}))


def product_text(product):
    if set(product) != {'name', 'category', 'itemPrice', 'searchTerms', 'availableColors'}:
        raise ValueError('Exact public product facts required')
    return 'Encode public store product facts as data.\n' + canonical(product)


def eligible(request_value, product):
    validate_request(request_value)
    if request_value['itemBudget'] is not None and product['itemPrice'] > request_value['itemBudget']:
        return False
    for r in request_value['hardRequirements']:
        a, v = r['attribute'], r['value']
        if a in ('stock', 'size'):
            return False
        if a == 'category' and v != product['category']:
            return False
        if a == 'color' and v not in product['availableColors']:
            return False
    return True


def public_catalog():
    bridge = prepare_current_catalog()
    snapshot = load_snapshot()
    products = [{**{k: deepcopy(p[k]) for k in ('name', 'category', 'itemPrice', 'searchTerms')},
                 'availableColors': [v['colorName'] for v in p['colorVariants']]}
                for p in snapshot['inputs']['products']]
    if len(products) != 16:
        raise ValueError('Expected actual 16-product roster')
    return products, bridge


def build():
    products, bridge = public_catalog()
    cases, labels, pairs = [], {'train': [], 'evaluation': []}, []

    def add(split, family, q, desired, avoided=(), exact=None):
        validate_request(q)
        rid = f'v4-{split}-{sum(c["split"] == split for c in cases):02d}'
        desired, avoided = sorted(desired), sorted(avoided)
        if set(desired) & set(avoided) or any(type(i) is not int or not 0 <= i < 16 for i in desired + avoided):
            raise ValueError('Unsupported or contradictory label')
        allowed = [i for i, p in enumerate(products) if eligible(q, p)]
        relevant = sorted(set(desired) & set(allowed))
        # -1 is an explicit soft avoidance; 0 means unrequested in this synthetic
        # case, never a claim that an unlabelled product cannot be useful.
        grades = [2 if i in relevant else -1 if i in avoided else 0 for i in range(16)]
        cases.append({'caseId': rid, 'split': split, 'family': family, 'request': q})
        labels[split].append({'caseId': rid, 'desiredProductIndices': desired,
            'explicitlyAvoidedProductIndices': avoided, 'eligibleProductIndices': allowed,
            'relevantProductIndices': relevant, 'relevanceByProduct': grades,
            'abstentionRequired': not relevant, 'exactNameTargetIndex': exact,
            'publicSupport': [{'productIndex': i, 'sourceFacts': deepcopy(products[i])}
                              for i in sorted(set(desired + avoided))],
            'rationale': 'Author-written synthetic matching of explicit public needs to '
                'declared names/categories/search terms; hard guards applied independently. '
                'Equal positive needs have equal relevance. No persona truth or unknown specifications.'})
        return rid

    # 32 single-need rows cover each function directly and through a sparse,
    # simulated merchant-observed search. Some profiles carry public declarations.
    framings = ('Help me find an item for this use.', 'Which item matches what I need?',
                'Recommend a product from this shop.', 'Show the closest match to my request.')
    for i, use in enumerate(USES):
        add('train', 'current-single-need', request(framings[i % 4], [f'I need {use}.']), [i])
        add('train', 'sparse-observed-search', request('Use my recent search to suggest an item.',
            searches=[SEARCHES[i]], profile_preferences=[f'I am looking for {use}.'] if i % 4 == 0 else []), [i])

    # A cycle supplies 16 distinct two-product sets. Eval multi-need combinations
    # below use different edges and do not occur in TRAIN.
    for i in range(16):
        j = (i + 1) % 16
        add('train', 'two-simultaneous-needs', request('Find products for both of my needs.',
            [f'I want {USES[i]}.', f'I also need {USES[j]}.']), [i, j])

    negative_rows = (
        (0, 4, 'Choose between a mug and a water bottle.', 'I am not looking for a bottle.'),
        (4, 0, 'Choose between a mug and a water bottle.', 'I am not looking for a mug.'),
        (1, 12, 'Choose between a daypack and a travel duffel.', 'No travel duffel for me.'),
        (12, 1, 'Choose between a daypack and a travel duffel.', 'No daypack for me.'),
        (5, 14, 'Choose between a grocery tote and a cable pouch.', 'Not a small cable pouch.'),
        (14, 5, 'Choose between a grocery tote and a cable pouch.', 'Not a grocery shopping tote.'),
        (6, 15, 'Choose between a throw blanket and a swimming towel.', 'I do not need a swimming towel.'),
        (15, 6, 'Choose between a throw blanket and a swimming towel.', 'I do not need a throw blanket.'))
    for i, j, query, negative in negative_rows:
        add('train', 'negative-only-two-option', request(query, avoid=[negative]), [i], [j])

    for i, previous in ((0, 'water bottle'), (2, 'cable pouch')):
        add('train', 'exact-name-overrides-stale-search', request('Find ' + products[i]['name'] + '.',
            searches=[previous]), [i], exact=i)
    for i, budget, requirements in (
        (0, 24, []), (0, 23, []), (3, 128, [{'attribute': 'color', 'value': 'Forest'}]),
        (3, 127, []), (10, None, [{'attribute': 'size', 'value': 'M'}]),
        (0, None, [{'attribute': 'stock', 'value': 'in-stock'}])):
        add('train', 'hard-constraint-or-empty', request('I need ' + USES[i] + '.',
            budget=budget, requirements=requirements), [i])

    def pair(family, qa, a, qb, b, avoid_a=(), avoid_b=()):
        ca = add('evaluation', family, qa, a, avoid_a)
        cb = add('evaluation', family, qb, b, avoid_b)
        pairs.append({'caseIds': [ca, cb], 'meaning': 'Both disjoint relevant target sets '
                      'must be recovered; a changed top choice alone is insufficient.'})

    pair('profile-counterfactual',
        request('Suggest one item using only my recent search.', searches=['a cup-style mug for drinking']), [0],
        request('Suggest one item using only my recent search.', searches=['a stainless bottle to hold water']), [4])
    pair('profile-counterfactual',
        request('Recommend either a daypack or a duffel using what I said.',
                profile_avoid=['Skip travel duffels.']), [1],
        request('Recommend either a daypack or a duffel using what I said.',
                profile_avoid=['Skip daypacks.']), [12], [12], [1])
    pair('current-preference-counterfactual',
        request('Pick either field notes or a cable organizer after reading my preference.', avoid=['Not a cable organizer.']), [2],
        request('Pick either field notes or a cable organizer after reading my preference.', avoid=['Not field notes.']), [14], [14], [2])
    pair('current-preference-counterfactual',
        request('Choose the clothing item I am asking for.', ['For my feet I want low-top canvas sneakers.'], ['Do not suggest a sweatshirt.']), [10],
        request('Choose the clothing item I am asking for.', ['I want a sweatshirt with a cotton crew neck.'], ['Do not suggest footwear.']), [11], [11], [10])

    multi_pairs = (
        (([0, 7], ['A mug for my drink.', 'A lamp for the table that works without a cord.'], []),
         ([4, 8], ['A water bottle made from steel.', 'A lunch box with a bamboo lid.'], [])),
        (([1, 14], ['A daypack for daily carrying.', 'A zip pouch for cable organization.'], [12]),
         ([5, 12], ['A tote for the grocery shop.', 'A canvas bag for weekend travel.'], [1])),
        (([3, 10], ['A hooded shell declared waterproof.', 'Low-top shoes made of canvas.'], [9]),
         ([9, 13], ['Cushioned headphones that go over the ears.', 'Sunglasses with round acetate frames.'], [3])),
        (([2, 15], ['Field notes for recording an idea.', 'A striped cotton towel for swim days.'], []),
         ([6, 11], ['A throw blanket made of cotton.', 'A crewneck sweatshirt made of cotton.'], [])))
    for a, b in multi_pairs:
        def multi(row):
            ids, needs, avoids = row
            return request('Help me cover these two separate needs.', needs,
                           ['Please skip ' + products[i]['name'] + '.' for i in avoids])
        pair('heldout-two-need-combination', multi(a), a[0], multi(b), b[0], a[2], b[2])

    for i, previous in ((9, 'travel backpack'), (14, 'notebooks'), (12, 'daily backpack')):
        add('evaluation', 'exact-name-overrides-stale-search',
            request('Please find the item named "' + products[i]['name'] + '".', searches=[previous]), [i], exact=i)
    for i, budget, requirements in (
        (4, 36, []), (4, 35, []), (14, None, [{'attribute': 'color', 'value': 'Navy'}]),
        (10, None, [{'attribute': 'size', 'value': '9'}]),
        (0, None, [{'attribute': 'stock', 'value': 'available-now'}])):
        add('evaluation', 'hard-constraint-or-empty',
            request('Show a matching item for this use.', ['My need is ' + USES[i] + '.'],
                    budget=budget, requirements=requirements), [i])

    packet = {'version': VERSION, 'products': products, 'publicCases': cases,
              'trainLabels': labels['train'], 'evaluationLabels': labels['evaluation'],
              'preferencePairs': pairs, 'bridge': {k: deepcopy(bridge[k]) for k in
              ('modelInputSha256', 'sourceReceipts', 'currentSourcesMatchPinnedInputs')}}
    audit(packet)
    return packet


def audit(packet):
    cases = packet['publicCases'];labels = packet['trainLabels'] + packet['evaluationLabels']
    products = packet['products'];byid = {x['caseId']: x for x in cases};truth = {x['caseId']: x for x in labels}
    if len(cases) != 88 or len(packet['trainLabels']) != 64 or len(packet['evaluationLabels']) != 24:
        raise ValueError('Fixed 64 TRAIN/24 evaluation cases required')
    if len(byid) != 88 or set(byid) != set(truth) or len({request_text(c['request']) for c in cases}) != 88:
        raise ValueError('Unique public requests and external label membership required')
    for c in cases:
        r = truth[c['caseId']];q = c['request'];validate_request(q)
        expected_split = packet['trainLabels'] if c['split'] == 'train' else packet['evaluationLabels']
        if r not in expected_split:
            raise ValueError('Labels must remain in their declared split')
        if len(r['relevanceByProduct']) != 16 or r['eligibleProductIndices'] != [i for i,p in enumerate(products) if eligible(q,p)]:
            raise ValueError('Full roster labels/independent guard eligibility required')
        relevant = sorted(set(r['desiredProductIndices']) & set(r['eligibleProductIndices']))
        if r['relevantProductIndices'] != relevant or r['abstentionRequired'] != (not relevant):
            raise ValueError('Hard-ineligible desired products cannot become positive labels')
        if set(r['desiredProductIndices']) & set(r['explicitlyAvoidedProductIndices']):
            raise ValueError('Contradictory desired/avoided labels')
        for field in ('desiredProductIndices', 'explicitlyAvoidedProductIndices'):
            if r[field] != sorted(set(r[field])) or any(type(i) is not int or not 0 <= i < 16 for i in r[field]):
                raise ValueError('Unique full-roster desired and avoided indices required')
        grades = [2 if i in relevant else -1 if i in r['explicitlyAvoidedProductIndices'] else 0 for i in range(16)]
        if r['relevanceByProduct'] != grades:
            raise ValueError('External full-roster relevance grades disagree with targets')
        support = [{'productIndex': i, 'sourceFacts': products[i]} for i in
                   sorted(set(r['desiredProductIndices'] + r['explicitlyAvoidedProductIndices']))]
        if r['publicSupport'] != support:
            raise ValueError('Label support must be literal public catalog facts')
        if r['exactNameTargetIndex'] is not None and products[r['exactNameTargetIndex']]['name'] not in q['query']:
            raise ValueError('Exact-name label requires literal current query evidence')
        for forbidden in ('caseId', 'split', 'family', 'targetProduct', 'desiredProduct', 'publicSupport', 'personaTruth', 'rationale'):
            if forbidden in request_text(q):
                raise ValueError('Private label/metadata crossed encoder boundary')
    train_sets = {tuple(x['relevantProductIndices']) for x in packet['trainLabels'] if len(x['relevantProductIndices']) == 2}
    eval_sets = {tuple(x['relevantProductIndices']) for x in packet['evaluationLabels'] if len(x['relevantProductIndices']) == 2}
    if len(train_sets) != 16 or len(eval_sets) != 8 or train_sets & eval_sets:
        raise ValueError('Heldout multi-need target sets must be distinct and unseen in TRAIN')
    paired = []
    for pair in packet['preferencePairs']:
        a,b = pair['caseIds'];qa,qb = byid[a]['request'],byid[b]['request']
        if (qa['query'] != qb['query'] or qa['itemBudget'] != qb['itemBudget']
                or qa['hardRequirements'] != qb['hardRequirements']
                or not truth[a]['relevantProductIndices'] or not truth[b]['relevantProductIndices']
                or set(truth[a]['relevantProductIndices']) & set(truth[b]['relevantProductIndices'])):
            raise ValueError('Counterfactual pair must hold task/guards fixed and target disjoint sets')
        paired.extend([a,b])
    if len(paired) != len(set(paired)) or len(paired) != 16:
        raise ValueError('Eight disjoint evaluation preference pairs required')
    counts = {split: {'cases': sum(c['split'] == split for c in cases),
        'multiNeed': sum(len(x['relevantProductIndices']) == 2 for x in packet[key]),
        'emptyTarget': sum(x['abstentionRequired'] for x in packet[key]),
        'families': dict(Counter(c['family'] for c in cases if c['split'] == split))}
        for split,key in (('train','trainLabels'),('evaluation','evaluationLabels'))}
    return {'status': 'cpu-data-contracts-passed', 'counts': counts,
            'preferencePairs': 8, 'trainMultiNeedSets': sorted(map(list,train_sets)),
            'evaluationMultiNeedSets': sorted(map(list,eval_sets)),
            'multiNeedSetOverlap': [], 'nativeForwards': 0, 'optimizerUpdates': 0,
            'labelProvenance': 'Author-written synthetic relevance, traceable to actual public product facts; '
                'no teacher calls, customer logs, hidden personas or inferred demographic traits.',
            'evaluationStatus': 'New author-written wording, not model-scored in v4; visible to data author '
                'and reviewers, not a sealed benchmark. All identities/functions TRAIN-seen.',
            'v3Status': 'Inspected v3 evaluation is DEVELOPMENT REPLAY. It is not included or renamed as fresh v4 evaluation.'}


def evaluate_rankings(rankings, public_cases, external_labels, pairs, k=6):
    """CPU evaluator. Targets never enter model inputs; no quality without outputs.

    A ranking may contain 0..k unique eligible products. Abstention is []; exact
    target-set recovery uses the first |relevant targets| outputs, not extra cards.
    """
    if type(k) is not int or k < 1:
        raise ValueError('Positive integer cutoff required')
    truth = {x['caseId']: x for x in external_labels};cases = {x['caseId']: x for x in public_cases}
    if set(rankings) != set(truth) or set(cases) != set(truth):
        raise ValueError('Fixed complete case denominator required')
    rows = {}
    for rid, chosen in rankings.items():
        t = truth[rid];target = set(t['relevantProductIndices'])
        if (type(chosen) is not list or len(chosen) > k or any(type(i) is not int or not 0 <= i < 16 for i in chosen)
                or len(set(chosen)) != len(chosen)):
            raise ValueError('Unique bounded full-roster indices required')
        safe = all(i in t['eligibleProductIndices'] for i in chosen)
        topn = chosen[:len(target)]
        rows[rid] = {'safe': safe, 'emptyTarget': not target,
            'hitAtOne': bool(target and chosen and chosen[0] in target),
            'allTargetsAtTheirCount': bool(target and set(topn) == target and safe),
            'relevantRetrieved': len(target & set(chosen)), 'relevantCount': len(target),
            'explicitAvoidanceInOutput': bool(set(chosen) & set(t['explicitlyAvoidedProductIndices'])),
            'correctEmptyAbstention': bool(not target and not chosen)}
    valid_pairs = [p['caseIds'] for p in pairs if all(rid in rows for rid in p['caseIds'])]
    nonempty = [r for r in rows.values() if not r['emptyTarget']];empty = [r for r in rows.values() if r['emptyTarget']]
    return {'fixedCases': len(rows), 'nonemptyCases': len(nonempty), 'emptyCases': len(empty),
        'hitAtOne': sum(r['hitAtOne'] and r['safe'] for r in nonempty),
        'allTargetsAtTheirCount': sum(r['allTargetsAtTheirCount'] for r in nonempty),
        'relevantRetrievedAtK': sum(r['relevantRetrieved'] for r in rows.values()),
        'relevantTargetDenominator': sum(r['relevantCount'] for r in rows.values()),
        'correctEmptyAbstentions': sum(r['correctEmptyAbstention'] for r in empty),
        'safeCases': sum(r['safe'] for r in rows.values()),
        'casesWithExplicitAvoidanceInOutput': sum(r['explicitAvoidanceInOutput'] for r in rows.values()),
        'fixedPreferencePairs': len(valid_pairs),
        'pairsWithBothTargetSetsRecovered': sum(all(rows[x]['allTargetsAtTheirCount'] for x in pair) for pair in valid_pairs),
        'rows': rows}


def freeze(out=DATA):
    packet = build();out = Path(out);out.mkdir(parents=True,exist_ok=False)
    files = {'catalog.json': packet['products'], 'public-cases.json': packet['publicCases'],
             'train-labels.json': packet['trainLabels'], 'evaluation-labels.json': packet['evaluationLabels'],
             'preference-pairs.json': packet['preferencePairs'], 'AUDIT.json': audit(packet)}
    for name,value in files.items():
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    receipt = {'version': VERSION, 'files': {n: sha(out/n) for n in files},
        'sourceHashes': {n: sha(ROOT/n) for n in SOURCE_FILES}, 'bridge': packet['bridge'],
        'counts': files['AUDIT.json']['counts'], 'nativeForwards': 0, 'optimizerUpdates': 0,
        'promotionStatus': 'CPU dataset only; no learned quality, latency or integration claim',
        'trainingBoundary': 'Only TRAIN labels may fit model parameters, calibration or thresholds. '
            'Do not use evaluation labels, targetsets, rationale, support or case metadata as predictive text.',
        'architecture': 'Current v3 single-label softmax cannot directly consume multi-label and empty targets. '
            'Use independent sigmoid relevance or a query-product compatibility scorer with multi-positive '
            'supervision and TRAIN-only no-match calibration; keep deterministic guards and exact-name routing. '
            'Case grades 0 are unrequested, not universal irrelevance; hard negatives are explicit avoidances.',
        'limitations': ['Synthetic observed searches and declarations, not real shopper histories.',
            'No hidden persona truth is generated. Do not infer age, gender, profession or tastes from a view.',
            'Public queries override stale observed searches; exact-name behavior remains explicit.',
            'All product identities and functions known; only request wording and multi-need combinations held out.',
            'Item budgets exclude delivery; size and stock unknown and therefore abstain when required.',
            'No image labels, adapted copy, conversion outcomes or browser timing.',
            'New wording is author-visible, not sealed or externally human-validated; v3 is inspected development replay.']}
    (out/'FREEZE.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return receipt


def load_frozen(out=DATA):
    """Reject source/data drift before a future runner consumes this package."""
    out = Path(out);receipt = json.loads((out/'FREEZE.json').read_text())
    if receipt['version'] != VERSION or receipt['sourceHashes'] != {n:sha(ROOT/n) for n in SOURCE_FILES}:
        raise ValueError('Frozen data sources changed')
    for name, expected in receipt['files'].items():
        if sha(out/name) != expected:
            raise ValueError('Frozen artifact changed: ' + name)
    read = lambda name: json.loads((out/name).read_text())
    packet = {'version': VERSION, 'products': read('catalog.json'), 'publicCases': read('public-cases.json'),
              'trainLabels': read('train-labels.json'), 'evaluationLabels': read('evaluation-labels.json'),
              'preferencePairs': read('preference-pairs.json'), 'bridge': receipt['bridge']}
    audit(packet)
    return packet


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, default=DATA)
    args = parser.parse_args()
    print(canonical(freeze(args.out)['counts']))
