"""Stdlib-only validation/projections for a fixed ten-pair shopper packet.

The shopper projection intentionally reads no evaluator gold or catalog file.
This module neither launches shoppers nor scores actual browser sessions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

DATA = Path(__file__).resolve().parent
REPO = DATA.parents[3]
VERSION = 'actual16-paired-shopper-packet.v1'
CONTRACT = 'shopper-session.v1'
FORBIDDEN = {'productId', 'productIds', 'variantId', 'variantIds', 'catalogIndex',
             'gold', 'evaluatorGold', 'acceptableProductIds', 'acceptableVariantIds',
             'candidateFunctionProductIds', 'expectedProductIds', 'correctSelector',
             'targetSelector', 'targetProductIndex', 'personaFamilyId', 'hiddenPersona'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(name, directory=DATA):
    return json.loads((Path(directory) / name).read_text())


def _exact(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError('Unexpected/missing packet fields')


def assert_no_gold(value):
    if type(value) is dict:
        if set(value) & FORBIDDEN:
            raise ValueError('Evaluator identities/labels may not enter shopper or public inputs')
        for item in value.values():
            assert_no_gold(item)
    elif type(value) is list:
        for item in value:
            assert_no_gold(item)


def validate_public(value):
    _exact(value, ('query', 'itemBudget', 'hardRequirements'))
    if type(value['query']) is not str or not value['query'].strip() or len(value['query']) > 200:
        raise ValueError('Bounded nonempty public query required')
    # Browser v1 enters only the public query. Constraints are deliberately
    # written into it; there is no hidden service-only arm advantage.
    if value['itemBudget'] is not None or value['hardRequirements'] != []:
        raise ValueError('Version1 browser context uses query only, no typed-only arm advantage')
    assert_no_gold(value)
    return value


def validate_private(value):
    _exact(value, ('goal', 'hardConstraints', 'preferences'))
    if type(value['goal']) is not str or not value['goal'].strip():
        raise ValueError('Private shopper goal required')
    ids = []
    for kind in ('hardConstraints', 'preferences'):
        if type(value[kind]) is not list:
            raise ValueError('Typed needs required')
        for need in value[kind]:
            _exact(need, ('id', 'label', 'attribute', 'operator', 'value'))
            if not all(type(need[k]) is str and need[k].strip() for k in ('id', 'label', 'attribute')):
                raise ValueError('Explicit need identity/label/attribute required')
            if need['operator'] not in ('equals', 'includes', 'at-most', 'less-than'):
                raise ValueError('Unsupported need operator')
            if need['attribute'].lower().replace('-', '').replace('_', '') in ('productid', 'variantid', 'catalogindex', 'gold', 'expectedproduct'):
                raise ValueError('Need attributes must not encode answer selectors')
            if need['operator'] in ('at-most', 'less-than') and type(need['value']) not in (int, float):
                raise ValueError('Numeric budget required')
            ids.append(need['id'])
    if len(set(ids)) != len(ids):
        raise ValueError('Need identities must be unique')
    assert_no_gold(value)
    return value


def _index(rows, key='caseId'):
    result = {r[key]: r for r in rows}
    if len(rows) != len(result):
        raise ValueError('Duplicate scenario identity')
    return result


def shopper_case(case_id, directory=DATA):
    """Only public/private/plan files enter the framework-facing case."""
    public = _index(read('public-observations.json', directory))[case_id]
    private = _index(read('shopper-instructions.json', directory))[case_id]
    plan = _index(read('pair-plan.json', directory)['schedule'])[case_id]
    _exact(public, ('caseId', 'observableContext'))
    _exact(private, ('caseId', 'privateShopperNeeds'))
    return {'contractVersion': CONTRACT, 'caseId': case_id, 'split': plan['split'], 'seed': plan['seed'],
            'armOrder': deepcopy(plan['armOrder']), 'privateShopperNeeds': deepcopy(validate_private(private['privateShopperNeeds'])),
            'observableContext': deepcopy(validate_public(public['observableContext']))}


def recommender_payload(case_id, directory=DATA):
    row = _index(read('public-observations.json', directory))[case_id]
    return deepcopy(validate_public(row['observableContext']))


def shopper_assignment(case_id, arm, directory=DATA, base_url='http://127.0.0.1:4312/'):
    if arm not in ('generic', 'local-model'):
        raise ValueError('Unknown paired arm')
    parsed = urlsplit(base_url)
    if base_url not in ('http://127.0.0.1:4312/', 'http://shop.dubhacks.localhost:1356/') or parsed.query or parsed.fragment:
        raise ValueError('Reviewed identical local base URL required')
    case = shopper_case(case_id, directory)
    assignment = {'contractVersion': CONTRACT, 'caseId': case_id, 'arm': arm,
                  'entryUrl': base_url + ('?model=local' if arm == 'local-model' else ''),
                  'shopperNeeds': case['privateShopperNeeds'], 'recommenderContext': case['observableContext'],
                  'instructions': read('pair-plan.json', directory)['shopperInstructions']}
    assert_no_gold(assignment)
    return assignment


def _need_status(need, product, variant=None):
    facts = {**product['facts'], 'itemPrice': product['itemPrice'], 'originalName': product['originalName']}
    if variant:
        facts['selectedColor'] = variant['selectedColor']
    if need['attribute'] not in facts:
        return 'unknown'
    actual, wanted = facts[need['attribute']], need['value']
    if need['operator'] == 'equals':
        return 'met' if actual == wanted else 'not-met'
    if need['operator'] == 'includes' and type(actual) is list:
        return 'met' if wanted in actual else 'not-met'
    if type(actual) in (int, float) and type(wanted) in (int, float):
        return 'met' if (actual <= wanted if need['operator'] == 'at-most' else actual < wanted) else 'not-met'
    return 'unknown'


def expected_variants(case, gold, catalog):
    variants, unknowns = [], set()
    for product in catalog['products']:
        if product['productId'] not in gold['candidateFunctionProductIds']:
            continue
        for variant in product['variants']:
            statuses = [_need_status(n, product, variant) for n in case['privateShopperNeeds']['hardConstraints']]
            if all(s == 'met' for s in statuses):
                variants.append((product['productId'], variant['variantId']))
            for need, status in zip(case['privateShopperNeeds']['hardConstraints'], statuses, strict=True):
                if status == 'unknown':
                    unknowns.add(need['attribute'])
    return variants, sorted(unknowns)


def validate_packet(directory=DATA, *, verify_live=False):
    directory = Path(directory)
    manifest = read('manifest.json', directory)
    if manifest['version'] != VERSION:
        raise ValueError('Wrong packet version')
    for name, expected in manifest['files'].items():
        if sha(directory / name) != expected:
            raise ValueError('Frozen packet file changed: ' + name)
    public = read('public-observations.json', directory)
    private = read('shopper-instructions.json', directory)
    plan = read('pair-plan.json', directory)
    gold = _index(read('evaluator-gold.json', directory))
    binding = read('catalog-binding.json', directory)
    catalog = binding['catalogSnapshot']
    ids = [row['caseId'] for row in public]
    if len(ids) != 10 or set(ids) != set(gold) or set(ids) != set(_index(private)) or set(ids) != set(_index(plan['schedule'])):
        raise ValueError('Fixed ten-case membership required')
    if len(catalog['products']) != 16 or sum(len(p['variants']) for p in catalog['products']) != 48 or catalog['sha256'] != digest({'products': catalog['products']}):
        raise ValueError('Exact16/48 independently bound catalog required')
    orders = [row['armOrder'] for row in plan['schedule']]
    if orders.count(['generic', 'local-model']) != 5 or orders.count(['local-model', 'generic']) != 5:
        raise ValueError('Predeclared five AB/five BA orders required')
    runs = [r for p in plan['schedule'] for r in p['runs']]
    if len(runs) != 20 or len({r['runId'] for r in runs}) != 20 or any(r['initialStatus'] != 'unrun' for r in runs):
        raise ValueError('Twenty unique planned smoke runs required; results live separately')
    if plan['budget']['maxActions'] != 24 or plan['budget']['maxWallSeconds'] != 120:
        raise ValueError('Fixed budget changed')
    if plan['isolation']['freshTabRequiredTruth'] != {'sharedStorage': True, 'storageCleared': False}:
        raise ValueError('Fresh tabs must not claim storage isolation')
    statuses = []
    for case_id in ids:
        case = shopper_case(case_id, directory)
        variants, unknowns = expected_variants(case, gold[case_id], catalog)
        if sorted(v for _, v in variants) != sorted(gold[case_id]['acceptableVariantIds']) or sorted({p for p, _ in variants}) != sorted(gold[case_id]['acceptableProductIds']):
            raise ValueError('Gold contradicts declared functions/hard facts: ' + case_id)
        status = 'satisfiable' if variants else 'unverifiable-required-fact' if unknowns else 'known-empty'
        if status != gold[case_id]['eligibilityStatus'] or unknowns != gold[case_id]['unknownRequiredAttributes']:
            raise ValueError('Empty/unknown distinction changed: ' + case_id)
        a, b = shopper_assignment(case_id, 'generic', directory), shopper_assignment(case_id, 'local-model', directory)
        if {k: v for k, v in a.items() if k not in ('arm', 'entryUrl')} != {k: v for k, v in b.items() if k not in ('arm', 'entryUrl')}:
            raise ValueError('Paired assignments differ beyond model mode')
        statuses.append(status)
    if statuses.count('satisfiable') != 8 or statuses.count('known-empty') != 1 or statuses.count('unverifiable-required-fact') != 1:
        raise ValueError('Fixed satisfiable/abstention coverage required')
    # Source/public bindings are checked without invoking a model or browser.
    source_data = directory.parent / 'store_catalog_v1'
    for name, expected in binding['sourceDataFileSha256'].items():
        if sha(source_data / name) != expected:
            raise ValueError('Source catalog receipt changed')
    if verify_live:
        from ..store_demo_catalog import runtime_export
        if digest(runtime_export()) != binding['rawExportSha256']:
            raise ValueError('Active runtime public catalog differs')
        for name, expected in binding['liveInputSourceHashes'].items():
            if sha(REPO / name) != expected:
                raise ValueError('Active input source changed: ' + name)
    return {'status': 'CPU packet valid', 'pairs': 10, 'plannedRuns': 20, 'executedRunsClaimed': 0,
            'satisfiableCases': 8, 'knownEmpty': 1, 'unknownRequiredFact': 1,
            'livePublicCatalogVerified': verify_live, 'nativeExecution': False, 'browserExecution': False}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--directory', type=Path, default=DATA)
    p.add_argument('--verify-live', action='store_true')
    args = p.parse_args()
    print(canonical(validate_packet(args.directory, verify_live=args.verify_live)))


if __name__ == '__main__':
    main()
