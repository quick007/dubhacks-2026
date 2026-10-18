"""CPU-only deterministic paired supervision; never load a tokenizer or model."""
from __future__ import annotations
import argparse
import copy
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[2]
OLD = ROOT / 'semantic_copy_data'
SEED = 19075
VERSION = 'semantic-copy-pairs-v1'
TAGS = ('compact', 'durable', 'organizer', 'lightweight', 'roomy', 'water-resistant',
        'packable', 'padded', 'reflective', 'foldable', 'ventilated', 'adjustable',
        'structured', 'compressible', 'rolltop', 'expandable', 'mesh', 'quilted')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def source_inventory():
    """Explicit inventory of previously inspected catalog/request fixtures."""
    sources = [OLD / f'cases.{s}.jsonl' for s in ('train', 'dev', 'test')]
    sources += [ROOT / 'SCHEDULER_IDENTITY_CASES.json',
                REPO / 'data/ranker/datasets/v2/catalog.json']
    sources += sorted((REPO / 'data/ranker/datasets/v2/public').glob('*.jsonl'))
    names = set(); products = set(); variants = set(); triples = set(); prefs = set()

    def visit(value):
        if isinstance(value, list):
            for item in value: visit(item)
        elif isinstance(value, dict):
            if all(k in value for k in ('id', 'title', 'tags', 'presentationVariants')):
                names.add(value['title'].split()[0].lower()); products.add(value['id'])
                triples.add(tuple(sorted(value['tags'])))
                variants.update(v['id'] for v in value['presentationVariants'])
            if 'statedPreferences' in value:
                prefs.add(tuple(sorted(value['statedPreferences'])))
            for nested in value.values(): visit(nested)

    for path in sources:
        if path.suffix == '.jsonl': visit(records(path))
        else: visit(json.loads(path.read_text()))
    return {'sourceSha256': {str(p.relative_to(REPO)): sha(p) for p in sources},
        'names': sorted(names), 'productIds': sorted(products), 'variantIds': sorted(variants),
        'tagCombinations': sorted(triples), 'preferenceSets': sorted(prefs)}


def plan_for(request, catalog, original=None):
    from ..page_plan import make_page_plan
    if original:
        cards = [c for s in original['sections'] for c in s['products']]
        order = [c['productId'] for c in cards]
        images = {c['productId']: c['imageCaptionId'] for c in cards}
    else:
        order = list(catalog)
        images = {pid: next(v['id'] for v in p['presentationVariants']
                  if v['presentationStyle'] == request['context']['presentationPreferences'][0])
                  for pid, p in catalog.items()}
    decisions = {'rankedProducts': [{'productId': pid, 'score': float(6 - i)} for i, pid in enumerate(order)],
        'presentationActions': {pid: {'productId': pid, 'variantId': images[pid]} for pid in order}}
    plan = make_page_plan(request, catalog, decisions, 6)
    plan['method']['selectionAndOrder'] = 'frozen synthetic fixture order; no learned ranking quality claim'
    assert [(c['productId'], c['imageCaptionId']) for s in plan['sections'] for c in s['products']] == [(p, images[p]) for p in order]
    return plan


def case_and_fields(request, catalog, split, family, original=None, **metadata):
    from ..semantic_copy_data.generator import public_messages
    plan = plan_for(request, catalog, original)
    common, evidence = public_messages(plan, request, catalog)
    targets = []
    for section in evidence['sections']:
        i = section['index']; tags = section['universalTags']
        value = tags[0].capitalize() + ' travel choices' if tags else 'Everyday travel choices'
        targets.append({'rowId': f'heading-{i}', 'field': f'S{i}', 'kind': 'heading',
            'sectionIndex': i, 'maxTokens': 16, 'targetValue': value})
    for card in evidence['cards']:
        i = card['index']; support = card['requiredObservedTags']
        title = card['identity'] + ' for ' + (support[0] if support else 'everyday') + ' travel'
        for letter, kind, cap, value in (('T', 'title', 24, title),
                ('B', 'badge', 8, support[0] + ' choice' if support else 'NONE')):
            targets.append({'rowId': f'{kind}-{i}', 'field': f'{letter}{i}', 'kind': kind,
                            'cardIndex': i, 'maxTokens': cap, 'targetValue': value})
    case = {'schemaVersion': 1, 'datasetVersion': VERSION, 'synthetic': True,
        'split': split, 'caseId': request['recordId'], 'personaFamilyId': family,
        'selectionSource': 'frozen fixture; ranking and image quality unlabeled',
        'request': request, 'catalog': catalog, 'pendingPagePlan': plan,
        'publicEvidence': evidence, 'targets': targets,
        'targetSource': 'audited synthetic teacher grammar; never inference fallback', **metadata}
    fields = []
    for target in targets:
        i = target.get('cardIndex', target.get('sectionIndex'))
        bound = 'section' if target['kind'] == 'heading' else 'card'
        suffix = {'role': 'user', 'content': f"Write only the fresh {target['kind']} value for {bound} {i} using its public evidence and {target['kind']} rules. End after this one line."}
        messages = [common, suffix]
        fields.append({'schemaVersion': 1, 'datasetVersion': VERSION, 'split': split,
            'caseId': case['caseId'], 'personaFamilyId': family, **target,
            'messages': messages, 'messagesSha256': digest(messages),
            'targetSource': case['targetSource'], 'inferenceMetadataIsNotModelInput': True})
    return case, fields


def pair_choices(old_cases):
    candidates = []
    for index, case in enumerate(old_cases):
        pairs = sorted({pair for p in case['catalog'].values()
                        for pair in itertools.combinations(sorted(p['tags']), 2)})
        candidates += [(index, p) for p in pairs]
    counts = Counter(); chosen = []

    def key(item):
        _, pair = item
        return (sum(counts[t] == 0 for t in pair), -sum(counts[t] for t in pair), digest(item))

    # Every old six-card fixture is used, preserving all 72 training products.
    for index in range(12):
        option = max((c for c in candidates if c[0] == index), key=key)
        chosen.append(option); counts.update(option[1])
    for _ in range(6):
        option = max((c for c in candidates if c not in chosen), key=key)
        chosen.append(option); counts.update(option[1])
    assert set(counts) == set(TAGS)
    return chosen


def fresh_tagsets(rng, preferences, strategy, used):
    a, b = preferences
    c = rng.choice([t for t in TAGS if t not in preferences])
    if strategy == 0: wants, shared = [a, a, a, b, None, None], [a, None]
    elif strategy == 1: wants, shared = [a, b, None, a, b, None], [None, None]
    else: wants, shared = [a, None, None, b, b, b], [c, b]
    options = list(itertools.combinations(sorted(TAGS), 3))
    choices = [[tags for tags in options if tags not in used and
                set(tags) & set(preferences) == ({wanted} if wanted else set()) and
                (shared[i // 3] is None or shared[i // 3] in tags)] for i, wanted in enumerate(wants)]
    for _ in range(10000):
        selected = [rng.choice(values) for values in choices]
        if len(set(selected)) != 6: continue
        if all(set.intersection(*(set(t) for t in selected[j:j + 3])) == ({shared[j // 3]} if shared[j // 3] else set()) for j in (0, 3)):
            used.update(selected)
            return selected
    raise ValueError('Could not build fresh exact-intersection case')


def build():
    inventory = source_inventory(); rng = random.Random(SEED)
    old_cases = records(OLD / 'cases.train.jsonl')
    cases = {s: [] for s in ('train', 'dev', 'test')}; fields = {s: [] for s in cases}; pairs = []
    for number, (index, tags) in enumerate(pair_choices(old_cases)):
        original = old_cases[index]; family = 'copy-pair-family-' + digest([SEED, number])[:16]
        siblings = []
        for arm, tag in enumerate(tags):
            request = copy.deepcopy(original['request'])
            request['recordId'] = 'copy-pair-' + digest([family, arm])[:16]
            request['context']['task'] = 'Choose daypacks for everyday travel with ' + tag + ' details.'
            request['context']['statedPreferences'] = [tag]
            case, rows = case_and_fields(request, copy.deepcopy(original['catalog']), 'train', family,
                original['pendingPagePlan'], sourceCaseId=original['caseId'], pairArm=arm,
                publicPreferenceSet=[tag], pairedFamilyId=family)
            cases['train'].append(case); fields['train'] += rows; siblings.append(case)
        shared = sorted(pid for pid, p in original['catalog'].items() if set(tags) <= set(p['tags']))
        pairs.append({'familyId': family, 'sourceCaseId': original['caseId'],
            'caseIds': [c['caseId'] for c in siblings], 'singletonPreferences': list(tags),
            'sameCatalogSha256': digest(original['catalog']), 'sameProductOrderAndImages': True,
            'mustChangeTitleAndBadgeForProductIds': shared})
    used = set(map(tuple, inventory['tagCombinations']))
    old_prefs = set(map(tuple, inventory['preferenceSets']))
    available = [p for p in itertools.combinations(sorted(TAGS), 2) if p not in old_prefs]
    rng.shuffle(available)
    names = [a + b for a in ('Zeli', 'Tavu', 'Neri', 'Qori', 'Vesu', 'Wali')
             for b in ('nor', 'ris', 'lan', 'ven', 'dor', 'mar')]
    rng.shuffle(names)
    for ordinal in range(6):
        split = 'dev' if ordinal < 3 else 'test'; preferences = list(available[ordinal])
        if ordinal % 2: preferences.reverse()
        tagsets = fresh_tagsets(rng, preferences, ordinal % 3, used)
        catalog = {}
        for index, tags in enumerate(tagsets):
            name = names[ordinal * 6 + index]; assert name.lower() not in inventory['names']
            capacity = 13 + (ordinal * 6 + index) * 7 % 27
            title = f'{name} {capacity}L Daypack'; pid = 'pair-fresh-p-' + digest([name, capacity, tags])[:16]
            variants = []
            for style in ('detail-view', 'in-use-view'):
                vid = 'pair-fresh-v-' + digest([pid, style])[:16]
                variants.append({'id': vid, 'title': title if style == 'detail-view' else title + ' — ' + tags[0],
                    'imageDescription': f'Text description of {title} with its listed catalog facts in {style}.',
                    'imageKey': 'synthetic-copy-pairs/' + vid, 'facts': [f'{capacity}L', *tags],
                    'presentationStyle': style, 'compatibleSurfaces': ['compact-card', 'detail-panel']})
            catalog[pid] = {'id': pid, 'title': title, 'category': 'backpacks',
                'description': f'{capacity}-litre daypack; ' + ', '.join(tags) + '.',
                'tags': list(tags), 'price': 49 + index * 6, 'shippingCost': 5,
                'inStock': True, 'presentationVariants': variants}
        case_id = 'copy-pair-fresh-' + digest([SEED, ordinal])[:16]
        request = {'recordId': case_id, 'asOf': '2026-10-01T12:00:00Z',
            'candidateProductIds': list(catalog), 'history': [],
            'context': {'task': 'Choose daypacks for everyday travel with ' + ' and '.join(preferences) + ' details.',
                'statedBudget': 200, 'statedRequirements': [], 'statedPreferences': preferences,
                'presentationSurface': 'compact-card', 'presentationPreferences': ['detail-view' if ordinal % 2 else 'in-use-view']},
            'presentationCompatibility': {pid: {v['id']: True for v in p['presentationVariants']} for pid, p in catalog.items()}}
        case, rows = case_and_fields(request, catalog, split, 'copy-pair-fresh-family-' + digest([SEED, split, ordinal])[:16],
            freshness='new synthetic request/products/combinations frozen before next native attempt',
            source='fresh synthetic teacher grammar; not human validated production heldout')
        cases[split].append(case); fields[split] += rows
    return cases, fields, pairs, inventory


def audit(cases, fields, pairs, inventory):
    from ..evidence_plan import validate_generated
    from ..fusion import _normalized, identity_errors
    old_train = records(OLD / 'cases.train.jsonl')
    old_products = {pid: p for c in old_train for pid, p in c['catalog'].items()}
    actual_products = {pid: p for c in cases['train'] for pid, p in c['catalog'].items()}
    assert actual_products == old_products and len(actual_products) == 72
    summary = {}; value_sets = {}; memberships = {}
    support_counts = Counter()
    for split, count in (('train', 36), ('dev', 3), ('test', 3)):
        assert len(cases[split]) == count and len(fields[split]) == count * 14
        names = set(); pids = set(); vids = set(); triples = set(); preferences = set(); values = set()
        counts = Counter()
        for case in cases[split]:
            plan = case['pendingPagePlan']; catalog = case['catalog']; targets = case['targets']
            assert len(targets) == 14 and sum(t['maxTokens'] for t in targets) == 224
            raw = '\n'.join(t['field'] + ': ' + t['targetValue'] for t in targets)
            _, quality = validate_generated(raw, 'stop', plan, case['request'], catalog)
            assert quality['allFieldsFulfilled'], (case['caseId'], quality)
            counts['teacherPagesPassingStrictValidator'] += 1
            preferences.add(tuple(sorted(case['request']['context']['statedPreferences'])))
            names.update(p['title'].split()[0].lower() for p in catalog.values()); pids.update(catalog)
            vids.update(v['id'] for p in catalog.values() for v in p['presentationVariants'])
            triples.update(tuple(sorted(p['tags'])) for p in catalog.values())
            cards = [c for s in plan['sections'] for c in s['products']]
            case_supported = Counter()
            for target in targets:
                text = target['targetValue']; kind = target['kind']
                if text != 'NONE': values.add(_normalized(text))
                assert '\n' not in text and not re.search(r'\b[STB]\d+:', text)
                if kind == 'heading':
                    assert 3 <= len(text.split()) <= 5
                    common = case['publicEvidence']['sections'][target['sectionIndex'] - 1]['universalTags']
                    counts['positiveUniversalHeadings' if common else 'emptyUniversalHeadings'] += 1
                else:
                    card = cards[target['cardIndex'] - 1]; product = catalog[card['productId']]
                    support = [f.rsplit(':', 1)[1] for f in card['approvedFactIds']]
                    if kind == 'title':
                        assert not identity_errors(text, product, catalog)
                        assert _normalized(product['title']) in _normalized(text) and 4 <= len(text.split()) <= 8
                        counts['titlesPassingIdentity'] += 1
                        if support:
                            counts['supportedTitles'] += 1; case_supported.update(support)
                            if split == 'train': support_counts.update(support)
                    elif support:
                        assert text != 'NONE'; counts['supportedBadges'] += 1
                    else:
                        assert text == 'NONE'; counts['requiredNoneBadges'] += 1
            assert counts['requiredNoneBadges'] > 0
            if split != 'train':
                assert set(case_supported) == set(case['request']['context']['statedPreferences'])
                assert sum(case_supported.values()) == 4
        for row in fields[split]:
            assert digest(row['messages']) == row['messagesSha256']
            message_text = canonical(row['messages'])
            for key in ('personaFamilyId', 'pairedFamilyId', 'targetValue', 'sourceCaseId'):
                assert key not in message_text
            assert row['caseId'] not in message_text and row['personaFamilyId'] not in message_text
            if row['targetValue'] != 'NONE': assert row['targetValue'].lower() not in message_text.lower()
        summary[split] = {'requests': count, 'fieldExamples': len(fields[split]),
            'uniqueProducts': len(pids), **dict(counts)}
        value_sets[split] = values
        memberships[split] = {'names': names, 'productIds': pids, 'variantIds': vids,
            'tagCombinations': triples, 'preferenceSets': preferences}
    assert set(support_counts) == set(TAGS) and all(support_counts[t] > 0 for t in TAGS)
    for a, b in itertools.combinations(memberships, 2):
        for key in memberships[a]: assert not memberships[a][key] & memberships[b][key], (a, b, key)
    for split in ('dev', 'test'):
        for key in memberships[split]:
            assert not memberships[split][key] & set(map(tuple, inventory[key]) if key in ('tagCombinations', 'preferenceSets') else inventory[key]), key
    by_id = {c['caseId']: c for c in cases['train']}; changed = 0
    assert len(pairs) == 18
    for pair in pairs:
        a, b = [by_id[cid] for cid in pair['caseIds']]
        assert a['catalog'] == b['catalog'] and digest(a['catalog']) == pair['sameCatalogSha256']
        assert len(a['request']['context']['statedPreferences']) == len(b['request']['context']['statedPreferences']) == 1
        assert a['request']['context']['statedPreferences'] != b['request']['context']['statedPreferences']
        for key in ('history', 'candidateProductIds', 'asOf', 'presentationCompatibility'):
            assert a['request'][key] == b['request'][key]
        ac = [c for s in a['pendingPagePlan']['sections'] for c in s['products']]
        bc = [c for s in b['pendingPagePlan']['sections'] for c in s['products']]
        assert [(c['productId'], c['imageCaptionId']) for c in ac] == [(c['productId'], c['imageCaptionId']) for c in bc]
        assert pair['mustChangeTitleAndBadgeForProductIds']
        for pid in pair['mustChangeTitleAndBadgeForProductIds']:
            i = next(i for i, c in enumerate(ac, 1) if c['productId'] == pid)
            av = {t['field']: t['targetValue'] for t in a['targets']}
            bv = {t['field']: t['targetValue'] for t in b['targets']}
            assert av[f'T{i}'] != bv[f'T{i}'] and av[f'B{i}'] != bv[f'B{i}']
            assert av[f'B{i}'] != 'NONE' and bv[f'B{i}'] != 'NONE'; changed += 1
    return {'status': 'passed', 'nativeExecution': False, 'modelCalls': 0, 'trainingExecuted': False,
        'pairedFamilies': 18, 'sharedProductsWithChangedTitleAndBadgeAcrossPair': changed,
        'oldTrainingProductsByteEquivalent': 72, 'positiveTrainingTitleCountsByAttribute': dict(sorted(support_counts.items())),
        'summary': summary, 'normalizedNonNoneTargetOverlapBetweenSplits': {
            a + '/' + b: len(value_sets[a] & value_sets[b]) for a, b in itertools.combinations(value_sets, 2)},
        'targetOverlapDisclosure': 'Shared short heading/badge grammar and NONE are intentional, not a held-out wording claim; product title completions use disjoint new names.'}


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('mode', choices=('write', 'check')); args = parser.parse_args()
    cases, fields, pairs, inventory = build(); result = audit(cases, fields, pairs, inventory)
    outputs = {f'{kind}.{split}.jsonl': ''.join(canonical(row) + '\n' for row in rows)
        for kind, groups in (('cases', cases), ('fields', fields)) for split, rows in groups.items()}
    outputs['pairs.jsonl'] = ''.join(canonical(row) + '\n' for row in pairs)
    outputs['exclusions.json'] = json.dumps(inventory, indent=2) + '\n'
    replay = {'status': 'previously inspected replay; never fresh held-out evidence',
        'cases': {s: {'path': str((OLD / f'cases.{s}.jsonl').relative_to(REPO)),
                     'sha256': sha(OLD / f'cases.{s}.jsonl'), 'caseIds': [c['caseId'] for c in records(OLD / f'cases.{s}.jsonl')]}
                  for s in ('dev', 'test')}, 'separateDenominatorsRequired': True}
    outputs['inspected_replays.json'] = json.dumps(replay, indent=2) + '\n'
    outputs['AUDIT.json'] = json.dumps(result, indent=2) + '\n'
    manifest = {'schemaVersion': 1, 'datasetVersion': VERSION, 'seed': SEED,
        'frozenBeforeNextNativeOrTrainingAttempt': True, 'modelCalls': 0, 'nativeExecution': False,
        'generatorSha256': sha(Path(__file__)), 'sourceSha256': inventory['sourceSha256'],
        'validatorSha256AtAudit': {name: sha(ROOT / name) for name in ('evidence_plan.py', 'fusion.py')},
        'fileSha256': {name: hashlib.sha256(body.encode()).hexdigest() for name, body in outputs.items()},
        'trainFile': 'fields.train.jsonl', 'developmentFile': 'fields.dev.jsonl', 'evaluationOnlyFile': 'fields.test.jsonl',
        'proposedTrainingLater': {'epochs': 1, 'individualUpdates': 504, 'executed': False,
            'selection': 'final fixed step, no dev/test checkpoint selection'},
        'promptContract': 'same public_messages and independent values-only field request; metadata excluded',
        'audit': result,
        'limitations': ['Synthetic grammar, not human validated production data or measured model improvement.',
            'All targets are supervision only; no runtime teacher-answer fallback.',
            'Only explicit public preferences; no history or latent persona truths.',
            'Train catalogs and fixture selection are reused; all fresh ranking/image quality is unlabeled.',
            'Common heading/badge wording and NONE overlap across cases/splits intentionally.',
            'Fresh cases disjoint from the explicit hashed exclusion inventory, not a claim about unseen external data.',
            'Tokenizer target+EOS, context and native cap fit not measured in this data-only preparation.',
            'Previously inspected old dev/test are separate replay references, never fresh holdouts.']}
    outputs['manifest.json'] = json.dumps(manifest, indent=2) + '\n'
    if args.mode == 'write':
        if any((HERE / name).exists() for name in outputs): raise FileExistsError('Frozen package exists; never overwrite')
        for name, body in outputs.items(): (HERE / name).write_text(body)
    else:
        for name, body in outputs.items():
            if (HERE / name).read_text() != body: raise AssertionError('Frozen artifact changed: ' + name)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
