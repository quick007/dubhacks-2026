"""Freeze projected STB+C teacher supervision, without native/model/tokenizer calls."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from .layout import VERSION, assemble_payload, build_layout, canonical
from ..above_fold import REQUIRED, project_first_section

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / 'semantic_copy_pairs'
DATA_VERSION = 'first-section-stbc-teacher-v1'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def records(path): return [json.loads(line) for line in path.read_text().splitlines() if line]


def parent_hashes():
    manifest = json.loads((PARENT / 'manifest.json').read_text())
    for name, expected in manifest['fileSha256'].items():
        if sha(PARENT / name) != expected: raise ValueError('Parent frozen payload changed: ' + name)
    return {'parentManifestSha256': sha(PARENT / 'manifest.json'),
        'parentFileSha256': manifest['fileSha256']}


def build():
    cases = {}; fields = {}; summary = {}; values = {}; all_cases = {}
    for split, count in (('train', 36), ('dev', 3), ('test', 3)):
        source_cases = records(PARENT / f'cases.{split}.jsonl')
        assert len(source_cases) == count
        cases[split] = []; fields[split] = []; counts = Counter(); atoms = Counter(); targets_seen = set(); copy_seen = set()
        for original in source_cases:
            full_plan = copy.deepcopy(original['pendingPagePlan'])
            root, jobs, evidence = build_layout(full_plan, original['request'], original['catalog'])
            retained = {t['field']: copy.deepcopy(t) for t in original['targets'] if t['field'] in REQUIRED}
            cards = full_plan['sections'][0]['products']
            for i, card in enumerate(cards, 1):
                product = original['catalog'][card['productId']]
                support = [f.rsplit(':', 1)[1] for f in card['approvedFactIds']]
                tag = support[0] if support else product['tags'][0]
                value = f'A {tag} daypack for everyday travel.'
                retained[f'C{i}'] = {'rowId': f'copy-{i}', 'field': f'C{i}', 'kind': 'copy',
                    'cardIndex': i, 'maxTokens': 32, 'targetValue': value}
                counts['supportedCopyTargets' if support else 'factualUnobservedCopyTargets'] += 1
                atoms.update(support); copy_seen.add(value)
            targets = [retained[f] for f in REQUIRED]
            teacher_rows = [{**job, 'text': retained[job['fields'][0]]['targetValue'],
                'finishReason': 'stop'} for job in jobs]
            result = assemble_payload(teacher_rows, full_plan, original['request'], original['catalog'])
            assert result['validation']['valid'], (original['caseId'], result['validation'])
            counts['teacherPayloadsPassingStrictValidation'] += 1
            counts['titlesPassingCurrentIdentity'] += 3
            counts['requiredNoneBadges'] += sum(t['kind'] == 'badge' and t['targetValue'] == 'NONE' for t in targets)
            counts['positiveUniversalHeadings' if evidence['sections'][0]['universalTags'] else 'emptyUniversalHeadings'] += 1
            case = {'schemaVersion': 1, 'datasetVersion': DATA_VERSION, 'split': split,
                'caseId': original['caseId'], 'personaFamilyId': original['personaFamilyId'],
                'sourceParentCaseId': original['caseId'], 'parentCasesInspectedDuringSTBPreparation': True,
                'newNativeInferenceExecuted': False, 'synthetic': True,
                'request': copy.deepcopy(original['request']), 'catalog': copy.deepcopy(original['catalog']),
                'fullSourcePagePlan': full_plan, 'sourceFullPageRequestedCardCount': full_plan['requestedCardCount'],
                'pendingPagePlan': project_first_section(full_plan),
                'publicEvidence': evidence, 'targets': targets,
                'targetSource': 'audited synthetic teacher grammar, not model output or runtime fallback'}
            cases[split].append(case); all_cases[case['caseId']] = case
            for target, job in zip(targets, jobs):
                assert target['field'] == job['fields'][0]
                messages = [root, *job['messages']]
                content = canonical(messages)
                assert original['caseId'] not in content and original['personaFamilyId'] not in content
                assert 'targetValue' not in content and 'sourceParentCaseId' not in content
                if target['targetValue'] != 'NONE':
                    assert target['targetValue'].lower() not in content.lower(), 'Exact teacher answer leaked into input'
                    targets_seen.add(target['targetValue'])
                fields[split].append({'schemaVersion': 1, 'datasetVersion': DATA_VERSION, 'split': split,
                    'caseId': original['caseId'], 'personaFamilyId': original['personaFamilyId'], **target,
                    'messages': messages, 'messagesSha256': digest(messages),
                    'targetSource': case['targetSource'], 'inferenceMetadataIsNotModelInput': True})
        assert len(fields[split]) == count * 10
        summary[split] = {'requests': count, 'fieldExamples': len(fields[split]),
            'copyExamples': count * 3, 'uniqueCopyTargetStrings': len(copy_seen),
            'positiveObservedCopyCountsByAttribute': dict(sorted(atoms.items())), **dict(counts)}
        values[split] = targets_seen
    visible_pairs = []
    for pair in records(PARENT / 'pairs.jsonl'):
        a, b = [all_cases[c] for c in pair['caseIds']]
        visible = {c['productId'] for c in a['pendingPagePlan']['sections'][0]['products']}
        shared = visible & set(pair['mustChangeTitleAndBadgeForProductIds'])
        for pid in shared:
            i = next(i for i, c in enumerate(a['pendingPagePlan']['sections'][0]['products'], 1) if c['productId'] == pid)
            av = {t['field']: t['targetValue'] for t in a['targets']}; bv = {t['field']: t['targetValue'] for t in b['targets']}
            assert all(av[f'{p}{i}'] != bv[f'{p}{i}'] for p in ('T', 'B', 'C'))
        visible_pairs.append({'familyId': pair['familyId'], 'caseIds': pair['caseIds'],
            'sharedChangedProductsVisible': sorted(shared), 'visibleCounterfactualRemains': bool(shared)})
    audit = {'status': 'passed', 'modelCalls': 0, 'nativeExecution': False, 'tokenizerCalls': 0,
        'trainingExecuted': False, 'teacherValidationUsesSyntheticStopNotNativeEOS': True,
        'summary': summary, 'pairedFamiliesRetainingVisibleContrast': sum(p['visibleCounterfactualRemains'] for p in visible_pairs),
        'pairedFamiliesOriginal': 18, 'nonNoneExactTargetOverlapAcrossSplits': {
            'train/dev': len(values['train'] & values['dev']), 'train/test': len(values['train'] & values['test']),
            'dev/test': len(values['dev'] & values['test'])},
        'noFreshHeldoutClaim': 'Parent cases inspected during STB data preparation; no new inference yet. New copy grammar is deliberately shared.'}
    return cases, fields, visible_pairs, audit


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('mode', choices=('write', 'check')); args = parser.parse_args()
    sources = parent_hashes()
    dependencies = [ROOT / n for n in ('above_fold.py', 'ABOVE_FOLD_WORKLOAD.json', 'branch_local_copy.py',
        'parallel_decode.py', 'scheduler_identity_experiment.py', 'fusion.py', 'evidence_plan.py', 'page_plan.py')]
    dependency_hashes = {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    cases, fields, pairs, audit = build()
    if dependency_hashes != {str(p.relative_to(ROOT)): sha(p) for p in dependencies} or sources != parent_hashes():
        raise ValueError('Parent/dependency changed during CPU audit; retry after writer finishes')
    outputs = {f'{kind}.{split}.jsonl': ''.join(canonical(row) + '\n' for row in rows)
        for kind, groups in (('cases', cases), ('fields', fields)) for split, rows in groups.items()}
    outputs['projected_pairs.jsonl'] = ''.join(canonical(p) + '\n' for p in pairs)
    outputs['AUDIT.json'] = json.dumps(audit, indent=2) + '\n'
    manifest = {'schemaVersion': 1, 'datasetVersion': DATA_VERSION, 'layoutVersion': VERSION,
        'frozenBeforeNative': True, 'modelCalls': 0, 'nativeExecution': False, 'trainingExecuted': False,
        **sources, 'dependencySha256': dependency_hashes,
        'sourceSha256': {name: sha(HERE / name) for name in ('layout.py', 'data.py')},
        'fileSha256': {name: hashlib.sha256(body.encode()).hexdigest() for name, body in outputs.items()},
        'trainFile': 'fields.train.jsonl', 'developmentFile': 'fields.dev.jsonl', 'evaluationOnlyFile': 'fields.test.jsonl',
        'counts': {'train': 360, 'dev': 30, 'test': 30}, 'caps': {'heading': 16, 'title': 24, 'badge': 8, 'copy': 32},
        'totalCapPerView': 208, 'queued504STBTrainingChanged': False, 'audit': audit,
        'limitations': ['Teacher grammar supervision only, not native generated output or runtime fallback.',
            'Exact fixed first-section projection loses organizer positive support and six of eighteen paired contrasts.',
            'Names/catalog/request/splits inherited unchanged from reviewed parent corpus; not a new unseen holdout.',
            'Short copy/headings/badge targets repeat within and across splits; no unseen wording claim.',
            'Actual tokenizer target+EOS/context fit, loaded-model cache controls and measured producer remain pending root review.',
            'No ranking, image pixels, browser paint, full-page validation, conversion or latency improvement established.']}
    outputs['manifest.json'] = json.dumps(manifest, indent=2) + '\n'
    if args.mode == 'write':
        if any((HERE / name).exists() for name in outputs): raise FileExistsError('Frozen extension already exists')
        for name, body in outputs.items(): (HERE / name).write_text(body)
    else:
        for name, body in outputs.items():
            if (HERE / name).read_text() != body: raise AssertionError('Frozen artifact/dependency changed: ' + name)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__': main()
