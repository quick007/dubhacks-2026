"""Read-only CPU revalidation of direct semantic-copy evaluation outputs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'semantic_copy_data'
RESULTS = ROOT.parents[2] / 'data/ranker/pretrained/semantic-copy-direct-v1/results.json'
SPLITS = ('dev', 'test')
ARMS = ('frozen', 'learned')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def stats(values):
    return {'count': len(values), 'sum': sum(values),
            'median': statistics.median(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def pending(plan):
    """Keep actual ranking, images and evidence; discard saved generated copy."""
    result = copy.deepcopy(plan)
    for section in result['sections']:
        section['heading'] = {'text': None, 'source': 'pending'}
        section['copy'] = {'text': None, 'source': 'none'}
        for card in section['products']:
            card['title'] = {'text': None, 'source': 'pending'}
            card['badge'] = {'text': None, 'source': 'none' if card['badgeDecision'] == 'NONE' else 'pending'}
    return result


def field_details(rows, replay, case):
    from .fusion import _field_claim_errors, _normalized, identity_errors
    sections = replay['pagePlan']['sections']
    cards = [c for s in sections for c in s['products']]
    quality = replay['quality']
    output = []
    global_stop = all(r['finishReason'] == 'stop' for r in rows)
    global_plain = 'unfinished-or-markup-output' not in quality['rejections']
    for row, wire in zip(rows, replay['wireRows']):
        field = row['fields'][0]; index = int(field[1:]) - 1
        kind = row['kind']; raw = row['text']
        if kind == 'heading':
            section = sections[index]
            members = [case['catalog'][c['productId']] for c in section['products']]
            common = sorted(set.intersection(*(set(p['tags']) for p in members)))
            facts = {'memberIdentities': [p['title'] for p in members], 'universalTags': common}
            product = {'title': 'Everyday travel choices', 'tags': common, 'category': 'backpacks'}
            accepted = section['heading']['source'] == 'generated'; support = []
        else:
            card = cards[index]; product = case['catalog'][card['productId']]
            support = [f.rsplit(':', 1)[1] for f in card['approvedFactIds']]
            facts = {'selectedIdentity': product['title'], 'productId': card['productId'],
                'imageCaptionId': card['imageCaptionId'], 'tags': product['tags'],
                'observedSupport': support, 'approvedTitles': [product['title'],
                    *(v['title'] for v in product['presentationVariants'])]}
            if kind == 'title':
                accepted = card['title']['source'] == 'generated'
            else:
                accepted = card['badge']['source'] == 'generated' or (
                    raw == 'NONE' and global_stop and global_plain and
                    not wire['valueProtocolRejections'] and
                    not any(r.startswith(field + ':') for r in quality['rejections']))
        body_claims = [] if kind == 'badge' and raw == 'NONE' else _field_claim_errors(raw, product, case['catalog'])
        entry = {'field': field, 'kind': kind, 'rawValue': raw, 'accepted': accepted,
            'finishReason': row['finishReason'], 'wordCount': len(raw.split()),
            'outputTokens': len(row['tokenIds']), 'arSteps': row['arSteps'],
            'maxTokens': row['maxTokens'], 'productFacts': facts,
            'factualClaimErrors': body_claims, 'protocolRejections': wire['valueProtocolRejections'],
            'requestedWordRangeMet': (3 <= len(raw.split()) <= 5) if kind == 'heading'
                else ((4 <= len(raw.split()) <= 8) if kind == 'title' else None)}
        if kind == 'title':
            entry['identityErrors'] = identity_errors(raw, product, case['catalog'])
            entry['copiesApprovedTitle'] = _normalized(raw) in set(map(_normalized, facts['approvedTitles']))
        if kind in ('title', 'badge'):
            entry['observedSupportMentioned'] = any(re.search(r'\b' + re.escape(_normalized(t)) + r'\b', _normalized(raw)) for t in support)
            entry['observedSupportAvailable'] = bool(support)
        if kind == 'badge':
            entry.update(rawExactNone=raw == 'NONE', noneRequired=not support,
                         rawNonePolicyCorrect=(raw == 'NONE') if not support else True)
        output.append(entry)
    return output


def analyze(report, cases, source_hashes=None):
    """Reject changed saved metrics; never patch a saved arm or use teacher text."""
    from .scheduler_identity_experiment import assemble_values_legacy_v1 as assemble_values
    from .fusion import TITLE_IDENTITY_POLICY
    expected = {}
    for split in SPLITS:
        if len(cases[split]) != 3 or len({c['caseId'] for c in cases[split]}) != 3:
            raise ValueError('Expected exactly three unique frozen cases per split')
        for case in cases[split]:
            for arm in ARMS:
                expected[(split, case['caseId'], arm)] = case
    seen = set(); evaluated = []
    for arm in report['arms']:
        key = (arm['split'], arm['caseId'], arm['arm'])
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate evaluation arm: ' + str(key))
        seen.add(key); case = expected[key]
        if arm.get('status') != 'completed-attempt':
            evaluated.append({'split': key[0], 'caseId': key[1], 'arm': key[2],
                              'status': arm.get('status'), 'blocker': arm.get('blocker')})
            continue
        plan = pending(arm['pagePlan']); rows = arm['rows']
        actual_cards = [c for s in plan['sections'] for c in s['products']]
        expected_fields = {f'S{i}': ('heading', 16) for i in (1, 2)} | {
            f'{letter}{i}': (kind, cap) for i in range(1, 7)
            for letter, kind, cap in (('T', 'title', 24), ('B', 'badge', 8))}
        if len(actual_cards) != 6 or len(plan['sections']) != 2 or len(rows) != 14:
            raise ValueError('Evaluation silently changed six-card/fourteen-field denominator')
        row_fields = [r['fields'][0] for r in rows if len(r['fields']) == 1]
        if len(row_fields) != 14 or set(row_fields) != set(expected_fields):
            raise ValueError('Trusted scheduler metadata membership changed')
        if any((r['kind'], r['maxTokens']) != expected_fields[r['fields'][0]] for r in rows):
            raise ValueError('Trusted field kind or cap changed')
        replay = assemble_values(copy.deepcopy(rows), plan, case['request'], case['catalog'])
        if replay['quality'] != arm['quality']:
            raise ValueError('Saved/current quality mismatch for ' + str(key) + ': ' +
                             json.dumps({'saved': arm['quality'], 'current': replay['quality']}))
        tokens = sum(len(r['tokenIds']) for r in rows); ar = sum(r['arSteps'] for r in rows)
        if tokens != arm['actualOutputTokens'] or ar != arm['actualRowArSteps']:
            raise ValueError('Saved token/AR totals disagree with raw row receipts')
        if arm['totalOutputBudget'] != 224:
            raise ValueError('Saved summed output budget changed')
        details = field_details(rows, replay, case)
        evaluated.append({'split': key[0], 'caseId': key[1], 'arm': key[2],
            'status': 'completed-attempt', 'qualityMatchesSaved': True,
            'quality': replay['quality'], 'fields': details,
            'selectedProductsAndImages': [[s['sectionId'], [[c['productId'], c['imageCaptionId']]
                for c in s['products']]] for s in plan['sections']],
            'actualOutputTokens': tokens, 'actualRowArSteps': ar,
            'externalFullCompletionMs': arm['externalFullCompletionMs'],
            'timing': arm['timing'], 'memory': arm['memory']})
    summaries = {}
    for split in SPLITS:
        summaries[split] = {}
        for name in ARMS:
            selected = [a for a in evaluated if a['split'] == split and a['arm'] == name]
            completed = [a for a in selected if a['status'] == 'completed-attempt']
            valid = [a for a in completed if a['quality']['allFieldsFulfilled']]
            fields = [f for a in completed for f in a['fields']]
            required = [f for f in fields if f['kind'] == 'badge' and f['noneRequired']]
            alignment = [f for f in fields if f['kind'] == 'title' and f['observedSupportAvailable']]
            badge_alignment = [f for f in fields if f['kind'] == 'badge' and f['observedSupportAvailable']]
            summaries[split][name] = {
                'fixedCases': 3, 'recorded': len(selected), 'completed': len(completed),
                'failedRecorded': sum(a['status'] != 'completed-attempt' for a in selected),
                'unrecorded': 3 - len(selected), 'acceptedPages': len(valid),
                'acceptedPageRateFixedCases': len(valid) / 3,
                'acceptedTitles': sum(f['accepted'] for f in fields if f['kind'] == 'title'),
                'fixedTitleDenominator': 18,
                'acceptedBadgesIncludingNone': sum(f['accepted'] for f in fields if f['kind'] == 'badge'),
                'acceptedGeneratedBadges': sum(a['quality']['generatedBadgeCount'] for a in completed),
                'acceptedNoneBadges': sum(f['accepted'] and f['rawExactNone'] for f in fields if f['kind'] == 'badge'),
                'fixedBadgeDenominator': 18,
                'acceptedHeadings': sum(a['quality']['generatedHeadingCount'] for a in completed),
                'fixedHeadingDenominator': 6,
                'strictCorrectNonePages': sum(a['quality']['correctNone'] is True for a in completed),
                'strictCorrectNoneEvaluablePages': sum(a['quality']['correctNone'] is not None for a in completed),
                'requiredNoneFieldsExecuted': len(required),
                'requiredNoneRawCorrect': sum(f['rawExactNone'] for f in required),
                'requiredNoneStrictAccepted': sum(f['accepted'] for f in required),
                'supportedTitleFieldsExecuted': len(alignment),
                'supportedTitleRawMention': sum(f['observedSupportMentioned'] for f in alignment),
                'supportedTitleAcceptedAndAligned': sum(f['accepted'] and f['observedSupportMentioned'] for f in alignment),
                'supportedBadgeFieldsExecuted': len(badge_alignment),
                'supportedBadgeRawMention': sum(f['observedSupportMentioned'] for f in badge_alignment),
                'supportedBadgeAcceptedAndAligned': sum(f['accepted'] and f['observedSupportMentioned'] for f in badge_alignment),
                'supportedBadgeAcceptedNone': sum(f['accepted'] and f['rawExactNone'] for f in badge_alignment),
                'outputTokensAll': stats([a['actualOutputTokens'] for a in completed]),
                'outputTokensValidOnly': stats([a['actualOutputTokens'] for a in valid]),
                'arStepsAll': stats([a['actualRowArSteps'] for a in completed]),
                'arStepsValidOnly': stats([a['actualRowArSteps'] for a in valid]),
                'fullCompletionMsAll': stats([a['externalFullCompletionMs'] for a in completed]),
                'fullCompletionMsValidOnly': stats([a['externalFullCompletionMs'] for a in valid]),
                'maxObservedMlxPeakBytes': max((a['memory'].get('mlxPeakBytes', 0) for a in completed), default=None)}
    pairs = []
    for split in SPLITS:
        for case in cases[split]:
            pair = [a for a in evaluated if a['split'] == split and a['caseId'] == case['caseId'] and a['status'] == 'completed-attempt']
            if len(pair) == 2:
                pairs.append({'split': split, 'caseId': case['caseId'],
                    'sameProductsImagesAndGrouping': pair[0]['selectedProductsAndImages'] == pair[1]['selectedProductsAndImages']})
    return {'schemaVersion': 1, 'sourceReadOnly': True, 'nativeExecution': False,
        'identityPolicy': TITLE_IDENTITY_POLICY, 'sourceSha256': source_hashes or {},
        'nativeProvenance': {key: report.get(key) for key in ('executionCommit',
            'sourceSha256', 'helperSourceSha256', 'runtime', 'modelRevision',
            'tokenizerSha256', 'inputHashesVerified', 'headsCheckpointSha256',
            'checkpoint', 'controls', 'config')},
        'runStatus': report.get('status'), 'runBlocker': report.get('blocker'),
        'trainingStepsRecorded': len(report.get('steps', [])), 'runWallMs': report.get('wallMs'),
        'completeFixedEvaluation': all(summaries[s][a]['completed'] == 3 for s in SPLITS for a in ARMS),
        'residentLoadMs': report.get('loadMs'), 'runMemory': report.get('memory'),
        'trainingStepWallMs': stats([s['ms'] for s in report.get('steps', [])]),
        'evaluationFullCompletionWallMs': sum(a['externalFullCompletionMs'] for a in evaluated if a['status'] == 'completed-attempt'),
        'summaries': summaries, 'selectionPairs': pairs, 'records': evaluated,
        'limitations': [
            'Small fictional teacher-grammar split; no production or human-quality claim.',
            'Strict acceptance is lexical policy plus identity; requested word ranges are separately diagnostic.',
            'Ranking/image quality has no oracle; paired selections are checked only for equality.',
            'Cumulative native peak memory includes training/previous arms; not isolated per-arm allocation.',
            'Missing arms remain in fixed three-case denominators; unavailable latency is not zero.',
            'Valid-only latency has no value when no pages pass; all-completion latency is not usable-page throughput.',
            'Run wall includes load, controls, training, evaluation and serialization; phase timers overlap.']}


def render(analysis):
    def num(value):
        return 'unavailable' if value is None else f'{value:.1f}'
    lines = ['# Direct semantic-copy independent CPU analysis', '',
        f"Run status: `{analysis['runStatus']}`. Recorded training steps: {analysis['trainingStepsRecorded']}. "
        f"All twelve fixed evaluation arms completed: {analysis['completeFixedEvaluation']}. "
        'Original artifacts were read only. Every completed arm was reassembled from unchanged raw values, '
        'its actual product/image/grouping plan with copy cleared, and frozen request/catalog; saved and current quality agree exactly.', '',
        '| Split / arm | Pages / 3 | Titles / 18 | Badges incl. NONE / 18 | Headings / 6 | Completed / missing | All median ms | Valid median ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split in SPLITS:
        for name in ARMS:
            s = analysis['summaries'][split][name]
            lines.append(f"| {split} / {name} | {s['acceptedPages']} | {s['acceptedTitles']} | {s['acceptedBadgesIncludingNone']} | {s['acceptedHeadings']} | {s['completed']} / {s['unrecorded']} | {num(s['fullCompletionMsAll']['median'])} | {num(s['fullCompletionMsValidOnly']['median'])} |")
    bad_pairs = [p for p in analysis['selectionPairs'] if not p['sameProductsImagesAndGrouping']]
    lines += ['', f"Completed paired selections compared: {len(analysis['selectionPairs'])}; product/image/grouping mismatches: {len(bad_pairs)}.", '',
              f"Total run wall: {num(analysis['runWallMs'])} ms; resident load: {num(analysis['residentLoadMs'])} ms; "
              f"summed evaluation completion wall: {num(analysis['evaluationFullCompletionWallMs'])} ms. "
              'Timers overlap; do not add these phases. Memory receipts and complete raw field/fact diagnostics are in the companion JSON.', '',
              '| Split / arm | Generated badges | Accepted NONE | Raw required NONE correct / executed | Strict correct-NONE pages / evaluable | Supported title accepted+aligned / executed | Tokens | AR steps | Peak bytes* |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split in SPLITS:
        for name in ARMS:
            s = analysis['summaries'][split][name]
            lines.append(f"| {split} / {name} | {s['acceptedGeneratedBadges']} | {s['acceptedNoneBadges']} | {s['requiredNoneRawCorrect']} / {s['requiredNoneFieldsExecuted']} | {s['strictCorrectNonePages']} / {s['strictCorrectNoneEvaluablePages']} | {s['supportedTitleAcceptedAndAligned']} / {s['supportedTitleFieldsExecuted']} | {s['outputTokensAll']['sum']} | {s['arStepsAll']['sum']} | {s['maxObservedMlxPeakBytes']} |")
    lines += ['', '*Native peaks are cumulative across training and previous arms.', '',
              '## Actual saved examples', '',
              'For each completed arm, show the first rejected field of each kind; if none fail, show its first accepted title. '
              'No wording is repaired. The JSON retains every field, selected identity, tags, observed support, universal tags, token count and stop reason.', '']
    for a in analysis['records']:
        if a['status'] != 'completed-attempt':
            continue
        examples = [next((f for f in a['fields'] if f['kind'] == kind and not f['accepted']), None)
                    for kind in ('heading', 'title', 'badge')]
        examples = [f for f in examples if f is not None]
        if not examples:
            examples = [next(f for f in a['fields'] if f['kind'] == 'title')]
        lines += [f"### {a['split']} / {a['caseId']} / {a['arm']}", '',
                  f"Page accepted: {a['quality']['allFieldsFulfilled']}; {a['actualOutputTokens']} output tokens, "
                  f"{a['actualRowArSteps']} AR steps, {a['externalFullCompletionMs']:.1f} ms.", '']
        for f in examples:
            lines += [f"- {f['field']} ({'accepted' if f['accepted'] else 'rejected'}, {f['finishReason']}, "
                      f"{f['outputTokens']}/{f['maxTokens']} tokens): {json.dumps(f['rawValue'], ensure_ascii=False)}. "
                      f"Facts: {json.dumps(f['productFacts'], ensure_ascii=False)}"]
        lines += ['', 'Strict rejections: ' + json.dumps(a['quality']['rejections']) + '.', '']
    lines += ['## Limits', ''] + ['- ' + s for s in analysis['limitations']]
    if analysis['runBlocker']:
        lines += ['', 'Recorded run blocker: ' + json.dumps(analysis['runBlocker'])]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--report', type=Path, default=ROOT / 'SEMANTIC_COPY_ANALYSIS.md')
    parser.add_argument('--json', type=Path, default=ROOT / 'SEMANTIC_COPY_ANALYSIS.json')
    args = parser.parse_args()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    input_paths = [args.results, args.data / 'manifest.json',
                   *(args.data / name for name in manifest['fileSha256']),
                   *(ROOT / name for name in ('semantic_copy_analysis.py', 'semantic_copy_direct.py',
                       'semantic_copy_training.py', 'evidence_plan.py', 'fusion.py',
                       'parallel_decode.py', 'scheduler_identity_experiment.py'))]
    if args.report.resolve() == args.json.resolve() or any(out.resolve() in {p.resolve() for p in input_paths} for out in (args.report, args.json)):
        raise ValueError('Analysis outputs must be distinct from original inputs')
    hashes = {str(p): sha(p) for p in input_paths}
    for name, expected in manifest['fileSha256'].items():
        if sha(args.data / name) != expected:
            raise ValueError('Frozen data hash changed: ' + name)
    report = json.loads(args.results.read_text())
    if report.get('inputHashesVerified') != manifest['fileSha256']:
        raise ValueError('Native report did not verify this exact frozen package')
    cases = {split: records(args.data / f'cases.{split}.jsonl') for split in SPLITS}
    result = analyze(report, cases, hashes)
    if hashes != {str(p): sha(p) for p in input_paths}:
        raise ValueError('Input/source changed during analysis; retry after writer finishes')
    args.json.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    args.report.write_text(render(result))
    print(json.dumps({'summaries': result['summaries'], 'runStatus': result['runStatus']}, indent=2))


if __name__ == '__main__':
    main()
