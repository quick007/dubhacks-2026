"""CPU audit of saved branch-local copy; keep policy and strict-wire counts apart."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

from .semantic_copy_analysis import ROOT, field_details, pending, sha, stats

ARMS = ('frozen', 'global-copy-adapter')
PACKET = ROOT / 'BRANCH_LOCAL_CASES.json'
RESULTS = ROOT / 'BRANCH_LOCAL_RESULTS.json'


def strict_fields(rows, replay, case):
    """Additional raw-row accounting only; never alter saved/replayed quality."""
    fields = field_details(rows, replay, case)
    global_ok = all(r['finishReason'] == 'stop' for r in rows) and 'unfinished-or-markup-output' not in replay['quality']['rejections']
    for entry in fields:
        legacy = entry.pop('accepted'); errors = []
        if not legacy: errors.append('not-accepted-by-original-policy')
        if not global_ok: errors.append('original-global-finish-or-plain-output-failed')
        errors += ['wire:' + e for e in entry['protocolRejections']]
        errors += ['factual:' + e for e in entry['factualClaimErrors']]
        if entry['kind'] == 'title':
            errors += ['identity:' + e for e in entry['identityErrors']]
            if entry['copiesApprovedTitle']: errors.append('copies-approved-title')
            if entry['observedSupportAvailable'] and not entry['observedSupportMentioned']:
                errors.append('missing-observed-supported-tag')
        if entry['kind'] == 'badge':
            if entry['noneRequired'] and not entry['rawExactNone']:
                errors.append('raw-NONE-required')
            if not entry['rawExactNone'] and entry['observedSupportAvailable'] and not entry['observedSupportMentioned']:
                errors.append('missing-observed-supported-tag')
        sentinel = entry['kind'] == 'badge' and entry['rawExactNone']
        if not sentinel and not 5 <= len(entry['rawValue']) <= 90:
            errors.append('raw-field-character-length')
        entry.update(acceptedByOriginalFieldAccounting=legacy,
                     strictWireAndSemanticAccepted=not errors,
                     strictRawRejections=errors)
    return fields


def analyze(report, packet, hashes=None):
    from .scheduler_identity_experiment import assemble_values_legacy_v1 as assemble_values
    if packet['casesAlreadyInspected'] is not True or report['endToEndRequestTiming'] is not False:
        raise ValueError('Copy-stage inspected replay scope changed')
    cases = {(c['split'], c['caseId']): c for c in packet['cases']}
    if len(cases) != 6 or any(sum(s == split for s, _ in cases) != 3 for split in ('dev', 'test')):
        raise ValueError('Fixed inspected case denominator changed')
    seen = set(); output = []
    for arm in report['arms']:
        key = (arm['split'], arm['caseId'], arm['arm'])
        if key[:2] not in cases or key[2] not in ARMS or key in seen:
            raise ValueError('Unexpected/duplicate branch arm')
        seen.add(key); case = cases[key[:2]]
        if arm['status'] != 'completed-copy-attempt':
            raise ValueError('Only completed recorded arms supported; absent arms stay missing')
        if arm['endToEndRequestTiming'] is not False or arm['totalOutputBudget'] != 224 or len(arm['rows']) != 14:
            raise ValueError('Copy stage or unchanged fourteen-field/224-cap contract changed')
        expected = {f'S{i}': ('heading', 16) for i in (1, 2)} | {
            f'{letter}{i}': (kind, cap) for i in range(1, 7)
            for letter, kind, cap in (('T', 'title', 24), ('B', 'badge', 8))}
        rows = arm['rows']; labels = [r['fields'][0] for r in rows if len(r['fields']) == 1]
        if len(labels) != 14 or set(labels) != set(expected) or any((r['kind'], r['maxTokens']) != expected[r['fields'][0]] for r in rows):
            raise ValueError('Trusted row membership/caps changed')
        plan = pending(case['plan'])
        def bindings(p):
            return [[(c['productId'], c['imageCaptionId'], c['approvedFactIds']) for c in s['products']] for s in p['sections']]
        if bindings(plan) != bindings(arm['pagePlan']): raise ValueError('Locked product/image/evidence plan changed')
        replay = assemble_values(copy.deepcopy(rows), plan, case['request'], case['catalog'])
        if replay['quality'] != arm['quality']:
            raise ValueError('Original-policy saved/current quality mismatch: ' + str(key))
        tokens = sum(len(r['tokenIds']) for r in rows); ar = sum(r['arSteps'] for r in rows)
        if tokens != arm['actualOutputTokens'] or ar != arm['actualRowArSteps']:
            raise ValueError('Raw token/AR receipt mismatch')
        details = strict_fields(rows, replay, case)
        output.append({'split': key[0], 'caseId': key[1], 'arm': key[2],
            'savedQualityMatchesCurrentOriginalPolicy': True, 'originalQuality': arm['quality'],
            'strictWireAndSemanticPageAccepted': arm['quality']['allFieldsFulfilled'] and all(f['strictWireAndSemanticAccepted'] for f in details),
            'fields': details, 'actualOutputTokens': tokens, 'actualRowArSteps': ar,
            'copyStageMs': arm['externalCopyStageMs'], 'timing': arm['timing'], 'memory': arm['memory'],
            'inputInventory': arm['inputInventory']})
    summary = {}
    for split in ('dev', 'test'):
        summary[split] = {}
        for arm_name in ARMS:
            selected = [a for a in output if a['split'] == split and a['arm'] == arm_name]
            fields = [f for a in selected for f in a['fields']]
            valid = [a for a in selected if a['originalQuality']['allFieldsFulfilled']]
            strict_valid = [a for a in selected if a['strictWireAndSemanticPageAccepted']]
            counts = {}
            for label, flag in (('originalFieldAccounting', 'acceptedByOriginalFieldAccounting'),
                                ('strictWireAndSemantic', 'strictWireAndSemanticAccepted')):
                counts[label] = {kind: sum(f[flag] for f in fields if f['kind'] == kind) for kind in ('title', 'badge', 'heading')}
                counts[label]['generatedBadge'] = sum(f[flag] and not f['rawExactNone'] for f in fields if f['kind'] == 'badge')
                counts[label]['acceptedNoneBadge'] = sum(f[flag] and f['rawExactNone'] for f in fields if f['kind'] == 'badge')
                counts[label]['supportedTitleAcceptedAndAligned'] = sum(f[flag] and f['observedSupportMentioned'] for f in fields if f['kind'] == 'title' and f['observedSupportAvailable'])
            summary[split][arm_name] = {'fixedInspectedCases': 3, 'completed': len(selected), 'missing': 3 - len(selected),
                'originalAcceptedPages': len(valid), 'strictAcceptedPages': len(strict_valid),
                'fixedFieldDenominators': {'title': 18, 'badge': 18, 'heading': 6}, **counts,
                'originalSavedGeneratedBadgeCount': sum(a['originalQuality']['generatedBadgeCount'] for a in selected),
                'strictCorrectNonePagesOriginalPolicy': sum(a['originalQuality']['correctNone'] is True for a in selected),
                'requiredNoneRawCorrect': sum(f['rawExactNone'] for f in fields if f['kind'] == 'badge' and f['noneRequired']),
                'requiredNoneFieldsExecuted': sum(f['kind'] == 'badge' and f['noneRequired'] for f in fields),
                'protocolMalformedRows': sum(bool(f['protocolRejections']) for f in fields),
                'originallyAcceptedMalformedRows': sum(f['acceptedByOriginalFieldAccounting'] and bool(f['protocolRejections']) for f in fields),
                'copyStageMsAll': stats([a['copyStageMs'] for a in selected]),
                'copyStageMsValidOnlyOriginal': stats([a['copyStageMs'] for a in valid]),
                'copyStageMsValidOnlyStrict': stats([a['copyStageMs'] for a in strict_valid]),
                'outputTokens': stats([a['actualOutputTokens'] for a in selected]),
                'arSteps': stats([a['actualRowArSteps'] for a in selected]),
                'cumulativeNativePeakBytes': max((a['memory']['mlxPeakBytes'] for a in selected), default=None)}
    return {'schemaVersion': 1, 'analysisScope': 'inspected old dev/test replay; copy stage only; original saved counts plus additional stricter raw-row accounting',
        'sourceReadOnly': True, 'nativeExecution': False, 'runStatus': report['status'],
        'runWallMs': report['wallMs'], 'residentLoadMs': report.get('loadMs'), 'runMemory': report.get('memory'),
        'sourceSha256': hashes or {}, 'nativeSourceReceipt': {k: report.get(k) for k in ('executionCommit', 'sourceSha256', 'packetSha256', 'learnedCheckpointSha256')},
        'completeFixedRoster': len(output) == 12, 'summaries': summary, 'records': output,
        'limitations': ['No new training; global-copy adapter checkpoint is reused under a different copy layout.',
            'All six requests were previously inspected; this is not fresh generalization evidence.',
            'Copy-stage times exclude ranking and are not comparable with direct end-to-end request completion.',
            'Original counts preserve the first-line parsing/provenance defect; stricter counts do not rewrite them.',
            'Global unfinished-output conservatism is retained in both accounts.',
            'Strict raw checks still do not establish human usefulness or fluency.',
            'Cumulative native peak includes controls/previous arms, not isolated arm memory.']}


def render(result):
    def number(v): return 'unavailable' if v is None else f'{v:.1f}'
    lines = ['# Branch-local copy independent audit', '', result['analysisScope'] + '.', '',
        f"Run status `{result['runStatus']}`; all twelve arms recorded: {result['completeFixedRoster']}. "
        'Raw values were reassembled with the locked packet plan/request/catalog under the original policy; every saved quality record matches exactly. '
        'Original fields and a separate stricter raw-wire account follow; no saved outputs or pass rates are rewritten.', '',
        '| Inspected split / arm | Original pages / 3 | Strict pages / 3 | Titles original→strict / 18 | Badges incl NONE original→strict / 18 | Headings original→strict / 6 | Malformed rows / originally counted | Copy median all ms | Copy median valid ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split, arms in result['summaries'].items():
        for name, s in arms.items():
            old = s['originalFieldAccounting']; new = s['strictWireAndSemantic']
            lines.append(f"| {split} / {name} | {s['originalAcceptedPages']} | {s['strictAcceptedPages']} | {old['title']}→{new['title']} | {old['badge']}→{new['badge']} | {old['heading']}→{new['heading']} | {s['protocolMalformedRows']} / {s['originallyAcceptedMalformedRows']} | {number(s['copyStageMsAll']['median'])} | {number(s['copyStageMsValidOnlyStrict']['median'])} |")
    lines += ['', f"Run wall {result['runWallMs']:.1f} ms; load {number(result['residentLoadMs'])} ms. "
              'Copy timing includes fresh root, repeated fact suffixes, generation and validation, and excludes ranking. '
              'Do not compare these numbers directly with the old global end-to-end times. Complete token/AR, memory, '
              'input inventory, original policy, raw fields/facts and stricter rejections are retained in the companion JSON.', '',
              '## Malformed rows counted by original field provenance', '']
    for arm in result['records']:
        for field in arm['fields']:
            if field['protocolRejections']:
                lines += [f"- {arm['split']} / {arm['caseId']} / {arm['arm']} / {field['field']}: "
                    f"raw {json.dumps(field['rawValue'], ensure_ascii=False)}; "
                    f"stop={field['finishReason']}, tokens={field['outputTokens']}/{field['maxTokens']}; "
                    f"original-counted={field['acceptedByOriginalFieldAccounting']}, "
                    f"strict-counted={field['strictWireAndSemanticAccepted']}; "
                    f"wire={json.dumps(field['protocolRejections'])}; facts={json.dumps(field['productFacts'], ensure_ascii=False)}."]
    lines += ['', '## Suggested prospective codec correction', '',
        'Keep original raw values and wire errors as immutable evidence. After ordinary validation, reject the entire trusted field whose raw value is empty, multiline or contains a generated field label: '
        'set its returned field text to null and provenance to rejected, and recompute generated counts and accepted alignment from that rejected plan. '
        'Never retain a malformed row’s first parsed line as an accepted field. Do not strip/repair/retry text, invent NONE or rewrite old reports. '
        'Keep the existing global finish rule. This is a proposed fix, not a codec or validator edit in this audit.', '',
        'Regression coverage should include first-line-valid malformed heading/title/badge, supported and required NONE rows, '
        'embedded field labels and Unicode line separators; assert raw bytes preserved, affected provenance rejected, '
        'generated counts excluded and page false.', '', '## Limits', '']
    lines += ['- ' + s for s in result['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--packet', type=Path, default=PACKET)
    parser.add_argument('--report', type=Path, default=ROOT / 'BRANCH_LOCAL_ANALYSIS.md')
    parser.add_argument('--json', type=Path, default=ROOT / 'BRANCH_LOCAL_ANALYSIS.json')
    args = parser.parse_args()
    paths = [args.results, args.packet, *(ROOT / n for n in ('branch_local_analysis.py', 'semantic_copy_analysis.py',
        'branch_local_benchmark.py', 'branch_local_copy.py', 'scheduler_identity_experiment.py', 'parallel_decode.py', 'evidence_plan.py', 'fusion.py'))]
    if args.report.resolve() == args.json.resolve() or any(out.resolve() in {p.resolve() for p in paths} for out in (args.report, args.json)):
        raise ValueError('Output must not overwrite original artifact/source')
    hashes = {str(p): sha(p) for p in paths}
    report = json.loads(args.results.read_text()); packet = json.loads(args.packet.read_text())
    if report['status'] != 'completed': raise ValueError('Wait for final native save')
    if report['packetSha256'] != sha(args.packet): raise ValueError('Frozen packet hash mismatch')
    if report['sourceSha256'] != sha(ROOT / 'branch_local_benchmark.py'): raise ValueError('Native source receipt mismatch')
    if packet['layoutSourceSha256'] != sha(ROOT / 'branch_local_copy.py'): raise ValueError('Layout receipt mismatch')
    result = analyze(report, packet, hashes)
    if hashes != {str(p): sha(p) for p in paths}: raise ValueError('Input/source changed during audit')
    args.json.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    args.report.write_text(render(result))
    print(json.dumps(result['summaries'], indent=2))


if __name__ == '__main__': main()
