"""Read-only CPU audit of the fixed ten-field endpoint benchmark receipts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from .data import HERE, ROOT, sha
from .layout import assemble_payload, build_layout
from ..above_fold import CAPS, CORE, REQUIRED, payload_hash, project_first_section
from ..semantic_copy_analysis import field_details, stats

WIDTHS = (1, 3, 6)
MODES = ('frozen', 'paired-STB-adapter')
CONTROL_FIELDS = {1: ('C1',), 3: ('S1', 'T1', 'C1'),
                  6: ('S1', 'T1', 'B1', 'C1', 'T2', 'C2')}
KINDS = {'S': 'heading', 'T': 'title', 'B': 'badge', 'C': 'copy'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def serialized(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def check_roster(report, packet):
    cases = {(c['split'], c['caseId']): c for c in packet['cases']}
    require(len(cases) == len(packet['cases']) == 6, 'Exactly six distinct fixed cases required')
    require(all(sum(s == split for s, _ in cases) == 3 for split in ('dev', 'test')),
            'Three fixed cases per inspected split required')
    require(packet['widths'] == list(WIDTHS) and packet['modelModes'] == list(MODES)
            and packet['fixedArms'] == report['fixedArms'] == 36, 'Fixed 36-arm configuration changed')
    expected = {(s, cid, m, w) for s, cid in cases for m in MODES for w in WIDTHS}
    keys = [(a['split'], a['caseId'], a['mode'], a['width']) for a in report['arms']]
    require(len(keys) == len(set(keys)) == 36 and set(keys) == expected,
            'Missing, duplicate or unexpected fixed arm')
    return cases


def control_fingerprint(arm):
    return [(r['rowId'], r['rawSampledTokenIds'], r['finishReason'], r['stopTokenId'],
             r['arSteps'], r['distributionHashes']) for r in arm['rows']]


def check_controls(controls):
    require(set(controls) == set(MODES), 'Both adapter modes require controls')
    summaries = {}
    for mode in MODES:
        require(set(controls[mode]) == set(map(str, WIDTHS)), 'All actual matched widths required')
        summaries[mode] = {}
        for width in WIDTHS:
            control = controls[mode][str(width)]
            require(control['status'] == 'same-width-exact' and control['actualLanes'] == width,
                    'Control exactness/actual lane receipt changed')
            arms = control['arms']
            require(set(arms) == {'fork', 'independent', 'repeat'}, 'All three cache controls required')
            for kind, arm in arms.items():
                rows = arm['rows']; timing = arm['timing']
                require([r['fields'][0] for r in rows] == list(CONTROL_FIELDS[width])
                        and len({r['rowId'] for r in rows}) == width,
                        'Control did not use N actual distinct fields')
                require(all(r['maxTokens'] == 7 and len(r['distributionHashes']) == r['arSteps']
                            == len(r['rawSampledTokenIds']) for r in rows), 'Control distribution/sample count changed')
                admissions = timing['admissions']
                require(timing['maximumRowsEmittingTogether'] == width
                        and timing['nativePrefillStepSize'] == 256
                        and len(admissions) == 1 and admissions[0]['admitted'] == width
                        and admissions[0]['activeBound'] == width
                        and len(set(admissions[0]['rowKeys'])) == width,
                        'Control did not exercise actual width at one admission')
                require(arm['rootHash'] == arms['fork']['rootHash'], 'Control root hashes differ')
                if kind == 'independent':
                    require(arm['independentRootHashes'] == [arm['rootHash']] * width,
                            'Every independent lane must rebuild the complete same root')
                else:
                    require(not arm['independentRootHashes'], 'Unexpected independent root receipt')
                require(control_fingerprint(arm) == control_fingerprint(arms['fork']),
                        'Fatal same-width control token/distribution mismatch')
            summaries[mode][str(width)] = {'actualLanes': width,
                'fields': list(CONTROL_FIELDS[width]), 'forkRebuildRepeatExact': True,
                'rootHash': arms['fork']['rootHash'],
                'independentCompleteRootRebuilds': len(arms['independent']['independentRootHashes']),
                'sampledArSteps': sum(r['arSteps'] for r in arms['fork']['rows']),
                'distributionValuesPersisted': 'SHA-256 per full finite normalized logprob vector, not raw logits'}
    return summaries


def field_analysis(rows, replay, case):
    from ..fusion import _field_claim_errors, _normalized
    from ..scheduler_identity_experiment import assemble_values
    view = project_first_section(case['plan'])
    selected = {r['fields'][0]: r for r in rows}
    core = assemble_values([copy.deepcopy(selected[f]) for f in CORE], view, case['request'], case['catalog'])
    details = field_details([selected[f] for f in CORE], core, case)
    errors = replay['validation']['rejections']
    for f in details:
        f['acceptedByCorePolicy'] = f.pop('accepted')
        f['accepted'] = f['acceptedByCorePolicy'] and not any(e.startswith(f['field'] + ':') for e in errors)
        f['viewRejectionsForField'] = [e for e in errors if e.startswith(f['field'] + ':') or e.startswith('core:' + f['field'] + ':')]
        f['coreGlobalRejections'] = [e for e in core['quality']['rejections'] if not re.match(r'[STB]\d+:', e)]
    for i, card in enumerate(view['sections'][0]['products'], 1):
        field = f'C{i}'; row = selected[field]; raw = row['text']; product = case['catalog'][card['productId']]
        support = [f.rsplit(':', 1)[1] for f in card['approvedFactIds']]
        copied = _normalized(raw) in {_normalized(t) for t in [product.get('description', ''), product.get('category', ''),
            *(v.get('imageDescription', '') for v in product['presentationVariants'])] if t}
        own_errors = [e for e in errors if e.startswith(field + ':')]
        details.append({'field': field, 'kind': 'copy', 'rawValue': raw,
            'accepted': replay['validation']['copyChecks'][field]['copyAccepted'] and not own_errors,
            'finishReason': row['finishReason'], 'wordCount': len(raw.split()),
            'outputTokens': len(row['tokenIds']), 'arSteps': row['arSteps'], 'maxTokens': row['maxTokens'],
            'productFacts': {'selectedIdentity': product['title'], 'productId': card['productId'],
                'imageCaptionId': card['imageCaptionId'], 'description': product['description'],
                'tags': product['tags'], 'observedSupport': support},
            'factualClaimErrors': _field_claim_errors(raw, product, case['catalog']),
            'copiesCatalogBody': copied, 'observedSupportAvailable': bool(support),
            'observedSupportMentioned': any(re.search(r'\b' + re.escape(_normalized(t)) + r'\b', _normalized(raw)) for t in support),
            'viewRejectionsForField': own_errors})
    for f in details:
        row = selected[f['field']]
        f['sampledTokensIncludingStop'] = len(row['rawSampledTokenIds'])
        f['stopTokenId'] = row['stopTokenId']
        f['nativeSampledCapMet'] = len(row['rawSampledTokenIds']) <= row['maxTokens']
    return details


def check_attempt(arm, case, decode=None, eos_ids=None):
    rows = arm['rawRows']; root, jobs, evidence = build_layout(case['plan'], case['request'], case['catalog'])
    require([r['fields'][0] for r in rows] == list(REQUIRED), 'All same ten required fields must remain ordered')
    require(sum(r['maxTokens'] for r in rows) == 208, '208 sampled-token cap changed')
    for row, job in zip(rows, jobs, strict=True):
        require(all(row.get(k) == job[k] for k in ('rowId', 'key', 'fields', 'kind', 'maxTokens', 'messages')),
                'Generated job differs from public-only frozen layout')
        require(row['arSteps'] == len(row['rawSampledTokenIds']) <= row['maxTokens'], 'Sampled token/AR/cap receipt mismatch')
        if decode is not None:
            require(decode(row['tokenIds']) == row['text'], 'Raw text differs from actual sampled token decoding')
        if eos_ids is not None and row['finishReason'] == 'stop':
            require(row['stopTokenId'] in eos_ids, 'Stop token is not a tokenizer EOS')
    require(arm['publicEvidence'] == evidence and arm['inputInventory'] == case['inventory'],
            'Public inputs changed or teacher target entered serving layout')
    replay = assemble_payload(copy.deepcopy(rows), case['plan'], case['request'], case['catalog'], require_native_receipts=True)
    for key in ('validation', 'payload', 'payloadSha256', 'rawRows', 'nativeReceiptsChecked'):
        require(replay[key] == arm[key], 'Saved/current reassembly differs: ' + key)
    encoded = serialized(arm['payload']); digest = hashlib.sha256(encoded).hexdigest()
    require(digest == payload_hash(arm['payload']) == arm['payloadSha256'] == arm['payloadCompletionReceipt']['payloadSha256']
            and len(encoded) == arm['payloadBytes'], 'Actual serialized payload hash/byte receipt mismatch')
    receipt = arm['payloadCompletionReceipt']; ready = receipt['validatedAndSerializedAtMs']
    require(ready == arm['attemptCompletionMs'] and ready >= max(receipt['metadataReadyAtMs'], arm['lastRequiredTokenConservativeLowerBoundMs']),
            'Payload-ready receipt predates required completion/metadata')
    require(arm['copyStagePayloadReadyMs'] == (ready if replay['validation']['valid'] else None),
            'Invalid payload must have no accepted ready time')
    require(sum(len(r['tokenIds']) for r in rows) == arm['actualOutputTokens']
            and sum(r['arSteps'] for r in rows) == arm['actualArSteps'], 'Token/AR total mismatch')
    timing = arm['timing']; width = arm['width']
    require(timing['nativePrefillStepSize'] == 256 and timing['maximumRowsEmittingTogether'] <= width
            and sum(a['admitted'] for a in timing['admissions']) == 10
            and all(0 < a['admitted'] <= a['activeBound'] <= width for a in timing['admissions']), 'Active admission bound changed')
    require(max(arm['memory']['mlxActiveBytes'], arm['memory']['mlxPeakBytes']) <= 7 * 2**30, 'Seven-GiB memory guard exceeded')
    return {'split': arm['split'], 'caseId': arm['caseId'], 'mode': arm['mode'], 'width': width,
        'status': arm['status'], 'validView': replay['validation']['valid'], 'rejections': replay['validation']['rejections'],
        'savedValidationPayloadAndRawRowsExact': True, 'noTeacherTargetInPublicJobs': True,
        'metadataBindingValidated': not any(':payload-' in e or e.startswith('payload-') for e in replay['validation']['rejections']),
        'actualSerializedPayloadSha256': digest, 'actualSerializedBytes': len(encoded),
        'readyReceipt': receipt, 'copyStagePayloadReadyMs': arm['copyStagePayloadReadyMs'],
        'attemptCompletionMs': ready, 'lastRequiredTokenConservativeLowerBoundMs': arm['lastRequiredTokenConservativeLowerBoundMs'],
        'actualOutputTokens': arm['actualOutputTokens'], 'actualArSteps': arm['actualArSteps'],
        'fields': field_analysis(rows, replay, case), 'timing': timing, 'preparation': arm['preparation'], 'memory': arm['memory'],
        'metadata': [{k: c[k] for k in ('productId', 'imageCaptionId', 'imageKey', 'originalName', 'price', 'shippingCost')}
                     for c in arm['payload']['cards']]}


def analyze(report, packet, decode=None, eos_ids=None):
    require(report['status'] in ('completed', 'blocked'), 'Wait for terminal native report before overall analysis')
    cases = check_roster(report, packet); controls = check_controls(report['controls'])
    records = []
    for arm in report['arms']:
        if arm['status'] == 'completed-attempt':
            records.append(check_attempt(arm, cases[(arm['split'], arm['caseId'])], decode, eos_ids))
        else:
            records.append({k: arm.get(k) for k in ('split', 'caseId', 'mode', 'width', 'status', 'error')})
    summary = []
    for split in ('dev', 'test'):
        for mode in MODES:
            for width in WIDTHS:
                group = [a for a in records if (a['split'], a['mode'], a['width']) == (split, mode, width)]
                completed = [a for a in group if a['status'] == 'completed-attempt']; valid = [a for a in completed if a['validView']]
                fields = [f for a in completed for f in a['fields']]
                support = [f for f in fields if f.get('observedSupportAvailable')] if fields else []
                none_required = [f for f in fields if f['kind'] == 'badge' and f['noneRequired']]
                summary.append({'split': split, 'mode': mode, 'width': width, 'fixedViews': 3,
                    'completed': len(completed), 'failed': sum(a['status'] == 'failed' for a in group),
                    'notAttempted': sum(a['status'] == 'not-attempted' for a in group), 'validViews': len(valid),
                    'acceptedFields': {kind: sum(f['accepted'] for f in fields if f['kind'] == kind) for kind in KINDS.values()},
                    'requestedWordRangeMissesDiagnosticOnly': {kind: sum(f.get('requestedWordRangeMet') is False for f in fields if f['kind'] == kind)
                                                             for kind in ('title', 'heading')},
                    'fixedFieldDenominators': {'heading': 3, 'title': 9, 'badge': 9, 'copy': 9},
                    'requiredNoneRawCorrect': sum(f['rawExactNone'] for f in none_required),
                    'requiredNoneExecuted': len(none_required),
                    'supportedTitleRawMention': sum(f['observedSupportMentioned'] for f in support if f['kind'] == 'title'),
                    'supportedCopyRawMention': sum(f['observedSupportMentioned'] for f in support if f['kind'] == 'copy'),
                    'supportedTitleExecuted': sum(f['kind'] == 'title' for f in support),
                    'supportedCopyExecuted': sum(f['kind'] == 'copy' for f in support),
                    'completionMsAll': stats([a['attemptCompletionMs'] for a in completed]),
                    'readyMsValidOnly': stats([a['copyStagePayloadReadyMs'] for a in valid]),
                    'outputTokensAll': stats([a['actualOutputTokens'] for a in completed]),
                    'arStepsAll': stats([a['actualArSteps'] for a in completed]),
                    'maximumCumulativePeakBytes': max((a['memory']['mlxPeakBytes'] for a in completed), default=None)})
    pairs = []
    for split, cid in cases:
        for mode in MODES:
            group = {a['width']: a for a in records if (a['split'], a['caseId'], a['mode']) == (split, cid, mode)
                     and a['status'] == 'completed-attempt'}
            for width in (3, 6):
                if 1 not in group or width not in group: continue
                base, batch = group[1], group[width]
                both = base['validView'] and batch['validView']
                pairs.append({'split': split, 'caseId': cid, 'mode': mode, 'comparedWidth': width,
                    'bothCompleteViewsValid': both, 'sequentialCompletionMs': base['attemptCompletionMs'],
                    'batchCompletionMs': batch['attemptCompletionMs'],
                    'validPayloadSpeedup': base['attemptCompletionMs'] / batch['attemptCompletionMs'] if both else None,
                    'exactRawStringsAcrossWidths': [f['rawValue'] for f in base['fields']] == [f['rawValue'] for f in batch['fields']],
                    'crossWidthStringDifferenceIsNotStateControlFailure': True})
    return {'schemaVersion': 1, 'scope': 'fixed three-card ten-field copy-stage endpoint; inspected dev/test replay',
        'nativeExecutionByAnalyzer': False, 'sourceArtifactsReadOnly': True, 'runStatus': report['status'],
        'runBlocker': report.get('blocker'), 'runWallMs': report['wallMs'], 'residentLoadMs': report.get('residentLoadMs'),
        'runMemory': report.get('memory'), 'fixedArms': 36, 'controls': controls, 'summaries': summary, 'matchedCaseWidthPairs': pairs, 'records': records,
        'limitations': ['Locked recorded ranking/image/section decisions; no ranking latency or ranking-quality measurement.',
            'Actual serialized validated payload handoff; browser image loading, rendering and paint are excluded.',
            'All parent dev/test cases were inspected during data preparation; not fresh sealed holdouts.',
            'The paired-STB adapter received no C supervision; this is transfer to three fresh body-copy jobs.',
            'Same ten fields/caps at widths 1/3/6. Cross-width strings may differ numerically; only same-width controls demand exactness.',
            'Policy acceptance is lexical/factual/identity evidence, not human fluency/usefulness or conversion evidence.',
            'Three cases per split and one timed attempt per shape: no throughput stability or latency-tail guarantee.',
            'Cumulative peak includes controls and prior arms; it is not an isolated per-arm peak.',
            'No comparison against dropping below-fold fields, full-page speedup or full-request view time.']}


def render(result):
    def number(v): return '—' if v is None else f'{v:.1f}'
    lines = ['# Independent complete-card endpoint audit', '', result['scope'] + '.', '',
        f"Native status `{result['runStatus']}`; fixed denominator 36 arms. All saved raw values were reassembled under the unchanged strict policy; "
        'saved payloads, validation, sampled caps, serialization hashes, metadata binding and ready receipts agree exactly.', '',
        'Matched-width controls cover both modes and actual widths 1/3/6, with full-root independent rebuilds for every lane. '
        'Fork/rebuild/repeat sampled IDs and full-distribution hashes are exact within each width. Controls are diagnostic and excluded from quality denominators.', '',
        '| Split / mode / width | Valid views / 3 | Titles / 9 | Badges incl NONE / 9 | Headings / 3 | Body copies / 9 | Completion median all ms | Ready median valid ms | Tokens median | AR steps median |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in result['summaries']:
        c = s['acceptedFields']
        lines.append(f"| {s['split']} / {s['mode']} / {s['width']} | {s['validViews']} | {c['title']} | {c['badge']} | {c['heading']} | {c['copy']} | {number(s['completionMsAll']['median'])} | {number(s['readyMsValidOnly']['median'])} | {number(s['outputTokensAll']['median'])} | {number(s['arStepsAll']['median'])} |")
    lines += ['', f"Native run wall {result['runWallMs']:.1f} ms; resident load {number(result['residentLoadMs'])} ms. "
        'Run wall includes load and controls after preparation/model-byte verification. Attempt completion includes fresh public root, repeated fact suffixes, generation-stream synchronization, '
        'strict validation and actual JSON serialization. Invalid views contribute only to all-attempt completion, never accepted ready latency. '
        'The last-required-token bridge is only a conservative lower bound; ready is measured directly after complete payload serialization.', '',
        '## Per-case complete view outcomes', '', '| Split / case | Frozen w1 / w3 / w6 | Paired STB w1 / w3 / w6 |', '| --- | --- | --- |']
    keys = list(dict.fromkeys((a['split'], a['caseId']) for a in result['records']))
    for split, cid in keys:
        columns = []
        for mode in MODES:
            group = [next(a for a in result['records'] if (a['split'], a['caseId'], a['mode'], a['width']) == (split, cid, mode, w)) for w in WIDTHS]
            columns.append(' / '.join(('valid' if a['validView'] else 'invalid') + f" {a['attemptCompletionMs']:.0f} ms" if a['status'] == 'completed-attempt' else a['status'] for a in group))
        lines.append(f"| {split} / {cid} | {columns[0]} | {columns[1]} |")
    lines += ['', '## Matched valid-view timing comparisons', '',
        'Ratios below compare the same case and model mode only when both complete payloads pass. '
        'Different valid-case subsets must not be described as a matched speedup. One attempt per shape is not a stable throughput estimate.', '',
        '| Split / case / mode | Width | Sequential ms | Batched ms | Sequential / batched |', '| --- | ---: | ---: | ---: | ---: |']
    for pair in result['matchedCaseWidthPairs']:
        if pair['bothCompleteViewsValid']:
            lines.append(f"| {pair['split']} / {pair['caseId']} / {pair['mode']} | {pair['comparedWidth']} | "
                f"{pair['sequentialCompletionMs']:.1f} | {pair['batchCompletionMs']:.1f} | {pair['validPayloadSpeedup']:.3f} |")
    if not any(p['bothCompleteViewsValid'] for p in result['matchedCaseWidthPairs']):
        lines += ['', 'No matched pair has two accepted complete payloads; no usable-view speedup is established.']
    lines += ['', '## Raw body-copy evidence', '']
    # Fixed width 3 shows every case/model, rather than selecting good examples.
    for a in result['records']:
        if a['status'] != 'completed-attempt' or a['width'] != 3: continue
        lines += [f"### {a['split']} / {a['caseId']} / {a['mode']}", '']
        for f in a['fields']:
            if f['kind'] != 'copy': continue
            facts = f['productFacts']
            lines.append(f"- {f['field']} raw {json.dumps(f['rawValue'], ensure_ascii=False)}; selected `{facts['selectedIdentity']}`, "
                f"tags {json.dumps(facts['tags'])}, observed support {json.dumps(facts['observedSupport'])}; "
                f"accepted={f['accepted']}, stop={f['finishReason']}, text tokens={f['outputTokens']}, sampled incl stop={f['sampledTokensIncludingStop']}/{f['maxTokens']}; "
                f"rejections={json.dumps(f['viewRejectionsForField'])}.")
    lines += ['', '## Failed raw fields', '']
    for a in result['records']:
        if a['status'] != 'completed-attempt': continue
        for f in a['fields']:
            if f['accepted']: continue
            lines.append(f"- {a['split']} / {a['caseId']} / {a['mode']} / w{a['width']} / {f['field']}: "
                f"{json.dumps(f['rawValue'], ensure_ascii=False)}; stop={f['finishReason']}, sampled={f['sampledTokensIncludingStop']}/{f['maxTokens']}; "
                f"rejections={json.dumps(f.get('viewRejectionsForField', []))}; factual={json.dumps(f['factualClaimErrors'])}; "
                f"core/global={json.dumps(f.get('coreGlobalRejections', []))}.")
    lines += ['', '## Limits', ''] + ['- ' + item for item in result['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--results', type=Path, default=ROOT.parents[2] / 'data/ranker/pretrained/above-fold-copy-v1/results.json')
    p.add_argument('--packet', type=Path, default=HERE / 'BENCHMARK_PACKET_V1.json')
    p.add_argument('--json', type=Path, default=HERE / 'BENCHMARK_AUDIT.json')
    p.add_argument('--report', type=Path, default=HERE / 'BENCHMARK_AUDIT.md')
    args = p.parse_args()
    report = json.loads(args.results.read_text()); packet = json.loads(args.packet.read_text())
    require(report['status'] in ('completed', 'blocked'), 'Wait for terminal native report')
    inputs = [args.results, args.packet, Path(packet['preflightPath']), Path(packet['adapter']['fitReportPath']),
        Path(packet['adapter']['checkpointPath']), HERE / 'audit.py', *(ROOT / n for n in packet['sources'])]
    require(len({args.json.resolve(), args.report.resolve()}) == 2 and not any(out.resolve() in {x.resolve() for x in inputs}
            for out in (args.json, args.report)), 'Audit output cannot overwrite source inputs')
    before = {str(x): sha(x) for x in inputs}
    require(report['packetSha256'] == sha(args.packet), 'Native packet hash differs')
    require(packet['sources'] == report['sourceSha256'] == {n: sha(ROOT / n) for n in packet['sources']}, 'Frozen source receipt differs')
    require(sha(Path(packet['preflightPath'])) == packet['preflightSha256'], 'Preflight hash changed')
    require(sha(Path(packet['adapter']['fitReportPath'])) == packet['adapter']['fitReportSha256']
            and sha(Path(packet['adapter']['checkpointPath'])) == packet['adapter']['checkpointSha256'], 'Final adapter receipt changed')
    require(sha(HERE / 'manifest.json') == packet['datasetManifestSha256'], 'Frozen dataset manifest changed')
    for name, expected in packet['dataSha256'].items():
        require(sha(HERE / name) == expected, 'Frozen teacher data changed: ' + name)
        before[str(HERE / name)] = expected
    before[str(HERE / 'manifest.json')] = sha(HERE / 'manifest.json')
    for mode, widths in report['controls'].items():
        for width, control in widths.items():
            for kind, arm in control['arms'].items():
                path = args.results.parent / ('controls-' + mode) / f'width-{width}-{kind}.json'
                require(json.loads(path.read_text()) == arm, 'Native control file differs from embedded receipt')
                before[str(path)] = sha(path)
    # Local tokenizer is CPU-only; no MLX/model module, download or inference.
    from transformers import AutoTokenizer
    from .benchmark import MODEL
    for name, expected in packet['tokenizerSha256'].items():
        require(sha(MODEL / name) == expected, 'Pinned tokenizer changed')
        before[str(MODEL / name)] = expected
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    result = analyze(report, packet, tokenizer.decode, {tokenizer.eos_token_id})
    result['artifactSha256'] = before
    result['executionCommit'] = report['executionCommit']
    result['originalInputsUnchangedAfterAudit'] = before == {str(x): sha(Path(x)) for x in before}
    require(result['originalInputsUnchangedAfterAudit'], 'Original evidence changed during audit')
    args.json.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    args.report.write_text(render(result))
    print(json.dumps({'runStatus': result['runStatus'], 'summaries': result['summaries']}, indent=2))


if __name__ == '__main__': main()
