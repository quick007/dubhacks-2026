"""CPU corruption tests for independent endpoint evidence checks; no model load."""
import copy
import json
from pathlib import Path
import unittest

from .audit import MODES, WIDTHS, check_attempt, check_controls, check_roster, serialized
from .data import HERE, records
from .layout import assemble_payload, build_layout


def teacher_fixture():
    source = records(HERE / 'cases.dev.jsonl')[0]
    case = {'split': 'dev', 'caseId': source['caseId'], 'plan': source['fullSourcePagePlan'],
            'request': source['request'], 'catalog': source['catalog'], 'inventory': {'CPUFixture': True}}
    _, jobs, evidence = build_layout(case['plan'], case['request'], case['catalog'])
    targets = {f['field']: f['targetValue'] for f in records(HERE / 'fields.dev.jsonl') if f['caseId'] == source['caseId']}
    rows = [{**r, 'text': targets[r['fields'][0]], 'tokenIds': [1],
             'rawSampledTokenIds': [1, 2], 'stopTokenId': 2, 'arSteps': 2,
             'completedMs': i + 1., 'finishReason': 'stop'} for i, r in enumerate(jobs)]
    result = assemble_payload(rows, case['plan'], case['request'], case['catalog'], require_native_receipts=True)
    arm = {**result, 'split': 'dev', 'caseId': case['caseId'], 'mode': 'frozen', 'width': 3,
        'status': 'completed-attempt', 'inputInventory': case['inventory'], 'publicEvidence': evidence,
        'payloadBytes': len(serialized(result['payload'])),
        'payloadCompletionReceipt': {'payloadSha256': result['payloadSha256'], 'metadataReadyAtMs': 1., 'validatedAndSerializedAtMs': 20.},
        'attemptCompletionMs': 20., 'lastRequiredTokenConservativeLowerBoundMs': 15.,
        'copyStagePayloadReadyMs': 20. if result['validation']['valid'] else None,
        'actualOutputTokens': 10, 'actualArSteps': 20,
        'timing': {'nativePrefillStepSize': 256, 'maximumRowsEmittingTogether': 3,
            'admissions': [{'admitted': 3, 'activeBound': 3}, {'admitted': 3, 'activeBound': 3},
                           {'admitted': 3, 'activeBound': 3}, {'admitted': 1, 'activeBound': 1}]},
        'preparation': {}, 'memory': {'mlxActiveBytes': 10, 'mlxPeakBytes': 11}}
    return arm, case


class AuditTests(unittest.TestCase):
    def test_teacher_fixture_is_only_cpu_check_not_model_evidence(self):
        arm, case = teacher_fixture()
        result = check_attempt(arm, case)
        self.assertTrue(result['validView'])
        self.assertEqual(sum(f['kind'] == 'copy' and f['accepted'] for f in result['fields']), 3)
        self.assertTrue(all(f['accepted'] for f in result['fields']))

    def test_early_ready_or_changed_catalog_metadata_is_rejected(self):
        arm, case = teacher_fixture(); altered = copy.deepcopy(arm)
        altered['payloadCompletionReceipt']['validatedAndSerializedAtMs'] = 9.
        altered['attemptCompletionMs'] = 9.
        with self.assertRaisesRegex(ValueError, 'predates'): check_attempt(altered, case)
        altered = copy.deepcopy(arm); altered['payload']['cards'][0]['price'] += 1
        with self.assertRaisesRegex(ValueError, 'reassembly differs'): check_attempt(altered, case)

    def test_teacher_text_in_job_or_changed_raw_decoding_is_rejected(self):
        arm, case = teacher_fixture(); changed = copy.deepcopy(arm)
        changed['rawRows'][0]['messages'][0]['content'] += ' Teacher answer: ' + changed['rawRows'][0]['text']
        with self.assertRaisesRegex(ValueError, 'public-only'): check_attempt(changed, case)
        with self.assertRaisesRegex(ValueError, 'sampled token decoding'):
            check_attempt(arm, case, lambda _: 'wrong text')

    def test_fixed_roster_cannot_silently_drop_failure(self):
        packet = json.loads((HERE / 'BENCHMARK_PACKET_V1.json').read_text())
        report = {'fixedArms': 36, 'arms': [{'split': c['split'], 'caseId': c['caseId'], 'mode': m, 'width': w}
            for c in packet['cases'] for m in MODES for w in WIDTHS]}
        check_roster(report, packet)
        report['arms'].pop()
        with self.assertRaisesRegex(ValueError, 'Missing'): check_roster(report, packet)

    def test_claimed_width_without_actual_distinct_rows_is_rejected(self):
        from .audit import CONTROL_FIELDS
        controls = {}
        for mode in MODES:
            controls[mode] = {}
            for width in WIDTHS:
                arms = {}
                for kind in ('fork', 'independent', 'repeat'):
                    rows = [{'rowId': f, 'fields': [f], 'maxTokens': 7, 'rawSampledTokenIds': [1],
                        'arSteps': 1, 'distributionHashes': ['distribution'], 'finishReason': 'length', 'stopTokenId': None}
                        for f in CONTROL_FIELDS[width]]
                    arms[kind] = {'rows': rows, 'rootHash': 'full-root',
                        'independentRootHashes': ['full-root'] * width if kind == 'independent' else [],
                        'timing': {'maximumRowsEmittingTogether': width, 'nativePrefillStepSize': 256,
                            'admissions': [{'admitted': width, 'activeBound': width, 'rowKeys': list(CONTROL_FIELDS[width])}]}}
                controls[mode][str(width)] = {'status': 'same-width-exact', 'actualLanes': width, 'arms': arms}
        check_controls(controls)
        controls['frozen']['6']['arms']['independent']['rows'][0]['fields'] = ['T1']
        with self.assertRaisesRegex(ValueError, 'actual distinct'): check_controls(controls)


if __name__ == '__main__': unittest.main()
