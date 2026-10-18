"""CPU synthetic fixtures for policy accounting; no native output fabrication."""
import copy
import unittest

from .branch_local_analysis import analyze, render, strict_fields
from .semantic_copy_analysis import DATA, records
from .scheduler_identity_experiment import assemble_values_legacy_v1 as assemble_values, encode_value_rows


class BranchLocalAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = [dict(c, split=s) for s in ('dev', 'test') for c in records(DATA / f'cases.{s}.jsonl')]
        cls.packet = {'casesAlreadyInspected': True, 'cases': [dict(c, plan=c['pendingPagePlan']) for c in cls.cases]}

    def arm(self):
        case = self.cases[0]
        rows = [{'rowId': t['rowId'], 'kind': t['kind'], 'fields': [t['field']],
                 'maxTokens': t['maxTokens'], 'text': t['targetValue'],
                 'finishReason': 'stop', 'tokenIds': [100, 101], 'arSteps': 3} for t in case['targets']]
        result = assemble_values(rows, case['pendingPagePlan'], case['request'], case['catalog'])
        return {'split': case['split'], 'caseId': case['caseId'], 'arm': 'frozen',
            'status': 'completed-copy-attempt', 'rows': rows, **result,
            'endToEndRequestTiming': False, 'totalOutputBudget': 224,
            'actualOutputTokens': 28, 'actualRowArSteps': 42,
            'externalCopyStageMs': 100., 'timing': {}, 'memory': {'mlxPeakBytes': 1024}, 'inputInventory': {}}

    def refresh(self, arm):
        case = self.cases[0]
        result = assemble_values(arm['rows'], case['pendingPagePlan'], case['request'], case['catalog'])
        arm.update(result)

    def report(self, arm):
        return {'status': 'completed', 'endToEndRequestTiming': False,
                'wallMs': 100., 'loadMs': 1., 'arms': [arm]}

    def test_clean_values_equal_counts_and_fixed_missing_denominators(self):
        arm = self.arm(); before = copy.deepcopy(arm)
        result = analyze(self.report(arm), self.packet)
        self.assertEqual(before, arm)
        s = result['summaries']['dev']['frozen']
        self.assertEqual(s['originalFieldAccounting'], s['strictWireAndSemantic'])
        self.assertEqual((s['originalAcceptedPages'], s['strictAcceptedPages']), (1, 1))
        self.assertEqual((s['fixedInspectedCases'], s['completed'], s['missing']), (3, 1, 2))
        self.assertEqual(s['strictWireAndSemantic']['badge'], 6)
        self.assertIn('Copy-stage times exclude ranking', render(result))

    def test_valid_first_line_does_not_make_malformed_row_strictly_accepted(self):
        for kind in ('heading', 'title', 'badge'):
            with self.subTest(kind=kind):
                arm = self.arm()
                row = next(r for r in arm['rows'] if r['kind'] == kind and r['text'] != 'NONE')
                row['text'] += '\nEveryday travel'
                # Explicit legacy projection fixture: clean first-line generated
                # plan retained despite wire error. Do not require a future
                # corrected codec to reproduce this old accounting defect.
                wire, failures = encode_value_rows(arm['rows'])
                quality = copy.deepcopy(arm['quality'])
                quality['allFieldsFulfilled'] = False
                quality['rejections'] += failures
                replay = {'pagePlan': arm['pagePlan'], 'quality': quality, 'wireRows': wire}
                fields = strict_fields(arm['rows'], replay, self.cases[0])
                field = next(f for f in fields if f['field'] == row['fields'][0])
                self.assertTrue(field['acceptedByOriginalFieldAccounting'])
                self.assertFalse(field['strictWireAndSemanticAccepted'])
                self.assertIn('multiline-or-line-separator', field['protocolRejections'])
                self.assertEqual(field['rawValue'], row['text'])
                self.assertFalse(quality['allFieldsFulfilled'])

    def test_global_length_does_not_rescue_other_clean_fields(self):
        arm = self.arm(); arm['rows'][0]['finishReason'] = 'length'; self.refresh(arm)
        result = analyze(self.report(arm), self.packet)
        s = result['summaries']['dev']['frozen']
        self.assertEqual(s['strictWireAndSemantic']['badge'], 0)
        self.assertEqual(s['requiredNoneRawCorrect'], 3)
        self.assertIsNone(s['copyStageMsValidOnlyStrict']['median'])

    def test_embedded_label_and_required_none_are_not_repaired(self):
        arm = self.arm()
        row = next(r for r in arm['rows'] if r['kind'] == 'badge' and r['text'] == 'NONE')
        row['text'] = row['fields'][0] + ': NONE'
        self.refresh(arm)
        result = analyze(self.report(arm), self.packet)
        field = next(f for f in result['records'][0]['fields'] if f['field'] == row['fields'][0])
        self.assertEqual(field['rawValue'], row['text'])
        self.assertFalse(field['strictWireAndSemanticAccepted'])
        self.assertIn('raw-NONE-required', field['strictRawRejections'])

    def test_original_quality_drift_and_bad_timing_scope_rejected(self):
        arm = self.arm(); arm['quality']['generatedBadgeCount'] += 1
        with self.assertRaisesRegex(ValueError, 'quality mismatch'):
            analyze(self.report(arm), self.packet)
        arm = self.arm(); arm['endToEndRequestTiming'] = True
        with self.assertRaisesRegex(ValueError, 'Copy stage'):
            analyze(self.report(arm), self.packet)


if __name__ == '__main__': unittest.main()
