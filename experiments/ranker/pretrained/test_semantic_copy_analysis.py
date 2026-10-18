"""CPU fixtures exercise analysis, never represent real generated output."""
import copy
import unittest

from .semantic_copy_analysis import DATA, analyze, pending, records, render
from .scheduler_identity_experiment import assemble_values_legacy_v1 as assemble_values


class SemanticCopyAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {s: records(DATA / f'cases.{s}.jsonl') for s in ('dev', 'test')}

    def arm(self, split='dev', name='frozen', case_index=0):
        case = self.cases[split][case_index]
        rows = [{'rowId': t['rowId'], 'kind': t['kind'], 'fields': [t['field']],
                 'maxTokens': t['maxTokens'], 'text': t['targetValue'],
                 'finishReason': 'stop', 'tokenIds': [100, 101], 'arSteps': 3}
                for t in case['targets']]
        result = assemble_values(rows, case['pendingPagePlan'], case['request'], case['catalog'])
        return {'split': split, 'caseId': case['caseId'], 'arm': name,
            'status': 'completed-attempt', 'rows': rows, **result,
            'totalOutputBudget': 224, 'actualOutputTokens': 28, 'actualRowArSteps': 42,
            'externalFullCompletionMs': 100., 'timing': {'fullCompletionMs': 95.},
            'memory': {'mlxPeakBytes': 1024}}

    def report(self, arms):
        return {'status': 'started', 'arms': arms, 'steps': [{'ms': 2.}],
                'wallMs': 200., 'loadMs': 20., 'memory': {'mlxPeakBytes': 1024}}

    def requality(self, arm):
        case = next(c for c in self.cases[arm['split']] if c['caseId'] == arm['caseId'])
        arm['quality'] = assemble_values(arm['rows'], pending(arm['pagePlan']),
            case['request'], case['catalog'])['quality']

    def test_fixed_denominators_no_mutation_and_valid_none(self):
        arm = self.arm(); before = copy.deepcopy(arm)
        result = analyze(self.report([arm]), self.cases)
        self.assertEqual(arm, before)
        s = result['summaries']['dev']['frozen']
        self.assertEqual((s['fixedCases'], s['completed'], s['unrecorded']), (3, 1, 2))
        self.assertEqual((s['acceptedPages'], s['acceptedTitles'], s['acceptedHeadings']), (1, 6, 2))
        self.assertEqual((s['acceptedBadgesIncludingNone'], s['acceptedGeneratedBadges'], s['acceptedNoneBadges']), (6, 3, 3))
        self.assertEqual(s['acceptedPageRateFixedCases'], 1 / 3)
        self.assertFalse(result['completeFixedEvaluation'])
        absent = result['summaries']['test']['learned']
        self.assertEqual(absent['unrecorded'], 3)
        self.assertIsNone(absent['fullCompletionMsValidOnly']['median'])
        self.assertEqual(s['fullCompletionMsValidOnly']['median'], 100.)
        self.assertIn('unavailable', render(result))

    def test_global_truncation_does_not_rescue_raw_none(self):
        arm = self.arm(); arm['rows'][0]['finishReason'] = 'length'; self.requality(arm)
        result = analyze(self.report([arm]), self.cases)
        s = result['summaries']['dev']['frozen']
        self.assertEqual(s['acceptedPages'], 0)
        self.assertEqual(s['acceptedTitles'], 0)
        self.assertEqual(s['acceptedBadgesIncludingNone'], 0)
        self.assertEqual(s['requiredNoneRawCorrect'], 3)
        self.assertEqual(s['requiredNoneStrictAccepted'], 0)
        self.assertEqual(s['strictCorrectNonePages'], 0)
        self.assertEqual(s['fullCompletionMsAll']['median'], 100.)
        self.assertIsNone(s['fullCompletionMsValidOnly']['median'])

    def test_supported_none_allowed_and_unsupported_non_none_rejected(self):
        arm = self.arm()
        card_fields = [f for f in arm['wireRows'] if f['kind'] == 'badge']
        supported = next(f['fields'][0] for f in card_fields if f['rawValue'] != 'NONE')
        required = next(f['fields'][0] for f in card_fields if f['rawValue'] == 'NONE')
        for row in arm['rows']:
            if row['fields'][0] == supported: row['text'] = 'NONE'
            if row['fields'][0] == required: row['text'] = 'Everyday travel'
        self.requality(arm)
        s = analyze(self.report([arm]), self.cases)['summaries']['dev']['frozen']
        self.assertEqual(s['acceptedBadgesIncludingNone'], 5)
        self.assertEqual(s['supportedBadgeAcceptedNone'], 1)
        self.assertEqual(s['requiredNoneRawCorrect'], 2)
        self.assertEqual(s['acceptedPages'], 0)

    def test_wrong_identity_is_replayed_and_saved_quality_drift_rejected(self):
        arm = self.arm()
        next(r for r in arm['rows'] if r['kind'] == 'title')['text'] = 'Meet your everyday travel companion'
        with self.assertRaisesRegex(ValueError, 'Saved/current quality mismatch'):
            analyze(self.report([arm]), self.cases)
        self.requality(arm)
        result = analyze(self.report([arm]), self.cases)
        title = next(f for f in result['records'][0]['fields'] if f['kind'] == 'title')
        self.assertFalse(title['accepted'])
        self.assertEqual(title['identityErrors'], ['missing_selected_product_identity'])

    def test_receipts_and_fixed_membership_fail_closed(self):
        for patch, message in (({'actualOutputTokens': 0}, 'token/AR'),
                ({'actualRowArSteps': 0}, 'token/AR'), ({'totalOutputBudget': 1}, 'output budget')):
            arm = self.arm(); arm.update(patch)
            with self.subTest(patch=patch), self.assertRaisesRegex(ValueError, message):
                analyze(self.report([arm]), self.cases)
        arm = self.arm(); arm['rows'][0]['maxTokens'] -= 1
        with self.assertRaisesRegex(ValueError, 'kind or cap'):
            analyze(self.report([arm]), self.cases)
        arm = self.arm()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            analyze(self.report([arm, arm]), self.cases)
        with self.assertRaisesRegex(ValueError, 'three unique'):
            analyze(self.report([]), {**self.cases, 'test': self.cases['test'][:2]})

    def test_paired_selection_and_recorded_failure_remain_visible(self):
        frozen = self.arm(); learned = self.arm(name='learned')
        failed = {'split': 'test', 'caseId': self.cases['test'][0]['caseId'],
                  'arm': 'learned', 'status': 'failed', 'blocker': 'fixture-only'}
        result = analyze(self.report([frozen, learned, failed]), self.cases)
        self.assertTrue(result['selectionPairs'][0]['sameProductsImagesAndGrouping'])
        self.assertEqual(result['summaries']['test']['learned']['failedRecorded'], 1)
        self.assertEqual(result['summaries']['test']['learned']['unrecorded'], 2)
        self.assertEqual(result['summaries']['test']['learned']['fullCompletionMsAll']['count'], 0)

    def test_generated_field_label_is_preserved_and_not_rescued(self):
        arm = self.arm()
        row = next(r for r in arm['rows'] if r['kind'] == 'badge' and r['text'] == 'NONE')
        row['text'] = row['fields'][0] + ': NONE'
        self.requality(arm)
        result = analyze(self.report([arm]), self.cases)
        field = next(f for f in result['records'][0]['fields'] if f['field'] == row['fields'][0])
        self.assertEqual(field['rawValue'], row['text'])
        self.assertFalse(field['accepted'])
        self.assertIn('unexpected-generated-field-label', field['protocolRejections'])
        self.assertFalse(result['records'][0]['quality']['allFieldsFulfilled'])


if __name__ == '__main__':
    unittest.main()
