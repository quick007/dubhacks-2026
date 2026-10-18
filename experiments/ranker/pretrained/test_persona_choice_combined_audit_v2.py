"""Derived metric-view identity regression; original predictions stay immutable."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from . import audit_persona_choice_combined_v1 as original
from . import audit_persona_choice_combined_v2 as current
from . import test_persona_choice_combined_audit_v1 as fixtures


class DerivedIdentityTests(unittest.TestCase):
    def test_reverse_metric_view_normalizes_case_without_mutating_original(self):
        products,profiles,labels,canonical=fixtures.quality_fixture()
        reverse=[]
        for row in canonical:
            reverse.append({**deepcopy(row),'orderLabel':'reversed','caseId':row['profileId']+'-reversed'})
        before=deepcopy(reverse)
        with self.assertRaisesRegex(original.AuditError,'case identity'):
            original.metrics(original.choose(reverse,profiles,'reversed'),profiles,labels,products,32)
        metric_view=current.choose(reverse,profiles,'reversed')
        self.assertEqual(current.metrics(metric_view,profiles,labels,products,32)['primaryTop1'],32)
        self.assertEqual(reverse,before)
        self.assertTrue(all(row['caseId']==row['profileId']+'-canonical' for row in metric_view))

    def test_control_view_joins_original_gold_and_normalizes_derived_identity_only(self):
        products,profiles,labels,canonical=fixtures.quality_fixture(); controls=[]
        for row in canonical:
            controls.append({**deepcopy(row),'profileId':'shuffled-'+row['profileId'],
                             'sourceProfileId':row['profileId'],'controlName':'shuffled-profile',
                             'caseId':'shuffled-'+row['profileId']+'-canonical'})
        before=deepcopy(controls)
        with self.assertRaisesRegex(original.AuditError,'case identity'):
            original.metrics(original.choose(controls,profiles,control='shuffled-public-profile'),profiles,labels,products,32)
        metric_view=current.choose(controls,profiles,control='shuffled-public-profile')
        result=current.metrics(metric_view,profiles,labels,products,32)
        self.assertEqual((result['fixedDenominator'],result['groups'],result['primaryTop1']),(32,16,32))
        self.assertEqual(controls,before)
        for source,derived in zip(controls,metric_view,strict=True):
            self.assertEqual(source['scores16'],derived['scores16'])
            self.assertEqual(source['order'],derived['order'])
            self.assertEqual(derived['profileId'],source['sourceProfileId'])

    def test_original_saved_identity_still_rejects_before_metric_normalization(self):
        _,profiles,_,rows=fixtures.quality_fixture()
        rows[0]['caseId']='wrong-canonical'
        with self.assertRaisesRegex(current.AuditError,'case identity'):
            current.index_rows(rows,profiles,both_orders=False)

    def test_all16_original_fake_admission_math_tests_pass_for_v2(self):
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(fixtures.CombinedAuditTests)
        result=unittest.TestResult()
        with patch.object(fixtures,'audit',current):suite.run(result)
        self.assertEqual(result.testsRun,16)
        self.assertFalse(result.errors,result.errors);self.assertFalse(result.failures,result.failures)


if __name__=='__main__':unittest.main()
