"""CPU preflight and rejection-policy checks; no model import/load."""
import unittest
from .q001 import preflight,badge_score,summarize


class Q001Tests(unittest.TestCase):
    def setUp(self):
        self.records=preflight()['records']
        self.supported=self.records[0]
        self.none=next(r for r in self.records if r['caseId']=='evidence-control-no-history-no-preferences')

    def test_supported_and_none_paths(self):
        self.assertTrue(badge_score('B1: Water-resistant','stop',self.supported)['acceptedGeneratedBadge'])
        self.assertTrue(badge_score('B1: NONE','stop',self.supported)['acceptedNone'])
        self.assertTrue(badge_score('B1: NONE','stop',self.none)['acceptedNone'])
        self.assertFalse(badge_score('B1: Compact','stop',self.none)['acceptedBadgeRow'])

    def test_truncation_binding_duplicate_and_claims_are_rejected(self):
        for text,finish in [('B1: Water-resistant','length'),('B2: Water-resistant','stop'),
            ('B1: Water-resistant\nB1: Water-resistant','stop'),('B1: B1: Water-resistant','stop'),
            ('B1: Waterproof','stop'),('B1: <b>Water-resistant</b>','stop'),('B1: 999L Water-resistant','stop')]:
            with self.subTest(text=text,finish=finish):
                self.assertFalse(badge_score(text,finish,self.supported)['acceptedBadgeRow'])

    def test_failed_arm_with_partial_quality_cannot_count_as_accepted(self):
        result=summarize([{'variant':'baseline','status':'failed','quality':{'acceptedBadgeRow':True}}])
        self.assertEqual(result['baseline'],{'fixedDenominator':3,'completed':0,'acceptedBadgeRows':0})


if __name__=='__main__':unittest.main()
