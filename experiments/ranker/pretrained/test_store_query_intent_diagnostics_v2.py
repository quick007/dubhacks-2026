import unittest
from .store_query_intent_diagnostics_v2 import summarize
class Tests(unittest.TestCase):
    def test_matched_context_sensitivity_and_contrast_counts(self):
        rows=[];p=[]
        for label in (0,1):
            for index in range(8):
                rows.append({'caseId':f'{label}-{index}','query':f'query{label}','intent':'specific' if label else 'broad','labelSpecific':label,'contextIndex':index,
                    'matchedContext':{'itemBudget':None if index<2 else 50,'hardRequirements':[] if index%2==0 else [{'attribute':'color','value':'Navy'}]}})
                p.append((.7 if label else .1)+.01*index)
        result=summarize(rows,p)
        self.assertEqual(result['overall']['classBalancedAccuracy'],1)
        self.assertEqual(result['contrastSubsets']['broadNullBudget']['rows'],2)
        self.assertEqual(result['contrastSubsets']['specificNumericBudgetAndGuard']['rows'],3)
        self.assertAlmostEqual(result['maximumPerQueryContextDelta'],.07)
        self.assertFalse(result['selectionOrThresholdTuning'])
    def test_incomplete_or_duplicate_cases_fail(self):
        with self.assertRaises(ValueError):summarize([],[])
        row={'caseId':'x','query':'q','intent':'broad','labelSpecific':0,'contextIndex':0,'matchedContext':{}}
        with self.assertRaises(ValueError):summarize([row,row],[.1,.1])
        with self.assertRaises(ValueError):summarize([row],[.1])
if __name__=='__main__':unittest.main()
