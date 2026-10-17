"""CPU wire-policy gates; scheduler identity cannot rescue malformed values."""
import copy
import unittest
from .scheduler_identity_experiment import encode_value_rows,values_layout,prepare_cases,summarize


class SchedulerIdentityTests(unittest.TestCase):
    def row(self,raw,kind='badge',field='B1'):
        return {'rowId':kind+'-1','fields':[field],'kind':kind,'text':raw,'finishReason':'stop'}

    def test_exact_none_and_fresh_value_preserved(self):
        rows=[self.row('NONE'),self.row('Fresh factual wording','title','T1')]
        original=copy.deepcopy(rows);encoded,failures=encode_value_rows(rows)
        self.assertFalse(failures);self.assertEqual(rows,original)
        self.assertEqual(encoded[0]['rawValue'],'NONE');self.assertEqual(encoded[0]['text'],'B1: NONE')
        self.assertEqual(encoded[1]['rawValue'],'Fresh factual wording')
        self.assertEqual(encoded[1]['text'],'T1: Fresh factual wording')

    def test_empty_linebreak_and_labels_never_repaired(self):
        for raw in ('',' ','NONE\n','NONE\r','first\nsecond','a\u2028b','B1: NONE','B2: compact','value T2: item','b1: NONE','Ｂ１： NONE'):
            with self.subTest(raw=raw):
                encoded,failures=encode_value_rows([self.row(raw)])
                self.assertTrue(failures);self.assertEqual(encoded[0]['rawValue'],raw)
                self.assertEqual(encoded[0]['text'],'B1: '+raw)
        with self.assertRaises(ValueError):encode_value_rows([self.row('NONE'),self.row('NONE')])

    def test_same_evidence_fields_caps_for_values_layout(self):
        from .test_heading_experiment import HeadingExperimentTests
        from .shared_evidence_experiment import shared_layout
        plan,request,catalog=HeadingExperimentTests().fixture(6)
        old_common,old_rows,old_evidence=shared_layout(plan,request,catalog)
        common,rows,evidence=values_layout(plan,request,catalog)
        self.assertEqual(evidence,old_evidence);self.assertEqual(len(rows),14)
        self.assertEqual(sum(r['maxTokens'] for r in rows),224)
        self.assertEqual([{k:v for k,v in r.items() if k!='messages'} for r in rows],
                         [{k:v for k,v in r.items() if k!='messages'} for r in old_rows])
        self.assertIn('identity is supplied by the scheduler',common['content'])
        self.assertNotIn('its labeled plain line',common['content'])
        self.assertTrue(all('Do not output a label' in r['messages'][0]['content'] for r in rows))

    def test_fresh_frozen_roster_and_failure_denominators(self):
        packet=prepare_cases();heldout=packet['freshSyntheticHoldouts']
        self.assertEqual(len(heldout),2)
        for case in heldout:
            self.assertEqual(case['cards'],6);self.assertEqual(len(case['catalog']),6)
            self.assertTrue(case['heldOutFromProtocolAdjustment']);self.assertFalse(case['rankingQualityLabelsAvailable'])
        records=[{'freshHoldout':True,'arms':{n:{'status':'failed','quality':{'allFieldsFulfilled':True}}
            for n in ('generatedLabels','schedulerIdentity')}}]
        self.assertEqual(summarize(records,True)['schedulerIdentity'],{'fixedCases':2,'completed':0,'validPages':0,
            'validRateFixedCases':0,'failed':1,'unattempted':1,'fullCompletionMs':[],'validFullCompletionMs':[]})


if __name__=='__main__':unittest.main()
