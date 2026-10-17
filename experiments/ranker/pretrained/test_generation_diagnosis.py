"""Protocol regression tests; no model or GPU used."""
import copy
import unittest
from .generation_diagnosis import manifest_prompt,expected_fields,field_audit
from .evidence_plan import validate_generated


class ManifestTests(unittest.TestCase):
    def setUp(self):
        from .test_page_plan import PagePlanTests
        fixture=PagePlanTests();fixture.setUp()
        self.request,self.catalog,self.plan=fixture.request,fixture.catalog,fixture.plan()
    def test_one_card_exact_labels_without_control_leakage(self):
        self.plan['sections'][0]['products'][0]['badgeDecision']='GENERATE'
        text=manifest_prompt(self.plan,self.request,self.catalog,False)[0]['content']
        self.assertEqual(expected_fields(self.plan),['S1','T1','B1'])
        self.assertIn('S1, T1, B1',text);self.assertIn('After B1 stop',text)
        self.assertNotIn('S2',text);self.assertNotIn('T2',text);self.assertNotIn('GENERATE',text)
        self.assertIn('Compact pack',text)
    def test_six_card_exact_bindings_and_shared_heading_constraints(self):
        card=self.plan['sections'][0]['products'][0]
        self.plan['sections']=[{'products':[copy.deepcopy(card) for _ in range(3)]}, {'products':[copy.deepcopy(card) for _ in range(3)]}]
        text=manifest_prompt(self.plan,self.request,self.catalog,False)[0]['content']
        self.assertEqual(expected_fields(self.plan),['S1','S2','T1','B1','T2','B2','T3','B3','T4','B4','T5','B5','T6','B6'])
        self.assertIn('After B6 stop',text);self.assertNotIn('T7',text)
        self.assertIn('S2: heading for cards [4, 5, 6]',text)
        self.assertIn('T6/B6 bind ONLY to Compact pack',text)
    def test_diagnosis_distinguishes_novelty_evidence_and_factual_claim(self):
        self.request['context']['statedPreferences']=['compact']
        card=self.plan['sections'][0]['products'][0]
        card['approvedFactIds']=['fact:p:tag:compact']
        raw='S1: Everyday travel choices\nT1: Compact pack\nB1: GENERATE'
        audit=field_audit(raw,'stop',self.plan,self.request,self.catalog)
        self.assertIn('control_enum_copied_as_content',audit['categories'])
        self.assertIn('copied_approved_title_not_new',audit['categories'])
        raw='S1: Everyday travel choices\nT1: Waterproof compact travel companion\nB1: NONE'
        audit=field_audit(raw,'stop',self.plan,self.request,self.catalog)
        self.assertIn('forbidden_claim_terms',audit['categories'])
    def test_none_control_is_not_misclassified_as_lexical_content(self):
        raw='S1: Everyday travel choices\nT1: Your compact travel companion\nB1: NONE'
        audit=field_audit(raw,'stop',self.plan,self.request,self.catalog)
        self.assertTrue(audit['exactFieldOrder'])
        self.assertNotIn('conservative_lexical_rejection_needs_review',audit['categories'])
        reordered='T1: Your compact travel companion\nB1: NONE\nS1: Everyday travel choices'
        self.assertIn('wrong_field_order',field_audit(reordered,'stop',self.plan,self.request,self.catalog)['categories'])

    def test_factual_guard_unchanged_new_wording_retained(self):
        raw='S1: Everyday travel choices\nT1: Your compact travel companion\nB1: NONE'
        result,quality=validate_generated(raw,'stop',self.plan,self.request,self.catalog)
        self.assertTrue(quality['allFieldsFulfilled'])
        self.assertEqual(result['sections'][0]['products'][0]['title']['text'],'Your compact travel companion')
        _,bad=validate_generated(raw.replace('compact','waterproof'),'stop',self.plan,self.request,self.catalog)
        self.assertFalse(bad['allFieldsFulfilled'])
        _,short=validate_generated(raw,'length',self.plan,self.request,self.catalog)
        self.assertEqual(short['newTitleCount'],0)

if __name__=='__main__':unittest.main()
