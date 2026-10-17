import copy
import unittest
from .full_page_reliability import evidence_capsule,locked_prompt
from .evidence_plan import validate_generated


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        from .test_page_plan import PagePlanTests
        fixture=PagePlanTests();fixture.setUp();self.request=fixture.request;self.catalog=fixture.catalog;self.plan=fixture.plan()
    def test_empty_intersection_is_not_union(self):
        self.catalog['q']={**copy.deepcopy(self.catalog['p']),'id':'q','title':'Durable pack','tags':['durable']}
        card=copy.deepcopy(self.plan['sections'][0]['products'][0]);card['productId']='q'
        self.plan['sections'][0]['products'].append(card)
        capsule=evidence_capsule(self.plan,self.catalog)
        self.assertEqual(capsule['sections'][0]['universalTags'],[])
        self.assertEqual(capsule['sections'][0]['members'],[{'index':1,'tags':['compact']},{'index':2,'tags':['durable']}])
        text=locked_prompt(self.plan,self.request,self.catalog,False)[0]['content']
        self.assertIn('UNIVERSAL=EMPTY',text);self.assertIn('ZERO product attributes',text)
    def test_locked_identity_support_and_no_control_enum(self):
        self.plan['sections'][0]['products'][0]['approvedFactIds']=['fact:p:tag:compact']
        text=locked_prompt(self.plan,self.request,self.catalog,False)[0]['content']
        self.assertIn('identity=Compact pack',text);self.assertIn('title MUST mention=compact',text)
        self.assertIn('badge allowed=compact',text);self.assertNotIn('GENERATE',text)
        self.assertIn('After B1 stop',text);self.assertNotIn('S2',text)
    def test_unknown_preference_remains_unknown(self):
        capsule=evidence_capsule(self.plan,self.catalog)
        self.assertEqual(capsule['cards'][0]['requiredObservedTags'],[])
        text=locked_prompt(self.plan,self.request,self.catalog,False)[0]['content']
        self.assertIn('badge allowed=NONE ONLY',text)
        self.assertIn('title=identity only, no personal preference claim',text)
    def test_original_factual_and_novelty_guards_unchanged(self):
        raw='S1: Everyday travel choices\nT1: Your compact travel companion\nB1: NONE'
        result,quality=validate_generated(raw,'stop',self.plan,self.request,self.catalog)
        self.assertTrue(quality['allFieldsFulfilled'])
        self.assertEqual(result['sections'][0]['products'][0]['title']['source'],'generated')
        for changed in [raw.replace('Everyday travel choices','Waterproof travel choices'),raw.replace('Your compact travel companion','Compact pack')]:
            self.assertFalse(validate_generated(changed,'stop',self.plan,self.request,self.catalog)[1]['allFieldsFulfilled'])

if __name__=='__main__':unittest.main()
