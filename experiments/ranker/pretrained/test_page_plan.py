import copy
import unittest
from .page_plan import make_page_plan, validate_page_plan, history_reasons


class PagePlanTests(unittest.TestCase):
    def setUp(self):
        self.product={'id':'p','title':'Compact pack','tags':['compact'],'inStock':True,'price':10,'shippingCost':0,
                      'presentationVariants':[{'id':'v','title':'Pack','imageDescription':'Pack'}]}
        self.catalog={'p':self.product}
        self.request={'recordId':'r','asOf':'2026-10-01T00:00:00Z','candidateProductIds':['p'],'history':[],
                      'context':{'statedPreferences':[],'statedRequirements':[],'statedBudget':20},
                      'presentationCompatibility':{'p':{'v':True}}}
        self.decision={'rankedProducts':[{'productId':'p','score':1}], 'presentationAction':{'productId':'p','variantId':'v'}}
    def plan(self):return make_page_plan(self.request,self.catalog,self.decision,1)
    def test_version_no_history_none_and_shortfall(self):
        plan=self.plan();self.assertTrue(validate_page_plan(plan,self.request,self.catalog))
        self.assertEqual(plan['sections'][0]['products'][0]['badgeDecision'],'NONE')
        with self.assertRaises(ValueError):validate_page_plan(plan,self.request,self.catalog,True)
        plan['schemaVersion']='page-plan.v2'
        with self.assertRaises(ValueError):validate_page_plan(plan,self.request,self.catalog)
    def test_reject_duplicate_order_invented_fact_or_unavailable(self):
        original=self.plan()
        for change in ('order','fact','history','stock','image'):
            plan=copy.deepcopy(original);card=plan['sections'][0]['products'][0]
            if change=='order':plan['sections'][0]['productOrder']*=2
            if change=='fact':card['approvedFactIds']=['invented']
            if change=='history':card['observedHistoryReasonIds']=['private-persona']
            if change=='image':card['imageCaptionId']='unknown'
            if change=='stock':self.product['inStock']=False
            with self.assertRaises(ValueError):validate_page_plan(plan,self.request,self.catalog)
            self.product['inStock']=True
    def test_observed_save_only_and_future_private_invariance(self):
        self.request['history']=[{'at':'2026-09-01T00:00:00Z','productId':'p','type':'save','confidence':.9}]
        plan=self.plan();self.assertEqual(len(plan['sections'][0]['products'][0]['observedHistoryReasonIds']),1)
        baseline=history_reasons(self.request,self.catalog,self.product)
        self.request['hiddenPreferences']=['other']
        self.request['history'] += [{'at':'2099-01-01T00:00:00Z','productId':'p','type':'save','confidence':1}]*5
        self.assertEqual(baseline,history_reasons(self.request,self.catalog,self.product))
        self.request['history'][0]['type']='view'
        self.assertEqual({},history_reasons(self.request,self.catalog,self.product))
    def test_factual_generated_fields_and_duplicates(self):
        plan=self.plan();section=plan['sections'][0]
        section['heading']={'text':'Everyday travel choices','source':'generated'}
        section['products'][0]['title']={'text':'Compact travel pack','source':'generated'}
        self.assertTrue(validate_page_plan(plan,self.request,self.catalog,True))
        section['products'][0]['title']['text']='Your compact travel companion'
        with self.assertRaisesRegex(ValueError,'identity'):
            validate_page_plan(plan,self.request,self.catalog,True)
        section['products'][0]['title']['text']='Free shipping tomorrow'
        with self.assertRaises(ValueError):validate_page_plan(plan,self.request,self.catalog,True)

if __name__=='__main__':unittest.main()
