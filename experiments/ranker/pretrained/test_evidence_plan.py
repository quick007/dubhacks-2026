import copy
import unittest
import torch
from .evidence_plan import ranking_loss, update, validate_generated, prompt_for, CONFIG
from .backbone import ParallelHeads
from .test_page_plan import PagePlanTests


class EvidencePlanTests(PagePlanTests):
    def test_generated_title_heading_and_correct_none(self):
        plan=self.plan()
        text='S1: Everyday travel choices\nT1: Your compact travel companion\nB1: NONE'
        completed,quality=validate_generated(text,'stop',plan,self.request,self.catalog)
        self.assertTrue(quality['allFieldsFulfilled'])
        self.assertTrue(quality['correctNone'])
        self.assertEqual(completed['sections'][0]['products'][0]['title']['source'],'generated')
        _,bad=validate_generated(text.replace('NONE','Compact travel'),'stop',plan,self.request,self.catalog)
        self.assertFalse(bad['allFieldsFulfilled'])
        _,truncated=validate_generated(text,'length',plan,self.request,self.catalog)
        self.assertEqual(truncated['newTitleCount'],0)
    def test_grounded_plan_does_not_include_private_or_opaque_ids(self):
        self.request['context']['statedPreferences']=['compact']
        self.request['hiddenPreferences']='PRIVATE_SENTINEL'
        plan=self.plan();prompt=str(prompt_for(plan,self.request,self.catalog,True))
        self.assertNotIn('PRIVATE_SENTINEL',prompt)
        self.assertNotIn('fact:p:',prompt)
        self.assertIn('compact',prompt)
        self.assertNotEqual(prompt,str(prompt_for(plan,self.request,self.catalog,False)))
    def test_ranking_loss_real_update_and_action_alignment(self):
        request=copy.deepcopy(self.request);request['candidateProductIds']=['p','q']
        request['presentationCompatibility']={'p':{'v':True},'q':{'w':True}}
        actions=[{'productId':'p','variantId':'v'},{'productId':'q','variantId':'w'}]
        labels={'r':{'bestProductId':'p','presentationScores':{'p':{'v':3},'q':{'w':1}}}}
        cache={'r':(torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.]]),actions)}
        torch.set_num_threads(1);head=ParallelHeads(4)
        result=update(head,lambda:ranking_loss(head,{'r':request},labels,{},cache))
        self.assertGreater(result['firstGradientL2'],0)
        self.assertLess(result['lossAfter'],result['lossBefore'])
        self.assertGreater(head(cache['r'][0])[0,0],head(cache['r'][0])[1,0])

if __name__=='__main__':unittest.main()
