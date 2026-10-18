"""CPU-only formatter contract; no private gold or native model execution."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch
from . import store_qwen_lexical_choice_inputs_v2 as inputs
from .test_store_qwen_lexical_choice_pilot_v1 import Tokenizer

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog=inputs.load_public_catalog();cls.people=json.loads((inputs.pilot.HERE/'synthetic_history_v1/demo-people.json').read_text())['people']
    def test_all20_original_prompts_and_tokenids_exact_byte_parity(self):
        tokenizer=Tokenizer()
        with patch.dict(inputs.pilot.CONFIG,maxTokens=30000):frozen,_=inputs.pilot.build_rows(self.people,self.catalog,tokenizer)
        with patch.object(inputs,'MAX_TOKENS',30000):
            for person in self.people:
                for order in ('canonical','reversed'):
                    row=inputs.build_row(person['request'],self.catalog,tokenizer,person['id']+'-'+order,order)
                    prior=next(x for x in frozen if x['caseId']==row['caseId'])
                    for key in prior:
                        if key!='personId':self.assertEqual(row[key],prior[key],key)
                    self.assertEqual(row['tokenIds'][:row['catalogPrefixTokens']],row['catalogPrefixTokenIds'])
                    self.assertTrue(row['catalogPrefixText'].endswith('Public shopper observations:\n'))
                    self.assertNotIn('Query:',row['catalogPrefixText'])
    def test_zero_to_three_history_and_anonymous_requests_are_explicit(self):
        for count in range(4):
            request=deepcopy(self.people[0]['request']);request['publicHistory']=request['publicHistory'][:count]
            self.assertEqual(len(inputs.checked_request(request,self.catalog)['recentHistory']),count)
        value=inputs.checked_request({'query':'Browse the store'},self.catalog)
        self.assertEqual(value['recentHistory'],[]);self.assertEqual(value['background'],[]);self.assertEqual(value['hardRequirements'],[])
    def test_unknown_private_fields_and_stale_facts_fail(self):
        request=deepcopy(self.people[0]['request']);request['desiredLabel']='throw'
        with self.assertRaises(ValueError):inputs.checked_request(request,self.catalog)
        request=deepcopy(self.people[0]['request']);request['persona']['hiddenGoal']='throw'
        with self.assertRaises(ValueError):inputs.checked_request(request,self.catalog)
        request=deepcopy(self.people[0]['request']);request['publicHistory'][1]['selectedColor']='fake purple'
        with self.assertRaises(ValueError):inputs.checked_request(request,self.catalog)
        catalog=deepcopy(self.catalog);catalog[0]['price']+=1
        with self.assertRaises(ValueError):inputs.checked_catalog(catalog)
    def test_hardconstraints_and_budget_are_preserved_never_interpreted_away(self):
        request={'query':'A blue item','itemBudget':50,'hardRequirements':[{'attribute':'color','value':'Blue'}]}
        checked=inputs.checked_request(request,self.catalog)
        self.assertEqual(checked['hardRequirements'],request['hardRequirements']);self.assertEqual(checked['itemBudget'],50)
        with self.assertRaises(ValueError):inputs.checked_request({**request,'itemBudget':True},self.catalog)
    def test_history_overflow_times_duplicates_source_and_caseid_are_guarded(self):
        request=deepcopy(self.people[0]['request']);request['publicHistory']*=2
        with self.assertRaises(ValueError):inputs.checked_request(request,self.catalog)
        request=deepcopy(self.people[0]['request']);request['publicHistory'][1]['eventId']=request['publicHistory'][0]['eventId']
        with self.assertRaises(ValueError):inputs.checked_request(request,self.catalog)
        request=deepcopy(self.people[0]['request']);request['publicHistory'][0]['source']='invented'
        with self.assertRaises(ValueError):inputs.checked_request(request,self.catalog)
        with patch.object(inputs,'MAX_TOKENS',30000):
            row=inputs.build_row({'query':'Browse the store'},self.catalog,Tokenizer(),'opaque-session-only','canonical')
            self.assertNotIn('opaque-session-only',row['prompt'])
    def test_two_exact_prefixes_independent_of_profile_and_query(self):
        prefixes={}
        with patch.object(inputs,'MAX_TOKENS',30000):
            for person in self.people:
                for order in ('canonical','reversed'):
                    row=inputs.build_row(person['request'],self.catalog,Tokenizer(),'x',order)
                    signature=(tuple(row['catalogPrefixTokenIds']),row['catalogPrefixTextSha256'])
                    if order in prefixes:self.assertEqual(prefixes[order],signature)
                    prefixes[order]=signature
        self.assertNotEqual(prefixes['canonical'],prefixes['reversed'])
    def test_no_silent_truncation(self):
        with self.assertRaisesRegex(ValueError,'no truncation'):inputs.build_row(self.people[0]['request'],self.catalog,Tokenizer(),'x','canonical')
if __name__=='__main__':unittest.main()
