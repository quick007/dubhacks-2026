"""Product-bound public lexical routing CPU-only checks."""
import json
from copy import deepcopy
import unittest
from unittest.mock import patch
from . import store_qwen_lexical_choice_pilot_v1 as pilot

class Tokenizer:
    def __init__(self):
        self.routes={' '+label:2000+n for n,label in enumerate(pilot.PRODUCT_LABELS.values())};self.calls=[]
    def encode(self,text,add_special_tokens=False):
        for route,number in self.routes.items():
            if text.endswith(route):return [ord(x) for x in text[:-len(route)]]+[number]
        return [ord(x) for x in text]
    def decode(self,ids):return ''.join(next((s for s,v in self.routes.items() if v==x),chr(x)) for x in ids)
    def apply_chat_template(self,messages,**kwargs):self.calls.append(kwargs);return 'system\n'+messages[0]['content']+'\nuser\n'+messages[1]['content']+'\nassistant\n'
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.people=json.loads((pilot.HERE/'synthetic_history_v1/demo-people.json').read_text())['people'];cls.catalog=json.loads((pilot.HERE/'synthetic_history_v1/catalog-binding.json').read_text())['products']
    def test16factual_unique_whole_tokens_with_one_disclosed_material(self):
        routes=pilot.routing_tokens(Tokenizer(),self.catalog)
        self.assertEqual(len(routes),16);self.assertEqual(len({r['tokenId'] for r in routes}),16)
        self.assertEqual([r['productId'] for r in routes if r['labelKind']=='public-material-noun'],['sweatshirt'])
        self.assertEqual(next(r['code'] for r in routes if r['productId']=='sweatshirt'),'cotton')
    def test20rows_stable_product_label_token_through_order_swaps(self):
        tokenizer=Tokenizer()
        with patch.dict(pilot.CONFIG,maxTokens=30000):rows,_=pilot.build_rows(self.people,self.catalog,tokenizer)
        self.assertEqual(len(rows),20)
        for index in range(0,20,2):
            a,b=rows[index:index+2]
            self.assertEqual(a['publicRequestSha256'],b['publicRequestSha256'])
            self.assertEqual(a['choices'],list(reversed(b['choices'])))
            self.assertEqual({x['productId']:(x['code'],x['tokenId']) for x in a['choices']},{x['productId']:(x['code'],x['tokenId']) for x in b['choices']})
            self.assertTrue(a['prompt'].endswith('Choice:'));self.assertIn('cotton: Soft cotton crewneck',a['prompt'])
            self.assertNotIn('displayName',a['prompt']);self.assertNotIn('relevanceGrades',a['prompt'])
        self.assertTrue(all(x['enable_thinking'] is False for x in tokenizer.calls))
    def test_factual_multitoken_or_invented_label_refused(self):
        tokenizer=Tokenizer();del tokenizer.routes[' cotton']
        with self.assertRaisesRegex(ValueError,'not one whole token'):pilot.routing_tokens(tokenizer,self.catalog)
        catalog=deepcopy(self.catalog);catalog[0]['name']='Different';catalog[0]['category']='Something'
        with self.assertRaisesRegex(ValueError,'not an exact public'):pilot.routing_tokens(Tokenizer(),catalog)
    def test_no_truncation_for_long_inputs(self):
        with self.assertRaisesRegex(ValueError,'no truncation'):pilot.build_rows(self.people,self.catalog,Tokenizer())
if __name__=='__main__':unittest.main()
