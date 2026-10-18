"""CPU protocol checks; no model/MLX load or private annotations."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from . import store_qwen_choice_pilot_v1 as pilot

class Tokenizer:
    def __init__(self): self.calls=[]
    def encode(self,text,add_special_tokens=False):
        if len(text)>=2 and text[-2]==' ' and 'A'<=text[-1]<='P':return [ord(x) for x in text[:-2]]+[1000+ord(text[-1])-65]
        return [ord(x) for x in text]
    def decode(self,ids):return ''.join(' '+chr(x-1000+65) if x>=1000 else chr(x) for x in ids)
    def apply_chat_template(self,messages,**kwargs):
        self.calls.append(kwargs);return 'user\n'+messages[0]['content']+messages[1]['content']+'\nassistant\n'

class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.people=json.loads((pilot.HERE/'synthetic_history_v1/demo-people.json').read_text())['people']
        cls.catalog=json.loads((pilot.HERE/'synthetic_history_v1/catalog-binding.json').read_text())['products']
    def test16_unique_one_token_codes_and_decode(self):
        t=Tokenizer();routes=pilot.routing_tokens(t)
        self.assertEqual(len({x['tokenId'] for x in routes}),16)
        self.assertTrue(all(t.encode(x['continuation'])==[x['tokenId']] for x in routes))
    def test20_fixed_rows_and_order_remapping_no_gold(self):
        t=Tokenizer()
        # Fake tokenizer usescharacters, so preserve the real cap only for actual
        # native tokenizer preflight; this fixture tests routing/message control.
        with patch.dict(pilot.CONFIG,maxTokens=30000):rows,routes=pilot.build_rows(self.people,self.catalog,t)
        self.assertEqual(len(rows),20)
        self.assertEqual({r['order'] for r in rows},{'canonical','reversed'})
        self.assertEqual(rows[0]['choices'][0]['productId'],rows[1]['choices'][-1]['productId'])
        self.assertTrue(all(c['enable_thinking'] is False for c in t.calls))
        self.assertTrue(all(r['prompt'].endswith('Choice:') for r in rows))
        self.assertTrue(all('displayName' not in r['prompt'] and 'relevanceGrades' not in r['prompt'] for r in rows))
        self.assertEqual([r['tokenId'] for r in rows[0]['choices']],[r['tokenId'] for r in rows[1]['choices']])
    def test_cap_refuses_without_truncation(self):
        with self.assertRaisesRegex(ValueError,'no truncation'):pilot.build_rows(self.people,self.catalog,Tokenizer())
    def test_private_and_stale_fact_injection_rejected(self):
        from copy import deepcopy
        person=deepcopy(self.people[0]);person['request']['desiredProductId']='blanket'
        with self.assertRaises(ValueError):pilot.public_profile(person,self.catalog)
        person=deepcopy(self.people[0]);person['request']['publicHistory'][1]['selectedColor']='Invented purple'
        with self.assertRaises(ValueError):pilot.public_profile(person,self.catalog)
    def test_raw_hidden_last_projection_matches_installed_tied_or_untied_path(self):
        raw=np.array([[[3.,4.]]]);seen=[]
        lm=SimpleNamespace(args=SimpleNamespace(tie_word_embeddings=False),lm_head=lambda x:seen.append(x.copy()) or x*2)
        self.assertTrue(np.array_equal(pilot.project_last(lm,raw),raw*2));self.assertTrue(np.array_equal(seen[0],raw))
        lm.args.tie_word_embeddings=True;lm.model=SimpleNamespace(embed_tokens=SimpleNamespace(as_linear=lambda x:x*3))
        self.assertTrue(np.array_equal(pilot.project_last(lm,raw),raw*3))
    def test_one_full_fresh_body_forward_scores_all16_without_ar_or_normalization(self):
        class MX:
            float32=np.float32
            array=staticmethod(np.array);eval=staticmethod(lambda *x:None);synchronize=staticmethod(lambda:None)
            logsumexp=staticmethod(lambda x:np.logaddexp.reduce(x));argmax=staticmethod(np.argmax)
            all=staticmethod(np.all);isfinite=staticmethod(np.isfinite);get_peak_memory=staticmethod(lambda:1024)
        calls=[]
        class Language:
            args=SimpleNamespace(tie_word_embeddings=False)
            def lm_head(self,x):
                calls.append(('lm',x.copy()));return np.arange(1200,dtype=np.float32)[None,None,:]
        class Model:
            language_model=Language()
            def model(self,inputs,cache):calls.append(('body',inputs.copy(),cache));return np.array([[[3.,4.],[5.,12.]]])
        bb=SimpleNamespace(mx=MX(),model=Model(),tokenizer=Tokenizer(),check_memory=lambda:None,memory_budget=4096,runtime_memory=lambda:{})
        row={'prompt':'hi','tokenIds':[104,105],'caseId':'fake','personId':'p','order':'canonical','choices':[{'code':chr(65+i),'productId':str(i),'tokenId':1000+i} for i in range(16)],'publicPurchaseProductIds':['0']}
        receipt=pilot.score_row(bb,row)
        self.assertEqual(len(calls),2);self.assertEqual(calls[0][0],'body');self.assertIsNone(calls[0][2]);self.assertEqual(calls[1][0],'lm')
        self.assertTrue(np.array_equal(calls[1][1],np.array([[[5.,12.]]])))
        self.assertEqual(receipt['selectedProductId'],'15');self.assertEqual(len(receipt['scores']),16)
        self.assertEqual(receipt['autoregressiveSteps'],0);self.assertEqual(receipt['bodyForwardCalls'],1)
        self.assertAlmostEqual(sum(x['routingConditionalProbability'] for x in receipt['scores']),1)
    def test_no_execute_flag_never_imports_native_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'plan.json';value={'rows':[{}]*20,'config':pilot.CONFIG,'tokenRange':[1,2]}
            pilot.dump(p,value);pilot.dump(p.parent/'PLAN_FROZEN.json',{'planSha256':pilot.sha(p)})
            with patch.object(pilot,'model_text_and_stat_pins'):
                self.assertFalse(pilot.run(p,Path(directory)/'out')['nativeExecuted'])
                with self.assertRaisesRegex(ValueError,'native-owner root'):pilot.run(p,Path(directory)/'out',execute=True)
if __name__=='__main__':unittest.main()
