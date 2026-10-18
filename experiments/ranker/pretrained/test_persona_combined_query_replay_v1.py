"""Explicit synthetic CPU tensors/receipts; no training or production gold."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from . import audit_persona_combined_query_replay_v1 as audit
from . import audit_persona_choice_combined_v1 as evidence


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.old_threads)

    def test_query_head_independent_sigmoid_and_strict_original4096(self):
        query=torch.eye(1,4096)
        state={'linear.weight':torch.full((1,4096),.02),'linear.bias':torch.tensor([-.1])}
        expected=float(torch.tensor(-.08).sigmoid())
        self.assertAlmostEqual(audit.query_probability(state,query),expected,places=7)
        with self.assertRaises(evidence.AuditError):audit.query_probability(state,torch.ones(1,4096))
        with self.assertRaises(evidence.AuditError):audit.query_probability(state,query.double())
        state['target']=torch.ones(1)
        with self.assertRaises(evidence.AuditError):audit.query_probability(state,query)

    def test_independent_v5_matches_original_frozen_head_math(self):
        from .store_function_adaptation import FunctionHead
        catalog=torch.eye(16,4096);query=torch.eye(1,4096)
        head=FunctionHead({'catalog':catalog.double().mean(0).float(),'query':torch.zeros(4096)})
        with torch.inference_mode():
            expected=(head.probabilities('catalog',catalog)@head.probabilities('query',query)[0]).tolist()
        self.assertEqual(audit.v5_scores(head.state_dict(),query,catalog),expected)
        changed={k:v.detach().clone() for k,v in head.state_dict().items()};changed['catalog_mean'][0]+=.01
        with self.assertRaisesRegex(evidence.AuditError,'catalog mean'):audit.v5_scores(changed,query,catalog)

    def test_mechanical_certificate_pins_actual_head_and_exact250_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);artifact=root/'head.fake';artifact.write_bytes(b'CPU receipt fixture')
            head_sha=evidence.sha(artifact)
            receipt={'version':'query-intent-independent-TRAIN-audit.v3','status':'passed-independent-TRAIN-replay',
                'privateQualityLabelsRead':False,'sourceManifestSha256':audit.GATE_PLAN_SHA,'headSha256':head_sha,
                'auditSourceSha256':audit.MECHANICAL_AUDIT_SOURCE_SHA,
                'replay':{'observedReplayOptimizerSteps':250,'everyFinalParameterBitwiseMatches':True},
                'twoReloadsExactLogitParity':True,'reloadSavedParity':{'rows':1272,'exactSavedProbabilityParity':True},
                'inputSha256':{str(artifact):head_sha}}
            path=root/'audit.json';evidence.write(path,receipt)
            self.assertEqual(audit.mechanical_receipt(path,head_sha),receipt)
            changed=deepcopy(receipt);changed['replay']['observedReplayOptimizerSteps']=249
            path.write_text(__import__('json').dumps(changed))
            with self.assertRaises(evidence.AuditError):audit.mechanical_receipt(path,head_sha)
            path.write_text(__import__('json').dumps(receipt));artifact.write_bytes(b'tampered despite unchanged pass claim')
            with self.assertRaisesRegex(evidence.AuditError,'changed'):audit.mechanical_receipt(path,head_sha)

    def fixture(self):
        products=[{'productId':str(i),'name':'Product '+str(i),'category':'test','itemPrice':i+1,'searchTerms':[],
                   'colorVariants':[{'colorName':'Navy'}],'variants':[{'colorName':'Navy','variantId':str(i)+'-navy'}]} for i in range(16)]
        public=[];predictions=[];p=.35;v5=[i/32 for i in range(16)];raw=[float(16-i) for i in range(16)]
        soft=torch.tensor(raw).softmax(-1);scores=((1-p)*soft+p*(torch.tensor(v5)+.05*soft)).tolist()
        for i in range(172):
            for order in ('canonical','reversed'):
                row={'caseId':str(i)+'-'+order,'profileId':str(i),'userGroupId':'g'+str(i),'split':'fake-public',
                     'orderLabel':order,'sourceProfileId':None,'controlName':None,'publicRequest':{'query':'Browse the store'},
                     'publicRequestSha256':'explicit-fake-public-hash','queryTextSha256':'explicit-fake-query-hash'}
                public.append(row)
                request,policy,_=audit.context.checked_context_request(row['publicRequest'],products)
                guarded,checks=audit.guarded_order(scores,request,policy,products)
                predictions.append({k:v for k,v in row.items() if k!='publicRequest'}|
                    {'pSpecific':p,'v5Scores16':v5,'rawSourceScores16':raw,'scores16':scores,
                     'fusionPolicy':{'gateCheckpointSha256':'gatefake'},'order':guarded,'guardChecks':checks,
                     'rankedProductIds':[products[j]['productId'] for j in guarded],
                     'rawLearnedOrder':sorted(range(16),key=lambda j:(-scores[j],j)),
                     'copyGenerated':False,'variantsOrImagesScored':False})
        return products,public,predictions,{'explicit-fake-query-hash':(p,v5)}

    def test_all344_guarded_combined_rows_replay(self):
        products,public,rows,expected=self.fixture()
        audit.check_combined_rows(rows,public,expected,products,'gatefake')
        with self.assertRaisesRegex(evidence.AuditError,'denominator'):
            audit.check_combined_rows(rows[:-1],public,expected,products,'gatefake')

    def test_wrong_query_probability_v5_identity_or_claim_refused(self):
        products,public,rows,expected=self.fixture();original=deepcopy(rows)
        for mutation in [lambda r:r.update(pSpecific=.36),lambda r:r['v5Scores16'].__setitem__(0,.2),
                         lambda r:r.update(profileId='different'),lambda r:r.update(copyGenerated=True),
                         lambda r:r['fusionPolicy'].update(gateCheckpointSha256='other')]:
            rows=deepcopy(original);mutation(rows[0])
            with self.assertRaises(evidence.AuditError):audit.check_combined_rows(rows,public,expected,products,'gatefake')

    def test_correctly_rehashed_wrong_blend_and_guards_fail(self):
        products,public,rows,expected=self.fixture();original=deepcopy(rows)
        rows[0]['scores16'][0]+=.001
        with self.assertRaisesRegex(evidence.AuditError,'blend differs'):
            audit.check_combined_rows(rows,public,expected,products,'gatefake')
        rows=deepcopy(original);rows[0]['order']=rows[0]['order'][:-1]
        with self.assertRaisesRegex(evidence.AuditError,'guard/exact'):
            audit.check_combined_rows(rows,public,expected,products,'gatefake')

    def test_failed_input_admission_never_reads_private_quality(self):
        from unittest.mock import patch
        original=evidence.read;opened=[]
        def guarded(path):
            name=Path(path).name;opened.append(name)
            if 'labels' in name:raise AssertionError('unexpected private label read')
            if name=='PERSONA_COMBINED_QUERY_REPLAY_AUDIT_FROZEN_V1.json':return {'sha256':{'fake':'fake'}}
            return original(path)
        with tempfile.TemporaryDirectory() as temp,patch.object(evidence,'read',side_effect=guarded),patch.object(evidence,'pins'):
            result=audit.audit(Path(temp)/'missing-combined',Path(temp)/'missing-query',Path(temp)/'head',Path(temp)/'export',Path(temp)/'mechanical',Path(temp)/'out')
        self.assertEqual(result['status'],'failed');self.assertFalse(result['privateQualityLabelsRead'])
        self.assertTrue(result['servingPending']);self.assertFalse(result['overallPromotionPass'])
        self.assertFalse(any('labels' in name for name in opened))


if __name__=='__main__':unittest.main()
