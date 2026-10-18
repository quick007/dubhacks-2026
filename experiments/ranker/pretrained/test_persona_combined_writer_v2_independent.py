"""Independent fake writer-to-frozen-auditor bridge; no production artifacts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from . import store_persona_combined_predictions_v2 as writer
from . import store_persona_combined_query_v1 as queries
from . import audit_persona_combined_query_replay_v1 as verifier


class WriterV2IndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.old_threads)

    def fixture(self):
        products=[{'productId':str(i),'name':'Fake product '+str(i),'category':'Fake category',
                   'itemPrice':i+1,'searchTerms':[],'colorVariants':[{'colorName':'Navy'}],
                   'variants':[{'variantId':str(i)+'-navy','colorName':'Navy'}]} for i in range(16)]
        rows=[];request={'query':'Browse the store','itemBudget':50,'hardRequirements':[]}
        text=queries.runtime_text(request,products);key=queries.digest_text(text)
        for i in range(172):
            for order in ('canonical','reversed'):
                rows.append({'caseId':str(i)+'-'+order,'profileId':str(i),'userGroupId':'fakegroup'+str(i),
                    'split':'fake-public','orderLabel':order,'sourceProfileId':None,'controlName':None,
                    'publicRequest':deepcopy(request),'publicRequestSha256':queries.digest_text(queries.canonical(request)),
                    'queryTextSha256':key})
        raw={}
        for model_index,model in enumerate(writer.MODEL_FILES):
            values=[(i+1+model_index)/448 for i in range(16)]
            raw[model]=[{k:r[k] for k in ('caseId','profileId','userGroupId','split','orderLabel','sourceProfileId','controlName')}
                        |{'scores16':list(values)} for r in rows]
        # Real v5 head output is float32, whereas the raw TRAIN prior retains
        # Python JSON fractions. Keep those two provenance contracts distinct.
        v5=torch.tensor([i/73 for i in range(16)],dtype=torch.float32).tolist()
        return rows,raw,products,{key:(.37,v5)}

    def test_original_fraction_json_survives_all1376_rows_and_frozen_checker(self):
        rows,raw,products,query=self.fixture()
        self.assertNotEqual(raw['TRAIN-popularity'][0]['scores16'],torch.tensor(raw['TRAIN-popularity'][0]['scores16']).tolist())
        output=writer.compose_rows(rows,raw,query,products,gate_sha='a'*64)
        with tempfile.TemporaryDirectory() as temp:
            for model,records in output.items():
                path=Path(temp)/(model+'.json');path.write_text(json.dumps(records,allow_nan=False))
                loaded=json.loads(path.read_text())
                verifier.check_combined_rows(loaded,rows,query,products,'a'*64)
                for original,result in zip(raw[model],loaded,strict=True):
                    self.assertEqual(result['rawSourceScores16'],original['scores16'])
        self.assertEqual(sum(map(len,output.values())),1376)

    def test_cast_only_changes_blend_arithmetic_not_saved_raw_source(self):
        rows,raw,products,query=self.fixture()
        output=writer.compose_rows(rows,raw,query,products,gate_sha='a'*64)
        p,v5=next(iter(query.values()))
        for model,records in output.items():
            source=torch.tensor(raw[model][0]['scores16'],dtype=torch.float32)
            soft=source.softmax(-1);expected=((1-p)*soft+p*(torch.tensor(v5,dtype=torch.float32)+.05*soft)).tolist()
            self.assertEqual(records[0]['scores16'],expected)
            self.assertEqual(records[0]['rawLearnedOrder'],sorted(range(16),key=lambda i:(-expected[i],i)))

    def test_raw_producer_arrays_are_not_mutated_and_old_cast_repro_detected(self):
        rows,raw,products,query=self.fixture();before=deepcopy(raw)
        output=writer.compose_rows(rows,raw,query,products,gate_sha='a'*64)
        self.assertEqual(raw,before)
        record=output['TRAIN-popularity'][0]
        old_cast=torch.tensor(record['rawSourceScores16'],dtype=torch.float32).tolist()
        self.assertNotEqual(old_cast,raw['TRAIN-popularity'][0]['scores16'])
        self.assertEqual(record['rawSourceScores16'],raw['TRAIN-popularity'][0]['scores16'])


if __name__=='__main__':unittest.main()
