"""CPU contracts with explicitly artificial unit vectors; no native evidence."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from . import store_function_adaptation_v4 as m
from . import store_function_data_v4 as data_api


class StoreFunctionAdaptationV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = data_api.DATA
        cls.model = Path(m.DEFAULT_MODEL)
        cls.prior = Path('data/ranker/pretrained/catalog-user-fusion-v1-attempt-1')
        cls.prepared = m.prepare(cls.data,cls.model,cls.prior)
        cls.packet,cls.frozen,cls.products,cls.cases,cls.rows = cls.prepared
        cls.labels=json.loads((cls.data/'train-labels.json').read_text())

    def test_public_preparation_has104_whole_inputs_no_label_parsing(self):
        original=Path.read_text;parsed=[]
        def trace(path,*args,**kwargs):
            parsed.append(path.name)
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',trace):
            prepared=m.prepare(self.data,self.model,self.prior)
        self.assertNotIn('evaluation-labels.json',parsed)
        self.assertNotIn('train-labels.json',parsed)
        self.assertEqual(prepared,self.prepared)
        self.assertEqual(len(self.rows),104)
        self.assertEqual([r['split'] for r in self.rows],['train']*80+['evaluation']*24)
        self.assertTrue(all(r['batchShape']==[1,r['tokens']] and 0<r['tokens']<=512 for r in self.rows))
        self.assertFalse(self.packet['evaluationLabelsParsed'])
        self.assertEqual(self.packet['thresholdChosenBeforeEvaluation'],.5)
        for row in self.rows:
            for private in ('caseId','family','targetProductIndex','desiredProductIndices','publicSupport','personaTruth'):
                self.assertNotIn(private,row['text'])

    def test_train_mean_and_desired_targets_exclude_eval_and_eligibility(self):
        class Store:
            def __init__(self):self.members=[]
            def load(self,row):
                self.members.append(row)
                return torch.full((1,4096),float(len(self.members))),{}
        store=Store();q,c,targets,means,meta=m.fit_inputs(store,self.rows,self.cases,self.labels)
        self.assertEqual(len(store.members),80)
        self.assertTrue(all(r['split']=='train' for r in store.members))
        self.assertEqual(targets.shape,(64,16))
        self.assertEqual(meta['queryMembers'],[x['caseId'] for x in self.cases if x['split']=='train'])
        self.assertTrue(torch.equal(means['query'],q.double().mean(0).float()))
        self.assertTrue(torch.equal(means['catalog'],c.double().mean(0).float()))
        for i,label in enumerate(self.labels):
            self.assertEqual(targets[i].nonzero().flatten().tolist(),label['desiredProductIndices'])
            if not label['relevantProductIndices']:
                self.assertGreater(int(targets[i].sum()),0)
        with self.assertRaises(ValueError):m.fit_inputs(store,self.rows,self.cases,self.labels[:-1])

    def test_query_sigmoid_is_independent_and_catalog_softmax_is_identity_distribution(self):
        means={r:torch.zeros(4096) for r in ('query','catalog')};head=m.IndependentFunctionHead(means)
        self.assertEqual(sum(p.numel() for p in head.parameters()),131104)
        query=head.query_probabilities(torch.zeros((1,4096)))
        self.assertEqual(query.tolist(),[[.5]*16]);self.assertEqual(float(query.detach().sum()),8)
        self.assertAlmostEqual(float(head.catalog_probabilities(torch.zeros((1,4096))).detach().sum()),1)

    def test_exact_names_are_whole_query_recognition_not_incidental_or_negative_mentions(self):
        for query in ('Trail mug','trail mug.','Find Trail mug.', 'Please find the item named "Trail mug".',
                      'Show me Trail mug!', '  "Trail mug"  '):
            self.assertEqual(m.exact_query_target(query,self.products),0)
        for query in ('Do not show Trail mug.','I dislike Trail mug.','Trail mug or Everyday daypack',
                      'Find Trail mugs.','Show me something like Trail mug.', 'Not Trail mug'):
            self.assertIsNone(m.exact_query_target(query,self.products))
        self.assertEqual(sum(m.exact_query_target(c['request']['query'],self.products) is not None for c in self.cases),5)

    def test_fixed_threshold_full_rank_no_padding_and_constraints_precede_policy(self):
        q=data_api.request('Choose matching items.');scores=[.1]*16;scores[0]=.5;scores[4]=.8
        decision=m.decide(scores,q,self.products)
        self.assertEqual(decision['selectedProductIndices'],[4,0])
        self.assertEqual(len(decision['rawRank']),16);self.assertEqual(len(decision['unthresholdedTopSix']),6)
        self.assertEqual(decision['shortfallToSix'],4)
        self.assertEqual(m.decide([.499]*16,q,self.products)['selectedProductIndices'],[])
        exact=data_api.request('Find Trail mug.',searches=['bottle'],budget=23)
        routed=m.decide(scores,exact,self.products,apply_exact_policy=True)
        self.assertTrue(routed['exactNameRouteTaken']);self.assertTrue(routed['exactNameGuardRejected'])
        self.assertEqual(routed['selectedProductIndices'],[])
        exact['itemBudget']=24
        self.assertEqual(m.decide([.01]*16,exact,self.products,apply_exact_policy=True)['selectedProductIndices'],[0])
        self.assertEqual(m.decide([.01]*16,exact,self.products)['selectedProductIndices'],[])
        for attribute in ('stock','size'):
            request=data_api.request('Trail mug',requirements=[{'attribute':attribute,'value':'required'}])
            self.assertEqual(m.decide([1.]*16,request,self.products,apply_exact_policy=True)['selectedProductIndices'],[])
        for invalid in (True,float('nan'),float('inf'),-.1,1.1):
            with self.assertRaises(ValueError):m.decide([invalid]+[.5]*15,q,self.products)

    def test_exact_checkpoint_reload_and_exclusive_preservation(self):
        means={r:torch.zeros(4096) for r in ('query','catalog')};head=m.IndependentFunctionHead(means)
        meta={'meanSha256':{r:m.tensor_digest(v) for r,v in means.items()}}
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'head.pt';sha=m.save_head(path,head,meta);restored=m.load_head(path,sha,meta)
            self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in head.state_dict().items()))
            with self.assertRaises(FileExistsError):m.save_head(path,head,meta)
            with self.assertRaises(ValueError):m.load_head(path,'0'*64,meta)

    def test_prediction_controls_accept_no_labels_and_policy_counts_stay_separate(self):
        means={r:torch.zeros(4096) for r in ('query','catalog')}
        models={r:m.IndependentFunctionHead(means) for r in ('learned','random')}
        cases=[x for x in self.cases if x['split']=='evaluation'];queries=torch.zeros((24,4096));catalog=torch.zeros((16,4096))
        predicted=m.predict(models,torch.full((16,),.1),queries,catalog,cases,self.products)
        before=deepcopy(predicted)
        for arm,rows in predicted.items():
            self.assertEqual(rows['fixedCases'],24)
            for i,c in enumerate(cases):
                row=rows['cases'][c['caseId']];self.assertEqual(len(row['scores']),16)
                self.assertEqual(row['querySource'],None if arm=='prior' else cases[(i+1)%24]['caseId'] if arm=='permuted-query' else c['caseId'])
        evaluation=json.loads((self.data/'evaluation-labels.json').read_text());pairs=json.loads((self.data/'preference-pairs.json').read_text())
        result=m.evaluate(predicted,cases,evaluation,pairs)
        self.assertEqual(predicted,before)
        self.assertEqual(result['prior']['semantic']['hitAtOne'],0)
        self.assertEqual(result['prior']['policy']['hitAtOne'],3)
        self.assertEqual(result['prior']['exactNameRouteCount'],3)
        self.assertEqual(result['prior']['policyChangedOutputCount'],3)
        self.assertEqual(result['prior']['semantic']['fixedPreferencePairs'],8)
        train=[x for x in self.cases if x['split']=='train']
        shuffled=m.predict(models,torch.full((16,),.1),torch.zeros((64,4096)),catalog,train,self.products)
        for i,c in enumerate(train):
            self.assertEqual(shuffled['permuted-query']['cases'][c['caseId']]['querySource'],train[(i+17)%64]['caseId'])
        self.assertIn('not guaranteed disjoint',m.CONFIG['permutationClaim'])

    def _patch_cpu_run(self,stack,out,*,verify_error=None,close_error=None):
        stack.enter_context(patch.object(m,'prepare',return_value=deepcopy(self.prepared)))
        stack.enter_context(patch('experiments.ranker.pretrained.catalog_fusion.generalization_v2.native_owner_lock',return_value=nullcontext()))
        provenance={'evidenceSource':'cpu-contract-unit-vectors','modelRevision':'cpu-test',
            'weightsSha256':'1'*64,'tokenizerSha256':'2'*64,'templateSha256':'3'*64,
            'quantization':{'bits':4}}
        stack.enter_context(patch.object(m,'verify_model',side_effect=verify_error,return_value=provenance))
        stack.enter_context(patch.object(m,'assert_model_files'))
        calls=[]
        class CpuEncoder:
            evidence_source='cpu-contract-unit-vectors';load_seconds=0.
            def __init__(self,model):pass
            def encode(self,text,row):
                if row['split']=='evaluation':
                    assert (out/'HEADS_FROZEN.json').exists()
                    assert len(calls)>=80
                calls.append(row)
                v=torch.zeros((1,4096));v[0,int(hashlib.sha256(text.encode()).hexdigest(),16)%4096]=1.
                return v,{'batchRows':1,'paddedTokensPerRow':row['tokens'],'truncatedRows':0,
                          'backboneForwardCalls':1,'encodeTotalMs':0.}
            def memory(self):return {'cpuContractOnly':True}
            def close(self):
                if close_error:raise close_error
        stack.enter_context(patch.object(m,'NativeEncoder',CpuEncoder))
        stack.enter_context(patch('sys.stdout',new=io.StringIO()))
        return calls

    def test_full_mock_cpu_run104_once80plus80updates_freeze_before_eval_labels(self):
        original=Path.read_text;eval_reads=[]
        with tempfile.TemporaryDirectory() as td,ExitStack() as stack:
            out=Path(td)/'run';packet=Path(td)/'packet.json';packet.write_text(json.dumps(self.packet))
            calls=self._patch_cpu_run(stack,out)
            def trace(path,*args,**kwargs):
                if path.name=='evaluation-labels.json':
                    self.assertTrue((out/'PREDICTIONS_FROZEN.json').exists());eval_reads.append(path)
                return original(path,*args,**kwargs)
            stack.enter_context(patch.object(Path,'read_text',trace))
            report=m.run(out,self.data,packet,self.model,self.prior,'root')
            self.assertEqual(len(calls),104);self.assertEqual(len(eval_reads),1)
            self.assertEqual(len({tuple(x['member']) for x in calls}),104)
            self.assertEqual(report['nativeForwards'],104);self.assertEqual(report['learnedUpdates'],80);self.assertEqual(report['priorUpdates'],80)
            self.assertEqual(len(report['freshRequestEndpoints']),24)
            self.assertFalse(report['promoted']);self.assertFalse((out/'FAILED.json').exists())
            heads=json.loads((out/'HEADS_FROZEN.json').read_text())
            self.assertEqual(len(heads['trainingSteps']),80);self.assertEqual(len(heads['priorSteps']),80)
            self.assertFalse(heads['evaluationFeaturesEncoded'])
            predictions=json.loads((out/'PREDICTIONS_FROZEN.json').read_text())
            self.assertFalse(predictions['evaluationLabelsParsed'])
            for rid,endpoint in report['freshRequestEndpoints'].items():
                encoded=m.canonical(endpoint['payload']).encode()
                self.assertEqual(endpoint['payloadSha256'],hashlib.sha256(encoded).hexdigest())
                self.assertEqual(endpoint['payloadBytes'],len(encoded))
                cached=predictions['predictions']['evaluation']['learned']['cases'][rid]
                self.assertEqual(endpoint['payload'],{k:cached[k] for k in ('scores','semantic','policy')})
            self.assertEqual(len(json.loads((out/'RAW_FEATURES_FROZEN.json').read_text())['files']),208)
            self.assertEqual(len((out/'fresh-endpoints.jsonl').read_text().splitlines()),24)
            before=(out/'report.json').read_bytes()
            with self.assertRaises(FileExistsError):m.run(out,self.data,packet,self.model,self.prior,'root')
            self.assertEqual((out/'report.json').read_bytes(),before)

    def test_receipts_capture_verification_preparation_postprocess_and_cleanup_failure(self):
        for phase in ('prepare','verification','postprocess','cleanup'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as td,ExitStack() as stack:
                out=Path(td)/'failed';packet=Path(td)/'packet.json';packet.write_text(json.dumps(self.packet))
                self._patch_cpu_run(stack,out,verify_error=ValueError('verify') if phase=='verification' else None,
                                    close_error=ValueError('close') if phase=='cleanup' else None)
                if phase=='prepare':stack.enter_context(patch.object(m,'prepare',side_effect=ValueError('prepare')))
                if phase=='postprocess':stack.enter_context(patch.object(m,'evaluate',side_effect=ValueError('evaluate')))
                with self.assertRaises(ValueError):m.run(out,self.data,packet,self.model,self.prior,'root')
                receipt=json.loads((out/'FAILED.json').read_text())
                self.assertFalse(receipt['promotable']);self.assertEqual(receipt['fixedNativeForwards'],104)
                self.assertEqual(receipt['fixedEvaluationCases'],24)
                expected={'prepare':'cpu-preparation','verification':'model-verification',
                          'postprocess':'postfreeze-evaluation','cleanup':'native-cleanup'}[phase]
                self.assertEqual(receipt['phase'],expected)
                if phase in ('postprocess','cleanup'):
                    self.assertEqual(receipt['completedNativeForwards'],104)
                    self.assertEqual(receipt['completedEvaluationEndpoints'],24)
                    self.assertEqual(receipt['completedLearnedUpdates'],80)

    def test_cooperative_timeout_after_returned_forward_keeps_actual_count(self):
        with tempfile.TemporaryDirectory() as td,ExitStack() as stack:
            out=Path(td)/'failed';packet=Path(td)/'packet.json';packet.write_text(json.dumps(self.packet))
            calls=self._patch_cpu_run(stack,out)
            def guard(started):
                if calls:raise TimeoutError('after first returned forward')
            stack.enter_context(patch.object(m,'check_elapsed',side_effect=guard))
            with self.assertRaises(TimeoutError):m.run(out,self.data,packet,self.model,self.prior,'root')
            receipt=json.loads((out/'FAILED.json').read_text())
            self.assertEqual(receipt['completedNativeForwards'],1)
            self.assertEqual(receipt['serializedFeatures'],0)
            self.assertEqual(receipt['fixedNativeForwards'],104)
        with patch.object(m.time,'perf_counter',return_value=181):
            with self.assertRaises(TimeoutError):m.check_elapsed(0)
        with patch.object(m.time,'perf_counter',return_value=180):m.check_elapsed(0)


if __name__=='__main__':
    unittest.main()
