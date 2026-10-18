"""CPU contracts only. Synthetic unit vectors never count as model evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from .store_function_adaptation import (CONFIG, FunctionHead, atom_gap, build, fit_inputs,
    freeze, guard, guard_probes, label_predictions, load_head, predict, product_text,
    rank, read_public, request_text, save_head, validate_request)
from .store_function_adaptation import check_elapsed, write_failure, run

class StoreFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.products,cls.cases,cls.labels,cls.evaluation,cls.bridge=build()
    def test_counts_and_disjoint_independent_preference_pairs(self):
        self.assertEqual(len(self.products),16);self.assertEqual(len(self.cases),48)
        self.assertEqual(len(self.labels),32);self.assertEqual(len(self.evaluation['pairs']),8)
        self.assertEqual(len({request_text(c['request']) for c in self.cases}),48)
        self.assertEqual({x['targetProductIndex'] for x in self.labels},set(range(16)))
        lookup={c['caseId']:c for c in self.cases}
        for a,b in self.evaluation['pairs']:
            qa,qb=lookup[a]['request'],lookup[b]['request']
            self.assertEqual(qa['query'],qb['query']);self.assertNotEqual(qa['preferences'],qb['preferences'])
    def test_private_targets_ids_images_never_predictive_text(self):
        for c in self.cases:
            text=request_text(c['request'])
            for forbidden in (c['caseId'],'targetProductIndex','publicSupport','caseId','split','persona'):
                self.assertNotIn(forbidden,text)
        for p in self.products:
            text=product_text(p)
            for forbidden in ('image','productId','variantId','src','scenario','swatch'):self.assertNotIn(forbidden,text)
            self.assertIn(p['name'],text);self.assertIn(p['category'],text)
        q=copy.deepcopy(self.cases[0]['request']);q['targetProductIndex']=3
        with self.assertRaises(ValueError):request_text(q)
    def test_seven_factor_gap_is_not_repaired_by_invented_aliases(self):
        gap=atom_gap(self.products)
        self.assertEqual(gap['emptyAtomProducts'],15);self.assertEqual(gap['distinctFocusedProductTexts'],2)
        self.assertEqual(gap['exactDeclaredPositiveAtomsByProduct'][14],['organizer'])
        self.assertEqual(gap['exactDeclaredPositiveAtomsByProduct'][3],[])
    def test_guard_probes_precede_high_scores_and_disclose_shortfalls(self):
        for p in guard_probes(self.products):
            scores=[float(1000-i) for i in range(16)]
            r=rank(scores,p['request'],self.products,6)
            self.assertEqual(r['eligibleRank'],p['expectedEligibleProductIndices'])
            self.assertEqual(r['shortfall'],max(0,6-len(r['eligibleRank'])))
            self.assertTrue(all(r['guards'][i]['eligible'] for i in r['productIndices']))
        self.assertEqual(len(guard_probes(self.products)),8)
    def test_strict_types_finite_scores_and_unknown_hard_requirements(self):
        for budget in (True,float('nan'),float('inf'),-1):
            q=copy.deepcopy(self.cases[0]['request']);q['itemBudget']=budget
            with self.assertRaises(ValueError):validate_request(q)
        for score in (True,float('nan'),float('inf')):
            with self.assertRaises(ValueError):rank([score]+[0.]*15,self.cases[0]['request'],self.products)
        q=copy.deepcopy(self.cases[0]['request']);q['hardRequirements']=[{'attribute':'invented','value':'yes'}]
        with self.assertRaises(ValueError):guard(q,self.products[0])
    def test_train_mean_membership_excludes_heldout_features_and_labels(self):
        class Store:
            def __init__(self):self.loaded=[]
            def load(self,row):
                self.loaded.append(row);return torch.full((1,4096),float(len(self.loaded))),{}
        store=Store();rows=[{'role':'catalog','caseId':str(i)} for i in range(16)] + [{'role':'request','caseId':c['caseId']} for c in self.cases]
        q,c,t,means,meta=fit_inputs(store,rows,self.cases,self.labels)
        self.assertEqual(len(store.loaded),48)
        self.assertTrue(all(r['role']=='catalog' or r['caseId'].startswith('train-') for r in store.loaded))
        self.assertEqual(meta['queryMembers'],[x['caseId'] for x in self.cases if x['split']=='train'])
        self.assertTrue(torch.equal(means['query'],q.double().mean(0).float()))
        self.assertTrue(torch.equal(means['catalog'],c.double().mean(0).float()))
    def test_exact_checkpoint_state_reload_and_exclusive_save(self):
        from .catalog_fusion.inputs import tensor_digest
        means={r:torch.zeros(4096) for r in ('query','catalog')};model=FunctionHead(means)
        self.assertEqual(sum(x.numel() for x in model.parameters()),CONFIG['parameters'])
        meta={'meanSha256':{r:tensor_digest(x) for r,x in means.items()}}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'head.pt';sha=save_head(path,model,meta);restored=load_head(path,sha,meta)
            self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()))
            with self.assertRaises(FileExistsError):save_head(path,model,meta)
    def test_predictions_accept_no_labels_and_evaluator_does_not_mutate_scores(self):
        means={r:torch.zeros(4096) for r in ('query','catalog')};models={r:FunctionHead(means) for r in ('learned','random')}
        cases=[c for c in self.cases if c['split']=='evaluation'];q=torch.zeros((16,4096));c=torch.zeros((16,4096))
        predicted=predict(models,torch.zeros(16),q,c,cases,self.products);before=copy.deepcopy(predicted)
        labels={x['caseId']:x['targetProductIndex'] for x in self.evaluation['cases']}
        result=label_predictions(predicted,labels,self.evaluation['pairs'],self.products,cases)
        self.assertEqual(predicted,before)
        self.assertEqual(result['public-oracle']['topOneCorrect'],16)
        self.assertEqual(result['public-oracle']['pairsBothCorrect'],8)
        self.assertEqual(len(predicted['learned']['cases'][cases[0]['caseId']]['scores']),16)
    def test_frozen_data_hashes_reject_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'data';freeze(data);read_public(data)
            with self.assertRaises(FileExistsError):freeze(data)
            path=data/'train-labels.json';path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'artifact changed'):read_public(data)
    def test_elapsed_guard_rejects_overrun_including_last_forward(self):
        with patch('experiments.ranker.pretrained.store_function_adaptation.time.perf_counter',return_value=181):
            with self.assertRaises(TimeoutError):check_elapsed(0)
        with patch('experiments.ranker.pretrained.store_function_adaptation.time.perf_counter',return_value=180):
            check_elapsed(0)
    def test_failure_receipt_keeps_fixed_denominators_and_partial_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_failure(tmp,TimeoutError('late forward'),'fresh-evaluation-endpoints',64,[{}]*63,{'case':{}},0)
            receipt=json.loads((Path(tmp)/'FAILED.json').read_text())
            self.assertEqual(receipt['completedNativeForwards'],64)
            self.assertEqual(receipt['serializedFeatures'],63)
            self.assertEqual(receipt['fixedNativeForwards'],64)
            self.assertEqual(receipt['fixedEvaluationCases'],16)
            self.assertEqual(receipt['fixedTrainingCases'],32)
            self.assertFalse(receipt['promotable'])
    def test_outer_failure_receipt_and_preserved_existing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new'
            def failed(*args):
                out.mkdir();raise ValueError('verification failed')
            with patch('experiments.ranker.pretrained.store_function_adaptation._run',side_effect=failed):
                with self.assertRaises(ValueError):run(out,None,None,None,None,'root')
            receipt=(out/'FAILED.json').read_bytes()
            with patch('experiments.ranker.pretrained.store_function_adaptation._run',side_effect=FileExistsError('preserve')):
                with self.assertRaises(FileExistsError):run(out,None,None,None,None,'root')
            self.assertEqual((out/'FAILED.json').read_bytes(),receipt)

if __name__=='__main__':unittest.main()
