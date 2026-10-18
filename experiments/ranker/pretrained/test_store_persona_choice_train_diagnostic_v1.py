"""CPU protocol/math fixtures, never native or trained-quality evidence."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from . import store_persona_choice_train_diagnostic_v1 as d

class DiagnosticTests(unittest.TestCase):
    def plan(self,root):
        ids=['p'+str(i) for i in range(16)];rows=[]
        for i in range(448):
            for order in ('canonical','reversed'):
                choices=ids if order=='canonical' else ids[::-1]
                rows.append({'caseId':f'profile-{i}-{order}','profileId':f'profile-{i}','userGroupId':f'group-{i}',
                    'source':'old320' if i<320 else 'new128','order':order,'choices':[{'productId':p} for p in choices],
                    'publicRequest':{'query':'public fixture only','publicHistory':[]}})
        a=np.array([1.,2.],dtype=np.float32);b=np.zeros(2,dtype=np.float32)
        labels=root/'labels.json';labels.write_text(json.dumps({'labels':[{'profileId':f'profile-{i}',
            'relevanceGrades':{p:3 if p=='p0' else 0 for p in ids},'alreadyOwnedProductIds':[]} for i in range(448)]}))
        packet={'version':d.VERSION,'recipe':d.RECIPE,'sourcePlanSha256':d.PLAN_SHA,'adapterSha256':d.ADAPTER_SHA,
            'sha256':{},'run':str(root),'rows':rows,'canonicalProductIds':ids,
            'initialAdapterSha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in {'lora_a':a,'lora_b':b}.items()},
            'labelFiles':{'TRAIN-only':{'file':str(labels),'sha256':d.sha(labels)}}}
        plan=root/'plan-input.json';plan.write_text(json.dumps(packet));return plan,packet,a,b

    def test_default_preflight_never_enters_native_or_creates_output(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);plan,packet,a,b=self.plan(root)
            with patch.object(d,'owned_model',side_effect=AssertionError('native forbidden')),patch.object(d,'pincheck'):
                result=d.run(plan,d.sha(plan),root/'out')
            self.assertEqual(result['nativeCalls'],0);self.assertFalse((root/'out').exists())
            with patch.object(d,'pincheck'):
                with self.assertRaisesRegex(ValueError,'sole native owner'):d.run(plan,d.sha(plan),root/'out',execute=True)

    def test_route_remap_and_ties_use_canonical_catalog_order_in_both_prompt_orders(self):
        ids=['z'+str(i) for i in range(16)];row={'caseId':'p-reversed','profileId':'p','userGroupId':'g','source':'old320','order':'reversed','choices':[{'productId':p} for p in ids[::-1]]}
        pred=d.canonical_prediction(row,'base',[0.]*16,ids)
        self.assertEqual(pred['rankedProductIds'],ids)
        pred=d.canonical_prediction(row,'base',[20.]+[0.]*15,ids)
        self.assertEqual(pred['rankedProductIds'][0],ids[-1])
        with self.assertRaises(ValueError):d.canonical_prediction(row,'base',[float('nan')]*16,ids)

    def test_multi_positive_ce_and_maximum_grade_set_are_not_arbitrary_single_target(self):
        row={'profileId':'p','rankedProductIds':['b','a','c'],'scoresByProductId':{'a':1.,'b':2.,'c':0.}}
        labels={'p':{'relevanceGrades':{'a':3,'b':3,'c':0},'alreadyOwnedProductIds':['b']}}
        m=d.subset_metrics([row],labels)
        self.assertEqual(m['maximumGradeTop1'],1);self.assertEqual(m['ownedTop1'],1)
        self.assertAlmostEqual(m['routeCE'],np.log(np.exp(1)+np.exp(2)+1)-1.5)
        self.assertAlmostEqual(m['targetEntropy'],np.log(2))

    def test_all_predictions_freeze_before_label_file_join_and_cleanup_on_failure(self):
        for failing in (False,True):
            with self.subTest(failing=failing),tempfile.TemporaryDirectory() as name:
                root=Path(name);plan,packet,a,b=self.plan(root);out=root/'out';adapter=SimpleNamespace(lora_a=a,lora_b=b)
                mx=SimpleNamespace(eval=lambda *args:None,synchronize=lambda:None,load=lambda filename:{'lora_a':a+1,'lora_b':b+1})
                bb=SimpleNamespace(mx=mx,check_memory=lambda:None);reads=[]
                @contextmanager
                def owned(packet,report,save):
                    try:yield bb,adapter
                    finally:report.update(frozen927HashesUnchanged=True,shutdownCompleted=True,sourcePinsUnchanged=True);save()
                oldread=d.read
                def read(file):
                    if str(file)==str(root/'labels.json'):
                        self.assertTrue((out/'ALL_TRAIN_PREDICTIONS_FROZEN.json').exists());reads.append('TRAIN label file')
                    return oldread(file)
                def score(bb,adapter,item):
                    self.assertEqual(set(item),{'x','h','y','dtypes','routeTokenIds'})
                    if failing:raise RuntimeError('fixture projection failure')
                    return np.arange(16,dtype=np.float32)
                pure={'x':None,'h':None,'y':None,'dtypes':{},'routeTokenIds':list(range(16))}
                with patch.object(d,'owned_model',owned),patch.object(d,'pincheck'),patch.object(d.training.pilot,'check_output',lambda x:Path(x)),\
                     patch.object(d,'load_public_feature',return_value=pure),patch.object(d.scoring,'score',score),patch.object(d,'read',read):
                    if failing:
                        with self.assertRaisesRegex(RuntimeError,'projection failure'):d.run(plan,d.sha(plan),out,execute=True,owner='root')
                        report=oldread(out/'report.json');self.assertEqual(report['rows'][0]['status'],'failed');self.assertFalse(reads)
                        self.assertFalse((out/'ALL_TRAIN_PREDICTIONS_FROZEN.json').exists())
                    else:
                        report=d.run(plan,d.sha(plan),out,execute=True,owner='root')
                        self.assertEqual(report['completedLMProjectionCalls'],1792);self.assertEqual(report['nativeBodyForwards'],0)
                        self.assertEqual(reads,['TRAIN label file']);self.assertEqual(oldread(out/'ALL_TRAIN_PREDICTIONS_FROZEN.json')['predictions'],1792)
                    self.assertTrue(oldread(out/'report.json')['shutdownCompleted'])

if __name__=='__main__':unittest.main()
