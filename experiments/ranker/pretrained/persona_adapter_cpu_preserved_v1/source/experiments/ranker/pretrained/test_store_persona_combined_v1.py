from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import torch

from . import store_persona_combined_query_v1 as query
from . import store_persona_combined_predictions_v1 as combined
from . import store_persona_adapter_runtime_v2 as runtime


class PublicQueryPlanTests(unittest.TestCase):
    def full(self):
        rows = []
        for split, n in query.EXPECTED_NONFIT.items():
            for i in range(n//2):
                for order in ('canonical', 'reversed'):
                    pid = f'{split}-{i}'
                    rows.append({'caseId':pid+'-'+order,'profileId':pid,'userGroupId':pid,'split':split,'order':order,
                        'sourceProfileId':None,'controlName':None,'publicRequest':{'query':'Browse the store'},
                        'targetWeights':['TRAIN/dummy values must not survive'], 'privateGrade':'must not survive'})
        rows.append({'split':'train','privateTRAINTarget':'discard entire row'})
        return {'rows':rows}

    def test_all344_public_roster_projection_discards_targets_and_train_rows(self):
        result = query.project_rows(self.full())
        self.assertEqual(len(result),344)
        self.assertTrue(all(set(r)=={'caseId','profileId','userGroupId','split','sourceProfileId','controlName','publicRequest','orderLabel','scope'} for r in result))
        self.assertNotIn('targetWeights',json.dumps(result)); self.assertNotIn('privateGrade',json.dumps(result))
        changed=self.full();changed['rows'].pop(0)
        with self.assertRaisesRegex(ValueError,'all344'):query.project_rows(changed)
        changed=self.full();changed['rows'][0]['caseId']='wrong-order'
        with self.assertRaisesRegex(ValueError,'identity'):query.project_rows(changed)

    def test_same_full_query_text_dedups_context_and_order_but_preserves_guards(self):
        first={'query':'Browse the store','itemBudget':50,'persona':{'background':['observed cycling'],'colorPreferences':[]}}
        second=deepcopy(first);second['persona']['background']=['observed cooking']
        self.assertEqual(query.runtime_text(first),query.runtime_text(second))
        second['itemBudget']=100
        self.assertNotEqual(query.runtime_text(first),query.runtime_text(second))
        explicit={'query':'Trail mug under $20','itemBudget':50}
        text=query.runtime_text(explicit)
        self.assertIn('20.0',text);self.assertIn('Trail mug under $20',text)
        with self.assertRaises(ValueError):query.runtime_text({'query':'x'*501})

    def test_cpu_preflight_and_nonroot_execute_never_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json'; packet={'version':query.VERSION,'recipe':query.RECIPE,'sha256':{},'counts':{'unique':2}}
            query.dump(path,packet)
            result=query.extract(path,query.sha(path),Path(directory)/'out')
            self.assertFalse(result['nativeExecuted']);self.assertFalse((Path(directory)/'out').exists())
            with self.assertRaisesRegex(ValueError,'sole native owner'):
                query.extract(path,query.sha(path),Path(directory)/'out',execute=True,owner='someone-else')


class CombinedPredictionTests(unittest.TestCase):
    def setUp(self):
        from .store_catalog_v1 import load_snapshot
        snapshot=load_snapshot(); self.products=deepcopy(snapshot['inputs']['products'])
        for product,external in zip(self.products,snapshot['externalMapping']['products'],strict=True):
            product['productId']=external['productId'];product['variants']=[{'variantId':v['variantId'],'colorName':c['colorName']}
                for v,c in zip(external['colorVariants'],product['colorVariants'],strict=True)]
        self.rows=[]
        for order in ('canonical','reversed'):
            request={'query':'Browse the store'}; key=query.digest_text(query.runtime_text(request))
            self.rows.append({'caseId':'person-'+order,'profileId':'person','userGroupId':'group','split':'evaluation-main',
                'orderLabel':order,'sourceProfileId':None,'controlName':None,'publicRequest':request,
                'publicRequestSha256':query.digest_text(query.canonical(request)),'queryTextSha256':key})
        self.raw={name:[{k:r[k] for k in ('caseId','profileId','userGroupId','split','orderLabel','sourceProfileId','controlName')}
            |{'scores16':[float(16-i+m) for i in range(16)]} for r in self.rows] for m,name in enumerate(combined.MODEL_FILES)}
        self.values={self.rows[0]['queryTextSha256']:(.35,[i/32 for i in range(16)])}

    def compose(self):return combined.compose_rows(self.rows,self.raw,self.values,self.products,gate_sha='a'*64)

    def test_all_models_share_exact_gate_v5_formula_and_canonical_tie_order(self):
        output=self.compose()
        for name,records in output.items():
            for row in records:
                self.assertEqual(row['pSpecific'],.35);self.assertEqual(row['v5Scores16'],[i/32 for i in range(16)])
                raw=torch.tensor(self.raw[name][0]['scores16'])
                expected=runtime.blended_scores(torch.tensor([i/32 for i in range(16)]),raw,.35).tolist()
                self.assertEqual(row['scores16'],expected)
                self.assertEqual(row['rawLearnedOrder'],sorted(range(16),key=lambda i:(-expected[i],i)))
                self.assertFalse(row['copyGenerated']);self.assertFalse(row['variantsOrImagesScored'])
            self.assertEqual(records[0]['scores16'],records[1]['scores16'])
        self.assertEqual(set(output),set(combined.MODEL_FILES))

    def test_exact_identity_budget_guard_applies_identically_to_every_control(self):
        for row in self.rows:
            row['publicRequest']={'query':'Trail mug under $20'}
            row['publicRequestSha256']=query.digest_text(query.canonical(row['publicRequest']))
            row['queryTextSha256']=query.digest_text(query.runtime_text(row['publicRequest']))
        self.values={self.rows[0]['queryTextSha256']:(.5,[i/32 for i in range(16)])}
        output=self.compose()
        for records in output.values():
            self.assertEqual(records[0]['order'],[])
            self.assertTrue(records[0]['guardChecks']['exactNamePolicy']['ineligibleExactIdentityAbstention'])

    def test_missing_control_wrong_identity_nonfinite_or_request_change_rejects(self):
        original=deepcopy(self.raw)
        for mutate in [lambda: self.raw.pop('latest-owned-repeat'),
                       lambda: self.raw['learned'].pop(),
                       lambda: self.raw['learned'][0].update(profileId='another-person'),
                       lambda: self.raw['learned'][0].update(scores16=[float('nan')]*16)]:
            self.raw=deepcopy(original);mutate()
            with self.assertRaises(ValueError):self.compose()
        self.raw=original;self.rows[0]['publicRequest']['query']='changed'
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.compose()


class QueryNativeProofTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.packet={'version':query.VERSION,'recipe':query.RECIPE,'sha256':{},'uniqueQueries':[{'textSha256':'a'*64}]}
        query.dump(self.root/'plan.json',self.packet);self.plan_sha=query.sha(self.root/'plan.json')
        self.value={'version':query.VERSION,'sourceManifestSha256':self.plan_sha,'features':{'a'*64:torch.eye(1,4096)}}
        self.report={'version':query.VERSION,'sourceManifestSha256':self.plan_sha,'status':'completed-query-features',
            'adapterInstalled':False,'goldParsed':False,'optimizerUpdates':0,'attemptedNativeForwards':1,'physicalNativeForwards':1,
            'savedVectors':1,'expectedNativeForwards':1,'frozen927Unchanged':True,'sourcePinsUnchanged':True,'shutdownCompleted':True,
            'beforeParameterSha256':{str(i):'f'*64 for i in range(927)},'afterParameterSha256':{str(i):'f'*64 for i in range(927)}}
        self.save()
    def save(self):
        torch.save(self.value,self.root/'query-features.pt');self.report['queryFeaturesSha256']=query.sha(self.root/'query-features.pt')
        query.dump(self.root/'report.json',self.report)
    def test_completed_base_only_927_integrity_exact_bundle_and_no_silent_partial_reuse(self):
        _,features,_=combined.query_bundle(self.root,self.plan_sha)
        self.assertTrue(torch.equal(features['a'*64],torch.eye(1,4096)))
        original=deepcopy(self.report)
        for mutate in [lambda r:r.update(adapterInstalled=True),lambda r:r.update(shutdownCompleted=False),
                       lambda r:r.update(savedVectors=0),lambda r:r.update(physicalNativeForwards=0),
                       lambda r:r['afterParameterSha256'].update({'0':'changed'})]:
            self.report=deepcopy(original);mutate(self.report);self.save()
            with self.assertRaisesRegex(ValueError,'query proof'):combined.query_bundle(self.root,self.plan_sha)
        self.report=original;self.value['features']['a'*64]=torch.ones(1,4096);self.save()
        with self.assertRaisesRegex(ValueError,'unit-L2'):combined.query_bundle(self.root,self.plan_sha)

class CombinedWriterFreezeTests(unittest.TestCase):
    setUp = CombinedPredictionTests.setUp
    def test_full344_writer_freezes_all_baseline_files_with_actual_trainer_key_names(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=query.ROOT) as directory:
            root=Path(directory); raw=root/'raw'; raw.mkdir(); qrun=root/'query';qrun.mkdir(); out=root/'combined'
            full=PublicQueryPlanTests().full(); packet_manifest=root/'PUBLIC_MANIFEST.json';packet_manifest.write_text('{}')
            packet_sha=query.sha(packet_manifest);full['sha256']={str(packet_manifest):packet_sha}
            raw_plan=root/'full-plan.json';query.dump(raw_plan,full);raw_sha=query.sha(raw_plan)
            rows=query.project_rows(full);text=query.runtime_text({'query':'Browse the store'});text_sha=query.digest_text(text)
            for row in rows:row.update(queryTextSha256=text_sha,publicRequestSha256=query.digest_text(query.canonical(row['publicRequest'])))
            qplan={'version':query.VERSION,'recipe':query.RECIPE,'sha256':{str(raw_plan):raw_sha},'rows':rows,'uniqueQueries':[{'textSha256':text_sha}]}
            query.dump(qrun/'plan.json',qplan);qsha=query.sha(qrun/'plan.json')
            torch.save({'version':query.VERSION,'sourceManifestSha256':qsha,'features':{text_sha:torch.eye(1,4096)}},qrun/'query-features.pt')
            hashes={str(i):'a'*64 for i in range(927)}
            query.dump(qrun/'report.json',{'version':query.VERSION,'sourceManifestSha256':qsha,'status':'completed-query-features',
                'adapterInstalled':False,'goldParsed':False,'optimizerUpdates':0,'attemptedNativeForwards':1,'physicalNativeForwards':1,
                'savedVectors':1,'expectedNativeForwards':1,'frozen927Unchanged':True,'sourcePinsUnchanged':True,'shutdownCompleted':True,
                'beforeParameterSha256':hashes,'afterParameterSha256':hashes,'queryFeaturesSha256':query.sha(qrun/'query-features.pt')})
            adapter_sha='c'*64
            query.dump(raw/'report.json',{'status':'completed-fixed-fit-and-predictions','planSha256':raw_sha,'actualUpdates':3584,
                'privateEvaluationLabelsParsed':False,'frozen927HashesUnchanged':True,'shutdownCompleted':True,'sourcePinsUnchanged':True,
                'fit':{'checkpointSha256':adapter_sha}})
            for name in ('serving.json','adapter.safetensors','index.json','inference-parity.json','BASELINE_DEFINITIONS_FROZEN.json'):(raw/name).write_text('{}')
            records=[{k:r[k] for k in ('profileId','userGroupId','caseId','split','orderLabel','sourceProfileId','controlName')}
                |{'scores16':[i/97 for i in range(16)]} for r in rows]
            for filename in combined.MODEL_FILES.values():query.dump(raw/filename,records)
            query.dump(raw/'ALL_PREDICTIONS_FROZEN.json',{'checkpointSha256':adapter_sha,'privateEvaluationLabelsParsed':False,'nonfitPromptRecords':344,
                'learnedPredictionsSha256':query.sha(raw/'learned-nonfit-predictions.json'),
                'frozenBasePredictionsSha256':query.sha(raw/'frozen-base-nonfit-predictions.json'),
                'baselineFilesSha256':{name:query.sha(raw/(name+'-nonfit-predictions.json')) for name in ('training-popularity','latest-owned-repeat')}})
            serving={'adapter':{'sha256':adapter_sha,'file':'adapter.safetensors'},'publicCache':{'file':'index.json'},
                'inferenceParity':{'file':'inference-parity.json'},'nativeReportSha256':query.sha(raw/'report.json')}
            v5=root/'v5.json';v5.write_text('{}');(root/'learned.pt').write_bytes(b'CPU fake head');gate=root/'gate.pt';gate.write_bytes(b'CPU fake gate')
            vectors=torch.eye(16,4096)
            class Head:
                catalog_mean=vectors.double().mean(0).float()
                def eval(self):return self
                def requires_grad_(self,value):return self
                def probabilities(self,role,value):return torch.eye(16) if role=='catalog' else torch.ones(1,16)/16
            export={'featureProvenance':{},'head':{'file':'learned.pt','sha256':query.sha(root/'learned.pt'),'metadata':{}}}
            with patch.object(combined.runtime,'load_manifest',return_value=(serving,{})), \
                 patch.object(combined.context,'load_catalog_features',return_value={'products':self.products,'vectors':vectors,'pins':{},'provenance':{}}), \
                 patch.object(combined,'read_export',return_value=export),patch.object(combined,'load_head',return_value=Head()), \
                 patch.object(combined.gate,'load_intent_head',return_value=(object(),{'sourceManifestSha256':'train-only'})), \
                 patch.object(combined.gate,'specific_probability',return_value=.4):
                result=combined.write_predictions(raw,raw_plan,raw_sha,qrun,qsha,out,gate_checkpoint=gate,gate_sha=query.sha(gate),
                    v5_export=v5,v5_export_sha=query.sha(v5),catalog_feature_root=root,independent_packet_manifest=packet_manifest,
                    independent_packet_manifest_sha=packet_sha)
            self.assertEqual(result['rowsPerModel'],344);self.assertFalse(result['goldParsed'])
            freeze=json.loads((out/'ALL_MODELS_PREDICTIONS_FROZEN.json').read_text())
            self.assertEqual(set(freeze['models']),set(combined.MODEL_FILES));self.assertEqual(freeze['stagesCompleted'],{'adapter':True,'queryGate':True,'allControls':True})
            self.assertFalse(freeze['specific320LoRARegressionComputed'])
            for metadata in freeze['models'].values():
                self.assertEqual(query.sha(out/metadata['file']),metadata['sha256'])
                self.assertEqual(len(json.loads((out/metadata['file']).read_text())),344)
            self.assertEqual(len(freeze['caseIds']),344)


if __name__=='__main__':unittest.main()
