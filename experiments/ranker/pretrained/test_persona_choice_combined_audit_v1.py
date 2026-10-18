"""CPU fake evidence only. No production quality gold or native model access."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import audit_persona_choice_combined_v1 as audit


def quality_fixture():
    products = [{'productId':str(i),'itemPrice':i+1,'category':'test',
                 'colorVariants':[{'colorName':'Navy'}]} for i in range(16)]
    profiles=[]; labels=[]; predictions=[]
    for i in range(32):
        target=i//2; owned=(target+1)%16; alternate=(target+2)%16
        pid='p'+str(i); group='g'+str(i//2)
        request={'query':'Help me choose','itemBudget':100,'hardRequirements':[],
                 'publicHistory':[{'eventType':'purchase','productId':str(owned)}]}
        profiles.append({'profileId':pid,'userGroupId':group,'request':request})
        labels.append({'profileId':pid,'primaryProductId':str(target),
                       'relevanceGrades':{str(j):3 if j==target else 2 if j==alternate else 0 for j in range(16)},
                       'alreadyOwnedProductIds':[str(owned)]})
        scores=[float(j==target)*10+float(j==alternate) for j in range(16)]
        row={'profileId':pid,'userGroupId':group,'orderLabel':'canonical','scores16':scores}
        row['rawLearnedOrder']=audit.raw_order(row); row['order']=row['rawLearnedOrder']
        predictions.append(row)
    return products,profiles,labels,predictions


class CombinedAuditTests(unittest.TestCase):
    def test_exact32_and_whole16_groups(self):
        products,profiles,labels,rows=quality_fixture()
        result=audit.metrics(rows,profiles,labels,products,32)
        self.assertEqual((result['fixedDenominator'],result['groups'],result['primaryTop1'],
                          result['novelRelevantTop1'],result['strongPositiveTop3']),(32,16,32,32,32))
        with self.assertRaisesRegex(audit.AuditError,'denominator'):
            audit.metrics(rows[:-1],profiles,labels,products,32)

    def test_wrong_label_and_group_join_fail(self):
        products,profiles,labels,rows=quality_fixture()
        labels[0]['profileId']='missing'
        with self.assertRaisesRegex(audit.AuditError,'label join'):
            audit.metrics(rows,profiles,labels,products,32)
        labels[0]['profileId']='p0'; rows[0]['userGroupId']='wrong'
        with self.assertRaisesRegex(audit.AuditError,'group join'):
            audit.metrics(rows,profiles,labels,products,32)

    def test_duplicate_prediction_and_bool_scores_fail(self):
        _,profiles,_,rows=quality_fixture()
        rows[1]=deepcopy(rows[0])
        with self.assertRaisesRegex(audit.AuditError,'duplicate'):
            audit.index_rows(rows,profiles,both_orders=False)
        _,profiles,_,rows=quality_fixture(); rows[0]['scores16'][0]=True
        with self.assertRaisesRegex(audit.AuditError,'finite16'):
            audit.index_rows(rows,profiles,both_orders=False)

    def test_empty_shortfall_counts_failure_without_shrinking_denominator(self):
        products,profiles,labels,rows=quality_fixture(); rows[0]['order']=[]
        result=audit.metrics(rows,profiles,labels,products,32)
        self.assertEqual(result['primaryTop1'],31)
        self.assertEqual(result['fixedDenominator'],32)

    def test_owned_label_cannot_be_positive_in_core(self):
        products,profiles,labels,rows=quality_fixture(); labels[0]['alreadyOwnedProductIds']=[]
        with self.assertRaisesRegex(audit.AuditError,'ownership'):
            audit.metrics(rows,profiles,labels,products,32)

    def test_source_budget_and_typed_color_checked_beyond_top1(self):
        products,profiles,_,rows=quality_fixture()
        profiles[0]['request']['itemBudget']=15
        self.assertEqual(audit.facts(rows[:1],profiles,products)['violationCount'],1)
        profiles[0]['request']['hardRequirements']=[{'attribute':'color','value':'Navy'},{'attribute':'color','value':'Rust'}]
        self.assertGreater(audit.facts(rows[:1],profiles,products)['violationCount'],0)

    def test_typed_budget_cannot_hide_stricter_query_budget(self):
        self.assertEqual(audit.effective_constraints({'query':'Please stay under $20', 'itemBudget':90})[0],(20,True))

    def test_replacement_can_prefer_observed_owned_item(self):
        products,profiles,labels,rows=quality_fixture()
        profiles=profiles[:4];labels=labels[:4];rows=rows[:4]
        for i,(p,label,row) in enumerate(zip(profiles,labels,rows,strict=True)):
            p['userGroupId']=row['userGroupId']='replacement'+str(i)
            p['request']['publicHistory']=[{'eventType':'purchase','productId':label['primaryProductId']}]
            label['alreadyOwnedProductIds']=[label['primaryProductId']]
        result=audit.metrics(rows,profiles,labels,products,4,replacement=True)
        self.assertEqual((result['primaryTop1'],result['ownedTop1']),(4,4))

    def test_nonfinite_duplicate_json_and_artifact_escape_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.json';path.write_text('{"x":1,"x":2}')
            with self.assertRaises(audit.AuditError):audit.read(path)
            path.write_text('{"x":NaN}')
            with self.assertRaises(audit.AuditError):audit.read(path)
            with self.assertRaisesRegex(audit.AuditError,'escapes'):
                audit.local_file(temp,'../outside','x')

    def test_independent_fusion_endpoints_and_invalid_probability(self):
        source=[float(i) for i in range(16)]; v5=[.01*i for i in range(16)]
        self.assertEqual(audit.fuse(source,0,v5),audit.softmax(source))
        self.assertEqual(audit.fuse(source,1,v5),[b+.05*a for a,b in zip(audit.softmax(source),v5)])
        with self.assertRaises(audit.AuditError):audit.fuse(source,1.01,v5)

    def combined_fixture(self,temp):
        directory=Path(temp); head=directory/'head.fake';head.write_bytes(b'explicit fake CPU test head')
        profiles=[{'profileId':str(i),'userGroupId':'g'+str(i),'request':{}} for i in range(132)]
        raw=[]
        for p in profiles:
            for order in ('canonical','reversed'):
                scores=[float(i)/10 for i in range(16)]
                raw.append({**p,'orderLabel':order,'scores16':scores,'rawLearnedOrder':list(range(15,-1,-1)),'order':list(range(15,-1,-1))})
        adapter={'checkpointSha256':'adapterfake','profiles':profiles,'models':{name:deepcopy(raw) for name in audit.MODELS}}
        freeze={'goldParsed':False,'independentPacketManifestSha256':audit.PACKET_SHA,
                'adapterCheckpointSha256':'adapterfake','gateCheckpointSha256':audit.sha(head),
                'stagesCompleted':{'adapter':True,'queryGate':True,'allControls':True},
                'models':{},'sha256':{str(head):audit.sha(head)}}
        for name in audit.MODELS:
            rows=deepcopy(raw)
            for row in rows:
                row.update(rawSourceScores16=row['scores16'],pSpecific=.5,v5Scores16=[.01*i for i in range(16)],
                           fusionPolicy={'gateCheckpointSha256':audit.sha(head)})
                row['scores16']=audit.fuse(row['rawSourceScores16'],.5,row['v5Scores16'])
            path=directory/(name+'.json');audit.write(path,rows)
            freeze['models'][name]={'file':path.name,'sha256':audit.sha(path)}
        audit.write(directory/'ALL_MODELS_PREDICTIONS_FROZEN.json',freeze)
        return adapter,profiles,freeze

    def test_all264_fused_controls_verify(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter,profiles,_=self.combined_fixture(temp)
            result=audit.verify_combined(temp,adapter,profiles)
            self.assertEqual(set(result['models']),audit.MODELS)

    def test_missing_model_stage_refused_before_artifact_load(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter,profiles,freeze=self.combined_fixture(temp)
            freeze['stagesCompleted']['queryGate']=False
            (Path(temp)/'ALL_MODELS_PREDICTIONS_FROZEN.json').write_text(json.dumps(freeze))
            with self.assertRaisesRegex(audit.AuditError,'all models'):
                audit.verify_combined(temp,adapter,profiles)

    def test_rehashed_wrong_fusion_still_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter,profiles,freeze=self.combined_fixture(temp)
            item=freeze['models']['TRAIN-popularity'];path=Path(temp)/item['file'];rows=audit.read(path)
            rows[0]['scores16'][0]+=.1;rows[0]['rawLearnedOrder']=audit.raw_order(rows[0])
            path.write_text(json.dumps(rows));item['sha256']=audit.sha(path)
            (Path(temp)/'ALL_MODELS_PREDICTIONS_FROZEN.json').write_text(json.dumps(freeze))
            with self.assertRaisesRegex(audit.AuditError,'fusion arithmetic'):
                audit.verify_combined(temp,adapter,profiles)

    def test_rehashed_different_v5_control_fails_even_if_formula_correct(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter,profiles,freeze=self.combined_fixture(temp)
            item=freeze['models']['latest-owned-repeat'];path=Path(temp)/item['file'];rows=audit.read(path)
            rows[0]['v5Scores16'][0]+=.1
            rows[0]['scores16']=audit.fuse(rows[0]['rawSourceScores16'],rows[0]['pSpecific'],rows[0]['v5Scores16'])
            rows[0]['rawLearnedOrder']=audit.raw_order(rows[0]);path.write_text(json.dumps(rows));item['sha256']=audit.sha(path)
            (Path(temp)/'ALL_MODELS_PREDICTIONS_FROZEN.json').write_text(json.dumps(freeze))
            with self.assertRaisesRegex(audit.AuditError,'different gate/v5'):
                audit.verify_combined(temp,adapter,profiles)

    def test_admission_failure_never_opens_private_quality_json(self):
        original_read=audit.read; opened=[]
        def guarded_read(path):
            name=Path(path).name;opened.append(name)
            if name == 'PERSONA_CHOICE_COMBINED_ADJUDICATOR_FROZEN_V1.json':return {'sha256':{'fake':'fake'}}
            if 'labels' in name:raise AssertionError('private quality JSON opened before barrier')
            return original_read(path)
        with tempfile.TemporaryDirectory() as temp, patch.object(audit,'read',side_effect=guarded_read), patch.object(audit,'pins'), patch.object(audit,'verify_adapter',side_effect=audit.AuditError('fake missing checkpoint')):
            result=audit.run('fake-adapter','fake-combined',audit.HERE/'persona_choice_independent_eval_v3',Path(temp)/'audit')
        self.assertEqual(result['status'],'failed-admission-or-audit')
        self.assertFalse(result['privateQualityJSONParsed'])
        self.assertFalse(any('labels' in p for p in opened))

    def test_known_history_exact10_pairs_raw_demotion_and_novelty(self):
        products=[{'productId':str(i)} for i in range(16)]; pairs=[]; labels=[];rows=[]
        for i in range(10):
            pid='pair'+str(i);pair={'pairId':pid}
            for side in ('A','B'):
                profile={'profileId':pid+side,'request':{}}
                pair['profile'+side]=profile
                scores=([10.]+[0.]*15) if side=='A' else ([-10.,10.]+[0.]*14)
                for order in ('canonical','reversed'):
                    row={'profileId':profile['profileId'],'userGroupId':pid,'orderLabel':order,'scores16':scores}
                    row['order']=row['rawLearnedOrder']=audit.raw_order(row);rows.append(row)
            pairs.append(pair);labels.append({'pairId':pid,'changedPurchaseProductId':'0',
                'profileB':{'alreadyOwnedProductIds':['0'],'relevanceGrades':{str(j):3 if j==1 else 0 for j in range(16)}}})
        result=audit.history_metrics(rows,{'pairs':pairs},labels,products)
        self.assertEqual((result['rawOwnedPriorityDemotions'],result['novelRelevantAfterTop1']),(10,10))
        self.assertTrue(result['developmentOnly'])
        with self.assertRaises(audit.AuditError):audit.history_metrics(rows[:-1],{'pairs':pairs},labels,products)


if __name__=='__main__':
    unittest.main()
