"""Fake-only clear-objective admission regressions; no future quality labels."""
from copy import deepcopy
import tempfile
import hashlib
import struct
from pathlib import Path
import unittest
from unittest.mock import patch
from . import audit_persona_choice_clear_combined_v2 as audit
from . import test_persona_choice_combined_audit_v1 as original
from . import test_persona_choice_combined_audit_v2 as derived

class ClearAuditTests(unittest.TestCase):
    def schedule_fixture(self):
        recipe={'seed':19083,'rank':4,'scale':20,'learningRate':.0001,'epochs':14,'updatesExpected':3584,'teacherWeight':.25,'teacherTemperature':1,'oldAuthoredPriorityLabelsInOptimizer':False}
        rows=[]
        for name,n,purpose in [('clear-CE',256,'audited-clear128-maximum-grade-TRAIN-supervision'),('frozen-base-teacher',640,'frozen-B0-legacy-soft-target-distillation-no-authored-priority')]:
            for i in range(n):rows.append({'caseId':name+str(i),'profileId':name+str(i//2),'objective':name,'targetPurpose':purpose})
        ce=audit.expected_fixed_orders(256,14,19083);teacher=audit.expected_fixed_orders(640,6,20083)[:3584]
        plan={'recipe':recipe,'ceOrder':ce,'teacherOrder':teacher}
        schedule={'recipe':recipe,'legacyAuthoredTargetsUsed':False,'optimizerUpdates':0,'ceCases':[r['caseId'] for r in rows[:256]],'teacherCases':[r['caseId'] for r in rows[256:]],'ceOrder':ce,'teacherOrder':teacher}
        return plan,schedule,rows
    def test_exact_schedule_then_changed_join_or_order_rejected(self):
        p,s,r=self.schedule_fixture();audit.verify_clear_schedule(p,s,r)
        broken=deepcopy(s);broken['teacherOrder'][0]=(broken['teacherOrder'][0]+1)%640
        with self.assertRaisesRegex(audit.AuditError,'schedule'):audit.verify_clear_schedule(p,broken,r)
        broken=deepcopy(s);broken['ceCases'][0]='wrong'
        with self.assertRaisesRegex(audit.AuditError,'joins'):audit.verify_clear_schedule(p,broken,r)
    def test_authored_legacy_targets_or_weight_change_rejected(self):
        p,s,r=self.schedule_fixture();p['recipe']['teacherWeight']=1
        with self.assertRaisesRegex(audit.AuditError,'recipe'):audit.verify_clear_schedule(p,s,r)
        p,s,r=self.schedule_fixture();s['legacyAuthoredTargetsUsed']=True
        with self.assertRaisesRegex(audit.AuditError,'pre-update'):audit.verify_clear_schedule(p,s,r)
    def test_historical976_not_miscounted_as1240_fresh(self):
        extraction={'successfulExactGateSuffixForwards':2480,'reusedOriginalBasisRows':976,'freshPublicNumericalRows':264}
        with self.assertRaisesRegex(audit.AuditError,'historical basis'):audit.verify_reuse_and_joint_controls('fake',{}, {},extraction)
    def test_full_fake_reuse_roster_and_zero_nonzero_receipts(self):
        old={};entries=[]
        for i in range(976):
            r={'caseId':'old'+str(i),'frozenBaseRouteScores':[0.]*16}
            for k in ('archive','metadata','gate'):r[k+'Sha256']='fake-'+k
            old[r['caseId']]=deepcopy(r);entries.append({**r,'newPerTargetGateClaim':False,'originalBasisCaseId':r['caseId'],'originalBasisRun':'original'})
        entries.extend({'caseId':'fresh'+str(i),'newPerTargetGateClaim':True,'split':'evaluation-main'} for i in range(264))
        rows=[{'caseId':'c'+str(i),'objective':'clear-CE'} for i in range(2)]+[{'caseId':'t'+str(i),'objective':'frozen-base-teacher'} for i in range(2)]
        plan={'reuseEntries':old,'reuseFileSha256':{str(i):'fake' for i in range(2928)},'originalRun':'original','rows':rows}
        extraction={'rows':entries,'successfulExactGateSuffixForwards':528,'reusedOriginalBasisRows':976,'freshPublicNumericalRows':264}
        records=[]
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for i in range(2):
                for mode in ('zero','nonzero'):
                    b=struct.pack('<f',0. if mode=='zero' else 1e-5)*(4*4096)
                    parameter={'lora_b':{'shape':[4,4096],'dtype':'mlx.core.float32','sha256':hashlib.sha256(b'mlx.core.float32'+str((4,4096)).encode()+b).hexdigest()}}
                    g={'status':'exact-paired-objective-gate-passed','exact':True,'pairIndex':i,'adapterControl':mode,'clearCaseId':'c'+str(i),'teacherCaseId':'t'+str(i),'optimizerUpdates':0,'successfulNativeSuffixForwards':2,'native':{'gradients':{'lora_a':'fake','lora_b':'fake'}},'features':{'gradients':{'lora_a':'fake','lora_b':'fake'}},'sourceStatePreserved':True,'adapterPreserved':True,'sourceStateHashesBefore':['fake'],'sourceStateHashesAfter':['fake'],'adapterBefore':parameter,'adapterAfter':parameter}
                    records.append(g);audit.write(root/f'INITIAL_PAIRED_GATE_{i}_{mode}.json',g)
            report={'successfulExactGateSuffixForwards':544,'successfulInferenceControlSuffixForwards':8,'originalReuseSourceBytesUnchanged':True,'initialPairedObjectiveControls':records}
            audit.verify_reuse_and_joint_controls(root,plan,report,extraction)
            records[0]['adapterBefore']['lora_b']['sha256']='wrong'
            path=root/'INITIAL_PAIRED_GATE_0_zero.json';path.unlink();audit.write(path,records[0])
            with self.assertRaisesRegex(audit.AuditError,'B control'):audit.verify_reuse_and_joint_controls(root,plan,report,extraction)
    def test_original16_fake_denominator_admission_and_fusion_cases(self):
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(original.CombinedAuditTests);result=unittest.TestResult()
        with patch.object(original,'audit',audit):suite.run(result)
        self.assertEqual(result.testsRun,16);self.assertFalse(result.errors,result.errors);self.assertFalse(result.failures,result.failures)
    def test_derived_identity_preserves_raw_rows(self):
        products,profiles,labels,rows=original.quality_fixture();reverse=[]
        for r in rows:reverse.append({**deepcopy(r),'orderLabel':'reversed','caseId':r['profileId']+'-reversed'})
        saved=deepcopy(reverse);result=audit.metrics(audit.choose(reverse,profiles,'reversed'),profiles,labels,products,32)
        self.assertEqual(result['primaryTop1'],32);self.assertEqual(reverse,saved)
    def test_264_combined_admission_refused_before_private_quality_parse(self):
        packet=audit.HERE/'persona_choice_independent_eval_v4'
        profiles=[]
        for filename in ('main.public.json','replacement.public.json','controls.public.json'):profiles+=audit.read(packet/filename)['profiles']
        planrows=[{'split':'evaluation-main','profileId':p['profileId'],'publicRequest':p['request'],'userGroupId':p['userGroupId'],'controlName':p.get('controlName'),'sourceProfileId':p.get('sourceProfileId')} for p in profiles for order in ('canonical','reversed')]
        adapter={'plan':{'rows':planrows}};opened=[];real_read=audit.read
        def safe_read(path):
            opened.append(Path(path).name)
            if Path(path).name=='PERSONA_CHOICE_CLEAR_ADJUDICATOR_FROZEN_V2.json':return {'sha256':{'fake':'fake'}}
            if 'labels.private' in Path(path).name:raise AssertionError('private labels opened before344 history admission')
            return real_read(path)
        with tempfile.TemporaryDirectory() as name,patch.object(audit,'read',side_effect=safe_read),patch.object(audit,'pins'),patch.object(audit,'verify_adapter',return_value=adapter),patch.object(audit,'verify_combined',return_value={'models':{m:[{}]*264 for m in audit.MODELS}}):
            result=audit.run('fake-adapter','fake-combined',packet,Path(name)/'out')
        self.assertEqual(result['status'],'failed-admission-or-audit');self.assertIn('all344',result['error']['message']);self.assertFalse(result['privateQualityJSONParsed']);self.assertFalse(any('labels.private' in n for n in opened))
    def test_public_packet_controls_and_manifest_are_compatible_without_private_parse(self):
        packet=audit.HERE/'persona_choice_independent_eval_v4';manifest=audit.read(packet/'MANIFEST.json')
        main=audit.read(packet/'main.public.json')['profiles'];controls=audit.read(packet/'controls.public.json')['profiles']
        audit.public_controls(main,controls,manifest);self.assertEqual(audit.sha(packet/'MANIFEST.json'),audit.PACKET_SHA)

if __name__=='__main__':unittest.main()
