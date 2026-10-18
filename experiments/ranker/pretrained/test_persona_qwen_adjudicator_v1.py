"""CPU evidence tests using fabricated fixtures, never the actual fresh gold."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import audit_persona_qwen_candidate_v1 as audit


def profile(i):
    return {'profileId':f'p{i}', 'userGroupId':f'g{i//2}',
            'request':{'query':'Browse the store','itemBudget':None,'hardRequirements':[],
                       'persona':{'background':[],'colorPreferences':[]},'publicHistory':[]}}


def prediction(p, first=0):
    scores = [float(-i) for i in range(16)]; scores[first] = 20.
    order = sorted(range(16),key=lambda i:(-scores[i],i))
    return {'profileId':p['profileId'],'userGroupId':p['userGroupId'],'scores16':scores,
            'rawLearnedOrder':order,'order':order,
            'selected':[{'productId':f'product{i}','variantId':f'variant{i}',
                         'selectedColor':'Navy','image':f'/image{i}.webp'} for i in order[:6]]}


def catalog():
    products = [{'productId':f'product{i}','name':f'Public item{i}','category':'accessories',
                 'itemPrice':20.,'colorVariants':[{'colorName':'Navy'}],
                 'variants':[{'variantId':f'variant{i}','colorName':'Navy','images':[f'/image{i}.webp']}]} for i in range(16)]
    return {'products':products,'variants':[{**p['variants'][0],'productIndex':i} for i,p in enumerate(products)]}


def gold(i):
    grades = {f'product{j}':0 for j in range(16)}; grades['product0']=3; grades['product1']=2
    return {'profileId':f'p{i}','pairId':f'pair{i//2}','primaryProductId':'product0',
            'alreadyOwnedProductIds':['product15'],'relevanceGrades':grades}


def dump(path,value):
    Path(path).write_text(json.dumps(value))


class EvidenceTests(unittest.TestCase):
    def test_json_duplicate_and_nonfinite_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json'
            for raw in ('{"key":1,"key":2}','{"key":NaN}','{"key":Infinity}'):
                p.write_text(raw)
                with self.assertRaises(audit.AuditError): audit.read(p)

    def test_fixed_roster_scores_cards_and_group_checks(self):
        profiles=[profile(0),profile(1)]; rows=[prediction(p) for p in profiles]
        audit.exact_rows(rows,profiles,'fake')
        mutations=[lambda r:r.pop(), lambda r:r.append(deepcopy(r[0])),
                   lambda r:r[1].update(profileId='other'),lambda r:r[0].update(userGroupId='wrong'),
                   lambda r:r[0]['scores16'].__setitem__(0,True),
                   lambda r:r[0]['order'].__setitem__(1,0),lambda r:r[0]['selected'].pop(),
                   lambda r:r[0].update(rawLearnedOrder=list(reversed(r[0]['order'])))]
        for mutate in mutations:
            changed=deepcopy(rows);mutate(changed)
            with self.assertRaises(audit.AuditError): audit.exact_rows(changed,profiles,'fake')

    def make_freeze(self, root):
        packet={'rows':[]}; files={}; baselines={}
        (root/'head.pt').write_bytes(b'fake checkpoint identity only')
        for split,n in audit.PRIMARY_SPLITS.items():
            roster=[]
            for i in range(n):
                p=profile(i); p['profileId']=split+str(i); roster.append(p)
                packet['rows'].append({'split':split,'publicProfile':p})
            path=root/(split+'-predictions.json');dump(path,[prediction(p) for p in roster]);files[split]=audit.sha(path)
        independent=[r['publicProfile'] for r in packet['rows'] if r['split']=='independent-evaluation']
        for name in audit.BASELINES:
            path=root/('independent-'+name+'-predictions.json');dump(path,[prediction(p) for p in independent]);baselines[name]=audit.sha(path)
        value={'checkpointSha256':audit.sha(root/'head.pt'),'files':files,'baselineFiles':baselines,'independentGoldParsed':False}
        dump(root/'PREDICTIONS_FROZEN.json',value)
        return packet,value

    def test_admission_never_touches_a_private_label(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);packet,_=self.make_freeze(root)
            opened=[];original=audit.read
            def tracking(path):
                opened.append(Path(path).name)
                self.assertNotIn('labels',Path(path).name)
                return original(path)
            with patch.object(audit,'read',side_effect=tracking):
                _,outputs,baselines=audit.require_prediction_freeze(root,packet)
            self.assertEqual(sum(map(len,outputs.values())),112)
            self.assertEqual(sum(map(len,baselines.values())),160)
            self.assertEqual(len(opened),11)

    def test_admission_rejects_missing_baseline_early_gold_and_tampering(self):
        for mode in ('baseline','early','checkpoint','prediction','denominator'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d);packet,value=self.make_freeze(root)
                if mode=='baseline':value['baselineFiles'].pop(next(iter(audit.BASELINES)))
                if mode=='early':value['independentGoldParsed']=True
                if mode=='checkpoint':(root/'head.pt').write_bytes(b'tampered')
                if mode=='prediction':(root/'independent-evaluation-predictions.json').write_text('[]')
                if mode=='denominator':packet['rows'].pop()
                dump(root/'PREDICTIONS_FROZEN.json',value)
                with self.assertRaises(audit.AuditError): audit.require_prediction_freeze(root,packet)

    def test_independent_counts_fixed32_strong_not_merely_positive(self):
        profiles=[profile(i) for i in range(32)];rows=[prediction(p) for p in profiles];labels=[gold(i) for i in range(32)]
        result=audit.metrics(rows,labels,catalog()['products'],profiles)
        self.assertEqual((result['fixedDenominator'],result['primaryTop1'],result['novelRelevantTop1'],result['strongPositiveTop3'],result['bothPairSidesPrimary']),(32,32,32,32,16))
        for row in rows:
            row['order']=[2,3,4,0,1,5]+list(range(6,16));row['selected']=[{'productId':f'product{i}'} for i in row['order'][:6]]
        for label in labels:label['relevanceGrades']['product2']=1
        changed=audit.metrics(rows,labels,catalog()['products'],profiles)
        self.assertEqual((changed['primaryTop1'],changed['novelRelevantTop1'],changed['strongPositiveTop3']),(0,0,0))
        with self.assertRaises(audit.AuditError):audit.metrics(rows[:-1],labels,catalog()['products'],profiles)
        bad=deepcopy(labels);bad[-1]=deepcopy(bad[0])
        with self.assertRaises(audit.AuditError):audit.metrics(rows,bad,catalog()['products'],profiles)

    def test_development_demotion_is_rank_change_not_logit_change(self):
        public={'pairs':[]};labels=[];rows=[]
        for i in range(10):
            a=profile(i*2);b=profile(i*2+1);pair=f'pair{i}'
            a['userGroupId']=b['userGroupId']=pair
            public['pairs'].append({'pairId':pair,'profileA':a,'profileB':b})
            rows.extend([prediction(a),prediction(b)])
            rows[-1]['scores16']=[x-100 for x in rows[-1]['scores16']]
            labels.append({'pairId':pair,'changedPurchaseProductId':'product0',
                           'profileB':{'alreadyOwnedProductIds':['product0'],
                                       'relevanceGrades':{f'product{j}':2 if j==1 else 0 for j in range(16)}}})
        m=audit.history_metrics(rows,public,labels,catalog()['products'])
        self.assertEqual((m['ownedPriorityDemotions'],m['novelRelevantAfterTop1']),(0,0))
        for i in range(10):rows[i*2+1]=prediction(public['pairs'][i]['profileB'],first=1)
        m=audit.history_metrics(rows,public,labels,catalog()['products'])
        self.assertEqual((m['ownedPriorityDemotions'],m['novelRelevantAfterTop1']),(10,10))

    def test_budget_and_source_binding_checks(self):
        p=profile(0);r=prediction(p);c=catalog()
        p['request']['itemBudget']=90;p['request']['query']='Find something under $20'
        self.assertEqual(audit.effective_budget(p['request']),(20.,True))
        self.assertEqual(audit.factual_checks([r],[p],c)['violationCount'],16)
        p['request']['query']='Find something up to $20'
        self.assertEqual(audit.factual_checks([r],[p],c)['violationCount'],0)
        r['selected'][0]['image']='/other.webp'
        self.assertEqual(audit.factual_checks([r],[p],c)['violations'][0]['reason'],'canonical-image-binding')
        r=prediction(p);p['request']['hardRequirements']=[{'attribute':'color','value':'Sage'}]
        reasons={v['reason'] for v in audit.factual_checks([r],[p],c)['violations']}
        self.assertEqual(reasons,{'unsupported-or-unsatisfied-hard-color','variant-hard-color'})

    def test_seeded_state_change_and_shapes(self):
        import torch
        from torch import nn
        cfg={'seed':99,'hiddenWidth':64,'colorWidth':32,'colorDimensions':256}
        with torch.random.fork_rng():
            torch.manual_seed(99);c=nn.Linear(4096,64);p=nn.Linear(4096,64,bias=False);cc=nn.Linear(4096,32);cv=nn.Linear(256,32,bias=False)
        state={'scale':torch.tensor(2.),'ownership':torch.tensor(-.5),'context.weight':c.weight.detach(),'context.bias':c.bias.detach(),'product.weight':p.weight.detach(),'color_context.weight':cc.weight.detach(),'color_context.bias':cc.bias.detach(),'color_candidate.weight':cv.weight.detach()}
        report={'changedTensors':['ownership'],'ownershipCoefficient':-.5}
        self.assertEqual(audit.verify_state(state,report,cfg),['ownership'])
        bad=dict(state);bad['product.weight']=torch.zeros((63,4096))
        with self.assertRaises(audit.AuditError):audit.verify_state(bad,report,cfg)
        with self.assertRaises(audit.AuditError):audit.verify_state(state,{**report,'changedTensors':['ownership','scale']},cfg)

    def test_simple_baselines_use_only_train_gains_and_public_latest_purchase(self):
        products=catalog()['products'];train=[profile(i) for i in range(320)];labels=[]
        for i in range(320):labels.append({'profileId':f'p{i}','relevanceGrades':{f'product{j}':3 if j==i%16 else 0 for j in range(16)}})
        independent=[profile(1000),profile(1001)]
        independent[0]['request']['publicHistory']=[{'eventType':'purchase','productId':'product2','occurredAt':'2026-01-01T00:00:00Z'}, {'eventType':'purchase','productId':'product3','occurredAt':'2026-02-01T00:00:00Z'}, {'eventType':'view','productId':'product4','occurredAt':'2026-03-01T00:00:00Z'}]
        scores=audit.simple_baseline_scores(labels,train,products,independent)
        self.assertEqual(scores['training-popularity'],[[.0625]*16]*2)
        self.assertEqual(scores['most-recent-owned-repeat'][0][3],10.0625)
        self.assertEqual(scores['most-recent-owned-repeat'][0][4],.0625)
        with self.assertRaises(audit.AuditError):audit.simple_baseline_scores(labels[:-1],train,products,independent)

    def test_independent_score_formula_ownership_is_additive_not_mask(self):
        import torch
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(11)
            c=catalog();c['vectors']=torch.nn.functional.normalize(torch.randn(16,4096),dim=1)
            x=torch.nn.functional.normalize(torch.randn(2,4096),dim=1)
            state={'scale':torch.tensor(2.),'ownership':torch.tensor(-.5),
                   'context.weight':torch.randn(64,4096)*.01,'context.bias':torch.randn(64)*.01,
                   'product.weight':torch.randn(64,4096)*.01,
                   'color_context.weight':torch.randn(32,4096)*.01,'color_context.bias':torch.randn(32)*.01,
                   'color_candidate.weight':torch.randn(32,256)*.01}
        profiles=[profile(0),profile(1)]
        profiles[0]['request']['publicHistory']=[{'eventType':'purchase','productId':'product2'}]
        scores,variants=audit.replay_scores(state,x,c,profiles)
        plain,_=audit.replay_scores(state,x,c,profiles,owned_override=torch.zeros(2,16))
        expected=plain.clone();expected[0,2]-=.5
        self.assertTrue(torch.equal(scores,expected));self.assertTrue(torch.isfinite(variants).all())
        self.assertTrue(torch.equal(scores[1],plain[1]))

    def test_public_cache_exact_content_and_provenance(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);report=root/'report.json';dump(report,{'fixture':'not native evidence'})
            row={'caseId':'p0','split':'demo-nonfit','textSha256':'publictext','tokenIds':[1,2]}
            packet={'featureProvenance':{'fixture':True},'rows':[row]};features={'p0':torch.zeros(1,4096)}
            cache={'version':'public-persona-native-context-cache.v1','featureProvenance':packet['featureProvenance'],
                   'contextFormatterSha256':audit.sha(audit.HERE/'store_persona_qwen_v1.py'),'teacherOrLabels':False,
                   'sourceExtractionReportSha256':audit.sha(report),
                   'features':{'publictext':{'tensor':features['p0'],'tokenIds':[1,2],'textSha256':'publictext','split':'demo-nonfit'}}}
            path=root/'public-demo-features.pt';torch.save(cache,path)
            checkpoint={'publicDemoFeatureCacheFile':str(path),'publicDemoFeatureCacheSha256':audit.sha(path)}
            self.assertEqual(audit.verify_public_demo_cache(checkpoint,root,packet,features,report),1)
            cache['features']['publictext']['tokenIds']=[2,1];torch.save(cache,path)
            checkpoint['publicDemoFeatureCacheSha256']=audit.sha(path)
            with self.assertRaises(audit.AuditError):audit.verify_public_demo_cache(checkpoint,root,packet,features,report)
            cache['teacherOrLabels']=True;torch.save(cache,path);checkpoint['publicDemoFeatureCacheSha256']=audit.sha(path)
            with self.assertRaises(audit.AuditError):audit.verify_public_demo_cache(checkpoint,root,packet,features,report)


if __name__=='__main__':unittest.main()
