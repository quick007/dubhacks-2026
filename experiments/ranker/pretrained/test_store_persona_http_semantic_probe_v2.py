from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from . import store_persona_http_semantic_probe_v2 as probe
from . import store_persona_adapter_runtime_v2 as runtime
from .store_demo_engine import guarded_order,public_sections
from .store_candidate_engine_v1 import parallel_rows,parallel_payload


class LivePayloadFixtureTests(unittest.TestCase):
    """Explicit CPU fake network payloads; no model, port or HTTP call."""
    def setUp(self):
        self.products=probe.old.public_catalog();self.value={'query':'Browse the store'}
        self.expected={'adapterSha256':'a'*64,'v5HeadSha256':'b'*64,'gateSha256':'c'*64,'copyHeadSha256':'d'*64,
            'colorHeadSha256':runtime.COLOR_SHA}
        self.public={'promptTokenIdsSha256':'e'*64,'promptTokens':740,'queryTextSha256':'f'*64}

    def events(self,*,empty=False):
        if empty:self.value={'query':'Trail mug under $20'}
        request,policy,_=probe.contract.common_request(self.value,probe.formatter.load_public_catalog(),self.products)
        self.case={'request':self.value,'effectiveRequest':request,'effectivePolicy':policy}
        lexical=torch.arange(16.,0.,-1.);v5=torch.ones(16)/16;p=.3;scores=runtime.blended_scores(v5,lexical,p).tolist()
        order,_=guarded_order(scores,request,policy,self.products);cards=[]
        for index in order[:6]:
            product=self.products[index];variant=product['variants'][0]
            cards.append({'productIndex':index,'facts':{'name':product['name'],'category':product['category'],'itemPrice':product['itemPrice'],'selectedColor':variant['colorName']},
                'action':{'productId':product['productId'],'variantId':variant['variantId'],'image':variant['images'][0]}})
        native={'kind':'fresh-native-last16-rank-adapter','adapterCheckpointSha256':self.expected['adapterSha256'],'promptTokenIdsSha256':self.public['promptTokenIdsSha256'],
            'freshLearnedDecision':True,'generatedContinuationTokens':0,'copyAdapterParticipated':False,'optimizerUpdates':0,'targetWeightsProvided':False,
            'cachedScoringMode':'plain','cacheHit':False,'catalogPrefixCacheHit':False,'catalogPrefixCreatedThisRequest':True,'catalogPrefixCacheUsed':True,
            'catalogPrefixTokensReused':0,'catalogPrefixTokensPrefilled':721,'catalogPrefixPrefillForwards':3,'freshPublicPrefixTokens':3,
            'nativeRankSuffixTokens':16,'nativeBodyForwards':5,'attemptedNativeBodyForwards':5,'dynamicMode':'gradient-public-XHY-last16',
            'gradientEvaluationCalls':1,'nativeLanguageModelProjectionCalls':2,
            'targetFreeCapture':{'targetOrLossProvided':False,'optimizerUpdates':0,'scalar':'sum16publicroutinglogits','sourceCacheUnchanged':True,
                'adapterUnchanged':True,'promptTokenIdsSha256':self.public['promptTokenIdsSha256'],'nativeRouteScores':lexical.tolist()}}
        decision={'phraseWhitelistUsed':False,'failedOldProductRankingUsed':False,'rankAdapterAndCopyAdapterMixed':False,'hiddenProfileInputs':False,
            'lexicalRouteLogits16':lexical.tolist(),'v5Scores16':v5.tolist(),'scores16':scores,'learnedQuerySpecificProbability':p,
            'rankDecision':native,'queryFeature':{'cacheHit':False,'nativeBackboneForwardCalls':1},
            'colorFeature':{'kind':'fresh-base-color-context','cacheHit':False,'nativeBackboneForwardCalls':1,'checkpointSha256':runtime.COLOR_SHA,
                'oldProductScoresUsedForProductRanking':False,'withinProductVariantMathUnchanged':True}}
        provenance={'nativeCopyCheckpointSha256':probe.NATIVE_COPY_SHA,'rankCheckpointSha256':self.expected['adapterSha256'],'v5QueryCheckpointSha256':self.expected['v5HeadSha256'],'personaRanking':decision}
        metadata,_=public_sections(cards,[],self.products)
        shared={'requestId':'actual-cpu-fixture','eligibleCount':len(order),'shortfall':max(0,6-len(cards)),'provenance':provenance}
        events=[{'type':'rank','productIds':[self.products[i]['productId'] for i in order],
            'selectedProductIds':[c['action']['productId'] for c in cards],'variants':{c['action']['productId']:c['action']['variantId'] for c in cards},
            'sections':metadata,'timing':{'rankMs':100.,'readyMs':100.},**shared}]
        if cards:
            rows=parallel_rows(cards);fields=[]
            for row in rows:
                inp=row['input'];facts=inp.get('source',{}).get('facts');kind=inp['kind']
                text=('Store collection' if kind=='section' else f"{facts['selectedColor']} {facts['name']}" if kind=='title' else
                    f"{facts['selectedColor']} option" if kind=='badge' else f"Shown in {facts['selectedColor']} at ${facts['itemPrice']:.2f}.")
                fields.append({'text':text,'rejections':[],'accepted':True,'finishReason':'eos'})
            generated={'fields':fields}
            source={'kind':'fresh-parallel-head','generatedEarlier':False,'freshPerRequestGeneration':True,
                'checkpointSha256':self.expected['copyHeadSha256'],'feedforwardCalls':1}
            payload=lambda limit:parallel_payload(cards,rows,generated,limit)
        else:
            source={'kind':'no-generation','generatedEarlier':False,'freshPerRequestGeneration':False,'noEligibleProducts':True}
            payload=lambda limit:public_sections([],[],self.products)
        for kind,limit,timing in ([('firstThree',3,200.),('complete',6,201.)] if cards else [('complete',0,101.)]):
            sections,validation=payload(limit)
            events.append({'type':kind,'sections':sections,'validation':validation,**{k:validation[k] for k in ('requiredFields','acceptedFields','withheldFields')},
                'copySource':source,'timing':{'rankMs':100.,'readyMs':timing,'first3readyMs' if kind=='firstThree' else 'fullReadyMs':timing},**shared})
        return events

    def analyze(self,events):return probe.analyze_events(self.case,self.public,events,{'x-request-id':'actual-cpu-fixture'},self.products,self.expected,set())

    def test_full_six_twenty_fields_and_real_source_variant_metadata(self):
        result=self.analyze(self.events())
        self.assertTrue(result['protocolPassed'],result);self.assertTrue(result['qualityPassed'],result)
        self.assertEqual((result['actualCards'],result['acceptedFields'],result['requiredFields']),(6,20,20))
        self.assertEqual(result['rankDecision']['catalogPrefixTokensReused'],0)
        self.assertEqual(result['learnedGateCheckpointSha256'],self.expected['gateSha256'])

    def test_wrong_color_price_generation_identity_or_native_receipt_detected(self):
        original=self.events()
        for mutate in [lambda e:e[-1]['sections'][0]['cards'][0].update(selectedColor='Invented'),
                       lambda e:e[-1]['sections'][0]['cards'][0].update(bodyCopy='Shown in Green at $1.00.'),
                       lambda e:e[-1]['copySource'].update(freshPerRequestGeneration=False),
                       lambda e:e[-1]['provenance']['personaRanking']['rankDecision'].update(catalogPrefixTokensReused=721),
                       lambda e:e[-1]['provenance']['personaRanking']['rankDecision']['targetFreeCapture'].update(optimizerUpdates=1),
                       lambda e:e[-1]['provenance']['personaRanking'].update(phraseWhitelistUsed=True)]:
            changed=deepcopy(original);mutate(changed);result=self.analyze(changed)
            self.assertFalse(result['qualityPassed'],result)

    def test_empty_ineligible_exact_product_abstains_without_padded_or_fake_fresh_copy(self):
        result=self.analyze(self.events(empty=True))
        self.assertTrue(result['qualityPassed'],result);self.assertEqual(result['actualCards'],0);self.assertEqual(result['requiredFields'],0)


class PlanAndHealthTests(unittest.TestCase):
    def plan(self):
        path=probe.HERE/'persona_http_semantic_prepare_v1/plan.json';original=json.loads(path.read_text())
        return {'version':probe.VERSION,'fixedDenominator':320,'publicOnly':True,'privateGoldParsed':False,'acceptanceStateWritten':False,
            'originalRoster':{'file':str(path),'sha256':probe.ROSTER_SHA},'cases':deepcopy(original['cases']),
            'expectedPublic':{c['probeCaseId']:{} for c in original['cases']},'sourceSha256':{},'scope':'CPU fixture','timingScope':'CPU fixture'}
    def test_original_roster_exact_and_missing_or_changed_case_rejects(self):
        plan=self.plan();probe.validate_plan(plan)
        plan['cases'][0]['request']['query']='another query'
        with self.assertRaisesRegex(ValueError,'cases/order'):probe.validate_plan(plan)
    def test_failure_before_ready_preserves_all320_unattempted_no_posts(self):
        plan=self.plan();calls=[]
        with tempfile.TemporaryDirectory(dir=probe.ROOT) as directory:
            out=Path(directory)/'failed'
            def exchange(*args,**kwargs):calls.append(args[1]);return {'httpStatus':200,'rawBytes':b'{"ready":false}','headers':{}}
            result=probe.run(plan,out,'unused','a'*64,selection_fn=lambda *args:{'expected':{},'pins':{}},exchange_fn=exchange,
                opener_factory=lambda:None,deadline_context=lambda seconds:nullcontext())
            self.assertEqual(result['status'],'stopped');self.assertEqual(result['summary']['unattempted'],320)
            self.assertEqual(result['postRequestAttempts'],0);self.assertEqual(calls,['http://127.0.0.1:8767/health'])
            self.assertTrue((out/'results.json').exists())
    def test_only_explicit8767_and_no_accepted_profile_substitution(self):
        for url in ('http://127.0.0.1:8766','https://127.0.0.1:8767','http://example.com:8767','http://localhost:8767/?x=1'):
            with self.assertRaisesRegex(ValueError,'8767'):probe.run(self.plan(),'unused','unused','a'*64,base_url=url)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'accepted.json';path.write_text('{"version":"personalized-local-profile.v1","status":"accepted"}')
            with self.assertRaisesRegex(ValueError,'candidate'):probe.selection(path,probe.old.sha(path))

class CandidateEvidenceAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.records={};self.receipt={'version':'persona-adapter-serving-candidate.v1',
            'status':'offline-passed-query-mechanical-passed-serving-pending','overallPromotionPass':False,
            'sourceSha256':{str((probe.HERE/name).resolve()):probe.old.sha(probe.HERE/name) for name in probe.SOURCE_NAMES}}
        self.adapter_sha='a'*64;self.gate=self.root/'gate.pt';self.gate.write_bytes(b'CPU fake gate; no model load');self.gate_sha=probe.old.sha(self.gate)
        self.bundle=self.root/'query-features.pt';self.bundle.write_bytes(b'CPU fake query bundle')
        self.freeze=self.root/'ALL_MODELS_PREDICTIONS_FROZEN.json';self.freeze.write_text('{}');self.freeze_sha=probe.old.sha(self.freeze)
        self.records['offlineAudit']={'version':'persona-choice-combined-independent-audit.v1','status':'completed-offline-adjudication',
            'offlineCoreAndReplacementPass':True,'overallPromotionPass':False,'privateQualityJSONParsed':True,
            'sourceSha256':probe.old.sha(probe.HERE/'audit_persona_choice_combined_v1.py'),
            'gateSha256':probe.old.sha(probe.HERE/'PERSONA_CHOICE_COMBINED_PROMOTION_GATE_V2.json')}
        self.records['admissionBeforeGold']={'privateQualityJSONParsed':False,'adapterCheckpointSha256':self.adapter_sha,'combinedFreezeSha256':self.freeze_sha}
        self.records['queryMechanicalAudit']={'version':'query-intent-independent-TRAIN-audit.v3','status':'passed-independent-TRAIN-replay',
            'headSha256':self.gate_sha,'auditSourceSha256':probe.old.sha(probe.HERE/'store_query_intent_audit_v3.py'),
            'privateQualityLabelsRead':False,'sourceManifestSha256':runtime.INTENT_TRAIN_MANIFEST_SHA,
            'replay':{'observedReplayOptimizerSteps':250,'everyFinalParameterBitwiseMatches':True},'twoReloadsExactLogitParity':True,
            'reloadSavedParity':{'rows':1272,'exactSavedProbabilityParity':True},'inputSha256':{str(self.gate):self.gate_sha}}
        v5=Path('/Users/lukas/Documents/Projects/dubhacks-2026/data/ranker/pretrained/actual-store-function-v5-attempt-1/serving-head.json')
        self.v5=json.loads(v5.read_text());self.receipt['v5Export']={'file':str(v5),'sha256':probe.old.sha(v5)}
        self.receipt['gateCheckpoint']={'file':str(self.gate),'sha256':self.gate_sha}
        self.receipt['copyManifest']={'file':'/Users/lukas/Documents/Projects/dubhacks-2026/data/ranker/pretrained/store-parallel-text-train-all-demo-v1-attempt-2/serving-manifest.json','sha256':probe.COPY_MANIFEST_SHA}
        self.receipt['integrationManifest']={'file':str(probe.HERE/'store_demo_integration_personalized_adapter_v1/manifest.json'),'sha256':probe.INTEGRATION_MANIFEST_SHA}
        self.records['adapterManifest']={}
        self.records['startup']={'persona':{'kind':'persona-adapter-engine.v2','checkpointSha256':self.adapter_sha,'servingManifestSha256':'pending-replaced',
            'queryIntent':{'checkpointSha256':self.gate_sha,'sourceSha256':probe.old.sha(probe.HERE/'store_query_intent_gate_v3.py'),
                'sourceManifestSha256':runtime.INTENT_TRAIN_MANIFEST_SHA},'color':{'checkpointSha256':runtime.COLOR_SHA,'failedOldProductOrderingUsed':False}},
            'rankCheckpointSha256':self.v5['head']['sha256'],'copyCheckpointSha256':probe.NATIVE_COPY_SHA,
            'copyCandidate':{'checkpointSha256':'074169e31bf5f6b99cc5bc35c9b2e64d82396d5225b3a67cc2cf5f613fcc4889'},'sourceTensorsHashed':927,
            'memory':{'processBudgetBytes':7*2**30},'rankSessionProvenance':{'rankHead':{'servingExportSha256':probe.old.sha(v5)}},
            'imageDeliveryManifestSha256':probe.INTEGRATION_MANIFEST_SHA}
        self.records['queryCombinationReplay']={'version':'persona-combined-query-replay-audit.v1','status':'passed',
            'gateCheckpointSha256':self.gate_sha,'v5HeadSha256':self.v5['head']['sha256'],'adapterCheckpointSha256':self.adapter_sha,
            'combinedFreezeSha256':self.freeze_sha,'rows':344,'modelRowsChecked':1376,'uniqueQueryVectorsReplayed':292,
            'exactQueryProbabilityAndV5Parity':True,'actualFloat32BlendAndGuardsExact':True,'overallPromotionPass':False,'servingPending':True,
            'sourceSha256':{str((probe.HERE/'audit_persona_combined_query_replay_v1.py').resolve()):probe.old.sha(probe.HERE/'audit_persona_combined_query_replay_v1.py')},
            'privateQualityLabelsRead':False,'optimizerUpdates':0,'nativeCalls':0,'specific320BehavioralRegressionComputed':False,
            'queryBundleSha256':probe.old.sha(self.bundle),'inputSha256':{str(self.bundle):probe.old.sha(self.bundle),str(self.freeze):self.freeze_sha}}
        self.path=self.root/'candidate.json';self.save()
    def save(self):
        for key in ('offlineAudit','admissionBeforeGold','queryMechanicalAudit','adapterManifest'):
            path=self.root/(key+'.json');probe.old.write(path,self.records[key]);self.receipt[key]={'file':str(path),'sha256':probe.old.sha(path)}
        self.records['queryCombinationReplay']['mechanicalQueryAuditSha256']=self.receipt['queryMechanicalAudit']['sha256']
        self.records['startup']['persona']['servingManifestSha256']=self.receipt['adapterManifest']['sha256']
        for key in ('queryCombinationReplay','startup'):
            path=self.root/(key+'.json');probe.old.write(path,self.records[key]);self.receipt[key]={'file':str(path),'sha256':probe.old.sha(path)}
        probe.old.write(self.path,self.receipt)
    def select(self):
        with patch.object(runtime,'load_manifest',return_value=({'adapter':{'sha256':self.adapter_sha}},{})):
            return probe.selection(self.path,probe.old.sha(self.path))
    def test_all_three_passed_audits_selected_source_lifetime_and_health_bind(self):
        selected=self.select();self.assertFalse(selected['overallPromotionPass']);self.assertFalse(selected['acceptanceStateWritten'])
        self.assertTrue(probe.validate_health({'ready':True,'status':'ready','startup':selected['expected']['startup']},selected['expected']))
        changed=deepcopy(selected['expected']['startup']);changed['persona']['queryIntent']['checkpointSha256']='0'*64
        with self.assertRaisesRegex(ValueError,'startup differs'):probe.validate_health({'ready':True,'status':'ready','startup':changed},selected['expected'])
    def test_failed_offline_or_query_audit_rejected_even_with_rehashed_candidate(self):
        original=deepcopy(self.records)
        for mutate in [lambda r:r['offlineAudit'].update(offlineCoreAndReplacementPass=False),
                       lambda r:r['queryMechanicalAudit']['replay'].update(everyFinalParameterBitwiseMatches=False),
                       lambda r:r['queryCombinationReplay'].update(actualFloat32BlendAndGuardsExact=False),
                       lambda r:r['queryCombinationReplay'].update(adapterCheckpointSha256='0'*64)]:
            self.records=deepcopy(original);mutate(self.records);self.save()
            with self.assertRaises(ValueError):self.select()


if __name__=='__main__':unittest.main()
