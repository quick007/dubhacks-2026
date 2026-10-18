"""CPU fake receipts only; no native, model, HTTP, browser, private gold or fit."""
from copy import deepcopy
import hashlib,json
import unittest
from . import store_persona_warm_http_probe_v1 as warm
from .test_store_persona_http_semantic_probe_v2 import LivePayloadFixtureTests

class FakeTokenizer:
    def encode(self,text,**kwargs):return list(range(64))

def call(text,role=None):
    row={'textSha256':warm.text_sha(text),'paddedTokensPerRow':64,'nativeBackboneForwardCalls':1,'batchRows':1,
        'adaptersDisabled':True,'truncatedRows':0,'elapsedMs':1.0}
    if role:row['role']=role
    return row

def startup():
    rows=[];calls=[]
    for pid,value in warm.measurement.public_browse_requests():
        text=warm.color_formatter.public_context_text(value)
        rows.append({'publicPersonId':pid,'formatterTextSha256':warm.text_sha(text),'formatterTextBytes':len(text.encode()),
            'featureTensorSha256':'a'*64,'cacheKeySha256':warm.text_sha(text),'nativeBackboneForwardCalls':1,
            'encodingMs':1.0,'featureSource':'fresh-base-color-feature'})
        calls.append(call(text))
    pre={'version':'persona-adapter-runtime.v3','enabled':True,'publicPeopleSha256':warm.measurement.PUBLIC_PEOPLE_SHA,
        'requestQuery':'Browse the store','rows':rows,'baseEncodingCalls':calls,'nativeBackboneForwardCalls':10,'encodingMs':10.,
        'rankingWarmResponses':0,'copyWarmResponses':0,'responseWordsCached':False,'rankWholePromptFeaturesPrecomputedHere':False,
        'cacheProducerSourceSha256':warm.old.sha(warm.measurement.__file__)}
    pre['warmCacheSha256']=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'persona':{'measurementEngine':'persona-adapter-engine.v3-measurement','forceFreshQuery':True,
        'wholePromptRankCacheIncludesQuery':True,'noResponseWarmupOrTextCache':True,'featurePreencoding':pre}}

class WarmProofTests(unittest.TestCase):
    def test_startup_exact_ten_public_context_texts_tokens_and_separate_encodes(self):
        s=startup();result=warm.validate_warm_startup(s,FakeTokenizer())
        self.assertEqual(result['startupNativeColorEncodes'],10);self.assertEqual(result['copyWarmResponses'],0)
        for mutate in [lambda p:p.update(nativeBackboneForwardCalls=0),lambda p:p['rows'][0].update(formatterTextSha256='0'*64),
                       lambda p:p['baseEncodingCalls'][0].update(paddedTokensPerRow=63),lambda p:p.update(copyWarmResponses=1),
                       lambda p:p.update(warmCacheSha256='0'*64)]:
            wrong=deepcopy(s);mutate(wrong['persona']['featurePreencoding'])
            with self.assertRaises(ValueError):warm.validate_warm_startup(wrong,FakeTokenizer())
    def fixture(self,kind='preencoded-public-color-context'):
        f=LivePayloadFixtureTests();f.setUp();f.value=deepcopy(warm.measurement.public_browse_requests()[0][1]);events=f.events()
        f.public.update(warm.public_encoding_expectations(f.value,f.products,FakeTokenizer()));f.expected['startup']=startup()
        decision=events[0]['provenance']['personaRanking'];q=decision['queryFeature'];text=warm.query.runtime_text(f.value,f.products)
        q.update(forceFreshQuery=True,encodingMeasurement=call(text,'query'))
        color=decision['colorFeature'];caching=kind!='fresh-base-color-context';ctext=warm.color_formatter.public_context_text(f.value)
        color.update(kind=kind,cacheHit=caching,nativeBackboneForwardCalls=int(not caching),formatterTextSha256=warm.text_sha(ctext),
            cacheKeySha256=warm.text_sha(ctext),featureTensorSha256='a'*64,cacheProducerSourceSha256=warm.old.sha(warm.measurement.__file__),
            warmCacheSha256=f.expected['startup']['persona']['featurePreencoding']['warmCacheSha256'],responseWordsCached=False,
            cachePolicy='bounded exact full formatter text; query included',cacheCapacity=32,cacheEntries=10,
            cacheOrigin='startup-preencoding' if kind=='preencoded-public-color-context' else 'dynamic-request')
        calls=[call(text,'query')]+([] if caching else [call(ctext,'color-context')])
        decision['featureEncoding']={'baseCalls':calls,'nativeBackboneForwardCalls':len(calls),'baseB1EncoderForwardCalls':len(calls),
            'publicProfilePreencodingEnabled':True,'colorContextCacheIncludesActualQuery':True,'rankXHYCacheIncludesActualWholePromptAndQuery':True,
            'newUniqueQueryMayMissBothContextAndRankCaches':True,'responseWordsCached':False,'rankingFormulaUnchanged':True,
            'rankDecisionNativeBodyForwards':decision['rankDecision']['nativeBodyForwards']}
        return f,events
    def analyze(self,f,events):return warm.analyze_events(f.case,f.public,events,{'x-request-id':'actual-cpu-fixture'},f.products,f.expected,set())
    def test_full_factual_analyzer_preserves_math_twenty_fields_and_accepts_all_three_truthful_color_kinds(self):
        for kind in ['preencoded-public-color-context','exact-full-text-multi-color-context','fresh-base-color-context']:
            f,events=self.fixture(kind);result=self.analyze(f,events)
            self.assertTrue(result['protocolPassed'],result);self.assertTrue(result['qualityPassed'],result)
            self.assertEqual(result['acceptedFields'],20);self.assertEqual(result['queryFeature']['nativeBackboneForwardCalls'],1)
    def test_query_cache_reuse_even_reload_or_bad_live_text_token_is_rejected(self):
        f,events=self.fixture()
        for mutate in [lambda r:r['queryFeature'].update(cacheHit=True,nativeBackboneForwardCalls=0),
                       lambda r:r['queryFeature'].update(forceFreshQuery=False),
                       lambda r:r['featureEncoding']['baseCalls'][0].update(textSha256='0'*64),
                       lambda r:r['featureEncoding']['baseCalls'][0].update(paddedTokensPerRow=63)]:
            bad=deepcopy(events);mutate(bad[0]['provenance']['personaRanking']);self.assertFalse(self.analyze(f,bad)['protocolPassed'])
    def test_color_exact_full_actual_query_and_startup_tensor_hash_bound(self):
        f,events=self.fixture()
        for mutate in [lambda c:c.update(formatterTextSha256='0'*64),lambda c:c.update(featureTensorSha256='b'*64),
                       lambda c:c.update(warmCacheSha256='0'*64),lambda c:c.update(cacheProducerSourceSha256='0'*64),
                       lambda c:c.update(cacheCapacity=1),lambda c:c.update(responseWordsCached=True)]:
            bad=deepcopy(events);mutate(bad[0]['provenance']['personaRanking']['colorFeature']);self.assertFalse(self.analyze(f,bad)['protocolPassed'])
        changed=deepcopy(f.value);changed['query']='work bag'
        other=warm.public_encoding_expectations(changed,f.products,FakeTokenizer())
        self.assertNotEqual(other['colorTextSha256'],f.public['colorTextSha256'])
    def test_dynamic_new_query_color_and_rank_work_are_not_mislabeled_startup_work(self):
        f,events=self.fixture('fresh-base-color-context');result=self.analyze(f,events)
        self.assertEqual(len(result['featureEncoding']['baseCalls']),2)
        self.assertEqual(result['featureEncoding']['baseB1EncoderForwardCalls'],2)
        self.assertGreater(result['rankDecision']['nativeBodyForwards'],0)
        self.assertFalse(result['colorFeature']['cacheHit'])

if __name__=='__main__':unittest.main()
