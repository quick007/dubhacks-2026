from copy import deepcopy
import hashlib,json
import unittest
from . import store_persona_factorized_http_probe_v1 as factor
from .test_store_persona_warm_http_probe_v1 import FakeTokenizer,call
from .test_store_persona_http_semantic_probe_v2 import LivePayloadFixtureTests

class FactorizedTests(unittest.TestCase):
 def fixture(self):
  f=LivePayloadFixtureTests();f.setUp();f.value=deepcopy(factor.measurement.previous.public_browse_requests()[0][1]);events=f.events()
  f.public.update(factor.public_encoding_expectations(f.value,f.products,FakeTokenizer()));key=factor.measurement.profile_key(f.value)
  f.expected['startup']={'persona':{'featurePreencoding':{'rankProfileRows':[{'profileKeySha256':key,'pureFeatureSha256':'a'*64}],
   'colorPreencoding':{'warmCacheSha256':'b'*64,'rows':[{'formatterTextSha256':f.public['colorTextSha256'],'featureTensorSha256':'a'*64}]}}}}
  d=events[0]['provenance']['personaRanking'];q=d['queryFeature'];qt=factor.query.runtime_text(f.value,f.products);qc=call(qt,'query');q.update(forceFreshQuery=True,encodingMeasurement=qc)
  n=d['rankDecision'];n.update(kind='factorized-profile-XHY-feature-reuse',cacheHit=True,nativeBodyForwards=0,nativeLanguageModelProjectionCalls=1,gradientEvaluationCalls=0,catalogPrefixCacheUsed=False,
   profileKeySha256=key,pureFeatureSha256='a'*64,conditioningPolicy=factor.measurement.POLICY,actualQueryOrGuardFieldsUsedForContext=False,actualQueryUsedForFreshQueryHeadAndGuards=True,scoresOrRankingsOrResponseWordsCached=False)
  c=d['colorFeature'];c.update(kind='preencoded-public-color-context',cacheHit=True,nativeBackboneForwardCalls=0,formatterTextSha256=f.public['colorTextSha256'],cacheKeySha256=f.public['colorTextSha256'],featureTensorSha256='a'*64,
   cacheProducerSourceSha256=factor.old.sha(factor.measurement.__file__),warmCacheSha256='b'*64,responseWordsCached=False,cachePolicy='bounded exact neutral profile text; actual query/budget/hard requirements excluded',cacheCapacity=32,cacheEntries=10,cacheOrigin='startup-preencoding',profileKeySha256=key,conditioningPolicy=factor.measurement.POLICY,actualQueryOrGuardFieldsUsedForContext=False)
  d['featureEncoding']={'baseCalls':[qc],'nativeBackboneForwardCalls':1,'baseB1EncoderForwardCalls':1,'publicProfilePreencodingEnabled':True,'colorContextCacheIncludesActualQuery':False,'rankXHYCacheIncludesActualWholePromptAndQuery':False,'newUniqueQueryMayMissBothContextAndRankCaches':False,'responseWordsCached':False,'rankingFormulaUnchanged':True,'rankDecisionNativeBodyForwards':0,'profileFeatureKeyExcludesQueryBudgetHardRequirements':True,'profileCacheContainsPureFeaturesOnly':True}
  return f,events
 def analyze(self,f,e):return factor.analyze_events(f.case,f.public,e,{'x-request-id':'actual-cpu-fixture'},f.products,f.expected,set())
 def test_original_guard_math_copy_facts_and_neutral_cache_reuse(self):
  f,e=self.fixture();r=self.analyze(f,e);self.assertTrue(r['protocolPassed'],r);self.assertTrue(r['qualityPassed'],r);self.assertEqual(r['acceptedFields'],20)
 def test_query_and_guards_change_actual_query_hash_but_not_profile_key_or_color_text(self):
  f,_=self.fixture();other=deepcopy(f.value);other.update(query='work bag',itemBudget=90,hardRequirements=[{'attribute':'color','value':'Navy'}])
  a=factor.public_encoding_expectations(f.value,f.products,FakeTokenizer());b=factor.public_encoding_expectations(other,f.products,FakeTokenizer())
  self.assertEqual(a['profileKeySha256'],b['profileKeySha256']);self.assertEqual(a['colorTextSha256'],b['colorTextSha256']);self.assertNotEqual(a['queryTextSha256'],b['queryTextSha256'])
  self.assertEqual(factor.rank_request(other)['hardRequirements'],[]);self.assertIsNone(factor.rank_request(other)['itemBudget'])
 def test_changed_history_changes_profile_key(self):
  f,_=self.fixture();other=deepcopy(f.value);other['publicHistory']=other['publicHistory'][:-1]
  self.assertNotEqual(factor.measurement.profile_key(f.value),factor.measurement.profile_key(other))
 def test_wrong_profile_key_pure_feature_or_fresh_query_receipt_rejected(self):
  f,e=self.fixture()
  for mutate in [lambda d:d['rankDecision'].update(profileKeySha256='0'*64),lambda d:d['rankDecision'].update(pureFeatureSha256='0'*64),lambda d:d['queryFeature'].update(cacheHit=True,nativeBackboneForwardCalls=0),lambda d:d['colorFeature'].update(actualQueryOrGuardFieldsUsedForContext=True)]:
   bad=deepcopy(e);mutate(bad[0]['provenance']['personaRanking']);self.assertFalse(self.analyze(f,bad)['protocolPassed'])

if __name__=='__main__':unittest.main()
