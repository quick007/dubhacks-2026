import copy,hashlib,unittest
import torch
from . import store_expanded35_verification_v1 as audit
from .store_expanded_rank_v1 import checked_request,strict_eligible_order,expanded_scores
from .store_parallel_text_data_v1 import tokens,detokenize,SLOTS
from .store_demo_engine import ranking_input
from .store_function_adaptation import request_text
from .store_query_prefix_reuse_v1 import hashed
from .store_qwen_lexical_choice_inputs_v2 import load_public_catalog
class ExpandedVerificationTests(unittest.TestCase):
 def fixture(self,query='Browse the store'):
  products=audit.load_catalog();request=copy.deepcopy(audit.fixed_schedule()[0]['request']);request['query']=query;checked,policy,_=checked_request(request,products);families=[p['id'] for p in load_public_catalog()];v5=torch.linspace(0,1,35);lexical=torch.linspace(-1,1,16);p=.5;scores,family=expanded_scores(v5,lexical,p,products,families);order,guards=strict_eligible_order(scores.tolist(),checked,policy,products);cards=[];metas=[];values={};sections=[]
  for start in range(0,min(6,len(order)),3):
   group=[];heading='Your collection' if len(order[start:start+3])==3 else None;values['S'+str(start//3+1)]=heading
   for i,index in enumerate(order[start:start+3],start):
    product=products[index];variant=product['variants'][0];color=variant['colorName'];card={'productId':product['productId'],'variantId':variant['variantId'],'originalName':product['name'],'selectedColor':color,'itemPrice':product['itemPrice'],'title':product['name']+' in '+color,'badge':color+' option','bodyCopy':f"Shown in {color} at ${product['itemPrice']}."};group.append(card);cards.append(card);metas.append({**card,'image':variant['images'][0]});values.update({'T'+str(i+1):card['title'],'B'+str(i+1):card['badge'],'C'+str(i+1):card['bodyCopy']})
   sections.append({'heading':heading,'cards':group})
  strings=[values[field] for field,_,_ in audit.slot_fields(cards)];vocab=['<eos>','<pad>']+sorted({w for text in strings for w in tokens(text)});fields=[]
  for text in strings:
   ids=[vocab.index(w) for w in tokens(text)]+[0];ids+= [1]*(SLOTS-len(ids));fields.append({'slotIds':ids,'text':detokenize(tokens(text)),'accepted':True,'finishReason':'eos'})
  source={'kind':'fresh-parallel-head','generatedEarlier':False,'freshPerRequestGeneration':True,'feedforwardCalls':1,'checkpointSha256':'copy'} if cards else {'kind':'no-generation','noEligibleProducts':True,'freshPerRequestGeneration':False}
  text=request_text(ranking_input(checked));ids=[1,2,3];enc={'queryEncodingMode':'ordinary','queryFeatureCacheMode':'lru32','queryFeatureCacheHit':True,'noScoreOrRankingOrAnswerCache':True,'reusedPrefixTokens':0,'originalFormatterTextSha256':hashlib.sha256(text.encode()).hexdigest(),'wholeTextSha256':hashlib.sha256(text.encode()).hexdigest(),'actualEncodingText':text,'fullTokenIds':ids,'wholeTokenSha256':hashed(ids),'originalFormatterTokenSha256':hashed(ids),'wholeRequestTokens':3,'originalFormatterTokens':3,'paddedTokensPerRow':0,'freshSuffixTokens':0}
  d={'catalogProducts':35,'catalogSha256':audit.CATALOG_SHA,'newProductsTrainedByLoRA':False,'rankAdapterAndCopyAdapterMixed':False,'scores':scores.tolist(),'v5Scores':v5.tolist(),'lexicalRouteLogits16':lexical.tolist(),'familyPriorScores':family.tolist(),'learnedQuerySpecificProbability':p,'queryFeature':{'cacheHit':True,'nativeBackboneForwardCalls':0},'queryEncoding':enc,'rankDecision':{'freshLearnedDecision':True,'copyAdapterParticipated':False,'generatedContinuationTokens':0,'optimizerUpdates':0}}
  provenance={'personaRanking':d};shared={'requestId':'actual-id','provenance':provenance,'eligibleCount':len(order),'shortfall':max(0,6-len(cards))};rank={'type':'rank',**shared,'productIds':[products[i]['productId'] for i in order],'selectedProductIds':[c['productId'] for c in cards],'variants':{c['productId']:c['variantId'] for c in cards},'sections':[{'cards':metas}]};accepted=sum(x is not None for x in values.values());validation={'requiredFields':len(values),'acceptedFields':accepted,'withheldFields':len(values)-accepted,'fields':[{'field':k,'accepted':v is not None} for k,v in values.items()]};complete={'type':'complete',**shared,'sections':sections,'copySource':source,'validation':validation,**{k:validation[k] for k in ('requiredFields','acceptedFields','withheldFields')}};events=[rank]+([{'type':'firstThree',**shared,'sections':sections[:1],'copySource':source}] if cards else [])+[complete];receipt={'requestId':'actual-id','input':request,'events':events,'status':'completed','effectiveRequest':checked,'policy':policy,'finalOrder':order,'guards':guards,'parallelGeneration':{'fields':fields,'feedforwardCalls':1,'teacherStringsRead':False}};evidence={'products':products,'startup':{'persona':{'productIds':families}},'copyCheckpointSha256':'copy','vocabulary':vocab};return request,events,{'x-request-id':'actual-id'},evidence,receipt
 def test_actual35_105_catalog(self):self.assertEqual(len(audit.load_catalog()),35);self.assertEqual(sum(len(p['variants']) for p in audit.load_catalog()),105)
 def test_complete6_facts_slotids_and_fusion(self):
  args=self.fixture();r=audit.audit_events(*args);self.assertTrue(r['passed'],r['errors']);self.assertTrue(r['all20Accepted'])
 def test_changed_visible_price_payload_rejected(self):
  args=self.fixture();args[1][-1]['sections'][0]['cards'][0]['itemPrice']+=1;self.assertFalse(audit.audit_events(*args)['passed'])
 def test_actual_word_ids_must_decode_to_text(self):
  args=self.fixture();args[4]['parallelGeneration']['fields'][0]['slotIds'][0]=0;self.assertFalse(audit.audit_events(*args)['passed'])
 def test_full35_score_change_rejected(self):
  args=self.fixture();args[1][0]['provenance']['personaRanking']['scores'][34]+=.1;self.assertFalse(audit.audit_events(*args)['passed'])
 def test_category3_not_padded_or_called6(self):
  args=self.fixture('Work bags');r=audit.audit_events(*args);self.assertTrue(r['passed'],r['errors']);self.assertLessEqual(r['actualCards'],3);self.assertFalse(r['all20Accepted']);self.assertEqual(r['shortfall'],6-r['actualCards'])
 def test_empty_guard_is_no_generation(self):
  args=self.fixture('in stock size M');r=audit.audit_events(*args);self.assertTrue(r['passed'],r['errors']);self.assertEqual(r['actualCards'],0)
 def test_fixeddenominators_and_separate_groups(self):
  rows=audit.fixed_schedule();self.assertEqual(len(rows),16);self.assertEqual(sum(r['kind']=='warm-public-Browse' for r in rows),10);self.assertIsNone(audit.distribution([1,None],2)['p50']);self.assertIsNone(audit.distribution([1],10)['p95'])
if __name__=='__main__':unittest.main()
