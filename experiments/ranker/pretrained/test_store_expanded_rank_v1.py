from copy import deepcopy
from types import SimpleNamespace as NS
import unittest
import torch
from . import store_expanded_rank_v1 as rank
from . import store_qwen_lexical_choice_inputs_v2 as formatter


def catalog():
    family=formatter.load_public_catalog();products=[]
    tags=['mug','backpack','notebook','rain-jacket','bottle','tote','blanket','lamp','lunchbox','headphones','shoes','sweatshirt','travel-bag','sunglasses','organizer','towel']
    for i,p in enumerate(family):
        variants=[{'variantId':v['id'],'colorName':v['colorName'],'images':[{'src':'/test-only.png'}]} for v in p['variants']]
        products.append({'productId':p['id'],'name':p['name'],'title':p['name'],'category':p['category'],'itemPrice':p['price'],'searchTerms':[],
            'parentFamily':p['id'],'intentTags':[tags[i]],'variants':variants,'colorVariants':[{'colorName':v['colorName']}for v in variants]})
    for productid,name,price,tag,parent in [('rain-anorak','Rain anorak',88,'rain-jacket','jacket'),('commuter-raincoat','Commuter raincoat',148,'rain-jacket','jacket'),('work-briefcase','Work briefcase',96,'work-bag','pack'),('work-tote','Office carryall',74,'work-bag','tote'),('commuter-pack','Commuter pack',112,'work-bag','pack')]:
        v={'variantId':productid+'-ivory','colorName':'Ivory','images':[{'src':'/test-only.png'}]}
        products.append({'productId':productid,'name':name,'title':name,'category':tag,'itemPrice':price,'searchTerms':[],'parentFamily':parent,'intentTags':[tag],
            'variants':[v],'colorVariants':[{'colorName':'Ivory'}]})
    vectors=torch.zeros(len(products),4096);vectors[:,0]=1
    return {'products':products,'vectors':vectors,'variants':[{**v,'productIndex':i} for i,p in enumerate(products) for v in p['variants']]}


class Tests(unittest.TestCase):
    def test_strict_rain_and_work_categories_precede_any_personalized_score(self):
        c=catalog();scores=[1000.-i for i in range(len(c['products']))]
        for query,expected in [('Rain jackets',{'jacket','rain-anorak','commuter-raincoat'}),('Work bags',{'work-briefcase','work-tote','commuter-pack'}),('work tote',{'work-briefcase','work-tote','commuter-pack'}),('commuter backpacks',{'work-briefcase','work-tote','commuter-pack'}),('rain anoraks',{'jacket','rain-anorak','commuter-raincoat'})]:
            request,policy,_=rank.checked_request({'query':query},c['products'])
            order,guards=rank.strict_eligible_order(scores,request,policy,c['products'])
            self.assertEqual({c['products'][i]['productId'] for i in order},expected)
            self.assertFalse(guards['specificSearchEligibility']['learnedScoresUsedForEligibility'])
        request,policy,_=rank.checked_request({'query':'Browse the store'},c['products'])
        self.assertEqual(len(rank.strict_eligible_order(scores,request,policy,c['products'])[0]),len(c['products']))

    def test_hard_eligible_exact_identity_cannot_be_dropped_by_ambiguous_noun_filter(self):
        c=catalog();c['products'][14]['name']='Rain jacket organizer';c['products'][14]['title']='Rain jacket organizer'
        request,policy,_=rank.checked_request({'query':'Rain jacket organizer'},c['products'])
        order,guards=rank.strict_eligible_order([100-i for i in range(21)],request,policy,c['products'])
        self.assertEqual(order[0],14);self.assertEqual(guards['specificSearchEligibility']['hardEligibleExactIdentityPreserved'],[14])
        request,policy,_=rank.checked_request({'query':'Rain jacket organizer under $1'},c['products'])
        self.assertEqual(rank.strict_eligible_order([100-i for i in range(21)],request,policy,c['products'])[0],[])

    def test_phone_stand_source_does_not_claim_laptop_stand_function(self):
        c=catalog();p=c['products'][14];p['name']='Foldaway Phone Stand';p['title']=p['name'];p['intentTags']=['stand']
        for query,expected in [('phone stand',[14]),('mobile stands',[14]),('laptop stand',[])]:
            request,policy,_=rank.checked_request({'query':query},c['products']);order,_=rank.strict_eligible_order([100-i for i in range(21)],request,policy,c['products'])
            self.assertEqual(order,expected)

    def test_budget_exclusive_colors_exact_identity_and_unsupported_remain_hard(self):
        c=catalog();scores=list(range(len(c['products'])))
        for value,expected in [({'query':'Rain jackets under $1'},0),({'query':'Work bags','itemBudget':100},2),
            ({'query':'Work bags','hardRequirements':[{'attribute':'stock','value':'in-stock'}]},0),
            ({'query':'Work bags','hardRequirements':[{'attribute':'color','value':'Sage'},{'attribute':'color','value':'Ivory'}]},0),
            ({'query':'Work briefcase under $90'},0)]:
            request,policy,_=rank.checked_request(value,c['products']);self.assertEqual(len(rank.strict_eligible_order(scores,request,policy,c['products'])[0]),expected)

    def test_family_probability_not_duplicated_and_original_finite_float32_blend(self):
        c=catalog();ids=[p['id']for p in formatter.load_public_catalog()];lex=torch.arange(16,dtype=torch.float32);v5=torch.linspace(0,1,len(c['products']))
        scores,prior=rank.expanded_scores(v5,lex,.75,c['products'],ids)
        original=lex.softmax(-1)
        for family in ids:self.assertTrue(torch.allclose(prior[[p['parentFamily']==family for p in c['products']]].sum(),original[ids.index(family)]))
        self.assertTrue(torch.equal(scores,.25*prior+.75*(v5+.05*prior)))
        self.assertAlmostEqual(float(prior.sum()),1,places=6)

    def test_new_history_validated_against_actualsource_then_literal_observations_preserved_in_familyview(self):
        c=catalog();value={'query':'Browse the store','publicHistory':[{'eventId':'event','eventType':'purchase','itemName':'Work briefcase','category':'work-bag','productId':'work-briefcase','variantId':'work-briefcase-ivory','selectedColor':'Ivory','occurredAt':'2026-10-18T00:00:00Z','source':'storefront-action'}]}
        rank.checked_request(value,c['products']);snapshot=deepcopy(value)
        projected,proof=rank.family_profile_view(value,formatter.load_public_catalog());event=projected['publicHistory'][0]
        self.assertNotIn('productId',event);self.assertNotIn('variantId',event);self.assertTrue(event['external']);self.assertEqual(event['itemName'],'Work briefcase')
        formatter.checked_request(projected,formatter.load_public_catalog());self.assertEqual(value,snapshot);self.assertEqual(proof['newSkuObservationsRepresentedOutsideOriginalFamilyCatalog'],1)
        value['publicHistory'][0]['selectedColor']='invented'
        with self.assertRaises(ValueError):rank.checked_request(value,c['products'])

    def test_dynamic_learned_variant_projection_and_failure_restore_original_catalog(self):
        c=catalog();family=formatter.load_public_catalog();copy=object();bb=object();queries=[]
        branch=NS(catalog={'original':True},colors=object(),indices=object())
        original=(branch.catalog,branch.colors,branch.indices)
        def color(value,encode):
            self.assertEqual(len(branch.catalog['products']),21);return torch.arange(len(c['variants']),dtype=torch.float32),{'learned':True}
        branch.scores=color
        runtime=NS(manifest={'productIds':[p['id']for p in family]},public_catalog=family,bb=bb,ownership=NS(copy=copy),color=branch,
            assert_current=lambda:None,query_intent=NS(p_specific=lambda q:.5),encoding_calls=[],public_row=lambda value:value,
            lexical_scores=lambda row,cancelled:(torch.arange(16,dtype=torch.float32),{'fresh':True}))
        def encode(text):queries.append(text);return torch.ones(1,4096)/64,{'backboneForwardCalls':1}
        runtime.encode_base=encode
        head=NS(probabilities=lambda role,values:torch.ones(len(values),7)/7)
        adapter=rank.ExpandedRankAdapter(runtime,c,catalog_sha256='a'*64)
        result=adapter.rank({'query':'Work bags'},NS(bb=bb,head=head),copy)
        self.assertEqual(len(result[4]),3);self.assertEqual((branch.catalog,branch.colors,branch.indices),original)
        self.assertTrue(all(card['facts']['selectedColor']=='Ivory'for card in result[4]))
        def fail(*a):raise RuntimeError('test color failure')
        branch.scores=fail
        with self.assertRaisesRegex(RuntimeError,'color failure'):adapter.rank({'query':'Work bags'},NS(bb=bb,head=head),copy)
        self.assertEqual((branch.catalog,branch.colors,branch.indices),original)

if __name__=='__main__':unittest.main()
