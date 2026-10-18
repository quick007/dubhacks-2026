"""Source-bound expanded ranking; original16 LoRA supplies family priors only.

No native initialization, training, labels, response words or new-ID LoRA claim.
Explicit recognized product-type eligibility precedes learned personalization.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import torch

from .store_demo_engine import guarded_order, ranking_input
from .store_function_adaptation import request_text
from .store_persona_context_v1 import checked_context_request, source_colors
from .store_persona_adapter_contract_v2 import sha

VERSION = 'expanded-source-catalog-rank.v1'
# Explicit shopper product nouns only; these are eligibility, never rank boosts.
INTENT_ALIASES = {
 'rain-jacket': ('rain jacket','rain jackets','rainjacket','rainjackets','rain coat','rain coats','raincoat','raincoats','rain shell','rain shells','waterproof jacket','waterproof jackets','jacket for rain','jacket for the rain','rain anorak','rain anoraks','anorak','anoraks'),
 'work-bag': ('work bag','work bags','workbag','workbags','laptop bag','laptop bags','briefcase','briefcases','office bag','office bags','commuter bag','commuter bags','work tote','work totes','work backpack','work backpacks','commuter backpack','commuter backpacks','bag for work','bags for work'),
 'notebook': ('notebook','notebooks','paper notes','writing journal','writing journals'),
 'daypack': ('backpack','backpacks','daypack','daypacks'),
 'tote': ('tote','totes'), 'duffel': ('duffel','duffels','duffle','duffles'),
 'bottle': ('water bottle','water bottles','drinking bottle','drinking bottles'),
 'mug': ('mug','mugs','coffee cup','coffee cups'),
 'lunchbox': ('lunchbox','lunchboxes','lunch box','lunch boxes'),
 'lamp': ('lamp','lamps','desk light','desk lights'),
 'blanket': ('blanket','blankets'), 'towel': ('towel','towels'),
 'pouch': ('pouch','pouches','cable organizer','cable organizers'),
 'speaker': ('speaker','speakers'), 'clock': ('clock','clocks'),
 'planter': ('planter','planters','plant pot','plant pots'),
 'tray': ('tray','trays'), 'candle': ('candle','candles'),
}

INTENT_ALIASES.update({
 'backpack': INTENT_ALIASES['daypack'], 'travel-bag': INTENT_ALIASES['duffel'],
 'organizer': INTENT_ALIASES['pouch'], 'headphones': ('headphones','headphone','earphones'),
 'shoes': ('shoes','shoe','sneakers','sneaker'), 'sweatshirt': ('sweatshirt','sweatshirts','crewneck','crewnecks'),
 'sunglasses': ('sunglasses','sunglass'), 'cap': ('cap','caps','baseball hat'),
 'crossbody': ('crossbody','cross-body bag'), 'cubes': ('packing cubes','packing cube'),
 'cushion': ('cushion','cushions','pillow','pillows'), 'flask': ('flask','flasks'),
 'overshirt': ('overshirt','overshirts'), 'scarf': ('scarf','scarves'),
 'sleeve': ('laptop sleeve','laptop sleeves'), 'stand': ('phone stand','phone stands','mobile stand','mobile stands'),
 'laptop-stand': ('laptop stand','laptop stands'),
 'umbrella': ('umbrella','umbrellas'), 'wallet': ('wallet','wallets'),
 'washbag': ('washbag','washbags','wash bag','toiletry bag','toiletry bags'),
})

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def validate_catalog(catalog, family_ids):
    if type(catalog) is not dict or not {'products','variants','vectors'} <= set(catalog):
        raise ValueError('Explicit source products, real variants and native vectors required')
    products=catalog['products']; variants=catalog['variants']; vectors=catalog['vectors']
    if not 16 <= len(products) <= 128 or len(family_ids)!=16 or len(set(family_ids))!=16:
        raise ValueError('Original16 unique family routes and bounded expanded catalog required')
    ids=[p['productId'] for p in products]
    if len(set(ids))!=len(ids) or any(type(x) is not str or not x for x in ids):
        raise ValueError('Unique real product IDs required')
    if tuple(vectors.shape)!=(len(products),4096) or not torch.isfinite(vectors).all() or not torch.allclose(vectors.norm(dim=1),torch.ones(len(products)),atol=1e-4):
        raise ValueError('One genuine normalized finite4096 native public vector per product required')
    allvariants=[]
    for i,p in enumerate(products):
        if p.get('parentFamily') not in family_ids or not p.get('variants'):
            raise ValueError('Each source product needs explicit original parentFamily and real variants')
        if type(p.get('intentTags')) is not list or not p['intentTags'] or set(p['intentTags'])-INTENT_ALIASES.keys():
            raise ValueError('Explicit reviewed supported product-type eligibility tags required')
        for key in ('name','category','productId'):
            if type(p.get(key)) is not str or not p[key]: raise ValueError('Public product identity/category required')
        if type(p.get('itemPrice')) not in (int,float) or not 0<=p['itemPrice']<1e7:
            raise ValueError('Finite nonnegative real item price required')
        if type(p.get('searchTerms')) is not list or any(type(x)is not str for x in p['searchTerms']):
            raise ValueError('Public search terms required')
        if [v['colorName'] for v in p.get('colorVariants',[])] != [v['colorName'] for v in p['variants']]:
            raise ValueError('Matching source color guards/variant order required')
        for v in p['variants']:
            if not v.get('variantId') or not v.get('colorName') or not v.get('images'):
                raise ValueError('Actual variant/color/image metadata required')
            allvariants.append({**v,'productIndex':i})
    if allvariants!=variants or len({v['variantId'] for v in variants})!=len(variants):
        raise ValueError('Exact unique source product/variant expansion order required')
    return catalog


def checked_request(value, products):
    # Original public/event checks against the EXPANDED actual catalog first.
    request,policy,context=checked_context_request(value,products)
    def bounded(x):
        if isinstance(x,str) and len(x)>500: raise ValueError('All endpoint text <=500 characters; no clipping')
        if isinstance(x,list):
            for y in x: bounded(y)
        if isinstance(x,dict):
            for y in x.values(): bounded(y)
    bounded(value)
    persona=context['persona']
    if len(persona['background'])>6 or len(persona['colorPreferences'])>4 or len(context['publicHistory'])>12:
        raise ValueError('Public context bounds: background6/colors4/history12')
    return request,policy,context


def family_profile_view(value, family_catalog):
    """Keep literal new-SKU observations; no false mapping to an old store SKU.

    New source SKU history was checked against the expanded catalog beforehand.
    The original16 family prompt sees it as an external-to-family-catalog public
    observation with the exact actual item name/category/color/time preserved.
    """
    result=deepcopy(value); known={p['id'] for p in family_catalog}; converted=0
    for event in result.get('publicHistory',[]):
        if event.get('productId') is not None and event['productId'] not in known:
            event.pop('productId',None); event.pop('variantId',None); event['external']=True; converted+=1
    return result,{'newSkuObservationsRepresentedOutsideOriginalFamilyCatalog':converted,
        'literalObservedNameCategoryColorTimePreserved':True,'newSkuPretendedToBeTrainedRoute':False}


def recognized_intents(query):
    found=[]; matches=[]
    for tag,aliases in INTENT_ALIASES.items():
        for alias in sorted(aliases,key=len,reverse=True):
            for match in re.finditer(r'(?<!\w)'+re.escape(alias)+r'(?!\w)',query,re.I):
                # Explicit negated product nouns do not create positive intent.
                before=query[max(0,match.start()-32):match.start()]
                if re.search(r"\b(?:not|no|avoid|without|don['’]?t want)\s+(?:a\s+|any\s+)?$",before,re.I): continue
                if tag not in found: found.append(tag)
                matches.append({'tag':tag,'literal':match.group(),'span':[match.start(),match.end()]})
    # A specific compound product noun dominates its contained generic noun.
    matches=[m for m in matches if not any(n['span'][0]<=m['span'][0] and n['span'][1]>=m['span'][1] and n['span']!=m['span'] for n in matches)]
    found=list(dict.fromkeys(m['tag'] for m in matches))
    return found,matches


def strict_eligible_order(scores,request,policy,products):
    # Existing hard budgets/typed constraints/exact identity are unchanged.
    order,guards=guarded_order(scores,request,policy,products)
    tags,matches=recognized_intents(request['query'])
    allowed={i for i,p in enumerate(products) if not tags or set(tags)&set(p['intentTags'])}
    identity=guards['exactNamePolicy']
    matched_identity={i for i in identity['matchedProductIndices'] if i in order} if identity['applied'] else set()
    allowed.update(matched_identity)  # Identity cannot override prior hard guards.
    filtered=[i for i in order if i in allowed]
    guards['specificSearchEligibility']={'version':VERSION,'recognizedIntentTags':tags,'literalMatches':matches,
        'policy':'positive explicit product nouns: union of declared source tags; eligibility before learned ordering; no unsuitable padding',
        'eligibleIndices':sorted(allowed),'excludedProductIds':[p['productId'] for i,p in enumerate(products) if i not in allowed],
        'noRecognizedSpecificNounMeansBroadBrowse':not tags,'learnedScoresUsedForEligibility':False,'hardEligibleExactIdentityPreserved':sorted(matched_identity)}
    return filtered,guards


def expanded_scores(v5,lexical,p_specific,products,family_ids):
    if tuple(v5.shape)!=(len(products),) or tuple(lexical.shape)!=(16,) or not torch.isfinite(v5).all() or not torch.isfinite(lexical).all():
        raise ValueError('Finite genuine N-catalog scores and original16 family logits required')
    if type(p_specific) not in (int,float) or not 0<=p_specific<=1: raise ValueError('Valid learned query gate required')
    indices=torch.tensor([family_ids.index(p['parentFamily']) for p in products])
    counts=torch.bincount(indices,minlength=16).clamp_min(1)
    family=lexical.softmax(-1)[indices]/counts[indices]
    return (1-p_specific)*family+p_specific*(v5+.05*family),family


class ExpandedRankAdapter:
    """Compose an already attached original16 runtime with expanded real facts.

    New resident engine supplies session.head (existing v5 learned projections)
    and a source-verified expanded catalog; no old fixed16 loader is modified.
    """
    def __init__(self,runtime,catalog,*,catalog_sha256,source_pins=None):
        self.runtime=runtime; self.family_ids=list(runtime.manifest['productIds'])
        self.catalog=validate_catalog(catalog,self.family_ids); self.catalog_sha256=catalog_sha256
        self.pins={str(Path(__file__)):sha(__file__),**(source_pins or {})}
        self.colors,self.indices=source_colors(catalog)

    def assert_current(self):
        self.runtime.assert_current()
        if any(sha(p)!=s for p,s in self.pins.items()): raise ValueError('Expanded rank input/source changed')

    def rank(self,value,session,copy_adapter,*,cancelled=lambda:False):
        from .store_demo_engine import RequestCancelled
        self.assert_current(); runtime=self.runtime
        if runtime.bb is not session.bb or runtime.ownership.copy is not copy_adapter: raise ValueError('Shared verified native owner required')
        request,policy,context=checked_request(value,self.catalog['products'])
        if cancelled(): raise RequestCancelled('Cancelled before expanded ranking')
        querytext=request_text(ranking_input(request)); runtime.encoding_calls.clear()
        query,native=runtime.encode_base(querytext)
        family_value,historyproof=family_profile_view(value,runtime.public_catalog)
        row=runtime.public_row(family_value); lexical,decision=runtime.lexical_scores(row,cancelled=cancelled)
        with torch.no_grad():
            v5=session.head.probabilities('catalog',self.catalog['vectors']) @ session.head.probabilities('query',query)[0]
            p=float(runtime.query_intent.p_specific(query))
            scores,familyscores=expanded_scores(v5,lexical,p,self.catalog['products'],self.family_ids)
        order,guards=strict_eligible_order(scores.tolist(),request,policy,self.catalog['products'])
        if cancelled(): raise RequestCancelled('Cancelled before expanded source variants')
        # Acquire cached/fresh neutral public profile4096 through existing branch,
        # but score NEW source colors/catalog with its unchanged learned weights.
        colorbranch=runtime.color; oldcatalog,oldcolors,oldindices=colorbranch.catalog,colorbranch.colors,colorbranch.indices
        try:
            colorbranch.catalog=self.catalog; colorbranch.colors=self.colors; colorbranch.indices=self.indices
            variants,colorproof=colorbranch.scores(value,runtime.encode_base)
        finally:
            colorbranch.catalog, colorbranch.colors,colorbranch.indices=oldcatalog,oldcolors,oldindices
        cards=[]
        for i in order[:6]:
            choices=[(j,v) for j,v in enumerate(self.catalog['variants']) if v['productIndex']==i and all(r['attribute']!='color' or r['value']==v['colorName'] for r in request['hardRequirements'])]
            if not choices: raise ValueError('No actual variant satisfies hard colors')
            _,v=max(choices,key=lambda pair:(float(variants[pair[0]]),-pair[0])); product=self.catalog['products'][i]
            cards.append({'productIndex':i,'facts':{'name':product['name'],'category':product['category'],'itemPrice':product['itemPrice'],'selectedColor':v['colorName']},
                'action':{'productId':product['productId'],'variantId':v['variantId'],'image':deepcopy(v['images'][0])}})
        self.assert_current()
        if cancelled(): raise RequestCancelled('Cancelled after expanded ranking')
        querytrace=deepcopy(runtime.encoding_calls[0]) if runtime.encoding_calls else {}
        queryproof={'cacheHit':bool(querytrace.get('queryFeatureCacheHit',False)),
            'nativeBackboneForwardCalls':native.get('backboneForwardCalls',0),'native':native}
        return request,policy,order,guards,cards,{'version':VERSION,'catalogSha256':self.catalog_sha256,
            'catalogProducts':len(self.catalog['products']),'scores':scores.tolist(),'v5Scores':v5.tolist(),
            'familyPriorScores':familyscores.tolist(),'lexicalRouteLogits16':lexical.tolist(),'learnedQuerySpecificProbability':p,
            'familyPolicy':'original16 LoRA family probability divided across real family members; no new-ID training claim',
            'fusionFormula':'(1-p)*familyPrior+p*(v5N+0.05*familyPrior)',
            'queryFeature':queryproof,'queryEncoding':querytrace,'queryEncodingCalls':deepcopy(runtime.encoding_calls),'rankDecision':decision,'colorFeature':colorproof,
            'historyFamilyView':historyproof,'newProductsTrainedByLoRA':False,'freshNeuralCopyRequired':True,
            'rankAdapterAndCopyAdapterMixed':False,'modelPromotion':False}
