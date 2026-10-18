"""Pure public lexical-choice inputs, no native model, labels or training.

Same rendering as the frozen lexical pilot; explicit new history bound0–12.
Case IDs remain metadata and do not enter inference prompts.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
from . import store_qwen_lexical_choice_pilot_v1 as pilot

VERSION='public-qwen-lexical-choice-inputs.v2'
MAX_TOKENS=2048
CATALOG=pilot.HERE/'synthetic_history_v1/catalog-binding.json'
CATALOG_FILE_SHA256='bdd7e2f5f5bc41c147830b5ecd03b550794a237ad633d4e4233688ec223b8450'
CATALOG_FACTS_SHA256='ec36e27d6a5054c72a8552bbe48c7a5822769a6673d486eaf93fb3d7b7af490d'

def load_public_catalog():
    if pilot.sha(CATALOG)!=CATALOG_FILE_SHA256:raise ValueError('Frozen actual public catalog differs')
    return json.loads(CATALOG.read_text())['products']

def checked_catalog(catalog):
    if pilot.digest(catalog)!=CATALOG_FACTS_SHA256:raise ValueError('Canonical actual16products/48variants/facts/order required')
    return catalog

def _object(value,required,optional=()):
    if type(value) is not dict or not set(required)<=value.keys() or set(value)-set(required)-set(optional):raise ValueError('Missing, unknown or private public-input fields')
    return value

def _text(value):
    if type(value) is not str or not value.strip() or len(value)>600:raise ValueError('Bounded nonempty public text required')
    return value

def _texts(value,maximum):
    if type(value) is not list or len(value)>maximum:raise ValueError('Bounded public text array required')
    return [_text(x) for x in value]

def checked_request(value,catalog):
    checked_catalog(catalog)
    request=_object(value,{'query'},{'itemBudget','hardRequirements','persona','publicHistory'})
    query=_text(request['query']);budget=request.get('itemBudget')
    if budget is not None and (type(budget) not in (int,float) or not math.isfinite(budget) or budget<=0):raise ValueError('Finite positive item budget required')
    persona=_object(request.get('persona',{'background':[],'colorPreferences':[]}),{'background','colorPreferences'})
    background=_texts(persona['background'],6);preferences=_texts(persona['colorPreferences'],4)
    requirements=request.get('hardRequirements',[])
    if type(requirements) is not list or len(requirements)>12:raise ValueError('Bounded hard requirements required')
    for requirement in requirements:
        _object(requirement,{'attribute','value'});_text(requirement['attribute']);_text(requirement['value'])
    events=request.get('publicHistory',[])
    if type(events) is not list or not 0<=len(events)<=12:raise ValueError('Explicit public history bound0–12')
    products={p['id']:p for p in catalog};history=[];seen=set();previous=None
    for event in events:
        _object(event,{'eventId','eventType','itemName','category','occurredAt','source'},{'productId','variantId','selectedColor','external'})
        for key in ('eventId','itemName','category','occurredAt'):_text(event[key])
        if event['eventType'] not in ('view','cart','purchase') or event['source'] not in ('demo-seed','storefront-action'):raise ValueError('Unknown public event type/source')
        try:when=datetime.fromisoformat(event['occurredAt'].replace('Z','+00:00'))
        except (ValueError,TypeError) as error:raise ValueError('ISO public event time required') from error
        if when.tzinfo is None or (previous is not None and when<previous) or event['eventId'] in seen:raise ValueError('Ordered timezone-aware times and unique event IDs required')
        previous=when;seen.add(event['eventId'])
        if 'external' in event and event['external'] is not True:raise ValueError('External flag must be true')
        if 'selectedColor' in event:_text(event['selectedColor'])
        if event.get('external'):
            if 'productId' in event or 'variantId' in event:raise ValueError('External item cannot claim store SKU')
        else:
            product=products.get(event.get('productId'));variant=next((v for v in product['variants'] if v['id']==event.get('variantId')),None) if product else None
            if not variant or (event['itemName'],event['category'],event.get('selectedColor'))!=(product['name'],product['category'],variant['colorName']):raise ValueError('Stale actual store name/category/variant/color binding')
        history.append({k:deepcopy(event[k]) for k in ('eventType','itemName','category','selectedColor','occurredAt','external') if k in event})
    return {'query':query,'itemBudget':budget,'hardRequirements':deepcopy(requirements),'background':background,'colorPreferences':preferences,'recentHistory':history}

def build_messages(request,catalog,tokenizer,order='canonical'):
    if order not in ('canonical','reversed'):raise ValueError('Explicit canonical or reversed catalog order required')
    profile=checked_request(request,catalog);routes=pilot.routing_tokens(tokenizer,catalog)
    by_product={r['productId']:r for r in routes};products=catalog if order=='canonical' else list(reversed(catalog))
    choices=[dict(by_product[p['id']]) for p in products]
    public_catalog=[f"{c['code']}: {p['name']} | category: {p['category']} | item price: ${p['price']:.2f} | available colors: {', '.join(v['colorName'] for v in p['variants'])}" for c,p in zip(choices,products,strict=True)]
    observations=["Query: "+profile['query'],"Item budget: "+(f"${profile['itemBudget']:.2f}" if profile['itemBudget'] is not None else 'not specified'),
        "Hard requirements: "+pilot.canonical(profile['hardRequirements']),"Public background:"]
    observations += ['- '+line for line in profile['background']]
    observations += ["Stated soft color preferences: "+', '.join(profile['colorPreferences']),"Recent public history, oldest first:"]
    for event in profile['recentHistory']:
        line=f"- {event['eventType']} | {event['itemName']} | category: {event['category']}"
        if 'selectedColor' in event:line+=' | color: '+event['selectedColor']
        line+=' | occurred: '+event['occurredAt']+' | '+('external item, no store SKU' if event.get('external') else 'actual store item')
        observations.append(line)
    user='Public catalog (routing labels are product-bound, independent of order):\n'+'\n'.join(public_catalog)+'\nPublic shopper observations:\n'+'\n'.join(observations)+'\nSelect the best listed product routing label.'
    return [{'role':'system','content':pilot.SYSTEM},{'role':'user','content':user}],choices

def build_row(request,catalog,tokenizer,case_id,order):
    _text(case_id)
    messages,choices=build_messages(request,catalog,tokenizer,order)
    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)+pilot.CONFIG['assistantPrefix']
    ids=tokenizer.encode(prompt,add_special_tokens=False)
    if not 0<len(ids)<=MAX_TOKENS:raise ValueError(f'Whole public prompt has{len(ids)}tokens; no truncation')
    for route in choices:
        if tokenizer.encode(prompt+route['continuation'],add_special_tokens=False)!=ids+[route['tokenId']]:raise ValueError('Not a stable whole single next-token boundary')
    # Exact literal prefix ends before the first public query; never inferred
    # from an evaluation profile or cut at an approximate token count.
    boundary='Public shopper observations:\n'
    cut=prompt.index(boundary)+len(boundary)
    prefix_text=prompt[:cut];prefix_ids=tokenizer.encode(prefix_text,add_special_tokens=False)
    if not prefix_ids or ids[:len(prefix_ids)]!=prefix_ids:raise ValueError('Literal catalog-prefix token boundary merges; refuse approximate/spliced prefix')
    return {'caseId':case_id,'order':order,'publicRequestSha256':pilot.digest(request),'prompt':prompt,'promptSha256':pilot.digest(prompt),
        'tokenIds':ids,'tokenIdsSha256':pilot.digest(ids),'tokens':len(ids),'choices':choices,
        'catalogPrefixTokenIds':prefix_ids,'catalogPrefixTextSha256':pilot.digest(prefix_text),'catalogPrefixTokens':len(prefix_ids),
        'catalogPrefixBoundary':'exact literal Public shopper observations:newline, immediately before Query:',
        'catalogPrefixText':prefix_text,
        'publicPurchaseProductIds':[e['productId'] for e in request.get('publicHistory',[]) if e['eventType']=='purchase' and not e.get('external')]}

def source_pins():
    return {str(Path(__file__)):pilot.sha(__file__),str(Path(pilot.__file__)):pilot.sha(pilot.__file__),str(CATALOG):CATALOG_FILE_SHA256}
