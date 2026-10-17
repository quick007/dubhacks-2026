"""Versioned model-only page-plan interface. No storefront or private-file access."""
from __future__ import annotations
import hashlib
import json
from .boundary import guard_reasons, visible_history
from .fusion import supported_preferences, _field_claim_errors

VERSION = 'page-plan.v1'


def fact_id(product_id, tag):
    return f'fact:{product_id}:tag:{tag}'


def history_reasons(request, catalog, product):
    """References to observed actions, not claims about latent user intent."""
    reasons = {}
    for event in visible_history(request, catalog):
        # A weak/shared-session view is not sufficient evidence for a badge.
        if event.get('type') != 'save' or event.get('confidence', 0) < .7:
            continue
        common = set(catalog[event['productId']].get('tags', [])) & set(product['tags'])
        for tag in sorted(common):
            identity = [event['at'], event['productId'], event['type'], tag]
            rid = 'observed:' + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
            reasons[rid] = {'reasonId': rid, 'at': event['at'], 'eventType': 'save',
                            'confidence': event['confidence'], 'factTag': tag,
                            'meaning': 'Observed a save of an item with this attribute; current intent unknown.'}
    return reasons


def field(text=None, source='pending'):
    return {'text': text, 'source': source}


def make_page_plan(request, catalog, decision, card_count):
    if card_count not in (1, 3, 6):
        raise ValueError('Requested card count must be 1, 3 or 6')
    ranked = [row['productId'] for row in decision['rankedProducts']
              if not guard_reasons(request, catalog[row['productId']])
              and any(request.get('presentationCompatibility', {}).get(row['productId'], {}).values())][:card_count]
    cards = []
    for pid in ranked:
        product = catalog[pid]
        compatible = [v for v in product['presentationVariants']
                      if request['presentationCompatibility'].get(pid, {}).get(v['id'], False)]
        action = decision.get('presentationAction')
        variant = next((v for v in compatible if action and action['productId']==pid and action['variantId']==v['id']), compatible[0])
        explicit = set(supported_preferences(request, product))
        reasons = history_reasons(request, catalog, product)
        tags = sorted(explicit | {reason['factTag'] for reason in reasons.values()})
        cards.append({'productId': pid, 'imageCaptionId': variant['id'],
                      'approvedFactIds': [fact_id(pid, tag) for tag in tags],
                      'observedHistoryReasonIds': sorted(reasons),
                      'badgeDecision': 'GENERATE' if tags else 'NONE',
                      'title': field(), 'badge': field(None, 'none' if not tags else 'pending')})
    sections = []
    # Fixed grouping policy exposes structural decisions without pretending a
    # learned section-count head exists. Ordering originates in trained rankings.
    for index, group in enumerate((cards[:3], cards[3:])):
        if not group: continue
        sections.append({'sectionId': f'section-{index+1}',
                         'type': 'recommendations' if index==0 else 'alternatives',
                         'productOrder': [p['productId'] for p in group],
                         'products': group, 'heading': field(), 'copy': field(None, 'none')})
    return {'schemaVersion': VERSION, 'synthetic': True, 'recordId': request['recordId'],
            'requestedCardCount': card_count, 'duplicatePolicy': 'forbid',
            'sectionOrder': [s['sectionId'] for s in sections], 'sections': sections,
            'shortfall': card_count-len(cards),
            'method': {'structure': 'fixed max-three-card grouping; not a trained layout policy',
                       'selectionAndOrder': 'trained shared-backbone product scores plus known-constraint mask',
                       'evidence': 'deterministic observable projection; not a learned evidence planner',
                       'headingsAndTitles': 'pending actual same-backbone generation'}}


def validate_page_plan(plan, request, catalog, require_complete=False):
    """Semantic validator used in addition to the consumer TypeScript contract."""
    if plan.get('schemaVersion') != VERSION or plan.get('synthetic') is not True:
        raise ValueError('Unsupported page plan version or missing synthetic label')
    if plan.get('requestedCardCount') not in (1,3,6) or plan.get('duplicatePolicy') not in ('forbid','explicit-allow'):
        raise ValueError('Invalid card count or duplicate policy')
    sections=plan['sections']; ids=[s['sectionId'] for s in sections]
    if len(set(ids))!=len(ids) or set(plan['sectionOrder'])!=set(ids) or len(plan['sectionOrder'])!=len(ids):
        raise ValueError('Section order must reference every unique section exactly once')
    all_ids=[]
    for section in sections:
        if section['type'] not in ('recommendations','alternatives'):
            raise ValueError('Unknown section type')
        products=section['products']; pids=[p['productId'] for p in products]
        if len(pids)!=len(set(pids)) or set(section['productOrder'])!=set(pids) or len(section['productOrder'])!=len(pids):
            raise ValueError('Product order must reference every unique section product exactly once')
        all_ids+=pids
        for card in products:
            pid=card['productId']
            if pid not in request['candidateProductIds'] or pid not in catalog or guard_reasons(request,catalog[pid]):
                raise ValueError('Unknown/unavailable/ineligible product')
            product=catalog[pid]
            if not request['presentationCompatibility'].get(pid,{}).get(card['imageCaptionId'],False) or card['imageCaptionId'] not in {v['id'] for v in product['presentationVariants']}:
                raise ValueError('Unknown/incompatible image-caption ID')
            if len(set(card['approvedFactIds'])) != len(card['approvedFactIds']) or not set(card['approvedFactIds']) <= {fact_id(pid,t) for t in product['tags']}:
                raise ValueError('Invented/duplicate product fact')
            reasons=history_reasons(request,catalog,product)
            if len(set(card['observedHistoryReasonIds'])) != len(card['observedHistoryReasonIds']) or not set(card['observedHistoryReasonIds']) <= set(reasons):
                raise ValueError('Unobserved/duplicate history reason')
            supported=set(supported_preferences(request,product)) | {reasons[r]['factTag'] for r in card['observedHistoryReasonIds']}
            if card['badgeDecision'] not in ('NONE','GENERATE') or (card['badgeDecision']=='GENERATE' and not supported):
                raise ValueError('Badge generation needs observed support')
            if card['badgeDecision']=='NONE' and card['badge']['text'] is not None:
                raise ValueError('NONE must have empty badge')
            for name in ('title','badge'):
                value=card[name]
                if value['source'] not in ('pending','generated','rejected','none','approved-fallback'):
                    raise ValueError('Unknown generation provenance')
                if value['text'] is not None and (not isinstance(value['text'],str) or len(value['text'])>90 or _field_claim_errors(value['text'],product,catalog)):
                    raise ValueError('Unfactual/oversized generated card field')
                if require_complete and value['source']=='pending':raise ValueError('Incomplete generated card field')
        # Heading attributes apply to every card; only common tags are allowed.
        common=set.intersection(*(set(catalog[pid]['tags']) for pid in pids)) if pids else set()
        proxy={'title':'Everyday travel choices','tags':sorted(common),'category':'backpacks'}
        for name in ('heading','copy'):
            value=section[name]
            if value['source'] not in ('pending','generated','rejected','none'):
                raise ValueError('Unknown section generation provenance')
            if value['text'] is not None and (len(value['text'])>90 or _field_claim_errors(value['text'],proxy,catalog)):
                raise ValueError('Unsupported section claim')
            if require_complete and value['source']=='pending':raise ValueError('Incomplete section field')
    if plan['duplicatePolicy']=='forbid' and len(all_ids)!=len(set(all_ids)):
        raise ValueError('Duplicate product across sections')
    if len(all_ids)>plan['requestedCardCount'] or plan['shortfall']!=plan['requestedCardCount']-len(all_ids):
        raise ValueError('Incorrect card count/shortfall')
    return True
