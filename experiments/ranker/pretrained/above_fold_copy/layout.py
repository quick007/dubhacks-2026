"""CPU layout/assembly helpers; no model runtime is loaded or executed here."""
from __future__ import annotations
import copy
import json

from ..above_fold import CAPS, CORE, REQUIRED, payload_hash, project_first_section, validate_view
from ..branch_local_copy import RULES as STB_RULES, canonical, token_hash, _card_facts

VERSION = 'first-section-card-branch-copy-v1'
RULES = STB_RULES.replace('Public evidence: ',
    'Copy: write one NEW short factual plain sentence, 10-180 characters. '
    'Mention a supportedObservedTag in the copy itself when present; otherwise use only selected facts and ordinary framing. '
    'Do not copy the catalog description/category/image description. '
    'originalNameSubtitle must remain the exact selected identity in the completed payload; it does not relax title identity. '
    'Each field is independent: do not rely on another generated title, badge or copy value. '
    'Public evidence: ')


def build_layout(plan, request, catalog):
    from ..boundary import clean_product, visible_history
    from ..fusion import observable_request
    from ..page_plan import validate_page_plan
    view = project_first_section(plan); public = observable_request(request)
    validate_page_plan(view, public, catalog)
    cards = [_card_facts(i, card, catalog) for i, card in enumerate(view['sections'][0]['products'], 1)]
    for card in cards: card['originalNameSubtitle'] = card['identity']
    observations = [{**{k: event.get(k) for k in ('at', 'type', 'confidence', 'description', 'presentationStyle')},
        'productFacts': clean_product(catalog[event['productId']])}
        for event in visible_history(public, catalog)]
    root = {'asOf': public['asOf'], 'request': public['context'], 'observedHistory': observations}
    root_message = {'role': 'user', 'content': RULES + canonical(root)}
    universal = sorted(set.intersection(*(set(catalog[c['productId']]['tags']) for c in view['sections'][0]['products'])))
    section = {'index': 1, 'cardIndices': [1, 2, 3], 'universalTags': universal}
    rows = []
    for field in REQUIRED:
        prefix, i = field[0], int(field[1:]); kind = {'S': 'heading', 'T': 'title', 'B': 'badge', 'C': 'copy'}[prefix]
        facts = {'cards': cards, 'sections': [section]} if prefix == 'S' else {'cards': [cards[i - 1]]}
        target = f'section {i}' if prefix == 'S' else f'card {i}'
        messages = [{'role': 'user', 'content': 'Public bound evidence: ' + canonical(facts)},
            {'role': 'user', 'content': f'Write only the fresh {kind} value for {target} using its public evidence and {kind} rules. End after this one line.'}]
        rows.append({'rowId': f'{kind}-{i}', 'key': f'{kind}-{i}', 'fields': [field],
                     'kind': kind, 'maxTokens': CAPS[prefix], 'messages': messages})
    assert len(rows) == 10 and sum(r['maxTokens'] for r in rows) == 208
    return root_message, rows, {'layoutVersion': VERSION, 'root': root, 'cards': cards,
        'sections': [section], 'sameBackboneRequired': True, 'cardFactCacheReuseImplemented': False,
        'fullPageRequestedCardCount': plan['requestedCardCount'], 'fullPageValidated': False}


def token_inventory(tokenizer, root, rows):
    """Delegate exact root/suffix boundary checks; caller supplies CPU tokenizer."""
    from ..branch_local_copy import token_inventory as inventory
    result = inventory(tokenizer, root, rows)
    return {**result, 'layoutVersion': VERSION, 'rows': result['rows'], 'requiredRows': list(REQUIRED)}


def fork_root(root_state, root_message, expected_root_tokens, model, materialize=None):
    """Complete immutable-state fork, never prefix substitution/trim/splice.

    This CPU-neutral helper may be used later by a separately authorized native
    owner, who must materialize/evaluate leaves and prove alias/logit controls.
    """
    from ..parallel_decode import fork_complete_state
    if root_state.get('model') is not model or root_state.get('consumed'):
        raise ValueError('Require pristine root belonging to the exact loaded model')
    if root_state.get('messages') != [root_message] or root_state.get('tokens') != expected_root_tokens:
        raise ValueError('Root prompt/tokens differ; never splice a replacement prefix')
    if root_state.get('prefixSha256') != token_hash(expected_root_tokens):
        raise ValueError('Root token hash differs')
    return fork_complete_state(root_state, materialize)


def assemble_payload(rows, plan, request, catalog, *, require_native_receipts=False):
    """Join trusted values/metadata only; preserve all raw values without repair."""
    from ..above_fold import load_workload
    view = project_first_section(plan); selected = {}
    for row in rows:
        labels = row.get('fields', [])
        if len(labels) != 1 or labels[0] not in REQUIRED or labels[0] in selected:
            raise ValueError('Exactly one unique expected field per trusted row required')
        selected[labels[0]] = row
    if set(selected) != set(REQUIRED): raise ValueError('All ten required rows must exist; no NONE injection')
    static = load_workload()['staticVisibleMetadata']; cards = []
    for i, card in enumerate(view['sections'][0]['products'], 1):
        product = catalog[card['productId']]
        image = next(v for v in product['presentationVariants'] if v['id'] == card['imageCaptionId'])
        badge = selected[f'B{i}']['text']
        cards.append({'productId': card['productId'], 'imageCaptionId': card['imageCaptionId'],
            'imageKey': image['imageKey'], 'originalName': product['title'], 'price': product['price'],
            'shippingCost': product['shippingCost'], 'catalogCategory': product['category'],
            'title': selected[f'T{i}']['text'], 'bodyCopy': selected[f'C{i}']['text'],
            'badge': None if badge == 'NONE' else badge, 'addToCartLabel': static['addToCartLabel']})
    payload = {'schemaVersion': 'first-section-payload.v1', 'sectionId': view['sections'][0]['sectionId'],
        'heading': selected['S1']['text'], 'sectionIntro': static['sectionIntro'], 'cards': cards}
    validation = validate_view(rows, plan, request, catalog, payload)
    if require_native_receipts:
        for row in rows:
            raw = row.get('rawSampledTokenIds'); tokens = row.get('tokenIds'); stop = row.get('stopTokenId')
            ok = (isinstance(raw, list) and raw and isinstance(tokens, list)
                and all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in raw + tokens)
                and len(raw) <= row['maxTokens'] and isinstance(row.get('arSteps'), int)
                and not isinstance(row['arSteps'], bool) and row['arSteps'] == len(raw)
                and isinstance(stop, int) and not isinstance(stop, bool)
                and row.get('finishReason') == 'stop' and stop == raw[-1] and tokens == raw[:-1])
            if not ok:
                validation['valid'] = False
                validation['rejections'].append(row['fields'][0] + ':native-sampled-cap-or-EOS-receipt')
    # The copy branch explicitly conditions on observed support independently of
    # a generated title. This is additive prospective C policy, not old STB data.
    from ..fusion import _normalized
    import re
    for i, card in enumerate(view['sections'][0]['products'], 1):
        tags = [f.rsplit(':', 1)[1] for f in card['approvedFactIds']]
        raw = selected[f'C{i}']['text']
        if tags and not any(re.search(r'\b' + re.escape(_normalized(t)) + r'\b', _normalized(raw)) for t in tags):
            validation['valid'] = False; validation['rejections'].append(f'C{i}:body-independent-observed-support-missing')
    return {'payload': payload, 'validation': validation,
        'rawRows': copy.deepcopy(rows), 'payloadSha256': payload_hash(payload),
        'nativeReceiptsChecked': require_native_receipts, 'modelCallsByThisHelper': 0,
        'note': 'assembly is not generation; native execution/provenance must be proved by a future owner'}
