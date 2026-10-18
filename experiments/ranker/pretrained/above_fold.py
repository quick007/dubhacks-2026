"""Prospective CPU-only first-section payload contract; no UI or model execution."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

WORKLOAD = Path(__file__).with_name('ABOVE_FOLD_WORKLOAD.json')
CORE = ('S1', 'T1', 'B1', 'T2', 'B2', 'T3', 'B3')
REQUIRED = (*CORE, 'C1', 'C2', 'C3')
CAPS = {'S': 16, 'T': 24, 'B': 8, 'C': 32}


def payload_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_workload():
    return json.loads(WORKLOAD.read_text())


def project_first_section(plan):
    """Project a distinct validation scope; never change the full page plan."""
    view = copy.deepcopy(plan)
    first = view['sections'][0]
    if view['sectionOrder'][0] != first['sectionId']:
        raise ValueError('Ambiguous first section array/order binding')
    if len(first['products']) != 3 or first['productOrder'] != [c['productId'] for c in first['products']]:
        raise ValueError('Fixed view requires three explicitly ordered cards')
    view['sections'] = [first]; view['sectionOrder'] = [first['sectionId']]
    view['requestedCardCount'] = 3; view['shortfall'] = 0
    first['heading'] = {'text': None, 'source': 'pending'}
    first['copy'] = {'text': None, 'source': 'none'}
    for card in first['products']:
        card['title'] = {'text': None, 'source': 'pending'}
        card['badge'] = {'text': None, 'source': 'none' if card['badgeDecision'] == 'NONE' else 'pending'}
    view['method']['validationScope'] = 'first-section-payload.v1; full page remains separate'
    return view


def validate_view(rows, plan, request, catalog, payload, *, exact_product_ids=()):
    """Validate only required visible rows and the actual completed payload.

    Lower rows are irrelevant to this scope. Strict current title identity stays
    mandatory even though originalName is separately visible. C rows are a new
    contract and use the existing fusion copy factuality/length checks.
    """
    from .fusion import validate_copy, _normalized
    from .page_plan import validate_page_plan
    from .scheduler_identity_experiment import assemble_values, FORBIDDEN_SEPARATORS
    workload = load_workload(); errors = []; selected = {}
    try:
        view = project_first_section(plan)
        validate_page_plan(view, request, catalog)
    except (KeyError, ValueError, IndexError) as exc:
        return {'valid': False, 'rejections': ['invalid-visible-plan:' + str(exc)], 'payloadSha256': None}
    for row in rows:
        labels = row.get('fields', [])
        if not any(f in REQUIRED for f in labels): continue
        if len(labels) != 1:
            errors.append('ambiguous-visible-row'); continue
        field = labels[0]
        if field in selected: errors.append(field + ':duplicate-row'); continue
        selected[field] = row
        raw = row.get('text')
        if not isinstance(raw, str) or not raw.strip(): errors.append(field + ':empty-value'); continue
        if any(c in raw for c in FORBIDDEN_SEPARATORS): errors.append(field + ':multiline-value')
        if re.search(r'(?<![A-Za-z0-9])[STBC]\d+\s*:', unicodedata.normalize('NFKC', raw), re.I):
            errors.append(field + ':generated-field-label')
        if row.get('finishReason') != 'stop': errors.append(field + ':unfinished')
        if row.get('maxTokens') != CAPS[field[0]]: errors.append(field + ':changed-cap')
        if row.get('kind') != {'S': 'heading', 'T': 'title', 'B': 'badge', 'C': 'copy'}[field[0]]:
            errors.append(field + ':changed-kind')
    errors += [f + ':missing-row' for f in REQUIRED if f not in selected]
    core = None
    if all(f in selected and isinstance(selected[f].get('text'), str) for f in CORE):
        core = assemble_values([copy.deepcopy(selected[f]) for f in CORE], view, request, catalog)
        errors += ['core:' + e for e in core['quality']['rejections']]
    cards = view['sections'][0]['products']
    copies = {}
    for i, card in enumerate(cards, 1):
        field = f'C{i}'; title = selected.get(f'T{i}', {}).get('text'); row = selected.get(field)
        if not row or not isinstance(row.get('text'), str) or not isinstance(title, str): continue
        product = catalog[card['productId']]
        support = [f.rsplit(':', 1)[1] for f in card['approvedFactIds']]
        result = validate_copy('Title: ' + title + '\nCopy: ' + row['text'], product, catalog,
            'stop' if row.get('finishReason') == selected[f'T{i}'].get('finishReason') == 'stop' else 'length', support)
        if not result['copyAccepted']: errors += [field + ':' + e for e in result['fieldRejections']['copy']]
        approved = [product.get('description', ''), product.get('category', ''),
                    *(v.get('imageDescription', '') for v in product['presentationVariants'])]
        if _normalized(row['text']) in {_normalized(t) for t in approved if t}:
            errors.append(field + ':copied-catalog-body')
        copies[field] = result
    if not isinstance(payload, dict):
        errors.append('completed-payload-required'); payload = {}
    if payload.get('schemaVersion') != 'first-section-payload.v1': errors.append('payload-version')
    if payload.get('sectionId') != view['sections'][0]['sectionId']: errors.append('payload-section-binding')
    if payload.get('heading') != selected.get('S1', {}).get('text'): errors.append('payload-heading-mismatch')
    if payload.get('sectionIntro') != workload['staticVisibleMetadata']['sectionIntro']:
        errors.append('payload-section-intro')
    values = payload.get('cards', [])
    if not isinstance(values, list) or len(values) != 3:
        errors.append('payload-three-cards-required'); values = []
    for i, (card, value) in enumerate(zip(cards, values), 1):
        if not isinstance(value, dict): errors.append(f'card{i}:invalid-payload'); continue
        product = catalog[card['productId']]
        image = next(v for v in product['presentationVariants'] if v['id'] == card['imageCaptionId'])
        if not isinstance(image.get('imageKey'), str) or not image['imageKey']:
            errors.append(f'card{i}:missing-image-key')
        expected = {'productId': card['productId'], 'imageCaptionId': card['imageCaptionId'],
            'imageKey': image['imageKey'], 'originalName': product['title'], 'price': product['price'],
            'shippingCost': product['shippingCost'], 'catalogCategory': product['category'],
            'title': selected.get(f'T{i}', {}).get('text'), 'bodyCopy': selected.get(f'C{i}', {}).get('text'),
            'badge': None if selected.get(f'B{i}', {}).get('text') == 'NONE' else selected.get(f'B{i}', {}).get('text'),
            'addToCartLabel': workload['staticVisibleMetadata']['addToCartLabel']}
        for key, expected_value in expected.items():
            if key not in value or value[key] != expected_value: errors.append(f'card{i}:payload-{key}-mismatch')
        if isinstance(value.get('price'), bool) or isinstance(value.get('shippingCost'), bool):
            errors.append(f'card{i}:invalid-price-type')
    visible_ids = {c['productId'] for c in cards}
    for pid in exact_product_ids:
        if pid not in catalog or pid not in visible_ids: errors.append('resolved-exact-name-product-missing:' + str(pid))
    query = ' ' + _normalized(request.get('context', {}).get('task', '')) + ' '
    exact_names = {p['title'] for p in catalog.values() if ' ' + _normalized(p['title']) + ' ' in query}
    visible_names = {catalog[c['productId']]['title'] for c in cards}
    if not exact_names <= visible_names: errors.append('explicit-complete-catalog-name-missing-from-view')
    return {'valid': not errors, 'rejections': list(dict.fromkeys(errors)),
        'scope': 'first section and three complete cards; no full-page claim',
        'fullPageRequestedCardCount': plan.get('requestedCardCount'), 'fullPageValidated': False,
        'requiredFields': list(REQUIRED), 'coreQuality': core['quality'] if core else None,
        'copyChecks': copies, 'payloadSha256': payload_hash(payload),
        'belowFoldRowsAffectValidity': False, 'titleIdentityPolicy': 'unchanged current policy; subtitle relaxation not enabled',
        'renderedImageOrBrowserPaintMeasured': False}


def payload_timing(validation, rows, *, planning_prefix_ms, receipt=None):
    """All required completions plus actual validated/serialized payload receipt.

    Row completedMs is relative to native decode start. planning_prefix_ms is
    measured from copy-stage start to that same decode start, not a sum of
    overlapping phase timers. First-token times never enter this function.
    """
    def nonnegative(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    if not nonnegative(planning_prefix_ms): raise ValueError('Measured nonnegative pre-decode offset required')
    seen = {}; complete = True
    for row in rows:
        labels = row.get('fields', [])
        if len(labels) == 1 and labels[0] in REQUIRED:
            if labels[0] in seen: complete = False
            seen[labels[0]] = row.get('completedMs')
    complete = complete and set(seen) == set(REQUIRED) and all(nonnegative(v) for v in seen.values())
    lower = planning_prefix_ms + max(seen.values()) if complete else None
    result = {'measurementScope': 'copy-stage payload handoff; ranking and browser paint excluded',
        'allRequiredRowsCompleted': complete, 'lastRequiredTokenLowerBoundMs': lower,
        'copyStagePayloadReadyMs': None, 'fullRequestViewReadyMs': None,
        'payloadCompletionReceiptPresent': receipt is not None}
    if receipt is not None:
        if receipt.get('payloadSha256') != validation.get('payloadSha256'): raise ValueError('Receipt belongs to another payload')
        ready = receipt.get('validatedAndSerializedAtMs'); metadata = receipt.get('metadataReadyAtMs')
        if not nonnegative(ready) or not nonnegative(metadata): raise ValueError('Measured ready/metadata timestamps required')
        if lower is None or ready < max(lower, metadata): raise ValueError('Payload receipt predates required completion')
        if validation['valid']: result['copyStagePayloadReadyMs'] = ready
    return result
