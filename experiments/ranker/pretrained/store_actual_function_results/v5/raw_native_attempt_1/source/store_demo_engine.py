"""Opt-in resident store demo: learned ranking and fresh, guarded native copy.

The old training/evaluation modules and their strict benchmark are unchanged.
This prospective demo gate withholds individual unsafe fields, never repairs
them. Ordinary metadata remains usable when a generated field is withheld.
Only the owning worker thread may construct or call ResidentDemoEngine.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import time

from .store_catalog_v1 import canonical, load_snapshot
from .store_copy_data_v2 import CAPS, KIND, RULES, field_rejections, native_receipt, validate_facts
from .store_function_adaptation import rank, validate_request
from .store_rank_session import action_bindings, exact_name_order

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RANK = Path('data/ranker/pretrained/actual-store-function-v3-attempt-1')
DEFAULT_COPY = Path('data/ranker/pretrained/actual-store-copy-fit-v2-attempt-1')
DEFAULT_OUTPUT = Path('data/ranker/pretrained/store-demo-v1')
NEUTRAL_COPY_REQUEST = {'query': 'Show a clear collection of store products.',
    'preferences': [], 'itemBudget': None, 'hardRequirements': []}
COPY_DISCLOSURE = ('Fresh native adapted titles, badges, copy and headings; conditioned only on '
    'selected public product facts and a neutral browsing request. Shopper query influences ranking, '
    'not generation wording. Unsafe generated fields are withheld as null, never repaired or templated.')
TIMING_SCOPE = ('Request receipt through fresh query encoding, guarded learned ranking and native copy '
    'validation/JSON serialization; excludes startup, browser paint and image transfer.')
BUDGET_CLAUSE = re.compile(
    r'\b(under|below|less than|up to|max(?:imum)?(?: budget)?|budget(?: of| is)?)\s*:?\s*\$(\d+(?:\.\d{1,2})?)(?![\d.])', re.I)


class RequestCancelled(RuntimeError):
    pass


def checked_request(value):
    """Bounded public schema; explicit typed constraints never become soft cues.

    Typed budget is inclusive. Text 'under/below/less than' is exclusive;
    'up to/max/budget' is inclusive, and only an explicit dollar amount is read.
    This is deliberately not a general natural-language constraint parser.
    """
    if type(value) is not dict or not set(value) <= {'query', 'itemBudget', 'hardRequirements'}:
        raise ValueError('Expected query, optional itemBudget and hardRequirements only')
    query = value.get('query')
    if type(query) is not str or not query.strip() or len(query) > 2000:
        raise ValueError('query must contain 1–2000 characters')
    if re.search(r'\b(?:delivered|including shipping|with shipping|delivery included|shipping included)\b', query, re.I) and (
            value.get('itemBudget') is not None or BUDGET_CLAUSE.search(query)):
        raise ValueError('Delivered-cost budgets are unsupported; itemBudget covers item price only')
    requirements = value.get('hardRequirements', [])
    if type(requirements) is not list or len(requirements) > 12:
        raise ValueError('At most 12 explicit hard requirements allowed')
    waterproof = False
    base_requirements = []
    for requirement in requirements:
        if (type(requirement) is not dict or set(requirement) != {'attribute', 'value'}
                or type(requirement['value']) is not str or len(requirement['value']) > 120):
            raise ValueError('Hard requirements require {attribute,value} strings')
        if requirement['attribute'] == 'waterproof':
            if requirement['value'].casefold() not in ('true', 'yes', 'required'):
                raise ValueError('waterproof supports only an explicit positive requirement')
            waterproof = True
        else:
            base_requirements.append(deepcopy(requirement))
    if re.search(r'\bin[ -]stock\b', query, re.I):
        base_requirements.append({'attribute': 'stock', 'value': 'in-stock'})
    size_match = re.search(r'\bsize\s+(XXS|XS|S|M|L|XL|XXL|\d{1,3}(?:\.\d)?)\b', query, re.I)
    if size_match:
        base_requirements.append({'attribute': 'size', 'value': size_match[1]})
    request = {'query': query.strip(), 'preferences': [], 'itemBudget': value.get('itemBudget'),
        'hardRequirements': base_requirements}
    validate_request(request)
    exclusive = False
    budget_source = 'typed-inclusive' if request['itemBudget'] is not None else 'absent'
    if request['itemBudget'] is None:
        matches = list(BUDGET_CLAUSE.finditer(query))
        if matches:
            # All explicit ceilings apply; use the most restrictive one.
            bounds = [(Decimal(m[2]), m[1].casefold() in ('under', 'below', 'less than')) for m in matches]
            amount, exclusive = min(bounds, key=lambda pair: (pair[0], not pair[1]))
            request['itemBudget'] = float(amount)
            budget_source = 'query-exclusive' if exclusive else 'query-inclusive'
    # Only an unnegated literal request for waterproof is recognized. No
    # water-resistant=>waterproof conversion and no material/size/stock guessing.
    for match in re.finditer(r'\bwaterproof\b', query, re.I):
        preceding = query[max(0, match.start() - 40):match.start()].casefold()
        if not re.search(r"\b(?:not|no|without|avoid|don['’]?t|do not)\b[^,.!?;]*$", preceding):
            waterproof = True
    policy = {'budgetSource': budget_source, 'budgetExclusive': exclusive,
        'waterproofRequired': waterproof,
        'constraintParserScope': 'typed hard constraints, explicit dollar ceilings and unnegated literal waterproof only; other free text is ranking preference'}
    return request, policy


def ranking_input(request):
    """TRAIN-like head input, with the entire public shopper query retained."""
    return {'query': 'Choose a product for my stated use.', 'preferences': [request['query']],
        'itemBudget': request['itemBudget'], 'hardRequirements': deepcopy(request['hardRequirements'])}


def guarded_order(scores, request, policy, products):
    result = rank(scores, request, products, 6)
    extra = []
    desired_colors = [r['value'] for r in request['hardRequirements'] if r['attribute'] == 'color']
    for product in products:
        reasons = []
        if (policy['budgetExclusive'] and request['itemBudget'] is not None
                and product['itemPrice'] >= request['itemBudget']):
            reasons.append('exclusive-item-budget')
        evidence = ' '.join([product['name'], product['category'], *product['searchTerms']])
        if policy['waterproofRequired'] and not re.search(r'\bwaterproof\b', evidence, re.I):
            reasons.append('waterproof-not-declared')
        if desired_colors and not any(all(c['colorName'] == value for value in desired_colors) for c in product['colorVariants']):
            reasons.append('no-single-variant-satisfies-all-colors')
        extra.append(reasons)
    result['eligibleRank'] = [i for i in result['eligibleRank'] if not extra[i]]
    identity_query = BUDGET_CLAUSE.sub('', request['query']).strip(' ,.;:-')
    identity_request = {**request, 'query': identity_query}
    order, identity = exact_name_order(identity_request, products, result)
    if identity['matchedProductIndices'] and not identity['applied']:
        order = []
        identity['ineligibleExactIdentityAbstention'] = True
    identity['policy'] = 'whole public product name after removing recognized dollar ceiling; hard guards first; abstain if exact identity ineligible'
    return order, {'base': result['guards'], 'additionalReasons': extra, 'exactNamePolicy': identity, **policy}


def demo_layout(cards):
    """Same v2 field-message pattern and caps, generalized to 1–6 real cards."""
    if type(cards) is not list or not 1 <= len(cards) <= 6:
        raise ValueError('One through six actual eligible cards required')
    common = {'role': 'user', 'content': RULES + canonical(NEUTRAL_COPY_REQUEST)}
    rows = []
    for start in range(0, len(cards), 3):
        fields = [f'S{start // 3 + 1}']
        fields += [f'{prefix}{i + 1}' for i in range(start, min(start + 3, len(cards))) for prefix in ('T', 'B', 'C')]
        for field in fields:
            prefix, index = field[0], int(field[1:])
            if prefix == 'S':
                facts = {'genericSection': index}
            else:
                facts = deepcopy(validate_facts(cards[index - 1]['facts']))
                if prefix == 'B':
                    facts = {'badgeEvidence': {'selectedColor': facts['selectedColor']}}
            kind = KIND[prefix]
            rows.append({'rowId': f'{kind}-{index}', 'key': f'{kind}-{index}', 'fields': [field],
                'kind': kind, 'maxTokens': CAPS[prefix], 'messages': [{'role': 'user',
                    'content': f'Write only the fresh {kind} value, then stop. Bound evidence: ' + canonical(facts)}]})
    return common, rows


def _span(text, fact):
    return bool(fact and re.search(r'(?<!\w)' + re.escape(fact) + r'(?!\w)', text, re.I))


# This is a prospective lightweight safety gate, not the fixed v2 grammar or a
# comprehensive semantic proof. Whole catalog categories are supported phrases;
# individual category words are not automatically independent product claims.
FRAMING = set(('a an the this that these those it its is are in of for at with and or '
    'your our their you to from on as has have item price priced costs cost shown '
    'show listed selected select selection collection choices choice picks pick '
    'color colour colored coloured tone shade finish version edition option '
    'browse browsing explore discover view current everyday store shop products '
    'product available comes featuring features now here display displayed '
    'includes include offering offered choose enjoy find meet per each only '
    'take look through assortment goods merchandise curated consider considering '
    'look looks presented present ready new simple more another other').split())
UNSUPPORTED = re.compile(r'\b(?:waterproof|water-resistant|cotton|canvas|steel|acetate|leather|'
    r'wool|nylon|polyester|organic|recycled|sustainable|eco|durable|premium|best|perfect|'
    r'comfortable|breathable|warm|cooling|insulated|leakproof|wireless|bluetooth|noise|'
    r'water|hydration|rain|hiking|travel|weekend|lunch|meal|music|coffee|tea|beach|'
    r'swimming|cables|grocery|groceries|reading|shipping|delivery|free|discount|sale|'
    r'save|savings|warranty|stock|sizes?|lightweight|heavy|capacity|liters?|litres?)\b', re.I)


def practical_rejections(field, text, card, products, *, row=None):
    """Allow natural framing of exact source facts; reject obvious bad claims."""
    reasons = []
    if type(text) is not str or not text.strip():
        return ['empty-or-nontext']
    if (text != text.strip() or not re.fullmatch(r"[A-Za-z0-9 .,'’$()\-]+", text)
            or text.startswith(("'", '’')) or text.endswith(("'", '’'))
            or '\n' in text or '\r' in text or len(text) > {'S': 100, 'T': 140, 'B': 60, 'C': 240}[field[0]]
            or any(c in text for c in ('<', '>', '{', '}', '[', ']', '`', '|', '\\'))
            or re.search(r'^(?:title|badge|copy|heading|assistant|user)\s*:', text, re.I)):
        reasons.append('nonplain-protocol-or-length')
    if row is not None and not native_receipt(row, field):
        reasons.append('native-EOS-or-token-cap')
    if text.isupper() and text != 'NONE':
        reasons.append('uppercase-value')
    if field[0] == 'S':
        if not 2 <= len(text.split()) <= 9:
            reasons.append('heading-length')
        if re.search(r'[$\d]', text):
            reasons.append('heading-price-or-number')
        if any(_span(text, p['name']) or any(_span(text, c['colorName']) for c in p['colorVariants']) for p in products):
            reasons.append('heading-bound-identity-or-color')
        residual = text
    else:
        facts = card['facts']; name, color = facts['name'], facts['selectedColor']
        if field[0] == 'B' and text == 'NONE':
            return reasons + ([] if color is None else ['color-fact-present'])
        residual = text
        if field[0] != 'B':
            if not _span(text, name) or len(re.findall(re.escape(name), text, re.I)) != 1:
                reasons.append('complete-selected-identity-once-required')
            residual = re.sub(re.escape(name), ' ', residual, flags=re.I)
        if field[0] in ('T', 'B') and color and not _span(text, color):
            reasons.append('exact-selected-color-required')
        if field[0] == 'C' and not (_span(text, color) or _span(text, facts['category']) or '$' in text):
            reasons.append('body-needs-meaningful-bound-fact')
        if color:
            residual = re.sub(r'(?<!\w)' + re.escape(color) + r'(?!\w)', ' ', residual, flags=re.I)
        # Category permission is all-or-nothing, not token-bag implication.
        residual = re.sub(r'(?<!\w)' + re.escape(facts['category']) + r'(?!\w)', ' ', residual, flags=re.I)
        for product in products:
            if product['name'] != name and _span(residual, product['name']):
                reasons.append('cross-product-identity')
            for variant in product['colorVariants']:
                other = variant['colorName']
                if other != color and _span(residual, other):
                    reasons.append('cross-variant-color')
        amounts = re.findall(r'\$\d+(?:\.\d+)?', residual)
        if amounts and re.search(r'\bcollection of\b', text, re.I):
            reasons.append('ambiguous-collection-item-price-scope')
        for amount in amounts:
            try:
                if not re.fullmatch(r'\$\d+(?:\.\d{1,2})?', amount) or Decimal(amount[1:]) != Decimal(str(facts['itemPrice'])):
                    reasons.append('incorrect-or-malformed-item-price')
            except InvalidOperation:
                reasons.append('incorrect-or-malformed-item-price')
        residual = re.sub(r'\$\d+(?:\.\d+)?', ' ', residual)
        if '$' in residual or re.search(r'\d', residual):
            reasons.append('unbound-numeric-or-price-claim')
        if field[0] == 'C' and not text.endswith('.'):
            reasons.append('unfinished-body')
    if UNSUPPORTED.search(residual):
        reasons.append('unsupported-product-or-service-claim')
    if re.search(r'\bavailable\b(?!\s+in\b)|\bready\b', residual, re.I):
        reasons.append('unverified-availability')
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", residual.casefold())
    if any(token not in FRAMING for token in tokens):
        reasons.append('unverified-claim-language')
    if len(tokens) != len(set(tokens)) and any(tokens.count(token) >= 4 for token in set(tokens)):
        reasons.append('repetition')
    return sorted(set(reasons))


def public_sections(cards, rows, products, limit=None):
    selected = cards if limit is None else cards[:limit]
    by_field = {row['fields'][0]: row for row in rows}
    validation = []
    sections = []
    def value(field, card=None):
        row = by_field.get(field)
        reasons = ['not-yet-generated'] if row is None else practical_rejections(field, row.get('text'), card, products, row=row)
        validation.append({'field': field, 'accepted': not reasons, 'reasons': reasons})
        return row['text'] if row is not None and not reasons and row['text'] != 'NONE' else None
    for start in range(0, len(selected), 3):
        section = {'heading': value(f'S{start // 3 + 1}'), 'cards': []}
        for index in range(start, min(start + 3, len(selected))):
            card = selected[index]; facts = card['facts']; number = index + 1
            section['cards'].append({**deepcopy(card['action']), 'productIndex': card['productIndex'],
                'originalName': facts['name'], 'itemPrice': facts['itemPrice'], 'selectedColor': facts['selectedColor'],
                'catalogCategory': facts['category'], 'title': value(f'T{number}', card),
                'badge': value(f'B{number}', card), 'bodyCopy': value(f'C{number}', card)})
        sections.append(section)
    return sections, {'acceptedFields': sum(x['accepted'] for x in validation),
        'withheldFields': sum(not x['accepted'] for x in validation), 'requiredFields': len(validation),
        'fields': validation,
        'scope': 'prospective practical source-fact safety screen; not comprehensive English factuality proof or fixed strict v2 score'}


def decode_rows(bb, state, jobs, width, first_ready, cancelled):
    """Independent cache forks, continuous native generator, first section first."""
    if width not in (6, 10, 20):
        raise ValueError('Explicit width 6, 10 or 20 required')
    from mlx_lm.generate import BatchGenerator, StopSequences, generation_stream
    from mlx_lm.sample_utils import make_sampler
    from .parallel_decode import clone_prefix_state, exact_row_suffix
    from .store_copy_wide_v2 import guard_memory
    mx = bb.mx; output = {}; active = {}; pending = list(enumerate(jobs)); first_done = False
    first_count = 1 + 3 * min(3, sum(r['kind'] == 'title' for r in jobs))
    started = time.perf_counter(); rounds = 0
    generator = BatchGenerator(bb.model, completion_batch_size=width, prefill_batch_size=width,
        prefill_step_size=256, sampler=make_sampler(temp=0.0))
    def check_cancel():
        if cancelled():
            raise RequestCancelled('Client disconnected or cancelled stale request')
    def admit():
        check_cancel()
        count = min(width - len(active), len(pending))
        if count <= 0:
            return
        group = [pending.pop(0) for _ in range(count)]
        prompts, caches = [], []
        for _, row in group:
            branch = clone_prefix_state(bb, state)
            suffix, cache = exact_row_suffix(bb.tokenizer, row, branch, bb.model)
            if not 1 <= len(suffix) <= 256 or len(state['tokens']) + len(suffix) + row['maxTokens'] > 4096:
                raise ValueError('Complete public field exceeds bounded context; no truncation')
            prompts.append(suffix); caches.append(cache)
        stops = [[int(token)] for token in bb.tokenizer.eos_token_ids]
        uids = generator.insert(prompts, max_tokens=[row['maxTokens'] for _, row in group], caches=caches,
            all_tokens=[list(state['tokens']) for _ in group], samplers=[make_sampler(temp=0.0) for _ in group],
            logits_processors=[[] for _ in group], stop_sequences=[StopSequences(stops) for _ in group])
        for uid, (index, row), suffix in zip(uids, group, prompts, strict=True):
            active[uid] = index
            output[index] = {**deepcopy(row), 'tokenIds': [], 'rawSampledTokenIds': [],
                'stopTokenId': None, 'arSteps': 0, 'finishReason': None, 'suffixTokens': len(suffix)}
        del caches, cache, branch
        guard_memory(bb)
    try:
        admit()
        while active:
            check_cancel()
            if time.perf_counter() - started > 120:
                raise TimeoutError('Native copy exceeded 120-second demo request budget')
            responses = generator.next_generated(); mx.synchronize(generation_stream)
            if not responses:
                raise RuntimeError('Native generator stopped with unfinished fields')
            rounds += 1
            for response in responses:
                index = active[response.uid]; record = output[index]
                record['arSteps'] += 1; record['rawSampledTokenIds'].append(int(response.token))
                if response.finish_reason != 'stop':
                    record['tokenIds'].append(int(response.token))
                if response.finish_reason is not None:
                    record['finishReason'] = response.finish_reason
                    record['stopTokenId'] = int(response.token) if response.finish_reason == 'stop' else None
                    record['text'] = bb.tokenizer.decode(record['tokenIds'])
                    record['completedMs'] = (time.perf_counter() - started) * 1000
                    del active[response.uid]
            del responses, response
            guard_memory(bb); check_cancel()
            if not first_done and all(i in output and output[i]['finishReason'] is not None for i in range(first_count)):
                first_ready([output[i] for i in range(first_count)]); first_done = True
            admit()
        if len(output) != len(jobs) or not first_done:
            raise RuntimeError('Native field roster incomplete')
        return [output[i] for i in range(len(jobs))], {'decodeRounds': rounds,
            'decodeWallMs': (time.perf_counter() - started) * 1000, 'width': width,
            'sampledTokenCap': sum(r['maxTokens'] for r in jobs),
            'actualSampledTokens': sum(r['arSteps'] for r in output.values())}
    except BaseException as error:
        error.partial_rows = [output[i] for i in sorted(output)]
        raise
    finally:
        generator.close(); mx.synchronize(generation_stream)


class ResidentDemoEngine:
    """Thread-affine native engine. Service owns the lock for this whole lifetime."""
    def __init__(self, *, rank_run=DEFAULT_RANK, copy_run=DEFAULT_COPY, output=DEFAULT_OUTPUT,
                 width=10, memory_gib=7, integration_manifest=None, catalog_binding_factory=None):
        initialization_started = time.perf_counter()
        from .store_hybrid_benchmark_v2 import completed_adapter
        from .semantic_copy_feature_fit import MODEL, verify_model_files
        from .semantic_copy_training import parameter_hashes
        from .store_demo_catalog import DemoCatalogBinding
        from .store_rank_session import StoreRankSession
        if width not in (6, 10, 20) or memory_gib != 7:
            raise ValueError('Reviewed demo widths6/10/20 and7GiB budget required')
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.width = width; self.copy_run = Path(copy_run)
        fit, checkpoint = completed_adapter(copy_run)
        self.copy_sha = fit['checkpoint']['sha256']
        self.model_verification = verify_model_files(fit['preflight']['modelFiles'])
        binding_factory = DemoCatalogBinding if catalog_binding_factory is None else catalog_binding_factory
        self.binding = binding_factory() if integration_manifest is None else binding_factory(integration_manifest)
        import mlx.core as mx
        from mlx_lm.generate import generation_stream
        from mlx_lm.tuner.lora import LoRALinear
        from .backbone import FrozenBackbone
        from .evidence_plan import REVISION
        self.stream_context = mx.stream(generation_stream); self.stream_context.__enter__()
        self.bb = FrozenBackbone(MODEL, REVISION, 4096, memory_gib)
        self.before_hashes = parameter_hashes(self.bb.model.language_model)
        if len(self.before_hashes) != 927:
            raise ValueError('Expected927 frozen source tensors')
        self.adapter = LoRALinear.from_base(self.bb.model.layers[-1].mlp.down_proj, r=4, scale=20, dropout=0)
        self.bb.model.layers[-1].mlp.down_proj = self.adapter
        self.adapter.load_weights(list(mx.load(str(checkpoint)).items()), strict=False)
        self.adapter.freeze(); self.bb.model.eval(); mx.eval(self.adapter.parameters()); mx.synchronize()
        self.session = StoreRankSession(self.bb, rank_run, source_validator=self.binding.assert_current_sources)
        from .store_copy_wide_v2 import guard_memory
        guard_memory(self.bb)
        self.startup = {'rankCheckpointSha256': self.session.head_sha, 'copyCheckpointSha256': self.copy_sha,
            'sourceTensorsHashed': len(self.before_hashes), 'residentLoadMs': self.bb.load_ms,
            'engineInitializationMs': (time.perf_counter() - initialization_started) * 1000,
            'initializationScope': 'engine constructor including source/checkpoint verification, model load, frozen hashes and catalog head; excludes process imports, worker lock wait and browser',
            'memory': self.bb.runtime_memory(), 'width': width, 'nativeFallback': False}
        (self.output / 'startup.json').write_text(json.dumps(self.startup, indent=2) + '\n')

    def close(self):
        from .semantic_copy_training import parameter_hashes
        from .semantic_copy_feature_fit import digest
        from mlx_lm.generate import generation_stream
        after = {k.replace('.31.mlp.down_proj.linear.', '.31.mlp.down_proj.'): value
            for k, value in parameter_hashes(self.bb.model.language_model).items()}
        receipt = {'frozen927HashesUnchanged': after == self.before_hashes,
            'copyCheckpointUnchanged': digest(self.copy_run / 'adapter.safetensors') == self.copy_sha,
            'memory': self.bb.runtime_memory()}
        (self.output / 'shutdown.json').write_text(json.dumps(receipt, indent=2) + '\n')
        self.bb.mx.synchronize(generation_stream); self.stream_context.__exit__(None, None, None)
        return receipt

    def personalize(self, value, request_id, started, emit, cancelled):
        from .branch_local_copy import prepare_root
        from .fused_prefix import _complete_cache
        from .native_batch_diagnosis import state_hash
        from .store_copy_wide_v2 import guard_memory
        request, policy = checked_request(value)
        directory = self.output / 'requests' / request_id; directory.mkdir(parents=True, exist_ok=False)
        receipt = {'requestId': request_id, 'input': deepcopy(value), 'effectiveRequest': request,
            'policy': policy, 'copyConditioning': deepcopy(NEUTRAL_COPY_REQUEST), 'copyConditionDisclosure': COPY_DISCLOSURE,
            'status': 'running', 'events': [], 'rawRows': [], 'startup': self.startup}
        def save():
            temporary = directory / 'receipt.tmp'
            temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
            temporary.replace(directory / 'receipt.json')
        save()
        def output_event(event):
            event = {'requestId': request_id, **event}
            canonical(event)  # Validate complete finite JSON before a readiness receipt.
            ready = (time.perf_counter() - started) * 1000
            event.setdefault('timing', {})['readyMs'] = ready
            if event['type'] == 'firstThree':
                event['timing']['first3readyMs'] = ready
            elif event['type'] == 'complete':
                event['timing']['fullReadyMs'] = ready
            canonical(event)
            receipt['events'].append(deepcopy(event)); emit(event)
        try:
            if cancelled():
                raise RequestCancelled('Request cancelled before native admission')
            rank_result = self.session.rank_request(ranking_input(request), copy_adapter=self.adapter)
            order, guards = guarded_order(rank_result['scores16'], request, policy, self.session.products)
            cards = action_bindings(request, order[:6], self.session.snapshot)
            rank_ms = (time.perf_counter() - started) * 1000
            provenance = {'ranking': 'fresh frozen Qwen query encoding plus trained actual-store v3 head; public query retained as preference; deterministic hard guards and whole-query exact-name override',
                'images': 'existing deterministic storefront gallery; declared metadata; no learned image or pixel claim; model receipt retains first declared image binding', 'rankCheckpointSha256': self.session.head_sha,
                'copyCheckpointSha256': self.copy_sha, 'copy': COPY_DISCLOSURE, 'timingScope': TIMING_SCOPE,
                'hiddenProfileInputs': False, 'strictV2BenchmarkRelaxed': False,
                'demoValidation': 'prospective practical per-field safety screen, separate from unchanged strict benchmark'}
            receipt.update(ranking=rank_result, finalOrder=order, guards=guards, cards=cards, provenance=provenance)
            metadata, _ = public_sections(cards, [], self.session.products)
            output_event({'type': 'rank', 'productIds': [self.session.snapshot['externalMapping']['products'][i]['productId'] for i in order],
                'selectedProductIds': [card['action']['productId'] for card in cards],
                'variants': {card['action']['productId']: card['action']['variantId'] for card in cards},
                'sections': metadata, 'eligibleCount': len(order), 'shortfall': max(0, 6 - len(order)),
                'timing': {'rankMs': rank_ms}, 'provenance': provenance})
            save()
            def copied_event(kind, rows, limit):
                sections, validation = public_sections(cards, rows, self.session.products, limit)
                strict = []
                for row in rows:
                    field = row['fields'][0]
                    errors = field_rejections(field, row.get('text'), {'cards': cards})
                    if not native_receipt(row, field):
                        errors.append('native-sampled-cap-or-EOS-receipt')
                    strict.append({'field': field, 'accepted': not errors, 'rejections': errors})
                receipt.setdefault('strictV2FieldDiagnostics', {})[kind] = {
                    'fields': strict, 'acceptedFields': sum(r['accepted'] for r in strict),
                    'requiredFields': len(strict), 'completeFieldsPass': all(r['accepted'] for r in strict),
                    'scope': 'unchanged strict v2 per-field gate on this fresh demo request; variable card workloads are not the fixed benchmark'}
                event = {'type': kind, 'sections': sections,
                    **{key: validation[key] for key in ('acceptedFields', 'withheldFields', 'requiredFields')},
                    'validation': validation, 'timing': {'rankMs': rank_ms,
                        'copyMs': (time.perf_counter() - started) * 1000 - rank_ms},
                    'eligibleCount': len(order), 'shortfall': max(0, 6 - len(order)), 'provenance': provenance}
                output_event(event)
            if cards:
                if cancelled():
                    raise RequestCancelled('Request cancelled after ranking')
                copy_started = time.perf_counter()
                common, jobs = demo_layout(cards)
                state, preparation = prepare_root(self.bb, common)
                _complete_cache(state['cache'], len(state['tokens'])); original = state_hash(state['cache'])
                guard_memory(self.bb)
                rows, timing = decode_rows(self.bb, state, jobs, self.width,
                    lambda done: copied_event('firstThree', done, min(3, len(cards))), cancelled)
                receipt.update(rawRows=rows, preparation=preparation, decodeTiming=timing,
                    copyWallMs=(time.perf_counter() - copy_started) * 1000)
                if state_hash(state['cache']) != original:
                    raise AssertionError('Shared native source cache changed')
                copied_event('complete', rows, len(cards))
                del state
                self.bb.mx.clear_cache()
            else:
                copied_event('complete', [], 0)
            self.binding.assert_current_sources(self.session.snapshot['receipt'])
            receipt.update(status='completed', memory=guard_memory(self.bb))
            save()
        except BaseException as error:
            receipt.update(status='cancelled' if isinstance(error, RequestCancelled) else 'failed',
                error={'type': type(error).__name__, 'message': str(error)},
                rawRows=getattr(error, 'partial_rows', receipt['rawRows']),
                failedAtMs=(time.perf_counter() - started) * 1000)
            save(); raise
