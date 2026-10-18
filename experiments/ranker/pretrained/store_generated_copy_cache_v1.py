"""Versioned, auditable native-generated public-fact copy cache; no model load.

A hit is ALWAYS generated-earlier copy, never fresh per-request generation.
The caller owns learned ranking, the native worker lock, memory and controls.
Misses require a supplied native generator; no text templates or repair exist.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from .store_catalog_v1 import canonical
from .store_demo_engine import NEUTRAL_COPY_REQUEST, demo_layout, practical_rejections, public_sections

VERSION = 'actual-store-native-generated-copy-cache.v1'
WIDTHS = (6, 10, 20)
DISCLOSURE = ('Previously generated native model copy, reused from a versioned public product/color cache. '
    'Shopper query changes learned ordering and hard guards; cached wording is not generated fresh for this request. '
    'Unsafe native fields remain withheld as null. Images follow existing deterministic storefront metadata.')
HERE = Path(__file__).resolve().parent


def hashed(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    if type(value) is not str:
        raise ValueError('Explicit timezone-aware native generation timestamp required')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('Native timestamps must include timezone')
    return parsed


def digest_value(value):
    if type(value) is not str or len(value) != 64 or set(value) - set('0123456789abcdef'):
        raise ValueError('Exact lowercase SHA-256 provenance required')
    return value


def producer_spec(checkpoint_sha256, model_revision, tokenizer_sha256, *, native_width=10):
    if type(native_width) is not int or native_width not in WIDTHS or type(model_revision) is not str or not model_revision:
        raise ValueError('Explicit pinned model revision and supported native width required')
    return {'version': VERSION, 'checkpointSha256': digest_value(checkpoint_sha256),
        'modelRevision': model_revision, 'tokenizerSha256': digest_value(tokenizer_sha256),
        'nativeWidth': native_width, 'temperature': 0, 'neutralCopyRequest': deepcopy(NEUTRAL_COPY_REQUEST),
        'sourceSha256': {name: sha(HERE / name) for name in ('store_generated_copy_cache_v1.py',
            'store_demo_engine.py', 'store_copy_data_v2.py', 'branch_local_copy.py', 'parallel_decode.py')},
        'cacheSelection': 'first recorded native attempt per exact key; invalid fields withheld; no quality-based retry',
        'queryDependentCopy': False, 'freshHitClaimPermitted': False}


def catalog_card(snapshot, product_index, color_index):
    product = snapshot['inputs']['products'][product_index]
    external = snapshot['externalMapping']['products'][product_index]
    color, route = product['colorVariants'][color_index], external['colorVariants'][color_index]
    return {'productIndex': product_index, 'facts': {'name': product['name'], 'category': product['category'],
        'itemPrice': product['itemPrice'], 'selectedColor': color['colorName']},
        'action': {'productId': external['productId'], 'variantId': route['variantId'], 'image': deepcopy(route['images'][0])}}


def warm_plan(snapshot):
    """Eight fixed six-card/20-field/416-token requests cover all48 colors."""
    products = snapshot['inputs']['products']
    if len(products) != 16 or any(len(p['colorVariants']) != 3 for p in products):
        raise ValueError('Exact actual-store16 products/48 declared colors required')
    cards = [catalog_card(snapshot, pi, ci) for ci in range(3) for pi in range(16)]
    groups = []
    for start in range(0, 48, 6):
        selected = cards[start:start + 6]
        if len({c['action']['productId'] for c in selected}) != 6:
            raise AssertionError('Warm request must retain six distinct actual products')
        common, jobs = demo_layout(selected)
        groups.append({'groupId': f'catalog-colors-{start // 6:02d}', 'cards': selected,
            'common': common, 'jobs': jobs, 'fields': len(jobs), 'sampledCap': sum(j['maxTokens'] for j in jobs)})
    return {'version': VERSION + '/warm-plan', 'groups': groups, 'productColors': 48, 'nativeRequests': 8,
        'nativeFields': 160, 'sampledTokenCap': 8 * 416,
        'headingPolicy': 'first native S1/S2 attempt retained; later duplicate heading outputs archived but never selected by quality',
        'nativeExecution': False}


class NativeGenerationRequired(RuntimeError):
    def __init__(self, misses):
        self.misses = misses
        super().__init__('Native generation is required for cache misses; no fallback text is available')


class GeneratedCopyCache:
    def __init__(self, directory, snapshot, spec, *, clock=utc_now):
        self.root = Path(directory); self.snapshot = deepcopy(snapshot); self.spec = deepcopy(spec); self.clock = clock
        self.products = self.snapshot['inputs']['products']; self.plan = warm_plan(self.snapshot)
        expected = producer_spec(spec['checkpointSha256'], spec['modelRevision'], spec['tokenizerSha256'], native_width=spec['nativeWidth'])
        if self.spec != expected:
            raise ValueError('Current producer sources/config differ from exact cache specification')
        self.cards = {(c['action']['productId'], c['action']['variantId']): c for g in self.plan['groups'] for c in g['cards']}
        self.binding = {'version': VERSION, 'spec': self.spec,
            'catalogInputsSha256': hashed(self.snapshot['inputs']), 'externalMappingSha256': hashed(self.snapshot['externalMapping'])}
        self.binding_sha = hashed(self.binding)
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / 'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != self.binding:
                raise ValueError('Existing cache belongs to different public facts, actions, checkpoint or prompt; use a new namespace')
        else:
            with manifest.open('x') as file:
                file.write(canonical(self.binding) + '\n')
        (self.root / 'entries').mkdir(exist_ok=True); (self.root / 'attempts').mkdir(exist_ok=True)

    def assert_current(self):
        if json.loads((self.root / 'manifest.json').read_text()) != self.binding:
            raise ValueError('Immutable cache manifest changed')
        if any(sha(HERE / name) != value for name, value in self.spec['sourceSha256'].items()):
            raise ValueError('Cache producer/validator source changed; new versioned namespace required')

    def descriptors(self, cards):
        if not 1 <= len(cards) <= 6 or len({c['action']['productId'] for c in cards}) != len(cards):
            raise ValueError('One to six distinct real eligible products required; never pad shortfalls')
        for card in cards:
            key = card['action']['productId'], card['action']['variantId']
            if self.cards.get(key) != card:
                raise ValueError('Requested product/color facts or actual action differ from bound public catalog')
        common, jobs = demo_layout(cards)
        descriptors = []
        for section in range(1, (len(cards) + 2) // 3 + 1):
            descriptors.append({'kind': 'heading', 'identity': {'section': section},
                'facts': {'genericSection': section}, 'jobs': [j for j in jobs if j['fields'] == [f'S{section}']]})
        for number, card in enumerate(cards, 1):
            descriptors.append({'kind': 'card', 'identity': {k: card['action'][k] for k in ('productId', 'variantId')},
                'facts': deepcopy(card['facts']), 'jobs': [j for j in jobs if j['fields'][0] in (f'T{number}', f'B{number}', f'C{number}')]})
        for descriptor in descriptors:
            descriptor['keyBasis'] = {'bindingSha256': self.binding_sha, 'kind': descriptor['kind'],
                'identity': descriptor['identity'], 'facts': descriptor['facts'], 'commonMessage': common,
                'fieldPrompts': [{'kind': j['kind'], 'messages': j['messages'], 'maxTokens': j['maxTokens']} for j in descriptor['jobs']]}
            descriptor['key'] = hashed(descriptor['keyBasis'])
        return common, jobs, descriptors

    def _read(self, descriptor):
        path = self.root / 'entries' / (descriptor['key'] + '.json')
        if not path.exists():
            return None
        envelope = json.loads(path.read_text()); entry = envelope['entry']
        if envelope['sha256'] != hashed(entry) or entry['keyBasis'] != descriptor['keyBasis'] or entry['key'] != descriptor['key']:
            raise ValueError('Cached native entry or its prompt/fact provenance changed')
        attempt_path = self.root / 'attempts' / entry['attemptFile']
        if sha(attempt_path) != entry['attemptSha256']:
            raise ValueError('Original raw native generation archive changed')
        return entry

    def record_generation(self, cards, generated):
        """Persist raw native attempts, including rejected fields; never overwrite."""
        self.assert_current(); _, jobs, descriptors = self.descriptors(cards)
        started, completed = timestamp(generated['startedAt']), timestamp(generated['completedAt'])
        if completed < started:
            raise ValueError('Native generation completion precedes start timestamp')
        cost = generated['coldGenerationMs']
        if type(cost) not in (int, float) or not 0 <= cost < float('inf'):
            raise ValueError('Measured finite cold generation cost required')
        provenance = generated['provenance']
        if (provenance.get('nativeGeneration') is not True or provenance.get('fallback') is not False
                or provenance.get('checkpointSha256') != self.spec['checkpointSha256']
                or provenance.get('modelRevision') != self.spec['modelRevision']
                or provenance.get('tokenizerSha256') != self.spec['tokenizerSha256']
                or provenance.get('nativeWidth') != self.spec['nativeWidth']):
            raise ValueError('Exact native producer/checkpoint/tokenizer/width provenance required')
        digest_value(provenance.get('sameWidthControlsSha256'))
        rows = generated['rows']; by_field = {r['fields'][0]: r for r in rows}
        if len(rows) != len(jobs) or len(by_field) != len(jobs) or set(by_field) != {j['fields'][0] for j in jobs}:
            raise ValueError('Complete native six-card/20-field roster required; no omitted body fields')
        for job in jobs:
            row = by_field[job['fields'][0]]
            if any(row.get(k) != job[k] for k in ('rowId', 'kind', 'fields', 'maxTokens', 'messages')):
                raise ValueError('Native row differs from exact bound serving prompt/cap')
        attempt = {'version': VERSION + '/attempt', 'bindingSha256': self.binding_sha, 'cards': deepcopy(cards),
            'generation': deepcopy(generated), 'selectionPolicy': self.spec['cacheSelection']}
        attempt_name = hashed(attempt) + '.json'; attempt_path = self.root / 'attempts' / attempt_name
        if not attempt_path.exists():
            with attempt_path.open('x') as file:
                file.write(canonical(attempt) + '\n')
        inserted = []
        for descriptor in descriptors:
            if self._read(descriptor) is not None:
                continue
            kept = [deepcopy(by_field[j['fields'][0]]) for j in descriptor['jobs']]
            validations = []
            for row in kept:
                field = row['fields'][0]; card = None if field[0] == 'S' else cards[int(field[1:]) - 1]
                validations.append({'fieldKind': field[0], 'rejections': practical_rejections(field, row.get('text'), card, self.products, row=row)})
            entry = {'version': VERSION, 'key': descriptor['key'], 'keyBasis': descriptor['keyBasis'],
                'startedAt': generated['startedAt'], 'completedAt': generated['completedAt'], 'coldGenerationMs': cost,
                'provenance': deepcopy(provenance), 'rawRows': kept, 'validationAtGeneration': validations,
                'attemptFile': attempt_name, 'attemptSha256': sha(attempt_path)}
            path = self.root / 'entries' / (entry['key'] + '.json')
            with path.open('x') as file:
                file.write(canonical({'sha256': hashed(entry), 'entry': entry}) + '\n')
            inserted.append(entry['key'])
        return {'attemptFile': str(attempt_path), 'attemptSha256': sha(attempt_path), 'insertedKeys': inserted,
            'nativeColdGenerationMs': cost, 'nativeFieldCount': len(rows), 'rawAttemptsPreserved': True}

    def resolve(self, cards, *, native_generate=None, request_started=None):
        """Compose full real action payload; caller may start clock before ranking.

        A miss invokes supplied native_generate(cards) for the unchanged complete
        page, including all six cards/20 fields when six were selected. Existing
        cache hits are not overwritten or silently repaired by that new attempt.
        """
        started = time.perf_counter() if request_started is None else request_started
        self.assert_current(); _, jobs, descriptors = self.descriptors(cards)
        entries = {d['key']: self._read(d) for d in descriptors}
        misses = [d['key'] for d in descriptors if entries[d['key']] is None]
        initial_hits = set(entries) - set(misses); cold = None
        if misses:
            if native_generate is None:
                raise NativeGenerationRequired(misses)
            admitted = timestamp(self.clock())
            generated = native_generate(deepcopy(cards))
            if timestamp(generated['startedAt']) < admitted:
                raise ValueError('Native miss callback supplied an earlier archive instead of current request generation')
            cold = self.record_generation(cards, generated)
            entries = {d['key']: self._read(d) for d in descriptors}
            if any(entry is None for entry in entries.values()):
                raise RuntimeError('Native miss generation did not populate every required exact key')
        now = timestamp(self.clock()); rows = []; receipts = []
        for descriptor in descriptors:
            entry = entries[descriptor['key']]
            age = (now - timestamp(entry['completedAt'])).total_seconds() * 1000
            if age < 0:
                raise ValueError('Cached native generation timestamp is in the future')
            for raw, job in zip(entry['rawRows'], descriptor['jobs'], strict=True):
                # Only scheduler labels change. Exact field-user prompt, caps,
                # generated text, sampled tokens and native EOS remain intact.
                if (raw['messages'], raw['kind'], raw['maxTokens']) != (job['messages'], job['kind'], job['maxTokens']):
                    raise ValueError('Position-independent copy prompt equality is not established')
                row = {**deepcopy(raw), 'rowId': job['rowId'], 'key': job['key'], 'fields': job['fields'],
                    'cacheOrigin': {'key': entry['key'], 'originalRowId': raw['rowId'], 'originalFields': raw['fields'],
                        'generatedAt': entry['completedAt'], 'generatedEarlier': entry['key'] in initial_hits}}
                rows.append(row)
            receipts.append({'key': entry['key'], 'kind': descriptor['kind'], 'identity': descriptor['identity'],
                'hit': entry['key'] in initial_hits, 'generatedEarlier': entry['key'] in initial_hits,
                'generatedAt': entry['completedAt'], 'ageMs': age, 'originalColdGenerationMs': entry['coldGenerationMs'],
                'attemptSha256': entry['attemptSha256'], 'validationAtGeneration': entry['validationAtGeneration']})
        by_field = {r['fields'][0]: r for r in rows}; ordered = [by_field[j['fields'][0]] for j in jobs]
        sections, validation = public_sections(cards, ordered, self.products)
        cache = {'version': VERSION, 'bindingSha256': self.binding_sha, 'initialHits': len(initial_hits),
            'initialMisses': len(misses), 'nativeMissGeneration': cold is not None,
            'nativeColdGenerationMsThisRequest': 0 if cold is None else cold['nativeColdGenerationMs'],
            'copySource': 'generated-earlier' if not misses else 'fresh-native-on-miss' if not initial_hits else 'mixed-earlier-and-native-on-miss',
            'freshPerRequestGenerationClaim': False if not misses or initial_hits else True,
            'disclosure': DISCLOSURE, 'entries': receipts, 'imagesLearned': False}
        payload = {'version': VERSION + '/payload', 'sections': sections,
            **{k: validation[k] for k in ('acceptedFields', 'withheldFields', 'requiredFields')}, 'copyCache': cache}
        encoded = canonical(payload).encode(); ready = (time.perf_counter() - started) * 1000
        return {'payload': payload, 'encodedPayload': encoded, 'payloadSha256': hashlib.sha256(encoded).hexdigest(),
            'payloadBytes': len(encoded), 'validatedAndSerializedAtMs': ready, 'rawRows': ordered,
            'validation': validation, 'requestTimingIncludesCallerRanking': request_started is not None,
            'timingScope': 'caller start through fact/native-receipt revalidation, cache lookup and complete UTF8 payload serialization; no browser paint or image transfer',
            'cache': cache}


def personalize_with_cache(session, cache, value, *, copy_adapter=None, native_generate=None):
    """Composable candidate: fresh learned query ranking through serialized copy.

    Caller owns resident model verification/lock; this entry point imports no
    native runtime. It never substitutes copy latency for whole-request latency.
    """
    from .store_demo_engine import checked_request, guarded_order, ranking_input
    from .store_rank_session import action_bindings
    started = time.perf_counter()
    request, policy = checked_request(value)
    ranked = session.rank_request(ranking_input(request), copy_adapter=copy_adapter)
    order, guards = guarded_order(ranked['scores16'], request, policy, session.products)
    cards = action_bindings(request, order[:6], session.snapshot)
    if not cards:
        payload = {'version': VERSION + '/payload', 'sections': [], 'acceptedFields': 0, 'withheldFields': 0,
            'requiredFields': 0, 'copyCache': {'copySource': 'no-eligible-products', 'generatedEarlier': False,
                'nativeMissGeneration': False, 'nativeColdGenerationMsThisRequest': 0, 'freshPerRequestGenerationClaim': False}}
        encoded = canonical(payload).encode()
        return {'payload': payload, 'encodedPayload': encoded, 'payloadSha256': hashlib.sha256(encoded).hexdigest(),
            'payloadBytes': len(encoded), 'validatedAndSerializedAtMs': (time.perf_counter() - started) * 1000,
            'ranking': ranked, 'guards': guards, 'shortfall': 6, 'noPadding': True}
    result = cache.resolve(cards, native_generate=native_generate, request_started=started)
    result.update(ranking=ranked, guards=guards, shortfall=max(0, 6 - len(cards)), noPadding=True,
        scope='fresh learned query ranking, public hard guards, actual variant binding, cached/native-miss copy revalidation and complete UTF8 payload; no browser paint or image transfer')
    return result
