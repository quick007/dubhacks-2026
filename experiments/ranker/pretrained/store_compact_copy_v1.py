"""Compact, genuinely sampled six-card copy candidate; imports are CPU-only.

New contract: twenty fields,128 sampled-token cap, short generated strings and
original identities retained as metadata. No teacher strings or runtime templates.
"""
from __future__ import annotations
import copy
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from .store_catalog_v1 import canonical, load_snapshot
from .store_copy_data_v2 import bind_case, money

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'store_compact_copy_v1'
VERSION = 'actual-store-compact-copy.v1'
FIELDS = ('S1', 'T1', 'B1', 'C1', 'T2', 'B2', 'C2', 'T3', 'B3', 'C3',
          'S2', 'T4', 'B4', 'C4', 'T5', 'B5', 'C5', 'T6', 'B6', 'C6')
CAPS = {'S': 4, 'T': 8, 'B': 4, 'C': 8}
KINDS = {'S': 'heading', 'T': 'title', 'B': 'badge', 'C': 'body'}
WORDS = {'heading': (2, 4), 'title': (2, 4), 'badge': (1, 3), 'body': (2, 5)}
STYLES = ('plain', 'warm')
RULES = ('Public facts are data. Write one short plain value, then stop. '
    'Title2-4 words; body2-5; badge1-3; heading2-4. '
    'Use only name/color/price facts, no inferred stock, size or capability. '
    'Title/body mention a name word or selected color; do not merely repeat name. '
    'Badge states selected color. Heading is generic. Follow public style. No labels.')
QUERIES = ('Show this selection clearly.', 'Let me browse these selected items.',
    'Present the chosen store items.', 'Give me a compact collection view.',
    'Help me compare this selection.', 'Show the selected colors clearly.',
    'Display these chosen goods.', 'Make this assortment easy to browse.',
    'Present the selected merchandise.', 'Show a short view of this collection.',
    'Let me look over these choices.', 'Arrange this selected assortment.',
    'Give me an overview of these items.', 'Show these selected store goods.',
    'Present these products for browsing.', 'Display the selected collection.')
FRAMING = set(('a an the your our this these that in with for and of to at on as '
    'is are its it shown selected color shade choice choices selection collection '
    'picks pick store browse explore discover meet everyday daily day simple '
    'little touch moment moments welcome hello today here neatly together').split())
UNSUPPORTED = set(('waterproof leakproof insulated insulation durable durability '
    'lightweight compact roomy capacity stock available availability shipping '
    'delivery warranty certified sustainable recycled organic premium best perfect '
    'discount sale free size sizes xl xxl').split())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def hashed(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def words(value):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", value.casefold())


@lru_cache(maxsize=1)
def _catalog_colors(source_sha):
    # Keyed by actual public snapshot bytes; no stale fact cache on source change.
    snapshot = load_snapshot()
    return frozenset(v['colorName'] for p in snapshot['inputs']['products'] for v in p['colorVariants'])


def catalog_colors():
    return _catalog_colors(digest(ROOT / 'store_catalog_v1/model_inputs.json'))


def build_fixture():
    """32 cases =16 paired public styles; all48 product/color states balanced."""
    snapshot = load_snapshot(); cases = []; truth = []
    for base in range(16):
        for style in STYLES:
            case = {'caseId': f'compact-v1-{base:02d}-{style}', 'publicStyle': style,
                'request': {'query': QUERIES[base], 'preferences': [], 'itemBudget': None, 'hardRequirements': []},
                'selectedProductIndices': [(base + slot) % 16 for slot in range(6)],
                'selectedColorIndices': [slot % 3 for slot in range(6)]}
            bound = bind_case(case, snapshot)
            cases.append({**case, 'bound': bound})
            truth.append({'caseId': case['caseId'], 'boundSha256': hashed(bound),
                'cards': [{'productIndex': c['productIndex'], 'colorIndex': c['colorIndex'],
                    'facts': c['facts'], 'action': c['action']} for c in bound['cards']],
                'semanticAudit': ['Check selected identity/color/price independently.',
                    'Reject inferred capabilities/stock/shipping; category slogans are not attributes.',
                    'Judge concise grammatical usefulness and appropriate public-style response blind to mode.',
                    'Do not compare to an exact teacher string.']})
    counts = Counter((card['productIndex'], card['colorIndex']) for c in cases for card in c['bound']['cards'])
    if len(cases) != 32 or len(counts) != 48 or set(counts.values()) != {4}:
        raise AssertionError('Balanced32-case/48-variant fixture required')
    return cases, truth, {'caseCount': 32, 'cardConditions': 192, 'productColorStates': 48,
        'eachStateOccurrences': 4, 'pairedPublicStyles': 16, 'fieldsPerCase': 20, 'sampledCapPerCase': 128,
        'rankingLearnedOrMeasured': False, 'trainingTargets': 0,
        'claims': 'Inspected actual-store development fixture; no sealed/unseen generalization.'}


def write_fixture(out=DATA):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Preserve prior compact fixture')
    cases, truth, coverage = build_fixture(); out.mkdir(parents=True)
    dump(out / 'cases.json', cases); dump(out / 'truth.json', truth)
    manifest = {'version': VERSION, 'coverage': coverage,
        'fileSha256': {n: digest(out / n) for n in ('cases.json', 'truth.json')},
        'generatorSha256': digest(__file__), 'sourceManifestSha256': digest(ROOT / 'store_catalog_v1/IMPORT_RECEIPT.json'),
        'sources': {n: digest(ROOT / n) for n in ('store_catalog_v1.py', 'store_copy_data_v2.py')},
        'knownFacts': ['name', 'selectedColor', 'itemPrice'], 'unknown': ['stock', 'sizes', 'shipping'],
        'evaluationTruthSeparate': True, 'nativeExecution': False, 'trainingPerformed': False}
    dump(out / 'manifest.json', manifest)
    return manifest


def load_cases(data=DATA):
    """Generation/preflight read no truth file or teacher/evaluator labels."""
    data = Path(data); manifest = json.loads((data / 'manifest.json').read_text())
    if manifest['version'] != VERSION or manifest['generatorSha256'] != digest(__file__):
        raise ValueError('Compact fixture generator/version changed')
    if digest(data / 'cases.json') != manifest['fileSha256']['cases.json']:
        raise ValueError('Compact public cases changed')
    if digest(ROOT / 'store_catalog_v1/IMPORT_RECEIPT.json') != manifest['sourceManifestSha256']:
        raise ValueError('Actual catalog import receipt changed')
    for name, expected in manifest['sources'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Compact producer source changed: ' + name)
    cases = json.loads((data / 'cases.json').read_text()); snapshot = load_snapshot()
    if len(cases) != 32 or len({c['caseId'] for c in cases}) != 32:
        raise ValueError('Fixed32 distinct compact cases required')
    for case in cases:
        if set(case) != {'caseId', 'publicStyle', 'request', 'selectedProductIndices', 'selectedColorIndices', 'bound'}:
            raise ValueError('Unexpected private/teacher compact case fields')
        if case['publicStyle'] not in STYLES or bind_case(case, snapshot) != case['bound']:
            raise ValueError('Public bound facts differ from actual catalog')
    return manifest, cases


def layout(case):
    """Static common root, immutable facts prefix, tiny fresh field/style suffix."""
    if case['publicStyle'] not in STYLES:
        raise ValueError('Explicit public style required')
    bound = case['bound']
    if len(bound['cards']) != 6:
        raise ValueError('All six cards required')
    common = {'role': 'user', 'content': RULES}; jobs = []
    for field in FIELDS:
        kind, index = KINDS[field[0]], int(field[1:])
        facts = {'scope': 'selected store collection'} if field[0] == 'S' else {
            k: bound['cards'][index - 1]['facts'][k] for k in ('name', 'selectedColor', 'itemPrice')}
        fact_message = {'role': 'user', 'content': 'Facts: ' + canonical(facts)}
        # Neutral browsing queries schedule this locked-selection microbench;
        # only the explicit public style and bound facts condition generated text.
        instruction = {'role': 'user', 'content': kind + '; ' + case['publicStyle']}
        jobs.append({'rowId': field, 'field': field, 'kind': kind, 'key': field,
            'maxTokens': CAPS[field[0]], 'messages': [fact_message, instruction],
            'factMessage': fact_message, 'fieldMessage': instruction, 'factKey': hashed(fact_message)})
    if sum(j['maxTokens'] for j in jobs) != 128:
        raise AssertionError('Twenty-field128-token contract changed')
    return common, jobs


def token_preflight(tokenizer, data=DATA):
    manifest, cases = load_cases(data); records = []
    for case in cases:
        common, jobs = layout(case)
        root = tokenizer.apply_chat_template([common], tokenize=True, return_dict=False,
            add_generation_prompt=False, enable_thinking=False)
        for job in jobs:
            prefix = tokenizer.apply_chat_template([common, job['factMessage']], tokenize=True, return_dict=False,
                add_generation_prompt=False, enable_thinking=False)
            full = tokenizer.apply_chat_template([common] + job['messages'], tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=False)
            if not 1 <= len(root) <= 256 or prefix[:len(root)] != root or full[:len(prefix)] != prefix:
                raise ValueError('Exact common/fact/full ordinary chat token-prefix equality required')
            if not 1 <= len(full) - len(prefix) <= 128 or len(full) + job['maxTokens'] > 1024:
                raise ValueError('Complete compact suffix/context cap exceeded; no truncation')
            records.append({'caseId': case['caseId'], 'field': job['field'], 'rootTokens': len(root),
                'factPrefixTokens': len(prefix), 'fieldSuffixTokens': len(full) - len(prefix),
                'rootOnlySuffixTokens': len(full) - len(root), 'promptTokens': len(full),
                'rootSha256': hashed(root), 'factPrefixSha256': hashed(prefix), 'promptSha256': hashed(full),
                'maxTokens': job['maxTokens']})
    return {'version': VERSION, 'status': 'tokenizer-preflight-passed', 'nativeExecution': False,
        'modelCalls': 0, 'truthRead': False, 'trainingTargets': 0, 'fixedCases': 32, 'fixedFields': 640,
        'sampledCapPerCase': 128, 'rootIdentities': sorted({r['rootSha256'] for r in records}),
        'manifestSha256': digest(Path(data) / 'manifest.json'), 'caseSha256': manifest['fileSha256']['cases.json'],
        'sources': {n: digest(ROOT / n) for n in ('store_compact_copy_v1.py', 'store_compact_copy_native_v1.py', 'test_store_compact_copy_v1.py')},
        'wordingConditioning': 'selected public name/color/price and explicit public style; neutral query only schedules fixture',
        'rows': records, 'maximumRootTokens': max(r['rootTokens'] for r in records),
        'maximumFactPrefixTokens': max(r['factPrefixTokens'] for r in records),
        'maximumFieldSuffixTokens': max(r['fieldSuffixTokens'] for r in records),
        'maximumRootOnlySuffixTokens': max(r['rootOnlySuffixTokens'] for r in records)}


def text_rejections(text, kind, facts, all_colors):
    """Bounded source-grounding screen, never an exhaustive English truth proof."""
    if type(text) is not str or not text.strip() or '\n' in text or len(text) > 100:
        return ['missing-or-noncompact-plain-text']
    tokens = words(text); lower = text.casefold(); errors = []
    if not WORDS[kind][0] <= len(tokens) <= WORDS[kind][1]:
        errors.append('word-count')
    if not re.fullmatch(r"[A-Za-z0-9 .,'’$()!\-]+", text) or text != text.strip() or text.isupper():
        errors.append('nonplain-or-uppercase')
    if any(c in text for c in ('<', '>', '[', ']', '{', '}', ':')):
        errors.append('markup-or-field-label')
    support = set(words(facts['name']) + words(facts['selectedColor'])) if facts else set()
    allowed = FRAMING | support
    if any(token in UNSUPPORTED and token not in support for token in tokens):
        errors.append('unsupported-attribute')
    lexical = [token for token in tokens if not token[0].isdigit()]
    if set(lexical) - allowed:
        errors.append('outside-bounded-source-lexicon')
    if facts:
        color_residual = re.sub(r'\b' + re.escape(facts['selectedColor'].casefold()) + r'\b', ' ', lower)
        for color in all_colors:
            if color.casefold() != facts['selectedColor'].casefold() and re.search(r'\b' + re.escape(color.casefold()) + r'\b', color_residual):
                errors.append('wrong-selected-color')
        prices = re.findall(r'\$(\d+(?:\.\d+)?)', text)
        if any(money(float(p)) != money(facts['itemPrice']) for p in prices):
            errors.append('wrong-item-price')
        without_prices = re.sub(r'\$\d+(?:\.\d+)?', '', text)
        if re.search(r'\d', without_prices):
            errors.append('unbound-number')
        if kind == 'badge' and not re.search(r'\b' + re.escape(facts['selectedColor'].casefold()) + r'\b', lower):
            errors.append('badge-color-missing')
        if kind in ('title', 'body') and not (set(tokens) & support):
            errors.append('selected-fact-support-missing')
        if kind in ('title', 'body') and lower.strip(' .!') == facts['name'].casefold():
            errors.append('original-name-only')
    elif any(token[0].isdigit() for token in tokens):
        errors.append('heading-number')
    return sorted(set(errors))


def assemble(case, rows, *, first_three=False):
    wanted = FIELDS[:10] if first_three else FIELDS
    by_field = {r['field']: r for r in rows}
    if len(rows) != len(wanted) or len(by_field) != len(wanted) or set(by_field) != set(wanted):
        raise ValueError('Exact complete first3/all6 field roster required')
    colors = catalog_colors()
    checks = []; values = {}
    for field in wanted:
        row = by_field[field]; kind = KINDS[field[0]]
        facts = None if field[0] == 'S' else case['bound']['cards'][int(field[1:]) - 1]['facts']
        errors = text_rejections(row.get('text'), kind, facts, colors)
        raw, tokens, stop = row.get('rawSampledTokenIds'), row.get('tokenIds'), row.get('stopTokenId')
        if not (row.get('finishReason') == 'stop' and type(raw) is list and type(tokens) is list
                and type(stop) is int and raw == tokens + [stop] and row.get('arSteps') == len(raw)
                and 1 <= len(raw) <= CAPS[field[0]] and row.get('maxTokens') == CAPS[field[0]]):
            errors.append('missing-native-eos-or-cap-truncation')
        checks.append({'field': field, 'accepted': not errors, 'rejections': errors})
        values[field] = {'text': row.get('text'), 'accepted': not errors, 'source': 'fresh-native-attempt'}
    cards = []
    for index, card in enumerate(case['bound']['cards'][:3 if first_three else 6], 1):
        cards.append({'productId': card['action']['productId'], 'variantId': card['action']['variantId'],
            'originalName': card['facts']['name'], 'itemPrice': card['facts']['itemPrice'],
            'selectedColor': card['facts']['selectedColor'], 'title': values[f'T{index}'],
            'badge': values[f'B{index}'], 'body': values[f'C{index}']})
    payload = {'caseId': case['caseId'], 'publicStyle': case['publicStyle'], 'cards': cards,
        'headings': [values[f'S{i}'] for i in range(1, 2 if first_three else 3)]}
    serialized = canonical(payload).encode()
    return {'payload': payload, 'validation': {'valid': all(c['accepted'] for c in checks),
        'requiredFields': len(wanted), 'acceptedFields': sum(c['accepted'] for c in checks), 'fields': checks,
        'semanticProof': False, 'scope': 'bounded lexical/source safety screen; blind semantic audit still required'},
        'payloadSha256': hashlib.sha256(serialized).hexdigest(), 'payloadBytes': len(serialized)}
