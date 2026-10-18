"""CPU-only packet for an isolated Qwen-feature parallel word decoder.

Teachers are offline synthetic supervision. Neither feature text nor the
serving row contract contains target text, product IDs, private profiles, split
membership or teacher construction indices. This is not the live copy path.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

from .store_catalog_v1 import canonical, digest, load_snapshot, MODEL_INPUTS_SHA256
from .store_copy_data_v2 import money, validate_facts

VERSION = 'actual-store-parallel-word-data.v1'
ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'store_parallel_text_v1'
KINDS = ('section', 'title', 'badge', 'copy')
STYLES = ('plain', 'browse', 'compact')
SLOTS = 14  # lexical tokens plus an explicitly learned EOS; no AR token loop
EOS, PAD = '<eos>', '<pad>'
TOKEN_RE = re.compile(r"\$\d+(?:\.\d{1,2})?|[A-Za-z]+(?:[-'][A-Za-z]+)*|[.,:;!?]")
SAFE_WORDS = frozenset(('in', 'shown', 'at', 'color', 'item', 'option', 'selected', 'displayed'))
SECTION_WORDS = frozenset(('your', 'selected', 'collection', 'browse', 'these', 'choices', 'store',
                           'more', 'explore', 'the', 'current', 'selection'))


def tokens(text):
    if type(text) is not str or not text.strip():
        raise ValueError('Nonempty lexical text required')
    found = list(TOKEN_RE.finditer(text))
    cursor = 0
    for match in found:
        if text[cursor:match.start()].strip():
            raise ValueError('Unrecognized lexical input; no silent removal')
        cursor = match.end()
    if text[cursor:].strip() or not found:
        raise ValueError('Unrecognized lexical input; no silent removal')
    return [m.group() for m in found]


def detokenize(values):
    """Whitespace/punctuation joining only; never inserts source facts."""
    return re.sub(r'\s+([.,:;!?])', r'\1', ' '.join(values))


def fact_text(facts):
    validate_facts(facts)
    return 'Public product facts: ' + canonical(facts)


def public_record(facts):
    text = fact_text(facts)
    return {'featureKey': digest(text), 'text': text, 'textSha256': hashlib.sha256(text.encode()).hexdigest(),
            'facts': copy.deepcopy(facts)}


def serving_row(kind, style, facts=None, *, section_index=None):
    if kind not in KINDS or style not in STYLES:
        raise ValueError('Declared field kind/style required')
    if kind == 'section':
        if facts is not None or type(section_index) is not int or section_index not in (0, 1):
            raise ValueError('Section uses selected-feature mean and explicit section index only')
        return {'kind': kind, 'style': style, 'sectionIndex': section_index}
    if section_index is not None:
        raise ValueError('Card row cannot contain section index')
    return {'kind': kind, 'style': style, 'source': public_record(facts)}


def teacher(kind, style, facts=None, section_index=0):
    """OFFLINE ONLY: not called by the decoder or its serving feature adapter."""
    style_index = STYLES.index(style)
    if kind == 'section':
        return (('Your selected collection', 'Browse these choices', 'Selected store choices'),
                ('More selected choices', 'Explore the collection', 'Current store selection'))[section_index][style_index]
    validate_facts(facts)
    name, color, price = facts['name'], facts['selectedColor'], '$' + money(facts['itemPrice'])
    if color is None:
        raise ValueError('Candidate requires an explicit selected public color')
    if kind == 'title':
        return (f'{name} in {color}', f'{color} {name}', f'{name}, {color}')[style_index]
    if kind == 'badge':
        return (f'{color} selected', f'In {color}', f'{color} displayed')[style_index]
    if kind == 'copy':
        return (f'Shown in {color} at {price}.', f'Color: {color}. Item: {price}.',
                f'{color} option at {price}.')[style_index]
    raise ValueError('Declared field kind required')


def field_rejections(kind, text, facts=None):
    """Bounded lexical/relationship safety screen, never a teacher comparison.

    Names remain canonical subtitles and prices remain canonical UI metadata.
    This candidate's short body states selected color and item price. It does
    not repeat the complete product name inside the body as copy-v2 required.
    """
    if kind not in KINDS:
        raise ValueError('Declared field kind required')
    try:
        lexical = tokens(text)
    except ValueError:
        return ['invalid-lexical-text']
    if len(lexical) >= SLOTS:
        return ['over-word-slot-cap']
    if kind == 'section':
        lower = [v.casefold() for v in lexical]
        return [] if 2 <= len(lower) <= 5 and all(v in SECTION_WORDS for v in lower) and len(set(lower)) == len(lower) else ['unsupported-section-words']
    validate_facts(facts)
    name, color = facts['name'], facts['selectedColor']
    if color is None:
        return ['missing-selected-color']
    remainder = text
    required = [color] + ([name] if kind == 'title' else [])
    for span in sorted(required, key=len, reverse=True):
        pattern = r'(?<![A-Za-z])' + re.escape(span) + r'(?![A-Za-z])'
        if len(re.findall(pattern, remainder)) != 1:
            return ['missing-or-repeated-bound-fact']
        remainder = re.sub(pattern, ' ', remainder)
    prices = re.findall(r'\$\d+(?:\.\d{1,2})?', remainder)
    if kind == 'copy':
        from decimal import Decimal
        if len(prices) != 1 or Decimal(prices[0][1:]) != Decimal(money(facts['itemPrice'])):
            return ['wrong-or-missing-item-price']
        remainder = remainder.replace(prices[0], ' ')
        if not text.endswith('.'):
            return ['unfinished-copy']
    elif prices:
        return ['unexpected-price']
    remaining_words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", remainder.casefold())
    if any(w not in SAFE_WORDS for w in remaining_words) or re.search(r'\d|\$', remainder):
        return ['unsupported-fact-or-word']
    # A color/price word bag is not a complete body. Require ordinary relation.
    lower = text.casefold()
    if kind == 'copy' and not (('shown in ' in lower and ' at ' in lower) or
                               ('color: ' in lower and '. item: ' in lower) or ' option at ' in lower):
        return ['missing-color-price-relation']
    return []


def heldout_colors(products):
    """Greedy public-lexicon split; never uses model outcomes/eval teachers.

    A heldout color's exact word forms must still have positive training
    supervision in another retained source binding. Products with no such
    choice retain all colors in TRAIN and are explicitly not held out.
    """
    selected = {}
    def coverage(held):
        count = Counter()
        for pi, product in enumerate(products):
            for ci, color in enumerate(product['colorVariants']):
                if held.get(pi) != ci:
                    count.update(tokens(product['name']) + tokens(color['colorName']))
        return count
    for pi, product in enumerate(products):
        candidates = sorted(range(len(product['colorVariants'])), key=lambda ci: ((ci - pi) % 3, ci))
        for ci in candidates:
            candidate = {**selected, pi: ci}
            count = coverage(candidate)
            if all(count[word] > 0 for p, c in candidate.items()
                   for word in tokens(products[p]['colorVariants'][c]['colorName'])):
                selected = candidate
                break
    return selected


def build(snapshot):
    products = snapshot['inputs']['products']
    if len(products) != 16 or any(len(p['colorVariants']) != 3 for p in products):
        raise ValueError('Pinned actual 16-product/48-variant catalog required')
    held = heldout_colors(products)
    records, index = [], {}
    for pi, product in enumerate(products):
        for ci, variant in enumerate(product['colorVariants']):
            facts = {'name': product['name'], 'category': product['category'],
                     'itemPrice': product['itemPrice'], 'selectedColor': variant['colorName']}
            record = public_record(facts)
            records.append(record)
            index[pi, ci] = record
    labels = {'train': [], 'eval': []}
    for (pi, ci), record in index.items():
        split = 'eval' if held.get(pi) == ci else 'train'
        for style in STYLES:
            for kind in KINDS[1:]:
                labels[split].append({'rowId': f'p{pi}-c{ci}-{style}-{kind}',
                    'input': serving_row(kind, style, record['facts']), 'featureKeys': [record['featureKey']],
                    'targetText': teacher(kind, style, record['facts'])})
    cases = {'train': [], 'eval': []}
    for split in cases:
        eligible = list(range(16)) if split == 'train' else sorted(held)
        if len(eligible) < 6:
            raise ValueError('Six distinct heldout source bindings required')
        for case_index in range(len(eligible)):
            selected = [eligible[(case_index + offset) % len(eligible)] for offset in range(6)]
            chosen = []
            for offset, pi in enumerate(selected):
                choices = [ci for ci in range(3) if held.get(pi) != ci]
                ci = choices[(case_index + offset) % len(choices)] if split == 'train' else held[pi]
                chosen.append(index[pi, ci])
            for style in STYLES:
                case_id = f'{split}-six-{case_index:02d}-{style}'
                rows = []
                for section in range(2):
                    field = f'S{section + 1}'
                    keyset = [r['featureKey'] for r in chosen[section * 3:section * 3 + 3]]
                    inp = serving_row('section', style, section_index=section)
                    rows.append({'field': field, 'input': inp, 'featureKeys': keyset})
                    labels[split].append({'rowId': f'{case_id}-{field}', 'input': inp, 'featureKeys': keyset,
                                         'targetText': teacher('section', style, section_index=section)})
                    for card in range(section * 3, section * 3 + 3):
                        record = chosen[card]
                        for prefix, kind in zip(('T', 'B', 'C'), KINDS[1:], strict=True):
                            rows.append({'field': f'{prefix}{card + 1}', 'input': serving_row(kind, style, record['facts']),
                                         'featureKeys': [record['featureKey']]})
                cases[split].append({'caseId': case_id, 'rows': rows,
                    'productIndices': selected, 'sourceBindingKeys': [r['featureKey'] for r in chosen]})
    # Lexicon includes public source words, not heldout teacher text. In this
    # split all eval label lexical items must also occur positively in TRAIN.
    vocabulary = [EOS, PAD] + sorted({w for row in labels['train'] for w in tokens(row['targetText'])} |
                                    {w for row in records for w in tokens(row['facts']['name']) + tokens(row['facts']['selectedColor'])})
    positive = Counter(w for row in labels['train'] for w in tokens(row['targetText']))
    eval_words = set(w for row in labels['eval'] for w in tokens(row['targetText']))
    audit = {'products': 16, 'publicFeatureRecords': len(records),
        'heldoutBindingCount': len(held), 'heldoutColorIndexByProduct': held,
        'productsWithoutHeldoutColor': [i for i in range(16) if i not in held],
        'splitPolicy': 'public color-word coverage; heldout combos only, catalog identities reused',
        'bindingOverlap': 0, 'vocabularySize': len(vocabulary),
        'evalWordsMissingPositiveTrainSupervision': sorted(w for w in eval_words if positive[w] == 0),
        'maximumTeacherLexicalTokens': max(len(tokens(r['targetText'])) for rows in labels.values() for r in rows),
        'fieldRows': {s: len(rows) for s, rows in labels.items()},
        'sixCardCases': {s: len(rows) for s, rows in cases.items()},
        'nativeCalls': 0, 'trainingUpdates': 0, 'pixelInputs': False, 'hiddenPersonaInputs': False}
    if audit['evalWordsMissingPositiveTrainSupervision']:
        raise ValueError('Heldout label words lack positive TRAIN supervision')
    for rows in labels.values():
        for row in rows:
            inp = row['input']; facts = inp.get('source', {}).get('facts')
            if len(tokens(row['targetText'])) >= SLOTS or field_rejections(inp['kind'], row['targetText'], facts):
                raise ValueError('Offline teacher fails bounded factual contract: ' + row['targetText'])
    return {'schemaVersion': VERSION, 'catalogInputsCanonicalSha256': MODEL_INPUTS_SHA256,
        'slotsPerField': SLOTS, 'vocabulary': vocabulary, 'featureRecords': records,
        'labels': labels, 'cases': cases, 'audit': audit}


def write_packet(packet, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    payloads = {'public-feature-contract.json': {'schemaVersion': VERSION, 'records': packet['featureRecords'],
        'expectedFeatureWidth': 4096, 'representation': 'frozen Qwen final hidden state, last-valid-token, unit L2',
        'modelRevision': '8b2b98c00a6b4d291155e4890773ca8f769aee53',
        'requiredNativeForwards': 48, 'copyAdapterEnabled': False, 'truncationAllowed': False},
        'vocabulary.json': packet['vocabulary'], 'audit.json': packet['audit']}
    for split in ('train', 'eval'):
        payloads[f'labels.{split}.json'] = packet['labels'][split]
        payloads[f'cases.{split}.json'] = packet['cases'][split]
    for name, value in payloads.items():
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    manifest = {'schemaVersion': VERSION, 'status': 'cpu-packet-only-not-native-quality',
        'catalogInputsCanonicalSha256': packet['catalogInputsCanonicalSha256'], 'slotsPerField': SLOTS,
        'sourceSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'files': {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in payloads},
        'inputLabelSeparation': 'public feature contract and cases never contain target text; labels separate',
        'scope': 'candidate only; no live promotion, template fallback, full UI or native latency claim'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    packet = build(load_snapshot())
    manifest = write_packet(packet, args.out)
    print(json.dumps({'out': str(args.out), 'audit': packet['audit'], 'status': manifest['status']}))


if __name__ == '__main__':
    main()
