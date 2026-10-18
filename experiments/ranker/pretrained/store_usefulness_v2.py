"""CPU-only offline candidate; never changes serving text, gates or readiness.

The pinned v1 practical screen is reused verbatim. Only C-field dollar notation
is normalized in a private scoring string after exact Decimal source binding.
This conservative lexical score is not semantic proof or a human usefulness study.
"""
from __future__ import annotations

import argparse
import ast
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

VERSION = 'store-usefulness-offline-candidate.v2'
ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'store_ar_comparison_results/v1'
PINS = {
    'store_demo_engine.py': 'a1e52eebc7ea90103d27309b5d960ae4676db28c4e207ccec3ebaa3a3564639c',
    'store_copy_data_v2.py': 'ac424d601d8dc97d480d3c22ed0175de002e46281c7c35f9054f00bb32a8082e',
}
NUMBER = r'(?:0|[1-9][0-9]*)(?:\.[0-9]{1,2})?'
# Plain English "dollars" is interpreted as the same store dollar currency as $.
# USD/foreign currencies, grouped/signed amounts and words-as-numbers abstain.
WORD_PRICE = re.compile(r'(?<![\w$€£¥.,+\-])(' + NUMBER + r')[ \t]+dollars\b', re.I)
PRESERVE_RAW_REASONS = {'nonplain-protocol-or-length', 'uppercase-value',
    'native-EOS-or-token-cap', 'unfinished-body', 'repetition'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def frozen_screen():
    """Extract reviewed CPU definitions, not native/model module imports."""
    env = {'re': re, 'Decimal': Decimal, 'InvalidOperation': InvalidOperation}
    wanted = {'store_copy_data_v2.py': {'CAPS', 'native_receipt'},
              'store_demo_engine.py': {'FRAMING', 'UNSUPPORTED', '_span', 'practical_rejections'}}
    for name, names in wanted.items():
        path = ARCHIVE / 'source' / name
        if sha(path) != PINS[name]:
            raise ValueError('Frozen screen source hash mismatch: ' + name)
        nodes = []
        for node in ast.parse(path.read_text()).body:
            defined = ({node.name} if isinstance(node, ast.FunctionDef) else
                {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else set())
            if defined & names:
                nodes.append(node)
        if len(nodes) != len(names):
            raise ValueError('Frozen screen definitions incomplete: ' + name)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env['practical_rejections']


def score_field(field, text, card, products, *, row=None):
    """Keep raw receipt/text intact; only source-bound price syntax can differ."""
    if not re.fullmatch(r'[STBC][1-6]', field) or (field[0] == 'S' and field not in ('S1', 'S2')):
        raise ValueError('Unknown copy field')
    screen = frozen_screen()
    original = screen(field, text, card, products, row=row)
    normalized, changes, extra = text, [], []
    if field[0] == 'C' and isinstance(text, str):
        facts = card['facts']
        amount = Decimal(str(facts['itemPrice']))
        matches = list(WORD_PRICE.finditer(text))
        # Dollar-symbol matches ending the sentence need a separate boundary:
        # the frozen checker remains authoritative for malformed $ syntax.
        symbol_count = len(re.findall(r'\$\d', text))
        if len(matches) + symbol_count > 1:
            extra.append('multiple-item-price-mentions')
        for match in matches:
            if Decimal(match.group(1)) != amount:
                extra.append('incorrect-equivalent-dollar-price')
            else:
                changes.append({'originalSpan': match.group(0), 'start': match.start(),
                    'end': match.end(), 'sourceItemPrice': str(amount), 'notationOnly': True})
        if not extra:
            for change in reversed(changes):
                normalized = normalized[:change['start']] + '$' + format(amount, '.2f') + normalized[change['end']:]
        else:
            changes = []
    reasons = sorted(set(screen(field, normalized, card, products, row=row)) |
        (set(original) & PRESERVE_RAW_REASONS) | set(extra))
    if row is not None and row.get('text') != text:
        reasons = sorted(set(reasons) | {'raw-text-does-not-match-original-row'})
    return {'field': field, 'rawText': text, 'accepted': not reasons, 'reasons': reasons,
        'frozenPracticalAccepted': not original, 'frozenPracticalReasons': original,
        'equivalentPriceSpansForScoringOnly': changes,
        'nativeEvidence': 'original independent field EOS/cap receipt checked' if row is not None else
            'no per-field EOS claim; caller must retain whole AR stream completion evidence'}


def _archived_result_path(archive, record):
    """Resolve within the durable archive, never trust old absolute data paths."""
    old = Path(record['resultPath']).parts
    marker = 'store-ar-comparison-v1-attempt-1'
    rel = Path(*old[old.index(marker) + 1:])
    path = archive / rel
    if not path.resolve().is_relative_to(archive.resolve()):
        raise ValueError('Receipt escapes archive')
    return path


def score_archive(archive=ARCHIVE):
    archive = Path(archive)
    results = json.loads((archive / 'results.json').read_text())
    if results.get('status') != 'completed-comparison' or len(results['records']) != 6:
        raise ValueError('Completed fixed six-arm comparison required')
    if ({r['arm'] for r in results['records']} != {'A', 'B'} or
        any(sum(r['arm'] == arm for r in results['records']) != 3 for arm in ('A', 'B'))):
        raise ValueError('Exactly three timed arms per system required')
    catalog_path = ROOT / 'store_catalog_v1/model_inputs.json'
    products = json.loads(catalog_path.read_text())['products']
    arms = []
    for rec in results['records']:
        path = _archived_result_path(archive, rec)
        item = {'caseId': rec['caseId'], 'arm': rec['arm'], 'receiptArchiveRelativePath': str(path.relative_to(archive)),
            'frozenUsableFirstThreeMs': rec.get('usableFirstThreeMs'),
            'frozenUsableFullMs': rec.get('usableFullMs'), 'observedV2ReadinessMs': None}
        try:
            item['receiptSha256'] = sha(path)
            data = json.loads(path.read_text())
            if rec['arm'] == 'A':
                if data.get('finishReason') != 'stop' or len(data['full']['rawFields']) != 20:
                    raise ValueError('Whole AR stream incomplete')
                public = [c for s in data['full']['payload']['sections'] for c in s['cards']]
                rows = {f['field']: {'text': f['text']} for f in data['full']['rawFields']}
                cards = [{'productIndex': c['productIndex'], 'facts': {'name': c['originalName'],
                    'selectedColor': c['selectedColor'], 'itemPrice': c['itemPrice'], 'category': c['catalogCategory']}}
                    for c in public]
            else:
                if data.get('status') != 'completed':
                    raise ValueError('Independent native request incomplete')
                cards = data['cards']; rows = {r['fields'][0]: r for r in data['rawRows']}
            if len(cards) != 6 or len({c['productIndex'] for c in cards}) != 6 or len(rows) != 20:
                raise ValueError('Incomplete six-card/twenty-field roster')
            for card in cards:
                fact = card['facts']; source = products[card['productIndex']]
                if (fact['name'] != source['name'] or fact['itemPrice'] != source['itemPrice'] or
                    fact['category'] != source['category'] or fact['selectedColor'] not in
                    [v['colorName'] for v in source['colorVariants']]):
                    raise ValueError('Card facts do not match public catalog')
            scored = []
            for field, row in rows.items():
                card = None if field[0] == 'S' else cards[int(field[1:]) - 1]
                scored.append(score_field(field, row['text'], card, products,
                    row=row if rec['arm'] == 'B' else None))
            for label, count in [('firstThree', 3), ('fullSix', 6)]:
                core = [f for f in scored if f['field'][0] in ('T', 'C') and int(f['field'][1:]) <= count]
                optional = [f for f in scored if
                    (f['field'][0] == 'S' and int(f['field'][1:]) <= (count+2)//3) or
                    (f['field'][0] == 'B' and int(f['field'][1:]) <= count)]
                item[label] = {'coreTitleBodyPass': len(core) == 2 * count and all(f['accepted'] for f in core),
                    'coreAccepted': sum(f['accepted'] for f in core), 'coreRequired': 2 * count,
                    'optionalAccepted': sum(f['accepted'] for f in optional), 'optionalRequired': len(optional),
                    'allFieldsPass': all(f['accepted'] for f in core + optional)}
            item.update(status='scored-offline', fields=scored)
        except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
            item.update(status='failed-offline-score', error=str(error), firstThree={'coreTitleBodyPass': False},
                        fullSix={'coreTitleBodyPass': False})
        arms.append(item)
    assert len({(a['caseId'], a['arm']) for a in arms}) == 6
    summary = {}
    for arm in ('A', 'B'):
        selected = [a for a in arms if a['arm'] == arm]
        summary[arm] = {'fixedTimedArms': 3, 'scoredArms': sum(a['status'] == 'scored-offline' for a in selected),
            'firstThreeCorePass': sum(a['firstThree']['coreTitleBodyPass'] for a in selected),
            'fullSixCorePass': sum(a['fullSix']['coreTitleBodyPass'] for a in selected),
            'acceptedFields': sum(f['accepted'] for a in selected for f in a.get('fields', [])),
            'requiredFields': 60,
            'notationAcceptedOnlyByV2': sum(f['accepted'] and not f['frozenPracticalAccepted'] for a in selected for f in a.get('fields', []))}
    return {'version': VERSION, 'execution': 'CPU-only prospective offline candidate on archived raw outputs',
        'observedV2Readiness': False, 'originalTextChanged': False, 'oldGatesOrTimestampsChanged': False,
        'inferenceHiddenTruthUsed': False, 'warmupsScored': 0, 'fixedTimedArms': 6,
        'scope': 'Source-bound equivalent dollar notation plus unchanged frozen lexical safety checks; title/body core separate from optional heading/badge; relevance remains independent.',
        'limitations': ['This candidate was designed after inspecting v1 failure modes; not a preregistered independent benchmark.',
            'No live readiness or speedup may be inferred; old usable timestamps stay null.',
            'Not comprehensive English factuality, relevance, conversion, budget or hard-constraint evaluation.',
            'Only plain $ numbers and decimal-number dollars are supported; USD/other currencies and number words abstain.'],
        'provenance': {'scorerSha256': sha(Path(__file__)), 'frozenScreenSources': PINS,
            'archivePath': str(archive),
            'definitionReuse': 'AST extraction of exact hash-pinned CPU functions/constants; no model/runtime imports',
            'resultsSha256': sha(archive / 'results.json'), 'catalogSha256': sha(catalog_path)},
        'summary': summary, 'arms': arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior scored candidates')
    report = score_archive(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
