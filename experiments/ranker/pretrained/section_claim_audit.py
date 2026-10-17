"""Independent CPU-only audit of all 22 preserved manifest section headings.

No model construction, generation, training, private labels, or acceptance changes.
Run: python -m experiments.ranker.pretrained.section_claim_audit
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .fusion import _field_claim_errors
from .generation_diagnosis import cases_and_saved

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATASET = ROOT / 'data/ranker/datasets/v2'
BENCHMARK = ROOT / 'data/ranker/pretrained/generation-diagnosis-v1/benchmark-reviewed.json'


def normalize(text):
    return ' '.join(re.findall(r'[a-z0-9]+', unicodedata.normalize('NFKC', text).lower()))


def tokens(text):
    return set(re.findall(r'[a-z]+|\d+(?:\.\d+)?', unicodedata.normalize('NFKC', text).lower()))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_heading(text, members, catalog):
    """Semantic tag test is independent of the existing proxy/error categories.

    A section tag is a universal claim: every actual member must carry it.
    Existing lexical errors are retained as policy evidence, never used to
    rescue acceptance or automatically label a claim a factual hallucination.
    """
    common = set.intersection(*(set(p['tags']) for p in members)) if members else set()
    universe = {tag for p in catalog.values() for tag in p['tags']}
    claimed = sorted(tag for tag in universe if text is not None and
                     re.search(r'\b' + re.escape(normalize(tag)) + r'\b', normalize(text)))
    violations = [{'claimedTag': tag, 'missingFromProductIds': [p['id'] for p in members if tag not in p['tags']]}
                  for tag in claimed if tag not in common]
    result = {'universalIntersection': sorted(common), 'claimedTags': claimed,
              'universalClaimViolations': violations, 'genericProxyErrors': [],
              'actualSingleProductErrors': None, 'unrecognizedProxyTokens': [],
              'unknownFramingTokens': [], 'singleProductProxyFalsePositive': False,
              'categories': []}
    if text is None:
        result['categories'].append('missing_or_unparseable_heading')
        return result
    proxy = {'title': 'Everyday travel choices', 'tags': sorted(common), 'category': 'backpacks'}
    proxy_errors = _field_claim_errors(text, proxy, catalog)
    result['genericProxyErrors'] = proxy_errors
    outside = set()
    for error in proxy_errors:
        if error.startswith('outside_factual_or_generic_lexicon:'):
            outside.update(error.split(':', 1)[1].split(','))
    tag_tokens = set().union(*(tokens(tag) for tag in universe)) if universe else set()
    unknown_framing = outside - tag_tokens
    if len(members) == 1:
        actual_errors = _field_claim_errors(text, members[0], catalog)
        result['actualSingleProductErrors'] = actual_errors
        result['singleProductProxyFalsePositive'] = bool(proxy_errors and not actual_errors and not violations)
        # Known single-product identity is not an unknown framing claim.
        unknown_framing -= tokens(members[0]['title'])
    result['unrecognizedProxyTokens'] = sorted(outside)
    result['unknownFramingTokens'] = sorted(unknown_framing)
    if violations:
        result['categories'].append('true_universal_tag_claim_failure')
    if result['singleProductProxyFalsePositive']:
        result['categories'].append('single_product_generic_or_numeric_proxy_false_positive')
    if unknown_framing:
        result['categories'].append('unknown_or_disallowed_framing_needs_review')
    if not proxy_errors and not violations:
        result['categories'].append('supported_section_wording_under_existing_policy')
    if proxy_errors and not result['categories']:
        result['categories'].append('other_policy_rejection_needs_review')
    return result


def self_tests():
    def p(pid, name, tags):
        return {'id': pid, 'title': name, 'category': 'backpacks', 'tags': tags,
                'description': name, 'price': 20, 'shippingCost': 0}
    grove = p('g', 'Grove 23L Daypack', ['water-resistant', 'compact'])
    cedar = p('c', 'Cedar 26L Daypack', ['durable', 'compact'])
    river = p('r', 'River 26L Daypack', ['durable', 'organizer'])
    field = p('f', 'Field 20L Daypack', ['water-resistant', 'roomy'])
    catalog = {x['id']: x for x in (grove, cedar, river, field)}
    bad = classify_heading('Compact durable daypacks', [grove, cedar, river], catalog)
    assert bad['universalIntersection'] == []
    assert {v['claimedTag'] for v in bad['universalClaimViolations']} == {'compact', 'durable'}
    assert bad['universalClaimViolations'][0]['missingFromProductIds'] == ['r']
    identity = classify_heading('Field 20L Daypack', [field], catalog)
    assert identity['singleProductProxyFalsePositive'] and not identity['actualSingleProductErrors']
    factual = classify_heading('Water-resistant roomy travel choices', [field], catalog)
    assert not factual['universalClaimViolations'] and not factual['genericProxyErrors']
    generic = classify_heading('Everyday travel choices', [grove, cedar, river], catalog)
    assert generic['claimedTags'] == [] and not generic['universalClaimViolations']
    assert classify_heading(None, [grove], catalog)['categories'] == ['missing_or_unparseable_heading']
    return {'checksPassed': 5, 'scope': 'CPU-only independent universal-tag and proxy repro assertions'}


def audit():
    benchmark = json.loads(BENCHMARK.read_text())
    # Public case reconstruction reads catalog and saved public case plans only;
    # it does not invoke the benchmark/model/private grading entry points.
    public_cases = {case['caseId']: case for case in cases_and_saved(DATASET)}
    records = []
    for index, row in enumerate(benchmark['records']):
        case = public_cases[row['caseId']]
        saved = row['arms']['manifest']
        raw = saved['generation']['text']
        parsed = {}
        for line in raw.splitlines():
            match = re.fullmatch(r'(S\d+): (.+)', line.strip())
            if match:
                parsed.setdefault(match[1], []).append(match[2])
        sections = []
        for number, section in enumerate(saved['pagePlan']['sections'], 1):
            key = f'S{number}'
            values = parsed.get(key, [])
            text = values[0] if len(values) == 1 else None
            members = [case['catalog'][card['productId']] for card in section['products']]
            detail = classify_heading(text, members, case['catalog'])
            sections.append({'field': key, 'sectionId': section['sectionId'], 'text': text,
                             'rawOccurrences': values, 'memberProductIds': section['productOrder'],
                             'members': [{'productId': p['id'], 'title': p['title'], 'tags': p['tags']} for p in members],
                             'originalFieldProvenance': section['heading'], **detail})
        records.append({'caseIndex': index, 'caseId': row['caseId'], 'source': row['source'],
                        'requestedCards': row['cards'], 'finishReason': saved['generation']['finishReason'],
                        'originalAllFieldsFulfilled': saved['quality']['allFieldsFulfilled'],
                        'hasTrueUniversalClaimFailure': any(s['universalClaimViolations'] for s in sections),
                        'hasSingleProductProxyFalsePositive': any(s['singleProductProxyFalsePositive'] for s in sections),
                        'sections': sections})
    cache_bad = {r['caseId'] for r in records if r['source'] == 'cache' and r['hasTrueUniversalClaimFailure']}
    assert cache_bad == {'cache-3-changed-profile-B', 'cache-5-changed-catalog-unavailable',
                         'cache-6-cold-catalog', 'cache-7-warm-catalog',
                         'cache-8-cold-catalog', 'cache-9-warm-catalog'}, cache_bad
    false_proxy = {r['caseId'] for r in records if r['hasSingleProductProxyFalsePositive']}
    assert false_proxy == {'evidence-r-36f617c0fe5c7f6164', 'evidence-r-4171cae29da747fcef'}, false_proxy
    assert len(records) == 22 and sum(r['source'] == 'cache' for r in records) == 10
    all_sections = [s for r in records for s in r['sections']]
    checks = self_tests()
    checks['preservedPopulationAssertionsPassed'] = 3
    return {'schemaVersion': 'section-claim-audit.v1', 'synthetic': True,
            'scope': 'All 22 prior manifest outputs only; no current reliability-comparison outputs or private persona labels.',
            'acceptanceChanged': False, 'gpuOrModelsLoaded': False,
            'population': {'responses': len(records), 'cacheResponses': 10, 'evidenceResponses': 12,
                           'sectionFields': len(all_sections)},
            'summary': {'trueUniversalFailureResponses': sum(r['hasTrueUniversalClaimFailure'] for r in records),
                        'cacheTrueUniversalFailureResponses': len(cache_bad),
                        'evidenceSingleProductProxyFalsePositiveResponses': len(false_proxy),
                        'trueUniversalFailureSectionFields': sum(bool(s['universalClaimViolations']) for s in all_sections),
                        'singleProductProxyFalsePositiveSectionFields': sum(s['singleProductProxyFalsePositive'] for s in all_sections),
                        'sectionCategories': dict(Counter(c for s in all_sections for c in s['categories']))},
            'assertions': checks,
            'inputs': {str(p.relative_to(ROOT)): digest(p) for p in (
                BENCHMARK, DATASET / 'catalog.json', HERE / 'fusion.py',
                HERE / 'generation_diagnosis.py', Path(__file__).resolve())},
            'method': ['Tag claims are detected against known public catalog tags, normalized for punctuation/case.',
                       'Each claimed tag must be present on every actual section member, using intersection rather than union.',
                       'Proxy false positives require exactly one member, proxy rejection, actual-product lexical acceptance, and no universal-tag violation.',
                       'Existing proxy errors and original provenance remain visible; no text is rescued or accepted by this diagnostic.'],
            'limitations': ['Finite tag phrase audit, not unrestricted semantic truth verification.',
                            'Actual-product lexical acceptance alone is not a general proof of factuality.',
                            'Historical inspected synthetic cases; this does not validate new prompt results or generalization.'],
            'records': records}


def write_results(report):
    (HERE / 'SECTION_CLAIM_AUDIT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# Independent section-claim audit', '',
             'CPU-only results for all 22 preserved prior manifest responses. No model/GPU was loaded; no validator, acceptance rule, generated output, or current comparison source was modified.', '',
             'Run: `.tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.section_claim_audit`', '',
             '## Findings', '',
             f"- {report['population']['sectionFields']} section fields audited across 10 cache and 12 evidence responses.",
             '- Six cache responses contain true universal tag-claim failures. This is not explained by conservative generic-word rejection.',
             '- The six-card evidence response adds another true universal failure response.',
             '- Two single-product evidence responses are generic/numeric section-proxy false positives: the actual selected Field 20L product accepts the wording, while the generic section proxy drops its brand/capacity.',
             '- All assertions passed: five independent synthetic checks and three preserved-population checks. Existing strict acceptance remains unchanged.', '',
             '## Actual failing/proxy examples', '',
             '| Case / field | Actual heading | Member facts | Finding |',
             '| --- | --- | --- | --- |']
    for row in report['records']:
        for section in row['sections']:
            if not section['universalClaimViolations'] and not section['singleProductProxyFalsePositive']:
                continue
            facts = '; '.join(p['title'] + ': ' + ', '.join(p['tags']) for p in section['members'])
            if section['singleProductProxyFalsePositive']:
                finding = 'Proxy false positive; actual selected product lexical check passes.'
            else:
                finding = 'Not universal: ' + ', '.join(v['claimedTag'] for v in section['universalClaimViolations'])
            lines.append('| ' + row['caseId'] + ' / ' + section['field'] + ' | ' + section['text'].replace('|', '\\|') + ' | ' + facts + ' | ' + finding + ' |')
    lines += ['', '## Interpretation', '',
              'Grove 23L is compact but not durable; River 26L is durable/organizer but not compact. Their shared section has an empty tag intersection, so “compact durable” is a genuine universal-claim error. Field 14L and Ridge 23L are lightweight/roomy, but their section also contains River 14L (durable/compact), making “lightweight roomy” false for the whole section.', '',
              'Changed-profile B still selects River 26L even though the observable preferences are lightweight/roomy. A heading claiming lightweight/roomy for that selected River is incorrect. Product locking can improve output faithfulness without repairing recommendation selection.', '',
              'The two Field 20L headings are factual single-product identity/attribute statements: “Field 20L Daypack” and “Field 20L Daypack is a water-resistant roomy bag.” They remain rejected under the existing generic section proxy. This audit labels the mechanism without relaxing acceptance.', '',
              '“Options” in the six-card evidence heading is additionally an unknown framing token; that lexical issue does not erase the separate true compact/durable universal-claim error. Missing/unknown-format fields and unknown framing are separate categories in the JSON.', '',
              'The proposed next prompt should expose the UNIVERSAL intersection explicitly; empty intersections allow generic framing only. Selected identity/capacity and true product facts must remain distinct from the narrower observable preference/history evidence used for badges. No outcome from that next prompt is included here.', '',
              'The JSON contains every section’s actual product IDs/tags, detected claimed tags, missing-member counterexamples, original provenance, proxy errors, actual-product checks and source hashes. Tag phrase matching is a bounded diagnostic, not unrestricted semantic factuality proof.']
    (HERE / 'SECTION_CLAIM_AUDIT.md').write_text('\n'.join(lines) + '\n')


def main():
    report = audit()
    write_results(report)
    print(json.dumps({'population': report['population'], 'summary': report['summary'], 'assertions': report['assertions']}, indent=2))


if __name__ == '__main__':
    main()
