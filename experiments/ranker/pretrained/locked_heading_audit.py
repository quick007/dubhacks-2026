"""CPU-only, read-only input audit of the completed locked-evidence headings.

Writes only LOCKED_HEADING_AUDIT.json/.md beside this independent audit module.
No generation, model construction, training, validator edits, or text acceptance.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .section_claim_audit import classify_heading, normalize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / 'data/ranker/pretrained/full-page-reliability-v1'
CATEGORIES = {
    'accepted': 'accepted_supported',
    'proxy': 'single_product_proxy_false_positive',
    'genuine': 'genuine_universal_or_section_binding_failure',
    'policy': 'other_lexical_or_semantic_policy_rejection',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity_binding(text, members, catalog):
    """Titles identify names, not necessarily unique SKUs; expose all matching IDs.

    A named title absent from EVERY member title is a definite section-binding
    error. Duplicate catalog titles alone never establish a wrong SKU.
    """
    wording = normalize(text or '')
    named = {}
    for product in catalog.values():
        title = product['title']
        if re.search(r'\b' + re.escape(normalize(title)) + r'\b', wording):
            named.setdefault(title, []).append(product['id'])
    member_titles = {product['title'] for product in members}
    wrong = sorted(set(named) - member_titles)
    return {'namedCatalogTitles': [{'title': title, 'matchingCatalogProductIds': sorted(ids)}
                                  for title, ids in sorted(named.items())],
            'wrongSectionIdentityTitles': wrong,
            'mentionsOnlyOneMemberIdentity': len(named) == 1 and not wrong and len(members) > 1}


def audit():
    benchmark_path = RUN / 'benchmark.json'
    cases_path = RUN / 'case-plan.json'
    benchmark = json.loads(benchmark_path.read_text())
    cases = {case['caseId']: case for case in json.loads(cases_path.read_text())['cases']}
    rows = []
    for case_index, record in enumerate(benchmark['records']):
        saved = record['arms']['lockedEvidence']
        public = cases[record['caseId']]
        catalog = public['catalog']
        raw = saved['generation']['text']
        card_count = sum(len(section['products']) for section in saved['pagePlan']['sections'])
        expected_labels = [f'S{i}' for i in range(1, len(saved['pagePlan']['sections']) + 1)] + [
            f'{kind}{i}' for i in range(1, card_count + 1) for kind in ('T', 'B')]
        actual_labels = []
        for line in raw.splitlines():
            match = re.fullmatch(r'([STB]\d+): (.+)', line)
            assert match is not None, ('Unexpected output format', record['caseId'], line)
            actual_labels.append(match[1])
        assert actual_labels == expected_labels, ('Wrong output field order', record['caseId'])
        parsed = {}
        for line in raw.splitlines():
            match = re.fullmatch(r'(S\d+): (.+)', line.strip())
            if match:
                parsed.setdefault(match[1], []).append(match[2])
        for index, section in enumerate(saved['pagePlan']['sections'], 1):
            field = f'S{index}'
            assert len(parsed.get(field, [])) == 1, (record['caseId'], field, parsed)
            text = parsed[field][0]
            members = [catalog[card['productId']] for card in section['products']]
            claims = classify_heading(text, members, catalog)
            binding = identity_binding(text, members, catalog)
            capsule_tags = saved['capsule']['sections'][index - 1]['universalTags']
            assert sorted(capsule_tags) == claims['universalIntersection'], 'Saved capsule intersection mismatch'
            if claims['universalClaimViolations'] or binding['wrongSectionIdentityTitles']:
                category = CATEGORIES['genuine']
            elif claims['singleProductProxyFalsePositive']:
                category = CATEGORIES['proxy']
            elif claims['genericProxyErrors']:
                category = CATEGORIES['policy']
            else:
                category = CATEGORIES['accepted']
            assert (category == CATEGORIES['accepted']) == (section['heading']['source'] == 'generated')
            rows.append({'caseIndex': case_index, 'caseId': record['caseId'], 'source': record['source'],
                         'requestedCards': record['cards'], 'field': field, 'sectionId': section['sectionId'],
                         'rawHeading': text, 'rawCompleteGeneration': raw,
                         'finishReason': saved['generation']['finishReason'],
                         'originalFieldProvenance': section['heading'],
                         'originalPageFulfilled': saved['quality']['allFieldsFulfilled'],
                         'members': [{'productId': p['id'], 'title': p['title'], 'tags': p['tags']} for p in members],
                         'savedCapsuleUniversalTags': capsule_tags, 'exclusiveCategory': category,
                         **binding, 'claimAudit': claims})
    counts = Counter(row['exclusiveCategory'] for row in rows)
    expected = {CATEGORIES['accepted']: 4, CATEGORIES['proxy']: 10,
                CATEGORIES['genuine']: 9, CATEGORIES['policy']: 2}
    assert dict(counts) == expected, dict(counts)
    assert len(benchmark['records']) == 22 and len(rows) == 25
    assert all(row['finishReason'] == 'stop' for row in rows)
    assert benchmark['manifestBaselineRawMatchesPrior'] == 22
    wrong_identity = [row for row in rows if row['wrongSectionIdentityTitles']]
    assert [(row['caseIndex'], row['field']) for row in wrong_identity] == [(8, 'S2'), (9, 'S2')]
    proxy_rows = [row for row in rows if row['exclusiveCategory'] == CATEGORIES['proxy']]
    assert all(len(row['members']) == 1 and not row['claimAudit']['actualSingleProductErrors']
               for row in proxy_rows)
    files = [benchmark_path, cases_path, Path(__file__).resolve(), HERE / 'section_claim_audit.py',
             HERE / 'fusion.py', HERE / 'full_page_reliability.py', HERE / 'evidence_plan.py', HERE / 'page_plan.py']
    return {'schemaVersion': 'locked-heading-audit.v1', 'synthetic': True,
            'scope': 'All completed lockedEvidence section fields; existing manifest baseline and strict acceptance are preserved.',
            'gpuOrModelsLoaded': False, 'modelInvocations': 0, 'acceptanceChanged': False,
            'population': {'responses': 22, 'sectionFields': 25, 'cacheSectionFields': 12,
                           'evidenceSectionFields': 13},
            'exclusiveCounts': dict(counts),
            'acceptedBySource': {source: sum(row['source'] == source and row['exclusiveCategory'] == CATEGORIES['accepted']
                                             for row in rows) for source in ('cache', 'evidence')},
            'genuineFailureDetails': {'wrongSectionIdentityFields': len(wrong_identity),
                                     'nonuniversalAttributeFields': sum(bool(row['claimAudit']['universalClaimViolations'])
                                                                       for row in rows)},
            'baselineReplayMatches': benchmark['manifestBaselineRawMatchesPrior'],
            'assertions': {'exclusiveCountsPass': True, 'populationPass': True, 'allNativeStopPass': True,
                           'savedUniversalIntersectionsPass': True, 'storedAcceptanceUnchangedPass': True,
                           'twoWrongSectionIdentitiesPass': True, 'allTenProxyActualProductChecksPass': True,
                           'independentExactOutputFieldOrderPass': True},
            'inputs': {str(path.relative_to(ROOT)): sha256(path) for path in files},
            'interpretation': [
                'Zero accepted headings applies only to the 13 evidence headings, not all locked headings: 4/25 are accepted overall.',
                'Ten rejections are factual single-product names/capacities/attributes rejected by the generic section proxy.',
                'Nine fields violate universal/member-binding constraints; two name a product absent from that section, although the named product facts may be catalog-true in isolation.',
                'Two further fields fail lexical/policy checks. Existing rejection is retained; these diagnostics never rescue text.',
                'All locked responses stop normally and the manifest baseline replays 22/22, so this is not evidence of field-format failure or stale-cache corruption.',
                'The locked prompt encouraged full product-name sentences in headings while the strict generic proxy omits brand/capacity identity facts.'],
            'limitations': ['Finite catalog-tag/identity/lexical analysis, not general semantic truth proof.',
                            'Historical inspected synthetic cases; no new-generation or human conversion claim.',
                            'Prompt serialization combines multiple instructions, so causal effects of individual clauses are not identified.'],
            'fields': rows}


def write_results(report):
    (HERE / 'LOCKED_HEADING_AUDIT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# Independent locked-heading audit', '',
             'The completed locked-evidence arm has **4/25 accepted headings**: 4/12 cache headings and 0/13 evidence headings. Zero does not describe the overall arm.', '',
             'Reproduce from the repository root:', '', '```sh',
             'PYTHONDONTWRITEBYTECODE=1 .tools/pretrained-venv/bin/python -m experiments.ranker.pretrained.locked_heading_audit',
             '```', '',
             'Only this module and its JSON/Markdown outputs are written. No GPU/model loading, generation, training, validator/acceptance edits, core experiment edits or commits occur.', '',
             '## Mutually exclusive results', '',
             '| Classification | Heading fields |', '| --- | ---: |',
             '| Accepted supported wording | 4 |',
             '| Single-product generic/numeric proxy false positive | 10 |',
             '| Genuine universal/member-binding failure | 9 |',
             '| Other lexical or semantic-policy rejection | 2 |', '',
             'All count, population, native-stop, saved intersection and original-acceptance assertions pass. All 22 preserved manifest baseline outputs replay exactly. The 25-field exclusive classification contains no double counting.', '',
             '## Exact examples and counterexamples', '',
             '| Case / field | Raw heading | Actual members and tags | Classification |',
             '| --- | --- | --- | --- |']
    for field in report['fields']:
        members = '; '.join(p['title'] + ' [' + ', '.join(p['tags']) + ']' for p in field['members'])
        lines.append('| ' + str(field['caseIndex']) + ' / ' + field['field'] + ' | ' + field['rawHeading'].replace('|', '\\|')
                     + ' | ' + members + ' | ' + field['exclusiveCategory'] + ' |')
    lines += ['', '## What the failures mean', '',
              '**Proxy false positives:** “The River 26L Daypack is a durable, organizer backpack.” (case 3) is factual for the selected River 26L, but the generic section proxy rejects `River/26L`. Similarly, all nine single-product evidence proxy cases retain correct selected identities/attributes but are rejected for brand/capacity tokens. These are not established fabricated product facts; all remain rejected under the unchanged strict policy.', '',
              '**Universal/member failures:** River 26L lacks compact (case 5). The three-card section has Grove 23L without durable and River 26L without compact, so “durable, compact” is not universal (cases 6/7). In cases 8/9, S2 says “The Cedar 26L Daypack is a durable and compact option.” but section 2 contains Field 14L, Ridge 23L and River 14L, not Cedar 26L. That is wrong section identity/binding; Cedar’s isolated attribute facts are catalog-true. The case-21 multi-card headings also assert attributes not shared by every section member.', '',
              '**Other lexical/policy cases:** case 16 says “The Cedar 26L Daypack is a practical choice for carrying your laptop and other essentials.” Its actual-product lexical check rejects `other/essentials`; the generic proxy additionally rejects brand/capacity. Case 20 says “This choice meets needs of everyday travel and short trips.” and is rejected for `meets/short`, without a detected unsupported catalog-tag claim. These diagnoses do not authorize broader wording or establish universal semantic safety.', '',
              '**Style conflict:** the locked serialization emphasizes selected identity and factual tags globally. Many headings consequently become full product-name sentences, while the existing generic heading proxy intentionally has no product identity/capacity vocabulary. Missing or malformed output and native truncation do not explain this run: all 22 locked responses stop normally and preserve exact field order.', '',
              'The JSON includes exact complete raw generations, case/section/product IDs, actual member tags, claimed-tag intersections and missing-member counterexamples, named-identity evidence, original provenance, proxy-versus-actual lexical checks and SHA-256 hashes. Counts describe this fixed synthetic run only; they imply neither human conversion lift nor generalization.']
    (HERE / 'LOCKED_HEADING_AUDIT.md').write_text('\n'.join(lines) + '\n')


def main():
    report = audit()
    write_results(report)
    print(json.dumps({key: report[key] for key in ('population', 'exclusiveCounts', 'acceptedBySource',
                                                 'genuineFailureDetails', 'assertions')}, indent=2))


if __name__ == '__main__':
    main()
