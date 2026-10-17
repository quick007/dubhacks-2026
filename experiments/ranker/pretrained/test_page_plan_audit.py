"""CPU-only adversarial page-plan audit. Failures are actionable validator gaps.

These tests assert intended boundaries; no GPU/model load or generation occurs.
"""
import copy
import json
from pathlib import Path
import unittest

from .page_plan import fact_id, history_reasons, validate_page_plan
from .evidence_plan import validate_generated
from . import test_page_plan as _base


class PagePlanAuditTests(unittest.TestCase):
    setUp = _base.PagePlanTests.setUp
    plan = _base.PagePlanTests.plan

    def complete(self):
        plan = self.plan()
        completed, quality = validate_generated('S1: Everyday travel choices\nT1: Your compact travel companion\nB1: NONE', 'stop', plan, self.request, self.catalog)
        self.assertTrue(quality['allFieldsFulfilled'])
        return completed

    def test_contract_fixture_semantics_and_declared_field_shapes(self):
        fixture = json.loads(Path(__file__).with_name('page-plan.v1.fixture.json').read_text())
        catalog = {product['id']: product for product in fixture['catalog']['products']}
        plan = fixture['pagePlan']
        self.assertTrue(validate_page_plan(plan, fixture['request'], catalog))
        self.assertIs(type(plan['requestedCardCount']), int)
        self.assertEqual(plan['recordId'], fixture['request']['recordId'])
        self.assertEqual(plan['sectionOrder'], [s['sectionId'] for s in plan['sections']])
        self.assertEqual(sum(len(s['products']) for s in plan['sections']), 6)
        for section in plan['sections']:
            self.assertEqual(section['productOrder'], [p['productId'] for p in section['products']])
            for value in (section['heading'], section['copy'], *(v for c in section['products'] for v in (c['title'], c['badge']))):
                self.assertIn(value['source'], ('pending','generated','rejected','none','approved-fallback'))
                self.assertTrue(value['text'] is None or isinstance(value['text'], str))
        with self.assertRaises(ValueError):
            validate_page_plan(plan, fixture['request'], catalog, True)

    def test_cross_section_duplicate_needs_explicit_allow(self):
        plan = self.plan(); plan['requestedCardCount'] = 3; plan['shortfall'] = 1
        section = copy.deepcopy(plan['sections'][0]); section['sectionId'] = 'section-2'
        plan['sections'].append(section); plan['sectionOrder'].append('section-2')
        with self.assertRaises(ValueError): validate_page_plan(plan, self.request, self.catalog)
        plan['duplicatePolicy'] = 'explicit-allow'
        self.assertTrue(validate_page_plan(plan, self.request, self.catalog))
        section['products'] *= 2; section['productOrder'] *= 2; plan['shortfall'] = 0
        with self.assertRaises(ValueError): validate_page_plan(plan, self.request, self.catalog)

    def test_unshared_heading_attribute_is_rejected(self):
        fixture = json.loads(Path(__file__).with_name('page-plan.v1.fixture.json').read_text())
        catalog = {p['id']: p for p in fixture['catalog']['products']}
        plan = fixture['pagePlan']; section = plan['sections'][0]
        common = set.intersection(*(set(catalog[c['productId']]['tags']) for c in section['products']))
        noncommon = next(t for c in section['products'] for t in catalog[c['productId']]['tags'] if t not in common)
        section['heading'] = {'text': noncommon + ' travel choices', 'source': 'generated'}
        with self.assertRaises(ValueError): validate_page_plan(plan, fixture['request'], catalog)

    def test_private_unknown_weak_and_canonical_future_history_cannot_justify_badge(self):
        baseline = self.plan()
        changed = copy.deepcopy(self.request)
        changed['hiddenPreferences'] = ['compact']; changed['privateTruth'] = 'compact'
        changed['history'] = [
            {'at':'2099-01-01T00:00:00Z','productId':'p','type':'save','confidence':1},
            {'at':'2026-09-01T00:00:00Z','productId':'unknown','type':'save','confidence':1},
            {'at':'2026-09-01T00:00:00Z','productId':'p','type':'view','confidence':1},
            {'at':'2026-09-02T00:00:00Z','productId':'p','type':'save','confidence':.35}]
        self.assertEqual({}, history_reasons(changed, self.catalog, self.product))
        self.assertEqual(baseline['sections'][0]['products'][0]['badgeDecision'], 'NONE')
        invented = copy.deepcopy(baseline)
        invented['sections'][0]['products'][0]['observedHistoryReasonIds'] = ['observed:made-up']
        with self.assertRaises(ValueError): validate_page_plan(invented, changed, self.catalog)

    def test_truncation_markup_and_duplicate_lines_do_not_fulfill_page(self):
        good = 'S1: Everyday travel choices\nT1: Your compact travel companion\nB1: NONE'
        for raw, finish in ((good,'length'), (good+'\nT1: Compact travel pack','stop'), (good.replace('Your','<b>Your</b>'),'stop')):
            _, quality = validate_generated(raw, finish, self.plan(), self.request, self.catalog)
            self.assertFalse(quality['allFieldsFulfilled'])

    # Intended-rejection tests below expose concrete gaps in the audited source.
    def test_completed_generated_provenance_requires_nonempty_text(self):
        plan = self.complete(); plan['sections'][0]['products'][0]['title'] = {'text':None,'source':'generated'}
        with self.assertRaises(ValueError): validate_page_plan(plan, self.request, self.catalog, True)

    def test_fallback_provenance_requires_an_approved_title(self):
        plan = self.complete(); plan['sections'][0]['products'][0]['title'] = {'text':'Your compact travel companion','source':'approved-fallback'}
        with self.assertRaises(ValueError): validate_page_plan(plan, self.request, self.catalog, True)

    def test_record_identity_must_match_observable_request(self):
        plan = self.plan(); plan['recordId'] = 'different-request'
        with self.assertRaises(ValueError): validate_page_plan(plan, self.request, self.catalog)

    def test_explicit_negative_preference_is_not_positive_badge_support(self):
        self.request['context']['statedPreferences'] = ['not compact']
        self.assertEqual(self.plan()['sections'][0]['products'][0]['badgeDecision'], 'NONE')

    def test_offset_encoded_future_save_is_not_observed(self):
        # 23:30 at UTC-02 is 01:30Z on Oct 1, AFTER the midnight request cutoff.
        self.request['history'] = [{'at':'2026-09-30T23:30:00-02:00','productId':'p','type':'save','confidence':.9}]
        self.assertEqual(history_reasons(self.request, self.catalog, self.product), {})

    def test_fact_reference_cannot_launder_unobserved_attribute_alignment(self):
        self.product['tags'].append('durable')
        self.request['context']['statedPreferences'] = ['compact']
        plan = self.plan(); card = plan['sections'][0]['products'][0]
        card['approvedFactIds'] = [fact_id('p','durable')]
        with self.assertRaises(ValueError): validate_page_plan(plan, self.request, self.catalog)

    def test_no_observed_support_does_not_report_perfect_observed_alignment(self):
        _, quality = validate_generated('S1: Everyday travel choices\nT1: Your compact travel companion\nB1: NONE', 'stop', self.plan(), self.request, self.catalog)
        self.assertIsNone(quality['titleObservedAlignmentRate'])

    def test_control_characters_cannot_be_accepted_as_generated_plaintext(self):
        _, quality = validate_generated('S1: Everyday travel choices\nT1: Your compact\x00 travel companion\nB1: NONE', 'stop', self.plan(), self.request, self.catalog)
        self.assertFalse(quality['allFieldsFulfilled'])


if __name__ == '__main__': unittest.main()
