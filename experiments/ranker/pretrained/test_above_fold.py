"""CPU fixtures for a complete scoped payload, never measured model/UI output."""
import copy
import unittest

from .above_fold import REQUIRED, payload_hash, payload_timing, validate_view
from .semantic_copy_analysis import DATA, records
from .scheduler_identity_experiment import assemble_values


class AboveFoldTests(unittest.TestCase):
    def fixture(self):
        case = records(DATA / 'cases.train.jsonl')[0]
        rows = [{'rowId': t['rowId'], 'fields': [t['field']], 'kind': t['kind'],
                 'maxTokens': t['maxTokens'], 'text': t['targetValue'],
                 'finishReason': 'stop', 'completedMs': 100. + i}
                for i, t in enumerate(case['targets'])]
        cards = case['pendingPagePlan']['sections'][0]['products']; values = []
        by_field = {r['fields'][0]: r for r in rows}
        for i, card in enumerate(cards, 1):
            product = case['catalog'][card['productId']]
            tag = product['tags'][0]
            text = f'A {tag} daypack for everyday travel.'
            rows.append({'rowId': f'copy-{i}', 'fields': [f'C{i}'], 'kind': 'copy',
                'maxTokens': 32, 'text': text, 'finishReason': 'stop', 'completedMs': 200. + i})
            image = next(v for v in product['presentationVariants'] if v['id'] == card['imageCaptionId'])
            badge = by_field[f'B{i}']['text']
            values.append({'productId': card['productId'], 'imageCaptionId': card['imageCaptionId'],
                'imageKey': image['imageKey'], 'originalName': product['title'],
                'price': product['price'], 'shippingCost': product['shippingCost'],
                'catalogCategory': product['category'], 'title': by_field[f'T{i}']['text'],
                'badge': None if badge == 'NONE' else badge, 'bodyCopy': text, 'addToCartLabel': 'Add to cart'})
        payload = {'schemaVersion': 'first-section-payload.v1', 'sectionId': case['pendingPagePlan']['sections'][0]['sectionId'],
            'heading': by_field['S1']['text'], 'sectionIntro': 'A few of our favorites.', 'cards': values}
        return case, rows, payload

    def validate(self, case, rows, payload, **kwargs):
        return validate_view(rows, case['pendingPagePlan'], case['request'], case['catalog'], payload, **kwargs)

    def test_complete_view_and_no_mutation(self):
        case, rows, payload = self.fixture(); before = copy.deepcopy((case, rows, payload))
        result = self.validate(case, rows, payload)
        self.assertTrue(result['valid'], result['rejections'])
        self.assertEqual(before, (case, rows, payload))
        self.assertEqual(result['payloadSha256'], payload_hash(payload))

    def test_below_fold_length_does_not_invalidate_view_but_full_page_stays_false(self):
        case, rows, payload = self.fixture()
        row = next(r for r in rows if r['fields'] == ['S2']); row['finishReason'] = 'length'; row['text'] = 'unfinished'
        self.assertTrue(self.validate(case, rows, payload)['valid'])
        full = assemble_values([r for r in rows if r['kind'] != 'copy'], case['pendingPagePlan'], case['request'], case['catalog'])
        self.assertFalse(full['quality']['allFieldsFulfilled'])

    def test_body_copy_is_required_and_factual(self):
        case, rows, payload = self.fixture()
        missing = [r for r in rows if r['fields'] != ['C1']]
        self.assertIn('C1:missing-row', self.validate(case, missing, payload)['rejections'])
        next(r for r in rows if r['fields'] == ['C1'])['text'] = 'A guaranteed waterproof travel daypack.'
        payload['cards'][0]['bodyCopy'] = 'A guaranteed waterproof travel daypack.'
        self.assertFalse(self.validate(case, rows, payload)['valid'])

    def test_original_subtitle_does_not_relax_current_title_identity(self):
        case, rows, payload = self.fixture()
        next(r for r in rows if r['fields'] == ['T1'])['text'] = 'Meet your everyday travel companion'
        payload['cards'][0]['title'] = 'Meet your everyday travel companion'
        result = self.validate(case, rows, payload)
        self.assertFalse(result['valid'])
        self.assertTrue(any('missing_selected_product_identity' in e for e in result['rejections']))

    def test_wire_error_and_payload_metadata_mismatch_rejected(self):
        case, rows, payload = self.fixture()
        next(r for r in rows if r['fields'] == ['B1'])['text'] += '\n'
        self.assertFalse(self.validate(case, rows, payload)['valid'])
        case, rows, payload = self.fixture(); payload['cards'][0]['price'] += 1
        self.assertIn('card1:payload-price-mismatch', self.validate(case, rows, payload)['rejections'])
        case, rows, payload = self.fixture(); payload['cards'][0]['imageKey'] = 'wrong-image'
        self.assertFalse(self.validate(case, rows, payload)['valid'])

    def test_exact_complete_name_query_and_resolved_product_must_show(self):
        case, rows, payload = self.fixture()
        below = case['pendingPagePlan']['sections'][1]['products'][0]['productId']
        case['request']['context']['task'] = 'Please show ' + case['catalog'][below]['title']
        self.assertIn('explicit-complete-catalog-name-missing-from-view', self.validate(case, rows, payload)['rejections'])
        case['request']['context']['task'] = 'Choose travel daypacks.'
        self.assertFalse(self.validate(case, rows, payload, exact_product_ids=[below])['valid'])

    def test_timing_uses_all_required_completions_and_bound_payload_receipt(self):
        case, rows, payload = self.fixture(); validation = self.validate(case, rows, payload)
        next(r for r in rows if r['fields'] == ['C3'])['completedMs'] = 1000.
        next(r for r in rows if r['fields'] == ['S2'])['completedMs'] = 9999.
        lower_only = payload_timing(validation, rows, planning_prefix_ms=50.)
        self.assertEqual(lower_only['lastRequiredTokenLowerBoundMs'], 1050.)
        self.assertIsNone(lower_only['copyStagePayloadReadyMs'])
        receipt = {'payloadSha256': validation['payloadSha256'], 'validatedAndSerializedAtMs': 1060., 'metadataReadyAtMs': 10.}
        result = payload_timing(validation, rows, planning_prefix_ms=50., receipt=receipt)
        self.assertEqual(result['copyStagePayloadReadyMs'], 1060.)
        self.assertIsNone(result['fullRequestViewReadyMs'])
        with self.assertRaisesRegex(ValueError, 'predates'):
            payload_timing(validation, rows, planning_prefix_ms=50., receipt={**receipt, 'validatedAndSerializedAtMs': 1049.})
        with self.assertRaisesRegex(ValueError, 'another payload'):
            payload_timing(validation, rows, planning_prefix_ms=50., receipt={**receipt, 'payloadSha256': 'wrong'})

    def test_invalid_or_incomplete_payload_gets_no_ready_latency(self):
        case, rows, payload = self.fixture(); validation = self.validate(case, rows, payload)
        receipt = {'payloadSha256': validation['payloadSha256'], 'validatedAndSerializedAtMs': 300., 'metadataReadyAtMs': 10.}
        result = payload_timing({**validation, 'valid': False}, rows, planning_prefix_ms=10., receipt=receipt)
        self.assertIsNone(result['copyStagePayloadReadyMs'])
        result = payload_timing(validation, rows[:-1], planning_prefix_ms=10.)
        self.assertFalse(result['allRequiredRowsCompleted'])
        self.assertIsNone(result['lastRequiredTokenLowerBoundMs'])


if __name__ == '__main__': unittest.main()
