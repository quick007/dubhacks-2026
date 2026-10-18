"""CPU contract checks: source facts, constraints, variable real-card rosters."""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .store_catalog_v1 import load_snapshot
from .store_copy_data_v2 import CAPS, layout as frozen_layout
from .store_demo_engine import (NEUTRAL_COPY_REQUEST, checked_request, demo_layout,
    guarded_order, practical_rejections, public_sections, ranking_input)
from .store_demo_service import ModelWorker, handler_for
from .store_rank_session import action_bindings


class DemoContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = load_snapshot()
        cls.products = cls.snapshot['inputs']['products']
        request, _ = checked_request({'query': 'browse'})
        cls.cards = action_bindings(request, list(range(6)), cls.snapshot)

    def test_variable_layout_preserves_real_roster_and_caps(self):
        for count in range(1, 7):
            _, rows = demo_layout(self.cards[:count])
            self.assertEqual(len(rows), 3 * count + (count + 2) // 3)
            self.assertEqual(sum(r['kind'] == 'title' for r in rows), count)
            self.assertEqual(sum(r['maxTokens'] for r in rows), 64 * count + 16 * ((count + 2) // 3))
            self.assertFalse(any('targetValue' in r or 'text' in r for r in rows))
        with self.assertRaises(ValueError):
            demo_layout([])

    def test_six_layout_same_as_frozen_v2(self):
        self.assertEqual(demo_layout(self.cards), frozen_layout({'request': NEUTRAL_COPY_REQUEST, 'cards': self.cards}))

    def test_head_input_preserves_entire_raw_query_as_preference(self):
        request, _ = checked_request({'query': 'I want water, not music, under $50'})
        result = ranking_input(request)
        self.assertEqual(result['preferences'], [request['query']])
        self.assertEqual(result['itemBudget'], 50)
        self.assertEqual(result['query'], 'Choose a product for my stated use.')

    def test_budget_strictness_and_exact_identity(self):
        product = self.products[0]
        price = product['itemPrice']
        request, policy = checked_request({'query': f"{product['name']} under ${price}"})
        order, checks = guarded_order([1.] * 16, request, policy, self.products)
        self.assertEqual(order, [])
        self.assertTrue(checks['exactNamePolicy']['ineligibleExactIdentityAbstention'])
        request, policy = checked_request({'query': f"{product['name']} up to ${price}"})
        order, _ = guarded_order([1.] * 16, request, policy, self.products)
        self.assertEqual(order[0], 0)

    def test_typed_budget_is_inclusive_and_overrides_query_budget(self):
        request, policy = checked_request({'query': 'under $1', 'itemBudget': 24})
        self.assertEqual(request['itemBudget'], 24)
        self.assertFalse(policy['budgetExclusive'])

    def test_conservative_ceilings_and_delivered_budget(self):
        request, policy = checked_request({'query': 'under $40 but up to $25'})
        self.assertEqual(request['itemBudget'], 25)
        self.assertFalse(policy['budgetExclusive'])
        request, policy = checked_request({'query': 'up to $25 and under $25'})
        self.assertTrue(policy['budgetExclusive'])
        with self.assertRaisesRegex(ValueError, 'Delivered-cost'):
            checked_request({'query': 'something delivered under $25'})

    def test_delivery_alias_budget_is_not_silently_treated_as_item_only(self):
        for phrase in ('including delivery', 'with delivery', 'including shipping', 'delivery included'):
            with self.subTest(phrase=phrase), self.assertRaisesRegex(ValueError, 'Delivered-cost'):
                checked_request({'query': f'a mug under $30 {phrase}'})
            with self.subTest(typed=phrase), self.assertRaisesRegex(ValueError, 'Delivered-cost'):
                checked_request({'query': f'a mug {phrase}', 'itemBudget': 30})
        request, _ = checked_request({'query': 'a mug with delivery'})
        self.assertIsNone(request['itemBudget'])

    def test_unknown_stock_size_fail_closed_and_colors_bind_actual_variant(self):
        for query in ('something in stock', 'a jacket size M'):
            request, policy = checked_request({'query': query})
            self.assertEqual(guarded_order([1.] * 16, request, policy, self.products)[0], [])
        color = self.products[0]['colorVariants'][1]['colorName']
        request, policy = checked_request({'query': 'browse', 'hardRequirements': [{'attribute': 'color', 'value': color}]})
        order, _ = guarded_order([1.] * 16, request, policy, self.products)
        cards = action_bindings(request, order[:6], self.snapshot)
        self.assertTrue(all(c['facts']['selectedColor'] == color for c in cards))
        self.assertEqual(cards[0]['action']['variantId'], self.snapshot['externalMapping']['products'][0]['colorVariants'][1]['variantId'])

    def test_only_positive_declared_waterproof_supported(self):
        request, policy = checked_request({'query': 'I need waterproof'})
        order, _ = guarded_order([1.] * 16, request, policy, self.products)
        self.assertTrue(order)
        self.assertTrue(all('waterproof' in (' '.join([self.products[i]['category'], *self.products[i]['searchTerms']])).lower() for i in order))
        self.assertFalse(checked_request({'query': 'not waterproof'})[1]['waterproofRequired'])

    def test_contradictory_colors_abstain_instead_of_variant_error(self):
        request, policy = checked_request({'query': 'browse', 'hardRequirements': [
            {'attribute': 'color', 'value': 'Sage'}, {'attribute': 'color', 'value': 'Ivory'}]})
        self.assertEqual(guarded_order([1.] * 16, request, policy, self.products)[0], [])

    def test_unknown_and_private_request_fields_rejected(self):
        for value in ({'query': 'browse', 'hiddenProfile': {}}, {'query': 'browse', 'itemBudget': True},
                {'query': 'browse', 'hardRequirements': [{'attribute': 'material', 'value': 'cotton'}]}):
            with self.assertRaises(ValueError):
                checked_request(value)

    def test_practical_gate_accepts_natural_source_grounded_copy_outside_old_grammar(self):
        card = self.cards[0]; facts = card['facts']
        text = f"{facts['name']} comes in {facts['selectedColor']}."
        self.assertEqual(practical_rejections('C1', text, card, self.products), [])
        category = f"{facts['name']} is a {facts['category']}."
        self.assertEqual(practical_rejections('C1', category, card, self.products), [])

    def test_gate_rejects_incorrect_prices_malformed_prices_missing_identity(self):
        card = self.cards[0]; facts = card['facts']; name, color = facts['name'], facts['selectedColor']
        for price in ('$999', '$24.000', '$24.0.0', '$twenty'):
            text = f'{name} in {color} is listed at {price}.'
            self.assertTrue(practical_rejections('C1', text, card, self.products))
        self.assertTrue(practical_rejections('C1', f'Shown in {color}.', card, self.products))

    def test_collection_price_scope_is_withheld_without_rewriting(self):
        card = self.cards[0]; facts = card['facts']
        text = f"This collection of {facts['name']} in {facts['selectedColor']} has an item price of ${facts['itemPrice']:.2f}."
        self.assertIn('ambiguous-collection-item-price-scope',
            practical_rejections('C1', text, card, self.products))

    def test_gate_rejects_cross_identity_color_and_unsupported_use(self):
        card = self.cards[0]; facts = card['facts']; name, color = facts['name'], facts['selectedColor']
        other = self.cards[1]['facts']
        for text in (f"{name} in {other['selectedColor']}.",
                f"{name} in {color} with {other['name']}.",
                f'{name} in {color} for hiking.', f'{name} in {color} has free shipping.'):
            self.assertTrue(practical_rejections('C1', text, card, self.products), text)

    def test_native_required_and_failed_field_withheld_without_repair(self):
        facts = self.cards[0]['facts']
        rows = [{'fields': ['T1'], 'text': f"{facts['name']} in {facts['selectedColor']}"}]
        sections, result = public_sections(self.cards[:1], rows, self.products)
        self.assertIsNone(sections[0]['cards'][0]['title'])
        self.assertEqual(sections[0]['cards'][0]['originalName'], facts['name'])
        self.assertEqual(result['requiredFields'], 4)
        self.assertEqual(result['acceptedFields'], 0)
        self.assertEqual(result['withheldFields'], 4)
        self.assertEqual(public_sections([], [], self.products)[0], [])

    def test_valid_native_rows_accept_fields_and_leave_raw_text_exact(self):
        card = self.cards[0]; facts = card['facts']
        values = {'S1': 'Browse the collection', 'T1': f"{facts['name']} in {facts['selectedColor']}",
            'B1': f"In {facts['selectedColor']}", 'C1': f"{facts['name']} comes in {facts['selectedColor']}."}
        rows = []
        for field, text in values.items():
            rows.append({'fields': [field], 'text': text, 'rawSampledTokenIds': [1, 2], 'tokenIds': [1],
                'stopTokenId': 2, 'arSteps': 2, 'finishReason': 'stop', 'maxTokens': CAPS[field[0]]})
        sections, result = public_sections([card], rows, self.products)
        self.assertEqual(result['acceptedFields'], 4)
        self.assertEqual(sections[0]['cards'][0]['bodyCopy'], values['C1'])


class HTTPContractTests(unittest.TestCase):
    def setUp(self):
        from http.server import ThreadingHTTPServer
        class FakeWorker:
            def health(self):
                return {'status': 'ready', 'ready': True}
            def submit(self, value, request_id, started):
                from concurrent.futures import Future
                import queue
                events = queue.Queue(); cancel = threading.Event(); future = Future()
                events.put({'type': 'rank', 'requestId': request_id, 'productIds': [], 'selectedProductIds': [], 'variants': {}, 'shortfall': 6, 'timing': {'rankMs': 1}})
                events.put({'type': 'complete', 'requestId': request_id, 'sections': [], 'acceptedFields': 0, 'withheldFields': 0, 'requiredFields': 0})
                future.set_result(True); events.put(None)
                return events, cancel, future
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(FakeWorker()))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()

    def test_real_http_ndjson_and_cors(self):
        request = Request(self.url + '/personalize', data=b'{"query":"browse"}',
            headers={'Content-Type': 'application/json', 'Origin': 'http://shop.dubhacks.localhost:1356'})
        with urlopen(request) as response:
            self.assertEqual(response.headers['Access-Control-Allow-Origin'], 'http://shop.dubhacks.localhost:1356')
            self.assertEqual([json.loads(line)['type'] for line in response], ['rank', 'complete'])

    def test_unknown_origin_and_bad_request_rejected_before_model(self):
        request = Request(self.url + '/personalize', data=b'{"query":"browse"}', headers={'Origin': 'https://other.example'})
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 403)
        request = Request(self.url + '/personalize', data=b'{"query":"browse","hiddenTruth":{}}')
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 400)


if __name__ == '__main__':
    unittest.main()
