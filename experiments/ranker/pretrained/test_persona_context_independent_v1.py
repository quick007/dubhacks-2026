"""Independent boundary tests; unit vectors below are synthetic CPU fixtures only."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import unittest

import torch

from . import store_persona_context_v1 as model


HERE = Path(__file__).resolve().parent


class IndependentPersonaContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        public = json.loads((HERE / 'store_catalog_v1/model_inputs.json').read_text())['products']
        mapping = json.loads((HERE / 'store_catalog_v1/external_mapping.json').read_text())['products']
        cls.products = []
        for product, routes in zip(public, mapping, strict=True):
            cls.products.append({**product, 'productId': routes['productId'], 'variants': [
                {**v, 'colorName': c['colorName']} for v, c in zip(routes['colorVariants'], product['colorVariants'], strict=True)]})
        cls.catalog = {'products': cls.products, 'vectors': torch.eye(16, 4096)}
        cls.request = json.loads((HERE / 'synthetic_history_v1/profiles.public.json').read_text())['profiles'][0]['request']

    def test_private_outcome_and_extra_persona_fields_rejected(self):
        for key, value in [('targetProductIndex', 6), ('chosen', {'productId': 'blanket'}), ('rationale', 'private gold')]:
            request = deepcopy(self.request); request[key] = value
            with self.assertRaises(ValueError):
                model.checked_context_request(request, self.products)
        request = deepcopy(self.request); request['persona']['recommendedProductIds'] = ['blanket']
        with self.assertRaises(ValueError):
            model.checked_context_request(request, self.products)

    def test_observed_catalog_identity_cannot_carry_spoofed_public_facts(self):
        for key, value in [('itemName', 'Invented laptop'), ('category', 'Invented category'), ('selectedColor', 'Invented ultraviolet')]:
            request = deepcopy(self.request)
            next(e for e in request['publicHistory'] if 'productId' in e)[key] = value
            with self.assertRaises(ValueError, msg=key):
                model.checked_context_request(request, self.products)

    def test_event_ids_and_absolute_time_shift_do_not_change_features(self):
        shifted = deepcopy(self.request)
        for index, event in enumerate(shifted['publicHistory']):
            event['eventId'] = 'unrelated-public-event-' + str(index)
            moment = datetime.fromisoformat(event['occurredAt'].replace('Z', '+00:00'))
            event['occurredAt'] = (moment - timedelta(days=100)).isoformat()
        context = model.checked_context_request(self.request, self.products)[2]
        changed = model.checked_context_request(shifted, self.products)[2]
        a, b = model.context_features(context, self.catalog), model.context_features(changed, self.catalog)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))

    def test_current_profile_encoder_does_not_claim_query_encoding(self):
        changed = deepcopy(self.request); changed['query'] = 'I need a towel for the swimming pool.'
        a = model.context_features(model.checked_context_request(self.request, self.products)[2], self.catalog)
        b = model.context_features(model.checked_context_request(changed, self.products)[2], self.catalog)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
        # The integration must retain a separate semantic-query path; this is a
        # measured limitation of this context encoder, not a complete rank test.

    def test_baseline_variants_obey_the_same_explicit_color_guard(self):
        request = deepcopy(self.request)
        request['hardRequirements'] = [{'attribute': 'color', 'value': 'Navy'}]
        profile = {'profileId': 'cpu-fixture', 'userGroupId': 'cpu-group', 'request': request}
        result = model.baseline_prediction(profile, [0.] * 16, self.catalog)
        self.assertTrue(result['selected'])
        for selected in result['selected']:
            product = next(p for p in self.products if p['productId'] == selected['productId'])
            variant = next(v for v in product['variants'] if v['variantId'] == selected['variantId'])
            self.assertEqual(variant['colorName'], 'Navy')

    def test_query_ceiling_cannot_be_relaxed_by_profile_budget(self):
        request = deepcopy(self.request); request['itemBudget'] = 90
        request['query'] = 'Show useful things under $20'
        parsed, policy, _ = model.checked_context_request(request, self.products)
        self.assertEqual(parsed['itemBudget'], 20)
        self.assertIs(policy['budgetExclusive'], True)
        request['itemBudget'] = 12; request['query'] = 'Show useful things under $90'
        parsed, policy, _ = model.checked_context_request(request, self.products)
        self.assertEqual(parsed['itemBudget'], 12)
        self.assertIs(policy['budgetExclusive'], False)

    def test_exact_name_still_abstains_when_stricter_query_budget_excludes_it(self):
        request = deepcopy(self.request); request['itemBudget'] = 90
        request['query'] = 'Market canvas tote under $20'
        profile = {'profileId': 'cpu-fixture', 'userGroupId': 'cpu-group', 'request': request}
        result = model.baseline_prediction(profile, [0.] * 16, self.catalog)
        self.assertEqual(result['order'], [])

    def test_explicit_query_color_conflicting_with_typed_color_abstains(self):
        request = deepcopy(self.request); request['itemBudget'] = 90
        request['query'] = 'Show products in Navy'
        request['hardRequirements'] = [{'attribute': 'color', 'value': 'Sage'}]
        profile = {'profileId': 'cpu-fixture', 'userGroupId': 'cpu-group', 'request': request}
        result = model.baseline_prediction(profile, [0.] * 16, self.catalog)
        self.assertEqual(result['order'], [])


if __name__ == '__main__':
    unittest.main()
