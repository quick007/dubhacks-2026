"""Identity regressions; product names optional when mentioned facts are unique."""
import copy
import unittest
from .fusion import identity_errors, validate_copy


class TitleIdentityTests(unittest.TestCase):
    def setUp(self):
        self.p = {'id': 'p', 'title': 'Pine 18L Daypack', 'category': 'backpacks',
                  'tags': ['compact', 'laptop-sleeve'], 'presentationVariants': [],
                  'price': 40, 'shippingCost': 5}
        self.catalog = {'p': self.p,
            'r': {**self.p, 'id': 'r', 'title': 'River 18L Daypack', 'tags': ['durable', 'roomy']},
            'a': {**self.p, 'id': 'a', 'title': 'Alder 24L Daypack', 'tags': ['compact', 'lightweight']}}

    def test_name_or_unambiguous_facts(self):
        for title in ('Pine daypack for everyday travel', 'Compact 18L daypack for travel'):
            self.assertEqual(identity_errors(title, self.p, self.catalog), [])
            self.assertTrue(validate_copy('Title: '+title+'\nCopy: A compact daypack for travel.', self.p, self.catalog)['titleAccepted'])
        for title in ('Meet the compact everyday travel choice', 'Everyday 18L daypack for travel',
                      'Compact daypack for everyday travel', 'Meet your everyday travel companion',
                      'Compact 18 daypack for everyday travel', 'l everyday travel',
                      'Pineapple compact daypack for travel'):
            self.assertTrue(identity_errors(title, self.p, self.catalog), title)

    def test_foreign_name_and_full_catalog_collision(self):
        self.assertIn('other_product_identity', identity_errors('River compact 18L daypack', self.p, self.catalog))
        catalog = copy.deepcopy(self.catalog)
        catalog['f'] = {**self.p, 'id': 'f', 'title': 'Field 18L Daypack', 'tags': ['compact', 'organizer']}
        self.assertIn('ambiguous_public_fact_identity', identity_errors('Compact 18L daypack for travel', self.p, catalog))
        self.assertEqual(identity_errors('Compact laptop-sleeve 18L daypack', self.p, catalog), [])
        catalog['f']['tags'] = self.p['tags']
        self.assertTrue(identity_errors('Compact laptop sleeve 18L daypack', self.p, catalog))

    def test_compound_tag_and_order_invariance(self):
        catalog = copy.deepcopy(self.catalog)
        catalog['f'] = {**self.p, 'id': 'f', 'title': 'Field 18L Daypack', 'tags': ['compact']}
        for title in ('Laptop sleeve 18L daypack', 'laptop-sleeve 18L daypack'):
            self.assertEqual(identity_errors(title, self.p, catalog), [])
            self.assertEqual(identity_errors(title, self.p, dict(reversed(list(catalog.items())))), [])
        self.assertTrue(identity_errors('Laptop 18L daypack', self.p, catalog))

    def test_page_path_rejects_generic_title(self):
        from .semantic_copy_data.generator import build
        from .evidence_plan import validate_generated
        case = build()[0]['train'][0]
        raw = '\n'.join(t['field']+': '+('Meet the everyday travel choice' if t['field']=='T1' else t['targetValue']) for t in case['targets'])
        _, quality = validate_generated(raw, 'stop', case['pendingPagePlan'], case['request'], case['catalog'])
        self.assertIn('T1:missing_selected_product_identity', quality['rejections'])


if __name__ == '__main__':
    unittest.main()
