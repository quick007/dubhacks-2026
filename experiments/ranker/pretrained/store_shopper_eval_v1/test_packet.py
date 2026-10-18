"""Meaningful fixed gold/privacy/pairing checks; stdlib only, no native work."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import validation as packet


class ShopperPacketTests(unittest.TestCase):
    def test_packet_fixed_pairs_and_budget(self):
        report = packet.validate_packet()
        self.assertEqual(report['pairs'], 10)
        self.assertEqual(report['plannedRuns'], 20)
        self.assertEqual(report['executedRunsClaimed'], 0)
        self.assertEqual(report['satisfiableCases'], 8)
        self.assertEqual((report['knownEmpty'], report['unknownRequiredFact']), (1, 1))

    def test_shop_projection_never_reads_gold_or_catalog(self):
        original = packet.read
        names = []
        def guarded(name, directory=packet.DATA):
            names.append(name)
            if name in ('evaluator-gold.json', 'catalog-binding.json'):
                raise AssertionError('Gold/catalog leaked into shopper projection')
            return original(name, directory)
        with patch.object(packet, 'read', side_effect=guarded):
            for index in range(1, 11):
                cid = f'shopper-pair-{index:02d}'
                for arm in ('generic', 'local-model'):
                    assignment = packet.shopper_assignment(cid, arm)
                    packet.assert_no_gold(assignment)
        self.assertNotIn('evaluator-gold.json', names)

    def test_public_input_is_exact_observation_only_and_private_key_injection_rejected(self):
        for index in range(1, 11):
            cid = f'shopper-pair-{index:02d}'
            public = packet.recommender_payload(cid)
            self.assertEqual(set(public), {'query', 'itemBudget', 'hardRequirements'})
            for key in ('privateShopperNeeds', 'evaluatorGold', 'acceptableProductIds', 'hiddenPersona'):
                with self.assertRaises(ValueError):
                    packet.validate_public({**public, key: {}})
            private = packet.shopper_case(cid)['privateShopperNeeds']
            with self.assertRaises(ValueError):
                packet.validate_private({**private, 'correctSelector': '#buy-correct-product'})

    def test_pair_conditions_equal_except_model_mode(self):
        for index in range(1, 11):
            cid = f'shopper-pair-{index:02d}'
            a, b = (packet.shopper_assignment(cid, arm) for arm in ('generic', 'local-model'))
            self.assertEqual(a['entryUrl'] + '?model=local', b['entryUrl'])
            for key in ('shopperNeeds', 'recommenderContext', 'instructions'):
                self.assertEqual(a[key], b[key])
            case = packet.shopper_case(cid)
            self.assertEqual(case['seed'], 2026101800 + index)
        plan = packet.read('pair-plan.json')
        self.assertEqual(sum(r['armOrder'][0] == 'generic' for r in plan['schedule']), 5)
        self.assertFalse(plan['isolation']['crossArmDecisionsOrMemoryPermitted'])
        self.assertTrue(plan['isolation']['freshTabRequiredTruth']['sharedStorage'])

    def test_semantic_goals_have_real_known_products_and_variant_binding(self):
        catalog = packet.read('catalog-binding.json')['catalogSnapshot']
        gold = packet._index(packet.read('evaluator-gold.json'))
        expected = {'01': ['notebook'], '02': ['bottle'], '03': ['lamp'], '04': ['tote'],
                    '05': ['organizer'], '06': ['mug'], '07': ['tote', 'duffel'], '08': ['jacket']}
        for suffix, correct in expected.items():
            row = gold['shopper-pair-' + suffix]
            self.assertEqual(row['acceptableProductIds'], correct)
            variants, unknowns = packet.expected_variants(packet.shopper_case(row['caseId']), row, catalog)
            self.assertTrue(variants)
            self.assertFalse(unknowns)
        mug = next(p for p in catalog['products'] if p['productId'] == 'mug')
        cobalt = next(v['variantId'] for v in mug['variants'] if v['selectedColor'] == 'Cobalt')
        self.assertEqual(gold['shopper-pair-06']['acceptableVariantIds'], [cobalt])

    def test_budget_boundary_notebook_substitution_and_unknown_stock_are_distinct(self):
        catalog = packet.read('catalog-binding.json')['catalogSnapshot']
        gold = packet._index(packet.read('evaluator-gold.json'))
        case = packet.shopper_case('shopper-pair-09')
        variants, unknowns = packet.expected_variants(case, gold[case['caseId']], catalog)
        self.assertEqual(variants, [])
        self.assertEqual(unknowns, [])
        inclusive = deepcopy(case)
        inclusive['privateShopperNeeds']['hardConstraints'][0]['operator'] = 'at-most'
        variants, _ = packet.expected_variants(inclusive, gold[case['caseId']], catalog)
        self.assertTrue(variants, 'Exactly $24 must pass inclusive but fail exclusive $24')
        stock_case = packet.shopper_case('shopper-pair-10')
        variants, unknowns = packet.expected_variants(stock_case, gold[stock_case['caseId']], catalog)
        self.assertEqual(variants, [])
        self.assertEqual(unknowns, ['stockVerified'])
        self.assertTrue(all('stockVerified' not in p['facts'] for p in catalog['products']))

    def test_negative_backpack_constraint_removes_an_otherwise_eligible_bag(self):
        catalog = packet.read('catalog-binding.json')['catalogSnapshot']
        gold = packet._index(packet.read('evaluator-gold.json'))['shopper-pair-07']
        case = packet.shopper_case(gold['caseId'])
        before, _ = packet.expected_variants(case, gold, catalog)
        self.assertNotIn('pack', {p for p, _ in before})
        case['privateShopperNeeds']['hardConstraints'] = [n for n in case['privateShopperNeeds']['hardConstraints'] if n['id'] != 'avoid-backpack']
        without_negative, _ = packet.expected_variants(case, gold, catalog)
        self.assertIn('pack', {p for p, _ in without_negative})

    def test_external_personas_not_gold_and_only_local_origins_allowed(self):
        binding = packet.read('catalog-binding.json')
        self.assertTrue(binding['existingAssetPersonasAreNotGold'])
        self.assertTrue(binding['annotatedFunctionFactsAreEvaluatorOnly'])
        for url in ('https://example.com/', 'http://127.0.0.1:4312/?gold=notebook', 'http://127.0.0.1:4312/#confirmed'):
            with self.assertRaises(ValueError):
                packet.shopper_assignment('shopper-pair-02', 'generic', base_url=url)


if __name__ == '__main__':
    unittest.main()
