"""Public contract gold cases; CPU only, no model-quality or NLI claim."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from .store_demo_engine import (NEUTRAL_COPY_REQUEST, checked_request, demo_layout,
    guarded_order, practical_rejections, public_sections, ranking_input)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'store_transparency_v1'


class PublicTransparencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((DATA / 'public-cases.json').read_text())
        cls.labels = {r['caseId']: r for name in ('train-labels.json', 'evaluation-labels.json')
                      for r in json.loads((DATA / name).read_text())}
        cls.products = json.loads((ROOT / 'store_catalog_v1/model_inputs.json').read_text())['products']
        cls.mapping = json.loads((ROOT / 'store_catalog_v1/external_mapping.json').read_text())['products']

    def card(self, product_index=0):
        p, mapping = self.products[product_index], self.mapping[product_index]
        return {'productIndex': product_index,
                'facts': {'name': p['name'], 'category': p['category'], 'itemPrice': p['itemPrice'],
                          'selectedColor': p['colorVariants'][0]['colorName']},
                'action': {'productId': mapping['productId'], 'variantId': mapping['colorVariants'][0]['variantId'],
                           'image': deepcopy(mapping['colorVariants'][0]['images'][0])}}

    def test_frozen_small_packet_is_untrained_and_gold_is_external(self):
        manifest = json.loads((DATA / 'manifest.json').read_text())
        self.assertEqual(manifest['counts'], {'train': 8, 'evaluation': 4})
        self.assertFalse(manifest['trainingPerformed'])
        self.assertFalse(manifest['nativeExecution'])
        self.assertFalse(manifest['sealedEvaluation'])
        self.assertEqual(len(self.cases), 12)
        self.assertEqual(len(self.labels), 12)
        self.assertEqual(sum(c['split'] == 'train' for c in self.cases), 8)
        self.assertEqual(sum(c['split'] == 'eval' for c in self.cases), 4)
        for name, sha in manifest['files'].items():
            self.assertEqual(hashlib.sha256((DATA / name).read_bytes()).hexdigest(), sha)
        self.assertEqual(hashlib.sha256((ROOT / manifest['catalogSource']).read_bytes()).hexdigest(), manifest['catalogSha256'])
        for case in self.cases:
            predictive = json.dumps(case['publicInput'])
            for forbidden in ('"hiddenPersona":', '"personaFamilyId":', 'targetExplanation', 'claimStatus', case['caseId']):
                self.assertNotIn(forbidden, predictive)
            self.assertNotIn(self.labels[case['caseId']]['targetExplanation'], predictive)

    def test_public_claims_and_gold_support_bind_real_catalog_facts(self):
        for case in self.cases:
            public, label = case['publicInput'], self.labels[case['caseId']]
            if 'productFacts' in public:
                index = label['support']['catalogProductIndex']
                self.assertEqual(public['productFacts'], self.card(index)['facts'])
        supported = next(c for c in self.cases if c['caseId'] == 'transparency-train-03')
        self.assertIn('$24.00', supported['publicInput']['candidateCopy'])
        self.assertEqual(self.labels[supported['caseId']]['claimStatus'], 'supported')
        # Absent heat retention evidence must remain unknown, not a physical
        # negative label such as "this mug cannot retain heat".
        heat = self.labels['transparency-train-04']
        self.assertEqual(heat['claimStatus'], 'unsupported-performance')
        self.assertIn('duration', heat['support']['unknownClaims'])

    def test_budget_gold_wins_over_adversarial_scores_and_exact_name(self):
        for case in self.cases:
            if case['task'] != 'public-budget-decision':
                continue
            label = self.labels[case['caseId']]
            if label['requestRejected']:
                with self.assertRaisesRegex(ValueError, label['expectedErrorContains']):
                    checked_request(case['publicInput']['request'])
                self.assertIsNone(label['namedProductEligible'])
                continue
            request, policy = checked_request(case['publicInput']['request'])
            # The $128 jacket scores highest. Neither this score nor a cheaper
            # notebook can override an ineligible exact-mug instruction.
            scores = [0.] * 16
            scores[3], scores[2] = 1e6, 1e5
            order, evidence = guarded_order(scores, request, policy, self.products)
            self.assertEqual(0 in order, label['namedProductEligible'])
            self.assertEqual(not order, label['exactIdentityAbstention'])
            if order:
                self.assertEqual(order[0], 0)
                self.assertTrue(all(self.products[i]['itemPrice'] <= request['itemBudget'] for i in order))
            self.assertFalse(evidence['exactNamePolicy']['learned'])

    def test_grounded_and_unsupported_claim_gold_against_actual_screen(self):
        for case in self.cases:
            if case['task'] != 'grounded-copy-claim':
                continue
            public, label = case['publicInput'], self.labels[case['caseId']]
            card = self.card(label['support']['catalogProductIndex'])
            rejected = bool(practical_rejections(public['field'], public['candidateCopy'], card, self.products))
            self.assertEqual(rejected, label['shouldWithhold'], case['caseId'])

    def test_disclosure_gold_marks_false_attribution_instead_of_overclaiming(self):
        observed_false = set()
        for case in self.cases:
            if case['task'] != 'pipeline-disclosure':
                continue
            public, label = case['publicInput'], self.labels[case['caseId']]
            # A hand-authored public truth table checks semantic gold labels.
            # This is not a claim that arbitrary prose is machine-classified.
            mismatches = {k for k, v in public['claimedOperations'].items() if v != public['operationFacts'][k]}
            self.assertEqual(mismatches, set(label.get('mismatchedOperations', [])))
            self.assertEqual(bool(mismatches), label['shouldWithhold'])
            observed_false.update(mismatches)
        self.assertEqual(observed_false, {'budgetGuardsLearned', 'imageSelectionLearned',
                                         'displayedItemPriceGenerated', 'queryConditionsCopyWording'})

    def test_real_input_boundary_query_changes_rank_input_but_not_copy(self):
        a, _ = checked_request({'query': 'a bottle for drinking water'})
        b, _ = checked_request({'query': 'a light for my bedside table'})
        self.assertNotEqual(ranking_input(a), ranking_input(b))
        self.assertEqual(ranking_input(a)['preferences'], [a['query']])
        cards = [self.card(0)]
        common_a, jobs_a = demo_layout(cards)
        common_b, jobs_b = demo_layout(deepcopy(cards))
        self.assertEqual((common_a, jobs_a), (common_b, jobs_b))
        prompt = json.dumps([common_a, jobs_a])
        self.assertNotIn(a['query'], prompt)
        self.assertNotIn(b['query'], prompt)
        self.assertIn(NEUTRAL_COPY_REQUEST['query'], prompt)
        with self.assertRaises(ValueError):
            checked_request({'query': 'browse', 'hiddenPersona': {'prefers': 'luxury'}})

    def test_withheld_native_text_is_not_relabelled_generated_catalog_metadata(self):
        case = next(c for c in self.cases if c['task'] == 'native-metadata-distinction')
        label = self.labels[case['caseId']]
        sections, receipt = public_sections([self.card(0)], [], self.products)
        visible = sections[0]['cards'][0]
        self.assertEqual(receipt['acceptedFields'], 0)
        self.assertEqual(receipt['withheldFields'], 4)
        self.assertIsNone(visible['title'])
        self.assertIsNone(visible['badge'])
        self.assertIsNone(visible['bodyCopy'])
        self.assertEqual(visible['originalName'], 'Trail mug')
        self.assertEqual(visible['itemPrice'], 24)
        self.assertEqual(label['nativeFieldsActuallyPresent'], [])
        self.assertEqual(label['catalogFieldsVisible'], ['originalName', 'itemPrice', 'selectedColor'])


if __name__ == '__main__':
    unittest.main()
