"""Independent source/price counterexamples and immutable archived-run checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from .store_usefulness_v2 import ARCHIVE, ROOT, frozen_screen, score_archive, score_field, sha


class EquivalentDollarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.products = json.loads((ROOT / 'store_catalog_v1/model_inputs.json').read_text())['products']
        cls.card = {'facts': {'name': 'Everyday steel bottle', 'selectedColor': 'Graphite',
            'category': 'Stainless steel bottle', 'itemPrice': 36}, 'productIndex': 4}

    def assess(self, amount, **kwargs):
        return score_field('C1', f'Everyday steel bottle in Graphite costs {amount}.',
            kwargs.pop('card', self.card), self.products, **kwargs)

    def test_equivalent_store_dollars_with_raw_text_preserved(self):
        for amount in ('$36', '$36.00', '36 dollars', '36.00 dollars', '36.0 dollars'):
            with self.subTest(amount=amount):
                result = self.assess(amount)
                self.assertTrue(result['accepted'], result)
                self.assertEqual(result['rawText'], f'Everyday steel bottle in Graphite costs {amount}.')
        self.assertFalse(self.assess('36 dollars')['frozenPracticalAccepted'])

    def test_decimal_source_not_integer_rounding(self):
        card = deepcopy(self.card); card['facts']['itemPrice'] = 36.25
        self.assertTrue(self.assess('36.25 dollars', card=card)['accepted'])
        for amount in ('36 dollars', '36.24 dollars', '36.26 dollars', '$36.250'):
            self.assertFalse(self.assess(amount, card=card)['accepted'], amount)

    def test_wrong_prices_and_extra_number_or_duplicate_price(self):
        for amount in ('$35.99', '35 dollars', '136 dollars', '36 dollars and 2',
            '36 dollars and $36', '$36 and $36', '36 dollars and 36 dollars',
            '36 dollars per 2 items', '36 dollars with 0 shipping'):
            self.assertFalse(self.assess(amount)['accepted'], amount)

    def test_foreign_or_unspecified_currency_never_normalized(self):
        for amount in ('36 euros', '€36', '£36', '¥36', '36 CAD', 'CAD 36',
            'USD 36', '36 USD', '$36 CAD', '36 dollars EUR'):
            self.assertFalse(self.assess(amount)['accepted'], amount)

    def test_malformed_and_signed_prices_rejected(self):
        for amount in ('36.000 dollars', '36..00 dollars', '36,00 dollars',
            '036 dollars', '-36 dollars', '+36 dollars', '3e1 dollars',
            'thirty-six dollars', '36dollars', '$36.000', '$36.00.00',
            '$36 dollars', '$ 36', '$36,00', '36 dollars.00'):
            self.assertFalse(self.assess(amount)['accepted'], amount)

    def test_price_normalization_does_not_hide_claims(self):
        for suffix in ('with free shipping', 'with a discount', 'for waterproof protection',
            'with durable premium quality', 'and is available now', 'and has a 2 year warranty'):
            text = f'Everyday steel bottle in Graphite costs 36 dollars {suffix}.'
            self.assertFalse(score_field('C1', text, self.card, self.products)['accepted'], text)

    def test_wrong_identity_color_missing_identity_and_unfinished(self):
        for text in ('Trail mug in Graphite costs 36 dollars.',
            'Everyday steel bottle in Cream costs 36 dollars.',
            'This selected item in Graphite costs 36 dollars.',
            'Everyday steel bottle in Graphite costs 36 dollars',
            'Everyday steel bottle in Graphite'):
            self.assertFalse(score_field('C1', text, self.card, self.products)['accepted'], text)

    def test_collection_scope_and_original_protocol_limits_preserved(self):
        for text in ('This collection of Everyday steel bottle in Graphite costs 36 dollars.',
            'Everyday steel bottle in Graphite costs 36 dollars.\n',
            'Everyday steel bottle in Graphite costs 36 dollars.<script>',
            'Everyday steel bottle in Graphite costs 36 dollars.' + ' '*200):
            self.assertFalse(score_field('C1', text, self.card, self.products)['accepted'], text)

    def test_nonprice_field_decisions_exactly_reuse_frozen_screen(self):
        for field, texts in [('T1', ['Everyday steel bottle in Graphite', 'Trail mug in Cream']),
            ('B1', ['In Graphite', 'Graphite hydration', 'In Cream']),
            ('S1', ['Your selected collection', 'Essential Daily Hydration'])]:
            for text in texts:
                card = None if field.startswith('S') else self.card
                self.assertEqual(score_field(field, text, card, self.products)['reasons'],
                    frozen_screen()(field, text, card, self.products))

    def test_original_native_receipt_failures_remain_rejected(self):
        path = ARCHIVE / 'hybrid/requests/878c00f2-599c-4a3a-bc59-8d472823dd98/receipt.json'
        data = json.loads(path.read_text()); row = deepcopy(data['rawRows'][3])
        def evaluate(receipt):
            return score_field('C1', row['text'], self.card, self.products, row=receipt)
        self.assertTrue(evaluate(row)['accepted'])
        before = deepcopy(row)
        evaluate(row)
        self.assertEqual(row, before)
        for key, value in [('finishReason', 'length'), ('stopTokenId', -1), ('arSteps', 999)]:
            bad = deepcopy(row); bad[key] = value
            self.assertIn('native-EOS-or-token-cap', evaluate(bad)['reasons'])
        self.assertIn('raw-text-does-not-match-original-row', self.assess('36 dollars', row=row)['reasons'])

    def test_source_price_swap_changes_acceptance_without_changing_text(self):
        before = deepcopy(self.card); good = self.assess('36 dollars')
        swapped = deepcopy(self.card); swapped['facts']['itemPrice'] = 24
        bad = self.assess('36 dollars', card=swapped)
        self.assertTrue(good['accepted']); self.assertFalse(bad['accepted'])
        self.assertEqual(good['rawText'], bad['rawText']); self.assertEqual(self.card, before)


class ArchivedCandidateTests(unittest.TestCase):
    def test_fixed_six_arms_full_coverage_and_no_native_imports_or_mutation(self):
        originals = {p: sha(p) for p in [ARCHIVE/'results.json', *ARCHIVE.glob('ar-*/result.json'),
            *ARCHIVE.glob('hybrid/requests/*/receipt.json')]}
        before_modules = set(sys.modules)
        report = score_archive()
        self.assertEqual({p: sha(p) for p in originals}, originals)
        self.assertFalse({'torch', 'mlx', 'mlx.core', 'mlx_lm'} & (set(sys.modules)-before_modules))
        self.assertEqual(len(report['arms']), 6)
        self.assertEqual(sum(len(a['fields']) for a in report['arms']), 120)
        self.assertEqual(report['summary']['A']['notationAcceptedOnlyByV2'], 18)
        self.assertEqual(report['summary']['B']['notationAcceptedOnlyByV2'], 0)
        self.assertEqual(report['summary']['A']['fullSixCorePass'], 3)
        self.assertEqual(report['summary']['B']['fullSixCorePass'], 0)
        self.assertEqual(sum(f['frozenPracticalAccepted'] for a in report['arms']
            if a['arm']=='A' for f in a['fields']), 30)
        self.assertEqual(sum(f['frozenPracticalAccepted'] for a in report['arms']
            if a['arm']=='B' for f in a['fields']), 54)
        self.assertTrue(all(a['observedV2ReadinessMs'] is None and
            a['frozenUsableFirstThreeMs'] is None and a['frozenUsableFullMs'] is None for a in report['arms']))

    def test_missing_receipts_still_count_all_six_as_failed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'results.json').write_bytes((ARCHIVE/'results.json').read_bytes())
            report = score_archive(path)
        self.assertEqual(len(report['arms']), 6)
        self.assertTrue(all(a['status']=='failed-offline-score' for a in report['arms']))
        self.assertEqual(report['summary']['A']['requiredFields'], 60)
        self.assertEqual(report['summary']['B']['requiredFields'], 60)
        self.assertEqual(report['summary']['A']['fullSixCorePass'], 0)


if __name__ == '__main__':
    unittest.main()
