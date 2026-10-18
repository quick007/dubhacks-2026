"""CPU packet/data boundary checks; no native model or tokenizer imports."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import store_parallel_text_data_v1 as data


class ParallelTextDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).resolve().parent / 'store_catalog_v1'
        cls.snapshot = {'inputs': json.loads((base / 'model_inputs.json').read_text())}
        cls.packet = data.build(cls.snapshot)

    def test_actual_membership_compositional_split_and_positive_word_coverage(self):
        packet = self.packet
        self.assertEqual(len(packet['featureRecords']), 48)
        self.assertEqual(packet['audit']['heldoutBindingCount'], 14)
        self.assertEqual(packet['audit']['productsWithoutHeldoutColor'], [13, 15])
        self.assertEqual(packet['audit']['evalWordsMissingPositiveTrainSupervision'], [])
        train = {r['featureKeys'][0] for r in packet['labels']['train'] if r['input']['kind'] != 'section'}
        evaluation = {r['featureKeys'][0] for r in packet['labels']['eval'] if r['input']['kind'] != 'section'}
        self.assertEqual((len(train), len(evaluation), len(train & evaluation)), (34, 14, 0))
        positive = Counter(w for r in packet['labels']['train'] for w in data.tokens(r['targetText']))
        self.assertTrue(all(positive[w] > 0 for r in packet['labels']['eval'] for w in data.tokens(r['targetText'])))

    def test_full_six_cards_twenty_fields_and_balanced_heldout_products(self):
        for split, count in (('train', 48), ('eval', 42)):
            self.assertEqual(len(self.packet['cases'][split]), count)
            appearances = Counter()
            for case in self.packet['cases'][split]:
                self.assertEqual(len(case['rows']), 20)
                self.assertEqual(len(set(case['productIndices'])), 6)
                self.assertEqual({r['field'] for r in case['rows']},
                    {'S1', 'S2'} | {f'{prefix}{i}' for prefix in 'TBC' for i in range(1, 7)})
                appearances.update(case['productIndices'])
            self.assertEqual(set(appearances.values()), {18})

    def test_exact_public_features_and_input_label_separation(self):
        for record in self.packet['featureRecords']:
            self.assertEqual(record, data.public_record(record['facts']))
            for forbidden in ('targetText', 'split', 'teacher', 'persona', 'productIndex', 'variantId', 'productId'):
                self.assertNotIn(forbidden, record['text'])
        for cases in self.packet['cases'].values():
            for case in cases:
                for row in case['rows']:
                    self.assertNotIn('targetText', data.canonical(row))
        # Serving fact preparation remains independent of the offline teacher.
        original = data.teacher
        try:
            data.teacher = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('offline teacher called'))
            data.serving_row('copy', 'plain', self.packet['featureRecords'][0]['facts'])
        finally:
            data.teacher = original

    def test_teachers_pass_grounding_and_slot_contract(self):
        self.assertEqual(self.packet['audit']['maximumTeacherLexicalTokens'], 9)
        for rows in self.packet['labels'].values():
            for row in rows:
                inp = row['input']
                self.assertLess(len(data.tokens(row['targetText'])), data.SLOTS)
                self.assertEqual(data.field_rejections(inp['kind'], row['targetText'], inp.get('source', {}).get('facts')), [])

    def test_wrong_color_price_and_unsupported_claims_rejected(self):
        facts = self.packet['featureRecords'][0]['facts']
        good = data.teacher('copy', 'plain', facts)
        self.assertEqual(data.field_rejections('copy', good, facts), [])
        for broken in (good.replace(facts['selectedColor'], 'Magenta'), good.replace('$24.00', '$25.00'),
                       good + ' Waterproof.', good.replace('$24.00', '$24.001'), good[:-1],
                       'Sage $24.00.', 'Shown in Sage at $24.00. Free delivery.'):
            self.assertTrue(data.field_rejections('copy', broken, facts), broken)
        counterfactual = {**facts, 'itemPrice': 25}
        self.assertTrue(data.field_rejections('copy', good, counterfactual))

    def test_lexical_decoder_never_silently_drops_unknown_bytes(self):
        for invalid in ('Price €24', 'Sage ☀', 'This is 99 dollars', 'x\n<script>'):
            with self.assertRaises(ValueError):
                data.tokens(invalid)
        self.assertEqual(data.detokenize(['Color', ':', 'Slate', 'blue', '.', 'Item', ':', '$24.00', '.']),
                         'Color: Slate blue. Item: $24.00.')

    def test_packet_manifest_and_no_native_side_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'packet'
            manifest = data.write_packet(self.packet, out)
            self.assertEqual(manifest['status'], 'cpu-packet-only-not-native-quality')
            self.assertEqual(len(manifest['files']), 7)
            public = json.loads((out / 'public-feature-contract.json').read_text())
            self.assertEqual(public['requiredNativeForwards'], 48)
            self.assertFalse(public['copyAdapterEnabled'])
            self.assertNotIn('targetText', (out / 'public-feature-contract.json').read_text())
            with self.assertRaises(FileExistsError):
                data.write_packet(self.packet, out)


if __name__ == '__main__':
    unittest.main()
