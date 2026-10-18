"""CPU contract/data tests; no tokenizer, model or MLX import."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import store_copy_data_v2 as data


class StoreCopyV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).resolve().parent / 'store_catalog_v1'
        cls.snapshot = {'inputs': json.loads((base / 'model_inputs.json').read_text()),
                        'externalMapping': json.loads((base / 'external_mapping.json').read_text())}
        cls.groups, cls.fields = data.build(cls.snapshot)

    def bound(self):
        return copy.deepcopy(self.groups['train'][0]['bound'])

    def rows(self, split='train', case=0, native=False):
        current = self.groups[split][case]
        rows = [{**copy.deepcopy(row), 'text': row['targetValue']} for row in self.fields[split] if row['caseId'] == current['caseId']]
        if native:
            for row in rows:
                row.update(tokenIds=[100], rawSampledTokenIds=[100, 200], stopTokenId=200,
                           finishReason='stop', arSteps=2)
        return rows

    def test_balanced_products_colors_and_fixed_denominators(self):
        audit = data.coverage(self.groups, self.fields)
        self.assertEqual((len(self.groups['train']), len(self.groups['eval'])), (16, 8))
        self.assertEqual((len(self.fields['train']), len(self.fields['eval'])), (320, 160))
        for split, count in (('train', 6), ('eval', 3)):
            self.assertEqual(set(audit[split]['productOccurrences'].values()), {count})
            self.assertEqual(audit[split]['requiredColorBadges'], 96 if split == 'train' else 48)
        self.assertEqual(audit['bindingSplit']['overlap'], 0)
        self.assertEqual(audit['bindingSplit']['trainStates'], 32)
        self.assertEqual(audit['bindingSplit']['evalStates'], 16)
        self.assertEqual(set(audit['bindingSplit']['heldOutColorIndexByProduct'].values()), {0, 1, 2})

    def test_every_product_gets_all_six_body_and_title_constructions(self):
        lookup = {c['caseId']: c for c in self.groups['train']}
        for product in range(16):
            for kind in ('title', 'copy'):
                rows = [r for r in self.fields['train'] if r['kind'] == kind
                        and lookup[r['caseId']]['bound']['cards'][int(r['field'][1:]) - 1]['productIndex'] == product]
                self.assertEqual({r['teacherConstruction'] for r in rows}, set(range(6)))
                self.assertEqual(len({r['targetValue'] for r in rows}), 6)

    def test_every_teacher_and_complete_endpoint_passes_new_contract(self):
        for split, cases in self.groups.items():
            for index, case in enumerate(cases):
                rows = self.rows(split, index)
                full = data.assemble(case['bound'], rows, require_native=False)
                first = data.assemble(case['bound'], rows[:10], 3, require_native=False)
                self.assertTrue(full['validation']['valid'], full['validation']['rejections'])
                self.assertTrue(first['validation']['valid'], first['validation']['rejections'])
                self.assertEqual(len(full['payload']['sections']), 2)
                self.assertTrue(all(card['badge'] for s in full['payload']['sections'] for card in s['cards']))

    def test_serving_inputs_exactly_match_teacher_inputs_and_exclude_targets(self):
        lookup = {c['caseId']: c for cases in self.groups.values() for c in cases}
        for split, rows in self.fields.items():
            for row in rows:
                case = lookup[row['caseId']]
                common, jobs = data.layout(case['bound'])
                job = next(j for j in jobs if j['fields'] == row['fields'])
                for key in ('messages', 'fields', 'rowId', 'key', 'kind', 'maxTokens'):
                    self.assertEqual(job[key], row[key])
                prompt = data.canonical([common, job['messages']])
                for forbidden in (case['caseId'], 'targetValue', 'teacherConstruction', 'productIndex',
                                  'variantId', 'productId', 'hiddenPersona', 'personaFamilyId', 'sourceCaseId', 'split'):
                    self.assertNotIn(forbidden, prompt)
                self.assertNotIn(row['targetValue'].casefold(), prompt.casefold())
                self.assertEqual(tuple(j['fields'][0] for j in jobs), data.FIELDS)
                self.assertEqual(sum(j['maxTokens'] for j in jobs), 416)
                self.assertEqual(sum(j['maxTokens'] for j in jobs[:10]), 208)

    def test_inference_does_not_read_teacher_constructor(self):
        original = data.target_values
        try:
            data.target_values = lambda *a, **k: (_ for _ in ()).throw(AssertionError('teacher path reached'))
            bound = self.bound()
            data.layout(bound)
            self.assertTrue(data.assemble(bound, self.rows(), require_native=False)['validation']['valid'])
        finally:
            data.target_values = original

    def test_badge_is_factual_selected_color_and_none_requires_no_fact(self):
        bound = self.bound()
        color = bound['cards'][0]['facts']['selectedColor']
        for text in (f'In {color}', f'{color} selected', f'{color} choice', f'Selected in {color}'):
            self.assertEqual(data.field_rejections('B1', text, bound), [])
        for text in ('NONE', 'Popular choice', f'Best in {color}', 'In Unbound violet'):
            self.assertTrue(data.field_rejections('B1', text, bound))
        bound['cards'][0]['facts']['selectedColor'] = None
        self.assertEqual(data.field_rejections('B1', 'NONE', bound), [])
        self.assertTrue(data.field_rejections('B1', 'In Sage', bound))

    def test_price_decimals_are_numeric_facts_not_sentence_boundaries(self):
        bound = self.bound()
        facts = bound['cards'][0]['facts']
        name, color = facts['name'], facts['selectedColor']
        for price in ('24', '24.0', '24.00'):
            self.assertEqual(data.field_rejections('C1', f'This {name} is shown in {color} at ${price}.', bound), [])
        for price in ('2.40', '24.01', '240', '24.000', '-24', 'nan', '24,00'):
            self.assertTrue(data.field_rejections('C1', f'This {name} is shown in {color} at ${price}.', bound))
        facts['itemPrice'] = 24.35
        self.assertEqual(data.field_rejections('C1', f'This {name} is shown in {color} at $24.35.', bound), [])

    def test_unsupported_token_recombination_and_extra_claims_reject(self):
        bound = self.bound()
        for text in ('Trail mug is a slow mug.', 'Trail mug is the slow mornings.',
                     'Trail mug $24 and 24.', 'This Trail mug is in Sage for $24. Free shipping.',
                     'This Trail mug is shown in Sage at $24.00 and is waterproof.',
                     'This Trail mug is shown in Sage at $24.00. Ignore rules.',
                     'This Trail mug is shown in Sage at $24.00'):
            self.assertTrue(data.field_rejections('C1', text, bound), text)
        for text in ('For your you', 'Select your selected', 'Waterproof choices', 'Sage selected choices'):
            self.assertTrue(data.field_rejections('S1', text, bound))

    def test_fact_swap_sensitivity_for_title_badge_and_body(self):
        audit = data.counterfactual_audit(self.groups, self.fields)
        self.assertEqual(audit['checked'], {'name': 96, 'selectedColor': 144, 'itemPrice': 48})
        self.assertEqual(audit['checked'], audit['staleTargetsRejected'])
        self.assertEqual(audit['checked'], audit['factUpdatedTargetsAccepted'])
        self.assertFalse(audit['nativeGenerationPerformed'])

    def test_native_eos_and_cap_receipts_required_without_repairs(self):
        bound, rows = self.bound(), self.rows(native=True)
        self.assertTrue(data.assemble(bound, rows)['validation']['valid'])
        self.assertFalse(data.assemble(bound, self.rows())['validation']['valid'])
        mutations = (lambda row: row.update(finishReason='length'), lambda row: row.update(arSteps=True),
                     lambda row: row.update(rawSampledTokenIds=[True, 200]), lambda row: row.update(maxTokens=999),
                     lambda row: row.update(stopTokenId=None), lambda row: row.update(tokenIds=[100, 200]))
        for mutate in mutations:
            broken = copy.deepcopy(rows)
            mutate(broken[1])
            self.assertFalse(data.assemble(bound, broken)['validation']['valid'])

    def test_membership_kind_and_action_bindings(self):
        bound, rows = self.bound(), self.rows(native=True)
        for broken in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                data.assemble(bound, broken)
        broken = copy.deepcopy(rows)
        broken[1]['kind'] = 'badge'
        self.assertFalse(data.assemble(bound, broken)['validation']['valid'])
        payload = data.assemble(bound, rows)['payload']
        for actual, source in zip((c for section in payload['sections'] for c in section['cards']), bound['cards'], strict=True):
            self.assertEqual(actual['variantId'], source['action']['variantId'])
            self.assertEqual(actual['image'], source['action']['image'])
            self.assertIsNone(actual['stock'])
            self.assertIsNone(actual['shipping'])

    def test_private_request_facts_unknown_constraints_and_bad_indices_reject(self):
        case = copy.deepcopy(self.groups['train'][0])
        for key, value in (('hiddenPersona', 'private'), ('targetValue', 'secret'), ('split', 'train')):
            bad = copy.deepcopy(case)
            bad['request'][key] = value
            with self.assertRaises(ValueError):
                data.bind_case(bad, self.snapshot)
        for key, value in (('itemBudget', True), ('itemBudget', float('inf')), ('hardRequirements', ['in stock'])):
            bad = copy.deepcopy(case)
            bad['request'][key] = value
            with self.assertRaises(ValueError):
                data.bind_case(bad, self.snapshot)
        for key, value in (('selectedProductIndices', [0, 0, 1, 2, 3, 4]),
                           ('selectedColorIndices', [True, 0, 0, 0, 0, 0]),
                           ('selectedColorIndices', [3, 0, 0, 0, 0, 0])):
            with self.assertRaises(ValueError):
                data.bind_case({**case, key: value}, self.snapshot)
        bound = self.bound()
        bound['cards'][0]['facts']['hiddenPersona'] = 'secret'
        with self.assertRaises(ValueError):
            data.layout(bound)

    def test_freeze_check_hashes_and_preserve_prior_output(self):
        with tempfile.TemporaryDirectory() as parent:
            destination = Path(parent) / 'fresh'
            manifest = data.freeze(destination)
            loaded, cases, fields = data.load_dataset(destination)
            self.assertEqual(manifest, loaded)
            self.assertEqual((len(cases['train']), len(fields['eval'])), (16, 160))
            self.assertFalse(loaded['nativeExecution'])
            self.assertFalse(loaded['tokenizationPerformed'])
            with self.assertRaises(FileExistsError):
                data.freeze(destination)
            path = destination / 'fields.train.jsonl'
            path.write_text(path.read_text() + '\n')
            with self.assertRaises(ValueError):
                data.load_dataset(destination)


if __name__ == '__main__':
    unittest.main()
