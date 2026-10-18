"""Offline gold audit only; never import in native encoding, fitting or serving."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
DATA = HERE / 'persona_qwen_independent_eval_v2'


def read(name): return json.loads((DATA / name).read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


class IndependentQwenEvalPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public = read('public.json')
        cls.profiles = {p['profileId']: p for p in cls.public['profiles']}
        cls.labels = {l['profileId']: l for l in read('labels.private.json')['labels']}
        cls.products = {p['id']: p for p in json.loads((HERE / 'synthetic_history_v1/catalog-binding.json').read_text())['products']}

    def test_frozen_hashes_and_preoutcome_gate(self):
        manifest = read('MANIFEST.json')
        for name, expected in manifest['files'].items(): self.assertEqual(sha(DATA / name), expected)
        self.assertIs(manifest['trainAllowed'], False); self.assertIs(manifest['modelSelectionAllowed'], False)
        self.assertIs(manifest['predictionFreezeRequiredBeforePrivateLabels'], True)
        recipe = read('RECIPE_FROZEN.json')
        self.assertEqual(recipe['catalogBindingSha256'], sha(HERE / 'synthetic_history_v1/catalog-binding.json'))
        self.assertEqual(recipe['authoringSourceSha256'], sha(HERE / 'author_persona_qwen_independent_eval_v2.py'))
        gate = recipe['agreedRootGateBeforeOutcomes']
        self.assertEqual((gate['novelRelevantTop1AtLeast'], gate['highestGradeTop1AtLeast'], gate['acceptablePositiveInTop3AtLeast']), (24, 20, 28))

    def test_exact_32_cases_16_pairs_balanced_known_catalog(self):
        self.assertEqual(len(self.profiles), 32); self.assertEqual(set(self.profiles), set(self.labels))
        self.assertEqual(len(self.public['pairs']), 16)
        self.assertEqual(Counter(Counter(p['userGroupId'] for p in self.profiles.values()).values()), {2: 16})
        self.assertEqual(Counter(l['primaryProductId'] for l in self.labels.values()), {p: 2 for p in self.products})

    def test_same_public_context_with_only_observed_purchase_identity_changed(self):
        for pair in self.public['pairs']:
            pa, pb = (self.profiles[i] for i in pair['profileIds'])
            a, b = pa['request'], pb['request']
            self.assertEqual({k:v for k,v in a.items() if k != 'publicHistory'}, {k:v for k,v in b.items() if k != 'publicHistory'})
            self.assertEqual(a['publicHistory'][1:], b['publicHistory'][1:])
            old, new = a['publicHistory'][0], b['publicHistory'][0]
            for key in ('eventId', 'eventType', 'source', 'occurredAt'): self.assertEqual(old[key], new[key])
            self.assertIs(old['external'], True); self.assertNotIn('productId', old)
            self.assertEqual(new['productId'], self.labels[pa['profileId']]['primaryProductId'])
            self.assertEqual(self.labels[pb['profileId']]['relevanceGrades'][new['productId']], 0)

    def test_future_labels_stay_out_of_public_input_and_history_facts_are_real(self):
        for pid, profile in self.profiles.items():
            self.assertEqual(set(profile), {'profileId', 'userGroupId', 'sessionId', 'request'})
            request = profile['request']
            self.assertEqual(set(request), {'query', 'itemBudget', 'hardRequirements', 'persona', 'publicHistory'})
            self.assertEqual(set(request['persona']), {'background', 'colorPreferences'})
            self.assertEqual(len(request['publicHistory']), 4)
            observed = set()
            for e in request['publicHistory']:
                if e.get('external'):
                    self.assertNotIn('productId', e); self.assertNotIn('variantId', e)
                else:
                    product = self.products[e['productId']]
                    variant = next(v for v in product['variants'] if v['id'] == e['variantId'])
                    self.assertEqual((e['itemName'], e['category'], e['selectedColor']), (product['name'], product['category'], variant['colorName']))
                    observed.add(product['id'])
            self.assertNotIn(self.labels[pid]['primaryProductId'], observed)

    def test_gold_only_supports_unowned_eligible_items_with_real_variants(self):
        for pid, label in self.labels.items():
            r = self.profiles[pid]['request']; grades = label['relevanceGrades']
            self.assertEqual(set(grades), set(self.products))
            self.assertEqual([p for p,g in grades.items() if g == 3], [label['primaryProductId']])
            self.assertTrue(all(type(g) is int and 0 <= g <= 3 for g in grades.values()))
            owned = {e['productId'] for e in r['publicHistory'] if e['eventType'] == 'purchase' and 'productId' in e}
            self.assertEqual(set(label['alreadyOwnedProductIds']), owned)
            self.assertTrue(all(grades[p] == 0 for p in owned))
            allowed = set()
            for p, g in grades.items():
                if not g: continue
                product = self.products[p]; self.assertLessEqual(product['price'], r['itemBudget'])
                allowed.update(v['id'] for v in product['variants'] if all(h['value'] == v['colorName'] for h in r['hardRequirements']))
            self.assertLessEqual(set(label['acceptableVariantIds']), allowed)
            self.assertLessEqual(set(label['preferredVariantIds']), set(label['acceptableVariantIds']))
            self.assertTrue(label['preferredVariantIds'])


if __name__ == '__main__': unittest.main()
