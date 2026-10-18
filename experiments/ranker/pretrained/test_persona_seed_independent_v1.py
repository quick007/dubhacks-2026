"""Independent stdlib audit of frozen synthetic profiles, split and public facts."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parent / 'synthetic_history_v1'


def read(name):
    return json.loads((DATA / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IndependentPersonaSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = read('profiles.public.json')['profiles']
        cls.demo = read('demo-people.json')['people']
        cls.split = read('split.json')
        cls.binding = read('catalog-binding.json')
        cls.products = {p['id']: p for p in cls.binding['products']}
        cls.labels = {s: read('labels.' + s + '.json')['labels'] for s in ('train', 'heldout')}

    def test_all_frozen_files_and_source_bytes(self):
        manifest = read('manifest.json')
        for name, expected in manifest['files'].items():
            self.assertEqual(sha(DATA / name), expected, name)
        for name, expected in manifest['sourceSha256'].items():
            self.assertEqual(sha(ROOT / name), expected, name)
        freeze = read('PUBLIC_SPLIT_FROZEN.json')
        self.assertIs(freeze['publicAndSplitPublishedBeforeLabels'], True)
        for name, expected in freeze['files'].items():
            self.assertEqual(sha(DATA / name), expected)

    def test_family_group_session_and_demo_separation(self):
        self.assertEqual(len(self.profiles), 200)
        self.assertEqual(len({p['profileId'] for p in self.profiles}), 200)
        self.assertEqual(len({p['sessionId'] for p in self.profiles}), 200)
        self.assertEqual(Counter(Counter(p['userGroupId'] for p in self.profiles).values()), {2: 100})
        tr, ev = (set(self.split[s + 'ProfileIds']) for s in ('train', 'heldout'))
        tg, eg = (set(self.split[s + 'GroupIds']) for s in ('train', 'heldout'))
        tf, ef = (set(self.split[s + 'FamilyIds']) for s in ('train', 'heldout'))
        self.assertEqual((len(tr), len(ev), len(tg), len(eg), len(tf), len(ef)), (160, 40, 80, 20, 20, 5))
        self.assertFalse(tr & ev or tg & eg or tf & ef)
        self.assertEqual(tr | ev, {p['profileId'] for p in self.profiles})
        group_rows = {g['userGroupId']: g for g in self.split['groups']}
        for p in self.profiles:
            group = group_rows[p['userGroupId']]
            self.assertIn(p['profileId'], group['profileIds'])
            self.assertEqual(p['profileId'] in tr, group['split'] == 'train')
            self.assertEqual(group['familyId'] in tf, group['split'] == 'train')
        self.assertEqual(len(self.demo), 10)
        demo_groups = set(self.split['demoGroupIds'])
        self.assertEqual(len(demo_groups), 10)
        self.assertFalse(demo_groups & (tg | eg))

    def test_public_requests_have_no_gold_and_only_factual_past_events(self):
        self.assertEqual(len(self.products), 16)
        self.assertEqual(sum(len(p['variants']) for p in self.products.values()), 48)
        self.assertEqual(sha(ROOT / self.binding['source']), self.binding['sourceSha256'])
        raw = json.loads((ROOT / self.binding['source']).read_text())
        self.assertEqual(list(self.products), [p['id'] for p in raw['products']])
        cutoff = datetime.fromisoformat('2026-10-18T00:00:00+00:00')
        for row in [*self.profiles, *self.demo]:
            r = row['request']
            self.assertEqual(set(r), {'query', 'itemBudget', 'hardRequirements', 'persona', 'publicHistory'})
            self.assertEqual(set(r['persona']), {'background', 'colorPreferences'})
            self.assertEqual(r['hardRequirements'], [])
            self.assertTrue(4 <= len(r['publicHistory']) <= 12)
            times = []
            for event in r['publicHistory']:
                self.assertLessEqual(set(event), {'eventId', 'eventType', 'itemName', 'category', 'occurredAt', 'source', 'external', 'selectedColor', 'productId', 'variantId'})
                self.assertIn(event['eventType'], ('view', 'cart', 'purchase'))
                self.assertEqual(event['source'], 'demo-seed')
                time = datetime.fromisoformat(event['occurredAt'].replace('Z', '+00:00'))
                self.assertIsNotNone(time.utcoffset()); self.assertLess(time, cutoff); times.append(time)
                if event.get('external') is True:
                    self.assertNotIn('productId', event); self.assertNotIn('variantId', event)
                else:
                    product = self.products[event['productId']]
                    variant = next(v for v in product['variants'] if v['id'] == event['variantId'])
                    self.assertEqual((event['itemName'], event['category']), (product['name'], product['category']))
                    self.assertEqual(event['selectedColor'], variant['colorName'])
            self.assertEqual(times, sorted(times))
            self.assertEqual(len({e['eventId'] for e in r['publicHistory']}), len(times))

    def test_separate_labels_align_to_split_and_observed_ownership(self):
        by_id = {p['profileId']: p for p in self.profiles}
        variant_ids = {v['id'] for p in self.products.values() for v in p['variants']}
        for split, labels in self.labels.items():
            self.assertEqual({r['profileId'] for r in labels}, set(self.split[split + 'ProfileIds']))
            for label in labels:
                request = by_id[label['profileId']]['request']
                grades = label['relevanceGrades']
                self.assertEqual(set(grades), set(self.products))
                self.assertTrue(all(type(v) is int and 0 <= v <= 3 for v in grades.values()))
                self.assertEqual(max(grades.values()), 3)
                self.assertGreaterEqual(sum(v > 0 for v in grades.values()), 2)
                owned = {e['productId'] for e in request['publicHistory'] if e['eventType'] == 'purchase' and 'productId' in e}
                self.assertEqual(set(label['alreadyOwnedProductIds']), owned)
                self.assertTrue(all(grades[p] == 0 for p in owned))
                for pid, grade in grades.items():
                    if grade > 0:
                        self.assertLessEqual(self.products[pid]['price'], request['itemBudget'])
                self.assertLessEqual(set(label['preferredVariantIds']), set(label['acceptableVariantIds']))
                self.assertLessEqual(set(label['acceptableVariantIds']), variant_ids)

    def test_reported_class_coverage_is_synthetic_and_limited(self):
        primary = {s: Counter(pid for label in rows for pid, grade in label['relevanceGrades'].items() if grade == 3) for s, rows in self.labels.items()}
        self.assertEqual(set(primary['train']), set(self.products))
        self.assertEqual(len(primary['heldout']), 5)
        self.assertEqual(set(primary['heldout'].values()), {8})
        for name in ('profiles.public.json', 'demo-people.json'):
            provenance = read(name)['provenance']
            self.assertEqual(provenance['kind'], 'agent-authored-synthetic')
            self.assertIs(provenance['browserObserved'], False)
            self.assertIs(provenance['humanConversionClaim'], False)

    def test_history_only_diagnostic_pairs_are_matched_and_never_training(self):
        manifest = read('history-only-probes.manifest.json')
        self.assertIs(manifest['trainAllowed'], False)
        self.assertIs(manifest['modelSelectionAllowed'], False)
        self.assertIs(manifest['predictionFreezeRequiredBeforePrivateLabels'], True)
        self.assertEqual(manifest['frozenBaseManifestSha256'], sha(DATA / 'manifest.json'))
        for name, expected in manifest['files'].items():
            self.assertEqual(sha(DATA / name), expected)
        for name, expected in manifest['sourceSha256'].items():
            self.assertEqual(sha(ROOT / name), expected)
        pairs = read('history-only-probes.public.json')['pairs']
        labels = {r['pairId']: r for r in read('history-only-probes.labels.json')['labels']}
        self.assertEqual(len(pairs), 10)
        self.assertEqual({r['pairId'] for r in pairs}, set(labels))
        fit_ids = {p['profileId'] for p in self.profiles}
        for pair in pairs:
            a, b = (pair[key]['request'] for key in ('profileA', 'profileB'))
            self.assertEqual({k:v for k,v in a.items() if k != 'publicHistory'}, {k:v for k,v in b.items() if k != 'publicHistory'})
            self.assertEqual(a['publicHistory'][1:], b['publicHistory'][1:])
            old, new = a['publicHistory'][0], b['publicHistory'][0]
            for key in ('eventId', 'eventType', 'source', 'occurredAt'):
                self.assertEqual(old[key], new[key])
            self.assertEqual(old['eventType'], 'purchase')
            self.assertIs(old['external'], True)
            self.assertNotIn('productId', old)
            label = labels[pair['pairId']]
            pid = label['changedPurchaseProductId']
            self.assertEqual(new['productId'], pid)
            self.assertEqual(label['profileA']['relevanceGrades'][pid], 3)
            self.assertEqual(label['profileB']['relevanceGrades'][pid], 0)
            self.assertIn(pid, label['profileB']['alreadyOwnedProductIds'])
            self.assertNotIn(pid, label['profileA']['alreadyOwnedProductIds'])
            for key in ('profileA', 'profileB'):
                self.assertNotIn(pair[key]['profileId'], fit_ids)
                self.assertEqual(pair[key]['profileId'], label[key]['profileId'])


if __name__ == '__main__':
    unittest.main()
