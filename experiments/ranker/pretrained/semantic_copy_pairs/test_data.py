"""CPU checks of frozen synthetic data; no model, tokenizer, fitting or inference."""
import copy
from collections import defaultdict
import json
import unittest

from .generator import HERE, OLD, TAGS, audit, build, records


class PairedCopyDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.fields, cls.pairs, cls.inventory = build()

    def test_complete_audit_and_all_positive_atoms(self):
        result = audit(self.cases, self.fields, self.pairs, self.inventory)
        self.assertEqual(result, json.loads((HERE / 'AUDIT.json').read_text()))
        self.assertEqual(set(result['positiveTrainingTitleCountsByAttribute']), set(TAGS))
        self.assertGreaterEqual(min(result['positiveTrainingTitleCountsByAttribute'].values()), 2)
        self.assertEqual(result['summary']['train']['fieldExamples'], 504)

    def test_pair_changes_only_observable_preference_context(self):
        by_id = {c['caseId']: c for c in self.cases['train']}
        for pair in self.pairs:
            a, b = [by_id[c] for c in pair['caseIds']]
            left, right = copy.deepcopy(a['request']), copy.deepcopy(b['request'])
            left.pop('recordId'); right.pop('recordId')
            for key in ('task', 'statedPreferences'):
                left['context'].pop(key); right['context'].pop(key)
            self.assertEqual(left, right)
            self.assertEqual(a['catalog'], b['catalog'])
            self.assertEqual([t for t in a['targets'] if t['kind'] == 'heading'],
                             [t for t in b['targets'] if t['kind'] == 'heading'])

    def test_every_case_requires_none_and_fresh_cases_support_both_atoms(self):
        for split, cases in self.cases.items():
            for case in cases:
                self.assertTrue(any(t['kind'] == 'badge' and t['targetValue'] == 'NONE' for t in case['targets']))
                if split != 'train':
                    supported = {t for c in case['publicEvidence']['cards'] for t in c['requiredObservedTags']}
                    self.assertEqual(supported, set(case['request']['context']['statedPreferences']))
                    self.assertEqual(len(supported), 2)

    def test_frozen_records_exactly_match_reproducible_builder(self):
        for split in ('train', 'dev', 'test'):
            self.assertEqual(records(HERE / f'cases.{split}.jsonl'), self.cases[split])
            self.assertEqual(records(HERE / f'fields.{split}.jsonl'), self.fields[split])
        self.assertEqual(records(HERE / 'pairs.jsonl'), self.pairs)

    def test_training_catalog_mutation_and_unchanged_pair_target_fail(self):
        bad = copy.deepcopy(self.cases)
        next(iter(bad['train'][0]['catalog'].values()))['description'] += ' changed'
        with self.assertRaises(AssertionError): audit(bad, self.fields, self.pairs, self.inventory)
        bad = copy.deepcopy(self.cases)
        by_id = {c['caseId']: c for c in bad['train']}
        pair = self.pairs[0]; a, b = [by_id[cid] for cid in pair['caseIds']]
        pid = pair['mustChangeTitleAndBadgeForProductIds'][0]
        card = next(c['index'] for c in b['publicEvidence']['cards']
                    if b['catalog'][pid]['title'] == c['identity'])
        old_value = next(t['targetValue'] for t in a['targets'] if t['field'] == f'B{card}')
        next(t for t in b['targets'] if t['field'] == f'B{card}')['targetValue'] = old_value
        with self.assertRaises(AssertionError): audit(bad, self.fields, self.pairs, self.inventory)

    def test_old_training_and_inspected_replay_sources_remain_distinct(self):
        old_products = {pid: p for c in records(OLD / 'cases.train.jsonl') for pid, p in c['catalog'].items()}
        products = {pid: p for c in self.cases['train'] for pid, p in c['catalog'].items()}
        self.assertEqual(products, old_products)
        replay = json.loads((HERE / 'inspected_replays.json').read_text())
        self.assertTrue(replay['separateDenominatorsRequired'])
        for split in ('dev', 'test'):
            self.assertTrue(set(replay['cases'][split]['caseIds']).isdisjoint(c['caseId'] for c in self.cases[split]))

    def test_duplicate_training_prompts_never_have_contradictory_targets(self):
        answers = defaultdict(set)
        for row in self.fields['train']:
            answers[row['messagesSha256']].add(row['targetValue'])
        self.assertEqual(len(answers), 462)
        self.assertTrue(all(len(values) == 1 for values in answers.values()))


if __name__ == '__main__': unittest.main()
