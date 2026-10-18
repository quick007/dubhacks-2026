"""CPU-only layout/data/snapshot tests. Synthetic teacher values are not outputs."""
import copy
import unittest

from .data import PARENT, build, records
from .layout import assemble_payload, build_layout, fork_root, token_hash, token_inventory
from ..test_branch_local_copy import StubTokenizer


class Cache:
    def __init__(self, state): self.state = state
    @classmethod
    def from_state(cls, state): return cls(state)


class CopyLayoutTests(unittest.TestCase):
    def fixture(self):
        case = records(PARENT / 'cases.train.jsonl')[0]
        root, jobs, evidence = build_layout(case['pendingPagePlan'], case['request'], case['catalog'])
        targets = {t['field']: t['targetValue'] for t in case['targets']}
        for card in evidence['cards']:
            tag = (card['supportedObservedTags'] or card['productFacts']['tags'])[0]
            targets[f'C{card["index"]}'] = f'A {tag} daypack for everyday travel.'
        rows = [{**j, 'text': targets[j['fields'][0]], 'finishReason': 'stop'} for j in jobs]
        return case, root, jobs, evidence, rows

    def test_ten_jobs_same_card_facts_no_sibling_outputs_or_mutation(self):
        case = records(PARENT / 'cases.train.jsonl')[0]; before = copy.deepcopy(case)
        root, jobs, evidence = build_layout(case['pendingPagePlan'], case['request'], case['catalog'])
        self.assertEqual(case, before); self.assertEqual(len(jobs), 10)
        self.assertEqual(sum(j['maxTokens'] for j in jobs), 208)
        for i in range(1, 4):
            messages = [next(j for j in jobs if j['fields'] == [f'{p}{i}'])['messages'][0] for p in ('T', 'B', 'C')]
            self.assertEqual(messages[0], messages[1]); self.assertEqual(messages[1], messages[2])
            self.assertEqual(evidence['cards'][i - 1]['originalNameSubtitle'], evidence['cards'][i - 1]['identity'])
        self.assertFalse(evidence['cardFactCacheReuseImplemented'])

    def test_teacher_payload_and_strict_independent_copy_support(self):
        c, _, _, evidence, rows = self.fixture()
        result = assemble_payload(rows, c['pendingPagePlan'], c['request'], c['catalog'])
        self.assertTrue(result['validation']['valid'], result['validation']['rejections'])
        index = next(card['index'] for card in evidence['cards'] if card['supportedObservedTags'])
        next(r for r in rows if r['fields'] == [f'C{index}'])['text'] = 'A daypack for everyday travel.'
        result = assemble_payload(rows, c['pendingPagePlan'], c['request'], c['catalog'])
        self.assertIn(f'C{index}:body-independent-observed-support-missing', result['validation']['rejections'])

    def test_malformed_copy_and_changed_original_binding_fail_without_repair(self):
        c, _, _, _, rows = self.fixture(); row = next(r for r in rows if r['fields'] == ['C1'])
        row['text'] += '\n'
        result = assemble_payload(rows, c['pendingPagePlan'], c['request'], c['catalog'])
        self.assertFalse(result['validation']['valid']); self.assertEqual(result['rawRows'], rows)

    def test_native_receipt_checks_include_EOS_inside_sampled_cap(self):
        c, _, _, _, rows = self.fixture()
        for r in rows: r.update(tokenIds=[1], rawSampledTokenIds=[1, 99], stopTokenId=99, arSteps=2)
        result = assemble_payload(rows, c['pendingPagePlan'], c['request'], c['catalog'], require_native_receipts=True)
        self.assertTrue(result['validation']['valid'])
        r = next(r for r in rows if r['fields'] == ['C1'])
        r.update(tokenIds=[1] * 32, rawSampledTokenIds=[1] * 32 + [99], stopTokenId=99, arSteps=33)
        result = assemble_payload(rows, c['pendingPagePlan'], c['request'], c['catalog'], require_native_receipts=True)
        self.assertIn('C1:native-sampled-cap-or-EOS-receipt', result['validation']['rejections'])

    def test_full_snapshot_isolation_includes_recurrent_convolution_and_KV(self):
        _, root, _, _, _ = self.fixture(); model = object(); tokens = [10, 11]
        state = {'model': model, 'messages': [root], 'tokens': tokens, 'prefixSha256': token_hash(tokens),
            'consumed': False, 'cache': [Cache({'recurrent': [[1]], 'convolution': [[2]], 'kv': [[3]], 'offset': 2})]}
        branch = fork_root(state, root, tokens, model)
        for key in ('recurrent', 'convolution', 'kv'): branch['cache'][0].state[key][0][0] = 100
        self.assertEqual(state['cache'][0].state, {'recurrent': [[1]], 'convolution': [[2]], 'kv': [[3]], 'offset': 2})
        self.assertIs(branch['model'], model)
        with self.assertRaisesRegex(ValueError, 'never splice'): fork_root(state, root, [42], model)
        with self.assertRaisesRegex(ValueError, 'pristine'): fork_root({**state, 'consumed': True}, root, tokens, model)

    def test_exact_boundary_stub_and_all_frozen_teacher_views(self):
        _, root, jobs, _, _ = self.fixture(); inventory = token_inventory(StubTokenizer(), root, jobs)
        self.assertEqual(len(inventory['rows']), 10)
        cases, fields, pairs, audit = build()
        self.assertEqual([len(fields[s]) for s in ('train', 'dev', 'test')], [360, 30, 30])
        self.assertEqual(audit['pairedFamiliesRetainingVisibleContrast'], 12)
        self.assertNotIn('organizer', audit['summary']['train']['positiveObservedCopyCountsByAttribute'])
        self.assertEqual(audit['summary']['train']['teacherPayloadsPassingStrictValidation'], 36)


if __name__ == '__main__': unittest.main()
