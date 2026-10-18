"""CPU-only conventional baseline parser, source validation and preflight tests."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from .store_autoregressive_baseline import (CONFIG, build_messages, evaluate_full,
    first_section, full_sections, paired_hybrid_readout, prepare, prompt_tokens,
    summarize_pairs, validate_sections)
from .store_catalog_v1 import load_snapshot
from .store_demo_engine import checked_request
from .store_rank_session import action_bindings


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = load_snapshot()
        cls.value = {'query': 'a mug for my morning routine'}
        request, _ = checked_request(cls.value)
        cards = action_bindings(request, list(range(6)), cls.snapshot)
        cls.sections = []
        for start in (0, 3):
            section = {'heading': 'Browse the collection', 'cards': []}
            for card in cards[start:start + 3]:
                f = card['facts']
                section['cards'].append({'name': f['name'], 'title': f"{f['name']} in {f['selectedColor']}",
                    'badge': f"In {f['selectedColor']}",
                    'copy': f"{f['name']} is shown in {f['selectedColor']} at ${f['itemPrice']:.2f}."})
            cls.sections.append(section)

    def test_complete_twenty_fresh_fields_and_six_real_actions(self):
        result = validate_sections(self.sections, self.value, self.snapshot)
        self.assertEqual(result['cardCount'], 6)
        self.assertEqual(result['validation']['requiredFields'], 20)
        self.assertEqual(result['validation']['acceptedFields'], 20)
        self.assertEqual(result['payload']['selectedProductIndices'], list(range(6)))
        self.assertTrue(result['validation']['allStrictTextFieldsAccepted'])
        self.assertTrue(result['validation']['usableCopyProxy'])
        self.assertTrue(all(not row['perFieldEOS'] for row in result['rawFields']))

    def test_first_three_parsed_only_when_actual_section_closed(self):
        raw = '{"sections":[' + json.dumps(self.sections[0])
        self.assertIsNone(first_section(raw[:-1]))
        first = first_section(raw)
        self.assertEqual(first, self.sections[0])
        result = validate_sections([first], self.value, self.snapshot, first=True)
        self.assertEqual(result['cardCount'], 3)
        self.assertEqual(result['validation']['requiredFields'], 10)

    def test_no_fence_substring_or_prefix_repair(self):
        valid = json.dumps({'sections': self.sections})
        for raw in ('```json\n' + valid + '\n```', 'Here is your result: ' + valid,
                'junk' + valid, '{"other":[],"sections":[' + json.dumps(self.sections[0]) + ']}'):
            self.assertIsNone(first_section(raw))
        with self.assertRaises(ValueError):
            full_sections(valid + ' extra')

    def test_duplicate_keys_rejected(self):
        raw = '{"sections":[{"heading":"One","heading":"Two","cards":[]}]}'
        self.assertIsNone(first_section(raw))
        with self.assertRaises(ValueError):
            full_sections(raw)

    def test_structural_failures_not_repaired_or_padded(self):
        for sections in ([self.sections[0]], self.sections + [self.sections[0]],
                [{'heading': 'Browse the collection', 'cards': self.sections[0]['cards'][:2]}]):
            with self.assertRaises(ValueError):
                validate_sections(sections, self.value, self.snapshot)
        repeated = deepcopy(self.sections)
        repeated[1]['cards'][0] = deepcopy(repeated[0]['cards'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_sections(repeated, self.value, self.snapshot)

    def test_guard_violation_cannot_be_hidden_by_copy_abstention(self):
        with self.assertRaises(ValueError):
            validate_sections(self.sections, {'query': 'browse', 'itemBudget': 25}, self.snapshot)
        with self.assertRaises(ValueError):
            validate_sections(self.sections, {'query': 'Everywhere rain jacket'}, self.snapshot)

    def test_false_field_is_null_without_mutating_raw_generation(self):
        sections = deepcopy(self.sections)
        bad = sections[0]['cards'][0]['copy'].replace('$24.00', '$999.00')
        sections[0]['cards'][0]['copy'] = bad
        result = validate_sections(sections, self.value, self.snapshot)
        self.assertIsNone(result['payload']['sections'][0]['cards'][0]['bodyCopy'])
        self.assertEqual(next(row for row in result['rawFields'] if row['field'] == 'C1')['text'], bad)
        self.assertFalse(result['validation']['usableCopyProxy'])
        self.assertEqual(result['validation']['acceptedFields'], 19)

    def test_natural_copy_and_strict_grammar_scores_separate(self):
        sections = deepcopy(self.sections)
        original = sections[0]['cards'][0]
        color = self.snapshot['inputs']['products'][0]['colorVariants'][0]['colorName']
        original['copy'] = f"{original['name']} comes in {color}."
        result = validate_sections(sections, self.value, self.snapshot)
        self.assertTrue(result['validation']['usableCopyProxy'])
        self.assertFalse(result['validation']['allStrictTextFieldsAccepted'])

    def test_global_eos_distinct_from_complete_json(self):
        raw = json.dumps({'sections': self.sections})
        self.assertTrue(evaluate_full(raw, 'stop', self.value, self.snapshot)['usefulComplete'])
        self.assertFalse(evaluate_full(raw, 'length', self.value, self.snapshot)['usefulComplete'])

    def test_shortfall_workload_is_explicitly_different(self):
        value = {'query': 'I need waterproof'}
        messages, public, guards = build_messages(value, self.snapshot)
        self.assertEqual(public['requestedCardCount'], 1)
        self.assertEqual(len(public['catalog']), 16)
        self.assertEqual(public['neutralCopyRequest']['preferences'], [])
        self.assertEqual(public['actualPublicRequest']['query'], value['query'])
        card = deepcopy(self.sections[1]['cards'][0])
        result = validate_sections([{'heading': 'Browse the collection', 'cards': [card]}], value, self.snapshot)
        self.assertEqual(result['cardCount'], 1)
        self.assertEqual(result['validation']['requiredFields'], 4)
        self.assertEqual(result['payload']['shortfall'], 5)

    def test_entire_public_catalog_without_routes_images_hidden_truth(self):
        messages, public, _ = build_messages(self.value, self.snapshot)
        self.assertEqual(len(public['catalog']), 16)
        for product in public['catalog']:
            self.assertEqual(set(product), {'name', 'category', 'itemPrice', 'searchTerms', 'availableColors', 'selectedColor'})
        self.assertNotIn('targetValue', json.dumps(messages))
        self.assertNotIn('persona', json.dumps(messages))

    def test_bounded_token_preflight_does_not_truncate(self):
        class FakeTokenizer:
            def __init__(self, count):
                self.count = count
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return list(range(self.count))
        tokenizer = FakeTokenizer(3072)
        self.assertEqual(len(prompt_tokens(tokenizer, [])), 3072)
        self.assertFalse(tokenizer.kwargs['enable_thinking'])
        with self.assertRaises(ValueError):
            prompt_tokens(FakeTokenizer(3073), [])

    def test_prepare_freezes_exact_requests_workload_and_optional_hybrid_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); requests = root / 'requests.json'; hybrid = root / 'hybrid.json'
            requests.write_text(json.dumps([{'caseId': 'mug', 'request': self.value},
                {'caseId': 'rain', 'request': {'query': 'waterproof'}}]))
            hybrid.write_text(json.dumps({'input': self.value, 'status': 'completed', 'events': [], 'provenance': {'copyCheckpointSha256': 'example'}}))
            manifest = prepare(requests, root / 'preflight', hybrid_receipts=[hybrid])
            self.assertFalse(manifest['nativeExecution'])
            self.assertFalse(manifest['tokenizerPreflight'])
            self.assertEqual([c['requiredCopyFields'] for c in manifest['cases']], [20, 4])
            self.assertEqual(manifest['cases'][1]['workload'], 'separate-guard-shortfall-workload')
            self.assertEqual(manifest['pairedHybridReceipts'][0]['caseIds'], ['mug'])
            with self.assertRaises(FileExistsError):
                prepare(requests, root / 'preflight')

    def test_paired_as_served_proxy_requires_each_title_and_body_not_badges(self):
        result = validate_sections(self.sections, self.value, self.snapshot)
        sections = result['payload']['sections']
        first = deepcopy(sections[:1]); first[0]['cards'][0]['badge'] = None
        full = deepcopy(sections); full[1]['cards'][0]['bodyCopy'] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            path.write_text(json.dumps({'input': self.value, 'status': 'completed', 'cards': [{}] * 6,
                'events': [{'type': 'firstThree', 'sections': first, 'acceptedFields': 9, 'requiredFields': 10, 'timing': {'readyMs': 4000}},
                    {'type': 'complete', 'sections': full, 'acceptedFields': 19, 'requiredFields': 20, 'timing': {'readyMs': 5000}}]}))
            paired = paired_hybrid_readout(path, self.snapshot)
            self.assertEqual(paired['cardCount'], 6)
            self.assertTrue(paired['endpoints'][0]['usableCopyProxy'])
            self.assertFalse(paired['endpoints'][1]['usableCopyProxy'])
            self.assertFalse(paired['endpoints'][0]['asServedAllPracticalFieldsAccepted'])
            summary = summarize_pairs([paired])['six-card-twenty-field']
            self.assertEqual(summary['firstThree']['usableCopyProxyPasses'], 1)
            self.assertEqual(summary['complete']['usableCopyProxyPasses'], 0)
            self.assertEqual(summary['complete']['medianUsableReadyMs'], None)


if __name__ == '__main__':
    unittest.main()
