"""CPU-only compact candidate contracts. Mock token rows are not native evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import store_compact_copy_v1 as compact
from . import store_compact_copy_native_v1 as native


class ChatTokenizer:
    def __init__(self):
        self.vocab = {}
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        tokens = []
        for message in messages:
            if message['role'] != 'user':
                raise AssertionError('Ordinary user evidence turns required')
            tokens.append(1)
            for word in message['content'].split():
                self.vocab.setdefault(word, len(self.vocab) + 100)
                tokens.append(self.vocab[word])
            tokens.append(2)
        return tokens + ([3] if add_generation_prompt else [])


class CompactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.truth, cls.coverage = compact.build_fixture()

    def rows(self, case):
        _, jobs = compact.layout(case); rows = []
        for job in jobs:
            if job['kind'] == 'heading':
                text = 'Your selection'
            else:
                color = case['bound']['cards'][int(job['field'][1:]) - 1]['facts']['selectedColor']
                text = color if job['kind'] == 'badge' else ('Selected ' + color + ' choice' if job['kind'] == 'title' else 'Shown in ' + color)
            rows.append({**job, 'text': text, 'tokenIds': [100], 'rawSampledTokenIds': [100, 99],
                'stopTokenId': 99, 'arSteps': 2, 'finishReason': 'stop'})
        return rows

    def test_imports_do_not_import_mlx_or_load_model(self):
        subprocess.run([sys.executable, '-c', 'import sys; import experiments.ranker.pretrained.store_compact_copy_native_v1; '
            'assert not any(n == "mlx" or n.startswith("mlx.") for n in sys.modules)'], check=True, capture_output=True)

    def test_balanced32_public_style_cases_and48_actual_variant_states(self):
        self.assertEqual(self.coverage['caseCount'], 32)
        self.assertEqual(self.coverage['cardConditions'], 192)
        self.assertEqual(self.coverage['productColorStates'], 48)
        self.assertEqual(self.coverage['eachStateOccurrences'], 4)
        for first, second in zip(self.cases[::2], self.cases[1::2], strict=True):
            self.assertEqual(first['bound']['cards'], second['bound']['cards'])
            self.assertEqual(first['request'], second['request'])
            self.assertEqual([first['publicStyle'], second['publicStyle']], ['plain', 'warm'])
        self.assertEqual(len(self.truth), 32)
        self.assertEqual(self.coverage['trainingTargets'], 0)

    def test_static_root_immutable_facts_and_fresh_style_suffix_twenty_fields(self):
        roots = set()
        for case in self.cases:
            root, jobs = compact.layout(case); roots.add(compact.canonical(root))
            self.assertEqual(tuple(j['field'] for j in jobs), compact.FIELDS)
            self.assertEqual(len(jobs), 20)
            self.assertEqual(sum(j['maxTokens'] for j in jobs), 128)
            self.assertEqual(sum(j['maxTokens'] for j in jobs[:10]), 64)
            for job in jobs:
                self.assertNotIn(case['publicStyle'], job['factMessage']['content'])
                self.assertIn(case['publicStyle'], job['fieldMessage']['content'])
                self.assertNotIn(case['request']['query'], job['fieldMessage']['content'])
                self.assertNotIn('targetValue', job); self.assertNotIn('text', job)
        self.assertEqual(len(roots), 1)
        _, first = compact.layout(self.cases[0]); _, second = compact.layout(self.cases[1])
        self.assertEqual([j['factKey'] for j in first], [j['factKey'] for j in second])
        self.assertNotEqual(first[0]['fieldMessage'], second[0]['fieldMessage'])

    def test_preflight_requires_exact_common_fact_full_chat_prefix_without_truth_read(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'fixture'; compact.write_fixture(data)
            (data / 'truth.json').unlink()
            result = compact.token_preflight(ChatTokenizer(), data)
            self.assertFalse(result['nativeExecution']); self.assertFalse(result['truthRead'])
            self.assertEqual(result['modelCalls'], 0); self.assertEqual(result['fixedFields'], 640)
            self.assertEqual(len(result['rootIdentities']), 1)
            self.assertTrue(all(r['factPrefixTokens'] < r['promptTokens'] and r['rootTokens'] < r['factPrefixTokens'] for r in result['rows']))
            broken = ChatTokenizer(); original = broken.apply_chat_template
            def changed(messages, **kwargs):
                tokens = original(messages, **kwargs)
                if len(messages) > 2:
                    tokens[0] = 999
                return tokens
            broken.apply_chat_template = changed
            with self.assertRaisesRegex(ValueError, 'token-prefix equality'):
                compact.token_preflight(broken, data)

    def test_manifest_rejects_changed_public_cases_and_hidden_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'fixture'; compact.write_fixture(data)
            with self.assertRaises(FileExistsError):
                compact.write_fixture(data)
            cases = json.loads((data / 'cases.json').read_text()); cases[0]['teacherTarget'] = 'forbidden'
            compact.dump(data / 'cases.json', cases)
            with self.assertRaisesRegex(ValueError, 'public cases changed'):
                compact.load_cases(data)
            manifest = json.loads((data / 'manifest.json').read_text()); manifest['fileSha256']['cases.json'] = compact.digest(data / 'cases.json')
            compact.dump(data / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'private/teacher'):
                compact.load_cases(data)

    def test_all20_generated_values_original_metadata_and_actual_serialization(self):
        for case in self.cases:
            rows = self.rows(case)
            full = compact.assemble(case, rows); first = compact.assemble(case, rows[:10], first_three=True)
            self.assertTrue(full['validation']['valid'], full['validation'])
            self.assertTrue(first['validation']['valid'])
            self.assertFalse(full['validation']['semanticProof'])
            self.assertEqual(full['validation']['requiredFields'], 20)
            self.assertEqual(len(full['payload']['cards']), 6)
            self.assertEqual(full['payload']['cards'][0]['originalName'], case['bound']['cards'][0]['facts']['name'])
            self.assertEqual(full['payloadSha256'], compact.hashed(full['payload']))
            with self.assertRaisesRegex(ValueError, 'field roster'):
                compact.assemble(case, rows[:-1])

    def test_cap_finish_is_failure_not_short_text_success_and_no_word_repair(self):
        case = self.cases[0]; rows = self.rows(case); rows[1]['finishReason'] = 'length'; rows[1]['stopTokenId'] = None
        result = compact.assemble(case, rows)
        self.assertFalse(result['validation']['valid'])
        self.assertIn('missing-native-eos-or-cap-truncation', result['validation']['fields'][1]['rejections'])
        self.assertEqual(result['payload']['cards'][0]['title']['text'], rows[1]['text'])

    def test_factual_swaps_and_unsupported_claims_reject_without_exact_teacher_match(self):
        facts = {'name': 'Trail mug', 'selectedColor': 'Cobalt', 'itemPrice': 24}
        check = lambda text, kind='body': compact.text_rejections(text, kind, facts, {'Cobalt', 'Sage', 'Ivory'})
        for text in ('Shown in Cobalt', 'Your Cobalt choice', 'Cobalt for your day'):
            self.assertEqual(check(text), [])
        self.assertIn('wrong-selected-color', check('Shown in Sage'))
        self.assertIn('wrong-item-price', check('Cobalt at $23'))
        self.assertIn('unsupported-attribute', check('Waterproof Cobalt choice'))
        self.assertIn('selected-fact-support-missing', check('For your day'))
        self.assertIn('original-name-only', check('Trail mug', 'title'))
        self.assertIn('nonplain-or-uppercase', check('COBALT CHOICE'))
        self.assertEqual(check('Cobalt', 'badge'), [])
        self.assertIn('badge-color-missing', check('Selected shade', 'badge'))

    def test_native_api_rejects_unowned_call_without_importing_runtime(self):
        with self.assertRaisesRegex(ValueError, 'Root owner'):
            native.run_native_comparison(None, None, '/missing/adapter', 'sha', '/tmp/unused-compact', '/missing/preflight')

    def test_fixed32_summary_keeps_invalid_attempts_out_of_valid_speed_samples(self):
        arms = [{'caseId': c['caseId'], 'mode': mode, 'path': scheme, 'status': 'unattempted'}
            for mode in native.MODES for c in self.cases for scheme in native.PATHS]
        self.assertEqual(len(arms), 128)
        arms[0].update(status='completed-attempt', firstThree={'validation': {'valid': True}},
            fullSix={'validation': {'valid': False}}, firstThreeReadyMs=10, fullSixReadyMs=20)
        result = native.summary(arms)['frozen/common-root']
        self.assertEqual(result['fixedCases'], 32)
        self.assertEqual(result['validFirstThreeRateFixedDenominator'], 1/32)
        self.assertEqual(result['validFirstThreeMs'], [10])
        self.assertEqual(result['attemptFullSixMs'], [20])
        self.assertEqual(result['validFullSixMs'], [])

    def test_diagnostic8_plan_is_predeclared_paired_and_preserves_full32_plan(self):
        declaration = native.experiment_plan('diagnostic8')
        self.assertEqual(declaration['caseIndices'], [0, 1, 8, 9, 16, 17, 24, 25])
        self.assertEqual(declaration['paths'], ['fact-prefix'])
        self.assertEqual((declaration['fixedCases'], declaration['fixedArms']), (8, 16))
        cases = [self.cases[i] for i in declaration['caseIndices']]
        self.assertEqual({card['productIndex'] for c in cases for card in c['bound']['cards']}, set(range(16)))
        self.assertEqual(native.summary([], 8, ['fact-prefix'])['frozen/fact-prefix']['fixedCases'], 8)
        self.assertEqual(native.experiment_plan('full32')['fixedArms'], 128)
        with self.assertRaisesRegex(ValueError, 'Explicit'):
            native.experiment_plan('fastest-eval-results')

    def test_native_suffix_preserves_chat_turns_and_both_paths_have_identical_tokens(self):
        tokenizer = ChatTokenizer(); model = object(); common, jobs = compact.layout(self.cases[0]); job = jobs[1]
        encode = lambda messages, generate=False: tokenizer.apply_chat_template(messages, add_generation_prompt=generate)
        root = {'model': model, 'messages': [common], 'tokens': encode([common]), 'cache': [], 'consumed': False}
        fact = {'model': model, 'messages': [common, job['factMessage']],
            'tokens': encode([common, job['factMessage']]), 'cache': [], 'consumed': False}
        root_tokens, _ = native.exact_suffix(tokenizer, job['messages'], root, model)
        fact_tokens, _ = native.exact_suffix(tokenizer, [job['fieldMessage']], fact, model)
        self.assertEqual(root['tokens'] + root_tokens, fact['tokens'] + fact_tokens)
        self.assertEqual(root['tokens'] + root_tokens, encode([common] + job['messages'], True))
        with self.assertRaisesRegex(ValueError, 'pristine'):
            native.exact_suffix(tokenizer, job['messages'], root, model)

    def test_lru_is_bounded_and_detects_eviction_corruption_and_rebuild_mismatch(self):
        root = {'cache': [{'sha': 'root'}]}
        state_hash = lambda cache: cache[0]['sha']
        fact_state = lambda bb, source, message: {'cache': [{'sha': message['content']}]}
        with patch('experiments.ranker.pretrained.branch_local_copy.prepare_root', side_effect=lambda *a: (copy.deepcopy(root), {})), \
                patch('experiments.ranker.pretrained.native_batch_diagnosis.state_hash', side_effect=state_hash), \
                patch.object(native, '_fact_state', side_effect=fact_state):
            pool = native.PrefixPool(object(), {}, True)
            job = lambda key: {'factKey': key, 'factMessage': {'content': key}, 'fieldMessage': {'content': 'title'}}
            pool.source(job('a')); pool.source(job('a')); pool.source(job('b')); pool.source(job('c'))
            self.assertEqual((pool.hits, pool.misses, pool.evictions), (1, 3, 1))
            self.assertEqual(len(pool.entries), 2); self.assertTrue(pool.verify()['sourceCachesUnchanged'])
            with patch.object(native, '_fact_state', return_value={'cache': [{'sha': 'wrong'}]}):
                with self.assertRaisesRegex(AssertionError, 'fact prefix differs'):
                    pool.source(job('b'), rebuild=True)
            pool.entries['b'][0]['cache'][0]['sha'] = 'mutated'
            with self.assertRaisesRegex(AssertionError, 'Evicted source'):
                pool.source(job('d'))


if __name__ == '__main__':
    unittest.main()
