"""CPU-only exact prompt and supervised-stop alignment regressions."""
import copy
import unittest

from .semantic_copy_tokens import build_training_tokens


class StubTokenizer:
    eos_token_id = 99

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        return [10] + [20 + len(m['content']) for m in messages] + [11, 12]

    def encode(self, text, add_special_tokens=False):
        if text == '<|im_end|>':
            return [99]
        if text == 'embedded':
            return [30, 99]
        return [100 + ord(c) for c in text]


class SemanticCopyTokenTests(unittest.TestCase):
    def fixture(self):
        return ([{'role': 'system', 'content': 'rank facts'}],
            {'messages': [{'role': 'user', 'content': 'shared evidence'},
                          {'role': 'user', 'content': 'one title'}],
             'targetValue': 'new', 'maxTokens': 4, 'privateMetadata': 'never a model input'})

    def test_exact_turns_thinking_and_causal_alignment(self):
        rank, row = self.fixture();before = copy.deepcopy((rank, row));tok = StubTokenizer()
        result = build_training_tokens(tok, rank, row)
        self.assertEqual((rank, row), before)
        messages, options = tok.calls[0]
        self.assertEqual(messages, rank + row['messages'])
        self.assertEqual(options, {'tokenize': True, 'return_dict': False,
                                  'add_generation_prompt': True, 'enable_thinking': False})
        self.assertNotIn('targetValue', str(messages));self.assertNotIn('privateMetadata', str(messages))
        self.assertEqual(result['targetTokenIds'], tok.encode('new') + [99])
        self.assertEqual(result['fullTokenIds'], result['promptTokenIds'] + result['targetTokenIds'])
        self.assertEqual(result['teacherInputTokenIds'], result['fullTokenIds'][:-1])
        self.assertEqual(result['predictionStart'], len(result['promptTokenIds']) - 1)
        self.assertEqual(len(result['teacherInputTokenIds'][result['predictionStart']:]), result['supervisedTokenCount'])
        self.assertEqual(result['teacherInputTokenIds'][result['predictionStart'] + 1:], result['targetTokenIds'][:-1])

    def test_native_wrapper_and_plain_tokenizer_eos_receipts(self):
        rank, row = self.fixture();tok = StubTokenizer();tok.eos_token_ids = {98, 99}
        result = build_training_tokens(tok, rank, row)
        self.assertEqual(result['eosReceipt']['acceptedEosTokenIds'], [98, 99])
        self.assertEqual(result['eosReceipt']['stopDeclaredBy'], ['eos_token_ids', 'eos_token_id'])
        tok.eos_token_id = None
        self.assertEqual(build_training_tokens(tok, rank, row)['eosReceipt']['stopDeclaredBy'], ['eos_token_ids'])
        tok.eos_token_ids = {98}
        with self.assertRaisesRegex(ValueError, 'not a declared'):build_training_tokens(tok, rank, row)

    def test_no_context_target_or_stop_truncation(self):
        rank, row = self.fixture();tok = StubTokenizer()
        result = build_training_tokens(tok, rank, row)
        build_training_tokens(tok, rank, row, result['fullTokens'])
        with self.assertRaisesRegex(ValueError, 'context exceeds'):build_training_tokens(tok, rank, row, result['fullTokens'] - 1)
        with self.assertRaisesRegex(ValueError, 'field cap'):build_training_tokens(tok, rank, {**row, 'maxTokens': 3})
        with self.assertRaisesRegex(ValueError, 'embedded stop'):build_training_tokens(tok, rank, {**row, 'targetValue': 'embedded', 'maxTokens': 100})
        class SplitStop(StubTokenizer):
            def encode(self, text, **kwargs):
                return [99, 98] if text == '<|im_end|>' else super().encode(text, **kwargs)
        with self.assertRaisesRegex(ValueError, 'exactly one'):build_training_tokens(SplitStop(), rank, row)

    def test_input_hashes_ignore_metadata_and_include_teacher_forcing(self):
        rank, row = self.fixture();tok = StubTokenizer();base = build_training_tokens(tok, rank, row)
        changed_metadata = build_training_tokens(tok, rank, {**row, 'privateMetadata': 'different', 'caseId': 'other'})
        self.assertEqual(base['modelInputSha256'], changed_metadata['modelInputSha256'])
        changed_target = build_training_tokens(tok, rank, {**row, 'targetValue': 'alt'})
        self.assertEqual(base['promptSha256'], changed_target['promptSha256'])
        self.assertNotEqual(base['modelInputSha256'], changed_target['modelInputSha256'])

    def test_invalid_empty_contract_rejected(self):
        rank, row = self.fixture();tok = StubTokenizer()
        for patch in ({'targetValue': ''}, {'targetValue': ' '}, {'messages': []}, {'maxTokens': 0}, {'maxTokens': True}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):build_training_tokens(tok, rank, {**row, **patch})


if __name__ == '__main__':
    unittest.main()
