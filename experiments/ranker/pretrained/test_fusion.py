"""Offline fusion guards/protocol tests. Stubs do not constitute 9B model evidence."""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .fusion import (FusionModel, append_generation_suffix, build_shared_prefix,
                     generation_messages, load_trained_heads, native_decode,
                     observable_request, supported_preferences, validate_copy)


class TokenizerStub:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = ''.join(f"<{m['role']}>\n{m['content']}\n</turn>\n" for m in messages)
        if add_generation_prompt:
            text += '<assistant>\n'
        return self.encode(text) if tokenize else text


class BackboneStub:
    def __init__(self):
        self.model, self.tokenizer = object(), TokenizerStub()
        self.rows = []

    def encode(self, rows):
        self.rows.append(rows)
        return torch.ones((len(rows), 4)), {'tokenizeMs': 1.0, 'forwardMs': 2.0}


class HeadsStub:
    def __call__(self, values):
        return torch.tensor([[.8, .5] for _ in values])


def receipt(representation='row-batch-v1'):
    return {'verifiedHeadUpdate': True, 'representation': representation,
            'scope': 'unit-test stub; not actual training evidence'}


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.product = {'id': 'product-secret-id', 'title': 'Everyday pack', 'category': 'bags',
                        'description': 'A compact pack', 'price': 30, 'shippingCost': 5, 'inStock': True,
                        'tags': ['compact'], 'presentationVariants': [
                            {'id': 'variant-secret-a', 'title': 'Compact pack', 'imageDescription': 'The pack in use',
                             'facts': ['compact'], 'presentationStyle': 'in-use-view'},
                            {'id': 'variant-secret-b', 'title': 'Everyday pack', 'imageDescription': 'Close view',
                             'facts': ['compact'], 'presentationStyle': 'detail-view'}]}
        self.catalog = {self.product['id']: self.product,
                        'other': {**self.product, 'tags': ['water-resistant', 'durable']}}
        self.request = {'recordId': 'record-secret', 'userId': 'identity-secret', 'asOf': '2026-10-01T00:00:00Z',
                        'candidateProductIds': [self.product['id']],
                        'context': {'task': 'A compact travel bag', 'statedBudget': 40,
                                    'statedRequirements': ['compact'], 'statedPreferences': ['compact'],
                                    'presentationSurface': 'compact-card', 'presentationPreferences': ['detail-view']},
                        'history': [], 'presentationCompatibility': {self.product['id']: {
                            'variant-secret-a': False, 'variant-secret-b': True}}}
        self.valid = 'Title: Compact Everyday Pack for Your Travel\nCopy: A compact pack for your everyday travel.'

    def decoder(self, text=None, finish='stop'):
        self.calls = []
        def decode(model, tokenizer, messages, max_tokens, **kwargs):
            self.calls.append((model, tokenizer, messages, max_tokens, kwargs))
            return {'text': self.valid if text is None else text, 'finishReason': finish,
                    'timing': {'generationMs': 4.0, 'prefillAndFirstTokenMs': 3.0,
                               'autoregressiveAfterFirstTokenMs': 1.0}}
        return decode

    def test_novel_factual_copy_and_explicit_preference(self):
        result = validate_copy(self.valid, self.product, self.catalog, preferred_attributes=['compact'])
        self.assertTrue(result['accepted'], result)
        self.assertTrue(result['novelTitle'])
        self.assertEqual(supported_preferences(self.request, self.product), ['compact'])
        copied = self.valid.replace('Compact Everyday Pack for Your Travel', 'EVERYDAY PACK!')
        self.assertIn('title_matches_prewritten_approved_title', validate_copy(copied, self.product, self.catalog)['claimsRejected'])
        missing = self.valid.replace('Compact ', '').replace('compact ', '')
        self.assertIn('supported_observable_preference_not_demonstrated',
                      validate_copy(missing, self.product, self.catalog, preferred_attributes=['compact'])['claimsRejected'])

    def test_fail_closed_claims_markup_numbers_attributes_and_truncation(self):
        for replacement in ('water-resistant pack', 'durable pack', 'waterproof pack', 'free shipping pack',
                            '999 stars pack', '<b>compact</b> pack', 'compact charger pack', 'compact pack without limits'):
            candidate = self.valid.replace('compact pack', replacement)
            self.assertFalse(validate_copy(candidate, self.product, self.catalog)['accepted'], replacement)
        self.assertIn('generation_truncated_or_missing_stop', validate_copy(self.valid, self.product, self.catalog, 'length')['claimsRejected'])
        # A price/shipping numeral is not permission to claim an unrelated litre capacity.
        self.assertFalse(validate_copy(self.valid.replace('compact pack', '5L compact pack'), self.product, self.catalog)['accepted'])
        self.assertFalse(validate_copy('Title: ' + 'a' * 91 + '\nCopy: A compact pack for travel.', self.product, self.catalog)['accepted'])

    def test_rejected_optional_sentence_never_discards_or_contaminates_valid_novel_title(self):
        candidate = self.valid.replace('A compact pack for your everyday travel.', 'A compact pack with free shipping.')
        validation = validate_copy(candidate, self.product, self.catalog, preferred_attributes=['compact'])
        self.assertFalse(validation['accepted'])
        self.assertTrue(validation['titleAccepted'])
        self.assertFalse(validation['copyAccepted'])
        result = FusionModel(BackboneStub(), HeadsStub(), receipt(), decoder=self.decoder(candidate)).infer(self.request, self.catalog)
        self.assertEqual(result['copy']['status'], 'generated-partial')
        self.assertTrue(result['copy']['generatedTitleFulfilled'])
        self.assertEqual(result['copy']['title'], self.valid.splitlines()[0][7:])
        self.assertIsNone(result['copy']['copy'])
        self.assertTrue(result['copy']['fieldRejections']['copy'])

    def test_staged_ordering_same_wrapper_and_real_completion_timing(self):
        backbone = BackboneStub()
        model = FusionModel(backbone, HeadsStub(), receipt(), decoder=self.decoder())
        events = model.infer_events(self.request, self.catalog)
        structured = next(events)
        self.assertEqual(structured['event'], 'structured')
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(structured['result']['presentationAction']['variantId'], 'variant-secret-b')
        complete = next(events)['result']
        self.assertIs(self.calls[0][0], backbone.model)
        self.assertIs(self.calls[0][1], backbone.tokenizer)
        self.assertEqual(self.calls[0][3], 64)
        self.assertEqual(complete['copy']['status'], 'generated')
        self.assertTrue(complete['copy']['generatedTitleFulfilled'])
        self.assertEqual(complete['copy']['personalization']['status'], 'supported-observable-attribute')
        self.assertFalse(complete['method']['jointlyTrained'])
        self.assertFalse(complete['method']['crossPathPromptCacheReuse'])
        self.assertGreaterEqual(complete['timing']['fullCompletionMs'], complete['timing']['uncachedStructuredMs'])
        self.assertIn('validationMs', complete['timing'])

    def test_private_labels_identity_future_history_never_condition_generation(self):
        backbone = BackboneStub()
        model = FusionModel(backbone, HeadsStub(), receipt(), decoder=self.decoder())
        model.infer(self.request, self.catalog)
        rows_before, messages_before = backbone.rows[-1], self.calls[-1][2]
        changed = copy.deepcopy(self.request)
        changed.update({'recordId': 'different-record', 'userId': 'different-user',
                        'hiddenPreferences': 'private-secret', 'labels': {'best': 'private-secret'},
                        'futureActions': 'future-secret', 'personaFamilyId': 'family-secret'})
        changed['context']['hiddenPreferences'] = 'nested-private-secret'
        changed['history'] = [{'at': '2027-01-01T00:00:00Z', 'productId': self.product['id'],
                               'description': 'future-secret', 'privatePreferences': 'secret'}]
        model.infer(changed, self.catalog)
        self.assertEqual(rows_before, backbone.rows[-1])
        self.assertEqual(messages_before, self.calls[-1][2])
        self.assertNotIn('secret', str(self.calls[-1][2]))
        self.assertNotIn('userId', observable_request(changed))

    def test_history_asymmetry_is_preserved_without_asserting_hidden_intent(self):
        request = copy.deepcopy(self.request)
        request['context']['statedRequirements'] = []
        request['context']['statedPreferences'] = []
        request['history'] = [{'at': '2026-09-01T00:00:00Z', 'productId': self.product['id'],
                               'type': 'view', 'confidence': .35, 'description': 'A short view; intent unknown.'}]
        backbone = BackboneStub()
        result = FusionModel(backbone, HeadsStub(), receipt(), decoder=self.decoder()).infer(request, self.catalog)
        self.assertIn('0.35', str(backbone.rows[-1]))
        self.assertEqual(result['copy']['personalization']['status'], 'generic-insufficient-supported-context')

    def test_fallback_is_never_claimed_generated_and_retry_is_bounded(self):
        result = FusionModel(BackboneStub(), HeadsStub(), receipt(), decoder=self.decoder('Title: Free shipping\nCopy: Free shipping for all.'), retries=1).infer(self.request, self.catalog)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(result['copy']['status'], 'fallback')
        self.assertEqual(result['copy']['source'], 'prewritten-approved-title')
        self.assertFalse(result['copy']['generatedTitleFulfilled'])
        self.assertTrue(result['copy']['claimsRejected'])
        truncated = FusionModel(BackboneStub(), HeadsStub(), receipt(), decoder=self.decoder(finish='length')).infer(self.request, self.catalog)
        self.assertFalse(truncated['copy']['generatedTitleFulfilled'])

    def test_abstention_does_not_generate(self):
        request = copy.deepcopy(self.request)
        request['context']['statedBudget'] = 1
        result = FusionModel(BackboneStub(), HeadsStub(), receipt(), decoder=self.decoder()).infer(request, self.catalog)
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(result['copy']['status'], 'abstained')

    def test_limits_training_receipt_and_same_object_invariant(self):
        for options in ({'max_tokens': 65}, {'retries': 2}, {'max_tokens': 31}):
            with self.assertRaises(ValueError):
                FusionModel(BackboneStub(), HeadsStub(), receipt(), **options)
        with self.assertRaises(ValueError):
            FusionModel(BackboneStub(), HeadsStub(), {})
        with self.assertRaises(ValueError):
            FusionModel(BackboneStub(), HeadsStub(), receipt('shared-prefix-v1'), retries=1)
        backbone = BackboneStub()
        model = FusionModel(backbone, HeadsStub(), receipt(), decoder=self.decoder())
        backbone.model = object()
        with self.assertRaisesRegex(RuntimeError, 'model object changed'):
            model.infer(self.request, self.catalog)

    def test_shared_prefix_all_facts_before_readouts_and_private_boundary(self):
        tokenizer = TokenizerStub()
        tokens, actions, positions, messages = build_shared_prefix(tokenizer, self.request, self.catalog, 10000)
        content = messages[-1]['content']
        self.assertLess(content.index('Approved title: Everyday pack'), content.index('Query product'))
        self.assertEqual(len(actions), 2)
        self.assertEqual(len(positions), 2)
        self.assertTrue(all(0 <= p < len(tokens) for p in positions))
        changed = copy.deepcopy(self.request)
        changed['privateTruth'] = 'do-not-use'
        self.assertEqual(tokens, build_shared_prefix(tokenizer, changed, self.catalog, 10000)[0])
        with self.assertRaisesRegex(ValueError, 'Refusing truncation'):
            build_shared_prefix(tokenizer, self.request, self.catalog, 10)

    def test_exact_full_cache_append_no_trim_fork_or_repeated_consumption(self):
        tokenizer, model = TokenizerStub(), object()
        tokens, _, _, messages = build_shared_prefix(tokenizer, self.request, self.catalog, 10000)
        recurrent_and_conv = ['recurrent-array', 'conv-array']
        cache = [recurrent_and_conv, ['attention-key', 'attention-value']]
        state = {'model': model, 'consumed': False, 'tokens': tokens, 'messages': messages, 'cache': cache}
        suffix, reused = append_generation_suffix(tokenizer, [{'role': 'user', 'content': 'New copy'}], state, model)
        self.assertIs(reused, cache)
        self.assertIs(reused[0], recurrent_and_conv)
        full = tokenizer.apply_chat_template(messages + [{'role': 'user', 'content': 'New copy'}], tokenize=True, add_generation_prompt=True)
        self.assertEqual(tokens + suffix, full)
        with self.assertRaises(ValueError):
            append_generation_suffix(tokenizer, [{'role': 'user', 'content': 'Again'}], state, model)
        state['consumed'] = False
        state['tokens'] = [123] + tokens[1:]
        with self.assertRaisesRegex(ValueError, 'exact cached prefix'):
            append_generation_suffix(tokenizer, [{'role': 'user', 'content': 'Again'}], state, model)
        self.assertFalse(state['consumed'])

    def test_native_stream_uses_exact_same_wrapper_cache_and_stops(self):
        tokenizer, model = TokenizerStub(), object()
        messages = [{'role': 'system', 'content': 'Facts'}, {'role': 'user', 'content': 'Public facts'}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=True)
        cache, seen = [['delta-state', 'conv-state'], ['key', 'value']], {}
        state = {'model': model, 'consumed': False, 'tokens': prefix, 'messages': messages, 'cache': cache}
        def stream(received_model, received_tokenizer, **kwargs):
            seen.update(kwargs)
            self.assertIs(received_model, model)
            self.assertIs(received_tokenizer, tokenizer)
            yield types.SimpleNamespace(text='Title: New pack', finish_reason=None, prompt_tokens=20,
                                        prompt_tps=100, generation_tokens=1)
            yield types.SimpleNamespace(text='\nCopy: A compact pack.', finish_reason='stop', prompt_tokens=20,
                                        prompt_tps=100, generation_tokens=2)
        mlx = types.ModuleType('mlx')
        mx = types.ModuleType('mlx.core'); mx.synchronize = lambda: None
        lm = types.ModuleType('mlx_lm'); lm.stream_generate = stream
        cache_module = types.ModuleType('mlx_lm.models.cache')
        cache_module.make_prompt_cache = lambda m: self.fail('shared path must not create a new cache')
        sample = types.ModuleType('mlx_lm.sample_utils'); sample.make_sampler = lambda temp: ('greedy', temp)
        with patch.dict('sys.modules', {'mlx': mlx, 'mlx.core': mx, 'mlx_lm': lm,
                                      'mlx_lm.models.cache': cache_module, 'mlx_lm.sample_utils': sample}):
            result = native_decode(model, tokenizer, [{'role': 'user', 'content': 'New copy'}], 64, state)
        self.assertIs(seen['prompt_cache'], cache)
        self.assertEqual(seen['max_tokens'], 64)
        self.assertNotIn('draft_model', seen)
        self.assertEqual(result['finishReason'], 'stop')
        self.assertEqual(result['sharedPrefixTokens'], len(prefix))
        self.assertIn('prefillAndFirstTokenMs', result['timing'])

    def test_trained_checkpoint_evidence_and_revision_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            heads = root / 'adaptation'; heads.mkdir()
            (heads / 'metadata.json').write_text(json.dumps({'sha256': 'test-checksum', 'backboneFrozen': True,
                                                           'representation': 'shared-prefix-v1'}))
            report = {'adaptation': {'epochs': 2, 'firstGradientL2': .2, 'meanAbsolutePredictionChange': .1}}
            (root / 'training.json').write_text(json.dumps(report))
            with patch('experiments.ranker.pretrained.cli.load_heads', return_value=HeadsStub()) as load:
                _, verified = load_trained_heads(heads, 4, 'pinned-revision')
                self.assertEqual(load.call_args.kwargs['expected_revision'], 'pinned-revision')
                self.assertTrue(verified['verifiedHeadUpdate'])
                self.assertEqual(verified['representation'], 'shared-prefix-v1')
            report['adaptation']['firstGradientL2'] = 0
            (root / 'training.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'firstGradientL2'):
                load_trained_heads(heads, 4, 'pinned-revision')


if __name__ == '__main__':
    unittest.main()
