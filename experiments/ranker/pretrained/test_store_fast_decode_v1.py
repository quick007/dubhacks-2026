"""CPU-only root cache identity, invalidation, immutability and boundedness."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from .store_fast_decode_v1 import ImmutableRootCache, digest


class RootReuseTests(unittest.TestCase):
    def fixture(self):
        calls = []; adapter = object(); hashes = ['adapter-A']
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                calls.append(('tokenize', kwargs))
                return [ord(c) for c in messages[0]['content']]
        bb = SimpleNamespace(tokenizer=Tokenizer(), revision='verified-revision',
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=adapter))]))
        def prepare(backbone, common):
            calls.append(('native-root-build', common['content']))
            tokens = backbone.tokenizer.apply_chat_template([common])
            return {'model': bb.model, 'tokens': tokens, 'messages': [deepcopy(common)], 'consumed': False,
                'prefixSha256': digest(tokens), 'cache': [{'source': common['content']}]}, {
                'rootTokens': len(tokens), 'rootPrepareMs': 500., 'freshEmptyCache': True}
        cache = ImmutableRootCache(bb, adapter, model_receipt={'verified': True},
            checkpoint_sha256='completed-adapter-sha', tokenizer_receipt={'tokenizer.json': 'pinned'},
            _prepare=prepare, _state_hash=digest, _adapter_hash=lambda: hashes[0],
            _guard=lambda bb: {'processBudgetBytes': 7 * 2 ** 30})
        return bb, cache, calls, hashes

    def test_same_root_across_shopper_a_b_a_builds_once_and_never_reuses_text(self):
        bb, cache, calls, hashes = self.fixture()
        common = {'role': 'user', 'content': 'neutral copy root'}
        states = []
        for shopper in ('water', 'light', 'water'):
            state, receipt = cache.prepare_root(bb, common)
            states.append(state)
            branch = deepcopy({key: value for key, value in state.items() if key != 'model'})
            branch['consumed'] = True; branch['cache'][0]['source'] += ' freshly generated token'
            cache.assert_immutable()
            self.assertFalse(receipt['generationReused'])
        self.assertIs(states[0], states[2])
        self.assertEqual(sum(call[0] == 'native-root-build' for call in calls), 1)
        self.assertEqual((cache.hits, cache.misses, cache.evictions), (2, 1, 0))
        self.assertFalse(cache.receipt()['generatedTextCached'])

    def test_exact_content_and_token_key_different_root_evicts_bounded_entry(self):
        bb, cache, calls, hashes = self.fixture()
        a = {'role': 'user', 'content': 'A'}; b = {'role': 'user', 'content': 'B'}
        _, first = cache.prepare_root(bb, a)
        _, second = cache.prepare_root(bb, b)
        _, third = cache.prepare_root(bb, a)
        self.assertNotEqual(first['commonRootCacheKeySha256'], second['commonRootCacheKeySha256'])
        self.assertEqual(first['commonRootCacheKeySha256'], third['commonRootCacheKeySha256'])
        self.assertEqual((cache.hits, cache.misses, cache.evictions), (0, 3, 2))
        self.assertEqual(cache.receipt()['entries'], 1)

    def test_native_adapter_bytes_and_active_adapter_identity_both_required(self):
        bb, cache, calls, hashes = self.fixture(); common = {'role': 'user', 'content': 'root'}
        cache.prepare_root(bb, common); hashes[0] = 'adapter-B'
        with self.assertRaisesRegex(ValueError, 'tensors changed'):
            cache.prepare_root(bb, common)
        hashes[0] = 'adapter-A'; bb.model.layers[-1].mlp.down_proj = object()
        with self.assertRaisesRegex(ValueError, 'active copy adapter'):
            cache.prepare_root(bb, common)

    def test_consumed_or_mutated_source_fails_without_repair(self):
        for change in ('cache', 'tokens', 'messages', 'consumed', 'prefixSha256'):
            bb, cache, calls, hashes = self.fixture()
            state, _ = cache.prepare_root(bb, {'role': 'user', 'content': 'root'})
            if change == 'cache':
                state['cache'][0]['source'] = 'corrupted'
            elif change == 'tokens':
                state['tokens'].append(99)
            elif change == 'messages':
                state['messages'][0]['content'] = 'different'
            elif change == 'consumed':
                state['consumed'] = True
            else:
                state['prefixSha256'] = 'different'
            with self.assertRaisesRegex(ValueError, 'changed or was consumed'):
                cache.assert_immutable()

    def test_backbone_tokenizer_revision_and_verified_receipt_invalidation(self):
        for mutation in ('model', 'tokenizer', 'revision', 'identity'):
            bb, cache, calls, hashes = self.fixture()
            if mutation == 'identity':
                cache.identity['adapterCheckpointSha256'] = 'unverified'
            else:
                setattr(bb, mutation, 'changed')
            with self.assertRaises(ValueError):
                cache.prepare_root(bb, {'role': 'user', 'content': 'root'})

    def test_memory_guard_cannot_be_silently_raised(self):
        bb, cache, calls, hashes = self.fixture()
        def fail(bb):
            raise MemoryError('active/peak exceeds7GiB')
        cache._guard = fail
        with self.assertRaises(MemoryError):
            cache.prepare_root(bb, {'role': 'user', 'content': 'root'})

    def test_alternative_factory_refused_and_clear_checks_integrity(self):
        bb, cache, calls, hashes = self.fixture()
        with self.assertRaisesRegex(ValueError, 'Alternate cache factory'):
            cache.prepare_root(bb, {'role': 'user', 'content': 'root'}, cache_factory=object())
        cache.prepare_root(bb, {'role': 'user', 'content': 'root'}); cache.clear()
        self.assertEqual(cache.receipt()['entries'], 0)


if __name__ == '__main__':
    unittest.main()
