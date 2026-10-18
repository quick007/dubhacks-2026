from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

import numpy as np
import torch

from . import store_persona_adapter_contract_v2 as contract
from .store_persona_adapter_runtime_v2 import blended_scores


def public_row(offset=0):
    ids = list(range(740)); ids[722] += offset
    return {'prompt': 'actual complete public prompt', 'tokenIds': ids,
        'tokenIdsSha256': contract.token_hash(ids), 'routeTokenIds': list(range(16)), 'catalogPrefixTokenIds': ids[:721]}


class AdapterContractTests(unittest.TestCase):
    def test_common_bounds_preserve_complete_values_and_reject_excess(self):
        from . import store_qwen_lexical_choice_inputs_v2 as formatter
        catalog = formatter.load_public_catalog()
        legacy = [{'productId': p['id'], 'name': p['name'], 'category': p['category'], 'itemPrice': p['price'],
            'variants': [{'variantId': v['id'], 'colorName': v['colorName']} for v in p['variants']]} for p in catalog]
        value = {'query': 'Browse the store', 'persona': {'background': ['observed fact'] * 6,
            'colorPreferences': ['blue'] * 4}, 'publicHistory': [
            {'eventId': f'observed-{i}', 'eventType': 'view', 'itemName': 'external public item',
             'category': 'public category', 'occurredAt': f'2026-10-{i + 1:02}T12:00:00Z',
             'source': 'demo-seed', 'external': True} for i in range(12)]}
        original = deepcopy(value)
        _, _, context = contract.common_request(value, catalog, legacy)
        self.assertEqual(value, original); self.assertEqual(len(context['publicHistory']), 12)
        for mutate in [lambda v: v['persona']['background'].append('extra'),
                       lambda v: v['persona']['colorPreferences'].append('extra'),
                       lambda v: v['publicHistory'].append(v['publicHistory'][-1]),
                       lambda v: v.update(query='x' * 501)]:
            changed = deepcopy(value); mutate(changed)
            with self.assertRaises(ValueError): contract.common_request(changed, catalog, legacy)

    def test_actual_last16_literal_prefix_and_no_targets(self):
        row = public_row(); result = contract.suffix_contract(row)
        self.assertEqual(result['suffixTokenIds'], row['tokenIds'][-16:])
        self.assertEqual(result['prefixTokenIds'], row['tokenIds'][:-16])
        self.assertEqual(result['catalogPrefixTokenIds'], row['tokenIds'][:721])
        for mutate in [lambda v: v.update(targetWeights=[1 / 16] * 16),
                       lambda v: v.update(caseId='hidden-person'),
                       lambda v: v.update(catalogPrefixTokenIds=v['tokenIds'][:720]),
                       lambda v: v['tokenIds'].append(42),
                       lambda v: v.update(routeTokenIds=[1] * 16)]:
            changed = deepcopy(row); mutate(changed)
            with self.assertRaises(ValueError): contract.suffix_contract(changed)

    def test_projection_identity_restores_copy_after_error_cancel_and_foreign_change(self):
        base = NS(); copy = NS(linear=base); rank = NS(linear=base)
        mlp = NS(down_proj=copy); model = NS(layers=[NS()] * 31 + [NS(mlp=mlp)])
        syncs = []
        owner = contract.ProjectionOwnership(model, copy, rank, lambda: syncs.append(1))
        for mode, expected in [('rank', rank), ('base', base), ('copy', copy)]:
            with owner.select(mode): self.assertIs(mlp.down_proj, expected)
            self.assertIs(mlp.down_proj, copy)
        class Cancelled(BaseException): pass
        for error in (RuntimeError('original native failure'), Cancelled('request cancelled')):
            with self.assertRaises(type(error)):
                with owner.select('rank'): raise error
            self.assertIs(mlp.down_proj, copy); self.assertTrue(owner.assert_idle())
        with self.assertRaisesRegex(RuntimeError, 'changed unexpectedly'):
            with owner.select('rank'): mlp.down_proj = NS()
        self.assertIs(mlp.down_proj, copy)
        with self.assertRaisesRegex(RuntimeError, 'Serial'):
            with owner.select('rank'):
                with owner.select('base'): pass
        self.assertIs(mlp.down_proj, copy); self.assertGreaterEqual(len(syncs), 12)

    def test_nested_adapter_other_base_and_sync_failure_cannot_leave_rank_active(self):
        base = NS(); copy = NS(linear=base); mlp = NS(down_proj=copy); model = NS(layers=[NS()] * 31 + [NS(mlp=mlp)])
        for rank in (copy, NS(linear=NS()), NS(linear=copy)):
            with self.assertRaises(ValueError): contract.ProjectionOwnership(model, copy, rank, lambda: None)
        calls = 0
        def synchronize():
            nonlocal calls
            calls += 1
            if calls == 2: raise RuntimeError('GPU completion failed')
        owner = contract.ProjectionOwnership(model, copy, NS(linear=base), synchronize)
        with self.assertRaisesRegex(RuntimeError, 'completion'):
            with owner.select('rank'): pass
        self.assertIs(mlp.down_proj, copy); self.assertTrue(owner.assert_idle())

    def test_learned_gate_formula_no_phrase_or_person_rules(self):
        v5 = torch.linspace(.01, .5, 16); lexical = torch.linspace(4, -3, 16)
        persona = lexical.softmax(-1)
        self.assertTrue(torch.equal(blended_scores(v5, lexical, 0.), persona))
        self.assertTrue(torch.equal(blended_scores(v5, lexical, 1.), v5 + .05 * persona))
        expected = .7 * persona + .3 * (v5 + .05 * persona)
        self.assertTrue(torch.equal(blended_scores(v5, lexical, .3), expected))
        for probability in (float('nan'), float('inf'), -1, 1.1, True):
            with self.assertRaises(ValueError): blended_scores(v5, lexical, probability)


class PublicCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.rows = [public_row(i) for i in range(20)]
        self.items = {'x': np.zeros((1, 16, 12288), np.float32),
            'h': np.zeros((1, 16, 4096), np.float32), 'y': np.zeros((1, 16, 4096), np.float32)}
        entries = {}
        for row in self.rows:
            key = row['tokenIdsSha256']; path = self.root / (key + '.npz')
            np.savez_compressed(path, **self.items)
            entries[key] = {'archive': path.name, 'archiveSha256': contract.sha(path),
                'dtypes': {key: 'mlx.core.float32' for key in ('x', 'h', 'y')},
                'routeTokenIds': list(range(16)), 'promptTokenIdsSha256': key, 'promptTokens': 740,
                'prefixPath': 'literal721 plus actual prompt-minus-last16', 'nativeInputSuffixTokens': 16}
        self.index = {'version': 'persona-choice-public-feature-cache.v1', 'entries': entries,
            'teacherOrTargetsIncluded': False, 'key': 'exact prompt token SHA', 'modelRevision': 'revision',
            'formatterSha256': 'f' * 64, 'baseModelFiles': {}, 'scoreSourceSha256': 'e' * 64, 'featureSourceSha256': 'd' * 64}
        self.path = self.root / 'index.json'; self.save()

    def save(self): self.path.write_text(json.dumps(self.index))

    def cache(self):
        return contract.PublicFeatureCache(self.path, contract.sha(self.path), routes=list(range(16)),
            revision='revision', formatter_sha='f' * 64, score_sha='e' * 64, feature_sha='d' * 64)

    def test_exact_public_feature_identity_hit_miss_owned_values_and_dtype_shape(self):
        cache = self.cache(); item = cache.get(self.rows[0])
        self.assertEqual(set(item), {'x', 'h', 'y', 'dtypes', 'routeTokenIds'})
        self.assertTrue(np.array_equal(item['x'], self.items['x']))
        item['x'].fill(99)
        self.assertTrue(np.array_equal(cache.get(self.rows[0])['x'], self.items['x']))
        self.assertIsNone(cache.get(public_row(22)))

    def test_index_unknown_targets_id_fields_and_local_paths_reject(self):
        key = self.rows[0]['tokenIdsSha256']; original = deepcopy(self.index)
        for mutate in [lambda v: v['entries'][key].update(targetWeights=[1 / 16] * 16),
                       lambda v: v['entries'][key].update(profileId='person'),
                       lambda v: v['entries'][key].update(archive='../outside.npz'),
                       lambda v: v.update(teacherOrTargetsIncluded=True),
                       lambda v: v.update(featureSourceSha256='0' * 64)]:
            self.index = deepcopy(original); mutate(self.index); self.save()
            with self.assertRaises(ValueError): self.cache()

    def test_original_archive_checksum_and_no_extra_npz_targets(self):
        cache = self.cache(); key = self.rows[0]['tokenIdsSha256']; path = self.root / (key + '.npz')
        path.write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'identity changed'): cache.get(self.rows[0])
        np.savez_compressed(path, **self.items, targetWeights=np.ones(16))
        self.index['entries'][key]['archiveSha256'] = contract.sha(path); self.save(); cache = self.cache()
        with self.assertRaisesRegex(ValueError, 'only source X/H/Y'): cache.get(self.rows[0])

class ServingManifestTests(unittest.TestCase):
    def setUp(self):
        from . import store_persona_adapter_runtime_v2 as runtime
        self.runtime = runtime
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'adapter.safetensors').write_bytes(b'CPU fixture only; never native loaded')
        (self.root / 'index.json').write_text('{}')
        self.adapter_sha = contract.sha(self.root / 'adapter.safetensors')
        self.cert = {'version': 'persona-choice-inference-parity.v1', 'adapterSha256': self.adapter_sha,
            'cachedScoringMode': 'plain', 'dynamicMode': 'gradient-public-XHY-last16',
            'cachedScoreMatchesTrainingFeaturesExact': True, 'dynamicMatchesCachedFeatureScoresExact': True,
            'zeroAndNonzeroControlPassed': True, 'learnedPostfitPassed': True,
            'helperSourceSha256': {'score': contract.sha(runtime.scoring.__file__), 'capture': contract.sha(runtime.inference.__file__)},
            'zeroNonzeroProofSha256': '1' * 64, 'postfitProofSha256': [c * 64 for c in '2345']}
        self.manifest = {'version': 'persona-choice-adapter-serving.v1',
            'adapter': {'file': 'adapter.safetensors', 'sha256': self.adapter_sha, 'rank': 4, 'scale': 20, 'target': contract.TARGET},
            'publicCache': {'file': 'index.json', 'sha256': contract.sha(self.root / 'index.json')},
            'modelRevision': runtime.REVISION, 'formatterSha256': contract.sha(runtime.formatter.__file__),
            'catalogSha256': runtime.formatter.CATALOG_FILE_SHA256, 'routeTokenIds': list(range(16)),
            'productIds': ['public-' + str(i) for i in range(16)],
            'catalogPrefixTokenIdsSha256': {'canonical': '6' * 64, 'reversed': '7' * 64},
            'actualSuffixTokens': 16, 'base927Unchanged': True, 'trainedUpdates': 3584,
            'postfitNativeControlsExact': True, 'scoreSourceSha256': contract.sha(runtime.scoring.__file__),
            'inferenceTargetFieldsRequired': False}
        self.path = self.root / 'serving.json'
        self.save()

    def save(self):
        parity = self.root / 'inference-parity.json'; parity.write_text(json.dumps(self.cert))
        self.manifest['inferenceParity'] = {'file': parity.name, 'sha256': contract.sha(parity)}
        self.path.write_text(json.dumps(self.manifest))

    def load(self): return self.runtime.load_manifest(self.path, contract.sha(self.path))

    def test_exact_certificate_is_co_located_and_bound_to_learned_adapter_and_helpers(self):
        manifest, cert = self.load()
        self.assertEqual(manifest, self.manifest); self.assertEqual(cert, self.cert)
        original = deepcopy(self.cert)
        for mutate in [lambda c: c.update(dynamicMode='plain-native-last16'),
                       lambda c: c.update(cachedScoringMode='gradient-public-sum16'),
                       lambda c: c.update(learnedPostfitPassed=False),
                       lambda c: c.update(adapterSha256='0' * 64),
                       lambda c: c['helperSourceSha256'].update(capture='0' * 64),
                       lambda c: c.update(postfitProofSha256=['2' * 64] * 4),
                       lambda c: c.update(targetWeights=[1 / 16] * 16)]:
            self.cert = deepcopy(original); mutate(self.cert); self.save()
            with self.assertRaises(ValueError): self.load()

    def test_stale_or_external_certificate_and_wrong_weights_reject(self):
        (self.root / 'inference-parity.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum'): self.load()
        self.save(); self.manifest['inferenceParity']['file'] = '../external.json'
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'co-located'): self.load()
        self.save(); (self.root / 'adapter.safetensors').write_bytes(b'altered')
        with self.assertRaisesRegex(ValueError, 'artifact changed'): self.load()


class EngineLifecycleTests(unittest.TestCase):
    def setUp(self):
        from . import store_persona_adapter_engine_v2 as engine
        from . import store_qwen_lexical_choice_inputs_v2 as formatter
        self.module = engine; self.events = []; self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'candidate'
        self.public = formatter.load_public_catalog()
        self.products = [{'productId': p['id'], 'name': p['name'], 'category': p['category'], 'itemPrice': p['price'],
            'variants': [{'variantId': v['id'], 'colorName': v['colorName']} for v in p['variants']]} for p in self.public]
        events = self.events; products = self.products; public = self.public
        class Base:
            def __init__(self, **kwargs):
                events.append('base-start'); self.startup = {}; self.bb = NS(
                    check_memory=lambda: None, runtime_memory=lambda: {'processBudgetBytes': 10, 'mlxActiveBytes': 0, 'mlxPeakBytes': 0})
                self.adapter = object(); self.session = NS(head_sha='v5', products=products,
                    snapshot={'receipt': {}, 'externalMapping': {'products': []}})
                self.copy_sha = 'copy'; self.copy_metadata = {}; self.binding = NS(assert_current_sources=lambda receipt: True)
            def _assert_candidate(self): events.append('base-check')
            def close(self): events.append('base-close'); return {'frozen927HashesUnchanged': True}
        class Runtime:
            def __init__(self, *args, **kwargs):
                events.append('runtime-init'); self.manifest = {'adapter': {'sha256': 'adapter'}}
                self.color = NS(pins={'color.pt': 'color'}); self.public_catalog = public; self.catalog = {'products': products}
                self.ownership = NS(assert_idle=lambda: True)
            def attach(self, *args): events.append('attach')
            def assert_current(self, *, full=False): events.append('full-check' if full else 'runtime-check')
            def rank(self, value, session, adapter, **kwargs):
                events.append('rank'); request, policy, _ = contract.common_request(value, public, products)
                return request, policy, [], [], [], {'hiddenProfileInputs': False}
        self.Base = Base; self.Runtime = Runtime
        self.intent = lambda *args, **kwargs: NS(provenance=lambda: {'pinned': True})
        self.cleanup = lambda engine, error: events.append(('cleanup', type(error).__name__))

    def construct(self, runtime=None, base=None, **changes):
        klass = self.module.engine_class(base_cls=base or self.Base, runtime_factory=runtime or self.Runtime,
            intent_factory=self.intent, cleanup=self.cleanup)
        args = dict(output=self.output, serving_manifest='serving.json', serving_sha='s', catalog_feature_root='catalog',
            color_head='color.pt', color_cache='cache.pt', query_intent_head='intent.pt', query_intent_sha='g')
        args.update(changes); return klass(**args)

    def test_pre_native_and_post_base_attach_failure_cleanup_and_no_duplicate_base_cleanup(self):
        class Before(self.Runtime):
            def __init__(self, *args, **kwargs): raise ValueError('invalid CPU manifest')
        with self.assertRaisesRegex(ValueError, 'CPU manifest'): self.construct(runtime=Before)
        self.assertNotIn('base-start', self.events); self.assertEqual(self.events, [('cleanup', 'ValueError')])
        self.events.clear()
        class After(self.Runtime):
            def attach(self, *args): raise RuntimeError('rank attach failed')
        with self.assertRaisesRegex(RuntimeError, 'attach failed'): self.construct(runtime=After)
        self.assertEqual(sum(isinstance(x, tuple) and x[0] == 'cleanup' for x in self.events), 1)
        self.events.clear()
        class SelfCleaningBase(self.Base):
            def __init__(self, **kwargs):
                self_cleanup = 'base-owned-cleanup'; self.events = None
                raise ValueError(self_cleanup)
        with self.assertRaisesRegex(ValueError, 'base-owned-cleanup'): self.construct(base=SelfCleaningBase)
        self.assertFalse(any(isinstance(x, tuple) and x[0] == 'cleanup' for x in self.events))

    def test_shutdown_always_closes_base_when_rank_integrity_fails(self):
        engine = self.construct()
        def fail(*, full=False):
            if full: raise ValueError('rank weights changed')
        engine.adapter_runtime.assert_current = fail
        with self.assertRaisesRegex(ValueError, 'rank weights changed'): engine.close()
        self.assertIn('base-close', self.events)
        self.assertFalse((self.output / 'shutdown.json').exists())

    def test_empty_hard_guard_result_no_generation_and_cancellation_receipts(self):
        import time
        from .store_demo_engine import RequestCancelled
        engine = self.construct(); output = []
        engine.personalize({'query': 'Browse the store'}, 'empty', time.perf_counter(), output.append, lambda: False)
        self.assertEqual([event['type'] for event in output], ['rank', 'complete'])
        self.assertEqual(output[-1]['copySource'], {'kind': 'no-generation', 'generatedEarlier': False,
            'freshPerRequestGeneration': False, 'noEligibleProducts': True})
        self.assertEqual(output[-1]['requiredFields'], 0); self.assertEqual(output[-1]['sections'], [])
        receipt = json.loads((self.output / 'requests/empty/receipt.json').read_text())
        self.assertEqual(receipt['status'], 'completed'); self.assertNotIn('parallelGeneration', receipt)
        with self.assertRaises(RequestCancelled):
            engine.personalize({'query': 'Browse the store'}, 'cancelled', time.perf_counter(), output.append, lambda: True)
        receipt = json.loads((self.output / 'requests/cancelled/receipt.json').read_text())
        self.assertEqual(receipt['status'], 'cancelled'); self.assertEqual(receipt['events'], [])

class NativeAttachmentContractTests(unittest.TestCase):
    """Fake native arrays exercise identity/loading; no MLX/model/device import."""
    def setUp(self):
        from .store_persona_adapter_runtime_v2 import AdapterPersonaRuntime, REVISION
        self.factory_calls = []; self.roster = {'lora_a': np.ones((12288, 4), np.float32),
            'lora_b': np.ones((4, 4096), np.float32)}
        self.base = NS(); self.copy = NS(linear=self.base, lora_a=np.zeros((12288, 4), np.float32),
            lora_b=np.zeros((4, 4096), np.float32))
        self.copy_before = {k: getattr(self.copy, k).tobytes() for k in ('lora_a', 'lora_b')}
        self.mlp = NS(down_proj=self.copy)
        class Rank:
            def __init__(inner, pristine): inner.linear = pristine; inner.keys = None
            def load_weights(inner, weights, strict):
                for k, v in weights: setattr(inner, k, v)
            def freeze(inner): pass
            def eval(inner): pass
            def unfreeze(inner, *, keys): inner.keys = keys
            def parameters(inner): return {'lora_a': inner.lora_a, 'lora_b': inner.lora_b}
        def factory(pristine, **kwargs):
            self.factory_calls.append((pristine, kwargs)); return Rank(pristine)
        mx = NS(eval=lambda *args: None, synchronize=lambda: None, float32=np.dtype('float32'),
            load=lambda path: self.roster, all=np.all, isfinite=np.isfinite)
        self.bb = NS(mx=mx, revision=REVISION, check_memory=lambda: None,
            model=NS(layers=[NS()] * 31 + [NS(mlp=self.mlp)]))
        self.runtime = AdapterPersonaRuntime.__new__(AdapterPersonaRuntime)
        self.runtime.bb = None; self.runtime.adapter_factory = factory; self.runtime.adapter_path = Path('cpu-only-fixture')
        self.runtime.assert_current = lambda **kwargs: True

    def test_separate_rank_ab_loads_from_pristine_copy_base_and_restores_idle_copy(self):
        self.runtime.attach(self.bb, self.copy)
        self.assertEqual(self.factory_calls[0][1], {'r': 4, 'scale': 20, 'dropout': 0})
        self.assertIs(self.factory_calls[0][0], self.base)
        self.assertIs(self.runtime.rank_adapter.linear, self.copy.linear)
        self.assertIsNot(self.runtime.rank_adapter, self.copy)
        self.assertEqual(self.runtime.rank_adapter.keys, ['lora_a', 'lora_b'])
        self.assertIs(self.mlp.down_proj, self.copy)
        for key in self.copy_before: self.assertEqual(getattr(self.copy, key).tobytes(), self.copy_before[key])
        with self.runtime.ownership.select('rank'): self.assertIs(self.mlp.down_proj, self.runtime.rank_adapter)
        self.assertIs(self.mlp.down_proj, self.copy)
        with self.assertRaisesRegex(ValueError, 'Attach once'): self.runtime.attach(self.bb, self.copy)

    def test_checkpoint_base_copy_extra_weights_or_wrong_shape_reject_before_switch(self):
        self.roster['linear.weight'] = np.zeros((1,), np.float32)
        with self.assertRaisesRegex(ValueError, 'only A/B'): self.runtime.attach(self.bb, self.copy)
        self.assertIs(self.mlp.down_proj, self.copy)
        del self.roster['linear.weight']; self.roster['lora_b'] = np.zeros((4096, 4), np.float32)
        with self.assertRaisesRegex(ValueError, 'Exact FP32'): self.runtime.attach(self.bb, self.copy)
        self.assertIs(self.mlp.down_proj, self.copy)

class ColdPrefixAndCleanupTests(unittest.TestCase):
    def test_original_failure_or_cancel_survives_sync_failure_with_copy_restored(self):
        from .store_demo_engine import RequestCancelled
        for primary in (RuntimeError('original inference failed'), RequestCancelled('original cancellation')):
            base=NS();copy=NS(linear=base);rank=NS(linear=base);mlp=NS(down_proj=copy)
            calls=[]
            def sync():
                calls.append(1)
                if len(calls)==2:raise OSError('secondary synchronization failed')
            owner=contract.ProjectionOwnership(NS(layers=[NS()]*31+[NS(mlp=mlp)]),copy,rank,sync)
            with self.assertRaises(type(primary)) as caught:
                with owner.select('rank'):raise primary
            self.assertIs(caught.exception,primary);self.assertIs(mlp.down_proj,copy);self.assertTrue(owner.assert_idle())
            self.assertTrue(any('secondary synchronization failed' in note for note in primary.__notes__))

    def test_first_cold_prefix_counts_three_prefills_then_warm_reuses721(self):
        from contextlib import contextmanager
        from unittest.mock import patch
        import sys
        from . import store_persona_adapter_runtime_v2 as rt
        from . import parallel_decode, native_batch_diagnosis
        runtime=rt.AdapterPersonaRuntime.__new__(rt.AdapterPersonaRuntime)
        row=public_row(); runtime.public_cache=NS(get=lambda row:None);runtime.catalog_state=None
        runtime.manifest={'adapter':{'sha256':'a'*64}};runtime.inference_policy={'cachedScoringMode':'plain','dynamicMode':'gradient-public-XHY-last16'}
        copy=object();rank=object();mlp=NS(down_proj=copy)
        @contextmanager
        def select(mode):
            mlp.down_proj=rank if mode=='rank' else None
            try:yield
            finally:mlp.down_proj=copy
        runtime.ownership=NS(select=select);runtime.rank_adapter=rank
        forward_tokens=[]
        def native(tokens,*,cache):
            self.assertIsNot(mlp.down_proj,rank)
            forward_tokens.append(len(tokens[0]));cache[0].state+=len(tokens[0]);return np.zeros((1,1,1))
        model=NS(model=native)
        runtime.bb=NS(model=model,tokenizer=NS(encode=lambda *args,**kwargs:row['tokenIds']),check_memory=lambda:None,
            mx=NS(array=np.array,eval=lambda *args:None,synchronize=lambda:None,all=np.all,isfinite=np.isfinite))
        def capture(bb,adapter,state,prompt,routes):
            self.assertIs(mlp.down_proj,rank);self.assertEqual(state['tokens'],prompt[:-16])
            return {'pure':'CPU fixture'}, {'sourceCacheUnchanged':True,'adapterUnchanged':True}
        modules={'mlx_lm':NS(),'mlx_lm.models':NS(),'mlx_lm.models.cache':NS(make_prompt_cache=lambda model:[NS(state=0)])}
        with patch.dict(sys.modules,modules),patch.object(native_batch_diagnosis,'state_hash',side_effect=lambda cache:str(cache[0].state)),\
             patch.object(rt.inference,'capture',side_effect=capture),patch.object(rt.scoring,'score',return_value=np.arange(16,dtype=np.float32)):
            # The clone API takes (backbone,state); copy only actual state.
            with patch.object(parallel_decode,'clone_prefix_state',side_effect=lambda bb,state:deepcopy(state)):
                _,cold=runtime.lexical_scores(row);_,warm=runtime.lexical_scores(row)
        self.assertEqual(forward_tokens,[256,256,209,3,3])
        self.assertFalse(cold['catalogPrefixCacheHit']);self.assertTrue(cold['catalogPrefixCreatedThisRequest'])
        self.assertEqual((cold['catalogPrefixTokensReused'],cold['catalogPrefixTokensPrefilled'],cold['catalogPrefixPrefillForwards']),(0,721,3))
        self.assertEqual(cold['nativeBodyForwards'],5)
        self.assertTrue(warm['catalogPrefixCacheHit']);self.assertFalse(warm['catalogPrefixCreatedThisRequest'])
        self.assertEqual((warm['catalogPrefixTokensReused'],warm['catalogPrefixTokensPrefilled'],warm['catalogPrefixPrefillForwards']),(721,0,0))
        self.assertEqual(warm['nativeBodyForwards'],2);self.assertIs(mlp.down_proj,copy)


if __name__ == '__main__': unittest.main()
