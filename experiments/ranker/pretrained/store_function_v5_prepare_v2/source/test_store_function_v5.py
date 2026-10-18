"""CPU-only data/fit/controls tests; no native inference or service changes."""
from collections import Counter
from copy import deepcopy
from contextlib import nullcontext
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from . import store_function_data_v5 as data
from . import store_function_adaptation_v5 as run
from .store_catalog_v1 import load_snapshot
from .store_demo_image_binding_v1 import ImageDeliveryCatalogBinding
from .store_demo_engine import checked_request, ranking_input
from .store_function_adaptation import request_text


class FunctionV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = data.build(); cls.snapshot = load_snapshot()

    def test_balanced_all16_function_training_and_distinct_fresh32(self):
        p = self.packet
        self.assertEqual(len(p['publicCases']), 128)
        self.assertEqual(Counter(r['targetProductIndex'] for r in p['trainLabels']), Counter({i: 6 for i in range(16)}))
        self.assertEqual(Counter(r['targetProductIndex'] for r in p['evaluationLabels']), Counter({i: 2 for i in range(16)}))
        texts = [request_text(c['request']) for c in p['publicCases']]
        self.assertEqual(len(set(texts)), 128)
        self.assertTrue(all(c['request']['query'] == 'Choose a product for my stated use.' for c in p['publicCases'] if c['split'] == 'train'))
        self.assertNotIn('hidden', json.dumps([c['request'] for c in p['publicCases']]))

    def test_new_training_exact_live_canonicalization_and_public_support(self):
        p = self.packet
        for i, phrases in enumerate(data.PHRASES):
            for j, phrase in enumerate(phrases):
                case = next(c for c in p['publicCases'] if c['caseId'] == f'v5-train-{i:02d}-{j}')
                request, _ = checked_request({'query': phrase})
                self.assertEqual(case['request'], ranking_input(request))
                self.assertEqual(set(case['request']), {'query', 'preferences', 'itemBudget', 'hardRequirements'})
                label = next(r for r in p['trainLabels'] if r['caseId'] == case['caseId'])
                self.assertEqual(label['targetProductIndex'], i)
                self.assertEqual(label['publicSupport'], {k: p['products'][i][k] for k in ('name', 'category', 'searchTerms')})

    def test_frozen_dataset_preserves_and_read_public_never_parses_eval_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'data'; receipt = data.freeze(path)
            self.assertEqual(receipt['counts']['newNativeForwards'], 96)
            with self.assertRaises(FileExistsError):
                data.freeze(path)
            original = Path.read_text
            def guarded(file, *args, **kwargs):
                if file.name == 'evaluation-labels.json':
                    raise AssertionError('Evaluation labels parsed by public read')
                return original(file, *args, **kwargs)
            with patch.object(Path, 'read_text', guarded):
                _, _, cases = run.read_public(path)
            self.assertEqual(len(cases), 128)
            (path / 'catalog.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'artifact|file changed'):
                run.read_public(path)

    def test_all16_deterministic_guard_controls(self):
        checks = run.guard_checks(data.guard_probes(self.packet['products']), self.packet['products'], self.snapshot)
        self.assertEqual(len(checks), 16)
        self.assertEqual([r['caseId'] for r in checks if not r['passed']], [])

    def test_train_fit_no_eval_arguments_fixed80_and_safe_checkpoint_reload(self):
        generator = torch.Generator().manual_seed(55)
        # Meaningful engineered class-separable features only; no native claim.
        queries = torch.zeros((96, 4096)); catalog = torch.zeros((16, 4096))
        targets = torch.arange(16).repeat_interleave(6)
        queries[torch.arange(96), targets] = 1; catalog[torch.arange(16), torch.arange(16)] = 1
        queries += torch.randn(queries.shape, generator=generator) * .00001
        learned, random, metadata, losses = run.fit_head(queries, catalog, targets, [str(i) for i in range(96)], 'fixture')
        self.assertEqual(len(losses), 80); self.assertEqual(metadata['config']['steps'], 80)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'head.pt'; checksum = run.base.save_head(path, learned, metadata)
            restored = run.base.load_head(path, checksum, metadata)
            self.assertTrue(all(torch.equal(v, restored.state_dict()[k]) for k, v in learned.state_dict().items()))
        cases = [{'caseId': str(i)} for i in range(32)]
        values = catalog.repeat(2, 1); labels = [{'caseId': str(i), 'targetProductIndex': i % 16} for i in range(32)]
        predicted = run.predictions({'learned': learned, 'random': random, 'frozen-v3': restored}, values, catalog, cases, self.packet['products'])
        scored = run.score_predictions(predicted, labels)
        self.assertEqual(scored['learned']['topOneCorrect'], 32)
        self.assertLess(scored['permuted-query']['topOneCorrect'], 32)
        self.assertEqual(scored['prior']['topOneCorrect'], 2)
        bad = targets.clone(); bad[0] = 1
        with self.assertRaisesRegex(ValueError, 'Six TRAIN targets'):
            run.fit_head(queries, catalog, bad, [], 'fixture')

    def test_native_requires_execute_and_refuses_listening_service_without_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary); packet = {'fixture': True}; packet_path = path / 'preflight.json'
            packet_path.write_text(json.dumps(packet)); out = path / 'out'
            with patch.object(run, 'prepare', return_value=packet):
                self.assertFalse(run.run(out, path, path, path, packet_path)['nativeExecuted'])
                with patch('experiments.ranker.pretrained.store_ar_comparison_run_v1.demo_listening', return_value=True):
                    with self.assertRaisesRegex(RuntimeError, 'listening'):
                        run.run(out, path, path, path, packet_path, execute=True, catalog_binding_factory=ImageDeliveryCatalogBinding)
            self.assertFalse(out.exists())

    def test_dataset_pin_change_rejected_before_eval_labels_parse(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'data'; data.freeze(directory)
            manifest = data.HERE / 'store_demo_integration_v8/manifest.json'
            packet = run.prepare(directory, data.PRIOR, manifest, catalog_binding_factory=ImageDeliveryCatalogBinding)
            (directory / 'evaluation-labels.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'dataset file changed'):
                run.assert_packet_pins(packet, directory, manifest)

    def test_injected_native_failure_preserves_fixed_eval_roster_and_shutdown(self):
        # Fake encoder/lock/hash/model verifier only; never acquire the real GPU.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); directory = root / 'data'; data.freeze(directory)
            manifest = data.HERE / 'store_demo_integration_v8/manifest.json'
            packet = run.prepare(directory, data.PRIOR, manifest, catalog_binding_factory=ImageDeliveryCatalogBinding); packet_path = root / 'packet.json'
            packet_path.write_text(json.dumps(packet)); closed = []
            class FakeEncoder:
                def __init__(inner, *args):
                    inner.calls = 0; inner.backbone = SimpleNamespace(model=SimpleNamespace(language_model=object()))
                def encode(inner, text, expected):
                    inner.calls += 1
                    if inner.calls == 2:
                        raise RuntimeError('injected CPU fake native failure')
                    return torch.ones((1, 4096)) / 64, {'fixtureOnly': True}
                def memory(inner):
                    return {'fixtureOnly': True}
                def close(inner):
                    closed.append(True)
            with patch('experiments.ranker.pretrained.store_ar_comparison_run_v1.demo_listening', return_value=False), \
                    patch('experiments.ranker.pretrained.catalog_fusion.generalization_v2.native_owner_lock', return_value=nullcontext()), \
                    patch('experiments.ranker.pretrained.catalog_fusion.runtime.NativeEncoder', FakeEncoder), \
                    patch('experiments.ranker.pretrained.catalog_fusion.runtime.verify_model', return_value=packet['featureProvenance']), \
                    patch('experiments.ranker.pretrained.semantic_copy_training.parameter_hashes', return_value={str(i): 'fixture' for i in range(927)}):
                result = run.run(root / 'run', directory, data.PRIOR, manifest, packet_path, execute=True, catalog_binding_factory=ImageDeliveryCatalogBinding)
            self.assertEqual(result['status'], 'failed-not-promoted'); self.assertEqual(result['nativeForwards'], 1)
            self.assertEqual(len(result['evaluationRoster']), 32)
            self.assertTrue(all(r['status'] == 'unattempted' for r in result['evaluationRoster']))
            self.assertEqual(result['activeRecord']['caseId'], 'v5-train-00-1')
            self.assertTrue(result['frozen927HashesUnchanged']); self.assertEqual(closed, [True])
            self.assertEqual(json.loads((root / 'run/report.json').read_text())['error']['type'], 'RuntimeError')

    def test_original_strict_default_rejects_image_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'data'; data.freeze(directory)
            manifest = data.HERE / 'store_demo_integration_v8/manifest.json'
            with self.assertRaisesRegex(ValueError, 'Exact reviewed demo integration receipt required'):
                run.prepare(directory, data.PRIOR, manifest)

    def test_explicit_binding_factory_is_forwarded_without_default_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); packet_path = root / 'packet.json'; packet = {'fixture': True}
            packet_path.write_text(json.dumps(packet)); factory = object()
            with patch.object(run, 'prepare', return_value=packet) as prepare:
                result = run.run(root / 'out', root, root, root, packet_path, catalog_binding_factory=factory)
                self.assertFalse(result['nativeExecuted'])
                prepare.assert_called_once_with(root, root, root, catalog_binding_factory=factory)

    def test_cpu_preflight_real_reuse_and_no_eval_parse_or_native_import(self):
        manifest = data.HERE / 'store_demo_integration_v8/manifest.json'
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'data'; data.freeze(directory)
            original = Path.read_text
            def guarded(path, *args, **kwargs):
                if path.name == 'evaluation-labels.json':
                    raise AssertionError('eval labels parsed during preflight')
                return original(path, *args, **kwargs)
            with patch.object(Path, 'read_text', guarded):
                packet = run.prepare(directory, data.PRIOR, manifest, catalog_binding_factory=ImageDeliveryCatalogBinding)
            self.assertEqual(len(packet['records']), 144)
            self.assertEqual(sum(r['origin'] == 'verified-v3-reuse' for r in packet['records']), 48)
            self.assertEqual(len(packet['reusedFeatureFilesSha256']), 96)
            self.assertFalse(packet['nativeExecuted']); self.assertFalse(packet['evaluationLabelsParsed'])
            self.assertLessEqual(packet['tokenRange'][1], 512)
            self.assertEqual(packet['catalogBinding']['class'], 'ImageDeliveryCatalogBinding')
            self.assertEqual(packet['preparationVersion'], run.PREPARATION_VERSION)
            self.assertEqual(len(packet['catalogBinding']['helperSourceSha256']), 2)


if __name__ == '__main__':
    unittest.main()
