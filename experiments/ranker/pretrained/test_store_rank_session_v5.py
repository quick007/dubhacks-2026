"""CPU serving-adapter tests using local checkpoints and an injected backbone.

No model load, native forward, service or quality-promotion action occurs.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch

from . import store_rank_session_v5 as serving
from .store_function_adaptation import validate_request

BASE = Path('data/ranker/pretrained/actual-store-function-v3-attempt-1')
EXPORT = Path('data/ranker/pretrained/actual-store-function-v5-attempt-1/serving-head.json')


class FakeBackbone:
    def __init__(self, provenance):
        self.revision = provenance['modelRevision']; self.hidden_size = 4096; self.calls = 0
        self.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [10, 11, 12])
        self.model = SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=object()))])
    def encode(self, texts):
        self.calls += 1
        values = torch.arange(1, 4097, dtype=torch.float32).reshape(1, -1); values /= values.norm()
        return values, {'batchRows': 1, 'backboneForwardCalls': 1, 'truncatedRows': 0,
            'paddedTokensPerRow': 3, 'fixtureOnlyNoNativeForward': True}


class V5RankServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.export = json.loads(EXPORT.read_text())
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); self.path = self.root / 'serving-head.json'
        self.value = deepcopy(self.export)
        shutil.copyfile(EXPORT.parent / 'learned.pt', self.root / 'learned.pt')
        self.save()
        self.bb = FakeBackbone(self.value['featureProvenance'])
    def save(self):
        self.path.write_text(json.dumps(self.value)); self.checksum = serving.sha(self.path)
    def session(self):
        return serving.V5RankSession(self.bb, BASE, head_export_path=self.path,
            head_export_sha256=self.checksum, source_validator=lambda receipt: True)

    def test_pure_export_only_no_new_run_reports_labels_teachers_or_gpu_load(self):
        original = Path.read_text
        def guarded(path, *args, **kwargs):
            if path.parent == self.root and path.name != 'serving-head.json':
                raise AssertionError('Only public serving export may be parsed in new run')
            if any(word in path.name.casefold() for word in ('evaluation', 'labels', 'teacher', 'replay')):
                raise AssertionError('Evaluation/training/teacher file read during serving')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', guarded):
            session = self.session()
        self.assertEqual(self.bb.calls, 0)
        self.assertEqual(session.report['nativeForwards'], 64)  # genuine original base, unchanged
        self.assertEqual(session.rank_head_provenance['newRunNativeForwards'], 96)
        self.assertEqual(session.rank_head_provenance['totalFeatureVectors'], 144)
        self.assertEqual(session.head_sha, self.value['head']['sha256'])
        self.assertFalse(session.rank_head_provenance['teacherOrEvaluationFilesReadAtServing'])
        self.assertTrue(session.integrity_receipt()['v5HeadCheckpointUnchanged'])

    def test_native96_total144_927_proof_and_encoding_contract_fail_closed(self):
        for key, wrong in [('newNativeForwards', 64), ('totalFeatureVectors', 96),
                ('frozen927HashesUnchanged', False), ('requestEncodingUnchanged', False),
                ('teacherLookupAtServing', True)]:
            self.value = deepcopy(self.export); self.value[key] = wrong; self.save()
            with self.assertRaisesRegex(ValueError, 'proof'):
                self.session()
        self.assertEqual(self.bb.calls, 0)

    def test_export_and_head_checksums_and_base_binding_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'export checksum'):
            serving.read_export(self.path, '0' * 64)
        self.value['sourceBaseRun'] = str(self.root / 'wrong-base'); self.save()
        with self.assertRaisesRegex(ValueError, 'exact original'):
            self.session()
        self.value = deepcopy(self.export); self.save()
        (self.root / 'learned.pt').write_bytes(b'bad model')
        with self.assertRaisesRegex(ValueError, 'weights checksum'):
            self.session()

    def test_public_catalog_tensor_receipt_pin_corruption_rejected(self):
        first = next(iter(self.value['originalCatalogFeatureFilesSha256']))
        self.value['originalCatalogFeatureFilesSha256'][first] = '0' * 64; self.save()
        with self.assertRaisesRegex(ValueError, 'tensor/receipt changed'):
            self.session()
        self.value = deepcopy(self.export); self.value['originalCatalogFeatureFilesSha256'].pop(first); self.save()
        with self.assertRaisesRegex(ValueError, 'Exactly32'):
            self.session()

    def test_public_product_text_mismatch_rejected(self):
        from .store_catalog_v1 import load_snapshot
        snapshot = deepcopy(load_snapshot()); snapshot['inputs']['products'][2]['name'] = 'Changed notebook'
        with patch('experiments.ranker.pretrained.store_rank_session.load_snapshot', return_value=snapshot):
            with self.assertRaisesRegex(ValueError, 'public catalog facts differ'):
                self.session()

    def test_query_provenance_hash_and_contiguous_timing_are_updated(self):
        session = self.session()
        request = {'query': 'Choose a product for my stated use.', 'preferences': ['paper for writing notes'],
            'itemBudget': None, 'hardRequirements': []}
        result = session.rank_request(request)
        self.assertEqual(self.bb.calls, 1); self.assertEqual(len(result['scores16']), 16)
        self.assertEqual(result['rankCheckpointSha256'], self.value['head']['sha256'])
        self.assertEqual(result['featureBaseProvenance']['originalRunNativeForwards'], 64)
        self.assertEqual(result['rankHeadProvenance']['newRunNativeForwards'], 96)
        payload = {k: v for k, v in result.items() if k not in ('payloadSha256', 'payloadBytes', 'freshRankToSerializedActionsMs')}
        encoded = serving.canonical(payload).encode()
        self.assertEqual(result['payloadBytes'], len(encoded))
        self.assertEqual(result['payloadSha256'], hashlib.sha256(encoded).hexdigest())
        self.assertGreaterEqual(result['freshRankToSerializedActionsMs'], result['baseSessionRankToSerializedActionsMs'])

    def test_runtime_head_change_rejected_before_encode(self):
        session = self.session(); (self.root / 'learned.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            session.rank_request({'query': 'browse', 'preferences': [], 'itemBudget': None, 'hardRequirements': []})
        self.assertEqual(self.bb.calls, 0)

    def test_factory_signature_preserves_original_base_run_and_validator(self):
        factory = serving.make_v5_rank_session_factory(self.path, self.checksum)
        calls = []
        session = factory(self.bb, BASE, source_validator=lambda receipt: calls.append(True))
        self.assertTrue(calls); self.assertEqual(session.run, BASE)
        self.assertEqual(session.startup_provenance['rankHead']['newRunNativeForwards'], 96)
        self.assertEqual(self.bb.calls, 0)


if __name__ == '__main__':
    unittest.main()
