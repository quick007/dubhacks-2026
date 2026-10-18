"""CPU followup architecture/provenance tests, no Qwen/MLX execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import store_parallel_text_cpu_v2 as v2
from . import store_parallel_text_demo_all_v1 as demo
from . import store_parallel_text_head_v1 as base
from . import store_parallel_text_data_v1 as data
from .store_parallel_text_run_v1 import load_packet


class ParallelFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.packet = load_packet()
        with base.bounded_cpu_threads(1):
            cls.bundle = base.engineered_fixture(cls.packet)
        cls.feature_map = base.validate_bundle(cls.bundle, cls.packet, fixture=True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_normalization_fits_unique_train_only(self):
        before = v2.normalization(self.packet, self.feature_map, enabled=True)
        train = set(v2.train_feature_keys(self.packet))
        changed = dict(self.feature_map)
        for key in changed.keys() - train:
            changed[key] = torch.ones(4096) * 123
        after = v2.normalization(self.packet, changed, enabled=True)
        self.assertEqual(len(train), 34)
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[1], after[1]))
        self.assertEqual(before[2], after[2])
        self.assertFalse(before[2]['evalFeatureKeysUsedForFit'])

    def test_shared_word_classifier_outputs_all_slots_once(self):
        mean, inverse, _ = v2.normalization(self.packet, self.feature_map, enabled=True)
        model = v2.ParallelTextV2(self.packet['vocabulary'], mean, inverse, width=16)
        rows = self.packet['labels']['train'][:4]
        features, conditions = base.serving_tensors(base.inference_rows(rows), self.feature_map)
        with base.bounded_cpu_threads(1):
            logits = model(features, conditions)
        self.assertEqual(logits.shape, (4, 14, 107))
        self.assertEqual(model.words.out_features, 107)
        self.assertEqual(model.position.shape, (14, 16))

    def test_positional_coverage_exposes_v1_label_gap(self):
        result = v2.position_coverage(self.packet)
        self.assertEqual(len(result['zeroPositiveTrainSlotWordPairs']), 20)
        self.assertEqual(len(result['affectedFields']), 22)
        self.assertTrue(any(word == '$68.00' for _, word in result['zeroPositiveTrainSlotWordPairs']))

    def test_literal_metrics_are_separate_from_format_acceptance(self):
        row = next(r for r in self.packet['labels']['eval'] if r['input']['kind'] == 'copy')
        facts = row['input']['source']['facts']
        values = v2.factual_metrics(row, {'text': f"{facts['selectedColor']} at ${facts['itemPrice']:.2f}."})
        self.assertTrue(values['selectedColor']); self.assertTrue(values['exactItemPrice'])
        self.assertIsNone(values['completeIdentity'])
        wrong = v2.factual_metrics(row, {'text': 'Cobalt at $999.00.'})
        self.assertFalse(wrong['exactItemPrice'])

    def test_demo_explicitly_trains_all_48_not_heldout(self):
        rows = demo.demo_labels(self.packet)
        card_rows = [r for r in rows if r['input']['kind'] != 'section']
        self.assertEqual(len(rows), 528)
        self.assertEqual(len(card_rows), 432)
        self.assertEqual(len({r['featureKeys'][0] for r in card_rows}), 48)
        self.assertEqual(demo.CONFIG['heldoutBindings'], 0)
        self.assertEqual({r['input']['style'] for r in card_rows}, set(data.STYLES))
        for row in rows:
            inp = row['input']
            self.assertEqual(data.field_rejections(inp['kind'], row['targetText'], inp.get('source', {}).get('facts')), [])

    def test_demo_load_rejects_other_scope_and_never_calls_teacher(self):
        model = base.ParallelTextHead(self.packet['vocabulary'], width=16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'head.pt'
            payload = {'schemaVersion': demo.VERSION, 'scope': 'wrong scope', 'state_dict': model.state_dict(),
                       'vocabulary': list(model.vocabulary), 'width': model.width}
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                demo.load_head(path)
            payload['scope'] = 'TRAIN-ALL public catalog demo; no heldout evaluation'
            torch.save(payload, path)
            with patch.object(data, 'teacher', side_effect=AssertionError('runtime teacher')):
                loaded, metadata = demo.load_head(path)
            self.assertEqual(loaded.width, 16)
            self.assertEqual(metadata['schemaVersion'], demo.VERSION)


if __name__ == '__main__':
    unittest.main()
