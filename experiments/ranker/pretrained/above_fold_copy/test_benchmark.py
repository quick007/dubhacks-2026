"""Pure CPU roster/checkpoint gates. No tokenizer/model is loaded by tests."""
import json
from pathlib import Path
import tempfile
import unittest

from .benchmark import CONTROL_FIELDS, control_jobs, completed_adapter
from .data import PARENT, records, sha
from .layout import build_layout


class BenchmarkTests(unittest.TestCase):
    def test_controls_use_actual_distinct_N_rows_including_copy(self):
        c = records(PARENT / 'cases.dev.jsonl')[0]
        _, jobs, _ = build_layout(c['pendingPagePlan'], c['request'], c['catalog'])
        for width in (1, 3, 6):
            chosen = control_jobs(jobs, width)
            self.assertEqual(len(chosen), width)
            self.assertEqual([r['fields'][0] for r in chosen], list(CONTROL_FIELDS[width]))
            self.assertEqual(len({r['rowId'] for r in chosen}), width)
            self.assertTrue(all(r['maxTokens'] == 7 for r in chosen))
        self.assertEqual({r['kind'] for r in control_jobs(jobs, 6)}, {'heading', 'title', 'badge', 'copy'})
        self.assertEqual(sum(r['maxTokens'] for r in jobs), 208)

    def test_completed_final_504_receipt_required_and_hash_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkpoint = root / 'adapter.safetensors'; checkpoint.write_bytes(b'CPU fixture, no tensors')
            path = root / 'results.json'
            fit = {'status': 'started', 'phase': 'fit', 'nativeUpdates': 0, 'optimizerUpdateCalls': 0, 'steps': []}
            path.write_text(json.dumps(fit))
            with self.assertRaisesRegex(ValueError, 'Completed final'): completed_adapter(path)
            fit.update(status='completed', phase='completed', nativeUpdates=504, optimizerUpdateCalls=504,
                steps=[{'status': 'completed', 'updateEvaluated': True}] * 504,
                checkpoint={'exact': True, 'sha256': sha(checkpoint)},
                controls={'frozenLanguageHashesUnchanged': True, 'learnedCacheProof': {'matchedForcedContinuationLogitsExact': True}})
            path.write_text(json.dumps(fit)); receipt = completed_adapter(path)
            self.assertEqual(receipt['nativeUpdates'], 504)
            self.assertFalse(receipt['CTrainingCoverage'])
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint receipt'): completed_adapter(path)


if __name__ == '__main__': unittest.main()
