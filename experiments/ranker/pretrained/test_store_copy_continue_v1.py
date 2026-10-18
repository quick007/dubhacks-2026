"""CPU-only continuation contracts; fixture receipts are never native evidence."""
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from . import store_copy_continue_v1 as fit
from . import store_copy_fit_v2 as original
from .semantic_copy_feature_fit import digest, dump


def completed_fixture():
    """Minimal synthetic receipts for control-flow tests only."""
    case_ids = ['fixture-eval-' + str(i) for i in range(8)]
    arms = [{'caseId': case_id, 'mode': mode, 'width': 6, 'status': 'completed-attempt',
        'firstThree': {'validation': {'valid': False}}, 'fullSix': {'validation': {'valid': False}},
        'firstThreeReceipt': {'validatedAndSerializedAtMs': 1}, 'fullSixReceipt': {'validatedAndSerializedAtMs': 2}}
        for case_id in case_ids for mode in original.MODES]
    checkpoint = {'sha256': fit.WARM_SHA, 'exact': True, 'parametersBeforeReload': {'fixture': 1},
        'parametersAfterReload': {'fixture': 1}, 'lossBeforeReload': 1., 'lossAfterReload': 1.}
    controls = {key: True for key in ('all320NonzeroBGatesPassed', 'fitStartedFromExactFrozenBaseZeroB',
        'learnedOriginalArchivedNativeParityExact', 'frozenLanguageHashesUnchanged',
        'postEvaluationFrozen927Unchanged', 'sourceFilesUnchanged', 'datasetManifestUnchanged', 'originalPriorUnchanged')}
    controls.update({mode: {'status': 'exact-controls-passed'} for mode in original.MODES})
    extraction = [{'index': i, 'gateExact': True, 'trueZeroBRestored': True} for i in range(320)]
    steps = [{'step': i, 'featureIndex': index, 'status': 'completed', 'updateEvaluated': True}
        for i, index in enumerate(original.shuffled_order())]
    return {'schemaVersion': 'actual-store-copy-fit.v2', 'status': 'completed', 'phase': 'completed',
        'config': original.CONFIG, 'nativeUpdates': 320, 'optimizerUpdateCalls': 320,
        'steps': steps, 'extraction': extraction, 'checkpoint': checkpoint, 'controls': controls,
        'frozenBaseParameterHashes': {str(i): 'fixture' for i in range(927)}, 'arms': arms,
        'summary': original.summary(arms),
        'preflight': {'splits': {'eval': {'cases': [{'caseId': case_id} for case_id in case_ids]}}}}


class ContinuationTests(unittest.TestCase):
    def test_import_never_loads_native_runtime(self):
        subprocess.run([sys.executable, '-c', 'import sys; import experiments.ranker.pretrained.store_copy_continue_v1; '
            'assert not any(k == "mlx" or k.startswith("mlx.") for k in sys.modules)'], check=True, capture_output=True)

    def test_schedule_is_three_fixed_distinct_train_only_permutations(self):
        orders = fit.permutations()
        self.assertEqual(orders, fit.permutations())
        self.assertEqual(len(orders), 3)
        self.assertEqual(len({tuple(order) for order in orders}), 3)
        for order in orders:
            self.assertEqual(sorted(order), list(range(320)))
        self.assertEqual(fit.CONFIG['updates'], 960)
        self.assertEqual(fit.CONFIG['priorUpdates'] + fit.CONFIG['updates'], fit.CONFIG['finalTotalUpdates'])
        self.assertEqual(fit.CONFIG['finalTotalUpdates'], 1280)
        self.assertEqual(fit.CONFIG['learningRate'], original.CONFIG['learningRate'])
        self.assertEqual(fit.CONFIG['memoryGiB'], 7)
        self.assertEqual(fit.CONFIG['residentHostFeatureRows'], 1)
        self.assertIn('fresh zero Adam', fit.CONFIG['initialization'])
        self.assertIn('not exact resume', fit.CONFIG['optimizerContinuity'])

    def test_original_completed_receipts_required_including_native_actual_counts(self):
        baseline = completed_fixture(); fit.validate_completed_original(baseline)
        mutations = [lambda r: r.update(status='started'), lambda r: r.update(phase='24-evaluation-arms'),
            lambda r: r.update(nativeUpdates=319), lambda r: r.update(optimizerUpdateCalls=319),
            lambda r: r['steps'][5].update(updateEvaluated=False), lambda r: r['steps'].pop(),
            lambda r: r['steps'][2].update(featureIndex=999), lambda r: r['extraction'][4].update(gateExact=False),
            lambda r: r['checkpoint'].update(sha256=original.PRIOR_SHA),
            lambda r: r['checkpoint'].update(lossAfterReload=2.),
            lambda r: r['controls'].update(postEvaluationFrozen927Unchanged=False),
            lambda r: r['arms'][0].update(status='running'), lambda r: r['arms'][0].update(width=3),
            lambda r: r['arms'].pop(), lambda r: r['summary']['frozen'].update(completed=7),
            lambda r: r['controls']['new-store'].update(status='failed-strict-control')]
        for mutation in mutations:
            report = copy.deepcopy(baseline); mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                fit.validate_completed_original(report)

    def test_original_inventory_detects_rewrite_new_files_and_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); file = root / 'original.txt'; file.write_text('original bytes')
            receipt = {'path': str(root), 'files': {'original.txt': {'bytes': file.stat().st_size, 'sha256': digest(file)}}}
            fit.assert_inventory_preserved(receipt)
            file.write_text('modified bytes')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                fit.assert_inventory_preserved(receipt)
            file.write_text('original bytes'); (root / 'extra.txt').write_text('new')
            with self.assertRaisesRegex(ValueError, 'inventory changed'):
                fit.assert_inventory_preserved(receipt)
            with self.assertRaisesRegex(ValueError, 'contained regular file'):
                fit.contained_file(root, '../outside')
            (root / 'alias').symlink_to(file)
            with self.assertRaisesRegex(ValueError, 'contained regular file'):
                fit.contained_file(root, 'alias')

    def test_preflight_uses_original_exact_tokenizer_contract_and_has_no_native_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary); baseline = completed_fixture()
            baseline['preflightSha256'] = 'fixture-preflight-sha'
            base = {'datasetManifestSha256': 'fixture-manifest', 'payloadSha256': {},
                'splits': baseline['preflight']['splits'], 'limitations': ['Temporary nonzero-B old claim', 'same catalog']}
            baseline['preflight'] = base; dump(path / 'results.json', baseline)
            with patch.object(fit, 'original_receipt', return_value={'fixture': True}), \
                    patch.object(original, 'preflight', return_value=base), \
                    patch.object(fit, 'model_input_receipt', return_value={'fixture': True}), \
                    patch.object(fit, 'digest', return_value='fixture-digest'):
                result = fit.preflight(object(), path)
                self.assertFalse(result['nativeExecution'])
                self.assertEqual(result['trainingUpdates'], 0)
                self.assertEqual(result['modelCalls'], 0)
                self.assertEqual(result['permutations'], fit.permutations())
                self.assertEqual(result['fixedArms'], 24)
                self.assertEqual(result['evalModes'], list(fit.MODES))
                self.assertIn('development replay', ' '.join(result['limitations']))
                self.assertNotIn('Temporary nonzero-B old claim', result['limitations'])
                changed = {**base, 'datasetManifestSha256': 'changed'}
                with patch.object(original, 'preflight', return_value=changed), self.assertRaisesRegex(ValueError, 'no longer matches'):
                    fit.preflight(object(), path)

    def test_fixed24_replay_roster_has_separate_continued_name_and_strict_latency(self):
        arms = fit.arm_roster(['fixture-' + str(i) for i in range(8)])
        self.assertEqual(len(arms), 24)
        self.assertEqual({a['mode'] for a in arms}, set(fit.MODES))
        self.assertEqual({a['split'] for a in arms}, {'eval-development-replay'})
        self.assertEqual({a['width'] for a in arms}, {6})
        arm = next(a for a in arms if a['mode'] == 'continued-store')
        arm.update(status='completed-attempt', firstThree={'validation': {'valid': True}}, fullSix={'validation': {'valid': False}},
            firstThreeReceipt={'validatedAndSerializedAtMs': 10}, fullSixReceipt={'validatedAndSerializedAtMs': 20})
        values = fit.summary(arms)
        self.assertNotIn('new-store', values)
        self.assertEqual(values['continued-store']['validFirstThreeReadyMs'], [10])
        self.assertEqual(values['continued-store']['validFullSixReadyMs'], [])
        self.assertEqual(values['continued-store']['validThreeCardRateFixedDenominator'], 1 / 8)

    def test_stage_receipts_record_timeout_and_restore_alarm(self):
        previous = signal.getsignal(signal.SIGALRM); report = {}; saves = []
        with self.assertRaisesRegex(TimeoutError, 'fixed 60-second'):
            with fit.stage_receipt(report, lambda: saves.append(copy.deepcopy(report)), 'fixture-native-stage', 60):
                signal.raise_signal(signal.SIGALRM)
        self.assertEqual(saves[0]['stages'][0]['status'], 'running')
        self.assertEqual(saves[-1]['stages'][0]['status'], 'failed')
        self.assertEqual(report['stages'][0]['error']['type'], 'TimeoutError')
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)
        self.assertGreaterEqual(report['stages'][0]['ms'], 0)

    def test_zero_adam_contract_checks_moments_and_does_not_require_zero_learning_rate(self):
        import numpy as np
        state = {'step': np.array(0), 'learning_rate': np.array(.0001),
            'lora_a': {key: np.zeros((12288, 4), dtype=np.float32) for key in ('m', 'v')},
            'lora_b': {key: np.zeros((4, 4096), dtype=np.float32) for key in ('m', 'v')}}
        self.assertTrue(fit.validate_zero_adam_state(state)['allMomentsExactlyZero'])
        for mutation in (lambda s: s.update(step=np.array(1)), lambda s: s['lora_b']['m'].__setitem__((0, 0), .1),
                lambda s: s['lora_a'].update(v=np.zeros((1, 4), dtype=np.float32)),
                lambda s: s['lora_b'].update(v=np.zeros((4, 4096), dtype=np.float64)),
                lambda s: s.update(learning_rate=np.array(float('nan')))):
            bad = copy.deepcopy(state); mutation(bad)
            with self.assertRaises(ValueError):
                fit.validate_zero_adam_state(bad)

    def test_stage_receipts_capture_operator_cancellation_and_success(self):
        report = {}
        with self.assertRaises(KeyboardInterrupt):
            with fit.stage_receipt(report, lambda: None, 'cancelled', 60):
                raise KeyboardInterrupt('fixture')
        self.assertEqual(report['stages'][0]['status'], 'failed')
        self.assertEqual(report['stages'][0]['error']['type'], 'KeyboardInterrupt')
        with fit.stage_receipt(report, lambda: None, 'completed', 60):
            pass
        self.assertEqual(report['stages'][1]['status'], 'completed')

    def test_native_cli_requires_root_and_review_before_runtime_import(self):
        result = subprocess.run([sys.executable, '-m', 'experiments.ranker.pretrained.store_copy_continue_v1',
            '--preflight-report', '/tmp/nonexistent-continuation-preflight.json', '--out', '/tmp/unused-continuation-native'],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires root owner', result.stderr)


if __name__ == '__main__':
    unittest.main()
