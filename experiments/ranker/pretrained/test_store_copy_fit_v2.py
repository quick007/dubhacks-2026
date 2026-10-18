"""CPU-only runner contracts: no model load, tokenizer download or MLX."""
import copy
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from . import store_copy_fit_v2 as fit
from . import store_copy_data_v2 as data


class WordTokenizer:
    """Deterministic complete-chat token fixture, no native model runtime."""
    eos_token_id = 99
    eos_token_ids = [99]
    def __init__(self):
        self.vocab = {}
    def encode(self, text, add_special_tokens=False):
        if text == '<|im_end|>':
            return [99]
        result = []
        for word in text.split():
            self.vocab.setdefault(word, len(self.vocab) + 100)
            result.append(self.vocab[word])
        return result
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        result = []
        for message in messages:
            if message['role'] != 'user':
                raise AssertionError('Old raw-focused role entered ordinary chat fixture')
            result += [1] + self.encode(message['content']) + [2]
        return result + ([3] if add_generation_prompt else [])


class StoreCopyFitV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.packet = fit.dataset_packet()

    def fields(self, split='train', index=0):
        case = self.packet[split]['cases'][index]
        return case, [f for f in self.packet[split]['fields'] if f['caseId'] == case['caseId']]

    def native_rows(self, split='eval', index=0):
        case, fields = self.fields(split, index)
        _, rows, ordered = fit.serving_rows(case, fields)
        for row, field in zip(rows, ordered, strict=True):
            row.update(text=field['targetValue'], tokenIds=[100], rawSampledTokenIds=[100, 99],
                stopTokenId=99, arSteps=2, finishReason='stop')
        return case, rows

    def test_import_is_cpu_only(self):
        subprocess.run([sys.executable, '-c', 'import sys; import experiments.ranker.pretrained.store_copy_fit_v2; '
            'assert not any(n == "mlx" or n.startswith("mlx.") for n in sys.modules)'],
            check=True, capture_output=True, text=True)

    def test_exact_serving_inputs_all_cases_and_no_inference_targets(self):
        for split, group in self.packet.items():
            for case in group['cases']:
                fields = [f for f in group['fields'] if f['caseId'] == case['caseId']]
                common, jobs, ordered = fit.serving_rows(case, fields)
                self.assertEqual(common['role'], 'user')
                self.assertEqual(tuple(r['fields'][0] for r in jobs), fit.FIELDS)
                self.assertEqual(sum(r['maxTokens'] for r in jobs), 416)
                self.assertEqual(sum(r['maxTokens'] for r in jobs[:10]), 208)
                for row, field in zip(jobs, ordered, strict=True):
                    self.assertEqual(row['messages'], field['messages'])
                    self.assertNotIn('targetValue', row)
        case, fields = self.fields(); bad = copy.deepcopy(fields)
        bad[0]['messages'][0]['content'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'serving renderer'):
            fit.serving_rows(case, bad)
        with self.assertRaisesRegex(ValueError, 'membership'):
            fit.serving_rows(case, fields[:-1])

    def test_fixed_shuffled_updates_train_only_and_no_warm_start(self):
        order = fit.shuffled_order()
        self.assertEqual(sorted(order), list(range(320)))
        self.assertNotEqual(order, list(range(320)))
        self.assertEqual(order, fit.shuffled_order())
        self.assertGreater(len({v // 20 for v in order[:20]}), 7)
        for count in (True, 319, 321, 320.):
            with self.assertRaises(ValueError):
                fit.shuffled_order(count)
        self.assertEqual(fit.CONFIG['residentHostFeatureRows'], 1)
        self.assertEqual(fit.CONFIG['totalFeatureArtifactBudgetBytes'], 5 * 2**30)
        self.assertIn('exactly zero B', fit.CONFIG['initialization'])
        train_ids = {c['caseId'] for c in self.packet['train']['cases']}
        eval_ids = {c['caseId'] for c in self.packet['eval']['cases']}
        self.assertFalse(train_ids & eval_ids)

    def test_preflight_uses_exact_ordinary_chat_and_eval_never_extracted(self):
        result = fit.preflight(WordTokenizer())
        self.assertEqual(result['modelCalls'], 0)
        self.assertEqual(result['trainingUpdates'], 0)
        self.assertEqual(result['fixedArms'], 24)
        self.assertEqual(result['gateWork']['trainRows'], 320)
        self.assertEqual(result['gateWork']['evaluationRowsExtracted'], 0)
        for split, count in (('train', 320), ('eval', 160)):
            rows = result['splits'][split]['rows']
            self.assertEqual(len(rows), count)
            self.assertTrue(all(0 < r['suffixTokens'] <= 256 for r in rows))
        self.assertLess(result['projectedUncompressedTrainingArtifactsWithReserveBytes'], 5 * 2**30)

    def test_probe_control_restores_true_zero_initial_even_when_gate_raises(self):
        initial, control = {'A': [1], 'B': [0]}, {'A': [1], 'B': [1]}
        current = copy.deepcopy(initial); history = []
        def restore(value):
            current.clear(); current.update(copy.deepcopy(value)); history.append(copy.deepcopy(value))
        def receipts():
            return copy.deepcopy(current)
        for failed in (False, True):
            try:
                with fit.temporary_probe_control(restore, receipts, initial, control, initial, control):
                    self.assertEqual(current, control)
                    if failed:
                        raise RuntimeError('injected probe failure')
            except RuntimeError:
                self.assertTrue(failed)
            self.assertEqual(current, initial)
        self.assertEqual(history, [control, initial, control, initial])
        current['B'] = [2]
        with self.assertRaisesRegex(AssertionError, 'before feature extraction'):
            with fit.temporary_probe_control(restore, receipts, initial, control, initial, control):
                self.fail('Wrong initial adapter entered a gate')

    def test_probe_control_restores_initial_if_control_installation_is_wrong(self):
        current = {'state': 'initial'}
        def restore(value):
            current['state'] = 'broken' if value == 'control' else value
        with self.assertRaisesRegex(AssertionError, 'gate parameters changed'):
            with fit.temporary_probe_control(restore, lambda: current['state'], 'initial', 'control', 'initial', 'control'):
                self.fail('Incorrect control entered native probe')
        self.assertEqual(current['state'], 'initial')

    def test_all24_fixed_arms_keep_failures_out_of_valid_speed_samples(self):
        arms = fit.arm_roster(self.packet)
        self.assertEqual(len(arms), 24)
        self.assertEqual({a['width'] for a in arms}, {6})
        self.assertEqual({a['mode'] for a in arms}, set(fit.MODES))
        arms[0]['status'] = 'failed'
        values = fit.summary(arms)['frozen']
        self.assertEqual(values['fixedCases'], 8)
        self.assertEqual(values['failed'], 1)
        self.assertEqual(values['validSixCardRateFixedDenominator'], 0)
        self.assertEqual(values['validFullSixReadyMs'], [])
        case, rows = self.native_rows()
        first = data.assemble(case['bound'], rows[:10], cards=3)
        full = data.assemble(case['bound'], rows, cards=6)
        arms[1].update(status='completed-attempt', firstThree=first, fullSix=full,
            firstThreeReceipt={'validatedAndSerializedAtMs': 10}, fullSixReceipt={'validatedAndSerializedAtMs': 20})
        values = fit.summary(arms)['failed-daypack']
        self.assertEqual(values['validSixCardRateFixedDenominator'], 1 / 8)
        self.assertEqual(values['validFullSixReadyMs'], [20])
        arms[1]['fullSix']['validation']['valid'] = False
        values = fit.summary(arms)['failed-daypack']
        self.assertEqual(values['attemptFullSixReadyMs'], [20])
        self.assertEqual(values['validFullSixReadyMs'], [])

    def test_serialization_hash_and_byte_count_are_actual_payload(self):
        case, rows = self.native_rows()
        result = data.assemble(case['bound'], rows)
        encoded = fit.encoded_payload(result)
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), result['payloadSha256'])
        self.assertEqual(len(encoded), result['payloadBytes'])
        result['payload']['sections'][0]['heading'] += ' changed'
        with self.assertRaisesRegex(AssertionError, 'serialized payload'):
            fit.encoded_payload(result)

    def case_stubs(self):
        bb = types.SimpleNamespace(mx=object(), runtime_memory=lambda: {'test': True})
        state = {'tokens': [1, 2], 'cache': [], 'messages': [{'role': 'user', 'content': 'test'}]}
        return bb, state

    def test_request_measures_root_and_real_first3_full6_serialization(self):
        case, rows = self.native_rows(); bb, state = self.case_stubs(); events = []
        def decode(backbone, source, jobs, width, first_ready):
            self.assertEqual(width, 6); self.assertEqual(len(jobs), 20)
            self.assertTrue(all('targetValue' not in row for row in jobs))
            events.append('decode'); first_ready(rows[:10]); events.append('first-ready')
            return rows, {'continuous': True}
        real_encode = fit.encoded_payload
        def encode(value):
            events.append('serialize-' + str(value['validation']['requiredFields']))
            return real_encode(value)
        with patch('experiments.ranker.pretrained.branch_local_copy.prepare_root', side_effect=lambda *a: (events.append('root') or state, {})), \
                patch('experiments.ranker.pretrained.fused_prefix._complete_cache'), \
                patch('experiments.ranker.pretrained.native_batch_diagnosis.state_hash', return_value='same-source'), \
                patch.object(fit, 'decode_complete_rows', side_effect=decode), patch.object(fit, 'encoded_payload', side_effect=encode):
            result = fit.run_case(bb, case['bound'])
        self.assertEqual(events, ['root', 'decode', 'serialize-10', 'first-ready', 'serialize-20'])
        self.assertTrue(result['firstThree']['validation']['valid'])
        self.assertTrue(result['fullSix']['validation']['valid'])
        self.assertGreaterEqual(result['copyAdmittedAtMs'], result['metadataReadyAtMs'])
        self.assertGreaterEqual(result['firstThreeReceipt']['validatedAndSerializedAtMs'], result['copyAdmittedAtMs'])
        self.assertGreaterEqual(result['fullSixReceipt']['validatedAndSerializedAtMs'], result['firstThreeReceipt']['validatedAndSerializedAtMs'])
        self.assertEqual(result['sourceCacheBefore'], result['sourceCacheAfter'])
        self.assertFalse(result['requestTimingIncludesCallerRanking'])

    def test_decode_failure_retains_observed_partial_rows(self):
        case, rows = self.native_rows(); bb, state = self.case_stubs()
        error = TimeoutError('injected request deadline'); error.partial_rows = rows[:6]
        with patch('experiments.ranker.pretrained.branch_local_copy.prepare_root', return_value=(state, {})), \
                patch('experiments.ranker.pretrained.fused_prefix._complete_cache'), \
                patch('experiments.ranker.pretrained.native_batch_diagnosis.state_hash', return_value='same-source'), \
                patch.object(fit, 'decode_complete_rows', side_effect=error):
            with self.assertRaises(TimeoutError) as caught:
                fit.run_case(bb, case['bound'])
        self.assertEqual(caught.exception.partial_result['rawRows'], rows[:6])

    def test_timeout_signal_handler_and_timer_restored(self):
        previous = signal.getsignal(signal.SIGALRM)
        with self.assertRaisesRegex(TimeoutError, 'fixed 120-second'):
            with fit.deadline(120, 'test gate'):
                signal.raise_signal(signal.SIGALRM)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    def test_native_cli_rejects_unowned_call_before_native_import(self):
        process = subprocess.run([sys.executable, '-m', 'experiments.ranker.pretrained.store_copy_fit_v2',
            '--preflight-report', '/tmp/nonexistent-store-copy-fit-preflight.json', '--out', '/tmp/unused-store-copy-fit-output'],
            capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('root owner', process.stderr)


if __name__ == '__main__':
    unittest.main()
