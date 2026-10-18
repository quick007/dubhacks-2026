"""CPU-only scheduler/instrumentation contracts; no model/native imports."""
from types import SimpleNamespace
import unittest

from .store_demo_decode_profile_v1 import REFERENCE_SHA256, assert_reference, decode_rows_profiled
from .store_demo_engine import RequestCancelled, demo_layout
from .store_catalog_v1 import load_snapshot
from .store_rank_session import action_bindings


class ProfileTests(unittest.TestCase):
    def fixture(self, count=6, suffix_tokens=40):
        request = {'query': 'browse', 'preferences': [], 'itemBudget': None, 'hardRequirements': []}
        cards = action_bindings(request, list(range(count)), load_snapshot())
        _, jobs = demo_layout(cards)
        calls = []; clock_value = [0.]
        def clock():
            clock_value[0] += .00001
            return clock_value[0]
        class Mx:
            def synchronize(self, stream):
                calls.append(('sync', stream))
        class Tokenizer:
            eos_token_ids = [99]
            def decode(self, tokens):
                return ' '.join(map(str, tokens))
        class Generator:
            def __init__(self, model, **kwargs):
                self.next_uid = 0; self.rows = {}; self.width = kwargs['completion_batch_size']
                calls.append(('construct', kwargs['completion_batch_size'], kwargs['prefill_batch_size'], kwargs['prefill_step_size']))
            def insert(self, prompts, **kwargs):
                uids = []
                for prompt, cap in zip(prompts, kwargs['max_tokens'], strict=True):
                    uid = self.next_uid; self.next_uid += 1; uids.append(uid)
                    self.rows[uid] = {'steps': 0, 'finishAt': 1 + prompt[0] % 3}
                self.assert_active()
                calls.append(('insert', len(prompts)))
                return uids
            def assert_active(self):
                if len(self.rows) > self.width:
                    raise AssertionError('Native generator over-admitted')
            def next_generated(self):
                responses = []
                for uid, row in list(self.rows.items()):
                    row['steps'] += 1
                    done = row['steps'] == row['finishAt']
                    responses.append(SimpleNamespace(uid=uid, token=99 if done else 1, finish_reason='stop' if done else None))
                    if done:
                        del self.rows[uid]
                calls.append(('next', len(responses)))
                return responses
            def close(self):
                calls.append(('close',))
        def clone(bb, state):
            calls.append(('clone',))
            return {'tokens': list(state['tokens'])}
        def suffix(tokenizer, row, branch, model):
            calls.append(('suffix', row['fields'][0]))
            index = next(i for i, job in enumerate(jobs) if job['fields'] == row['fields'])
            return [index] * suffix_tokens, object()
        deps = {'BatchGenerator': Generator, 'StopSequences': lambda stops: stops,
            'generation_stream': 'fake-stream', 'make_sampler': lambda **kwargs: kwargs,
            'clone_prefix_state': clone, 'exact_row_suffix': suffix,
            'guard_memory': lambda bb: calls.append(('guard',))}
        bb = SimpleNamespace(mx=Mx(), tokenizer=Tokenizer(), model=object())
        return bb, {'tokens': [4] * 328}, jobs, deps, clock, calls

    def test_live_reference_fingerprint_matches(self):
        self.assertEqual(assert_reference(), REFERENCE_SHA256)

    def test_same_complete_roster_admission_priority_and_stage_accounting(self):
        for count, width in ((1, 6), (6, 6), (6, 10), (6, 20)):
            bb, state, jobs, deps, clock, calls = self.fixture(count)
            first = []
            rows, timing = decode_rows_profiled(bb, state, jobs, width, first.append, lambda: False,
                _dependencies=deps, _clock=clock)
            self.assertEqual([r['fields'] for r in rows], [r['fields'] for r in jobs])
            self.assertEqual(len(first), 1)
            self.assertEqual(len(first[0]), 1 + 3 * min(3, count))
            profile = timing['profile']
            self.assertFalse(profile['schedulingChanged'])
            self.assertEqual(sum(a['count'] for a in profile['admissions']), len(jobs))
            self.assertLessEqual(profile['maximumAdmittedActiveRows'], width)
            self.assertGreater(profile['stageMs']['cacheCloneMaterialization'], 0)
            self.assertGreater(profile['stageMs']['nextGeneratedCall'], 0)
            self.assertGreater(profile['stageMs']['explicitGenerationStreamSync'], 0)
            self.assertGreater(profile['stageMs']['generatorCloseAndSync'], 0)
            self.assertEqual(sum(c[0] == 'clone' for c in calls), len(jobs))
            self.assertEqual(sum(c[0] == 'suffix' for c in calls), len(jobs))
            self.assertEqual(calls[-2][0], 'close')
            self.assertEqual(profile['status'], 'completed')
            self.assertEqual(profile['accountedStageMsIncludingClose'], sum(profile['stageMs'].values()))

    def test_current_refill_policy_admits_below_fold_before_first_ready(self):
        bb, state, jobs, deps, clock, calls = self.fixture()
        _, timing = decode_rows_profiled(bb, state, jobs, 10, lambda rows: None, lambda: False,
            _dependencies=deps, _clock=clock)
        self.assertTrue(any(a['belowFoldBeforeFirstSectionComplete'] for a in timing['profile']['admissions']))
        self.assertFalse(timing['profile']['GPUKernelPrefillDecodeSplitMeasured'])

    def test_cancellation_preserves_partial_profile_and_closes(self):
        bb, state, jobs, deps, clock, calls = self.fixture()
        def cancelled():
            return any(c[0] == 'next' for c in calls)
        with self.assertRaises(RequestCancelled) as caught:
            decode_rows_profiled(bb, state, jobs, 10, lambda rows: None, cancelled,
                _dependencies=deps, _clock=clock)
        self.assertTrue(caught.exception.partial_rows)
        self.assertEqual(caught.exception.partial_profile['status'], 'failed')
        self.assertTrue(any(c[0] == 'close' for c in calls))

    def test_context_failure_does_not_truncate_or_start_decode(self):
        bb, state, jobs, deps, clock, calls = self.fixture(suffix_tokens=257)
        with self.assertRaisesRegex(ValueError, 'no truncation') as caught:
            decode_rows_profiled(bb, state, jobs, 10, lambda rows: None, lambda: False,
                _dependencies=deps, _clock=clock)
        self.assertEqual(caught.exception.partial_rows, [])
        self.assertFalse(any(c[0] == 'next' for c in calls))
        self.assertTrue(any(c[0] == 'close' for c in calls))


if __name__ == '__main__':
    unittest.main()
