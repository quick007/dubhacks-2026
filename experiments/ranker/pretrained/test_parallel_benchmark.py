"""CPU-only checks for all-case denominators and unchanged generated token parity."""
import copy
import unittest
from .parallel_benchmark import parity, summary


class ParallelBenchmarkTests(unittest.TestCase):
    def arm(self):
        return {'status': 'completed-attempt', 'prefixSha256': 'prefix', 'rows': [
            {'rowId': 'title-1', 'tokenIds': [1, 2], 'finishReason': 'stop', 'promptSha256': 'p', 'maxTokens': 24}],
            'quality': {'allFieldsFulfilled': True, 'newTitleCount': 1, 'generatedHeadingCount': 1, 'generatedBadgeCount': 0},
            'timing': {'fullCompletionMs': 10.}}

    def test_parity_checks_raw_tokens_eos_prompt_and_budget(self):
        left, right = self.arm(), self.arm()
        self.assertTrue(parity(left, right)['exact'])
        for key, bad in [('tokenIds', [1, 3]), ('finishReason', 'length'), ('promptSha256', 'different'), ('maxTokens', 32)]:
            changed = copy.deepcopy(right)
            changed['rows'][0][key] = bad
            self.assertFalse(parity(left, changed)['exact'])
        changed = copy.deepcopy(right)
        changed['rows'] = []
        self.assertFalse(parity(left, changed)['exact'])

    def test_failed_unattempted_rejected_stay_in_denominator(self):
        valid = self.arm()
        invalid = self.arm()
        invalid['quality']['allFieldsFulfilled'] = False
        rows = [{'arms': {'batch2': valid}}, {'arms': {'batch2': invalid}},
                {'arms': {'batch2': {'status': 'failed'}}}, {'arms': {'batch2': {'status': 'not-attempted'}}}]
        result = summary(rows, 5)['batch2']
        self.assertEqual(result['validPageRateAllFixedCases'], .2)
        self.assertEqual(result['completedAttempts'], 2)
        self.assertEqual(result['failedAttempts'], 1)
        self.assertEqual(result['unattempted'], 2)
        self.assertEqual(result['validPageMs']['n'], 1)
        self.assertEqual(result['fullAttemptMs']['n'], 2)


if __name__ == '__main__':
    unittest.main()
