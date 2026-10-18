"""CPU safety checks for conditional output-difference measurement."""
import copy
import unittest
from .parallel_semantic_benchmark import comparison_policy, validate_controls


def controls():
    arms, comparisons = {}, {}
    for width in (1, 2, 3):
        for mode in ('fork', 'independent', 'full'):
            for repeat in (0, 1):
                arms[f'w{width}-{mode}-r{repeat}'] = {
                    'status':'completed', 'width':width, 'mode':mode,
                    'baseBefore':'base', 'baseAfter':'base', 'branchMatchesPristine':[True]*width,
                    'rows':{'badge-1':{'rawSampledTokenIds':[1]*7, 'finishReason':'length',
                        'probe':{'alignment':{'exact':True}, 'actualActiveWidth':width}}}}
            comparisons[f'w{width}-{mode}-r1-vs-repeat0'] = {'raw':{'exact':True}, 'logprobs':{'exact':True}}
        for repeat in (0,1):
            comparisons[f'w{width}-independent-r{repeat}-vs-fork'] = {
                'sameEarlierContextAllRows':True, 'raw':{'exact':True}, 'logprobs':{'exact':True}}
        if width > 1:
            for mode in ('fork','independent','full'):
                comparisons[f'w{width}-{mode}-r0-vs-width1'] = {'sameEarlierBadgeContext':True, 'raw':{'exact':False}}
    return {'status':'completed','pristineBaseHash':'base','arms':arms,'comparisons':comparisons,
            'memoryLimitBytes':6*2**30,'memory':{'mlxPeakBytes':6*2**30}}


def attempt():
    return {'status':'completed-attempt','prefixSha256':'prefix','rows':[{
        'rowId':'badge-1','promptSha256':'prompt','maxTokens':7,'tokenIds':[1],
        'rawSampledTokenIds':[1,9],'arSteps':2,'stopTokenId':9,'finishReason':'stop'}]}


class SemanticComparisonTests(unittest.TestCase):
    def test_peak_excursion_and_raised_budget_block_before_model_loading(self):
        for change in ('peak','budget'):
            value = controls()
            if change == 'peak':
                value['memory']['mlxPeakBytes'] += 1
            else:
                value['memoryLimitBytes'] += 1
            with self.assertRaisesRegex(ValueError,'fixed six-GiB peak budget'):
                validate_controls(value)

    def test_requires_completed_controls_and_identical_complete_prefixes(self):
        validate_controls(controls())
        for key in ('status','baseAfter','branchMatchesPristine'):
            value = controls()
            value['arms']['w2-independent-r0'][key] = {'status':'running','baseAfter':'changed','branchMatchesPristine':[True,False]}[key]
            with self.assertRaises(ValueError):
                validate_controls(value)
        value = controls(); del value['arms']['w3-full-r1']
        with self.assertRaises(ValueError):
            validate_controls(value)

    def test_rejects_alignment_repeat_and_causal_context_failures(self):
        for key, subkey in [('w2-independent-r0-vs-fork','raw'),('w3-full-r1-vs-repeat0','logprobs')]:
            value = controls(); value['comparisons'][key][subkey]['exact'] = False
            with self.assertRaises(ValueError):
                validate_controls(value)
        value = controls(); value['comparisons']['w2-full-r0-vs-width1']['sameEarlierBadgeContext'] = False
        with self.assertRaises(ValueError):
            validate_controls(value)
        value = controls(); value['arms']['w1-fork-r0']['rows']['badge-1']['probe']['alignment']['exact'] = False
        with self.assertRaises(ValueError):
            validate_controls(value)

    def test_raw_output_difference_is_retained_without_changing_input_contract(self):
        left, right = attempt(), attempt()
        right['rows'][0]['rawSampledTokenIds'][0] = 2
        right['rows'][0]['tokenIds'][0] = 2
        result = comparison_policy(left,right)
        self.assertFalse(result['exact'])
        self.assertTrue(result['rows'][0]['samePromptAndBudget'])
        for field in ('promptSha256','maxTokens','rowId'):
            changed = copy.deepcopy(right); changed['rows'][0][field] = 'changed'
            with self.assertRaises(ValueError):
                comparison_policy(left,changed)
        right['prefixSha256'] = 'different'
        with self.assertRaises(ValueError):
            comparison_policy(left,right)


if __name__ == '__main__':
    unittest.main()
