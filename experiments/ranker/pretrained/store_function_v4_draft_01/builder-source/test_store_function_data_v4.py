"""CPU-only data contracts; oracle fixtures are not learned-model evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from .store_function_data_v4 import (audit, build, eligible, evaluate_rankings,
    freeze, load_frozen, product_text, request, request_text, validate_request)


class StoreFunctionV4DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = build()

    def test_bounded_counts_and_all_functions_in_sparse_and_current_training(self):
        result = audit(self.packet)
        self.assertEqual(result['counts']['train']['cases'], 64)
        self.assertEqual(result['counts']['evaluation']['cases'], 24)
        self.assertEqual(result['counts']['train']['multiNeed'], 16)
        self.assertEqual(result['counts']['evaluation']['multiNeed'], 8)
        self.assertEqual(result['counts']['train']['emptyTarget'], 4)
        self.assertEqual(result['counts']['evaluation']['emptyTarget'], 3)
        labels = {x['caseId']: x for x in self.packet['trainLabels']}
        for family in ('current-single-need', 'sparse-observed-search'):
            cases = [c for c in self.packet['publicCases'] if c['family'] == family]
            self.assertEqual({tuple(labels[c['caseId']]['relevantProductIndices']) for c in cases},
                             {(i,) for i in range(16)})

    def test_only_public_request_fields_enter_encoder_no_hidden_truth(self):
        for c in self.packet['publicCases']:
            text = request_text(c['request'])
            for bad in ('caseId', 'split', 'family', 'relevanceByProduct', 'publicSupport', 'personaTruth'):
                self.assertNotIn(bad, text)
            poisoned = deepcopy(c['request']);poisoned['personaTruth'] = 'secret shopper truth'
            with self.assertRaises(ValueError):request_text(poisoned)
        bad = request('Suggest an item.');bad['observedProfile']['inferredGender'] = 'invented'
        with self.assertRaises(ValueError):validate_request(bad)
        for p in self.packet['products']:
            self.assertNotIn('image', product_text(p))
            self.assertNotIn('productId', product_text(p))

    def test_counterfactual_pairs_hold_task_budget_and_hard_guards_fixed(self):
        byid = {c['caseId']: c for c in self.packet['publicCases']}
        labels = {x['caseId']: x for x in self.packet['evaluationLabels']}
        pairs = self.packet['preferencePairs'];self.assertEqual(len(pairs), 8)
        for pair in pairs:
            a,b = pair['caseIds'];qa,qb = byid[a]['request'],byid[b]['request']
            for field in ('query', 'itemBudget', 'hardRequirements'):
                self.assertEqual(qa[field], qb[field])
            self.assertNotEqual(qa, qb)
            self.assertFalse(set(labels[a]['relevantProductIndices']) & set(labels[b]['relevantProductIndices']))

    def test_multi_need_target_sets_held_out_and_not_forced_to_single_class(self):
        train = {tuple(x['relevantProductIndices']) for x in self.packet['trainLabels'] if len(x['relevantProductIndices']) == 2}
        evaluation = {tuple(x['relevantProductIndices']) for x in self.packet['evaluationLabels'] if len(x['relevantProductIndices']) == 2}
        self.assertEqual(len(train), 16);self.assertEqual(len(evaluation), 8)
        self.assertFalse(train & evaluation)
        for r in self.packet['trainLabels'] + self.packet['evaluationLabels']:
            if len(r['relevantProductIndices']) == 2:
                self.assertEqual(r['relevanceByProduct'].count(2), 2)

    def test_inclusive_budget_colors_and_unknown_attribute_abstention(self):
        products = self.packet['products']
        self.assertTrue(eligible(request('A mug.', budget=24), products[0]))
        self.assertFalse(eligible(request('A mug.', budget=23), products[0]))
        self.assertTrue(eligible(request('A pouch.', requirements=[{'attribute':'color','value':'Navy'}]), products[14]))
        self.assertFalse(eligible(request('A pouch.', requirements=[{'attribute':'color','value':'Cobalt'}]), products[14]))
        for attribute in ('size', 'stock'):
            self.assertFalse(eligible(request('A mug.', requirements=[{'attribute':attribute,'value':'required'}]), products[0]))
        for value in (True, float('nan'), float('inf'), -1):
            with self.assertRaises(ValueError):validate_request(request('A mug.', budget=value))

    def test_exact_name_queries_explicitly_override_stale_profile(self):
        products = self.packet['products'];labels = {x['caseId']:x for x in self.packet['trainLabels']+self.packet['evaluationLabels']}
        cases = [x for x in self.packet['publicCases'] if x['family']=='exact-name-overrides-stale-search']
        self.assertEqual(len(cases), 5)
        for c in cases:
            r = labels[c['caseId']];i = r['exactNameTargetIndex']
            self.assertIn(products[i]['name'], c['request']['query'])
            self.assertTrue(c['request']['observedProfile']['recentSearches'])
            self.assertEqual(r['relevantProductIndices'], [i])
        self.assertFalse(eligible(request('Find Trail mug.', budget=23), products[0]))

    def test_explicit_negatives_are_labels_not_unsupported_fact_inventions(self):
        cases = {c['caseId']:c for c in self.packet['publicCases']}
        avoided_count = 0
        for r in self.packet['trainLabels'] + self.packet['evaluationLabels']:
            if r['explicitlyAvoidedProductIndices']:
                avoided_count += 1;q=cases[r['caseId']]['request']
                self.assertTrue(q['negativePreferences'] or q['observedProfile']['statedAvoidances'])
                self.assertFalse(set(r['desiredProductIndices']) & set(r['explicitlyAvoidedProductIndices']))
                self.assertTrue(all(r['relevanceByProduct'][i] == -1 for i in r['explicitlyAvoidedProductIndices']))
        self.assertEqual(avoided_count, 18)

    def test_label_tamper_or_private_schema_extensions_rejected(self):
        packet = deepcopy(self.packet);packet['trainLabels'][0]['relevanceByProduct'][0] = 0
        with self.assertRaises(ValueError):audit(packet)
        packet = deepcopy(self.packet);packet['trainLabels'][0]['publicSupport'][0]['sourceFacts']['itemPrice'] = 1
        with self.assertRaises(ValueError):audit(packet)
        packet = deepcopy(self.packet);packet['publicCases'][0]['request']['targetProductIndex'] = 0
        with self.assertRaises(ValueError):audit(packet)

    def test_fixed_denominator_multineed_oracle_and_empty_abstention_metrics(self):
        cases = [c for c in self.packet['publicCases'] if c['split']=='evaluation'];labels=self.packet['evaluationLabels']
        oracle = {r['caseId']:r['relevantProductIndices'] for r in labels}
        result = evaluate_rankings(oracle, cases, labels, self.packet['preferencePairs'])
        self.assertEqual(result['fixedCases'],24);self.assertEqual(result['nonemptyCases'],21)
        self.assertEqual(result['emptyCases'],3);self.assertEqual(result['hitAtOne'],21)
        self.assertEqual(result['allTargetsAtTheirCount'],21)
        self.assertEqual(result['relevantTargetDenominator'],29)
        self.assertEqual(result['correctEmptyAbstentions'],3)
        self.assertEqual(result['pairsWithBothTargetSetsRecovered'],8)
        collapsed = {rid:ids[:1] for rid,ids in oracle.items()}
        collapse_result = evaluate_rankings(collapsed,cases,labels,self.packet['preferencePairs'])
        self.assertEqual(collapse_result['hitAtOne'],21)
        self.assertEqual(collapse_result['allTargetsAtTheirCount'],13)
        self.assertEqual(collapse_result['pairsWithBothTargetSetsRecovered'],4)
        del oracle[next(iter(oracle))]
        with self.assertRaises(ValueError):evaluate_rankings(oracle,cases,labels,self.packet['preferencePairs'])

    def test_evaluator_rejects_invalid_outputs_and_discloses_guard_violations(self):
        cases=[c for c in self.packet['publicCases'] if c['split']=='evaluation'];labels=self.packet['evaluationLabels']
        oracle={r['caseId']:r['relevantProductIndices'] for r in labels};rid=labels[0]['caseId']
        for bad in ([0,0], [True], [16], list(range(7))):
            trial=deepcopy(oracle);trial[rid]=bad
            with self.assertRaises(ValueError):evaluate_rankings(trial,cases,labels,self.packet['preferencePairs'])
        empty = next(r for r in labels if not r['eligibleProductIndices'])
        trial=deepcopy(oracle);trial[empty['caseId']]=[0]
        result=evaluate_rankings(trial,cases,labels,self.packet['preferencePairs'])
        self.assertEqual(result['safeCases'],23)
        self.assertEqual(result['correctEmptyAbstentions'],2)

    def test_freeze_preserves_existing_outputs_and_rejects_changed_files(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'dataset';receipt=freeze(out);self.assertEqual(load_frozen(out),self.packet)
            self.assertEqual(receipt['nativeForwards'],0);self.assertEqual(receipt['optimizerUpdates'],0)
            with self.assertRaises(FileExistsError):freeze(out)
            path=out/'train-labels.json';path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'Frozen artifact changed'):load_frozen(out)

    def test_v3_is_inspected_replay_and_eval_wording_is_not_recycled(self):
        source=Path(__file__).parent/'store_function_v3'/'public-cases.json'
        old=json.loads(source.read_text());old_preferences={p for c in old for p in c['request']['preferences']}
        new=[c for c in self.packet['publicCases'] if c['split']=='evaluation']
        for c in new:
            self.assertFalse(set(c['request']['preferences']) & old_preferences)
        self.assertIn('DEVELOPMENT REPLAY',audit(self.packet)['v3Status'])


if __name__ == '__main__':
    unittest.main()
