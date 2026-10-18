"""CPU boundary/selection tests, including corruption and actual shortfalls."""
from copy import deepcopy
import unittest

from .jacket_inputs import load_snapshot
from .jacket_plan import build_plan, validate_plan, copy_jobs


class JacketPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = load_snapshot()
        cls.products = cls.snapshot['products']
        cls.tasks = cls.snapshot['tasks']

    def test_actual_shortfalls_no_fabricated_images_or_products(self):
        counts = [1,2,2,1,2,2,1,1,2,1]
        for task, count in zip(self.tasks, counts):
            plan = build_plan(task, self.products)
            self.assertEqual(plan['returnedCardCount'], count)
            self.assertEqual(plan['shortfall'], 3-count)
            self.assertEqual(validate_plan(plan, task, self.products), plan)
            self.assertTrue(all(c['imageAction'] is None for c in plan['cards']))
            self.assertEqual(plan['capabilities']['stock'], 'unknown')
            self.assertEqual(plan['capabilities']['shipping'], 'unknown')
            self.assertFalse(plan['selection']['learnedProvenanceEstablished'])
            self.assertEqual(len(copy_jobs(plan, task, self.products)), 1+3*count)

    def test_scores_cannot_override_hard_constraints(self):
        # Hiker: source0 Harbor and source3 Coastline qualify; 138-dollar
        # Ridgeline remains masked even with an arbitrarily high caller score.
        scores = [1.,2.,3.,4.,5.,1e20]
        plan = build_plan(self.tasks[1], self.products, scores=scores)
        self.assertEqual([c['productIndex'] for c in plan['cards']], [3,0])
        self.assertEqual(plan['selection']['source'], 'caller_supplied_scores')
        self.assertEqual([c['name'] for c in plan['cards']], [self.products[i]['name'] for i in [3,0]])

    def test_zero_eligible_is_explicit_and_generates_nothing(self):
        task = deepcopy(self.tasks[0]);task['budget'] = 0
        plan = build_plan(task, self.products, requested_cards=4)
        self.assertEqual((plan['cards'], plan['shortfall'], plan['grouping']['sections']), ([],4,[]))
        self.assertEqual(copy_jobs(plan, task, self.products), [])

    def test_empty_catalog_and_single_card_limit(self):
        self.assertEqual(build_plan(self.tasks[0], [])['shortfall'], 3)
        plan = build_plan(self.tasks[1], self.products, requested_cards=1)
        self.assertEqual(plan['returnedCardCount'], 1)
        self.assertEqual(plan['shortfall'], 0)
        self.assertEqual(plan['eligibleCount'], 2)

    def test_invalid_scores_and_counts(self):
        for scores in ([1.], [True]*6, [float('nan')]*6, [float('inf')]*6, [10**1000]*6, {'0':1}, ['1']*6):
            with self.assertRaises(ValueError):build_plan(self.tasks[0], self.products, scores=scores)
        for count in (True,False,0,-1,65,3.0,None):
            with self.assertRaises(ValueError):build_plan(self.tasks[0], self.products, requested_cards=count)

    def test_payload_tampering_and_false_provenance_rejected(self):
        original = build_plan(self.tasks[0], self.products)
        for mutate in (lambda p:p['cards'][0].update(productIndex=True),
                       lambda p:p['cards'][0].update(productIndex=99),
                       lambda p:p['cards'][0].update(price=0),
                       lambda p:p['cards'].append(deepcopy(p['cards'][0])),
                       lambda p:p['selection'].update(learnedProvenanceEstablished=True),
                       lambda p:p['grouping'].update(learned=True),
                       lambda p:p.update(shortfall=0),
                       lambda p:p['cards'][0].update(imageAction={'imageKey':'invented'})):
            plan=deepcopy(original);mutate(plan)
            with self.assertRaises(ValueError):validate_plan(plan, self.tasks[0], self.products)

    def test_prompt_support_binding_universal_intersection_and_independence(self):
        task=self.tasks[1];plan=build_plan(task,self.products);jobs=copy_jobs(plan,task,self.products)
        heading=jobs[0]['evidence']['support']['universalAttributes']
        self.assertEqual(heading, {'size':['L'],'waterproof':['yes']})
        self.assertTrue(all(j['evidence']['task']==task for j in jobs))
        self.assertTrue(all(len(j['messages'])==1 for j in jobs))
        for job in jobs[1:]:
            self.assertEqual(len(job['productIndices']),1)
            self.assertEqual(job['evidence']['products'],[self.products[job['productIndices'][0]]])
        jobs[1]['evidence']['products'][0]['attributes']['color'].append('invented')
        self.assertNotIn('invented',self.products[0]['attributes']['color'])
        self.assertNotIn('invented',jobs[2]['evidence']['products'][0]['attributes']['color'])
        jobs[1]['evidence']['support']['attributes']['color'].append('other')
        self.assertNotIn('other',jobs[2]['evidence']['support']['attributes']['color'])


if __name__ == '__main__':unittest.main()
