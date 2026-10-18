"""CPU privacy, pairing, boundary and coverage checks; no browser/native execution."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from . import validation as packet

class HeldOutPacketTests(unittest.TestCase):
    def test_strict_packet_and_fixed_budgets(self):
        report=packet.validate_packet()
        self.assertEqual((report['pairs'],report['plannedRuns'],report['satisfiableCases']),(10,20,8))
        self.assertEqual((report['knownEmpty'],report['unknownRequiredFact']),(1,1))
        self.assertEqual(report['executedRunsClaimed'],0)
        self.assertTrue(all(r['split']=='held-out' for r in packet.read('pair-plan.json')['schedule']))
        self.assertEqual(packet.read('pair-plan.json')['experimentId'],'actual-store-heldout-shopper-v2')
        self.assertEqual(packet.read('manifest.json')['experimentId'],'actual-store-heldout-shopper-v2')
        self.assertEqual(packet.read('preregistration.json')['experimentId'],'actual-store-heldout-shopper-v2')
        cohort=packet.read('cohort-preregistration.json')
        self.assertEqual(cohort['fixedDesign']['assignedSessions'],120)
        self.assertEqual(cohort['fixedDesign']['globalRandomSeed'],20261018)
        self.assertEqual(cohort['fixedDesign']['negativeArmAssignments'],24)
        self.assertEqual(cohort['runtimePilot']['pairs'],5)
        self.assertEqual(cohort['deadline']['utc'],'2026-10-18T13:00:00Z')

    def test_shopper_projection_cannot_read_gold_or_catalog(self):
        original=packet.read
        def guarded(name,directory=packet.DATA):
            if name in ('evaluator-gold.json','catalog-binding.json'):
                raise AssertionError('Evaluator bytes entered shopper projection')
            return original(name,directory)
        with patch.object(packet,'read',side_effect=guarded):
            for row in original('public-observations.json'):
                for arm in ('generic','local-model'):
                    packet.assert_no_gold(packet.shopper_assignment(row['caseId'],arm))

    def test_only_model_mode_differs_within_pair(self):
        for row in packet.read('public-observations.json'):
            a,b=[packet.shopper_assignment(row['caseId'],arm) for arm in ('generic','local-model')]
            self.assertEqual(a['entryUrl']+'?model=local',b['entryUrl'])
            for key in ('shopperNeeds','recommenderContext','instructions'):self.assertEqual(a[key],b[key])

    def test_private_colors_are_soft_and_never_public(self):
        gold=packet._index(packet.read('evaluator-gold.json'))
        for row in packet.read('public-observations.json')[:8]:
            cid=row['caseId'];case=packet.shopper_case(cid);prefs=case['privateShopperNeeds']['preferences']
            self.assertEqual(len(prefs),1)
            self.assertEqual(prefs[0]['attribute'],'selectedColor')
            self.assertNotIn(prefs[0]['value'].casefold(),row['observableContext']['query'].casefold())
            self.assertEqual(len(gold[cid]['acceptableVariantIds']),3,'Every visible color remains hard-constraint eligible')
            self.assertFalse(any(n['attribute']=='selectedColor' for n in case['privateShopperNeeds']['hardConstraints']))

    def test_strict_boundary_is_empty_but_inclusive_price_qualifies(self):
        row=packet.read('evaluator-gold.json')[8];case=packet.shopper_case(row['caseId'])
        catalog=packet.read('catalog-binding.json')['catalogSnapshot']
        self.assertEqual(packet.expected_variants(case,row,catalog),([],[]))
        inclusive=deepcopy(case)
        next(n for n in inclusive['privateShopperNeeds']['hardConstraints'] if n['attribute']=='itemPrice')['operator']='at-most'
        variants,unknown=packet.expected_variants(inclusive,row,catalog)
        self.assertEqual(len(variants),3);self.assertEqual(unknown,[])

    def test_unavailable_required_fact_cannot_be_assumed(self):
        row=packet.read('evaluator-gold.json')[9];catalog=packet.read('catalog-binding.json')['catalogSnapshot']
        variants,unknown=packet.expected_variants(packet.shopper_case(row['caseId']),row,catalog)
        self.assertEqual(variants,[]);self.assertEqual(unknown,row['unknownRequiredAttributes'])
        self.assertTrue(all(unknown[0] not in p['facts'] for p in catalog['products']))

    def test_six_packet_coverage_and_exact_public_wording_overlap(self):
        import json
        from collections import Counter
        queries=[];cells=Counter()
        for version in range(2,8):
            directory=packet.DATA.parent/f'store_shopper_eval_v{version}'
            public=json.loads((directory/'public-observations.json').read_text())
            private=json.loads((directory/'shopper-instructions.json').read_text())
            gold=json.loads((directory/'evaluator-gold.json').read_text())
            queries.extend(row['observableContext']['query'].strip().casefold() for row in public)
            for needs,answer in zip(private[:8],gold[:8],strict=True):
                cells[(answer['acceptableProductIds'][0],needs['privateShopperNeeds']['preferences'][0]['value'])]+=1
        self.assertEqual((len(queries),len(set(queries))),(60,60))
        self.assertEqual(len(cells),48);self.assertEqual(set(cells.values()),{1})
        old=json.loads((packet.DATA.parent/'store_shopper_eval_v1/public-observations.json').read_text())
        known=[row['observableContext']['query'] for row in old]
        training=json.loads((packet.DATA.parent/'store_function_v5/public-cases.json').read_text())
        known.extend(value for row in training for value in [row['request']['query'],*row['request'].get('preferences',[])])
        self.assertFalse(set(queries)&{value.strip().casefold() for value in known})

    def test_injection_and_external_origin_rejected(self):
        row=packet.read('public-observations.json')[0]
        with self.assertRaises(ValueError):packet.validate_public({**row['observableContext'],'evaluatorGold':{}})
        with self.assertRaises(ValueError):packet.shopper_assignment(row['caseId'],'generic',base_url='https://example.com/')

if __name__=='__main__':unittest.main()
