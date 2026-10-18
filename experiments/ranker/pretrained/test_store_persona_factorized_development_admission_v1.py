from copy import deepcopy
import unittest
from . import store_persona_factorized_development_admission_v1 as candidate


def shutdown():
    return {**{key: True for key in ('frozen927HashesUnchanged', 'copyCheckpointUnchanged', 'candidateAssetsUnchanged',
        'imageServingSourcesUnchanged', 'v5ServingExportUnchanged', 'v5HeadCheckpointUnchanged', 'rankAdapterRestoredToCopy',
        'rankAdapterAndPublicCacheUnchanged', 'learnedVariantAndQueryIntentAssetsUnchanged', 'adapterEngineSourcesUnchanged')},
        'headCheckpointSha256': candidate.development.FIXED['v5'], 'servingExportSha256': candidate.development.FIXED['rankExport'],
        'memory': {'processBudgetBytes': 7 * 2 ** 30, 'mlxPeakBytes': 6 * 2 ** 30}}


def browser():
    selected = {'expected': {'startup': {'actual': 'frozen-startup'}}}
    row = lambda index: {'index': index, 'protocol': {'passed': True},
        'adapterHttpAudit': {'protocolPassed': True, 'qualityPassed': True}}
    distribution = {'fixedDenominator': 10, 'valid': 10, 'rawMs': [225.] * 10}
    report = {'version': candidate.BROWSER_VERSION, 'developmentDemo': True, 'qualityGatePassed': False,
        'modelPromotion': False, 'acceptedProfile': False, 'conditioningPolicy': candidate.runtime.POLICY,
        'differentConditioningFromV2V3': True, 'candidateSelection': {**deepcopy(selected), 'pins': {'/prior.json': 'a' * 64}},
        'sourcesUnchanged': True, 'startupUnchanged': True, 'sourceHashes': {}, 'sourceHashesAfter': {},
        'healthBefore': {'body': deepcopy(selected['expected'])}, 'healthAfter': {'body': deepcopy(selected['expected'])},
        'summary': {'recorded': 13, 'protocolPassed': 13, 'unattempted': [],
            'loadedSwitchFirst3': distribution, 'loadedSwitchFull6': distribution},
        'attempts': [row(i) for i in range(13)], 'functionalArms': [{'kind': 'rapid-Bob-Maya-Bob', 'passed': True},
            {'kind': 'per-person-cart-and-checkout-invalidation', 'passed': True}],
        'distinctQueryAttempts': [row(i) for i in range(10)], 'coldProfileHttp': {'passed': True}}
    return report, selected


class AdmissionTests(unittest.TestCase):
    def validate(self, report=None):
        default, selected = browser()
        return candidate.validate_browser(report or default, selected, '/prior.json', 'a' * 64)

    def test_actual_smoke_counts_do_not_become_model_quality_acceptance(self):
        result = self.validate()
        self.assertTrue(result['factualSmokePassed']); self.assertFalse(result['v4ConditioningQualityValidated'])
        self.assertEqual(result['fixedBrowserRequests'], 13); self.assertEqual(result['distinctQueryRequests'], 10)

    def test_failed_partial_or_replaced_roster_rejected(self):
        for mutate in [lambda v: v['attempts'].pop(), lambda v: v['attempts'][0].update(index=1),
                       lambda v: v['attempts'][0]['adapterHttpAudit'].update(qualityPassed=False),
                       lambda v: v['functionalArms'].pop(), lambda v: v['distinctQueryAttempts'].pop(),
                       lambda v: v['coldProfileHttp'].update(passed=False),
                       lambda v: v['summary']['loadedSwitchFirst3'].update(valid=9)]:
            report, _ = browser(); mutate(report)
            with self.assertRaises(ValueError): self.validate(report)

    def test_distinct_query_http_protocol_and_both_distinct_functional_kinds_required(self):
        report, _ = browser(); report['distinctQueryAttempts'][0]['adapterHttpAudit']['protocolPassed'] = False
        with self.assertRaisesRegex(ValueError, 'ten distinct'): self.validate(report)
        report, _ = browser(); report['functionalArms'][1] = deepcopy(report['functionalArms'][0])
        with self.assertRaisesRegex(ValueError, 'rapid-switch'): self.validate(report)

    def test_exact_startup_source_and_prior_receipt_join_required(self):
        for mutate in [lambda v: v['candidateSelection']['pins'].update({'/prior.json': 'b' * 64}),
                       lambda v: v['healthAfter']['body']['startup'].update(actual='different'),
                       lambda v: v.update(sourcesUnchanged=False), lambda v: v['sourceHashesAfter'].update(changed='file')]:
            report, _ = browser(); mutate(report)
            with self.assertRaises(ValueError): self.validate(report)

    def test_quality_or_acceptance_relabeling_is_rejected(self):
        for key in ('qualityGatePassed', 'modelPromotion', 'acceptedProfile'):
            report, _ = browser(); report[key] = True
            with self.assertRaises(ValueError): self.validate(report)
        report, _ = browser(); report['version'] = 'store-persona-warm-browser-probe.v1'
        with self.assertRaises(ValueError): self.validate(report)

    def test_shutdown_all_controls_identity_and_memory_required(self):
        candidate.validate_shutdown(shutdown())
        for mutate in [lambda v: v.update(frozen927HashesUnchanged=False),
                       lambda v: v.update(rankAdapterRestoredToCopy=False), lambda v: v.update(adapterEngineSourcesUnchanged=False),
                       lambda v: v.update(headCheckpointSha256='b' * 64),
                       lambda v: v['memory'].update(mlxPeakBytes=8 * 2 ** 30)]:
            value = shutdown(); mutate(value)
            with self.assertRaises(ValueError): candidate.validate_shutdown(value)


if __name__ == '__main__': unittest.main()
