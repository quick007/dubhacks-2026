"""Actual V4 factual-smoke/shutdown annex; never model quality acceptance.

CPU reads/hashes only. All actual evidence must already exist. No native, HTTP,
browser or service starts; original development and accepted selectors unchanged.
"""
from copy import deepcopy
import json
from pathlib import Path

from . import store_persona_development_admission_v1 as development
from . import store_persona_factorized_http_probe_v1 as factual
from . import store_persona_adapter_runtime_v4 as runtime

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
VERSION = 'persona-factorized-development-admission.v1'
BROWSER_VERSION = 'store-persona-factorized-browser-probe.v1'
SELF = Path(__file__).resolve()


def need(value, message):
    if not value: raise ValueError(message)


def validate_shutdown(value):
    for key in ('frozen927HashesUnchanged', 'copyCheckpointUnchanged', 'candidateAssetsUnchanged',
                'imageServingSourcesUnchanged', 'v5ServingExportUnchanged', 'v5HeadCheckpointUnchanged',
                'rankAdapterRestoredToCopy', 'rankAdapterAndPublicCacheUnchanged',
                'learnedVariantAndQueryIntentAssetsUnchanged', 'adapterEngineSourcesUnchanged'):
        need(value.get(key) is True, 'Actual clean native shutdown proof missing: ' + key)
    need(value.get('headCheckpointSha256') == development.FIXED['v5'] and
        value.get('servingExportSha256') == development.FIXED['rankExport'], 'Shutdown v5 identity differs')
    memory = value.get('memory', {})
    need(memory.get('processBudgetBytes') == 7 * 2 ** 30 and
        type(memory.get('mlxPeakBytes')) is int and 0 < memory['mlxPeakBytes'] <= memory['processBudgetBytes'],
        'Actual fixed7GiB native peak/shutdown receipt required')


def validate_browser(report, selected, prior_path, prior_sha):
    need(report.get('version') == BROWSER_VERSION and report.get('developmentDemo') is True and
        report.get('qualityGatePassed') is False and report.get('modelPromotion') is False and
        report.get('acceptedProfile') is False and report.get('conditioningPolicy') == runtime.POLICY and
        report.get('differentConditioningFromV2V3') is True and not report.get('error') and not report.get('teardownErrors'),
        'Completed explicit V4 development browser smoke required; model quality remains unvalidated')
    candidate = report.get('candidateSelection', {})
    need(candidate.get('expected') == selected['expected'] and candidate.get('pins', {}).get(str(prior_path)) == prior_sha,
        'Actual probe must use this exact preceding V4 startup admission')
    need(report.get('sourcesUnchanged') is True and report.get('startupUnchanged') is True and
        report.get('sourceHashes') == report.get('sourceHashesAfter'), 'Actual probe source/startup lifetime changed')
    for stage in ('healthBefore', 'healthAfter'):
        need(report.get(stage, {}).get('body', {}).get('startup') == selected['expected']['startup'],
            'Probe live startup differs from selected V4 startup')
    summary = report.get('summary', {}); attempts = report.get('attempts', [])
    need(summary.get('recorded') == summary.get('protocolPassed') == len(attempts) == 13 and
        summary.get('unattempted') == [] and {r.get('index') for r in attempts} == set(range(13)),
        'All fixed13 actual browser requests required; no success-only replacement')
    for row in attempts:
        need(row.get('protocol', {}).get('passed') is True and
            row.get('adapterHttpAudit', {}).get('protocolPassed') is True and
            row.get('adapterHttpAudit', {}).get('qualityPassed') is True,
            'Actual request protocol/public-factual audit failed')
    for stage in ('loadedSwitchFirst3', 'loadedSwitchFull6'):
        distribution = summary.get(stage, {})
        need(distribution.get('fixedDenominator') == distribution.get('valid') == 10 and
            len(distribution.get('rawMs', [])) == 10 and all(type(x) in (int, float) and 0 < x < float('inf') for x in distribution['rawMs']),
            'All ten loaded profile readiness observations required')
    functional = report.get('functionalArms', [])
    need(len(functional) == 2 and [arm.get('kind') for arm in functional] ==
        ['rapid-Bob-Maya-Bob', 'per-person-cart-and-checkout-invalidation'] and
        all(arm.get('passed') is True for arm in functional),
        'Actual rapid-switch and cart/checkout functional smoke required')
    distinct = report.get('distinctQueryAttempts', [])
    need(len(distinct) == 10 and {r.get('index') for r in distinct} == set(range(10)) and all(
        row.get('protocol', {}).get('passed') is True and
        row.get('adapterHttpAudit', {}).get('protocolPassed') is True and
        row.get('adapterHttpAudit', {}).get('qualityPassed') is True
        for row in distinct), 'All ten distinct actual query factual observations required')
    need(report.get('coldProfileHttp', {}).get('passed') is True,
        'Actual separate unknown-history HTTP/factual smoke required; no browser timing inferred')
    return {'fixedBrowserRequests': 13, 'loadedProfileReadinessPairs': 10, 'functionalArms': 2,
        'distinctQueryRequests': 10, 'unknownHistoryHttpRequests': 1,
        'factualSmokePassed': True, 'v4ConditioningQualityValidated': False}


def evidence(record, base):
    selected = development.evidence(record, base)
    annex = record.get('factorizedServingEvidence', {})
    need(annex.get('version') == VERSION and annex.get('scope') == 'technical-factual-smoke-only' and
        annex.get('v4ConditioningQualityValidated') is False, 'Explicit V4 unvalidated-quality factual annex required')
    need(record.get('sourceSha256', {}).get(str(SELF)) == factual.old.sha(SELF), 'Current factorized annex helper source pin required')
    paths = {key: factual.previous.artifact(annex[key], base) for key in ('probeAdmission', 'shutdown', 'browserProbe')}
    prior = factual.selection(paths['probeAdmission'], annex['probeAdmission']['sha256'])
    need(prior['expected'] == selected['expected'], 'Current/base probe admission startup or assets differ')
    startup = selected['expected']['startup']
    factual.validate_warm_startup(startup)
    need(startup.get('sourceTensorsHashed') == 927 and startup.get('width') == 20 and
        startup.get('rankCheckpointSha256') == development.FIXED['v5'] and
        startup.get('copyCheckpointSha256') == factual.NATIVE_COPY_SHA and
        startup.get('memory', {}).get('processBudgetBytes') == 7 * 2 ** 30, 'Actual927/source/checkpoint/full-width startup required')
    need(paths['shutdown'].parent == Path(record['startup']['file']).resolve().parent,
        'Shutdown must belong to this actual startup lifetime')
    validate_shutdown(json.loads(paths['shutdown'].read_text()))
    report = json.loads(paths['browserProbe'].read_text())
    smoke = validate_browser(report, prior, paths['probeAdmission'], annex['probeAdmission']['sha256'])
    factual.old.pins_current(report['sourceHashes'])
    selected['pins'].update(report['sourceHashes'])
    selected['pins'].update({str(path): annex[key]['sha256'] for key, path in paths.items()})
    selected.update(factorizedServingEvidenceVersion=VERSION, conditioningPolicy=runtime.POLICY,
        factualSmoke=smoke, technicalSmokePassed=True, v4ConditioningQualityValidated=False,
        actualShutdownVerified=True, scope='V4 factorized-profile technical demo; actual factual/functional smoke passed; conditioning quality unvalidated; prior model quality failed')
    return selected


def selection(receipt_path, expected_sha):
    path = Path(receipt_path).resolve()
    need(factual.old.sha(path) == factual.contract.exact_sha(expected_sha), 'Exact root-selected V4 factual admission SHA required')
    selected = evidence(json.loads(path.read_text()), path.parent)
    selected['pins'][str(path)] = expected_sha
    return selected


def build_record(prior_receipt, prior_sha, shutdown, browser_probe):
    prior_receipt = Path(prior_receipt).resolve()
    factual.selection(prior_receipt, prior_sha)  # Actual V4 startup, never an invented record.
    record = deepcopy(json.loads(prior_receipt.read_text()))
    record['sourceSha256'][str(SELF)] = factual.old.sha(SELF)
    paths = {'probeAdmission': prior_receipt, 'shutdown': Path(shutdown).resolve(), 'browserProbe': Path(browser_probe).resolve()}
    record['factorizedServingEvidence'] = {'version': VERSION, 'scope': 'technical-factual-smoke-only',
        'v4ConditioningQualityValidated': False,
        **{key: {'file': str(path), 'sha256': factual.old.sha(path)} for key, path in paths.items()}}
    return record


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(__doc__)
    for name in ('development-receipt', 'shutdown', 'browser-probe', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--development-receipt-sha256', required=True)
    args = p.parse_args(argv)
    record = build_record(args.development_receipt, args.development_receipt_sha256, args.shutdown, args.browser_probe)
    selected = evidence(record, ROOT)
    need(args.out.is_absolute() and args.out.resolve().is_relative_to(ROOT) and not args.out.exists(), 'Unique absolute current-code-root output required')
    args.out.mkdir(parents=True)
    factual.old.write(args.out / 'receipt.json', record); factual.old.write(args.out / 'verification.json', selected)
    print(json.dumps({'version': VERSION, 'technicalSmokePassed': True, 'v4ConditioningQualityValidated': False,
        'qualityGatePassed': False, 'modelPromotion': False, 'receiptSha256': factual.old.sha(args.out / 'receipt.json')}, indent=2))


if __name__ == '__main__': main()
