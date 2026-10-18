"""Exclusive fresh-query benchmark of the explicit TRAIN-ALL parallel head.

Default is CPU preflight. No fitting, feature extraction, teacher loading or
serving changes occur. Offline relevance labels are opened only after all
prediction bytes have been frozen. Native GPU execution belongs to root.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch

from . import store_parallel_text_head_v1 as model
from . import store_parallel_text_data_v1 as data
from .store_parallel_text_demo_all_v1 import load_head
from .store_parallel_text_run_v1 import (FRONTIER, ROOT, DEFAULT_MANIFEST, demo_listening,
    frontier_cases, run_request, runtime_rows, summarize, sha, dump)
from .store_catalog_v1 import canonical, load_snapshot
from .store_demo_engine import checked_request, ranking_input, guarded_order
from .store_rank_session import action_bindings

VERSION = 'parallel-word-train-all-fresh-benchmark.v1'
SOURCES = ('store_parallel_text_demo_benchmark_v1.py', 'store_parallel_text_demo_all_v1.py',
    'store_parallel_text_run_v1.py', 'store_parallel_text_head_v1.py', 'store_parallel_text_data_v1.py',
    'store_rank_session.py', 'store_demo_engine.py', 'store_demo_catalog.py', 'store_copy_fit_v2.py',
    'store_copy_wide_v2.py', 'backbone.py', 'semantic_copy_training.py')
GUARDS = (
    {'probe': 'exclusive-exact-abstention', 'request': {'query': 'Trail mug under $24'}, 'expectedCount': 0},
    {'probe': 'inclusive-exact', 'request': {'query': 'Trail mug up to $24'}, 'expectedFirst': 0},
    {'probe': 'unknown-stock', 'request': {'query': 'something in stock'}, 'expectedCount': 0},
    {'probe': 'unknown-size', 'request': {'query': 'a jacket size M'}, 'expectedCount': 0},
    {'probe': 'waterproof-declared-only', 'request': {'query': 'I need waterproof'}},
    {'probe': 'contradictory-colors', 'request': {'query': 'browse', 'hardRequirements': [
        {'attribute': 'color', 'value': 'Sage'}, {'attribute': 'color', 'value': 'Ivory'}]}, 'expectedCount': 0},
    {'probe': 'typed-budget', 'request': {'query': 'browse', 'itemBudget': 24}},
    {'probe': 'nondefault-color', 'request': {'query': 'browse', 'hardRequirements': [{'attribute': 'color', 'value': 'Cream'}]}},
)


def load_runtime(head_run):
    """Pure public feature/head export loader; no teacher packet or labels."""
    head_run = Path(head_run); manifest = json.loads((head_run / 'serving-manifest.json').read_text())
    if (manifest['scope'] != 'TRAIN-ALL public catalog demo; no heldout evaluation' or
            manifest['runtimeTeacherLookup'] is not False or manifest['all48BindingsSupervised'] is not True):
        raise ValueError('Explicit public TRAIN-ALL runtime scope required')
    head_path, contract_path = Path(manifest['headFile']), Path(manifest['publicFeatureContractFile'])
    if head_path.absolute() != (head_run / 'head.pt').absolute() or contract_path.absolute() != (head_run / 'public-runtime-features.json').absolute():
        raise ValueError('Head/public contract must belong to selected candidate directory')
    if sha(head_path) != manifest['headSha256'] or sha(contract_path) != manifest['publicFeatureContractSha256']:
        raise ValueError('Candidate head/public contract checksum changed')
    public = json.loads(contract_path.read_text()); records = public['records']
    if public['teacherOrLabelContent'] is not False or len(records) != 48:
        raise ValueError('Pure48 public feature records required')
    expected = []
    for product in load_snapshot()['inputs']['products']:
        for variant in product['colorVariants']:
            expected.append(data.public_record({'name': product['name'], 'category': product['category'],
                'itemPrice': product['itemPrice'], 'selectedColor': variant['colorName']}))
    if records != expected:
        raise ValueError('Public feature source facts differ from actual frozen catalog')
    bundle_path = Path(manifest['featureBundleFile'])
    if sha(bundle_path) != manifest['featureBundleSha256'] or public['featureBundleSha256'] != manifest['featureBundleSha256']:
        raise ValueError('Native offline feature archive checksum changed')
    bundle = torch.load(bundle_path, map_location='cpu', weights_only=True)
    feature_map = model.validate_bundle(bundle, {'featureRecords': records})
    head, saved = load_head(head_path)
    if saved['featureBundleSha256'] != manifest['featureBundleSha256']:
        raise ValueError('Head was trained with another native feature archive')
    return head, feature_map, {'manifest': manifest, 'servingManifestSha256': sha(head_run / 'serving-manifest.json'),
        'publicRecordCount': len(records), 'teacherDataRead': False, 'nativeFeatureExtractionPerformed': False}


def preflight(head_run, manifest=DEFAULT_MANIFEST):
    from .store_demo_catalog import DemoCatalogBinding
    _, _, provenance = load_runtime(head_run)
    DemoCatalogBinding(manifest).assert_current_sources(load_snapshot()['receipt'])
    return {'version': VERSION, 'nativeExecution': False, 'cases': frontier_cases(), 'guardProbes': GUARDS,
        'runtime': provenance, 'integrationManifestSha256': sha(manifest),
        'sourceSha256': {name: sha(ROOT / name) for name in SOURCES},
        'scope': 'all48catalogbindings supervised; curated32publicqueries; no heldout generalization claim',
        'inference': 'request only; outer case IDs and offline relevance labels never enter inference',
        'timerPolicy': 'fresh Qwen query+v3rank+guards+20parallel learned fields+validation+freshness+canonical first3/full6JSON; excludesstartup/offline48features/TRAIN/archives/browser/images',
        'warmupPolicy': 'one full request excluded;32freshquerytimedrequests,noresponsecache',
        'styles': 'first8publiccasepositions; samecards3explicitstyles,head-only counterfactuals',
        'guards': '8separatefreshqueryguard-onlyprobes; variableeligiblecount, no copy or paddedsixcardclaim'}


def guard_probe(engine, fixture):
    start = time.perf_counter(); request, policy = checked_request(deepcopy(fixture['request']))
    before_queries = getattr(engine.session, 'forward_calls', None)
    try:
        ranked = engine.session.rank_request(ranking_input(request), copy_adapter=engine.adapter)
    except ValueError as error:
        desired = [r['value'] for r in request['hardRequirements'] if r['attribute'] == 'color']
        no_joint_variant = not any(any(all(color['colorName'] == value for value in desired)
            for color in product['colorVariants']) for product in engine.session.products)
        if not (fixture['probe'] == 'contradictory-colors' and fixture.get('expectedCount') == 0 and
                len(set(desired)) > 1 and no_joint_variant and
                str(error) == 'No single declared variant satisfies every hard color requirement'):
            raise
        # The legacy rank session binds actions before this demo's stronger
        # joint-variant filter. Its explicit rejection is a safe abstention,
        # not a successful ranked payload. Retain the exact observed error.
        after_queries = getattr(engine.session, 'forward_calls', None)
        return {'probe': fixture['probe'], 'status': 'expected-safe-rejection',
            'input': deepcopy(fixture['request']), 'ranked': None, 'order': [], 'cards': [],
            'expectedSafeError': {'type': type(error).__name__, 'message': str(error)},
            'checks': {'distinctHardColors': True, 'noJointDeclaredVariant': no_joint_variant,
                       'rankSessionRefusedActions': True}, 'passed': True,
            'nativeQueryForwardCounterDelta': after_queries - before_queries if type(before_queries) is int and type(after_queries) is int else None,
            'rankThroughJsonMs': None, 'guardRejectionWallMs': (time.perf_counter() - start) * 1000,
            'copyGenerationPerformed': False,
            'scope': 'separate guard-only expected ValueError/abstention; no ranked JSON or generated-copy readiness claim'}
    order, guards = guarded_order(ranked['scores16'], request, policy, engine.session.products)
    cards = action_bindings(request, order[:6], engine.session.snapshot)
    checks = {'expectedCount': fixture.get('expectedCount') is None or len(order) == fixture['expectedCount'],
        'expectedFirst': fixture.get('expectedFirst') is None or bool(order) and order[0] == fixture['expectedFirst'],
        'budget': request['itemBudget'] is None or all(c['facts']['itemPrice'] <= request['itemBudget'] for c in cards),
        'declaredColor': all(all(c['facts']['selectedColor'] == r['value'] for r in request['hardRequirements'] if r['attribute'] == 'color') for c in cards)}
    if policy['waterproofRequired']:
        checks['waterproofDeclared'] = all('waterproof' in ' '.join([engine.session.products[i]['name'], engine.session.products[i]['category'], *engine.session.products[i]['searchTerms']]).casefold() for i in order)
    canonical({'order': order, 'cards': cards, 'guards': guards})
    return {'probe': fixture['probe'], 'input': deepcopy(fixture['request']), 'ranked': ranked,
        'order': order, 'cards': cards, 'guards': guards, 'checks': checks, 'passed': all(checks.values()),
        'rankThroughJsonMs': (time.perf_counter() - start) * 1000, 'copyGenerationPerformed': False,
        'scope': 'separate variable-count guard-only probe; no padding, above-fold, generated-copy or six-card benchmark claim'}


def relevance_after_freeze(out, roster):
    out = Path(out)
    freeze_path = out / 'PREDICTIONS_FROZEN.json'
    if not freeze_path.is_file():
        raise ValueError('Freeze prediction bytes before reading any relevance labels')
    frozen = json.loads(freeze_path.read_text())
    for file, expected in frozen['files'].items():
        if sha(out / file) != expected:
            raise ValueError('Prediction changed after freeze')
    manifest = json.loads((FRONTIER / 'manifest.json').read_text())
    label_path = FRONTIER / 'offline-relevance-labels.json'
    if sha(label_path) != manifest['files']['offline-relevance-labels.json']:
        raise ValueError('Frozen offline relevance file changed')
    labels = {r['caseId']: r['expectedTopProductIndex'] for r in json.loads(label_path.read_text())}
    rows = []
    for record in roster:
        if record['status'] == 'completed-attempt':
            result = json.loads((out / record['file']).read_text()); expected = labels[record['caseId']]
            rows.append({'caseId': record['caseId'], 'expectedTop': expected, 'predictedTop': result['order'][0],
                'topOne': result['order'][0] == expected, 'inTopSix': expected in result['order'][:6]})
    return {'topOne': sum(r['topOne'] for r in rows), 'inTopSix': sum(r['inTopSix'] for r in rows),
        'completedDenominator': len(rows), 'plannedDenominator': 32, 'rows': rows,
        'labelsReadAfterPredictionFreeze': True, 'freezeSha256': sha(freeze_path),
        'scope': 'author-curated known-catalog development labels; no human usefulness/conversion or unseen generalization claim'}


def run_benchmark(out, head_run, *, run=False, manifest=DEFAULT_MANIFEST, _dependencies=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Unused output required; preserve every candidate')
    deps = {} if _dependencies is None else _dependencies
    listening = deps.get('service_listening', demo_listening)
    if run and listening():
        raise RuntimeError('8766listening; refuse native benchmark without stopping or bypassing it')
    prepared = deps.get('preflight', preflight)(head_run, manifest)
    out.mkdir(parents=True); dump(out / 'preflight.json', prepared)
    if not run:
        return {'status': 'cpu-preflight-only', 'nativeExecution': False}
    if _dependencies is None:
        from .catalog_fusion.generalization_v2 import native_owner_lock
        from .store_demo_engine import ResidentDemoEngine
        lock = lambda: native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'), 'root')
        factory = lambda: ResidentDemoEngine(output=out / 'resident', width=10, integration_manifest=manifest)
    else:
        lock, factory = deps['native_lock'], deps['engine_factory']
    report = {'version': VERSION, 'status': 'started', 'nativeExecution': True, 'runtime': prepared['runtime'],
        'scope': prepared['scope'], 'requests': [{'caseId': c['caseId'], 'input': c['request'],
            'file': f'request-{i:03d}.json', 'status': 'unattempted'} for i, c in enumerate(prepared['cases'])],
        'styleCounterfactuals': [], 'guardProbes': [{'file': f'guard-{i:02d}.json', 'probe': fixture['probe'],
            'status': 'unattempted', 'passed': False} for i, fixture in enumerate(prepared['guardProbes'])]}
    receipts = []; engine = None; started = time.perf_counter()
    def save():
        report['summary'] = summarize(receipts, report['requests']); dump(out / 'results.json', report)
    save()
    try:
        with ExitStack() as ownership:
            ownership.enter_context(lock())
            try:
                if listening():
                    raise RuntimeError('8766became active before load; refuse without automatic stop')
                engine = factory()
                if engine.startup.get('sourceTensorsHashed') != 927:
                    raise ValueError('Verified927 frozen source tensors required')
                head, feature_map, runtime = deps.get('load_runtime', load_runtime)(head_run)
                if runtime != prepared['runtime']:
                    raise ValueError('Runtime provenance changed since preflight')
                report['startup'] = engine.startup; save()
                from .store_copy_fit_v2 import deadline
                requester = deps.get('requester', run_request)
                with model.bounded_cpu_threads(1):
                    report['activeStage'] = 'unscored-warmup'; save()
                    try:
                        with deadline(30, 'TRAIN-ALL parallel complete warmup'):
                            warmup = requester(engine, head, feature_map, deepcopy(prepared['cases'][0]['request']))
                        dump(out / 'warmup.json', warmup)
                    except BaseException as error:
                        dump(out / 'warmup-failure.json', {'status': 'failed-unscored-warmup', 'error': {'type': type(error).__name__, 'message': str(error)}})
                        raise
                    for index, case in enumerate(prepared['cases']):
                        record = report['requests'][index]; record['status'] = 'running'
                        report['activeStage'] = 'timed-request'; report['activeCase'] = case['caseId']; save()
                        attempt = time.perf_counter(); path = out / record['file']
                        try:
                            with deadline(30, 'TRAIN-ALL parallel complete fresh request'):
                                result = requester(engine, head, feature_map, deepcopy(case['request']))
                        except BaseException as error:
                            failed = {'status': 'failed-attempt', 'input': deepcopy(case['request']), 'archiveCaseId': case['caseId'],
                                'failedAtMs': (time.perf_counter() - attempt) * 1000, 'error': {'type': type(error).__name__, 'message': str(error)}}
                            dump(path, failed); record.update(status='failed-attempt', error=failed['error'], sha256=sha(path)); save(); raise
                        receipts.append(result); dump(path, result)
                        record.update(status='completed-attempt', sha256=sha(path), timing=result['timing'], selectedProductIndices=result['order'][:6])
                        if index < 8:
                            variants = []
                            for style in data.STYLES:
                                rows = runtime_rows(result['cards'], style)
                                generated = model.generate(head, model.inference_rows(rows), feature_map)
                                variants.append({'publicStyle': style, 'rows': rows, 'generated': generated})
                            file = out / f'style-{index:03d}.json'; dump(file, {'samePublicSourceCards': result['cards'], 'variants': variants,
                                'scope': 'explicit style counterfactual on same cached sourcecards; head-only timing, not a freshquery request'})
                            texts = [[f['text'] for f in v['generated']['fields']] for v in variants]
                            report['styleCounterfactuals'].append({'file': file.name,
                                'changedFieldsPlainVsBrowse': sum(a != b for a, b in zip(texts[0], texts[1])),
                                'changedFieldsPlainVsCompact': sum(a != b for a, b in zip(texts[0], texts[2]))})
                        save(); print(canonical({'event': 'train-all-fresh-request', 'completed': index + 1, 'total': 32, 'fullMs': result['timing']['attemptFullSixMs']}), flush=True)
                    for index, fixture in enumerate(prepared['guardProbes']):
                        report['activeStage'] = 'guard-only'; report['activeGuard'] = fixture['probe']
                        report.pop('activeCase', None)
                        record = report['guardProbes'][index]; record['status'] = 'running'; save()
                        try:
                            with deadline(30, 'TRAIN-ALL separate guard-only fresh request'):
                                result = deps.get('guard_probe', guard_probe)(engine, fixture)
                        except BaseException as error:
                            failed = {'probe': fixture['probe'], 'status': 'failed-guard-attempt', 'input': fixture['request'],
                                'error': {'type': type(error).__name__, 'message': str(error)}}
                            dump(out / record['file'], failed)
                            record.update(status='failed-guard-attempt', error=failed['error']); save(); raise
                        file = out / f'guard-{index:02d}.json'; dump(file, result)
                        record.update(status=result.get('status', 'completed-guard'), passed=result['passed'], sha256=sha(file))
                        save()
                if len(receipts) != 32 or len(report['guardProbes']) != 8 or any(g['status'] in ('unattempted', 'running', 'failed-guard-attempt') for g in report['guardProbes']):
                    raise ValueError('Fixed32request/8guard workload incomplete')
                engine.binding.assert_current_sources(engine.session.snapshot['receipt'])
                if deps.get('source_hashes', lambda: {n: sha(ROOT / n) for n in SOURCES})() != prepared['sourceSha256']:
                    raise ValueError('Source changed during candidate benchmark')
                files = {p.name: sha(p) for p in out.iterdir() if p.name.startswith(('request-', 'style-', 'guard-')) and p.suffix == '.json'}
                dump(out / 'PREDICTIONS_FROZEN.json', {'version': VERSION, 'files': files,
                    'headSha256': runtime['manifest']['headSha256'], 'offlineRelevanceLabelsRead': False})
                report['offlineRelevance'] = deps.get('relevance', relevance_after_freeze)(out, report['requests'])
                report['status'] = 'completed-train-all-fresh-benchmark-not-promoted'
                report.pop('activeStage', None); report.pop('activeCase', None); report.pop('activeGuard', None)
            finally:
                if engine is not None:
                    report['shutdownIntegrity'] = engine.close()
                    if not all(report['shutdownIntegrity'].get(k) is True for k in ('frozen927HashesUnchanged', 'copyCheckpointUnchanged')):
                        report['status'] = 'failed-shutdown-integrity'
    except BaseException as error:
        report.update(status='failed-benchmark', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        report['totalWallMsIncludingStartupArchiveOfflineScoring'] = (time.perf_counter() - started) * 1000; save()
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--head-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    with model.bounded_cpu_threads(1):
        result = run_benchmark(args.out, args.head_run, run=args.run, manifest=args.manifest)
    print(canonical({'status': result['status'], 'nativeExecution': result['nativeExecution'], 'summary': result.get('summary')}))
    if args.run and result['status'] != 'completed-train-all-fresh-benchmark-not-promoted':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
