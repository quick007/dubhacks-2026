"""Prepare or run an exclusive counterbalanced fast-parallel host profile.

Default is CPU preparation. --run is required;8766 must already be closed by its
owner. No live hooks, automatic stop, changed sampling, GPU sync or text cache.
"""
from __future__ import annotations
import argparse
import ast
import re
from contextlib import ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import time
import uuid

from .store_fast_path_profile_v1 import measure_request, VERSION
from .store_candidate_engine_v1 import load_parallel_assets
from .store_rank_session_v5 import read_export, make_v5_rank_session_factory, sha
from .store_demo_engine import checked_request, guarded_order
from .store_catalog_v1 import load_snapshot

HERE = Path(__file__).resolve().parent
SOURCES = ('store_fast_path_profile_v1.py', 'store_fast_path_profile_run_v1.py',
    'store_candidate_image_service_v1.py', 'store_candidate_engine_v1.py', 'store_demo_engine.py',
    'store_demo_service.py', 'store_demo_image_binding_v1.py', 'store_demo_catalog.py', 'store_rank_session.py',
    'store_rank_session_v5.py', 'store_function_adaptation.py', 'store_parallel_text_head_v1.py',
    'store_parallel_text_data_v1.py', 'store_parallel_text_demo_all_v1.py', 'catalog_fusion/runtime.py',
    'catalog_fusion/inputs.py', 'backbone.py', 'store_candidate_browser_probe_v1.mjs',
    'store_copy_fit_v2.py', 'semantic_copy_training.py', 'store_copy_wide_v2.py', 'store_copy_data_v2.py')
BROWSER_QUERIES = HERE / 'store_candidate_browser_probe_v1.mjs'


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def public_cases(pairs=10):
    if pairs not in (10, 20):
        raise ValueError('Fixed10 or20 paired known-public requests required')
    match = re.search(r'export const QUERIES = Object\.freeze\((\[[\s\S]*?\])\);', BROWSER_QUERIES.read_text())
    if match is None:
        raise ValueError('Frozen existing browser QUERIES literal required')
    queries = ast.literal_eval(match[1])
    if type(queries) is not list or len(queries) != 4 or any(type(q) is not str or not q for q in queries):
        raise ValueError('Exactly four existing public browser requests required')
    cases = []; products = load_snapshot()['inputs']['products']
    for index in range(pairs):
        qi, cycle = index % 4, index // 4
        request = {'query': queries[qi]}; validated, policy = checked_request(request)
        order, _ = guarded_order([0.] * 16, validated, policy, products)
        if len(order) < 6:
            raise ValueError('Known public request cannot supply unchanged6-card workload')
        cases.append({'caseId': f'browser-{qi}-repeat-{cycle}', 'request': request,
            'group': f'public-query-{qi}', 'armOrder': ['A-original', 'B-profile'] if (qi + cycle) % 2 == 0 else ['B-profile', 'A-original']})
    return cases


def prepare(manifest, candidate_run, rank_export, rank_export_sha, pairs=10):
    from .store_demo_image_binding_v1 import ImageDeliveryCatalogBinding
    binding = ImageDeliveryCatalogBinding(manifest); snapshot = load_snapshot()
    binding.assert_current_sources(snapshot['receipt'])
    _, _, records, copy_metadata = load_parallel_assets(candidate_run)
    export = read_export(rank_export, rank_export_sha)
    return {'version': VERSION + '/run', 'nativeExecution': False, 'cases': public_cases(pairs),
        'publicQuerySource': str(BROWSER_QUERIES), 'publicQuerySourceSha256': sha(BROWSER_QUERIES),
        'distinctPublicQueries': 4, 'pairedRequests': pairs,
        'integrationManifest': str(manifest), 'integrationManifestSha256': sha(manifest),
        'candidateRun': str(candidate_run), 'rankExport': str(rank_export), 'rankExportSha256': rank_export_sha,
        'rankHeadSha256': export['head']['sha256'], 'rankFeatureBase': export['sourceBaseRun'],
        'parallelCopyAssets': copy_metadata, 'publicFactRecords': len(records),
        'sourceSha256': {name: sha(HERE / name) for name in SOURCES},
        'warmup': 'one A then one B on first complete public query, excluded; same persistent serving allocator/native policy, no per-request clear',
        'sampling': 'unchanged fresh Qwen B1 query + learned rank + one fresh20-field parallel word-head forward; no cached response text',
        'admissionScope': 'standalone immediate worker admission; zero HTTP queue, includes original schema/hard-policy and request-source checks',
        'profileAccounting': 'nested inclusive/exclusive host times, clipped at complete NDJSON serialization; existing bb.encode completion synchronized by original code, no added sync',
        'pairing': 'same queries A(original)/B(profile), alternating AB/BA across full frozen roster; exact outputs and word-slot IDs required',
        'limits': ['server worker critical path only; no network, browser, image or physical-paint evidence',
            'observer overhead measured from paired A/B; phases not automatically corrected or summed as inclusive',
            '4existing public browser queries repeated10/20 pairs; development profiling, not heldout population, generic-UI baseline or the separate cohort experiment']}


def assert_external_pins(preflight):
    if sha(preflight['integrationManifest']) != preflight['integrationManifestSha256']:
        raise ValueError('Frozen integration manifest changed')
    if sha(preflight['rankExport']) != preflight['rankExportSha256']:
        raise ValueError('Frozen v5 export changed')
    if sha(Path(preflight['rankExport']).parent / 'learned.pt') != preflight['rankHeadSha256']:
        raise ValueError('Frozen v5 head changed')
    if any(sha(path) != checksum for path, checksum in preflight['parallelCopyAssets']['assetSha256'].items()):
        raise ValueError('Frozen parallel copy assets changed')


def quantiles(values):
    if not values:
        return {'count': 0, 'medianMs': None, 'p95Ms': None}
    values = sorted(values)
    # Explicit nearest-rank p95 avoids interpolating an unobserved small sample.
    return {'count': len(values), 'medianMs': statistics.median(values),
        'p95Ms': values[max(0, math.ceil(.95 * len(values)) - 1)], 'minMs': values[0], 'maxMs': values[-1]}


def summarize(records, cases):
    completed = [r for r in records if r.get('status') == 'completed']
    result = {'plannedArms': len(cases) * 2, 'completedArms': len(completed),
        'failedArms': sum(r.get('status') == 'failed' for r in records),
        'unattemptedArms': len(cases) * 2 - len(records), 'byGroup': {}, 'pairedObserverDeltaMs': [],
        'exactParityPairs': 0, 'completePairs': 0, 'phasesExclusiveThroughFullSix': {}}
    for group in sorted({c.get('group', 'unclassified') for c in cases}):
        group_rows = [r for r in completed if r.get('group', 'unclassified') == group]
        result['byGroup'][group] = {arm: quantiles([r['timing']['fullSixWireReadyMs'] for r in group_rows if r['arm'] == arm])
            for arm in ('A-original', 'B-profile')}
    for case in cases:
        paired = {r['arm']: r for r in completed if r['caseId'] == case['caseId']}
        if len(paired) != 2:
            continue
        result['completePairs'] += 1
        same = paired['A-original']['parity']['sha256'] == paired['B-profile']['parity']['sha256']
        result['exactParityPairs'] += same
        a, b = paired['A-original']['timing']['fullSixWireReadyMs'], paired['B-profile']['timing']['fullSixWireReadyMs']
        result['pairedObserverDeltaMs'].append({'caseId': case['caseId'], 'group': case.get('group', 'unclassified'),
            'deltaMs': b - a, 'relativePercent': 100 * (b - a) / a if a else None, 'exactParity': same})
    result['observerOverhead'] = quantiles([r['deltaMs'] for r in result['pairedObserverDeltaMs']])
    for row in completed:
        if row['arm'] != 'B-profile':
            continue
        for phase, value in row['profile']['throughFullSixWire']['phases'].items():
            result['phasesExclusiveThroughFullSix'].setdefault(phase, []).append(value['exclusiveMs'])
    result['phasesExclusiveThroughFullSix'] = {k: quantiles(v) for k, v in result['phasesExclusiveThroughFullSix'].items()}
    result['comparisonScope'] = 'All arms use the same fast model; A/B measures instrumentation overhead, never generic UI latency'
    return result


def run(out, manifest, candidate_run, rank_export, rank_export_sha, *, pairs=10, execute=False, _dependencies=None):
    out = Path(out); deps = _dependencies or {}
    if out.exists():
        raise FileExistsError('Unused output required; preserve all prior attempts')
    preflight = deps.get('prepare', prepare)(manifest, candidate_run, rank_export, rank_export_sha, pairs)
    out.mkdir(parents=True); dump(out / 'preflight.json', preflight)
    if not execute:
        return {'status': 'cpu-prepared-only', 'nativeExecution': False, 'preflightSha256': sha(out / 'preflight.json')}
    from .store_ar_comparison_run_v1 import demo_listening
    from .catalog_fusion.generalization_v2 import native_owner_lock
    from .store_copy_fit_v2 import deadline
    from .store_fast_path_profile_v1 import ProfileCandidateEngine
    listening = deps.get('service_listening', demo_listening)
    if listening():
        raise RuntimeError('Demo8766 is listening; no stop, auto-kill or live attach permitted')
    factory = deps.get('factory', lambda: ProfileCandidateEngine(mode='parallel', candidate_run=candidate_run,
        integration_manifest=manifest, output=out / 'engine', width=10,
        rank_session_factory=make_v5_rank_session_factory(rank_export, rank_export_sha)))
    lock = deps.get('lock', lambda: native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'), 'root'))
    measure = deps.get('measure', measure_request); engine = None
    ownership = ExitStack()
    report = {'version': VERSION + '/run', 'status': 'running', 'nativeExecution': True,
        'preflightSha256': sha(out / 'preflight.json'), 'warmups': [], 'records': []}
    def save():
        report['summary'] = summarize(report['records'], preflight['cases']); dump(out / 'results.json', report)
    def one(case, arm, warmup=False):
        record = {'caseId': case['caseId'], 'arm': arm, **{k: case[k] for k in ('group', 'pairId') if k in case}}
        request_id = str(uuid.uuid4()); record['requestId'] = request_id
        try:
            with deadline(60, 'complete unchanged fast-parallel profile request'):
                value = measure(engine, deepcopy(case['request']), request_id, instrumented=arm == 'B-profile')
            return {**record, **value, 'warmupExcluded': warmup}
        except BaseException as error:
            record.update(status='failed', error={'type': type(error).__name__, 'message': str(error)},
                timing=getattr(error, 'profile_failure_timing', {}), partialProfile=getattr(error, 'profile_partial_trace', None))
            (report['warmups'] if warmup else report['records']).append(record); save(); raise
    save()
    try:
        ownership.enter_context(lock())
        if listening():
            raise RuntimeError('Demo8766 became active under owner lock; refusing native construction')
        for name, checksum in preflight['sourceSha256'].items():
            if sha(HERE / name) != checksum:
                raise ValueError('Frozen profile source changed: ' + name)
            target = out / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((HERE / name).read_bytes())
        deps.get('assert_external_pins', assert_external_pins)(preflight)
        engine = factory(); report['startupExcluded'] = engine.startup; save()
        for arm in ('A-original', 'B-profile'):
            report['warmups'].append(one(preflight['cases'][0], arm, True)); save()
        for case in preflight['cases']:
            for arm in case['armOrder']:
                report['records'].append(one(case, arm)); save()
            pair = report['records'][-2:]
            if pair[0]['parity']['sha256'] != pair[1]['parity']['sha256']:
                raise AssertionError('Exact rank/copy text/slot-ID/validation parity failed')
        deps.get('assert_external_pins', assert_external_pins)(preflight)
        report['status'] = 'completed-profile-not-optimized'
    except BaseException as error:
        report.update(status='failed-profile', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        try:
            if engine is not None:
                try:
                    report['shutdownIntegrity'] = engine.close()
                    required = ('frozen927HashesUnchanged', 'copyCheckpointUnchanged', 'v5ServingExportUnchanged', 'v5HeadCheckpointUnchanged')
                    if not all(report['shutdownIntegrity'].get(key) is True for key in required):
                        raise AssertionError('927/adapter/v5 export shutdown integrity not confirmed')
                except BaseException as error:
                    report.update(status='failed-profile', shutdownError={'type': type(error).__name__, 'message': str(error)})
            if any(sha(HERE / name) != checksum for name, checksum in preflight['sourceSha256'].items()):
                report.update(status='failed-profile', sourceIntegrityError='Profile source changed during run')
            save()
        finally:
            ownership.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--integration-manifest', type=Path, required=True); parser.add_argument('--candidate-run', type=Path, required=True)
    parser.add_argument('--rank-head-export', type=Path, required=True); parser.add_argument('--rank-head-export-sha256', required=True)
    parser.add_argument('--pairs', type=int, choices=(10, 20), default=10); parser.add_argument('--run', action='store_true'); args = parser.parse_args()
    result = run(args.out, args.integration_manifest, args.candidate_run, args.rank_head_export,
        args.rank_head_export_sha256, pairs=args.pairs, execute=args.run)
    print(json.dumps({'status': result['status'], 'nativeExecution': result['nativeExecution']}))
    return int(result['status'].startswith('failed'))


if __name__ == '__main__':
    raise SystemExit(main())
