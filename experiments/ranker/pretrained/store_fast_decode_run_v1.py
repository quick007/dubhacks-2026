"""Isolated width20+immutable-root versus width10 fresh-root comparison.

CPU preflight by default; root owner must explicitly request --run after the
service is stopped. All20fields/sixcards stay native and fresh. No live edit,
automatic service stop or generated-text cache. Default7GiB; explicit8GiB is
applied only after default initialization and recorded real-headroom checks.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch
import uuid

from .store_ar_comparison_run_v1 import DEFAULT_MANIFEST, cpu_preflight, demo_listening, fixed_cases, reset_runtime
from .store_autoregressive_baseline import paired_hybrid_readout
from .store_copy_data_v2 import CAPS, FIELDS
from .store_demo_performance_run_v1 import token_signature
from .store_fast_decode_v1 import ImmutableRootCache

ROOT = Path(__file__).resolve().parent
BASELINE = 'width10-fresh-root'
CANDIDATE = 'width20-immutable-root'
ORDER = ((BASELINE, CANDIDATE), (CANDIDATE, BASELINE), (BASELINE, CANDIDATE))
GIB = 2 ** 30


def validate_budget(memory_gib):
    if type(memory_gib) is not int or memory_gib not in (7, 8):
        raise ValueError('Explicit memory_gib must be7 or8; default7, never automatic')
    return memory_gib * GIB


def system_memory_snapshot():
    import psutil
    memory = psutil.virtual_memory(); swap = psutil.swap_memory()
    return {'totalBytes': memory.total, 'availableBytes': memory.available,
        'percentUsed': memory.percent, 'swapUsedBytes': swap.used, 'swapFreeBytes': swap.free}


def require_headroom(memory_gib, snapshot, *, stage, active_bytes=0):
    budget = validate_budget(memory_gib)
    if stage not in ('before-load', 'after-default-initialization'):
        raise ValueError('Explicit memory inspection stage required')
    if (type(snapshot.get('availableBytes')) is not int or type(snapshot.get('totalBytes')) is not int
            or snapshot['availableBytes'] < 0 or snapshot['totalBytes'] <= 0
            or type(active_bytes) is not int or active_bytes < 0):
        raise ValueError('Real nonnegative byte memory snapshot required')
    if memory_gib == 8:
        required = budget + 2 * GIB if stage == 'before-load' else max(0, budget - active_bytes) + 2 * GIB
    else:
        required = 6 * GIB if stage == 'before-load' else 512 * 2 ** 20
    if snapshot['availableBytes'] < required:
        raise MemoryError(f'Insufficient real headroom at{stage}: available{snapshot["availableBytes"]}bytes, '
            f'required{required}bytes for explicit{memory_gib}GiB experiment; budget not automatically changed')
    return {'stage': stage, 'snapshot': deepcopy(snapshot), 'requiredAvailableBytes': required,
        'requestedMemoryGiB': memory_gib, 'requestedBudgetBytes': budget,
        'activeNativeBytesAtInspection': active_bytes, 'passed': True,
        'policy': '8GiB:2GiB system reserve beyond maximum additional native allocation;7GiB:existing6GiB preload/512MiB runtime floors'}


def configure_experimental_budget(engine, memory_gib, snapshot):
    """Apply explicit caller configuration after unchanged default7GiB init."""
    budget = validate_budget(memory_gib); bb = engine.bb
    before = bb.runtime_memory()
    if bb.memory_budget != 7 * GIB or before['processBudgetBytes'] != 7 * GIB:
        raise ValueError('Expected verified unchanged default7GiB engine initialization')
    if max(before['mlxActiveBytes'], before['mlxPeakBytes']) > 7 * GIB:
        raise MemoryError('Default engine already exceeded7GiB before explicit reconfiguration')
    check = require_headroom(memory_gib, snapshot, stage='after-default-initialization', active_bytes=before['mlxActiveBytes'])
    previous_allocator_limit = None
    if memory_gib != 7:
        previous_allocator_limit = bb.mx.set_memory_limit(budget)
        bb.memory_budget = budget
    bb.check_memory()
    after = bb.runtime_memory()
    if after['processBudgetBytes'] != budget or max(after['mlxActiveBytes'], after['mlxPeakBytes']) > budget:
        raise MemoryError('Configured native active/peak memory differs from explicit budget')
    receipt = {'initialConstructorMemoryGiB': 7, 'requestedMemoryGiB': memory_gib,
        'configuredBudgetBytes': budget, 'budgetRaisedExplicitly': memory_gib == 8,
        'automaticIncrease': False, 'before': before, 'after': after, 'headroomCheck': check,
        'allocatorPreviousLimitBytes': previous_allocator_limit}
    engine.startup['initialDefaultInitializationMemory'] = deepcopy(before)
    engine.startup['memory'] = deepcopy(after)
    engine.startup['experimentalMemoryConfiguration'] = deepcopy(receipt)
    return receipt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(out, *, native=False, manifest=DEFAULT_MANIFEST, memory_gib=7, _dependencies=None):
    budget = validate_budget(memory_gib)
    out = Path(out); deps = {} if _dependencies is None else _dependencies
    if out.exists():
        raise FileExistsError('Unused output required; preserve prior candidate results')
    if native and deps.get('service_listening', demo_listening)():
        raise RuntimeError('Demo8766 still listening; do not stop/bypass it automatically')
    preflight = deepcopy(deps.get('preflight', cpu_preflight)(manifest))
    preflight.update(version='store-fast-decode-comparison.v1', pairedOrder=[list(order) for order in ORDER],
        memoryGiB=memory_gib, configuredBudgetBytes=budget, defaultConstructorMemoryGiB=7,
        memoryIncreaseExplicit=memory_gib == 8, automaticMemoryIncrease=False,
        memoryHeadroomPolicy='explicit8GiB requires10GiBavailable beforeload and(8GiB-active)+2GiB afterdefaultinit; default7preservesexistingfloors',
        candidateSourceSha256=sha(ROOT / 'store_fast_decode_v1.py'), runnerSourceSha256=sha(__file__),
        textFreshness='all20field values freshly natively sampled each request; prompt-state reuse only',
        scope='resident fresh query through full validated serialized6card/20field copy; no browser',
        parityPolicy='same-width fork/rebuild/repeat controls; no forced cross-width token equality; same-width cached miss/hit/A-B-A token equality')
    out.mkdir(parents=True)
    (out / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    if not native:
        return {'status': 'cpu-preflight-only', 'nativeExecution': False, 'preflight': preflight}
    if _dependencies is None:
        from .catalog_fusion.generalization_v2 import native_owner_lock
        from .store_demo_engine import ResidentDemoEngine
        lock = lambda: native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'), 'root')
        factory = lambda: ResidentDemoEngine(output=out / 'engine', width=10, memory_gib=7, integration_manifest=manifest)
    else:
        lock = deps['lock']; factory = deps['factory']
    clock = deps.get('clock', time.perf_counter); reset = deps.get('reset', reset_runtime)
    cache_factory = deps.get('cache_factory', ImmutableRootCache.from_engine)
    report = {'version': preflight['version'], 'status': 'started', 'nativeExecution': True,
        'preflightSha256': sha(out / 'preflight.json'), 'records': [], 'controls': [],
        'cacheParity': [], 'baselineWidth': 10, 'candidateWidth': 20,
        'crossWidthTokenEqualityRequired': False, 'generatedTextReused': False,
        'memoryGuardGiB': memory_gib, 'memoryGuardBytes': budget,
        'memoryGuardRaised': memory_gib == 8, 'memoryGuardRaisedAutomatically': False}
    engine = None; cache = None; ownership = ExitStack(); begin = clock()
    def save():
        summaries = {}
        for mode in (BASELINE, CANDIDATE):
            all_rows = [row for row in report['records'] if row['mode'] == mode and not row['excludedWarmupOrControl']]
            rows = [row for row in all_rows if row['status'] == 'completed']
            def median(key):
                values = [row[key] for row in rows if row.get(key) is not None]
                return statistics.median(values) if values else None
            summaries[mode] = {'attempts': len(all_rows), 'completed': len(rows),
                'cardCounts': [row['cardCount'] for row in rows], 'fieldCounts': [row['requiredFields'] for row in rows],
                'firstThreeAttemptMs': median('firstThreeAttemptMs'), 'fullAttemptMs': median('fullAttemptMs'),
                'usableFirstThreeMs': median('usableFirstThreeMs'), 'usableFullMs': median('usableFullMs'),
                'fullUsableProxyPasses': sum(row.get('usableFullMs') is not None for row in rows),
                'acceptedFields': sum(row['acceptedFields'] for row in rows),
                'requiredFields': sum(row['requiredFields'] for row in rows),
                'allEOS': all(row['allEOS'] for row in rows), 'actualSampledTokens': [row['actualSampledTokens'] for row in rows]}
        report['summary'] = summaries
        (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    def one(case, mode, *, excluded=False):
        reset(engine); engine.width = 10 if mode == BASELINE else 20
        request_id = str(uuid.uuid4()); events = []; path = Path(engine.output) / 'requests' / request_id / 'receipt.json'
        row = {'caseId': case['caseId'], 'mode': mode, 'excludedWarmupOrControl': excluded,
            'requestId': request_id, 'resultPath': str(path), 'status': 'running'}
        report['records'].append(row); save(); started = clock()
        try:
            with ExitStack() as stack:
                if mode == CANDIDATE:
                    stack.enter_context(patch('experiments.ranker.pretrained.branch_local_copy.prepare_root', cache.prepare_root))
                from .store_copy_fit_v2 import deadline
                with deadline(120, 'full native fast-decode candidate request'):
                    engine.personalize(case['request'], request_id, started, events.append, lambda: False)
            value = json.loads(path.read_text()); endpoints = {event['type']: event for event in value['events']}
            rows = value['rawRows']
            if (len(value['cards']) != 6 or len(rows) != 20 or tuple(row['fields'] for row in rows) != tuple([f] for f in FIELDS)
                    or sum(row['maxTokens'] for row in rows) != 416
                    or any(row['maxTokens'] != CAPS[row['fields'][0][0]] for row in rows)
                    or value['decodeTiming']['width'] != engine.width):
                raise ValueError('Must preserve complete6cards/20fields/originalcaps and actual declared width')
            paired = paired_hybrid_readout(path, engine.session.snapshot)
            evaluated = {endpoint['type']: endpoint for endpoint in paired['endpoints']}
            row.update(status=value['status'], cardCount=len(value['cards']),
                acceptedFields=endpoints['complete']['acceptedFields'], requiredFields=endpoints['complete']['requiredFields'],
                rankMs=endpoints['rank']['timing']['readyMs'],
                firstThreeAttemptMs=endpoints['firstThree']['timing']['readyMs'],
                fullAttemptMs=endpoints['complete']['timing']['readyMs'],
                usableFirstThreeMs=evaluated['firstThree']['usableReadyMs'], usableFullMs=evaluated['complete']['usableReadyMs'],
                preparation=value['preparation'], decodeTiming=value['decodeTiming'], memory=value['memory'],
                allEOS=all(row['finishReason'] == 'stop' for row in rows),
                actualSampledTokens=sum(row['arSteps'] for row in rows),
                actionSignatureSha256=hashlib.sha256(json.dumps(value['cards'], sort_keys=True).encode()).hexdigest())
            cache.assert_immutable()
            return value
        except BaseException as error:
            row.update(status='failed-attempt', error={'type': type(error).__name__, 'message': str(error)},
                failedAtMs=(clock() - started) * 1000)
            raise
        finally:
            row['outerWallIncludingArchiveMs'] = (clock() - started) * 1000; save()
    save()
    try:
        ownership.enter_context(lock())
        if deps.get('service_listening', demo_listening)():
            raise RuntimeError('Service became active before candidate load')
        snapshot_fn = deps.get('memory_snapshot', system_memory_snapshot)
        report['systemMemoryBeforeLoad'] = snapshot_fn()
        report['preloadHeadroomCheck'] = require_headroom(memory_gib, report['systemMemoryBeforeLoad'], stage='before-load')
        engine = factory()
        report['startupAtDefaultInitialization'] = deepcopy(engine.startup)
        report['systemMemoryBeforeBudgetConfiguration'] = snapshot_fn()
        report['memoryConfiguration'] = configure_experimental_budget(engine, memory_gib, report['systemMemoryBeforeBudgetConfiguration'])
        cache = cache_factory(engine); report['startup'] = deepcopy(engine.startup); save()
        if _dependencies is None:
            from .store_rank_session import action_bindings
            from .store_demo_engine import NEUTRAL_COPY_REQUEST
            from .store_copy_wide_v2 import native_controls
            from .store_copy_fit_v2 import deadline
            request = {'query': 'browse', 'preferences': [], 'itemBudget': None, 'hardRequirements': []}
            bound = {'request': NEUTRAL_COPY_REQUEST,
                'cards': action_bindings(request, list(range(6)), engine.session.snapshot)}
            for width in (10, 20):
                control_started = clock()
                with deadline(180, 'same-width fast-decode cache fork/rebuild/repeat controls'):
                    control = native_controls(engine.bb, bound, out / ('controls-width' + str(width)), width=width)
                if control['status'] != 'exact-controls-passed':
                    raise ValueError('Same-width native cache controls failed')
                report['controls'].append({'width': width, 'status': control['status'],
                    'wallMsExcluded': (clock() - control_started) * 1000}); save(); reset(engine)
        else:
            report['controls'] = deps['controls'](engine, out)
        cases = fixed_cases()
        one(cases[0], BASELINE, excluded=True)
        a_miss = one(cases[0], CANDIDATE, excluded=True)
        a_hit = one(cases[0], CANDIDATE, excluded=True)
        if token_signature(a_miss) != token_signature(a_hit):
            raise ValueError('Same-width fresh/cache-hit token or binding parity failed')
        report['cacheParity'].append({'test': 'width20-A-miss-versus-A-hit', 'exact': True}); save()
        for case, order in zip(cases, ORDER, strict=True):
            receipts = {mode: one(case, mode) for mode in order}
            if receipts[BASELINE]['cards'] != receipts[CANDIDATE]['cards'] or receipts[BASELINE]['finalOrder'] != receipts[CANDIDATE]['finalOrder']:
                raise ValueError('Width/root-cache candidate changed learned rank/action bindings')
        a_after_b = one(cases[0], CANDIDATE, excluded=True)
        if token_signature(a_miss) != token_signature(a_after_b):
            raise ValueError('Same-width cached-root A-B-A actual native token parity failed')
        report['cacheParity'].append({'test': 'width20-A-before-different-shoppers-versus-A-return', 'exact': True})
        report['cacheReceipt'] = cache.receipt(); report['status'] = 'completed-fast-decode-comparison'
    except BaseException as error:
        report.update(status='failed-fast-decode-comparison', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        if cache is not None:
            try:
                report['cacheReceipt'] = cache.receipt(); cache.clear()
            except BaseException as error:
                report.update(status='failed-fast-decode-comparison', cacheError={'type': type(error).__name__, 'message': str(error)})
        if engine is not None:
            try:
                report['shutdownIntegrity'] = engine.close()
                if not all(report['shutdownIntegrity'].get(key) is True for key in ('frozen927HashesUnchanged', 'copyCheckpointUnchanged')):
                    raise ValueError('Frozen base/checkpoint integrity failed')
            except BaseException as error:
                report.update(status='failed-fast-decode-comparison', shutdownError={'type': type(error).__name__, 'message': str(error)})
        if sha(ROOT / 'store_fast_decode_v1.py') != preflight['candidateSourceSha256'] or sha(__file__) != preflight['runnerSourceSha256']:
            report.update(status='failed-fast-decode-comparison', sourceError='candidate or runner changed during run')
        ownership.close(); report['wholeRunWallMsIncludingControlsWarmup'] = (clock() - begin) * 1000; save()
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--run', action='store_true')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--memory-gib', type=int, choices=(7, 8), default=7,
        help='Explicit experimental budget; default7.8requires recorded headroom checks; no automatic raise')
    args = parser.parse_args(); report = run(args.out, native=args.run, manifest=args.manifest, memory_gib=args.memory_gib)
    print(json.dumps({key: report[key] for key in ('status', 'nativeExecution')}))
    return 1 if report['status'].startswith('failed') else 0


if __name__ == '__main__':
    raise SystemExit(main())
