"""Explicit, exclusive native AR-vs-hybrid comparison driver.

Default CLI is CPU preflight only. --run is required for native work. The
driver refuses a listening demo service, never stops it, and cannot resolve a
pending approval on the caller's behalf. Preserve every previous output path.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import socket
import statistics
import time
import uuid

from .store_catalog_v1 import canonical, load_snapshot
from .store_autoregressive_baseline import (CONFIG as AR_CONFIG, build_messages, paired_hybrid_readout,
    prompt_tokens, run_resident_case)
from .store_copy_fit_v2 import deadline
from .store_demo_engine import DEFAULT_COPY, DEFAULT_RANK, NEUTRAL_COPY_REQUEST

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / 'store_demo_integration_v3/manifest.json'
REQUESTS = ROOT / 'store_ar_baseline_requests_v1.json'
FIXED_IDS = ('demo-water-six', 'demo-light-six', 'demo-notebook-six')
ARM_ORDER = (('A', 'B'), ('B', 'A'), ('A', 'B'))
ARM_NAMES = {'A': 'conventional-frozen-base', 'B': 'hybrid-v3-rank-copy-v2'}
SOURCES = ('store_ar_comparison_run_v1.py', 'store_autoregressive_baseline.py',
    'store_demo_engine.py', 'store_demo_catalog.py', 'store_copy_wide_v2.py',
    'store_rank_session.py', 'backbone.py', 'store_copy_data_v2.py', 'semantic_copy_training.py',
    'store_copy_fit_v2.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def demo_listening():
    try:
        with socket.create_connection(('127.0.0.1', 8766), timeout=.5):
            return True
    except ConnectionRefusedError:
        return False
    except OSError as error:
        raise RuntimeError('Could not establish whether demo8766 is stopped; refuse native launch') from error


def fixed_cases():
    cases = json.loads(REQUESTS.read_text())
    by_id = {case['caseId']: case for case in cases}
    selected = [by_id[case_id] for case_id in FIXED_IDS]
    expected_queries = ('a bottle for drinking water', 'a light for my bedside table', 'a notebook for quick notes')
    if any(case != {'caseId': case_id, 'request': {'query': query}} for case, case_id, query in
            zip(selected, FIXED_IDS, expected_queries, strict=True)):
        raise ValueError('Fixed three public query fixtures changed')
    return selected


def cpu_preflight(manifest=DEFAULT_MANIFEST, *, tokenizer=None):
    """Current reviewed runtime catalog and all complete six-card prompts."""
    from .store_demo_catalog import DemoCatalogBinding
    from .store_hybrid_benchmark_v2 import completed_adapter
    snapshot = load_snapshot(); binding = DemoCatalogBinding(manifest)
    binding.assert_current_sources(snapshot['receipt'])
    fit, checkpoint = completed_adapter(DEFAULT_COPY)
    rank = json.loads((DEFAULT_RANK / 'report.json').read_text())
    if tokenizer is None:
        from transformers import AutoTokenizer
        from .semantic_copy_feature_fit import MODEL
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    cases = []
    for original, order in zip(fixed_cases(), ARM_ORDER, strict=True):
        messages, public, _ = build_messages(original['request'], snapshot)
        tokens = prompt_tokens(tokenizer, messages)
        if public['requestedCardCount'] != 6 or len(public['catalog']) != 16:
            raise ValueError('Exact16-catalog/six-card comparator workload required')
        cases.append({**deepcopy(original), 'armOrder': list(order), 'catalogProducts': 16,
            'cards': 6, 'copyFields': 20, 'conventionalPromptTokens': len(tokens),
            'conventionalPromptSha256': hashlib.sha256(canonical(tokens).encode()).hexdigest(),
            'publicInputSha256': hashlib.sha256(canonical(public).encode()).hexdigest()})
    return {'version': 'store-ar-comparison-run.v1', 'nativeExecution': False, 'cases': cases,
        'armNames': ARM_NAMES, 'sourceSha256': {name: sha(ROOT / name) for name in SOURCES},
        'requestFixtureSha256': sha(REQUESTS), 'integrationManifest': str(manifest),
        'integrationManifestSha256': sha(manifest), 'copyCheckpointSha256': sha(checkpoint),
        'copyReportSha256': sha(DEFAULT_COPY / 'results.json'),
        'rankCheckpointSha256': rank['heads']['learned']['sha256'],
        'rankReportSha256': sha(DEFAULT_RANK / 'report.json'),
        'pinnedBackboneFilesFromCompletedFit': fit['preflight']['modelFiles'],
        'freshLargeShardVerification': 'ResidentDemoEngine constructor before native use; CPU preflight records completed-fit receipts only',
        'workload': 'same three public requests, complete16catalog, six real cards and20fresh copyfields per request',
        'timerPolicy': 'same caller perf_counter entry passed to both legacy AR helper and hybrid personalize; validated serialized first3/full6 readiness; startup/warmup/controls excluded and reported separately',
        'cachePolicy': 'one warm resident backbone; one untimed complete warmup per system; fresh request caches and allocator clear before each arm; no response/prompt-result caching',
        'approvalPolicy': 'explicit --run does not resolve parent-reported pending approval; caller must resolve it first; active service is never stopped or bypassed'}


def reset_runtime(engine):
    gc.collect(); engine.bb.mx.synchronize(); engine.bb.mx.clear_cache()
    from .store_copy_wide_v2 import guard_memory
    return guard_memory(engine.bb)


def cache_controls(engine, directory):
    from .store_copy_fit_v2 import deadline
    from .store_copy_wide_v2 import native_controls
    from .semantic_copy_training import disabled
    from .store_rank_session import action_bindings
    request = {'query': 'browse', 'preferences': [], 'itemBudget': None, 'hardRequirements': []}
    bound = {'request': deepcopy(NEUTRAL_COPY_REQUEST),
        'cards': action_bindings(request, list(range(6)), engine.session.snapshot)}
    controls = []
    for mode in ('frozen-base', 'copy-v2-adapter'):
        context = disabled(engine.bb.model, engine.adapter) if mode == 'frozen-base' else nullcontext()
        started = time.perf_counter()
        with context, deadline(180, 'comparison same-width complete native cache controls'):
            result = native_controls(engine.bb, bound, directory / mode, width=engine.width)
        if result['status'] != 'exact-controls-passed':
            raise ValueError('Required fork/rebuild/repeat native cache control did not pass')
        controls.append({'mode': mode, 'status': result['status'], 'width': engine.width,
            'wallMsExcluded': (time.perf_counter() - started) * 1000,
            'resultPath': str(directory / mode / 'results.json'),
            'scope': 'same-width independent-cache fork/rebuild/repeat distributions for first10 ordinary field jobs; not forced baseline JSON logits'})
        reset_runtime(engine)
    return controls


def summarize(records):
    result = {}
    for arm in ('A', 'B'):
        rows = [record for record in records if record['arm'] == arm]
        def median(key):
            values = [row[key] for row in rows if row.get(key) is not None]
            return statistics.median(values) if values else None
        result[ARM_NAMES[arm]] = {'attempts': len(rows), 'cardCounts': [row.get('cardCount') for row in rows],
            'copyFieldCounts': [row.get('copyFields') for row in rows],
            'firstThreeUsableProxyPasses': sum(row.get('usableFirstThreeMs') is not None for row in rows),
            'fullSixUsableProxyPasses': sum(row.get('usableFullMs') is not None for row in rows),
            'medianAttemptFirstThreeMs': median('firstThreeAttemptMs'),
            'medianAttemptFullMs': median('fullAttemptMs'),
            'medianUsableFirstThreeMs': median('usableFirstThreeMs'),
            'medianUsableFullMs': median('usableFullMs'),
            'failedAttempts': sum(row['status'] not in ('completed-attempt', 'completed') for row in rows),
            'actualSampledTokens': [row.get('actualSampledTokens') for row in rows]}
    return result


def run_comparison(out, *, run=False, manifest=DEFAULT_MANIFEST, _dependencies=None):
    """No native dependency is loaded until explicit run + service/lock checks."""
    out = Path(out)
    if out.exists():
        raise FileExistsError('Comparison output must be unused; preserve prior results')
    deps = {} if _dependencies is None else _dependencies
    if run and deps.get('service_listening', demo_listening)():
        raise RuntimeError('Demo service is listening on127.0.0.1:8766; refuse native comparison; no automatic stop')
    preflight = deps.get('preflight', cpu_preflight)(manifest)
    out.mkdir(parents=True)
    (out / 'preflight.json').write_text(json.dumps(preflight, indent=2, allow_nan=False) + '\n')
    if not run:
        return {'status': 'cpu-preflight-only', 'nativeExecution': False, 'preflight': preflight}
    if _dependencies is None:
        from .catalog_fusion.generalization_v2 import native_owner_lock
        from .store_demo_engine import ResidentDemoEngine
        lock = lambda: native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'), 'root')
        engine_factory = lambda: ResidentDemoEngine(output=out / 'hybrid', width=10, integration_manifest=manifest)
    else:
        lock = deps['native_lock']; engine_factory = deps['engine_factory']
    baseline = deps.get('baseline', run_resident_case)
    reset = deps.get('reset_runtime', reset_runtime)
    controls = deps.get('cache_controls', cache_controls)
    clock = deps.get('clock', time.perf_counter)
    report = {'version': preflight['version'], 'status': 'started', 'nativeExecution': True,
        'preflightSha256': sha(out / 'preflight.json'), 'records': [], 'warmups': [], 'controls': [],
        'fixedArmOrder': [list(order) for order in ARM_ORDER],
        'modelSystemTradeoff': 'A frozen-base ordinary single AR product decision+copy; B trained v3 rank head plus320-update adapted independent copy fields on same resident backbone',
        'qualityScope': 'same hard guards and practical/strict text screens; accepted title+body per card usable-copy proxy, not independent human usefulness; no speedup claim from failed proxies',
        'totalExecutionStarted': clock()}
    engine = None; ownership = ExitStack()
    def save():
        report['summary'] = summarize(report['records'])
        (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    def one_arm(case, arm, *, warmup=False):
        reset(engine)
        started = clock()
        record = {'caseId': case['caseId'], 'arm': arm, 'warmupExcluded': warmup,
            'input': deepcopy(case['request']), 'expectedCards': 6, 'expectedCopyFields': 20}
        if arm == 'A':
            directory = out / ('warmup-ar' if warmup else 'ar-' + case['caseId'])
            # The helper's per-response budget cannot cover a blocked prefill or
            # next() before the first yield. Bound the entire main-thread call.
            with deadline(AR_CONFIG['requestTimeoutSeconds'], 'conventional AR complete request'):
                value = baseline(engine.bb, case['request'], engine.session.snapshot, directory,
                    mode='frozen-base', adapter=engine.adapter, source_validator=engine.binding.assert_current_sources,
                    request_started=started)
            full = value.get('full'); first = value.get('firstThree')
            record.update(status=value['status'], **value['summary'],
                cardCount=full['cardCount'] if full else None,
                copyFields=full['validation']['requiredFields'] if full else None,
                allFirstThreeFieldsAccepted=first['validation']['allPracticalFieldsAccepted'] if first else False,
                allFullFieldsAccepted=full['validation']['allPracticalFieldsAccepted'] if full else False,
                actualSampledTokens=value.get('generatedTokens'), resultPath=str(directory / 'result.json'),
                error=value.get('error'))
            if record['error'] and record['error']['type'] in ('MemoryError', 'FloatingPointError'):
                raise RuntimeError('Native safety failure in conventional arm: ' + record['error']['message'])
        else:
            request_id = str(uuid.uuid4()); events = []
            # The service's decode-loop check starts after ranking/root prefill;
            # the standalone comparison bounds the same complete call as A.
            with deadline(AR_CONFIG['requestTimeoutSeconds'], 'hybrid complete request'):
                engine.personalize(case['request'], request_id, started, events.append, lambda: False)
            path = Path(engine.output) / 'requests' / request_id / 'receipt.json'
            value = json.loads(path.read_text()); paired = paired_hybrid_readout(path, engine.session.snapshot)
            endpoints = {event['type']: event for event in paired['endpoints']}
            first, full = endpoints.get('firstThree'), endpoints.get('complete')
            record.update(status=value['status'], cardCount=paired['cardCount'],
                copyFields=full['asServedRequiredFields'] if full else None,
                firstThreeAttemptMs=first['timing']['readyMs'] if first else None,
                fullAttemptMs=full['timing']['readyMs'] if full else None,
                usableFirstThreeMs=first['usableReadyMs'] if first else None,
                usableFullMs=full['usableReadyMs'] if full else None,
                allFirstThreeFieldsAccepted=first['asServedAllPracticalFieldsAccepted'] if first else False,
                allFullFieldsAccepted=full['asServedAllPracticalFieldsAccepted'] if full else False,
                actualSampledTokens=value.get('decodeTiming', {}).get('actualSampledTokens'), resultPath=str(path),
                sourceCacheImmutability='engine compares full common native source state hash before/after each request; failure raises')
        record['outerCallWallMsIncludingArchive'] = (clock() - started) * 1000
        record['timerPolicy'] = 'same caller entryperf_counter passed to both paths; ready times exclude later archive writes; outerCallWall includes archive writes'
        if record['status'] in ('completed', 'completed-attempt') and record['cardCount'] != 6:
            raise ValueError('Completed comparison silently changed six-card workload')
        return record
    save()
    try:
        ownership.enter_context(lock())
        if deps.get('service_listening', demo_listening)():
            raise RuntimeError('Demo service became active before native load; refuse without stopping it')
        load_started = clock(); engine = engine_factory()
        report['residentInitializationMsExcluded'] = (clock() - load_started) * 1000
        report['startup'] = engine.startup
        report['controls'] = controls(engine, out / 'controls')
        save()
        # Warm both complete systems once using the same first public case.
        for arm in ('A', 'B'):
            report['warmups'].append(one_arm(preflight['cases'][0], arm, warmup=True)); save()
        for case in preflight['cases']:
            for arm in case['armOrder']:
                arm_started = clock()
                try:
                    report['records'].append(one_arm(case, arm)); save()
                except BaseException as error:
                    report['records'].append({'caseId': case['caseId'], 'arm': arm, 'status': 'driver-failed-attempt',
                        'input': case['request'], 'failedAtMs': (clock() - arm_started) * 1000,
                        'error': {'type': type(error).__name__, 'message': str(error)}})
                    save(); raise
        report['status'] = 'completed-comparison'
    except BaseException as error:
        report.update(status='failed-comparison', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        if engine is not None:
            try:
                report['shutdownIntegrity'] = engine.close()
                if not all(report['shutdownIntegrity'].get(key) is True for key in ('frozen927HashesUnchanged', 'copyCheckpointUnchanged')):
                    report['status'] = 'failed-shutdown-integrity'
            except BaseException as error:
                report.update(status='failed-shutdown', shutdownError={'type': type(error).__name__, 'message': str(error)})
        ownership.close()
        report['totalExecutionWallMsIncludingSetupControlsWarmup'] = (clock() - report['totalExecutionStarted']) * 1000
        save()
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--run', action='store_true', help='Explicit native execution; resolve pending approval and stop service yourself first')
    args = parser.parse_args()
    result = run_comparison(args.out, run=args.run, manifest=args.manifest)
    print(canonical({'status': result['status'], 'nativeExecution': result['nativeExecution'], 'summary': result.get('summary')}))
    if args.run and result['status'] != 'completed-comparison':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
