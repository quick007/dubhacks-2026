"""Exclusive native public-copy warming and fresh-query cached-payload benchmark.

Default execution is CPU preparation. Only --run loads the single native engine;
8766 must already be stopped by its owner. Eight first-attempt complete pages
warm48 product/colors, after exact width10 controls. Cached wording is ALWAYS
generated earlier. Full serialized payload readiness includes fresh learned query
encoding, guards and real variant binding, excludes browser/network/image paint.
First3 availability is conservatively measured at full payload readiness: this
candidate does not implement staged first3 delivery. Useful title/body and all
20-field pass rates are separate; a low latency is never a quality pass.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from contextvars import ContextVar
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time

from .store_ar_comparison_run_v1 import DEFAULT_MANIFEST, demo_listening, reset_runtime
from .store_catalog_v1 import canonical, load_snapshot
from .store_demo_engine import DEFAULT_COPY, DEFAULT_RANK, NEUTRAL_COPY_REQUEST, demo_layout
from .store_generated_copy_cache_v1 import (GeneratedCopyCache, hashed, personalize_with_cache,
    producer_spec, sha, utc_now, warm_plan)

VERSION = 'store-generated-copy-cache-run.v1'
HERE = Path(__file__).resolve().parent
FRONTIER = HERE / 'store_speed_quality_frontier_v1'
SOURCES = ('store_generated_copy_cache_run_v1.py', 'store_generated_copy_cache_v1.py',
    'store_demo_engine.py', 'store_demo_catalog.py', 'store_copy_wide_v2.py',
    'store_copy_data_v2.py', 'store_rank_session.py', 'branch_local_copy.py',
    'parallel_decode.py', 'backbone.py', 'semantic_copy_training.py', 'store_copy_fit_v2.py')
FIRST3_SCOPE = 'conservative availability at full payload serialization; no staged delivery or browser-paint claim'
_REQUEST_DEADLINE_OWNED = ContextVar('generated_copy_cache_request_deadline_owned', default=False)


def public_frontier(directory=FRONTIER):
    directory = Path(directory); manifest = json.loads((directory / 'manifest.json').read_text())
    # Pin labels as an offline evaluation sidecar; do not read them here.
    for name, expected in manifest['files'].items():
        if sha(directory / name) != expected:
            raise ValueError('Pinned frontier file changed: ' + name)
    cases = json.loads((directory / 'public-requests.json').read_text())
    if (len(cases) != 32 or manifest['productFunctions'] != 16 or manifest['wordingsPerFunction'] != 2
            or len({c['caseId'] for c in cases}) != 32):
        raise ValueError('Fixed32 request-only development frontier required')
    for case in cases:
        if (set(case) != {'caseId', 'request', 'requestedCardCount', 'requestedCopyFields'}
                or set(case['request']) != {'query'} or type(case['request']['query']) is not str
                or not case['request']['query'].strip() or case['requestedCardCount'] != 6
                or case['requestedCopyFields'] != 20):
            raise ValueError('Only public shopper requests may enter inference; fixed six-card20-field roster required')
    return cases, manifest


def guard_probes(snapshot):
    product = snapshot['inputs']['products'][4]  # Real public bottle; not a hidden need.
    name, price = product['name'], product['itemPrice']
    color = product['colorVariants'][1]['colorName']
    return [
        {'caseId': 'guard-exact-name', 'request': {'query': name}, 'expectation': {'firstProductIndex': 4}},
        {'caseId': 'guard-exact-at-inclusive-budget', 'request': {'query': name, 'itemBudget': price},
            'expectation': {'firstProductIndex': 4, 'maximumPrice': price}},
        {'caseId': 'guard-exact-under-exclusive-price', 'request': {'query': name + f' under ${price}'},
            'expectation': {'cardCount': 0}},
        {'caseId': 'guard-exact-known-color', 'request': {'query': name,
            'hardRequirements': [{'attribute': 'color', 'value': color}]},
            'expectation': {'firstProductIndex': 4, 'selectedColor': color}},
        {'caseId': 'guard-unknown-color', 'request': {'query': 'Browse the store',
            'hardRequirements': [{'attribute': 'color', 'value': 'Ultraviolet-unlisted'}]},
            'expectation': {'cardCount': 0}},
        {'caseId': 'guard-unknown-stock', 'request': {'query': 'a bottle in stock'},
            'expectation': {'cardCount': 0}},
        {'caseId': 'guard-low-budget', 'request': {'query': 'Browse the store', 'itemBudget': 1},
            'expectation': {'cardCount': 0, 'maximumPrice': 1}},
    ]


def prepare(manifest=DEFAULT_MANIFEST, frontier=FRONTIER):
    """Read-only CPU pins; no tokenizer/model execution or native import."""
    from .store_demo_catalog import DemoCatalogBinding
    from .store_hybrid_benchmark_v2 import completed_adapter
    snapshot = load_snapshot(); binding = DemoCatalogBinding(manifest)
    binding.assert_current_sources(snapshot['receipt'])
    fit, checkpoint = completed_adapter(DEFAULT_COPY)
    files = fit['preflight']['modelFiles']; small = files['smallFileSha256']
    tokenizer_names = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'chat_template.jinja')
    token_pins = {name: small[name] for name in tokenizer_names}
    # Check allsmall tokenizer/config files now; constructor checks both large shards.
    for name, digest in small.items():
        if sha(Path(files['modelPath']) / name) != digest:
            raise ValueError('Pinned backbone/tokenizer file changed: ' + name)
    spec = producer_spec(sha(checkpoint), files['revision'], hashed(token_pins), native_width=10)
    cases, frontier_manifest = public_frontier(frontier)
    plan = warm_plan(snapshot)
    return {'version': VERSION, 'nativeExecution': False, 'snapshot': snapshot,
        'producerSpec': spec, 'modelFiles': files, 'tokenizerFilesSha256': token_pins,
        'integrationManifest': str(manifest), 'integrationManifestSha256': sha(manifest),
        'rankRun': str(DEFAULT_RANK), 'copyRun': str(DEFAULT_COPY),
        'frontierDirectory': str(frontier), 'frontierManifest': frontier_manifest,
        'frontierManifestSha256': sha(Path(frontier) / 'manifest.json'), 'cases': cases,
        'guardProbes': guard_probes(snapshot), 'warmPlan': plan,
        'sourceSha256': {name: sha(HERE / name) for name in SOURCES},
        'firstThreeTimingScope': FIRST3_SCOPE, 'labelsEnterInference': False,
        'copyConditioning': 'only neutral browsing request and selected public name/category/itemPrice/color; shopper independent',
        'limitations': ['curated inspected development frontier; not unseen generalization',
            'server payload only; no network/browser/image-transfer timing',
            'cached accepted text may still fail title/body usefulness; no automatic repair/retry',
            'initial native guard7GiB; driver cannot raise it',
            'one separate fresh-miss query then its warm repeat; not included in32 cached frontier latency']}


def native_generate(engine, cards, spec, control_sha, output):
    """One real unchanged selected page, fresh ordinary prefix, no teacher text."""
    from .branch_local_copy import prepare_root
    from .fused_prefix import _complete_cache
    from .native_batch_diagnosis import state_hash
    from .store_copy_fit_v2 import deadline
    from .store_demo_engine import decode_rows
    from .store_copy_wide_v2 import guard_memory
    output = Path(output)
    if output.exists():
        raise FileExistsError('Native attempt output must be unused')
    output.mkdir(parents=True)
    common, jobs = demo_layout(cards)
    if len(cards) == 6 and (len(jobs) != 20 or sum(r['maxTokens'] for r in jobs) != 416):
        raise ValueError('Native complete sixcard/twentyfield416-cap roster changed')
    receipt = {'status': 'running', 'cards': deepcopy(cards), 'common': common, 'jobs': jobs,
        'startedAt': utc_now(), 'provenance': {'nativeGeneration': True, 'fallback': False,
            'checkpointSha256': spec['checkpointSha256'], 'modelRevision': spec['modelRevision'],
            'tokenizerSha256': spec['tokenizerSha256'], 'nativeWidth': 10,
            'sameWidthControlsSha256': control_sha}}
    # Start wall immediately after actualUTCstart; request cold wall excludes receipt IO.
    started = time.perf_counter()
    try:
        # A cold miss runs INSIDE one()'s complete rank+copy deadline. Only our
        # explicit context can suppress the inner alarm; unrelated process alarms
        # still fail closed through deadline() on standalone warming calls.
        outer_owned = _REQUEST_DEADLINE_OWNED.get()
        receipt['deadlineOwner'] = 'complete-request' if outer_owned else 'native-generation'
        with (nullcontext() if outer_owned else deadline(120, 'cache warming complete native field request')):
            state, preparation = prepare_root(engine.bb, common)
            _complete_cache(state['cache'], len(state['tokens'])); before = state_hash(state['cache'])
            rows, timing = decode_rows(engine.bb, state, jobs, 10, lambda rows: None, lambda: False)
            # Preserve actual tokens even if a later decoding/provenance check fails.
            receipt.update(rows=rows, preparation=preparation, decodeTiming=timing)
            for row in rows:
                if row['text'] != engine.bb.tokenizer.decode(row['tokenIds']):
                    raise AssertionError('Archived native text differs from tokenizer decoding of actual generated IDs')
                if row['finishReason'] == 'stop' and row['stopTokenId'] not in engine.bb.tokenizer.eos_token_ids:
                    raise AssertionError('Native stop token is not an actual pinned tokenizer EOS')
                if (row['arSteps'] != len(row['rawSampledTokenIds'])
                        or row['tokenIds'] != (row['rawSampledTokenIds'][:-1] if row['finishReason'] == 'stop' else row['rawSampledTokenIds'])):
                    raise AssertionError('Native text/sample/EOS arithmetic inconsistent')
            if state_hash(state['cache']) != before:
                raise AssertionError('Native complete prefix changed during copy generation')
            memory = guard_memory(engine.bb)
        receipt.update(status='completed', rows=rows, preparation=preparation, decodeTiming=timing,
            prefixUnchanged=True, memory=memory, actualTokenizerDecodeAndEOSVerified=True)
    except BaseException as error:
        receipt.update(status='failed', error={'type': type(error).__name__, 'message': str(error)},
            partialRows=getattr(error, 'partial_rows', []))
        raise
    finally:
        receipt.update(completedAt=utc_now(), coldGenerationMs=(time.perf_counter() - started) * 1000)
        (output / 'native.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return receipt


def quality(result):
    """V1 unchanged screen: core T/C and allfields, never infer from nonnull count."""
    payload = result['payload']; cards = [c for s in payload['sections'] for c in s['cards']]
    validations = result.get('validation', {}).get('fields', [])
    by_field = {f['field']: f['accepted'] for f in validations}
    value = {'cardCount': len(cards), 'acceptedFields': payload['acceptedFields'],
        'withheldFields': payload['withheldFields'], 'requiredFields': payload['requiredFields']}
    for label, count in (('firstThree', 3), ('fullSix', 6)):
        core = [f'{kind}{i}' for i in range(1, count + 1) for kind in ('T', 'C')]
        optional = [f'S{i}' for i in range(1, (count + 2) // 3 + 1)] + [f'B{i}' for i in range(1, count + 1)]
        value[label] = {'coreAccepted': sum(by_field.get(k) is True for k in core),
            'coreRequired': len(core), 'coreTitleBodyPass': len(cards) >= count and all(by_field.get(k) is True for k in core),
            'optionalAccepted': sum(by_field.get(k) is True for k in optional), 'optionalRequired': len(optional),
            'allFieldsPass': len(cards) >= count and all(by_field.get(k) is True for k in core + optional)}
    return value


def check_guard(result, expectation):
    cards = [c for s in result['payload']['sections'] for c in s['cards']]
    failures = []
    if 'cardCount' in expectation and len(cards) != expectation['cardCount']:
        failures.append('cardCount')
    if 'firstProductIndex' in expectation and (not cards or cards[0]['productIndex'] != expectation['firstProductIndex']):
        failures.append('exactNameFirst')
    if 'maximumPrice' in expectation and any(c['itemPrice'] > expectation['maximumPrice'] for c in cards):
        failures.append('itemBudget')
    if 'selectedColor' in expectation and any(c['selectedColor'] != expectation['selectedColor'] for c in cards):
        failures.append('boundSelectedColor')
    if result.get('shortfall') != max(0, 6 - len(cards)) or result.get('noPadding') is not True:
        failures.append('shortfallOrPadding')
    return {'passed': not failures, 'failures': failures, 'expectation': expectation}


def summarize(records):
    frontier = [r for r in records if r['phase'] == 'frontier']
    completed = [r for r in frontier if r['status'] == 'completed']
    scored = [r for r in completed if type(r.get('offlineRelevance', {}).get('top1Matches')) is bool]
    values = [r['fullSerializedReadyMs'] for r in completed]
    return {'fixedFrontierRequests': 32, 'attempted': len(frontier), 'completed': len(completed),
        'failed': len(frontier) - len(completed), 'unattempted': 32 - len(frontier),
        'medianFullSerializedReadyMs': statistics.median(values) if values else None,
        'maximumFullSerializedReadyMs': max(values) if values else None,
        'under300MsFullPayloads': sum(t < 300 for t in values),
        'firstThreeCoreTitleBodyPasses': sum(r['quality']['firstThree']['coreTitleBodyPass'] for r in completed),
        'fullSixCoreTitleBodyPasses': sum(r['quality']['fullSix']['coreTitleBodyPass'] for r in completed),
        'all20FieldsPasses': sum(r['quality']['fullSix']['allFieldsPass'] for r in completed),
        'top1PublicFunctionMatches': sum(r['offlineRelevance']['top1Matches'] for r in scored) if scored else None,
        'top1PublicFunctionScored': len(scored), 'top1PublicFunctionUnscored': 32 - len(scored),
        'rateDenominator': 32, 'firstThreeTimingScope': FIRST3_SCOPE,
        'freshPerRequestCopyOnHits': False, 'latencyAloneIsNotUsefulQuality': True}


def verify_reuse(directory, preflight):
    """Read-only verification of a prior run's48colors, raw attempts and controls.

    A run may have failed AFTER warming. It must still prove completed warming,
    unchanged927/checkpoint/source shutdown, exact producer/facts, native receipt
    token verification and archived-source identity. Only this driver's version
    may differ; generated copy/validator sources and model inputs must match.
    """
    directory = Path(directory).resolve()
    prior_path = directory / 'results.json'; prior = json.loads(prior_path.read_text())
    old_preflight_path = directory / 'preflight.json'
    old = json.loads(old_preflight_path.read_text())
    if (sha(old_preflight_path) != prior['preflightSha256'] or prior['runtimeSourceHashesUnchanged'] is not True
            or any(prior['shutdownIntegrity'].get(k) is not True for k in ('frozen927HashesUnchanged', 'copyCheckpointUnchanged'))):
        raise ValueError('Warm reuse requires preserved preflight and successful927/checkpoint/source shutdown')
    if old['producerSpec'] != preflight['producerSpec'] or old['snapshot'] != preflight['snapshot']:
        raise ValueError('Warm reuse producer/checkpoint/tokenizer/public snapshot differs')
    if set(old['sourceSha256']) != set(preflight['sourceSha256']):
        raise ValueError('Warm reuse source inventory differs')
    paths = {prior_path, old_preflight_path}
    for name, expected in old['sourceSha256'].items():
        path = directory / 'source' / name; paths.add(path)
        if sha(path) != expected or (name != Path(__file__).name and expected != preflight['sourceSha256'][name]):
            raise ValueError('Warm reuse generated/native source changed: ' + name)
    control_path = directory / 'native-controls-width10/results.json'; paths.add(control_path)
    control = json.loads(control_path.read_text())
    if (sha(control_path) != prior['sameWidthControlsSha256'] or control != prior['sameWidthControls']
            or control['status'] != 'exact-controls-passed' or control['width'] != 10):
        raise ValueError('Original width10 controls changed or did not pass')
    records = prior['warmRecords']; groups = preflight['warmPlan']['groups']
    if len(records) != 8 or [r['groupId'] for r in records] != [g['groupId'] for g in groups]:
        raise ValueError('All8 original complete warm pages required')
    cache_root = directory / 'warm-cache'
    if not all((cache_root / name).exists() for name in ('manifest.json', 'entries', 'attempts')):
        raise ValueError('Original populated cache directories required; never create a reuse source')
    cache = GeneratedCopyCache(cache_root, preflight['snapshot'], preflight['producerSpec'])
    for record, group in zip(records, groups, strict=True):
        raw_path = directory / 'warm-native' / group['groupId'] / 'native.json'; paths.add(raw_path)
        raw = json.loads(raw_path.read_text())
        attempt_path = cache_root / 'attempts' / Path(record['attemptFile']).name
        if (record['status'] != 'completed' or record['cards'] != group['cards']
                or raw['status'] != 'completed' or raw['cards'] != group['cards']
                or raw.get('actualTokenizerDecodeAndEOSVerified') is not True
                or raw['provenance']['sameWidthControlsSha256'] != prior['sameWidthControlsSha256']
                or sha(attempt_path) != record['attemptSha256']
                or json.loads(attempt_path.read_text())['generation'] != raw):
            raise ValueError('Original warm native tokens/EOS/archive binding not fully verified')
        resolved = cache.resolve(group['cards'])
        if (resolved['cache']['initialMisses'] != 0 or resolved['cache']['copySource'] != 'generated-earlier'
                or len(resolved['rawRows']) != 20):
            raise ValueError('Original cache does not contain all48 product/color first attempts')
    paths.update(path for path in cache_root.rglob('*') if path.is_file())
    paths.update(path for path in (directory / 'native-controls-width10').rglob('*') if path.is_file())
    inventory = {str(path.relative_to(directory)): sha(path) for path in sorted(paths)}
    return {'sourceDirectory': str(directory), 'priorStatus': prior['status'],
        'priorResultsSha256': sha(prior_path), 'originalControlsSha256': prior['sameWidthControlsSha256'],
        'immutableFilesSha256': inventory, 'verifiedPublicProductColors': 48, 'reusedGeneratedEarlierOnly': True,
        'newNativeWarmingPages': 0, 'newRuntimeControlsStillRequired': True,
        'originalColdGenerationMs': [r['nativeColdGenerationMs'] for r in records]}


def run(out, *, native=False, manifest=DEFAULT_MANIFEST, frontier=FRONTIER, reuse_warm_cache=None, _dependencies=None):
    out = Path(out); deps = _dependencies or {}
    if out.exists():
        raise FileExistsError('Unused output required; preserve earlier cache attempts')
    if native and deps.get('service_listening', demo_listening)():
        raise RuntimeError('Demo8766 still listening; owner must stop it before exclusive native work')
    preflight = deepcopy(deps.get('prepare', prepare)(manifest, frontier))
    reuse = verify_reuse(reuse_warm_cache, preflight) if reuse_warm_cache is not None else None
    preflight['warmCacheReuse'] = reuse
    out.mkdir(parents=True); (out / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    if not native:
        return {'status': 'cpu-preflight-only', 'nativeExecution': False, 'preflight': preflight}
    if _dependencies is None:
        from .catalog_fusion.generalization_v2 import native_owner_lock
        from .store_demo_engine import ResidentDemoEngine
        from .store_copy_wide_v2 import native_controls
        lock = lambda: native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'), 'root')
        factory = lambda: ResidentDemoEngine(output=out / 'engine', width=10, memory_gib=7, integration_manifest=manifest)
        controls = lambda engine, bound, directory: native_controls(engine.bb, bound, directory, width=10)
    else:
        lock, factory, controls = deps['lock'], deps['factory'], deps['controls']
    reset = deps.get('reset', reset_runtime); generate = deps.get('generate', native_generate)
    report = {'version': VERSION, 'nativeExecution': True, 'status': 'started', 'warmRecords': [], 'records': [],
        'preflightSha256': sha(out / 'preflight.json'), 'firstThreeTimingScope': FIRST3_SCOPE,
        'noTemplateOrRepair': True, 'labelsEnterInference': False, 'generatedEarlierDisclosureRequired': True}
    engine = None; ownership = ExitStack(); run_started = time.perf_counter()
    def save():
        report['summary'] = summarize(report['records'])
        (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    def generation(cards, directory):
        return generate(engine, cards, preflight['producerSpec'], report['sameWidthControlsSha256'], directory)
    def one(case, phase, cache, *, miss=False):
        record = {'caseId': case['caseId'], 'phase': phase, 'request': deepcopy(case['request']), 'status': 'running'}
        report['records'].append(record); save()
        # No diskreceipt writes after entry to personalize_with_cache until its
        # full payload has been validated and actually serialized.
        started = time.perf_counter()
        try:
            from .store_copy_fit_v2 import deadline
            with deadline(120, 'complete fresh rank and cached/native-miss payload'):
                ownership_token = _REQUEST_DEADLINE_OWNED.set(True)
                try:
                    result = personalize_with_cache(engine.session, cache, deepcopy(case['request']), copy_adapter=engine.adapter,
                        native_generate=(lambda cards: generation(cards, out / 'fresh-miss-native')) if miss else None)
                    # Include function-call preparation and actual final serializer
                    # in the OUTER endpoint rather than omit a small call gap.
                    encoded = canonical(result['payload']).encode()
                    ready = (time.perf_counter() - started) * 1000
                finally:
                    _REQUEST_DEADLINE_OWNED.reset(ownership_token)
            if encoded != result['encodedPayload']:
                raise AssertionError('Reported endpoint bytes differ from final serialized public payload')
            copy = result['payload']['copyCache']
            record.update(status='completed', fullSerializedReadyMs=ready, firstThreeAvailableAtMs=ready,
                quality=quality(result), payloadSha256=result['payloadSha256'], payloadBytes=result['payloadBytes'],
                cache=copy, guards=result.get('guards'), shortfall=result['shortfall'],
                rank=result['ranking'], noPadding=result['noPadding'])
            if phase == 'frontier' and (record['quality']['cardCount'] != 6 or record['quality']['requiredFields'] != 20
                    or copy['copySource'] != 'generated-earlier' or copy['initialMisses'] != 0
                    or not all(e['generatedEarlier'] for e in copy['entries'])):
                raise AssertionError('Every frontier request must retain six real cards20 fields, all earlier native hits')
            if 'expectation' in case:
                record['guardCheck'] = check_guard(result, case['expectation'])
                if not record['guardCheck']['passed']:
                    raise AssertionError('Public guard probe failed')
            # Persist bytes that WERE serialized inside the reported endpoint.
            (out / (phase + '-' + case['caseId'] + '.payload.json')).write_bytes(result['encodedPayload'])
            archive = {key: value for key, value in result.items() if key != 'encodedPayload'}
            (out / (phase + '-' + case['caseId'] + '.json')).write_text(json.dumps(archive, indent=2) + '\n')
            return result
        except BaseException as error:
            record.update(status='failed', error={'type': type(error).__name__, 'message': str(error)},
                failedAtMs=(time.perf_counter() - started) * 1000)
            raise
        finally:
            record['outerWallIncludingReceiptMs'] = (time.perf_counter() - started) * 1000; save()
    save()
    try:
        ownership.enter_context(lock())
        if deps.get('service_listening', demo_listening)():
            raise RuntimeError('Demo8766 became active under owner lock; refuse native load')
        engine = factory(); report['startup'] = engine.startup
        if engine.copy_sha != preflight['producerSpec']['checkpointSha256']:
            raise ValueError('Resident verified adapter differs from preflight')
        if engine.session.snapshot != preflight['snapshot']:
            raise ValueError('Resident public snapshot differs from preflight')
        for name, expected in preflight['sourceSha256'].items():
            (out / 'source').mkdir(exist_ok=True)
            if sha(HERE / name) != expected:
                raise ValueError('Preflight native source changed: ' + name)
            (out / 'source' / name).write_bytes((HERE / name).read_bytes())
        first = preflight['warmPlan']['groups'][0]
        bound = {'request': deepcopy(NEUTRAL_COPY_REQUEST), 'cards': first['cards']}
        from .store_copy_data_v2 import layout
        if layout(bound) != demo_layout(first['cards']):
            raise ValueError('Control and actual generation prompts/caps differ')
        from .store_copy_fit_v2 import deadline
        with deadline(180, 'cache width10 fork/rebuild/repeat controls'):
            control = controls(engine, bound, out / 'native-controls-width10')
        if control['status'] != 'exact-controls-passed':
            raise AssertionError('Width10 fork/rebuild/repeat controls failed')
        report['sameWidthControls'] = control
        # Controls are pinned by actual written receipt, not merely the returned status.
        control_path = out / 'native-controls-width10/results.json'
        report['sameWidthControlsSha256'] = sha(control_path); save(); reset(engine)
        cache = GeneratedCopyCache((Path(reuse['sourceDirectory']) / 'warm-cache') if reuse else out / 'warm-cache',
            engine.session.snapshot, preflight['producerSpec'])
        if reuse:
            report['warmCacheReuse'] = reuse; save()
        else:
            for group in preflight['warmPlan']['groups']:
                record = {'groupId': group['groupId'], 'status': 'running', 'cards': group['cards']}
                report['warmRecords'].append(record); save()
                generated = generation(group['cards'], out / 'warm-native' / group['groupId'])
                recorded = cache.record_generation(group['cards'], generated)
                record.update(status='completed', startedAt=generated['startedAt'], completedAt=generated['completedAt'],
                    **recorded); save(); reset(engine)
        # No allocator clearing in timed ranking: same persistent resident cache
        # policy as serving, but fresh query encode on EVERY request.
        for case in preflight['cases']:
            one(case, 'frontier', cache)
        for case in preflight['guardProbes']:
            one(case, 'guards', cache)
        fresh_case = {'caseId': 'native-miss-then-hit', 'request': {'query': 'Browse products for an ordinary day'}}
        cold_cache = GeneratedCopyCache(out / 'isolated-cold-namespace', engine.session.snapshot, preflight['producerSpec'])
        reset(engine)
        cold = one(fresh_case, 'cold-miss', cold_cache, miss=True)
        hit = one(fresh_case, 'warm-repeat', cold_cache)
        if (cold['payload']['copyCache']['nativeMissGeneration'] is not True
                or hit['payload']['copyCache']['copySource'] != 'generated-earlier'
                or [r['text'] for r in cold['rawRows']] != [r['text'] for r in hit['rawRows']]):
            raise AssertionError('Real fresh-native miss then identical generated-earlier hit proof failed')
        if not _dependencies:
            if sha(Path(frontier) / 'manifest.json') != preflight['frontierManifestSha256']:
                raise ValueError('Frontier manifest changed after preflight')
            for name, expected_sha in preflight['frontierManifest']['files'].items():
                if sha(Path(frontier) / name) != expected_sha:
                    raise ValueError('Pinned posthoc frontier/public labels changed: ' + name)
        labels = deps.get('labels', lambda: json.loads((Path(frontier) / 'offline-relevance-labels.json').read_text()))()
        expected = {row['caseId']: row['expectedTopProductIndex'] for row in labels}
        if set(expected) != {case['caseId'] for case in preflight['cases']}:
            raise ValueError('Offline public function label roster differs from32 requests')
        for record in report['records']:
            if record['phase'] == 'frontier':
                result = json.loads((out / ('frontier-' + record['caseId'] + '.json')).read_text())
                cards = [c for s in result['payload']['sections'] for c in s['cards']]
                record['offlineRelevance'] = {'top1Matches': cards[0]['productIndex'] == expected[record['caseId']],
                    'expectedProductIndex': expected[record['caseId']], 'scoredAfterAllInference': True}
        report['status'] = 'completed-cache-candidate'
    except BaseException as error:
        for record in report['warmRecords']:
            if record['status'] == 'running':
                record.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        report.update(status='failed-cache-candidate', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        if engine is not None:
            try:
                report['shutdownIntegrity'] = engine.close()
                if not all(report['shutdownIntegrity'].get(k) is True for k in ('frozen927HashesUnchanged', 'copyCheckpointUnchanged')):
                    raise AssertionError('Shutdown927/checkpoint integrity failed')
            except BaseException as error:
                report.update(status='failed-cache-candidate', shutdownError={'type': type(error).__name__, 'message': str(error)})
        try:
            if any(sha(HERE / name) != value for name, value in preflight['sourceSha256'].items()):
                raise ValueError('Frozen runner/cache/native source changed during experiment')
            report['runtimeSourceHashesUnchanged'] = True
            if reuse:
                base = Path(reuse['sourceDirectory'])
                if any(sha(base / name) != value for name, value in reuse['immutableFilesSha256'].items()):
                    raise ValueError('Original reused warm cache/control/native receipt was modified')
                report['reusedOriginalFilesUnchanged'] = True
        except BaseException as error:
            report.update(status='failed-cache-candidate', sourceError={'type': type(error).__name__, 'message': str(error)})
        try:
            ownership.close()
        except BaseException as error:
            report.update(status='failed-cache-candidate', ownershipError={'type': type(error).__name__, 'message': str(error)})
        report['wholeRunWallMsIncludingControlsWarmupAndShutdown'] = (time.perf_counter() - run_started) * 1000
        report['warmSummary'] = {'fixedNativePages': 8, 'completed': sum(r['status'] == 'completed' for r in report['warmRecords']),
            'attempted': len(report['warmRecords']), 'unattempted': 8 - len(report['warmRecords']),
            'originalColdGenerationMs': [r['nativeColdGenerationMs'] for r in report['warmRecords'] if r['status'] == 'completed'],
            'latencyExcludedFromCachedRequests': True, 'nativeRequestsDoNotUseTeacherText': True}
        if reuse:
            report['warmSummary'].update(fixedNativePages=0, unattempted=0, reusedProductColors=48,
                originalColdGenerationMs=reuse['originalColdGenerationMs'], newNativeWarmingPages=0)
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--frontier', type=Path, default=FRONTIER)
    parser.add_argument('--reuse-warm-cache', type=Path,
        help='Prior run directory with immutable verified48color warm-cache/control/native receipts; never rewritten')
    args = parser.parse_args(); report = run(args.out, native=args.run, manifest=args.manifest, frontier=args.frontier,
        reuse_warm_cache=args.reuse_warm_cache)
    print(json.dumps({k: report[k] for k in ('status', 'nativeExecution')}))
    return int(report['status'].startswith('failed'))


if __name__ == '__main__':
    raise SystemExit(main())
