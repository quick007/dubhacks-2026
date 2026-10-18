"""Optional wider actual-store decoding; CPU imports never load a model.

Widths10 and20 retain all20 fields, six cards and416 sampled tokens. Each width
needs its OWN exact fork/rebuild/repeat native distribution controls. Cross-width
numeric equality is neither assumed nor required. Caller owns checkpoint,
root-owner lock, deadlines and the inspected memory budget; initial guard7GiB.
No frozen fit/session source is modified, and this module never emits fallback.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time

from .store_copy_data_v2 import CAPS, FIELDS, assemble, canonical, layout
from .semantic_copy_feature_fit import dump, hashed

WIDTHS = (10, 20)
DEFAULT_MEMORY_GIB = 7
MAX_SUFFIX = 256
MAX_CONTEXT = 4096
SAMPLED_CAP = 416


def validate_width(width):
    if type(width) is not int or width not in WIDTHS:
        raise ValueError('Explicit native width10 or20 required')
    return width


def guard_memory(bb):
    """Honor caller-inspected native guard, including peak; never raise it here."""
    bb.check_memory()
    memory = bb.runtime_memory()
    budget = memory['processBudgetBytes']
    if max(memory['mlxActiveBytes'], memory['mlxPeakBytes']) > budget:
        raise MemoryError('Wide decode active/peak exceeds caller-declared native budget')
    return memory


def serving_jobs(bound, width):
    validate_width(width)
    common, jobs = layout(bound)
    if tuple(row['fields'][0] for row in jobs) != tuple(FIELDS) or len(jobs) != 20:
        raise ValueError('Complete twenty-field actual-store layout required')
    if sum(row['maxTokens'] for row in jobs) != SAMPLED_CAP or sum(row['maxTokens'] for row in jobs[:10]) != 208:
        raise ValueError('Preserve native first3/full6 sampled caps208/416')
    if common['role'] != 'user' or any(len(row['messages']) != 1 or row['messages'][0]['role'] != 'user' for row in jobs):
        raise ValueError('Fresh ordinary chat root and independent field-user suffixes required')
    if any('targetValue' in row or 'text' in row for row in jobs):
        raise ValueError('Teacher/fallback values must never enter native serving jobs')
    return common, jobs


def serialized_payload(result):
    encoded = canonical(result['payload']).encode()
    if hashlib.sha256(encoded).hexdigest() != result['payloadSha256'] or len(encoded) != result['payloadBytes']:
        raise AssertionError('Actual payload serialization receipt differs')
    return encoded


def require_completed_checkpoint(copy_run):
    """Read-only candidate gate: reuse the completed v2 checkpoint unchanged."""
    from .store_hybrid_benchmark_v2 import completed_adapter
    from .store_copy_fit_v2 import DATA, assert_source_fresh
    report, checkpoint = completed_adapter(copy_run)
    assert_source_fresh(report['preflight'], DATA)
    return {'checkpointPath': str(checkpoint), 'checkpointSha256': report['checkpoint']['sha256'],
        'trainingUpdates': report['nativeUpdates'], 'retrainingPermitted': False,
        'widths': list(WIDTHS), 'defaultMemoryGiB': DEFAULT_MEMORY_GIB,
        'nativeControls': 'one independent same-width fork/rebuild/repeat proof per tested width and mode',
        'scope': 'complete first3/full6 action payloads; no browser paint or image transfer timing'}


def decode_complete_rows(bb, state, rows, width, on_first_complete):
    """One continuous generator, first-section priority, real ready callback.

    The callback is invoked once all ten first-section rows have ended. Its CPU
    validation/serialization cost remains inside full wall time. Remaining rows
    keep their independent full prefix forks; no generated sibling is an input.
    """
    validate_width(width)
    if (tuple(r['fields'][0] for r in rows) != tuple(FIELDS) or len(rows) != 20
            or any(r['maxTokens'] != CAPS[r['fields'][0][0]] for r in rows)
            or sum(r['maxTokens'] for r in rows) != SAMPLED_CAP):
        raise ValueError('All twenty unchanged actual-store fields and416 sampled tokens required')
    from mlx_lm.generate import BatchGenerator, StopSequences, generation_stream
    from mlx_lm.sample_utils import make_sampler
    from .parallel_decode import clone_prefix_state, exact_row_suffix
    mx = bb.mx; output = {}; active = {}; pending = list(enumerate(rows)); first_done = False
    started = time.perf_counter(); fork_ms = suffix_ms = native_ms = callback_ms = 0.
    rounds = max_live = 0; admissions = []; finish_order = []
    gen = BatchGenerator(bb.model, completion_batch_size=width, prefill_batch_size=width,
        prefill_step_size=256, sampler=make_sampler(temp=0.0))
    def admit():
        nonlocal fork_ms, suffix_ms
        count = min(width - len(active), len(pending))
        if count <= 0:
            return
        group = [pending.pop(0) for _ in range(count)]; begin = time.perf_counter()
        branches = [clone_prefix_state(bb, state) for _ in group]
        fork_ms += (time.perf_counter() - begin) * 1000; guard_memory(bb); begin = time.perf_counter()
        prompts = []; caches = []
        for (_, row), branch in zip(group, branches, strict=True):
            suffix, cache = exact_row_suffix(bb.tokenizer, row, branch, bb.model)
            if not 1 <= len(suffix) <= MAX_SUFFIX or len(state['tokens']) + len(suffix) + row['maxTokens'] > MAX_CONTEXT:
                raise ValueError('Serving suffix exceeds complete teacher-gated 256-token cap')
            prompts.append(suffix); caches.append(cache)
        suffix_ms += (time.perf_counter() - begin) * 1000
        stops = [[int(token)] for token in bb.tokenizer.eos_token_ids]
        uids = gen.insert(prompts, max_tokens=[r['maxTokens'] for _, r in group], caches=caches,
            all_tokens=[list(state['tokens']) for _ in group], samplers=[make_sampler(temp=0.0) for _ in group],
            logits_processors=[[] for _ in group], stop_sequences=[StopSequences(stops) for _ in group])
        for uid, (index, row), suffix in zip(uids, group, prompts, strict=True):
            active[uid] = index
            output[index] = {**copy.deepcopy(row), 'tokenIds': [], 'rawSampledTokenIds': [],
                'stopTokenId': None, 'arSteps': 0, 'finishReason': None, 'suffixTokens': len(suffix),
                'promptSha256': hashed(row['messages'])}
        admissions.append({'admitted': count, 'activeBound': len(active), 'rowKeys': [r['key'] for _, r in group],
            'atMs': (time.perf_counter() - started) * 1000})
        del branches, caches
    try:
        admit()
        while active:
            begin = time.perf_counter(); responses = gen.next_generated(); mx.synchronize(generation_stream)
            native_ms += (time.perf_counter() - begin) * 1000
            if not responses:
                raise RuntimeError('Native generator ended before every active row finished')
            rounds += 1; max_live = max(max_live, len(responses))
            for response in responses:
                index = active[response.uid]; record = output[index]
                record['arSteps'] += 1; record['rawSampledTokenIds'].append(int(response.token))
                if response.finish_reason != 'stop':
                    record['tokenIds'].append(int(response.token))
                if response.finish_reason is not None:
                    record['finishReason'] = response.finish_reason
                    record['stopTokenId'] = int(response.token) if response.finish_reason == 'stop' else None
                    record['completedMs'] = (time.perf_counter() - started) * 1000
                    record['text'] = bb.tokenizer.decode(record['tokenIds'])
                    finish_order.append(record['key']); del active[response.uid]
            del response, responses
            guard_memory(bb)
            if not first_done and all(i in output and output[i]['finishReason'] is not None for i in range(10)):
                begin = time.perf_counter(); on_first_complete([output[i] for i in range(10)])
                callback_ms += (time.perf_counter() - begin) * 1000; first_done = True
            admit()
        if len(output) != 20 or not first_done:
            raise RuntimeError('Complete first-section/full-field roster was not produced')
        return [output[i] for i in range(20)], {'cacheForkMaterializeMs': fork_ms,
            'suffixBoundaryPreparationMs': suffix_ms, 'nativeSuffixPrefillAndDecodeMs': native_ms,
            'firstSectionValidationSerializationCallbackMs': callback_ms,
            'continuousDecodeWallMs': (time.perf_counter() - started) * 1000,
            'nativeDecodeRounds': rounds, 'maximumRowsEmittingTogether': max_live,
            'nativePrefillStepSize': 256, 'rowArSteps': sum(r['arSteps'] for r in output.values()),
            'admissions': admissions, 'finishOrder': finish_order,
            'admissionPolicy': 'one continuous generator; section-one priority; refill each freed independent lane'}
    except Exception as exc:
        # Preserve observed rows even if generation fails before returning.
        setattr(exc, 'partial_rows', [output[i] for i in sorted(output)])
        raise
    finally:
        gen.close(); mx.synchronize(generation_stream)



def native_controls(bb, bound, out, *, width):
    """Actual requested-width/original-cap fork, independent rebuild and fork repeat.

    Hash every full native log-probability vector including EOS, preserving
    mismatch evidence before failing. Controls are outside request latency.
    No trimming, cache splice, token repair, tolerance, or cross-shape equality.
    """
    validate_width(width)
    from mlx_lm.generate import BatchGenerator, StopSequences, generation_stream
    from mlx_lm.sample_utils import make_sampler
    from .branch_local_copy import prepare_root
    from .fused_prefix import _complete_cache
    from .native_batch_diagnosis import state_hash
    from .parallel_decode import clone_prefix_state, exact_row_suffix
    out = Path(out)
    if out.exists():
        raise FileExistsError('Preserve prior exact-control evidence')
    out.mkdir(parents=True); common, jobs = serving_jobs(bound, width); jobs = jobs[:width]
    report = {'status': 'running', 'width': width, 'rows': copy.deepcopy(jobs), 'arms': [],
              'originalCaps': [r['maxTokens'] for r in jobs], 'latencyClaim': False,
              'comparisonScope': 'only same width fork/rebuild/repeat distributions; no cross-width equality',
              'commonSha256': hashed(common), 'rowMessagesSha256': hashed([r['messages'] for r in jobs])}
    mx = bb.mx
    reference = None
    try:
        report['stage'] = 'fresh-ordinary-root'
        dump(out / 'results.json', report)
        source, preparation = prepare_root(bb, common)
        _complete_cache(source['cache'], len(source['tokens'])); original = state_hash(source['cache'])
        report.update(sourceSha256=original, preparation=preparation, stage='matched-native-decode')
        for label in ('fork', 'independent-rebuild', 'fork-repeat'):
            prompts, caches = [], []
            for row in jobs:
                branch = prepare_root(bb, common)[0] if label == 'independent-rebuild' else clone_prefix_state(bb, source)
                if state_hash(branch['cache']) != original:
                    raise AssertionError('Independent rebuilt/forked complete cache differs')
                suffix, cache = exact_row_suffix(bb.tokenizer, row, branch, bb.model)
                if not 1 <= len(suffix) <= 256 or len(source['tokens']) + len(suffix) + row['maxTokens'] > 4096:
                    raise ValueError('Complete control suffix/context exceeds caps; no truncation')
                prompts.append(suffix); caches.append(cache)
            generator = BatchGenerator(bb.model, completion_batch_size=width, prefill_batch_size=width,
                prefill_step_size=256, sampler=make_sampler(temp=0.0))
            records = []; maximum = 0
            try:
                uids = generator.insert(prompts, max_tokens=[r['maxTokens'] for r in jobs], caches=caches,
                    all_tokens=[list(source['tokens']) for _ in jobs], samplers=[make_sampler(temp=0.0) for _ in jobs],
                    logits_processors=[[] for _ in jobs],
                    stop_sequences=[StopSequences([[int(t)] for t in bb.tokenizer.eos_token_ids]) for _ in jobs])
                by_uid = {uid: {'rowId': row['rowId'], 'rawSampledTokenIds': [], 'distributionSha256': [],
                    'finishReason': None, 'stopTokenId': None} for uid, row in zip(uids, jobs, strict=True)}
                records = [by_uid[uid] for uid in uids]
                report['activeArm'] = {'label': label, 'rows': records}
                del branch, cache, caches
                while responses := generator.next_generated():
                    maximum = max(maximum, len(responses))
                    for response in responses:
                        row = by_uid[response.uid]
                        if row['finishReason'] is not None:
                            raise AssertionError('Completed native UID emitted again')
                        probability = response.logprobs.astype(mx.float32); mx.eval(probability); mx.synchronize(generation_stream)
                        host = bb.np.array(probability)
                        if not bb.np.isfinite(host).all():
                            raise FloatingPointError('Nonfinite native distribution')
                        row['distributionSha256'].append(hashlib.sha256(host.tobytes()).hexdigest())
                        row['rawSampledTokenIds'].append(int(response.token))
                        if response.finish_reason is not None:
                            row['finishReason'] = response.finish_reason
                            row['stopTokenId'] = int(response.token) if response.finish_reason == 'stop' else None
                    del response, responses, probability, host
                    guard_memory(bb)
            finally:
                generator.close(); mx.synchronize(generation_stream)
            arm = {'label': label, 'rows': records, 'maximumRowsEmittingTogether': maximum,
                'sourceAfter': state_hash(source['cache']), 'memory': bb.runtime_memory()}
            report['arms'].append(arm); report.pop('activeArm', None); dump(out / (label + '.json'), arm)
            if maximum != width or any(r['finishReason'] is None for r in records):
                raise AssertionError('Control did not execute every row at actual requested width')
            if arm['sourceAfter'] != original:
                raise AssertionError('Original complete source cache mutated')
            if reference is None:
                reference = records
            elif reference != records:
                raise AssertionError('Strict samewidth token/EOS/full-distribution mismatch')
        report['status'] = 'exact-controls-passed'
        return report
    except Exception as exc:
        report.update(status='failed-strict-control', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        dump(out / 'results.json', report)



def run_case(bb, bound, *, width, request_started=None, stage_callback=None):
    """One continuous timed request; complete first3 callback and full6 endpoint."""
    validate_width(width)
    from .branch_local_copy import prepare_root
    from .fused_prefix import _complete_cache
    from .native_batch_diagnosis import state_hash
    started = time.perf_counter() if request_started is None else request_started
    stage_callback = stage_callback or (lambda stage: None)
    result = {}; stage_callback('public-metadata'); common, jobs = serving_jobs(bound, width)
    metadata_at = (time.perf_counter() - started) * 1000
    try:
        stage_callback('ordinary-root-prefill')
        state, preparation = prepare_root(bb, common); _complete_cache(state['cache'], len(state['tokens']))
        original = state_hash(state['cache']); admitted = (time.perf_counter() - started) * 1000
        def first_ready(completed):
            first = assemble(bound, completed, cards=3, require_native=True); serialized_payload(first)
            result['firstThree'] = first
            result['firstThreeReceipt'] = {'validatedAndSerializedAtMs': (time.perf_counter() - started) * 1000,
                'valid': first['validation']['valid'], 'payloadSha256': first['payloadSha256'], 'payloadBytes': first['payloadBytes']}
        stage_callback('independent-field-decode')
        rows, timing = decode_complete_rows(bb, state, jobs, width, first_ready)
        result['rawRows'] = rows; stage_callback('full-validation-serialization')
        full = assemble(bound, rows, cards=6, require_native=True); serialized_payload(full)
        full_at = (time.perf_counter() - started) * 1000
        if state_hash(state['cache']) != original:
            raise AssertionError('Complete ordinary source cache mutated during generation')
        result.update(status='completed-attempt', fullSix=full,
            fullSixReceipt={'validatedAndSerializedAtMs': full_at, 'payloadSha256': full['payloadSha256'],
                'payloadBytes': full['payloadBytes'], 'valid': full['validation']['valid']},
            metadataReadyAtMs=metadata_at, copyAdmittedAtMs=admitted, preparation=preparation, copyTiming=timing,
            sourceCacheBefore=original, sourceCacheAfter=state_hash(state['cache']), width=width, sampledCap=416,
            actualOutputTokens=sum(len(r['tokenIds']) for r in rows), actualArSteps=sum(r['arSteps'] for r in rows),
            requestTimingIncludesCallerRanking=request_started is not None,
            requestScope=('caller request start through copy ordinary root, validation, UTF8 serialization; browser paint excluded'
                if request_started is not None else 'resident fresh copy request including ordinary root, validation, UTF8 serialization; ranking and browser paint excluded'),
            memory=guard_memory(bb))
        return result
    except Exception as exc:
        result['rawRows'] = getattr(exc, 'partial_rows', result.get('rawRows', []))
        setattr(exc, 'partial_result', result)
        raise

