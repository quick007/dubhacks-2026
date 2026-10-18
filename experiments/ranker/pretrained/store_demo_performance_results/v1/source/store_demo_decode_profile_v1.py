"""Instrumented candidate copy of the live decoder; no serving source edits.

Not imported by the live service. Native owner must explicitly select this
candidate later, under the same model/width/inputs and exclusive worker lock.
Timers identify host-visible waits and phases, not isolated GPU kernel time.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import time

REFERENCE_SHA256 = '2b6157efbc60473f79d1f718aedb159601ed0b237706ba7e2d963766c60a2a74'
STAGES = ('generatorConstruction', 'cacheCloneMaterialization', 'suffixBoundaryPreparation',
    'generatorInsert', 'nextGeneratedCall', 'explicitGenerationStreamSync',
    'memoryGuards', 'firstSectionCallback', 'generatorCloseAndSync')


def assert_reference():
    from .store_demo_engine import decode_rows as live_decode
    current = hashlib.sha256(inspect.getsource(live_decode).encode()).hexdigest()
    if current != REFERENCE_SHA256:
        raise ValueError('Live decoder differs from reviewed profile reference; audit before comparison')
    return current


def native_dependencies():
    from mlx_lm.generate import BatchGenerator, StopSequences, generation_stream
    from mlx_lm.sample_utils import make_sampler
    from .parallel_decode import clone_prefix_state, exact_row_suffix
    from .store_copy_wide_v2 import guard_memory
    return {'BatchGenerator': BatchGenerator, 'StopSequences': StopSequences,
        'generation_stream': generation_stream, 'make_sampler': make_sampler,
        'clone_prefix_state': clone_prefix_state, 'exact_row_suffix': exact_row_suffix,
        'guard_memory': guard_memory}


def decode_rows_profiled(bb, state, jobs, width, first_ready, cancelled, *,
                         _dependencies=None, _clock=time.perf_counter):
    """Same admission order, sampling, forks, caps and callbacks as live decoder.

    Dependency/clock injection is for CPU tests only. No source monkeypatching,
    cached text, speculative model or scheduling change is performed.
    """
    from .store_demo_engine import RequestCancelled
    if width not in (6, 10, 20):
        raise ValueError('Explicit width 6, 10 or 20 required')
    reference = assert_reference()
    deps = native_dependencies() if _dependencies is None else _dependencies
    BatchGenerator, StopSequences = deps['BatchGenerator'], deps['StopSequences']
    stream, make_sampler = deps['generation_stream'], deps['make_sampler']
    clone_prefix_state, exact_row_suffix = deps['clone_prefix_state'], deps['exact_row_suffix']
    guard_memory = deps['guard_memory']
    mx = bb.mx; output = {}; active = {}; pending = list(enumerate(jobs)); first_done = False
    first_count = 1 + 3 * min(3, sum(r['kind'] == 'title' for r in jobs))
    started = _clock(); rounds = 0; stage_ms = {name: 0. for name in STAGES}
    profile = {'version': 'store-demo-native-decode-profile.v1', 'referenceDecodeSourceSha256': reference,
        'schedulingChanged': False, 'samplerChanged': False, 'rowsReordered': False,
        'GPUKernelPrefillDecodeSplitMeasured': False,
        'timerInterpretation': 'host wall durations; native work can be charged at nextGeneratedCall, explicit sync or internal materialization; nextGeneratedCall includes suffix prefill, cache batching, decode and implicit waits',
        'stageMs': stage_ms, 'admissions': [], 'iterations': [], 'maximumAdmittedActiveRows': 0,
        'maximumRowsEmittingTogether': 0, 'rowPreparation': [], 'firstSectionFieldCount': first_count,
        'firstSectionCallbackAtMs': None}
    begin = _clock()
    generator = BatchGenerator(bb.model, completion_batch_size=width, prefill_batch_size=width,
        prefill_step_size=256, sampler=make_sampler(temp=0.0))
    stage_ms['generatorConstruction'] += (_clock() - begin) * 1000
    def check_cancel():
        if cancelled():
            raise RequestCancelled('Client disconnected or cancelled stale request')
    def guarded_memory():
        begin = _clock()
        try:
            return guard_memory(bb)
        finally:
            stage_ms['memoryGuards'] += (_clock() - begin) * 1000
    def admit():
        check_cancel()
        count = min(width - len(active), len(pending))
        if count <= 0:
            return
        group = [pending.pop(0) for _ in range(count)]
        prompts, caches = [], []
        admission_started = _clock()
        for _, row in group:
            begin = _clock(); branch = clone_prefix_state(bb, state)
            clone_ms = (_clock() - begin) * 1000
            stage_ms['cacheCloneMaterialization'] += clone_ms
            begin = _clock(); suffix, cache = exact_row_suffix(bb.tokenizer, row, branch, bb.model)
            suffix_ms = (_clock() - begin) * 1000
            stage_ms['suffixBoundaryPreparation'] += suffix_ms
            if not 1 <= len(suffix) <= 256 or len(state['tokens']) + len(suffix) + row['maxTokens'] > 4096:
                raise ValueError('Complete public field exceeds bounded context; no truncation')
            prompts.append(suffix); caches.append(cache)
            profile['rowPreparation'].append({'field': row['fields'][0], 'cacheCloneMaterializationMs': clone_ms,
                'suffixBoundaryPreparationMs': suffix_ms, 'suffixTokens': len(suffix)})
        stops = [[int(token)] for token in bb.tokenizer.eos_token_ids]
        begin = _clock()
        uids = generator.insert(prompts, max_tokens=[row['maxTokens'] for _, row in group], caches=caches,
            all_tokens=[list(state['tokens']) for _ in group], samplers=[make_sampler(temp=0.0) for _ in group],
            logits_processors=[[] for _ in group], stop_sequences=[StopSequences(stops) for _ in group])
        insert_ms = (_clock() - begin) * 1000
        stage_ms['generatorInsert'] += insert_ms
        for uid, (index, row), suffix in zip(uids, group, prompts, strict=True):
            active[uid] = index
            output[index] = {**deepcopy(row), 'tokenIds': [], 'rawSampledTokenIds': [],
                'stopTokenId': None, 'arSteps': 0, 'finishReason': None, 'suffixTokens': len(suffix)}
        profile['maximumAdmittedActiveRows'] = max(profile['maximumAdmittedActiveRows'], len(active))
        profile['admissions'].append({'count': count, 'fields': [row['fields'][0] for _, row in group],
            'activeAfterAdmission': len(active), 'suffixTokens': [len(suffix) for suffix in prompts],
            'pendingAfterAdmission': len(pending), 'atMs': (_clock() - started) * 1000,
            'admissionWallMs': (_clock() - admission_started) * 1000,
            'belowFoldBeforeFirstSectionComplete': not first_done and any(index >= first_count for index, _ in group)})
        del caches, cache, branch
        guarded_memory()
    try:
        admit()
        while active:
            check_cancel()
            if _clock() - started > 120:
                raise TimeoutError('Native copy exceeded 120-second demo request budget')
            live_before = len(active)
            begin = _clock(); responses = generator.next_generated()
            next_ms = (_clock() - begin) * 1000; stage_ms['nextGeneratedCall'] += next_ms
            begin = _clock(); mx.synchronize(stream)
            sync_ms = (_clock() - begin) * 1000; stage_ms['explicitGenerationStreamSync'] += sync_ms
            if not responses:
                raise RuntimeError('Native generator stopped with unfinished fields')
            rounds += 1
            profile['maximumRowsEmittingTogether'] = max(profile['maximumRowsEmittingTogether'], len(responses))
            completed = []
            for response in responses:
                index = active[response.uid]; record = output[index]
                record['arSteps'] += 1; record['rawSampledTokenIds'].append(int(response.token))
                if response.finish_reason != 'stop':
                    record['tokenIds'].append(int(response.token))
                if response.finish_reason is not None:
                    record['finishReason'] = response.finish_reason
                    record['stopTokenId'] = int(response.token) if response.finish_reason == 'stop' else None
                    record['text'] = bb.tokenizer.decode(record['tokenIds'])
                    record['completedMs'] = (_clock() - started) * 1000
                    completed.append(record['fields'][0]); del active[response.uid]
            profile['iterations'].append({'round': rounds, 'activeBefore': live_before,
                'rowsEmitting': len(responses), 'finishedFields': completed, 'activeAfter': len(active),
                'nextGeneratedCallMs': next_ms, 'explicitSyncMs': sync_ms, 'atMs': (_clock() - started) * 1000})
            del responses, response
            guarded_memory(); check_cancel()
            if not first_done and all(i in output and output[i]['finishReason'] is not None for i in range(first_count)):
                begin = _clock(); first_ready([output[i] for i in range(first_count)])
                stage_ms['firstSectionCallback'] += (_clock() - begin) * 1000
                profile['firstSectionCallbackAtMs'] = (_clock() - started) * 1000
                first_done = True
            admit()
        if len(output) != len(jobs) or not first_done:
            raise RuntimeError('Native field roster incomplete')
        decode_ms = (_clock() - started) * 1000
        result = {'decodeRounds': rounds, 'decodeWallMs': decode_ms, 'width': width,
            'sampledTokenCap': sum(row['maxTokens'] for row in jobs),
            'actualSampledTokens': sum(row['arSteps'] for row in output.values()), 'profile': profile}
        profile['status'] = 'completed'; profile['decodeWallMsBeforeClose'] = decode_ms
        return [output[i] for i in range(len(jobs))], result
    except BaseException as error:
        profile.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        error.partial_rows = [output[i] for i in sorted(output)]
        error.partial_profile = profile
        raise
    finally:
        begin = _clock(); generator.close(); mx.synchronize(stream)
        stage_ms['generatorCloseAndSync'] += (_clock() - begin) * 1000
        profile['profileWallMsIncludingClose'] = (_clock() - started) * 1000
        profile['accountedStageMsIncludingClose'] = sum(stage_ms.values())
        profile['otherHostBookkeepingMs'] = profile['profileWallMsIncludingClose'] - profile['accountedStageMsIncludingClose']
