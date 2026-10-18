"""Actual-store final-down fit from the frozen base, with exact native gates.

Import and --preflight are CPU/tokenizer only. The native CLI requires the
root-owner lock and exact reviewed preflight. Earlier candidates stay intact.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import signal
import subprocess
import time

from .semantic_copy_feature_fit import (MODEL, ROOT, digest, dump, feature_bytes,
    hashed, load_feature, model_input_receipt, verify_model_files)
from .short_copy_fit import decode_complete_rows

DATA = ROOT / 'store_copy_v2'
PRIOR = Path('data/ranker/pretrained/short-copy-fit-v1/adapter.safetensors')
PRIOR_SHA = '85ce8215add1d4ce561441d6972c8e2958dbb73966bd89f53595ef65b7fddb57'
MODES = ('frozen', 'failed-daypack', 'new-store')
WIDTH = 6
FIELDS = ('S1', 'T1', 'B1', 'C1', 'T2', 'B2', 'C2', 'T3', 'B3', 'C3',
          'S2', 'T4', 'B4', 'C4', 'T5', 'B5', 'C5', 'T6', 'B6', 'C6')
CONFIG = {'seed': 19079, 'updates': 320, 'epochs': 1, 'rank': 4, 'scale': 20.,
    'learningRate': .0001, 'memoryGiB': 7, 'maximumContext': 4096,
    'maximumRoot': 2048, 'maximumSuffix': 256, 'prefillStep': 256,
    'singleHostFeatureBudgetBytes': 32 * 2**20,
    'totalFeatureArtifactBudgetBytes': 5 * 2**30, 'residentHostFeatureRows': 1,
    'trainCases': 16, 'evalCases': 8, 'fieldsPerCase': 20, 'sampledCapPerCase': 416,
    'width': 6, 'batchSize': 1, 'gateTimeoutSeconds': 120, 'updateTimeoutSeconds': 60,
    'requestTimeoutSeconds': 120, 'controlTimeoutSeconds': 180,
    'selection': 'final fixed step; no evaluation checkpoint selection',
    'order': 'one seeded permutation of all 320 individual train rows',
    'initialization': 'frozen base; seeded final-down rank4 A and exactly zero B',
    'gates': 'all 320 exact native/auxiliary/feature loss and A/B-gradient gates; no tolerance',
    'probeControl': 'fixed original A; temporary B=1e-5 FP32; restore exact zero-B initial after each probe',
    'evaluation': 'fixed 24 arms: all 8 evaluation requests x frozen/failed-daypack/new-store at width6'}
SOURCES = ('store_copy_fit_v2.py', 'test_store_copy_fit_v2.py', 'store_copy_data_v2.py',
    'store_copy_diagnostic.py', 'store_catalog_v1.py', 'store_model_bridge.py',
    'branch_local_copy.py', 'short_copy_fit.py', 'semantic_copy_feature_fit.py',
    'semantic_copy_gradient_features.py', 'semantic_copy_tokens.py',
    'semantic_copy_training.py', 'parallel_decode.py', 'fused_prefix.py',
    'native_batch_diagnosis.py', 'backbone.py', 'evidence_plan.py', 'fusion.py',
    'catalog_fusion/generalization_v2.py')


def shuffled_order(count=320, seed=CONFIG['seed']):
    if type(count) is not int or count != 320:
        raise ValueError('Exactly 320 individual updates required')
    order = list(range(count)); random.Random(seed).shuffle(order)
    return order


@contextmanager
def deadline(seconds, label):
    """Native main-thread wall guard, preserving the caller's signal state."""
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError('Existing process alarm prevents an independent native deadline')
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError(label + ' exceeded fixed ' + str(seconds) + '-second deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def temporary_probe_control(restore, receipts, initial, control, initial_receipt, control_receipt):
    """Always restore exact true initialization, including a failed gate."""
    if receipts() != initial_receipt:
        raise AssertionError('True zero-B initialization was not restored before feature extraction')
    try:
        restore(control)
        if receipts() != control_receipt:
            raise AssertionError('Fixed numerical gate parameters changed')
        yield
    finally:
        restore(initial)
        if receipts() != initial_receipt:
            raise AssertionError('Exact true zero-B restoration failed after gate')


def serving_rows(case, fields):
    """Rebuild inference inputs without exposing any teacher/evaluation labels."""
    from .store_copy_data_v2 import layout
    common, jobs = layout(case['bound'])
    if common.get('role') != 'user' or tuple(r['fields'][0] for r in jobs) != FIELDS:
        raise ValueError('Ordinary chat root and complete twenty-field serving roster required')
    if len(jobs) != 20 or sum(r['maxTokens'] for r in jobs) != 416:
        raise ValueError('All twenty fields and unchanged 416-token sampled cap required')
    by_id = {f['rowId']: f for f in fields}
    if len(fields) != 20 or len(by_id) != 20 or set(by_id) != {r['rowId'] for r in jobs}:
        raise ValueError('Exact frozen teacher field membership required')
    ordered = []
    for job in jobs:
        field = by_id[job['rowId']]
        if len(job['messages']) != 1 or job['messages'][0].get('role') != 'user':
            raise ValueError('One ordinary field-user suffix required')
        if (field['messages'], field['maxTokens'], field['kind'], field['field']) != (
                job['messages'], job['maxTokens'], job['kind'], job['fields'][0]):
            raise ValueError('Frozen teacher input differs from actual serving renderer')
        if any(k in job for k in ('targetValue', 'teacherOnlySyntheticStop', 'text')):
            raise ValueError('Teacher targets/metadata must never enter inference jobs')
        ordered.append(field)
    return common, jobs, ordered


def dataset_packet(data=DATA):
    from .store_copy_data_v2 import load_dataset
    from .store_catalog_v1 import load_snapshot
    from .store_model_bridge import assert_current_sources
    assert_current_sources(load_snapshot()['receipt'])
    manifest, cases, fields = load_dataset(data)
    packet = {s: {'cases': cases[s], 'fields': fields[s]} for s in ('train', 'eval')}
    seen = set()
    for split, n in (('train', 16), ('eval', 8)):
        group = packet[split]; ids = {c['caseId'] for c in group['cases']}
        if len(group['cases']) != n or len(ids) != n or len(group['fields']) != n * 20 or seen & ids:
            raise ValueError('Fixed disjoint 16train/8eval and 320/160 fields required')
        if {f['caseId'] for f in group['fields']} != ids:
            raise ValueError('Field/case split membership differs')
        seen |= ids
        for case in group['cases']:
            serving_rows(case, [f for f in group['fields'] if f['caseId'] == case['caseId']])
    return manifest, packet


def prior_receipt():
    from .store_copy_diagnostic import completed_copy_adapter
    if digest(PRIOR) != PRIOR_SHA:
        raise ValueError('Preserved failed-daypack checkpoint checksum differs')
    receipt = completed_copy_adapter(PRIOR.parent / 'results.json')
    if receipt['checkpointSha256'] != PRIOR_SHA:
        raise ValueError('Prior completed training receipt differs')
    return receipt


def preflight(tokenizer, data=DATA):
    """Exact serving chat tokens, including every evaluation target cap, no MLX."""
    from .semantic_copy_gradient_features import build_native_contract
    manifest, packet = dataset_packet(data)
    result = {'schemaVersion': 'actual-store-copy-fit.v2', 'status': 'tokenizer-preflight-passed',
        'nativeExecution': False, 'modelCalls': 0, 'trainingUpdates': 0,
        'config': CONFIG, 'evalModes': list(MODES), 'widths': [WIDTH], 'fixedCases': 8, 'fixedArms': 24,
        'datasetManifestSha256': digest(data / 'manifest.json'), 'payloadSha256': manifest['fileSha256'],
        'sources': {name: digest(ROOT / name) for name in SOURCES}, 'modelFiles': model_input_receipt(),
        'prior': prior_receipt(), 'splits': {},
        'limitations': ['Same actual-store catalog; inspected synthetic labels; no unseen catalog or sealed-holdout claim.',
            'Locked products/color/image metadata; copy fit does not learn product selection or image understanding.',
            'Fresh ordinary chat root per mode and request; no ranking prefix or cross-mode cache reuse.',
            'Readiness includes complete required copy fields, actual validation and UTF8 serialization; excludes ranking, images and browser paint.',
            'Eight fixed evaluation cases per mode; invalid attempts never become valid latency samples.',
            'No templates, fallback text, target-fed inference, truncation or reduced card count.',
            'Temporary nonzero-B adapter is a numerical gate control only; all optimizer updates start from exact frozen-base zero-B initialization.']}
    for split, group in packet.items():
        row_receipts = []; case_receipts = []
        for case in group['cases']:
            common, jobs, ordered = serving_rows(case, [f for f in group['fields'] if f['caseId'] == case['caseId']])
            root = tokenizer.apply_chat_template([common], tokenize=True, return_dict=False,
                add_generation_prompt=False, enable_thinking=False)
            if not 1 <= len(root) <= CONFIG['maximumRoot']:
                raise ValueError('Complete ordinary chat root exceeds explicit cap; no truncation')
            state = {'messages': [common], 'tokens': root, 'consumed': False}
            maximum = 0
            for job, field in zip(jobs, ordered, strict=True):
                contract = build_native_contract(tokenizer, state, field)
                prompt = tokenizer.apply_chat_template([common] + job['messages'], tokenize=True,
                    return_dict=False, add_generation_prompt=True, enable_thinking=False)
                if prompt != contract['promptTokenIds'] or len(prompt) + job['maxTokens'] > CONFIG['maximumContext']:
                    raise ValueError('Serving/teacher prompt mismatch or generation context cap exceeded')
                size = feature_bytes(len(contract['suffixTokenIds']))
                if size > CONFIG['singleHostFeatureBudgetBytes']:
                    raise MemoryError('One complete host feature row exceeds cap')
                maximum = max(maximum, len(contract['suffixTokenIds']))
                row_receipts.append({'caseId': case['caseId'], 'rowId': field['rowId'], 'field': field['field'],
                    'kind': field['kind'], 'maxTokens': field['maxTokens'], 'suffixTokens': len(contract['suffixTokenIds']),
                    'promptTokens': len(prompt), 'targetTokens': len(contract['targetTokenIds']),
                    'fullTeacherTokens': contract['fullTokens'], 'projectedHostFeatureBytes': size,
                    **{k: contract[k] for k in ('promptSha256', 'modelInputSha256', 'sourceTokenSha256', 'predictionStartInBranch')}})
            case_receipts.append({'caseId': case['caseId'], 'boundSha256': hashed(case['bound']),
                'rootTokens': len(root), 'rootTokenSha256': hashed(root), 'maximumSuffix': maximum})
        result['splits'][split] = {'caseCount': len(case_receipts), 'fieldCount': len(row_receipts),
            'cases': case_receipts, 'rows': row_receipts,
            'maximumRootTokens': max(c['rootTokens'] for c in case_receipts),
            'maximumSuffixTokens': max(r['suffixTokens'] for r in row_receipts),
            'projectedHostFeatureBytes': sum(r['projectedHostFeatureBytes'] for r in row_receipts)}
    projected = result['splits']['train']['projectedHostFeatureBytes'] + 2**20
    if projected > CONFIG['totalFeatureArtifactBudgetBytes']:
        raise MemoryError('Uncompressed complete train feature archive exceeds 5-GiB cap')
    result['projectedUncompressedTrainingArtifactsWithReserveBytes'] = projected
    result['globalPermutationSha256'] = hashed(shuffled_order())
    result['gateWork'] = {'trainRows': 320, 'evaluationRowsExtracted': 0,
        'nativeSuffixAutodiffPasses': 640, 'featureAutodiffPasses': 320, 'exactComparisons': 960}
    return result


def arm_roster(packet):
    return [{'caseId': c['caseId'], 'split': 'eval', 'mode': mode, 'width': WIDTH,
        'status': 'not-attempted'} for c in packet['eval']['cases'] for mode in MODES]


def summary(arms):
    result = {}
    for mode in MODES:
        selected = [a for a in arms if a['mode'] == mode]
        done = [a for a in selected if a['status'] == 'completed-attempt']
        first = [a for a in done if a['firstThree']['validation']['valid']]
        full = [a for a in done if a['fullSix']['validation']['valid']]
        result[mode] = {'fixedCases': 8, 'completed': len(done),
            'validThreeCardResults': len(first), 'validThreeCardRateFixedDenominator': len(first) / 8,
            'validSixCardResults': len(full), 'validSixCardRateFixedDenominator': len(full) / 8,
            **{name: sum(a['status'] == status for a in selected) for name, status in
                (('failed', 'failed'), ('unattempted', 'not-attempted'), ('inProgress', 'running'))},
            'attemptFirstThreeReadyMs': [a['firstThreeReceipt']['validatedAndSerializedAtMs'] for a in done],
            'validFirstThreeReadyMs': [a['firstThreeReceipt']['validatedAndSerializedAtMs'] for a in first],
            'attemptFullSixReadyMs': [a['fullSixReceipt']['validatedAndSerializedAtMs'] for a in done],
            'validFullSixReadyMs': [a['fullSixReceipt']['validatedAndSerializedAtMs'] for a in full],
            'rankingQualityClaim': False, 'browserPaintMeasured': False}
    return result


def encoded_payload(result):
    from .store_copy_data_v2 import canonical
    value = canonical(result['payload']).encode()
    if hashlib.sha256(value).hexdigest() != result['payloadSha256'] or len(value) != result['payloadBytes']:
        raise AssertionError('Actual UTF8 serialized payload receipt differs')
    return value


def run_case(bb, bound, *, request_started=None, stage_callback=None):
    """One continuous timed request; complete first3 callback and full6 endpoint."""
    from .store_copy_data_v2 import assemble, layout
    from .branch_local_copy import prepare_root
    from .fused_prefix import _complete_cache
    from .native_batch_diagnosis import state_hash
    started = time.perf_counter() if request_started is None else request_started
    stage_callback = stage_callback or (lambda stage: None)
    result = {}; stage_callback('public-metadata'); common, jobs = layout(bound)
    metadata_at = (time.perf_counter() - started) * 1000
    try:
        stage_callback('ordinary-root-prefill')
        state, preparation = prepare_root(bb, common); _complete_cache(state['cache'], len(state['tokens']))
        original = state_hash(state['cache']); admitted = (time.perf_counter() - started) * 1000
        def first_ready(completed):
            first = assemble(bound, completed, cards=3, require_native=True); encoded_payload(first)
            result['firstThree'] = first
            result['firstThreeReceipt'] = {'validatedAndSerializedAtMs': (time.perf_counter() - started) * 1000,
                'valid': first['validation']['valid'], 'payloadSha256': first['payloadSha256'], 'payloadBytes': first['payloadBytes']}
        stage_callback('independent-field-decode')
        rows, timing = decode_complete_rows(bb, state, jobs, WIDTH, first_ready)
        result['rawRows'] = rows; stage_callback('full-validation-serialization')
        full = assemble(bound, rows, cards=6, require_native=True); encoded_payload(full)
        full_at = (time.perf_counter() - started) * 1000
        if state_hash(state['cache']) != original:
            raise AssertionError('Complete ordinary source cache mutated during generation')
        result.update(status='completed-attempt', fullSix=full,
            fullSixReceipt={'validatedAndSerializedAtMs': full_at, 'payloadSha256': full['payloadSha256'],
                'payloadBytes': full['payloadBytes'], 'valid': full['validation']['valid']},
            metadataReadyAtMs=metadata_at, copyAdmittedAtMs=admitted, preparation=preparation, copyTiming=timing,
            sourceCacheBefore=original, sourceCacheAfter=state_hash(state['cache']), width=WIDTH, sampledCap=416,
            actualOutputTokens=sum(len(r['tokenIds']) for r in rows), actualArSteps=sum(r['arSteps'] for r in rows),
            requestTimingIncludesCallerRanking=request_started is not None,
            requestScope='resident fresh copy request including ordinary root, validation, UTF8 serialization; excludes ranking and browser paint',
            memory=bb.runtime_memory())
        return result
    except Exception as exc:
        result['rawRows'] = getattr(exc, 'partial_rows', result.get('rawRows', []))
        setattr(exc, 'partial_result', result)
        raise


def native_controls(bb, bound, out):
    """Actual width6/original-cap fork, independent rebuild and fork repeat.

    Hash every full native log-probability vector including EOS, preserving
    mismatch evidence before failing. Controls are outside request latency.
    No trimming, cache splice, token repair, tolerance, or cross-shape equality.
    """
    from .store_copy_data_v2 import layout
    from mlx_lm.generate import BatchGenerator, StopSequences, generation_stream
    from mlx_lm.sample_utils import make_sampler
    from .branch_local_copy import prepare_root
    from .fused_prefix import _complete_cache
    from .native_batch_diagnosis import state_hash
    from .parallel_decode import clone_prefix_state, exact_row_suffix
    out = Path(out)
    if out.exists():
        raise FileExistsError('Preserve prior exact-control evidence')
    out.mkdir(parents=True); common, jobs = layout(bound); jobs = jobs[:WIDTH]
    report = {'status': 'running', 'width': WIDTH, 'rows': copy.deepcopy(jobs), 'arms': [],
              'originalCaps': [r['maxTokens'] for r in jobs], 'latencyClaim': False}
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
            generator = BatchGenerator(bb.model, completion_batch_size=WIDTH, prefill_batch_size=WIDTH,
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
                    bb.check_memory()
            finally:
                generator.close(); mx.synchronize(generation_stream)
            arm = {'label': label, 'rows': records, 'maximumRowsEmittingTogether': maximum,
                'sourceAfter': state_hash(source['cache']), 'memory': bb.runtime_memory()}
            report['arms'].append(arm); report.pop('activeArm', None); dump(out / (label + '.json'), arm)
            if maximum != WIDTH or any(r['finishReason'] is None for r in records):
                raise AssertionError('Control did not execute every row at actual width6')
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


def assert_source_fresh(frozen, data):
    from .store_catalog_v1 import load_snapshot
    from .store_model_bridge import assert_current_sources
    assert_current_sources(load_snapshot()['receipt'])
    for name, sha in frozen['sources'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Frozen producer source changed: ' + name)
    if digest(data / 'manifest.json') != frozen['datasetManifestSha256']:
        raise ValueError('Frozen dataset manifest changed')
    for name, sha in frozen['payloadSha256'].items():
        if digest(data / name) != sha:
            raise ValueError('Frozen dataset payload changed: ' + name)
    if prior_receipt() != frozen['prior']:
        raise ValueError('Preserved prior checkpoint/report changed')


def run(out, preflight_path, data=DATA):
    """Serialized root-owned native extraction, fixed fit, and all24 arms."""
    function_started = time.perf_counter()
    import numpy as np
    import psutil
    from transformers import AutoTokenizer
    out, preflight_path, data = Path(out), Path(preflight_path), Path(data)
    if out.exists():
        raise FileExistsError('Preserve previous native attempt; use a new output directory')
    frozen = json.loads(preflight_path.read_text())
    # Establish an auditable failure receipt before expensive tokenizer/hashes.
    out.mkdir(parents=True); (out / 'features').mkdir(); (out / 'gates').mkdir()
    # Even preflight/tokenizer failures preserve the full fixed arm denominator.
    frozen_case_ids = [c['caseId'] for c in frozen.get('splits', {}).get('eval', {}).get('cases', [])]
    if len(frozen_case_ids) != 8 or len(set(frozen_case_ids)) != 8:
        raise ValueError('Reviewed preflight requires exactly eight distinct evaluation case IDs')
    arms = [{'caseId': case_id, 'split': 'eval', 'mode': mode, 'width': WIDTH, 'status': 'not-attempted'}
        for case_id in frozen_case_ids for mode in MODES]
    report = {'schemaVersion': 'actual-store-copy-fit.v2', 'status': 'started', 'phase': 'preflight-verification',
        'config': CONFIG, 'executionCommit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'preflightSha256': digest(preflight_path), 'preflight': frozen,
        'nativeUpdates': 0, 'optimizerUpdateCalls': 0, 'extraction': [], 'steps': [],
        'arms': arms, 'fixedArms': 24, 'controls': {}, 'featureArtifactBytes': 0,
        'wallScope': 'run entry including tokenizer/preflight/shard verification; excludes Python startup',
        'limitations': frozen.get('limitations', [])}
    def save(event=None):
        report['wallMs'] = (time.perf_counter() - function_started) * 1000
        report['summary'] = summary(arms); dump(out / 'results.json', report)
        if event:
            print(json.dumps({'event': event, 'phase': report['phase'], 'gates': len(report['extraction']),
                'updates': report['nativeUpdates'], 'completedArms': sum(a['status'] == 'completed-attempt' for a in arms),
                'wallMs': report['wallMs']}), flush=True)
    save('started')
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        if preflight(tokenizer, data) != frozen:
            raise ValueError('Current source/data/tokenizer differs from frozen preflight')
        _, packet = dataset_packet(data)
        if arms != arm_roster(packet):
            raise ValueError('Frozen preflight evaluation roster differs from current dataset')
        save()
        available = psutil.virtual_memory()
        report['hostMemoryBeforeLoad'] = {'availableBytes': available.available, 'totalBytes': available.total,
            'percent': available.percent, 'swap': psutil.swap_memory()._asdict()}
        if available.available < CONFIG['memoryGiB'] * 2**30:
            raise MemoryError('Less than declared 7 GiB available before native load')
        report['phase'] = 'verify-model'; save('verify-model')
        report['modelVerification'] = verify_model_files(frozen['modelFiles']); save()
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from mlx_lm.tuner.lora import LoRALinear
        from mlx_lm.generate import generation_stream
        from .backbone import FrozenBackbone
        from .evidence_plan import REVISION
        from .branch_local_copy import prepare_root
        from .fused_prefix import _complete_cache
        from .native_batch_diagnosis import state_hash
        from .semantic_copy_training import parameter_hashes, disabled
        from .semantic_copy_gradient_features import (FEATURE_KEYS, GradientFeatureProbeError,
            probe_gradient_features, feature_loss)
        class GuardedBackbone(FrozenBackbone):
            def check_memory(self):
                super().check_memory()
                if max(self.mx.get_peak_memory(), self.mx.get_active_memory()) > self.memory_budget:
                    raise MemoryError('Native active/peak exceeded declared 7 GiB')
        native_context = mx.stream(generation_stream); native_context.__enter__()
        mx.random.seed(CONFIG['seed']); bb = GuardedBackbone(MODEL, REVISION, 4096, CONFIG['memoryGiB'])
        model = bb.model; model.freeze(); model.eval(); report['residentLoadMs'] = bb.load_ms
        base_hashes = parameter_hashes(model.language_model)
        if len(base_hashes) != 927 or dict(tree_flatten(model.trainable_parameters())):
            raise AssertionError('Exactly 927 frozen language tensors and no base trainables required')
        report['frozenBaseParameterHashes'] = base_hashes
        adapter = LoRALinear.from_base(model.layers[-1].mlp.down_proj, r=4, scale=20, dropout=0)
        model.layers[-1].mlp.down_proj = adapter; adapter.lora_a = adapter.lora_a.astype(mx.float32)
        adapter.lora_b = mx.zeros((4, 4096), dtype=mx.float32); model.eval()
        def leaves():
            return dict(tree_flatten(adapter.trainable_parameters()))
        def parameter_receipts():
            result = {}
            for k, v in (('lora_a', adapter.lora_a), ('lora_b', adapter.lora_b)):
                if v.dtype != mx.float32:
                    raise ValueError('Adapter A/B must remain float32')
                host = np.array(v)
                if not np.isfinite(host).all():
                    raise FloatingPointError('Nonfinite adapter parameter')
                result[k] = {'dtype': str(v.dtype), 'shape': list(v.shape),
                    'sha256': hashlib.sha256(host.tobytes()).hexdigest()}
            return result
        def restore(parameters):
            for k, v in parameters.items():
                setattr(adapter, k, v)
            mx.eval(adapter.parameters()); mx.synchronize()
        def load_adapter(path):
            weights = mx.load(str(path))
            if set(weights) != {'lora_a', 'lora_b'} or tuple(weights['lora_a'].shape) != (12288, 4) or tuple(weights['lora_b'].shape) != (4, 4096):
                raise ValueError('Only final-down rank4 A/B checkpoint shapes allowed')
            restore(weights); parameter_receipts()
        def verify_base(stage):
            current = {k.replace('.31.mlp.down_proj.linear.', '.31.mlp.down_proj.'): v
                for k, v in parameter_hashes(model.language_model).items()}
            if current != base_hashes:
                raise AssertionError('Frozen language bytes changed at ' + stage)
            report['controls'][stage + 'Frozen927Unchanged'] = True
        roster = dict(tree_flatten(model.trainable_parameters()))
        if len(roster) != 2 or sum(v.size for v in roster.values()) != 65536 or any(
                '.31.mlp.down_proj.' not in k or k.rsplit('.', 1)[-1] not in ('lora_a', 'lora_b') for k in roster):
            raise AssertionError('Only layer31 final-down A/B, 65536 trainable floats required')
        mx.eval(adapter.parameters()); mx.synchronize()
        initial_params = leaves(); initial_receipt = parameter_receipts()
        initial_host = {k: np.array(v) for k, v in initial_params.items()}
        if np.any(initial_host['lora_b'] != 0):
            raise AssertionError('Frozen-base primary arm requires exactly zero initial B')
        control_params = {'lora_a': initial_params['lora_a'], 'lora_b': mx.full((4, 4096), 1e-5, dtype=mx.float32)}
        restore(control_params); control_receipt = parameter_receipts(); restore(initial_params)
        report['trainableParameters'] = {k: int(v.size) for k, v in roster.items()}
        report['initialAdapter'] = initial_receipt; report['numericalProbeControl'] = control_receipt
        report['controls']['initialBExactlyZero'] = True
        planned = {(r['caseId'], r['rowId']): r for r in frozen['splits']['train']['rows']}
        report['phase'] = '320-exact-gates'; save('extraction-started')
        for case in packet['train']['cases']:
            common, _, fields = serving_rows(case, [f for f in packet['train']['fields'] if f['caseId'] == case['caseId']])
            restore(initial_params)
            initial_state, _ = prepare_root(bb, common); _complete_cache(initial_state['cache'], len(initial_state['tokens']))
            initial_source_hash = state_hash(initial_state['cache']); del initial_state
            restore(control_params)
            state, prep = prepare_root(bb, common); _complete_cache(state['cache'], len(state['tokens']))
            source_hash = state_hash(state['cache'])
            if source_hash != initial_source_hash:
                raise AssertionError('Initial/probe complete ordinary prefix cache differs')
            restore(initial_params)
            report['controls'].setdefault('initialVsProbeSourceCacheExact', []).append({'caseId': case['caseId'], 'sha256': source_hash})
            for field in fields:
                index = len(report['extraction']); begin = time.perf_counter(); plan = planned[(case['caseId'], field['rowId'])]
                gate_path = out / 'gates' / f'{index:04d}.json'
                report['activeGate'] = {'index': index, 'caseId': case['caseId'], 'rowId': field['rowId'],
                    'startedAtWallMs': (begin - function_started) * 1000, 'status': 'running'}; save()
                try:
                    with temporary_probe_control(restore, parameter_receipts, initial_params, control_params,
                            initial_receipt, control_receipt), deadline(CONFIG['gateTimeoutSeconds'], 'feature gate'):
                        gate, item = probe_gradient_features(bb, adapter, state, field)
                except GradientFeatureProbeError as exc:
                    dump(gate_path, exc.report)
                    report['failedGate'] = {**report['activeGate'], 'receipt': str(gate_path)}
                    raise
                dump(gate_path, gate)
                if gate['status'] != 'exact-gate-passed' or item is None:
                    report['failedGate'] = {**report['activeGate'], 'receipt': str(gate_path)}
                    raise AssertionError('Strict row gate failed; zero optimizer updates permitted')
                if any(gate['contract'][k] != plan[k] for k in ('modelInputSha256', 'sourceTokenSha256', 'promptSha256')):
                    raise AssertionError('Native teacher tokens differ from exact serving preflight')
                size = sum(item[k].nbytes for k in FEATURE_KEYS)
                if size != plan['projectedHostFeatureBytes'] or size > CONFIG['singleHostFeatureBudgetBytes']:
                    raise MemoryError('Complete host feature row violates exact preflight')
                archive = out / 'features' / f'{index:04d}.npz'; metadata = archive.with_suffix('.json')
                np.savez_compressed(archive, **{k: item[k] for k in FEATURE_KEYS})
                dump(metadata, {k: v for k, v in item.items() if k not in FEATURE_KEYS})
                report['featureArtifactBytes'] += archive.stat().st_size + metadata.stat().st_size
                if report['featureArtifactBytes'] > CONFIG['totalFeatureArtifactBudgetBytes']:
                    raise MemoryError('Complete feature archive/metadata exceeds declared 5-GiB budget')
                if state_hash(state['cache']) != source_hash:
                    raise AssertionError('Complete train source cache mutated during feature gate')
                report['extraction'].append({'index': index, 'caseId': case['caseId'], 'rowId': field['rowId'], 'kind': field['kind'],
                    'archive': str(archive.relative_to(out)), 'metadata': str(metadata.relative_to(out)),
                    'archiveSha256': digest(archive), 'metadataSha256': digest(metadata), 'gateSha256': digest(gate_path),
                    'featureTensors': gate['featureTensors'], 'hostFeatureBytes': size, 'sourceStateHash': source_hash,
                    'modelInputSha256': plan['modelInputSha256'], 'gateExact': True,
                    'fixedNonzeroBControl': control_receipt, 'trueZeroBRestored': True,
                    'rootPreparation': prep, 'ms': (time.perf_counter() - begin) * 1000})
                report.pop('activeGate', None)
                del item, gate; gc.collect(); mx.clear_cache(); save('gate-completed')
            del state; gc.collect(); mx.clear_cache()
        if len(report['extraction']) != 320 or report['nativeUpdates'] or parameter_receipts() != initial_receipt:
            raise AssertionError('All320 exact gates and exact true initial adapter must precede every optimizer update')
        verify_base('postExtraction'); assert_source_fresh(frozen, data)
        report['controls']['all320NonzeroBGatesPassed'] = True
        report['controls']['fitStartedFromExactFrozenBaseZeroB'] = True
        optimizer = optim.Adam(learning_rate=CONFIG['learningRate'])
        grad_fn = nn.value_and_grad(adapter, lambda item: feature_loss(bb, adapter, item))
        order = shuffled_order(); report['permutation'] = order; report['phase'] = '320-updates'; save('fit-started')
        for step, index in enumerate(order):
            begin = time.perf_counter(); entry = report['extraction'][index]; item = load_feature(entry, out)
            with deadline(CONFIG['updateTimeoutSeconds'], 'feature loss and gradients'):
                loss, gradients = grad_fn(item); mx.eval(loss, gradients); mx.synchronize()
            flat = dict(tree_flatten(gradients))
            if set(flat) != {'lora_a', 'lora_b'} or not bool(mx.isfinite(loss).item()) or not all(bool(mx.all(mx.isfinite(v)).item()) for v in flat.values()):
                raise FloatingPointError('Wrong/nonfinite feature loss or A/B gradient roster')
            row = {'step': step, 'featureIndex': index, 'caseId': entry['caseId'], 'rowId': entry['rowId'], 'kind': entry['kind'],
                'loss': float(loss.item()), 'gradientL2': sum(float(mx.sum(v.astype(mx.float32)**2).item()) for v in flat.values())**.5,
                'status': 'update-attempted', 'updateEvaluated': False}
            report['steps'].append(row); optimizer.update(adapter, gradients); report['optimizerUpdateCalls'] += 1
            with deadline(CONFIG['updateTimeoutSeconds'], 'optimizer update evaluation'):
                mx.eval(adapter.parameters(), optimizer.state); mx.synchronize(); report['nativeUpdates'] += 1; bb.check_memory()
            arrays = list(leaves().values()) + [v for _, v in tree_flatten(optimizer.state) if hasattr(v, 'dtype')]
            if not all(bool(mx.all(mx.isfinite(v)).item()) for v in arrays):
                raise FloatingPointError('Nonfinite adapter/optimizer state after update')
            row.update(status='completed', updateEvaluated=True, ms=(time.perf_counter() - begin) * 1000)
            del item, gradients, loss, flat, arrays; save('update-completed')
        changes = {k: float(np.abs(np.array(v) - initial_host[k]).sum()) for k, v in leaves().items()}
        if report['nativeUpdates'] != 320 or report['optimizerUpdateCalls'] != 320 or not all(v > 0 for v in changes.values()):
            raise AssertionError('Incomplete actual320 updates or unchanged A/B')
        report['parameterAbsoluteChanges'] = changes; verify_base('postFit')
        report['phase'] = 'checkpoint-and-original-archive-parity'; save('reload-proof')
        item = load_feature(report['extraction'][0], out); before = float(feature_loss(bb, adapter, item).item())
        before_parameters = parameter_receipts(); checkpoint = out / 'adapter.safetensors'
        mx.save_safetensors(str(checkpoint), leaves()); load_adapter(checkpoint)
        after = float(feature_loss(bb, adapter, item).item()); reloaded_parameters = parameter_receipts()
        if before != after or before_parameters != reloaded_parameters:
            raise AssertionError('Exact A/B dtype/shape/bytes or feature loss changed on save/reload')
        report['checkpoint'] = {'sha256': digest(checkpoint), 'lossBeforeReload': before, 'lossAfterReload': after,
            'parametersBeforeReload': before_parameters, 'parametersAfterReload': reloaded_parameters, 'exact': True}
        del item
        # x/h/y precede the final down adaptation. Verify actual archived bytes
        # under the learned adapter and a fresh learned-mode ordinary root.
        case = packet['train']['cases'][0]
        common, _, fields = serving_rows(case, [f for f in packet['train']['fields'] if f['caseId'] == case['caseId']])
        state, _ = prepare_root(bb, common); _complete_cache(state['cache'], len(state['tokens']))
        if state_hash(state['cache']) != report['extraction'][0]['sourceStateHash']:
            raise AssertionError('Fresh learned-mode root differs from original archived source cache')
        try:
            gate, recaptured = probe_gradient_features(bb, adapter, state, fields[0])
        except GradientFeatureProbeError as exc:
            dump(out / 'learned-probe.json', exc.report); raise
        dump(out / 'learned-probe.json', gate)
        if gate['status'] != 'exact-gate-passed' or recaptured is None or gate['featureTensors'] != report['extraction'][0]['featureTensors']:
            raise AssertionError('Learned exact native replay or original archived x/h/y bytes differ')
        del recaptured
        item = load_feature(report['extraction'][0], out)
        loss, gradients = nn.value_and_grad(adapter, lambda: feature_loss(bb, adapter, item))()
        mx.eval(loss, gradients); mx.synchronize()
        def tensor_receipt(value):
            host = np.array(value.astype(mx.float32)); dtype = str(value.dtype)
            return {'dtype': dtype, 'shape': list(value.shape),
                'sha256': hashlib.sha256(dtype.encode() + str(tuple(value.shape)).encode() + host.tobytes()).hexdigest()}
        archived = {'loss': float(loss.item()), 'lossTensor': tensor_receipt(loss),
            'gradientTensors': {k: tensor_receipt(v) for k, v in tree_flatten(gradients)}}
        if archived != gate['arms']['direct']:
            raise AssertionError('Original archived learned loss/gradient bytes differ from fresh native direct replay')
        report['controls']['learnedOriginalArchivedNativeParityExact'] = True
        report['controls']['learnedOriginalArchiveParityScope'] = {'trainFeatureIndices': [0],
            'notAll320LearnedReplays': True, 'all320PassedBeforeFit': True}
        del item, gate, state, loss, gradients
        verify_base('postReplay'); assert_source_fresh(frozen, data)
        report['languageParameterTensorsHashed'] = 927
        report['controls']['frozenLanguageHashesUnchanged'] = True
        adapter.freeze(); model.eval(); report['phase'] = 'matched-width6-controls'; save('controls-started')
        first_eval_bound = packet['eval']['cases'][0]['bound']
        for mode in MODES:
            load_adapter(PRIOR if mode == 'failed-daypack' else checkpoint)
            with disabled(model, adapter) if mode == 'frozen' else nullcontext(), deadline(CONFIG['controlTimeoutSeconds'], 'width6 controls'):
                report['controls'][mode] = native_controls(bb, first_eval_bound, out / ('controls-' + mode))
            save('mode-controls-completed'); gc.collect(); mx.clear_cache()
        report['phase'] = '24-evaluation-arms'; save('evaluation-started')
        for case_index, case in enumerate(packet['eval']['cases']):
            mode_order = MODES if case_index % 2 == 0 else tuple(reversed(MODES))
            for mode in mode_order:
                load_adapter(PRIOR if mode == 'failed-daypack' else checkpoint)
                arm = next(a for a in arms if (a['caseId'], a['mode']) == (case['caseId'], mode))
                arm.update(status='running', stage='public-metadata'); save()
                # No report save or expensive work lies between clock start and
                # the request: stage callbacks are assignments only.
                begin = time.perf_counter()
                try:
                    with disabled(model, adapter) if mode == 'frozen' else nullcontext(), deadline(CONFIG['requestTimeoutSeconds'], 'copy request'):
                        result = run_case(bb, case['bound'], stage_callback=lambda stage: arm.update(stage=stage))
                    arm.update(result, stage='completed', boundSha256=hashed(case['bound']))
                except Exception as exc:
                    arm.update(getattr(exc, 'partial_result', {}), status='failed',
                        error={'type': type(exc).__name__, 'message': str(exc)},
                        rawRows=getattr(exc, 'partial_rows', getattr(exc, 'partial_result', {}).get('rawRows', [])),
                        failedAtMs=(time.perf_counter() - begin) * 1000)
                    save('arm-failed'); raise
                finally:
                    gc.collect(); mx.clear_cache()
                save('arm-completed')
        if len(arms) != 24 or any(a['status'] != 'completed-attempt' for a in arms):
            raise AssertionError('Fixed24 evaluation arms incomplete')
        if digest(checkpoint) != report['checkpoint']['sha256']:
            raise AssertionError('Learned checkpoint changed during evaluation')
        verify_base('postEvaluation'); assert_source_fresh(frozen, data)
        report['controls'].update(sourceFilesUnchanged=True, datasetManifestUnchanged=True, originalPriorUnchanged=True)
        report.update(status='completed', phase='completed', memory=bb.runtime_memory()); save('completed')
    except BaseException as exc:
        # Includes operator cancellation: retain active gate/step/arm receipts.
        for arm in arms:
            if arm['status'] == 'running':
                arm.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        report.update(status='blocked', blocker={'type': type(exc).__name__, 'message': str(exc)})
        if 'bb' in locals():
            report['memory'] = bb.runtime_memory()
        save('blocked'); raise
    finally:
        if 'native_context' in locals():
            mx.synchronize(generation_stream); native_context.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--preflight-report', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--native-owner', choices=['root'])
    parser.add_argument('--confirm-serialized', action='store_true')
    parser.add_argument('--reviewed-preflight-sha256')
    args = parser.parse_args()
    if args.preflight:
        if args.out is not None or args.preflight_report.exists():
            raise ValueError('Preflight writes one new report; no native output directory')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        receipt = preflight(tokenizer, args.data)
        import sys
        if any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules):
            raise AssertionError('Tokenizer-only preflight imported MLX')
        dump(args.preflight_report, receipt)
        print(json.dumps({s: {k: v for k, v in g.items() if k not in ('rows', 'cases')}
            for s, g in receipt['splits'].items()}), flush=True)
        return
    if args.out is None or args.native_owner != 'root' or not args.confirm_serialized or digest(args.preflight_report) != args.reviewed_preflight_sha256:
        raise ValueError('Native phase requires new output, root owner, serialization confirmation and exact reviewed preflight SHA')
    from .catalog_fusion.generalization_v2 import native_owner_lock
    with native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'), args.native_owner):
        run(args.out, args.preflight_report, args.data)


if __name__ == '__main__':
    main()

