"""Prospective 240-row C-supervised shared-prefix fit and complete-view benchmark.

Imports and tokenizer preflight execute no native model. Root alone launches
the native phase. Earlier scripts, validators, checkpoints and reports stay
read-only. Decisions are locked: all reported quality is copy quality only.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

from .semantic_copy_feature_fit import (MODEL, ROOT, digest, dump, feature_bytes,
    hashed, load_feature, model_input_receipt, records, verify_model_files)

DATA = ROOT / 'short_copy_v1'
WARM_START = Path('data/ranker/pretrained/semantic-copy-feature-fit-v1/adapter.safetensors')
WARM_SHA = '0b0aeda377a9c9ad7eb37b529c139dac84bc5acd4bb559176d40245865ffe1b2'
WIDTHS = (3, 6)
ALL_MODES = ('frozen', 'prior-STB', 'new-C')
DEFAULT_EVAL_MODES = ('prior-STB', 'new-C')
FIRST_FIELDS = ('S1', 'T1', 'B1', 'C1', 'T2', 'B2', 'C2', 'T3', 'B3', 'C3')
SECOND_FIELDS = ('S2', 'T4', 'B4', 'C4', 'T5', 'B5', 'C5', 'T6', 'B6', 'C6')
CONFIG = {'seed': 19077, 'updates': 240, 'epochs': 1, 'rank': 4, 'scale': 20.,
    'learningRate': .0001, 'memoryGiB': 7, 'maximumContext': 4096,
    'maximumSuffix': 256, 'maximumRawPrefix': 512, 'maximumCommonAppend': 512,
    'prefillStep': 256, 'singleHostFeatureBudgetBytes': 32 * 2**20,
    'totalFeatureArtifactBudgetBytes': 5 * 2**30, 'residentHostFeatureRows': 1,
    'trainCases': 12, 'fieldsPerCase': 20, 'sampledCapPerCase': 416,
    'selection': 'final fixed step; no dev/test checkpoint selection',
    'order': 'one seeded permutation of all 240 individual rows',
    'initialization': 'immutable completed prior-STB A/B checkpoint; no zero reset',
    'gates': 'all 240 exact nonzero-B loss and A/B-gradient comparisons; no tolerance'}
SOURCES = ('short_copy_fit.py', 'short_copy_data.py', 'fused_prefix.py',
    'semantic_copy_gradient_features.py', 'semantic_copy_tokens.py',
    'semantic_copy_feature_fit.py', 'semantic_copy_training.py', 'parallel_decode.py',
    'above_fold_copy/layout.py', 'above_fold.py', 'scheduler_identity_experiment.py',
    'fusion.py', 'page_plan.py', 'boundary.py', 'branch_local_copy.py', 'evidence_plan.py',
    'backbone.py', 'native_batch_diagnosis.py', 'ABOVE_FOLD_WORKLOAD.json',
    'semantic_copy_pairs/generator.py',
    'catalog_fusion/generalization_v2.py', 'test_short_copy_fit.py')


def shuffled_order(count=240, seed=CONFIG['seed']):
    if isinstance(count, bool) or count != 240:
        raise ValueError('Exactly 240 individual updates are required')
    indices = list(range(count)); random.Random(seed).shuffle(indices)
    return indices


def evaluation_modes(value):
    modes = tuple(value.split(',')) if isinstance(value, str) else tuple(value)
    if len(modes) != len(set(modes)) or not modes or set(modes) - set(ALL_MODES):
        raise ValueError('Distinct explicit frozen/prior-STB/new-C evaluation modes required')
    if 'new-C' not in modes:
        raise ValueError('The newly trained mode must be evaluated')
    return modes


def warm_start_receipt(path=WARM_START):
    if digest(path) != WARM_SHA:
        raise ValueError('Immutable prior-STB warm-start checkpoint SHA differs')
    from .above_fold_copy.benchmark import completed_adapter
    receipt = completed_adapter(path.parent / 'results.json')
    if receipt['checkpointSha256'] != WARM_SHA:
        raise ValueError('Completed prior-STB result/checkpoint mismatch')
    return receipt


def serving_rows(case, fields):
    from .short_copy_data import build_layout
    common, rows, evidence = build_layout(case['pendingPagePlan'], case['request'], case['catalog'])
    if len(rows) != 20 or sum(r['maxTokens'] for r in rows) != 416 or len(fields) != 20:
        raise ValueError('Exactly twenty unchanged S/T/B/C fields and 416 sampled tokens required')
    by_id = {f['rowId']: f for f in fields}
    if len(by_id) != 20 or set(by_id) != {r['rowId'] for r in rows}:
        raise ValueError('Frozen training field membership differs')
    if tuple(r['fields'][0] for r in rows) != FIRST_FIELDS + SECOND_FIELDS:
        raise ValueError('All first-section fields must precede all second-section fields')
    ordered = []
    for row in rows:
        field = by_id[row['rowId']]
        if len(row['messages']) != 1 or row['messages'][0]['role'] != 'user':
            raise ValueError('Exactly one independent field-user suffix required')
        if (field['messages'], field['maxTokens'], field['kind'], field['field']) != (
                row['messages'], row['maxTokens'], row['kind'], row['fields'][0]):
            raise ValueError('Frozen teacher input differs from serving renderer')
        # Scheduler metadata only; teacher text never enters a serving row.
        row['key'] = row['rowId']
        ordered.append(field)
    return common, rows, ordered, evidence


def dataset_packet(data=DATA):
    from .short_copy_data import load_dataset
    manifest, cases_by_split, fields_by_split = load_dataset(data)
    packet = {s: {'cases': cases_by_split[s], 'fields': fields_by_split[s]} for s in ('train', 'dev', 'test')}
    seen = set()
    for split, count in (('train', 12), ('dev', 3), ('test', 3)):
        cases, fields = packet[split]['cases'], packet[split]['fields']
        ids = {c['caseId'] for c in cases}
        if len(cases) != count or len(ids) != count or len(fields) != count * 20 or seen & ids:
            raise ValueError('Fixed split denominator or disjoint case IDs changed')
        if {f['caseId'] for f in fields} != ids:
            raise ValueError('Field/case membership differs')
        seen |= ids
        for case in cases:
            serving_rows(case, [f for f in fields if f['caseId'] == case['caseId']])
    return manifest, packet


def preflight(tokenizer, data=DATA, eval_modes=DEFAULT_EVAL_MODES):
    from .fused_prefix import RawPrefixTokenizer, focused_query_text
    from .semantic_copy_gradient_features import build_native_contract
    modes = evaluation_modes(eval_modes); tokenizer = RawPrefixTokenizer(tokenizer)
    manifest, packet = dataset_packet(data)
    result = {'schemaVersion': 1, 'status': 'tokenizer-preflight-passed', 'nativeExecution': False,
        'modelCalls': 0, 'trainingUpdates': 0, 'config': CONFIG, 'evalModes': list(modes),
        'widths': list(WIDTHS), 'fixedCases': 6, 'fixedArms': 6 * len(modes) * 2,
        'datasetManifestSha256': digest(data / 'manifest.json'), 'payloadSha256': manifest['fileSha256'],
        'datasetTokenPreflightSha256': digest(data / 'TOKEN_PREFLIGHT.json'),
        'sources': {n: digest(ROOT / n) for n in SOURCES}, 'modelFiles': model_input_receipt(),
        'warmStart': warm_start_receipt(), 'splits': {},
        'limitations': ['Locked recorded decisions; no ranking quality or integrated ranking latency claim.',
            'Six dev/test cases were previously inspected during data preparation; no sealed holdout claim.',
            'Three-card result requires all ten first-section fields, actual validation and serialization.',
            'Full completion requires both sections, all twenty fields and two complete three-card payloads.',
            'No templates, fallback copy, truncation or changed strict factuality/identity rules.',
            'Native 240-row gates and matched-width controls must pass before quality timings.']}
    for split, group in packet.items():
        cases = []; rows = []
        for case in group['cases']:
            common, jobs, fields, _ = serving_rows(case, [f for f in group['fields'] if f['caseId'] == case['caseId']])
            raw = {'role': 'raw_prefix', 'content': focused_query_text(case['request'], case['catalog'])}
            raw_tokens = tokenizer.apply_chat_template([raw], tokenize=True, return_dict=False,
                add_generation_prompt=False, enable_thinking=False)
            complete = tokenizer.apply_chat_template([raw, common], tokenize=True, return_dict=False,
                add_generation_prompt=False, enable_thinking=False)
            if not 1 <= len(raw_tokens) <= CONFIG['maximumRawPrefix'] or complete[:len(raw_tokens)] != raw_tokens:
                raise ValueError('Focused raw prefix exceeds cap or common message changed it')
            if not 1 <= len(complete) - len(raw_tokens) <= CONFIG['maximumCommonAppend']:
                raise ValueError('Common evidence append exceeds complete explicit cap')
            state = {'messages': [raw, common], 'tokens': complete, 'consumed': False}
            case_rows = []
            for field, job in zip(fields, jobs, strict=True):
                contract = build_native_contract(tokenizer, state, field)
                prompt = tokenizer.apply_chat_template(state['messages'] + job['messages'], tokenize=True,
                    return_dict=False, add_generation_prompt=True, enable_thinking=False)
                if prompt != contract['promptTokenIds'] or len(prompt) + job['maxTokens'] > CONFIG['maximumContext']:
                    raise ValueError('Serving/training prompt differs or generation cap exceeds context')
                size = feature_bytes(len(contract['suffixTokenIds']))
                if size > CONFIG['singleHostFeatureBudgetBytes']:
                    raise MemoryError('One full native feature row exceeds host cap')
                entry = {'caseId': case['caseId'], 'rowId': field['rowId'], 'field': field['field'],
                    'kind': field['kind'], 'maxTokens': field['maxTokens'], 'suffixTokens': len(contract['suffixTokenIds']),
                    'promptTokens': len(prompt), 'targetTokens': len(contract['targetTokenIds']),
                    'fullTeacherTokens': contract['fullTokens'], 'projectedHostFeatureBytes': size,
                    'promptSha256': contract['promptSha256'], 'modelInputSha256': contract['modelInputSha256'],
                    'sourceTokenSha256': contract['sourceTokenSha256'],
                    'predictionStartInBranch': contract['predictionStartInBranch']}
                rows.append(entry); case_rows.append(entry)
            cases.append({'caseId': case['caseId'], 'lockedPlanSha256': hashed(case['pendingPagePlan']),
                'rawTokens': len(raw_tokens), 'commonAppendTokens': len(complete) - len(raw_tokens),
                'sharedTokens': len(complete), 'rawTokensSha256': hashed(raw_tokens),
                'completeSharedSha256': hashed(complete), 'maximumSuffix': max(r['suffixTokens'] for r in case_rows)})
        result['splits'][split] = {'caseCount': len(cases), 'fieldCount': len(rows), 'cases': cases, 'rows': rows,
            'maximumRawTokens': max(c['rawTokens'] for c in cases),
            'maximumSharedTokens': max(c['sharedTokens'] for c in cases),
            'maximumSuffixTokens': max(r['suffixTokens'] for r in rows),
            'projectedHostFeatureBytes': sum(r['projectedHostFeatureBytes'] for r in rows)}
    projected = result['splits']['train']['projectedHostFeatureBytes'] + 2**20
    if projected > CONFIG['totalFeatureArtifactBudgetBytes']:
        raise MemoryError('Projected uncompressed complete feature artifacts exceed declared 5-GiB cap')
    result['projectedUncompressedTrainingArtifactsWithReserveBytes'] = projected
    result['archivePolicy'] = {'method': 'numpy.savez_compressed; lossless float32 host representation',
        'compressionRequiredToFitArtifactBudget': projected > CONFIG['totalFeatureArtifactBudgetBytes'],
        'actualArchivePlusMetadataByteGuard': CONFIG['totalFeatureArtifactBudgetBytes'],
        'ifCompressionInsufficient': 'stop during extraction; preserve exact evidence; zero optimizer updates'}
    result['globalPermutationSha256'] = hashed(shuffled_order())
    result['gateWork'] = {'rows': 240, 'nativeSuffixAutodiffPasses': 480,
        'featureAutodiffPasses': 240, 'exactComparisons': 720}
    return result


def section_projection(case, rows, section_index):
    """Trusted local-index projection, preserving text/token/native receipts verbatim."""
    if section_index not in (0, 1):
        raise ValueError('Exactly two three-card sections are supported')
    plan = copy.deepcopy(case['pendingPagePlan']); section = plan['sections'][section_index]
    plan['sections'] = [section]; plan['sectionOrder'] = [section['sectionId']]
    plan['requestedCardCount'] = 3; plan['shortfall'] = 0
    wanted = FIRST_FIELDS if section_index == 0 else SECOND_FIELDS
    mapping = dict(zip(wanted, FIRST_FIELDS, strict=True)); selected = []
    for row in rows:
        fields = row.get('fields', [])
        if len(fields) != 1 or fields[0] not in wanted:
            continue
        value = copy.deepcopy(row); original = fields[0]; local = mapping[original]
        value['sourceGlobalField'] = original; value['fields'] = [local]
        value['rowId'] = value['key'] = value['kind'] + '-' + local[1:]
        selected.append(value)
    if len(selected) != 10 or {r['fields'][0] for r in selected} != set(FIRST_FIELDS):
        raise ValueError('All ten unique section rows required; no missing-field repair')
    return plan, selected


def assemble_section(case, rows, section_index):
    from .above_fold_copy.layout import assemble_payload
    plan, local = section_projection(case, rows, section_index)
    return assemble_payload(local, plan, case['request'], case['catalog'], require_native_receipts=True)


def encoded_payload(result):
    encoded = json.dumps(result['payload'], sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()
    sha = hashlib.sha256(encoded).hexdigest()
    if result['payloadSha256'] != sha:
        raise ValueError('Actual serialized payload hash differs from assembly receipt')
    return encoded


def same_prefix_cache_proof(bb, adapter, case, common, field, expected_source_hash):
    """Exact cache proof at the ACTUAL raw/common/field prefill boundaries."""
    from mlx_lm.generate import generation_stream
    from .fused_prefix import prepare_fused_prefix, append_common_message
    from .native_batch_diagnosis import state_hash
    from .semantic_copy_training import disabled
    from .semantic_copy_gradient_features import build_native_contract
    mx = bb.mx; model = bb.model; sources = []; states = []; logits = []; token_receipts = []
    for enabled in (False, True):
        with disabled(model, adapter) if not enabled else nullcontext():
            mode = 'proof-adapter-enabled' if enabled else 'frozen-base'
            features, _, raw = prepare_fused_prefix(bb, case['request'], case['catalog'], feature_mode=mode); del features
            state, _ = append_common_message(bb, raw, common, current_mode=mode,
                max_append_tokens=CONFIG['maximumCommonAppend'])
            source_hash = state_hash(state['cache']); sources.append(source_hash)
            if source_hash != expected_source_hash:
                raise AssertionError('Actual raw/common off/on prefix cache differs from exact extraction source')
            contract = build_native_contract(bb.tokenizer, state, field)
            suffix = contract['promptTokenIds'][len(state['tokens']):]
            if not 1 <= len(suffix) <= CONFIG['maximumSuffix']:
                raise ValueError('Cache-proof field prompt suffix exceeds declared shape')
            hidden = model.model(mx.array([suffix]), cache=state['cache'])
            logit = model.language_model.lm_head(hidden[:, -1:])
            mx.eval(logit, [c.state for c in state['cache']]); mx.synchronize(generation_stream); bb.check_memory()
            if not bool(mx.all(mx.isfinite(logit)).item()):
                raise FloatingPointError('Nonfinite cache proof logits')
        states.append(state); logits.append(logit)
        token_receipts.append({'promptSha256': contract['promptSha256'],
            'forcedContinuation': contract['targetTokenIds'][:3], 'suffixTokens': len(suffix)})
    hashes = [state_hash(s['cache']) for s in states]
    effect = float(mx.max(mx.abs(logits[0].astype(mx.float32) - logits[1].astype(mx.float32))).item())
    if hashes[0] != hashes[1] or token_receipts[0] != token_receipts[1] or not effect > 0:
        raise AssertionError('Final-down adapter changed complete native cache or had no logit effect')
    continuations = []
    for state in states:
        logs = []; predictions = []
        for token in token_receipts[0]['forcedContinuation']:
            value = model(mx.array([[token]]), cache=state['cache'])[:, -1]
            mx.eval(value, [c.state for c in state['cache']]); mx.synchronize(generation_stream)
            if not bool(mx.all(mx.isfinite(value)).item()):
                raise FloatingPointError('Nonfinite continuation proof logits')
            logs.append(value); predictions.append(int(mx.argmax(value, axis=-1).item()))
        continuations.append((predictions, logs, state_hash(state['cache'])))
    exact = (continuations[0][0] == continuations[1][0] and continuations[0][2] == continuations[1][2]
        and all(bool(mx.array_equal(a, b).item()) for a, b in zip(continuations[0][1], continuations[1][1], strict=True)))
    if not exact:
        raise AssertionError('Actual-boundary base/adapter cache continuation differs')
    return {'sourceStateHashes': sources, 'promptStateHashes': hashes,
        'sourceMatchesActualExtractionBoundary': True, 'maxAbsLogitEffect': effect,
        'matchedForcedContinuationLogitsExact': True, 'tokenReceipt': token_receipts[0],
        'prefillBoundary': 'raw whole B1; complete common chunks256; whole field suffix B1',
        'noArbitraryTrimOrSplice': True}


def arm_roster(packet, modes=DEFAULT_EVAL_MODES):
    modes = evaluation_modes(modes)
    return [{'split': split, 'caseId': c['caseId'], 'mode': mode, 'width': width, 'status': 'not-attempted'}
        for split in ('dev', 'test') for c in packet[split]['cases'] for mode in modes for width in WIDTHS]


def summary(arms, modes=DEFAULT_EVAL_MODES):
    result = {}
    for split in ('dev', 'test'):
        for mode in evaluation_modes(modes):
            for width in WIDTHS:
                selected = [a for a in arms if (a['split'], a['mode'], a['width']) == (split, mode, width)]
                done = [a for a in selected if a['status'] == 'completed-attempt']
                valid_first = [a for a in done if a['firstSection']['validation']['valid']]
                valid_full = [a for a in done if a['fullValidation']['valid']]
                result[f'{split}/{mode}/width-{width}'] = {'fixedCases': 3,
                    'completed': len(done), 'validThreeCardResults': len(valid_first),
                    'validThreeCardRateFixedDenominator': len(valid_first) / 3,
                    'validSixCardResults': len(valid_full), 'validSixCardRateFixedDenominator': len(valid_full) / 3,
                    'failed': sum(a['status'] == 'failed' for a in selected),
                    'unattempted': sum(a['status'] == 'not-attempted' for a in selected),
                    'inProgress': sum(a['status'] == 'running' for a in selected),
                    'attemptFirstSectionReadyMs': [a['firstSectionReceipt']['validatedAndSerializedAtMs'] for a in done],
                    'validFirstSectionReadyMs': [a['firstSectionReceipt']['validatedAndSerializedAtMs'] for a in valid_first],
                    'attemptFullCompletionMs': [a['fullCompletionReceipt']['validatedAndSerializedAtMs'] for a in done],
                    'validFullCompletionMs': [a['fullCompletionReceipt']['validatedAndSerializedAtMs'] for a in valid_full],
                    'lockedDecisions': True, 'rankingQualityClaim': False, 'browserPaintClaim': False}
    return result


def decode_complete_rows(bb, state, rows, width, on_first_complete):
    """One continuous generator, first-section priority, real ready callback.

    The callback is invoked once all ten first-section rows have ended. Its CPU
    validation/serialization cost remains inside full wall time. Remaining rows
    keep their independent full prefix forks; no generated sibling is an input.
    """
    from mlx_lm.generate import BatchGenerator, StopSequences, generation_stream
    from mlx_lm.sample_utils import make_sampler
    from .parallel_decode import clone_prefix_state, exact_row_suffix
    if width not in WIDTHS or tuple(r['fields'][0] for r in rows) != FIRST_FIELDS + SECOND_FIELDS:
        raise ValueError('Fixed widths and complete twenty-field first-section-priority roster required')
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
        fork_ms += (time.perf_counter() - begin) * 1000; bb.check_memory(); begin = time.perf_counter()
        prompts = []; caches = []
        for (_, row), branch in zip(group, branches, strict=True):
            suffix, cache = exact_row_suffix(bb.tokenizer, row, branch, bb.model)
            if not 1 <= len(suffix) <= CONFIG['maximumSuffix']:
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
            bb.check_memory()
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


def run(out, preflight_path, data=DATA):
    """Root-owned native phase; CPU tests never call this entry point."""
    function_started = time.perf_counter()
    import numpy as np
    import psutil
    from transformers import AutoTokenizer
    if out.exists():
        raise FileExistsError('Preserve previous attempt; choose a new output directory')
    frozen = json.loads(preflight_path.read_text())
    modes = evaluation_modes(frozen['evalModes'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    expected = preflight(tokenizer, data, modes)
    if expected != frozen:
        raise ValueError('Current source/data/tokenizer differs from frozen preflight')
    if psutil.virtual_memory().available < CONFIG['memoryGiB'] * 2**30:
        raise MemoryError('Less than declared 7 GiB available before native load')
    manifest, packet = dataset_packet(data); out.mkdir(parents=True)
    (out / 'features').mkdir(); (out / 'gates').mkdir(); start = function_started
    arms = arm_roster(packet, modes)
    report = {'status': 'started', 'phase': 'verify-model', 'config': CONFIG,
        'executionCommit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'preflightSha256': digest(preflight_path), 'preflight': frozen,
        'nativeUpdates': 0, 'optimizerUpdateCalls': 0, 'extraction': [], 'steps': [],
        'arms': arms, 'fixedArms': len(arms), 'controls': {}, 'featureArtifactBytes': 0,
        'wallScope': 'run() entry including tokenizer/preflight verification and shard hashes; excludes Python/module startup',
        'limitations': frozen['limitations']}
    def save():
        report['wallMs'] = (time.perf_counter() - start) * 1000
        report['summary'] = summary(arms, modes); dump(out / 'results.json', report)
    save()
    try:
        report['modelVerification'] = verify_model_files(frozen['modelFiles']); save()
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from mlx_lm.tuner.lora import LoRALinear
        from .backbone import FrozenBackbone
        from .evidence_plan import REVISION
        from .fused_prefix import RawPrefixTokenizer, prepare_fused_prefix, append_common_message, matched_controls
        from .native_batch_diagnosis import state_hash
        from .semantic_copy_training import disabled, parameter_hashes
        from .semantic_copy_gradient_features import (FEATURE_KEYS, GradientFeatureProbeError,
            build_native_contract, probe_gradient_features, feature_loss)
        class GuardedBackbone(FrozenBackbone):
            def check_memory(self):
                super().check_memory()
                if max(self.mx.get_peak_memory(), self.mx.get_active_memory()) > self.memory_budget:
                    raise MemoryError('Native active/peak exceeded declared 7 GiB')
        from mlx_lm.generate import generation_stream
        native_context = mx.stream(generation_stream); native_context.__enter__()
        mx.random.seed(CONFIG['seed']); bb = GuardedBackbone(MODEL, REVISION, 4096, CONFIG['memoryGiB'])
        report['residentLoadMs'] = bb.load_ms
        bb.tokenizer = RawPrefixTokenizer(bb.tokenizer); model = bb.model; model.freeze(); model.eval()
        base_hashes = parameter_hashes(model.language_model)
        if len(base_hashes) != 927 or dict(tree_flatten(model.trainable_parameters())):
            raise AssertionError('Exactly all 927 frozen language tensor receipts required')
        adapter = LoRALinear.from_base(model.layers[-1].mlp.down_proj, r=4, scale=20, dropout=0)
        model.layers[-1].mlp.down_proj = adapter; model.eval()
        def load_adapter(path):
            weights = mx.load(str(path))
            if set(weights) != {'lora_a', 'lora_b'} or tuple(weights['lora_a'].shape) != (12288, 4) or tuple(weights['lora_b'].shape) != (4, 4096):
                raise ValueError('Only pinned final-down A/B shapes permitted')
            if not all(bool(mx.all(mx.isfinite(v)).item()) for v in weights.values()):
                raise FloatingPointError('Nonfinite checkpoint parameter')
            adapter.load_weights(list(weights.items()), strict=False); mx.eval(adapter.parameters()); mx.synchronize()
        load_adapter(WARM_START)
        roster = dict(tree_flatten(model.trainable_parameters()))
        if len(roster) != 2 or sum(v.size for v in roster.values()) != 65536 or any(
                '.31.mlp.down_proj.' not in k or k.rsplit('.', 1)[-1] not in ('lora_a', 'lora_b') for k in roster):
            raise AssertionError('Exactly layer31 down-projection A/B, 65536 trainable floats required')
        if not bool(mx.any(adapter.lora_b != 0).item()):
            raise ValueError('Nonzero warm-start B required before all gates')
        report['trainableParameters'] = {k: int(v.size) for k, v in roster.items()}
        initial = {k: np.array(v) for k, v in tree_flatten(adapter.trainable_parameters())}
        initial_params = {k: v for k, v in tree_flatten(adapter.trainable_parameters())}
        planned = {(v['caseId'], v['rowId']): v for v in frozen['splits']['train']['rows']}
        first_proof = None; report['phase'] = '240-exact-gates'; save()
        for case in packet['train']['cases']:
            common, _, fields, _ = serving_rows(case, [f for f in packet['train']['fields'] if f['caseId'] == case['caseId']])
            with disabled(model, adapter):
                features, raw_timing, raw = prepare_fused_prefix(bb, case['request'], case['catalog'], feature_mode='frozen-base')
                del features
                state, common_timing = append_common_message(bb, raw, common, current_mode='frozen-base',
                    max_append_tokens=CONFIG['maximumCommonAppend'])
            raw_hash = state_hash(raw['cache']); source_hash = state_hash(state['cache'])
            for field in fields:
                index = len(report['extraction']); begin = time.perf_counter(); plan = planned[(case['caseId'], field['rowId'])]
                if first_proof is None:
                    contract = build_native_contract(bb.tokenizer, state, field)
                    first_proof = (case, common, field, source_hash)
                    report['controls']['warmStartCacheProof'] = same_prefix_cache_proof(bb, adapter, *first_proof)
                gate_path = out / 'gates' / f'{index:04d}.json'
                try:
                    gate, item = probe_gradient_features(bb, adapter, state, field)
                except GradientFeatureProbeError as exc:
                    dump(gate_path, exc.report)
                    report['failedGate'] = {'index': index, 'caseId': case['caseId'], 'rowId': field['rowId'], 'receipt': str(gate_path)}
                    raise
                dump(gate_path, gate)
                if gate['status'] != 'exact-gate-passed' or item is None:
                    report['failedGate'] = {'index': index, 'caseId': case['caseId'], 'rowId': field['rowId'], 'receipt': str(gate_path)}
                    raise AssertionError('Strict row gate failed; no optimizer update permitted')
                if any(gate['contract'][k] != plan[k] for k in ('modelInputSha256', 'sourceTokenSha256', 'promptSha256')):
                    raise AssertionError('Native teacher tokens differ from preflight')
                size = sum(item[k].nbytes for k in FEATURE_KEYS)
                if size != plan['projectedHostFeatureBytes'] or size > CONFIG['singleHostFeatureBudgetBytes']:
                    raise MemoryError('Full host feature row violates preflight')
                archive = out / 'features' / f'{index:04d}.npz'; metadata = archive.with_suffix('.json')
                np.savez_compressed(archive, **{k: item[k] for k in FEATURE_KEYS})
                dump(metadata, {k: v for k, v in item.items() if k not in FEATURE_KEYS})
                report['featureArtifactBytes'] += archive.stat().st_size + metadata.stat().st_size
                if report['featureArtifactBytes'] > CONFIG['totalFeatureArtifactBudgetBytes']:
                    raise MemoryError('Total host feature artifact guard exceeded')
                if state_hash(raw['cache']) != raw_hash or state_hash(state['cache']) != source_hash:
                    raise AssertionError('Complete raw or common source state mutated')
                report['extraction'].append({'index': index, 'caseId': case['caseId'], 'rowId': field['rowId'], 'kind': field['kind'],
                    'archive': str(archive.relative_to(out)), 'metadata': str(metadata.relative_to(out)),
                    'archiveSha256': digest(archive), 'metadataSha256': digest(metadata), 'gateSha256': digest(gate_path),
                    'featureTensors': gate['featureTensors'], 'hostFeatureBytes': size, 'sourceStateHash': source_hash,
                    'rawStateHash': raw_hash, 'modelInputSha256': plan['modelInputSha256'], 'gateExact': True,
                    'rawPreparation': raw_timing, 'commonPreparation': common_timing,
                    'ms': (time.perf_counter() - begin) * 1000})
                del item, gate; gc.collect(); mx.clear_cache(); save()
            del state, raw; gc.collect(); mx.clear_cache()
        if len(report['extraction']) != 240:
            raise AssertionError('All 240 exact gates must precede every optimizer update')
        for k, v in initial_params.items():
            setattr(adapter, k, v)
        mx.eval(adapter.parameters()); mx.synchronize()
        if any(not np.array_equal(np.array(v), initial[k]) for k, v in tree_flatten(adapter.trainable_parameters())):
            raise AssertionError('Warm-start adapter changed before actual training')
        report['controls']['all240NonzeroBGatesPassed'] = True
        report['controls']['fitStartedFromExactWarmStart'] = True
        optimizer = optim.Adam(learning_rate=CONFIG['learningRate'])
        grad_fn = nn.value_and_grad(adapter, lambda item: feature_loss(bb, adapter, item))
        order = shuffled_order(); report['permutation'] = order; report['phase'] = '240-updates'; save()
        for step, index in enumerate(order):
            begin = time.perf_counter(); entry = report['extraction'][index]; item = load_feature(entry, out)
            loss, gradients = grad_fn(item); mx.eval(loss, gradients); mx.synchronize()
            flat = dict(tree_flatten(gradients))
            if set(flat) != {'lora_a', 'lora_b'} or not bool(mx.isfinite(loss).item()) or not all(bool(mx.all(mx.isfinite(v)).item()) for v in flat.values()):
                raise FloatingPointError('Wrong/nonfinite loss or A/B gradients')
            row = {'step': step, 'featureIndex': index, 'caseId': entry['caseId'], 'rowId': entry['rowId'], 'kind': entry['kind'],
                'loss': float(loss.item()), 'gradientL2': sum(float(mx.sum(v.astype(mx.float32)**2).item()) for v in flat.values())**.5,
                'status': 'update-attempted', 'updateEvaluated': False}
            report['steps'].append(row); optimizer.update(adapter, gradients); report['optimizerUpdateCalls'] += 1
            mx.eval(adapter.parameters(), optimizer.state); mx.synchronize(); report['nativeUpdates'] += 1; bb.check_memory()
            leaves = [v for _, v in tree_flatten(adapter.trainable_parameters())] + [v for _, v in tree_flatten(optimizer.state) if hasattr(v, 'dtype')]
            if not all(bool(mx.all(mx.isfinite(v)).item()) for v in leaves):
                raise FloatingPointError('Nonfinite adapter or optimizer state')
            row.update(status='completed', updateEvaluated=True, ms=(time.perf_counter() - begin) * 1000)
            del item, gradients, loss, flat, leaves; save()
        changes = {k: float(np.abs(np.array(v) - initial[k]).sum()) for k, v in tree_flatten(adapter.trainable_parameters())}
        if report['nativeUpdates'] != 240 or not all(v > 0 for v in changes.values()):
            raise AssertionError('Incomplete real updates or unchanged A/B')
        report['parameterAbsoluteChanges'] = changes
        report['controls']['learnedCacheProof'] = same_prefix_cache_proof(bb, adapter, *first_proof)
        item = load_feature(report['extraction'][0], out); before = float(feature_loss(bb, adapter, item).item())
        def adapter_receipts():
            result = {}
            for k, value in tree_flatten(adapter.trainable_parameters()):
                # A/B in this pinned checkpoint are FP32. Record actual dtype,
                # shape and exact parameter bytes, not only one scalar loss.
                if value.dtype != mx.float32:
                    raise ValueError('Pinned warm-start/new adapter A/B must remain FP32')
                host = np.array(value)
                result[k] = {'dtype': str(value.dtype), 'shape': list(value.shape),
                    'sha256': hashlib.sha256(host.tobytes()).hexdigest()}
            if set(result) != {'lora_a', 'lora_b'}:
                raise ValueError('Exact reload proof requires only A/B leaves')
            return result
        before_parameters = adapter_receipts()
        checkpoint = out / 'adapter.safetensors'; mx.save_safetensors(str(checkpoint), dict(tree_flatten(adapter.trainable_parameters())))
        load_adapter(checkpoint); after = float(feature_loss(bb, adapter, item).item())
        reloaded_parameters = adapter_receipts()
        if before != after or before_parameters != reloaded_parameters:
            raise AssertionError('Learned feature loss or A/B dtype/shape/bytes changed on checkpoint reload')
        report['checkpoint'] = {'sha256': digest(checkpoint), 'lossBeforeReload': before, 'lossAfterReload': after,
            'parametersBeforeReload': before_parameters, 'parametersAfterReload': reloaded_parameters, 'exact': True}
        del item
        # A fresh native replay checks the ORIGINAL archived feature item under
        # learned parameters, not merely another reconstructed feature loss.
        first_case = packet['train']['cases'][0]
        common, _, fields, _ = serving_rows(first_case, [f for f in packet['train']['fields'] if f['caseId'] == first_case['caseId']])
        with disabled(model, adapter):
            features, _, raw = prepare_fused_prefix(bb, first_case['request'], first_case['catalog'], feature_mode='frozen-base'); del features
            state, _ = append_common_message(bb, raw, common, current_mode='frozen-base', max_append_tokens=CONFIG['maximumCommonAppend'])
        try:
            gate, recaptured = probe_gradient_features(bb, adapter, state, fields[0])
        except GradientFeatureProbeError as exc:
            dump(out / 'learned-probe.json', exc.report)
            report['failedGate'] = {'stage': 'learned-probe', 'receipt': str(out / 'learned-probe.json')}
            raise
        dump(out / 'learned-probe.json', gate)
        if gate['status'] != 'exact-gate-passed' or recaptured is None or gate['featureTensors'] != report['extraction'][0]['featureTensors']:
            raise AssertionError('Learned native/feature replay or original feature bytes differ')
        # Release the newly recaptured full host row before opening the original
        # archive. At most one [1,L,D] x/h/y item is resident outside the helper.
        del recaptured
        item = load_feature(report['extraction'][0], out)
        archived_loss, archived_grads = nn.value_and_grad(adapter, lambda: feature_loss(bb, adapter, item))()
        mx.eval(archived_loss, archived_grads); mx.synchronize()
        def host_receipt(value):
            host = np.array(value.astype(mx.float32)); dtype = str(value.dtype)
            return {'dtype': dtype, 'shape': list(value.shape),
                'sha256': hashlib.sha256(dtype.encode() + str(tuple(value.shape)).encode() + host.tobytes()).hexdigest()}
        archived_direct = {'loss': float(archived_loss.item()), 'lossTensor': host_receipt(archived_loss),
            'gradientTensors': {k: host_receipt(v) for k, v in tree_flatten(archived_grads)}}
        if archived_direct != gate['arms']['direct']:
            raise AssertionError('Original archived learned loss/A/B gradient bytes differ from fresh native replay')
        report['controls']['learnedOriginalArchivedNativeParityExact'] = True
        del item, gate, state, raw, archived_loss, archived_grads
        after_hashes = {k.replace('.31.mlp.down_proj.linear.', '.31.mlp.down_proj.'): v
            for k, v in parameter_hashes(model.language_model).items()}
        if base_hashes != after_hashes or digest(WARM_START) != WARM_SHA:
            raise AssertionError('Frozen base or original warm-start checkpoint changed')
        report['controls']['frozenLanguageHashesUnchanged'] = True
        report['languageParameterTensorsHashed'] = len(base_hashes)
        adapter.freeze(); model.eval(); report['phase'] = 'matched-controls'; save()
        first_case = packet['dev']['cases'][0]
        common, jobs, _, _ = serving_rows(first_case, [f for f in packet['dev']['fields'] if f['caseId'] == first_case['caseId']])
        for mode in ALL_MODES:
            load_adapter(WARM_START if mode == 'prior-STB' else checkpoint)
            dest = out / ('controls-' + mode)
            with disabled(model, adapter) if mode == 'frozen' else nullcontext():
                report['controls'][mode] = matched_controls(bb, first_case['request'], first_case['catalog'], common, jobs,
                    {'featureMode': 'frozen-base' if mode == 'frozen' else WARM_SHA if mode == 'prior-STB' else digest(checkpoint),
                     'currentMode': 'frozen-base' if mode == 'frozen' else WARM_SHA if mode == 'prior-STB' else digest(checkpoint)}, dest, widths=WIDTHS)
            save(); gc.collect(); mx.clear_cache()
        report['phase'] = 'evaluation'; save(); expected_states = {}
        for case_index, (split, case) in enumerate((s, c) for s in ('dev', 'test') for c in packet[s]['cases']):
            mode_order = modes if case_index % 2 == 0 else tuple(reversed(modes))
            width_order = WIDTHS if case_index % 2 == 0 else tuple(reversed(WIDTHS))
            for mode in mode_order:
                load_adapter(WARM_START if mode == 'prior-STB' else checkpoint)
                for width in width_order:
                    arm = next(a for a in arms if (a['caseId'], a['mode'], a['width']) == (case['caseId'], mode, width))
                    arm['status'] = 'running'; arm['stage'] = 'public-metadata'; save(); begin = time.perf_counter()
                    try:
                        common, jobs, _, evidence = serving_rows(case, [f for f in packet[split]['fields'] if f['caseId'] == case['caseId']])
                        metadata_ready = (time.perf_counter() - begin) * 1000
                        with disabled(model, adapter) if mode == 'frozen' else nullcontext():
                            mode_sha = 'frozen-base' if mode == 'frozen' else WARM_SHA if mode == 'prior-STB' else digest(checkpoint)
                            arm['stage'] = 'raw-and-common-prefill'
                            features, raw_timing, raw = prepare_fused_prefix(bb, case['request'], case['catalog'], feature_mode=mode_sha)
                            del features
                            state, common_timing = append_common_message(bb, raw, common, current_mode=mode_sha,
                                max_append_tokens=CONFIG['maximumCommonAppend'])
                            original = state_hash(state['cache']); raw_original = state_hash(raw['cache'])
                            cache_pair = (raw_original, original, hashed(case['pendingPagePlan']))
                            if case['caseId'] in expected_states and cache_pair != expected_states[case['caseId']]:
                                raise AssertionError('Paired prior/new complete raw/common states or locked decisions differ')
                            expected_states[case['caseId']] = cache_pair
                            admitted = (time.perf_counter() - begin) * 1000; arm['stage'] = 'independent-field-decode'
                            def first_ready(completed):
                                result = assemble_section(case, completed, 0); encoded = encoded_payload(result)
                                ready = (time.perf_counter() - begin) * 1000
                                arm['firstSection'] = result
                                arm['firstSectionReceipt'] = {'payloadSha256': result['payloadSha256'], 'payloadBytes': len(encoded),
                                    'metadataReadyAtMs': metadata_ready, 'copyAdmittedAtMs': admitted,
                                    'validatedAndSerializedAtMs': ready, 'includesRawAndCommonPrefill': True,
                                    'browserPaintMeasured': False, 'requestStartMeaning': 'model-resident copy request; fixture decisions locked'}
                            rows, timing = decode_complete_rows(bb, state, jobs, width, first_ready)
                            arm['rawRows'] = rows
                            bb.check_memory(); arm['stage'] = 'full-validation-and-serialization'
                            second = assemble_section(case, rows, 1); encoded_payload(second)
                            full = {'schemaVersion': 'two-section-six-card-copy.v1',
                                'sections': [arm['firstSection']['payload'], second['payload']]}
                            full_bytes = json.dumps(full, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
                            valid = arm['firstSection']['validation']['valid'] and second['validation']['valid']
                            full_ready = (time.perf_counter() - begin) * 1000
                        if state_hash(state['cache']) != original or state_hash(raw['cache']) != raw_original:
                            raise AssertionError('Evaluation consumed/mutated complete source cache')
                        arm.update(status='completed-attempt', stage='completed', rawRows=rows, secondSection=second,
                            fullPayload=full, fullValidation={'valid': valid, 'requiredFieldCount': 20,
                                'sections': [arm['firstSection']['validation'], second['validation']]},
                            fullCompletionReceipt={'payloadSha256': hashlib.sha256(full_bytes).hexdigest(), 'payloadBytes': len(full_bytes),
                                'validatedAndSerializedAtMs': full_ready, 'includesFirstSectionCallback': True},
                            lockedPlanSha256=hashed(case['pendingPagePlan']), lockedDecisionRankingMs=None,
                            rawPreparation=raw_timing, commonPreparation=common_timing, copyTiming=timing,
                            publicEvidence=evidence, sharedStateHash=original, rawStateHash=raw_original,
                            actualOutputTokens=sum(len(r['tokenIds']) for r in rows), actualArSteps=sum(r['arSteps'] for r in rows),
                            memory=bb.runtime_memory())
                    except Exception as exc:
                        arm.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)},
                            rawRows=getattr(exc, 'partial_rows', arm.get('rawRows', [])),
                            failedAtMs=(time.perf_counter() - begin) * 1000)
                        save(); raise
                    finally:
                        if 'state' in locals():
                            del state
                        if 'raw' in locals():
                            del raw
                        gc.collect(); mx.clear_cache()
                    save()
        if digest(WARM_START) != WARM_SHA or digest(checkpoint) != report['checkpoint']['sha256']:
            raise AssertionError('Original or learned checkpoint changed during evaluation')
        final_hashes = {k.replace('.31.mlp.down_proj.linear.', '.31.mlp.down_proj.'): v
            for k, v in parameter_hashes(model.language_model).items()}
        if final_hashes != base_hashes:
            raise AssertionError('Frozen language bytes changed during benchmark')
        report['controls']['sourceFilesUnchanged'] = all(digest(ROOT / n) == sha for n, sha in frozen['sources'].items())
        report['controls']['datasetManifestUnchanged'] = digest(data / 'manifest.json') == frozen['datasetManifestSha256']
        if not report['controls']['sourceFilesUnchanged'] or not report['controls']['datasetManifestUnchanged']:
            raise AssertionError('Frozen producer sources or data manifest changed during native run')
        report.update(status='completed', phase='completed', memory=bb.runtime_memory()); save()
    except Exception as exc:
        report.update(status='blocked', blocker={'type': type(exc).__name__, 'message': str(exc)})
        if 'bb' in locals():
            report['memory'] = bb.runtime_memory()
        save(); raise
    finally:
        if 'native_context' in locals():
            mx.synchronize(generation_stream)
            native_context.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--preflight-report', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--eval-modes', default=','.join(DEFAULT_EVAL_MODES))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.preflight:
        if args.out is not None or args.preflight_report.exists():
            raise ValueError('Preflight writes exactly one new report; no native output directory')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        receipt = preflight(tokenizer, args.data, evaluation_modes(args.eval_modes))
        import sys
        if any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules):
            raise AssertionError('Tokenizer-only preflight unexpectedly imported MLX')
        dump(args.preflight_report, receipt)
        print(json.dumps({s: {k: v for k, v in group.items() if k not in ('rows', 'cases')}
            for s, group in receipt['splits'].items()}), flush=True)
    else:
        if args.out is None:
            raise ValueError('Native phase requires a new output directory')
        run(args.out, args.preflight_report, args.data)


if __name__ == '__main__':
    main()
