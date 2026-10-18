"""Prospective fixed 36-arm endpoint experiment. Native run requires root launch."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .data import HERE, PARENT, ROOT, records, sha
from .layout import VERSION, assemble_payload, build_layout, fork_root, token_inventory

MODEL = ROOT.parents[2] / '.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53'
WIDTHS = (1, 3, 6)
MODES = ('frozen', 'paired-STB-adapter')
CONTROL_FIELDS = {1: ('C1',), 3: ('S1', 'T1', 'C1'), 6: ('S1', 'T1', 'B1', 'C1', 'T2', 'C2')}
SOURCES = ('above_fold_copy/layout.py', 'above_fold_copy/data.py', 'above_fold_copy/benchmark.py',
    'above_fold.py', 'ABOVE_FOLD_WORKLOAD.json', 'branch_local_copy.py', 'parallel_decode.py',
    'shared_evidence_experiment.py', 'scheduler_identity_experiment.py', 'fusion.py', 'evidence_plan.py', 'page_plan.py',
    'semantic_copy_feature_fit.py', 'semantic_copy_training.py', 'semantic_copy_tokens.py', 'backbone.py', 'boundary.py')


def dump(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def control_jobs(rows, width):
    if width not in WIDTHS: raise ValueError('Fixed matched widths only')
    by_label = {r['fields'][0]: r for r in rows}
    return [{**copy.deepcopy(by_label[f]), 'maxTokens': 7} for f in CONTROL_FIELDS[width]]


def completed_adapter(fit_path):
    fit = json.loads(fit_path.read_text()); checkpoint = fit_path.parent / 'adapter.safetensors'
    if (fit.get('status') != 'completed' or fit.get('phase') != 'completed' or
            fit.get('nativeUpdates') != 504 or fit.get('optimizerUpdateCalls') != 504 or len(fit.get('steps', [])) != 504):
        raise ValueError('Completed final 504-update fit required; no interim checkpoint')
    if not all(s['status'] == 'completed' and s['updateEvaluated'] for s in fit['steps']):
        raise ValueError('Incomplete optimizer receipts')
    if not fit['checkpoint']['exact'] or sha(checkpoint) != fit['checkpoint']['sha256']:
        raise ValueError('Exact final adapter checkpoint receipt mismatch')
    if not fit['controls']['frozenLanguageHashesUnchanged']:
        raise ValueError('Frozen base proof missing')
    if not fit['controls']['learnedCacheProof']['matchedForcedContinuationLogitsExact']:
        raise ValueError('Learned cache proof missing')
    return {'fitReportPath': str(fit_path.resolve()), 'fitReportSha256': sha(fit_path),
        'checkpointPath': str(checkpoint.resolve()), 'checkpointSha256': sha(checkpoint),
        'nativeUpdates': 504, 'CTrainingCoverage': False, 'label': 'paired-STB-adapter; no C training'}


def preflight(tokenizer):
    from ..semantic_copy_tokens import build_training_tokens
    from importlib.metadata import version
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for name, expected in manifest['fileSha256'].items():
        if sha(HERE / name) != expected: raise ValueError('Frozen extension changed: ' + name)
    cases = []; training_caps = []
    for split in ('train', 'dev', 'test'):
        fields = records(HERE / f'fields.{split}.jsonl')
        for case in records(HERE / f'cases.{split}.jsonl'):
            root, jobs, _ = build_layout(case['fullSourcePagePlan'], case['request'], case['catalog'])
            inv = token_inventory(tokenizer, root, jobs)
            selected = [f for f in fields if f['caseId'] == case['caseId']]
            by_id = {f['rowId']: f for f in selected}; receipts = []
            for job in jobs:
                field = by_id[job['rowId']]
                if field['messages'] != [root] + job['messages']: raise ValueError('Serving/teacher prompt mismatch')
                token = build_training_tokens(tokenizer, [], field)
                receipts.append({'field': field['field'], 'sampledTargetTokensIncludingEOS': len(token['targetTokenIds']),
                    'promptTokens': token['promptTokens'], 'fullTeacherTokens': token['fullTokens'],
                    'promptSha256': token['promptSha256'], 'modelInputSha256': token['modelInputSha256']})
            if split == 'train': training_caps += receipts
            else:
                cases.append({'split': split, 'caseId': case['caseId'], 'request': case['request'], 'catalog': case['catalog'],
                    'plan': case['fullSourcePagePlan'], 'inventory': inv,
                    'universalTagsFirstSection': case['publicEvidence']['sections'][0]['universalTags'],
                    'teacherTokenFitOnly': receipts, 'parentSplitCasesPreviouslyInspectedDuringDataPreparation': True})
    if len(cases) != 6: raise ValueError('Fixed six inspected-parent cases required')
    if not any(c['universalTagsFirstSection'] for c in cases) or not any(not c['universalTagsFirstSection'] for c in cases):
        raise ValueError('Both universal and empty-intersection cases required')
    return {'schemaVersion': 1, 'status': 'CPU-tokenizer-preflight-passed', 'modelCalls': 0, 'nativeExecution': False,
        'layoutVersion': VERSION, 'datasetManifestSha256': sha(HERE / 'manifest.json'),
        'dataSha256': manifest['fileSha256'], 'sources': {n: sha(ROOT / n) for n in SOURCES},
        'tokenizerSha256': {n: sha(MODEL / n) for n in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')},
        'runtime': {n: version(n) for n in ('mlx', 'mlx-lm')},
        'cases': cases, 'fixedCases': 6, 'widths': list(WIDTHS), 'modelModes': list(MODES), 'fixedArms': 36,
        'requiredFields': ['S1', 'T1', 'B1', 'T2', 'B2', 'T3', 'B3', 'C1', 'C2', 'C3'], 'totalCapPerView': 208,
        'trainingCorpusPreflightRows': len(training_caps),
        'maximumTrainingTeacherContext': max(r['fullTeacherTokens'] for r in training_caps),
        'controls': {'caseId': cases[0]['caseId'], 'widths': list(WIDTHS), 'passes': ['fork', 'independent', 'repeat'],
            'actualLaneFields': {str(w): list(CONTROL_FIELDS[w]) for w in WIDTHS}, 'capPerControlRow': 7,
            'modelModes': list(MODES), 'diagnosticOnlyNotQualityDenominator': True},
        'limits': ['Copy-stage endpoint only; no ranking, UI paint, conversion or full-page speedup claim.',
            'Same ten fields and caps at all widths; cross-width numerical wording differences are compared semantically.',
            'Matched-width fork/rebuild/repeat tokens/distributions must be exact; mismatch aborts before timing arms.',
            'Paired-STB adapter received no C training; this experiment tests transfer, not the new C supervision fit.',
            'All parent cases inspected in data preparation; six fixed denominators, not new sealed holdouts.']}


def freeze(preflight_path, fit_path, packet_path):
    if packet_path.exists(): raise FileExistsError('Preserve frozen benchmark packet')
    value = json.loads(preflight_path.read_text())
    if value['sources'] != {n: sha(ROOT / n) for n in SOURCES}: raise ValueError('Preflight source changed')
    packet = {**value, 'preflightPath': str(preflight_path.resolve()), 'preflightSha256': sha(preflight_path),
        'adapter': completed_adapter(fit_path), 'memoryGuardGiB': 7, 'prefillStep': 256,
        'nativeLaunchRequiresRootReview': True}
    dump(packet_path, packet)


def matched_controls(bb, root_message, rows, mode_receipt, out):
    """Actual N-lane, same-shape controls; native owner only, not CPU preflight."""
    import importlib
    from .. import parallel_decode
    from ..branch_local_copy import prepare_root
    from ..native_batch_diagnosis import state_hash
    from ..shared_evidence_experiment import message_suffixes, same_outputs
    native = importlib.import_module('mlx_lm.generate'); results = {}
    for width in WIDTHS:
        chosen = control_jobs(rows, width); state, _ = prepare_root(bb, root_message)
        state['copyModeReceipt'] = mode_receipt; pristine = state_hash(state['cache']); arms = {}
        for kind in ('fork', 'independent', 'repeat'):
            distributions = {}; rebuilt = []
            original_class = native.BatchGenerator; original_clone = parallel_decode.clone_prefix_state
            class Observer(original_class):
                def insert(self, *args, **kwargs):
                    uids = super().insert(*args, **kwargs)
                    for uid, row in zip(uids, chosen, strict=True): distributions[uid] = {'rowId': row['rowId'], 'hashes': []}
                    return uids
                def next_generated(self):
                    responses = super().next_generated()
                    for response in responses:
                        vector = bb.np.array(response.logprobs.astype(bb.mx.float32))
                        if not bb.np.isfinite(vector).all(): raise FloatingPointError('Nonfinite control distribution')
                        distributions[response.uid]['hashes'].append(hashlib.sha256(vector.tobytes()).hexdigest())
                    return responses
            def clone(backbone, source):
                if source is not state or source.get('copyModeReceipt') != mode_receipt:
                    raise ValueError('Wrong root/adapter configuration; no adapted-prefix substitution')
                if kind == 'independent':
                    fresh, _ = prepare_root(backbone, root_message)
                    actual = state_hash(fresh['cache']); rebuilt.append(actual)
                    if actual != pristine: raise AssertionError('Independent complete root differs')
                    return fresh
                return fork_root(source, root_message, state['tokens'], bb.model,
                    lambda leaves: (bb.mx.eval(leaves), bb.mx.synchronize()))
            native.BatchGenerator = Observer; parallel_decode.clone_prefix_state = clone
            output = None
            try:
                with message_suffixes(): output, timing = parallel_decode.decode_rows(bb, state, chosen, width, prefill_step_size=256)
                bb.mx.synchronize(native.generation_stream); bb.check_memory()
                by_row = {d['rowId']: d['hashes'] for d in distributions.values()}
                for row in output:
                    row['distributionHashes'] = by_row[row['rowId']]
                    if len(row['distributionHashes']) != row['arSteps']: raise AssertionError('Control sample/hash count differs')
                if state_hash(state['cache']) != pristine: raise AssertionError('Pristine complete root mutated')
                if timing['maximumRowsEmittingTogether'] != width or len(output) != width:
                    raise AssertionError('Control did not exercise the claimed actual width')
                if kind == 'independent' and len(rebuilt) != width: raise AssertionError('Not every lane independently rebuilt')
                arms[kind] = {'rows': output, 'timing': timing, 'independentRootHashes': rebuilt, 'rootHash': pristine}
                if kind != 'fork' and not same_outputs(arms['fork']['rows'], output):
                    raise AssertionError('Fatal same-width state/rebuild/repeat distribution mismatch')
            except Exception as exc:
                dump(out / f'width-{width}-{kind}-failure.json', {'status': 'failed',
                    'error': {'type': type(exc).__name__, 'message': str(exc)},
                    'width': width, 'actualControlFields': list(CONTROL_FIELDS[width]),
                    'rowsIfReturned': output, 'observedDistributions': list(distributions.values()),
                    'independentRootHashes': rebuilt, 'referenceRootHash': pristine})
                raise
            finally:
                native.BatchGenerator = original_class; parallel_decode.clone_prefix_state = original_clone
            dump(out / f'width-{width}-{kind}.json', arms[kind]); gc.collect(); bb.mx.clear_cache()
        results[str(width)] = {'status': 'same-width-exact', 'actualLanes': width, 'arms': arms}
        del state; gc.collect(); bb.mx.clear_cache()
    return results


def run(packet_path, out):
    """No caller in this lane invokes this; root must separately review/launch."""
    import psutil
    from transformers import AutoTokenizer
    from ..semantic_copy_feature_fit import model_input_receipt, verify_model_files
    packet = json.loads(packet_path.read_text())
    if out.exists(): raise FileExistsError('Preserve previous native attempt')
    if psutil.virtual_memory().available < 7 * 2**30: raise MemoryError('Less than 7 GiB available')
    if packet['sources'] != {n: sha(ROOT / n) for n in SOURCES}: raise ValueError('Frozen producer source changed')
    if completed_adapter(Path(packet['adapter']['fitReportPath'])) != packet['adapter']: raise ValueError('Completed fit/adapter changed')
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    if preflight(tokenizer) != json.loads(Path(packet['preflightPath']).read_text()): raise ValueError('CPU token preflight changed')
    if sha(Path(packet['preflightPath'])) != packet['preflightSha256']: raise ValueError('Preflight file changed')
    verification = verify_model_files(model_input_receipt())
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.generate import generation_stream
    from mlx_lm.tuner.lora import LoRALinear
    from ..backbone import FrozenBackbone
    from ..branch_local_copy import prepare_root
    from ..evidence_plan import REVISION
    from ..native_batch_diagnosis import state_hash
    from ..parallel_decode import decode_rows
    from ..shared_evidence_experiment import message_suffixes
    from ..semantic_copy_training import disabled, null_context
    class GuardedBackbone(FrozenBackbone):
        def check_memory(self):
            super().check_memory()
            if max(self.mx.get_active_memory(), self.mx.get_peak_memory()) > 7 * 2**30:
                raise MemoryError('Active/peak native memory exceeded 7-GiB guard')
    out.mkdir(parents=True); start = time.perf_counter()
    report = {'status': 'started', 'packetSha256': sha(packet_path), 'executionCommit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'sourceSha256': packet['sources'], 'modelVerification': verification, 'fixedCases': 6, 'fixedArms': 36,
        'scope': 'first-section ten-field copy-stage payload only; locked recorded decisions',
        'CTrainingCoverage': False, 'controls': {}, 'arms': [dict(split=c['split'], caseId=c['caseId'], mode=m, width=w, status='not-attempted')
            for c in packet['cases'] for m in MODES for w in WIDTHS]}
    def save():
        report['wallMs'] = (time.perf_counter() - start) * 1000; dump(out / 'results.json', report)
    save()
    try:
        bb = GuardedBackbone(MODEL, REVISION, 2048, 7); model = bb.model; model.freeze(); model.eval()
        adapter = LoRALinear.from_base(model.layers[-1].mlp.down_proj, r=4, scale=20, dropout=0)
        weights = mx.load(packet['adapter']['checkpointPath'])
        if set(weights) != {'lora_a', 'lora_b'}: raise ValueError('Unexpected adapter checkpoint members')
        adapter.load_weights(list(weights.items()), strict=False); adapter.freeze(); model.layers[-1].mlp.down_proj = adapter
        model.eval(); mx.eval(model.parameters()); report['residentLoadMs'] = bb.load_ms
        if dict(tree_flatten(model.trainable_parameters())): raise ValueError('Benchmark model must be fully frozen')
        first = packet['cases'][0]; root, jobs, _ = build_layout(first['plan'], first['request'], first['catalog'])
        for mode in MODES:
            dest = out / ('controls-' + mode); dest.mkdir()
            with disabled(model, adapter) if mode == 'frozen' else null_context():
                receipt = 'frozen-base' if mode == 'frozen' else packet['adapter']['checkpointSha256']
                report['controls'][mode] = matched_controls(bb, root, jobs, receipt, dest)
            save(); gc.collect(); mx.clear_cache()
        # Rotate widths and model order by case, retaining all thirty-six arms.
        for index, case in enumerate(packet['cases']):
            modes = MODES if index % 2 == 0 else tuple(reversed(MODES))
            widths = WIDTHS[index % 3:] + WIDTHS[:index % 3]
            for mode in modes:
                for width in widths:
                    arm = next(a for a in report['arms'] if (a['caseId'], a['mode'], a['width']) == (case['caseId'], mode, width))
                    arm['status'] = 'running'; save(); started = time.perf_counter()
                    with disabled(model, adapter) if mode == 'frozen' else null_context():
                        root, jobs, evidence = build_layout(case['plan'], case['request'], case['catalog'])
                        inventory = token_inventory(bb.tokenizer, root, jobs)
                        if inventory != case['inventory']: raise ValueError('Native token inventory differs from CPU preflight')
                        metadata_ready = (time.perf_counter() - started) * 1000
                        state, preparation = prepare_root(bb, root); original = state_hash(state['cache'])
                        decoder_call_start = (time.perf_counter() - started) * 1000
                        with message_suffixes(): rows, timing = decode_rows(bb, state, jobs, width, prefill_step_size=256)
                        # BatchGenerator may queue its next step on this stream.
                        # Fence it explicitly before memory and readiness clocks;
                        # close()'s wired-limit cleanup fence is conditional.
                        mx.synchronize(generation_stream); bb.check_memory()
                        if state_hash(state['cache']) != original: raise AssertionError('Fresh same-mode root mutated')
                        result = assemble_payload(rows, case['plan'], case['request'], case['catalog'], require_native_receipts=True)
                        encoded = json.dumps(result['payload'], sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
                        ready = (time.perf_counter() - started) * 1000
                    valid = result['validation']['valid']
                    arm.update(status='completed-attempt', **result, inputInventory=inventory, publicEvidence=evidence,
                        payloadBytes=len(encoded), actualOutputTokens=sum(len(r['tokenIds']) for r in rows), actualArSteps=sum(r['arSteps'] for r in rows),
                        payloadCompletionReceipt={'payloadSha256': result['payloadSha256'], 'metadataReadyAtMs': metadata_ready,
                            'validatedAndSerializedAtMs': ready},
                        copyStagePayloadReadyMs=ready if valid else None, attemptCompletionMs=ready,
                        lastRequiredTokenConservativeLowerBoundMs=decoder_call_start + max(r['completedMs'] for r in rows),
                        rowClockBridge='outer decoder-call offset precedes internal wall_start; conservative lower bound only; actual ready receipt measured directly',
                        preparation=preparation, timing=timing, memory=bb.runtime_memory(), originalRootStateHash=original)
                    del state, rows; gc.collect(); mx.clear_cache(); save()
        report['status'] = 'completed'; report['memory'] = bb.runtime_memory(); save()
    except Exception as exc:
        report['status'] = 'blocked'; report['blocker'] = {'type': type(exc).__name__, 'message': str(exc)}
        if 'arm' in locals() and arm.get('status') == 'running':
            arm.update(status='failed', error=report['blocker'])
        report['controlArtifacts'] = sorted(str(p.relative_to(out)) for p in out.glob('controls-*/*.json'))
        if 'bb' in locals(): report['memory'] = bb.runtime_memory()
        save(); raise


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('action', choices=('preflight', 'freeze', 'run'))
    p.add_argument('--preflight', type=Path); p.add_argument('--packet', type=Path); p.add_argument('--fit-report', type=Path); p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.action == 'preflight':
        if a.preflight is None or a.preflight.exists(): raise ValueError('Supply one new CPU preflight report')
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
        dump(a.preflight, preflight(tok))
    elif a.action == 'freeze':
        if None in (a.preflight, a.fit_report, a.packet): raise ValueError('Freeze needs preflight, completed fit and new packet')
        freeze(a.preflight, a.fit_report, a.packet)
    else:
        if a.packet is None or a.out is None: raise ValueError('Root-reviewed native launch needs packet and new output directory')
        run(a.packet, a.out)


if __name__ == '__main__': main()
