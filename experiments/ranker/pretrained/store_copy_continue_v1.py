"""Fixed 960-update continuation of the completed actual-store 320-step fit.

No native imports at module import or tokenizer preflight. Original archives,
labels, prompts, validators and results remain immutable. This is a new Adam
trajectory from the exact 320-step adapter, not an optimizer-state resume.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

from . import store_copy_fit_v2 as original
from .semantic_copy_feature_fit import (MODEL, ROOT, digest, dump, hashed,
    load_feature, model_input_receipt, verify_model_files)

ORIGINAL = Path('data/ranker/pretrained/actual-store-copy-fit-v2-attempt-1')
WARM_SHA = '56c6c1f4199d8689f1920d871213a46d2d121fec4e5836ff70e33aaf98c7c7b8'
SEEDS = (19080, 19081, 19082)
MODES = ('frozen', 'failed-daypack', 'continued-store')
CONFIG = {**original.CONFIG, 'seed': SEEDS[0], 'updates': 960, 'epochs': 3,
    'priorUpdates': 320, 'finalTotalUpdates': 1280, 'epochSeeds': list(SEEDS),
    'initialization': 'exact completed actual-store 320-update adapter; fresh zero Adam state',
    'optimizerContinuity': 'original optimizer state was not saved; new Adam trajectory, not exact resume',
    'order': 'three predeclared independent seeded permutations of all 320 original TRAIN rows',
    'gates': 'reuse all320 original exact gates; learned original-feature0 native parity before and after training',
    'probeControl': 'learned adapter without numerical replacement; no temporary nonzero control',
    'evaluation': 'development replay: fixed8 original EVAL requests x frozen/failed-daypack/continued-store width6',
    'rootProof': 'all16 ordinary TRAIN root cache hashes equal original archive before and after continuation',
    'selection': 'final additional step960 / total1280 fixed before evaluation; no EVAL checkpoint selection'}
SOURCES = (*original.SOURCES, 'store_copy_continue_v1.py', 'test_store_copy_continue_v1.py')


def permutations():
    orders = []
    for seed in SEEDS:
        order = list(range(320)); random.Random(seed).shuffle(order); orders.append(order)
    return orders


def validate_completed_original(report):
    """Reject partial, changed, wrong-checkpoint or incomplete baseline evidence."""
    if report.get('schemaVersion') != 'actual-store-copy-fit.v2' or report.get('status') != 'completed' or report.get('phase') != 'completed':
        raise ValueError('Original actual-store run must be completed, including every evaluation arm')
    if report.get('config') != original.CONFIG or report.get('nativeUpdates') != 320 or report.get('optimizerUpdateCalls') != 320:
        raise ValueError('Original exact 320-update configuration/actual counts required')
    steps, extraction = report.get('steps', []), report.get('extraction', [])
    if len(steps) != 320 or [s.get('step') for s in steps] != list(range(320)) or any(
            s.get('status') != 'completed' or s.get('updateEvaluated') is not True for s in steps):
        raise ValueError('Original320 actual completed update receipts required')
    if [s.get('featureIndex') for s in steps] != original.shuffled_order():
        raise ValueError('Original update permutation differs')
    if len(extraction) != 320 or [r.get('index') for r in extraction] != list(range(320)) or any(
            r.get('gateExact') is not True or r.get('trueZeroBRestored') is not True for r in extraction):
        raise ValueError('Original all320 exact feature gates required')
    checkpoint = report.get('checkpoint', {})
    if checkpoint.get('sha256') != WARM_SHA or checkpoint.get('exact') is not True or (
            checkpoint.get('parametersBeforeReload') != checkpoint.get('parametersAfterReload')) or (
            checkpoint.get('lossBeforeReload') != checkpoint.get('lossAfterReload')):
        raise ValueError('Exact completed actual-store warm checkpoint/reload receipt required')
    if len(report.get('frozenBaseParameterHashes', {})) != 927:
        raise ValueError('Original927 frozen parameter hashes required')
    controls = report.get('controls', {})
    for key in ('all320NonzeroBGatesPassed', 'fitStartedFromExactFrozenBaseZeroB',
            'learnedOriginalArchivedNativeParityExact', 'frozenLanguageHashesUnchanged',
            'postEvaluationFrozen927Unchanged', 'sourceFilesUnchanged', 'datasetManifestUnchanged', 'originalPriorUnchanged'):
        if controls.get(key) is not True:
            raise ValueError('Missing original native control: ' + key)
    ids = [c['caseId'] for c in report['preflight']['splits']['eval']['cases']]
    expected = {(case_id, mode) for case_id in ids for mode in original.MODES}
    arms = report.get('arms', [])
    if len(ids) != 8 or len(set(ids)) != 8 or len(arms) != 24 or {
            (a.get('caseId'), a.get('mode')) for a in arms} != expected or any(
            a.get('status') != 'completed-attempt' or a.get('width') != 6 for a in arms):
        raise ValueError('Original fixed24 width6 evaluation arms must all have finished')
    if report.get('summary') != original.summary(arms):
        raise ValueError('Original summary does not reproduce fixed24 arm receipts')
    for mode in original.MODES:
        if controls.get(mode, {}).get('status') != 'exact-controls-passed':
            raise ValueError('Missing original exact width6 control: ' + mode)


def contained_file(root, relative):
    root = Path(root).resolve(); path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError('Original receipt path must be an existing contained regular file')
    return path


def original_receipt(path=ORIGINAL):
    """CPU-only immutable inventory, row-by-row SHA/metadata/tensor validation."""
    path = Path(path)
    report = json.loads((path / 'results.json').read_text()); validate_completed_original(report)
    if digest(path / 'adapter.safetensors') != WARM_SHA:
        raise ValueError('Completed actual-store warm checkpoint bytes differ')
    original.assert_source_fresh(report['preflight'], original.DATA)
    planned = report['preflight']['splits']['train']['rows']
    if len(planned) != 320:
        raise ValueError('Original preflight must contain320 TRAIN rows')
    for index, (entry, plan) in enumerate(zip(report['extraction'], planned, strict=True)):
        if (entry['caseId'], entry['rowId'], entry['modelInputSha256']) != (
                plan['caseId'], plan['rowId'], plan['modelInputSha256']):
            raise ValueError('Original feature row differs from frozen TRAIN serving roster')
        for key in ('archive', 'metadata'):
            contained_file(path, entry[key])
        gate_path = contained_file(path, f'gates/{index:04d}.json')
        if digest(gate_path) != entry['gateSha256']:
            raise ValueError('Original exact gate file changed')
        gate = json.loads(gate_path.read_text())
        if gate.get('status') != 'exact-gate-passed' or gate.get('sourceStatePreserved') is not True or (
                gate.get('featureTensors') != entry['featureTensors']) or (
                gate['contract']['modelInputSha256'] != plan['modelInputSha256']):
            raise ValueError('Original gate/feature/preflight identity differs')
        item = load_feature(entry, path)
        if (item['caseId'], item['rowId'], item['contract']['modelInputSha256']) != (
                entry['caseId'], entry['rowId'], plan['modelInputSha256']):
            raise ValueError('Original feature metadata identity differs')
        del item
    inventory = {}
    for item in sorted(path.rglob('*')):
        if item.is_file():
            relative = str(item.relative_to(path)); contained_file(path, relative)
            inventory[relative] = {'bytes': item.stat().st_size, 'sha256': digest(item)}
    return {'path': str(path.resolve()), 'resultsSha256': digest(path / 'results.json'),
        'checkpointSha256': WARM_SHA, 'files': inventory, 'fileCount': len(inventory),
        'all320OriginalRowsReverified': True, 'residentHostFeatureRows': 1,
        'originalSummary': report['summary'], 'originalTotalUpdates': 320,
        'optimizerStateSaved': False}


def preflight(tokenizer, prior=ORIGINAL):
    receipt = original_receipt(prior)
    report = json.loads((Path(prior) / 'results.json').read_text())
    base = original.preflight(tokenizer, original.DATA)
    if base != report['preflight']:
        raise ValueError('Original completed preflight no longer matches exact tokenizer/data/source')
    return {'schemaVersion': 'actual-store-copy-continuation.v1', 'status': 'tokenizer-preflight-passed',
        'nativeExecution': False, 'modelCalls': 0, 'trainingUpdates': 0, 'config': CONFIG,
        'original': receipt, 'originalPreflightSha256': report['preflightSha256'],
        'datasetManifestSha256': base['datasetManifestSha256'], 'payloadSha256': base['payloadSha256'],
        'sources': {name: digest(ROOT / name) for name in SOURCES}, 'modelFiles': model_input_receipt(),
        'evalModes': list(MODES), 'fixedArms': 24, 'widths': [6], 'splits': base['splits'],
        'permutations': permutations(), 'permutationSha256': hashed(permutations()),
        'limitations': [v for v in base['limitations'] if not v.startswith('Temporary nonzero-B')]
            + ['Original EVAL now inspected; every evaluation here is development replay, never unseen generalization.',
               'Fresh Adam state at320-update checkpoint; optimizer trajectory differs from uninterrupted1280-step training.',
               'Schedule fixed at960 additional TRAIN-only updates before continuation evaluation; no EVAL-target tuning.',
               'Features remain in the immutable original run; one complete host row resident; no re-extraction or replacement.',
               '320-step baseline and1280-total continuation results remain in separate run reports.']}


def arm_roster(case_ids):
    return [{'caseId': case_id, 'split': 'eval-development-replay', 'mode': mode,
        'width': 6, 'status': 'not-attempted'} for case_id in case_ids for mode in MODES]


def summary(arms):
    # Reuse exactly the original quality/latency arithmetic, with an explicit new
    # name for the continued arm. Original reports and mode constants untouched.
    renamed = [{**a, 'mode': 'new-store' if a['mode'] == 'continued-store' else a['mode']} for a in arms]
    values = original.summary(renamed)
    values['continued-store'] = values.pop('new-store')
    return values


@contextmanager
def stage_receipt(report, save, name, seconds):
    """Persist running/completed/failed receipt around every native stage."""
    record = {'stage': name, 'status': 'running', 'deadlineSeconds': seconds}
    report.setdefault('stages', []).append(record); report['phase'] = name; save()
    started = time.perf_counter()
    try:
        with original.deadline(seconds, name):
            yield record
        record['status'] = 'completed'
    except BaseException as exc:
        record.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        record['ms'] = (time.perf_counter() - started) * 1000
        save()


def assert_inventory_preserved(receipt):
    path = Path(receipt['path'])
    current = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    if current != set(receipt['files']):
        raise ValueError('Original run file inventory changed')
    for name, expected in receipt['files'].items():
        item = contained_file(path, name)
        if item.stat().st_size != expected['bytes'] or digest(item) != expected['sha256']:
            raise ValueError('Original run bytes changed: ' + name)


def validate_zero_adam_state(state):
    """CPU-testable distinction between zero moments and nonzero learning rate."""
    import numpy as np
    if set(state) != {'step', 'learning_rate', 'lora_a', 'lora_b'} or int(state['step'].item()) != 0:
        raise ValueError('Fresh Adam step0 and exact A/B state roster required')
    for key, shape in (('lora_a', (12288, 4)), ('lora_b', (4, 4096))):
        if set(state[key]) != {'m', 'v'}:
            raise ValueError('Fresh Adam requires exactly m/v moments for A/B')
        for value in state[key].values():
            host = np.array(value)
            if host.shape != shape or host.dtype != np.float32 or not np.isfinite(host).all() or np.any(host != 0):
                raise ValueError('Fresh Adam requires exact zero float32 A/B moments')
    if not np.isfinite(np.array(state['learning_rate'])).all() or float(state['learning_rate'].item()) <= 0:
        raise ValueError('Fresh Adam requires finite positive declared learning rate')
    return {'step': 0, 'allMomentsExactlyZero': True,
        'momentShapes': {'lora_a': [12288, 4], 'lora_b': [4, 4096]}}


def run(out, preflight_path, prior=ORIGINAL):
    started = time.perf_counter(); out, preflight_path, prior = Path(out), Path(preflight_path), Path(prior)
    if out.exists() or out.resolve().is_relative_to(prior.resolve()) or prior.resolve().is_relative_to(out.resolve()):
        raise FileExistsError('Separate new output directory required; preserve original run')
    out.mkdir(parents=True)
    report = {'schemaVersion': 'actual-store-copy-continuation.v1', 'status': 'started',
        'phase': 'preflight-load', 'config': CONFIG, 'nativeUpdates': 0, 'optimizerUpdateCalls': 0,
        'priorCompletedUpdates': 320, 'totalCompletedUpdates': 320, 'steps': [], 'stages': [],
        'arms': [], 'fixedArms': 24, 'controls': {}, 'featureArtifactsWritten': 0,
        'wallScope': 'run entry including tokenizer/preflight/all artifact verification; excludes Python startup'}
    def save(event=None):
        report['wallMs'] = (time.perf_counter() - started) * 1000
        if report['arms']:
            report['summary'] = summary(report['arms'])
        dump(out / 'results.json', report)
        if event:
            print(json.dumps({'event': event, 'phase': report['phase'], 'updates': report['nativeUpdates'],
                'totalUpdates': report['totalCompletedUpdates'], 'wallMs': report['wallMs']}), flush=True)
    save('started')
    try:
        import numpy as np
        import psutil
        from transformers import AutoTokenizer
        frozen = json.loads(preflight_path.read_text())
        case_ids = [c['caseId'] for c in frozen['splits']['eval']['cases']]
        if len(case_ids) != 8 or len(set(case_ids)) != 8:
            raise ValueError('Exactly8 predeclared evaluation cases required')
        report.update(preflightSha256=digest(preflight_path), preflight=frozen,
            arms=arm_roster(case_ids), limitations=frozen['limitations'],
            executionCommit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        with stage_receipt(report, save, 'preflight-verification', 180):
            tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
            if preflight(tokenizer, prior) != frozen:
                raise ValueError('Continuation data/source/model/original-run changed after reviewed preflight')
            baseline = json.loads((prior / 'results.json').read_text())
            _, packet = original.dataset_packet()
        available = psutil.virtual_memory()
        report['hostMemoryBeforeLoad'] = {'availableBytes': available.available, 'totalBytes': available.total,
            'percent': available.percent, 'swap': psutil.swap_memory()._asdict()}
        if available.available < CONFIG['memoryGiB'] * 2**30:
            raise MemoryError('Less than declared7 GiB available before native load')
        with stage_receipt(report, save, 'model-verification', 180):
            report['modelVerification'] = verify_model_files(frozen['modelFiles'])
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
        from .semantic_copy_gradient_features import (GradientFeatureProbeError,
            probe_gradient_features, feature_loss)
        class GuardedBackbone(FrozenBackbone):
            def check_memory(self):
                super().check_memory()
                if max(self.mx.get_peak_memory(), self.mx.get_active_memory()) > self.memory_budget:
                    raise MemoryError('Native active/peak exceeded unchanged7 GiB guard')
        native_context = mx.stream(generation_stream); native_context.__enter__()
        with stage_receipt(report, save, 'load-base-and-warm-adapter', 180):
            mx.random.seed(CONFIG['seed']); bb = GuardedBackbone(MODEL, REVISION, 4096, CONFIG['memoryGiB'])
            model = bb.model; model.freeze(); model.eval(); report['residentLoadMs'] = bb.load_ms
            base_hashes = parameter_hashes(model.language_model)
            if base_hashes != baseline['frozenBaseParameterHashes'] or len(base_hashes) != 927 or dict(tree_flatten(model.trainable_parameters())):
                raise AssertionError('Exactly original927 frozen language tensor hashes and no base trainables required')
            report['frozenBaseParameterHashes'] = base_hashes
            adapter = LoRALinear.from_base(model.layers[-1].mlp.down_proj, r=4, scale=20, dropout=0)
            model.layers[-1].mlp.down_proj = adapter; model.eval()
        def leaves():
            return dict(tree_flatten(adapter.trainable_parameters()))
        def parameter_receipts():
            result = {}
            for key in ('lora_a', 'lora_b'):
                value = getattr(adapter, key); host = np.array(value)
                if value.dtype != mx.float32 or not np.isfinite(host).all():
                    raise ValueError('Finite float32 adapter A/B required')
                result[key] = {'dtype': str(value.dtype), 'shape': list(value.shape),
                    'sha256': hashlib.sha256(host.tobytes()).hexdigest()}
            return result
        def load_adapter(path):
            weights = mx.load(str(path))
            if set(weights) != {'lora_a', 'lora_b'} or tuple(weights['lora_a'].shape) != (12288, 4) or tuple(weights['lora_b'].shape) != (4, 4096):
                raise ValueError('Only original final-down rank4 shapes allowed')
            for key, value in weights.items():
                setattr(adapter, key, value)
            mx.eval(adapter.parameters()); mx.synchronize(); parameter_receipts()
        def verify_base(label):
            with stage_receipt(report, save, label + '-frozen927-verification', 120):
                current = {k.replace('.31.mlp.down_proj.linear.', '.31.mlp.down_proj.'): v
                    for k, v in parameter_hashes(model.language_model).items()}
                if current != base_hashes:
                    raise AssertionError('Original frozen927 language tensor bytes changed at ' + label)
                report['controls'][label + 'Frozen927Unchanged'] = True
        with stage_receipt(report, save, 'warm-adapter-verification', 60):
            load_adapter(prior / 'adapter.safetensors')
            initial_receipts = parameter_receipts()
            if initial_receipts != baseline['checkpoint']['parametersAfterReload']:
                raise AssertionError('Loaded adapter differs from exact320-step completed checkpoint')
            initial_host = {k: np.array(v) for k, v in leaves().items()}
            roster = dict(tree_flatten(model.trainable_parameters()))
            if len(roster) != 2 or sum(v.size for v in roster.values()) != 65536 or any(
                    '.31.mlp.down_proj.' not in k or k.rsplit('.', 1)[-1] not in ('lora_a', 'lora_b') for k in roster):
                raise AssertionError('Only layer31 final-down65536 A/B float32 trainables allowed')
            report['initialAdapter'] = initial_receipts
            report['trainableParameters'] = {k: int(v.size) for k, v in roster.items()}
        def root_proof(label):
            proofs = report['controls'].setdefault(label + 'All16TrainRootCaches', [])
            for case in packet['train']['cases']:
                with stage_receipt(report, save, label + '-train-root-' + case['caseId'], 120):
                    common, _, _ = original.serving_rows(case, [f for f in packet['train']['fields'] if f['caseId'] == case['caseId']])
                    state, prep = prepare_root(bb, common); _complete_cache(state['cache'], len(state['tokens']))
                    sha = state_hash(state['cache'])
                    expected = {e['sourceStateHash'] for e in baseline['extraction'] if e['caseId'] == case['caseId']}
                    if expected != {sha}:
                        raise AssertionError('Learned ordinary root differs from original TRAIN source cache')
                    proofs.append({'caseId': case['caseId'], 'sha256': sha, 'originalExact': True, 'preparation': prep})
                    del state; gc.collect(); mx.clear_cache(); bb.check_memory()
            if len(proofs) != 16:
                raise AssertionError('All16 TRAIN roots must have exact original cache parity')
        def archive_native_parity(label):
            with stage_receipt(report, save, label + '-original-feature0-native-parity', 120):
                case = packet['train']['cases'][0]
                common, _, fields = original.serving_rows(case, [f for f in packet['train']['fields'] if f['caseId'] == case['caseId']])
                state, _ = prepare_root(bb, common); _complete_cache(state['cache'], len(state['tokens']))
                entry = baseline['extraction'][0]
                if state_hash(state['cache']) != entry['sourceStateHash']:
                    raise AssertionError('Fresh learned root differs from original archived source cache')
                gate_path = out / (label + '-learned-probe.json')
                try:
                    gate, recaptured = probe_gradient_features(bb, adapter, state, fields[0])
                except GradientFeatureProbeError as exc:
                    dump(gate_path, exc.report); raise
                dump(gate_path, gate)
                if gate['status'] != 'exact-gate-passed' or recaptured is None or gate['featureTensors'] != entry['featureTensors']:
                    raise AssertionError('Learned exact native gate or original archived x/h/y bytes differ')
                del recaptured
                item = load_feature(entry, prior)
                loss, gradients = nn.value_and_grad(adapter, lambda: feature_loss(bb, adapter, item))()
                mx.eval(loss, gradients); mx.synchronize()
                def tensor_receipt(value):
                    host = np.array(value.astype(mx.float32)); dtype = str(value.dtype)
                    return {'dtype': dtype, 'shape': list(value.shape),
                        'sha256': hashlib.sha256(dtype.encode() + str(tuple(value.shape)).encode() + host.tobytes()).hexdigest()}
                archived = {'loss': float(loss.item()), 'lossTensor': tensor_receipt(loss),
                    'gradientTensors': {k: tensor_receipt(v) for k, v in tree_flatten(gradients)}}
                dump(out / (label + '-original-archive-parity.json'), {'featureIndex': 0, 'archived': archived,
                    'native': gate['arms']['direct'], 'exact': archived == gate['arms']['direct'],
                    'originalArchiveSha256': entry['archiveSha256']})
                if archived != gate['arms']['direct']:
                    raise AssertionError('Original archive learned loss/A/B gradients differ from native direct replay')
                report['controls'][label + 'OriginalFeature0NativeLossGradientsExact'] = True
                del item, gate, state, loss, gradients; gc.collect(); mx.clear_cache(); bb.check_memory()
        root_proof('before'); archive_native_parity('before'); verify_base('beforeFit')
        with stage_receipt(report, save, 'fresh-adam-initialization', 60):
            optimizer = optim.Adam(learning_rate=CONFIG['learningRate'])
            optimizer.init(adapter.trainable_parameters())
            state = optimizer.state
            mx.eval(state); mx.synchronize()
            report['optimizerInitialization'] = {'freshInstance': True, 'priorOptimizerStateLoaded': False,
                'originalOptimizerStateUnavailable': True, 'sameUninterruptedAdamTrajectory': False,
                'stateKeysBeforeFirstUpdate': sorted(state), 'learningRate': CONFIG['learningRate'],
                'zeroStateValidation': validate_zero_adam_state(state)}
            grad_fn = nn.value_and_grad(adapter, lambda item: feature_loss(bb, adapter, item))
        report['permutations'] = permutations(); save('continuation-started')
        for epoch, order in enumerate(report['permutations']):
            for position, index in enumerate(order):
                step = epoch * 320 + position; entry = baseline['extraction'][index]
                row = {'step': step, 'totalStep': 320 + step, 'epoch': epoch + 1, 'epochPosition': position,
                    'featureIndex': index, 'caseId': entry['caseId'], 'rowId': entry['rowId'], 'kind': entry['kind'],
                    'status': 'loading-feature', 'updateEvaluated': False}
                report['steps'].append(row)
                with stage_receipt(report, save, f'update-{step:04d}-feature-load-and-gradient', CONFIG['updateTimeoutSeconds']):
                    item = load_feature(entry, prior)
                    loss, gradients = grad_fn(item); mx.eval(loss, gradients); mx.synchronize()
                    flat = dict(tree_flatten(gradients))
                    if set(flat) != {'lora_a', 'lora_b'} or not bool(mx.isfinite(loss).item()) or not all(
                            bool(mx.all(mx.isfinite(v)).item()) for v in flat.values()):
                        raise FloatingPointError('Wrong/nonfinite feature loss or A/B gradients')
                    row.update(loss=float(loss.item()), gradientL2=sum(float(mx.sum(v.astype(mx.float32)**2).item())
                        for v in flat.values())**.5, status='update-attempted')
                with stage_receipt(report, save, f'update-{step:04d}-optimizer', CONFIG['updateTimeoutSeconds']):
                    optimizer.update(adapter, gradients); report['optimizerUpdateCalls'] += 1
                    mx.eval(adapter.parameters(), optimizer.state); mx.synchronize()
                    report['nativeUpdates'] += 1; report['totalCompletedUpdates'] = 320 + report['nativeUpdates']
                    row['updateEvaluated'] = True
                    arrays = list(leaves().values()) + [v for _, v in tree_flatten(optimizer.state) if hasattr(v, 'dtype')]
                    if not all(bool(mx.all(mx.isfinite(v)).item()) for v in arrays):
                        raise FloatingPointError('Nonfinite adapter/optimizer state after actual update')
                    row.update(status='completed', updateEvaluated=True); bb.check_memory()
                del item, loss, gradients, flat, arrays
                save('update-completed')
        changes = {k: float(np.abs(np.array(v) - initial_host[k]).sum()) for k, v in leaves().items()}
        if report['nativeUpdates'] != 960 or report['optimizerUpdateCalls'] != 960 or report['totalCompletedUpdates'] != 1280 or not all(v > 0 for v in changes.values()):
            raise AssertionError('Exactly960 additional actual updates /1280 total and changed A/B required')
        report['parameterAbsoluteChanges'] = changes; verify_base('afterFit')
        checkpoint = out / 'adapter.safetensors'
        with stage_receipt(report, save, 'checkpoint-exact-reload', 120):
            item = load_feature(baseline['extraction'][0], prior)
            before = float(feature_loss(bb, adapter, item).item()); before_parameters = parameter_receipts()
            mx.save_safetensors(str(checkpoint), leaves()); load_adapter(checkpoint)
            after = float(feature_loss(bb, adapter, item).item()); after_parameters = parameter_receipts()
            if before != after or before_parameters != after_parameters:
                raise AssertionError('Exact A/B dtype/shape/bytes or original feature loss changed on reload')
            report['checkpoint'] = {'sha256': digest(checkpoint), 'lossBeforeReload': before, 'lossAfterReload': after,
                'parametersBeforeReload': before_parameters, 'parametersAfterReload': after_parameters,
                'exact': True, 'totalCompletedUpdates': 1280, 'continuationActualUpdates': 960}
            del item
        root_proof('after'); archive_native_parity('after'); verify_base('afterReplay')
        adapter.freeze(); model.eval()
        for mode in MODES:
            with stage_receipt(report, save, 'matched-width6-controls-' + mode, CONFIG['controlTimeoutSeconds']):
                load_adapter(original.PRIOR if mode == 'failed-daypack' else checkpoint)
                with disabled(model, adapter) if mode == 'frozen' else nullcontext():
                    report['controls'][mode] = original.native_controls(bb, packet['eval']['cases'][0]['bound'], out / ('controls-' + mode))
                gc.collect(); mx.clear_cache()
        for case_index, case in enumerate(packet['eval']['cases']):
            for mode in MODES if case_index % 2 == 0 else tuple(reversed(MODES)):
                arm = next(a for a in report['arms'] if (a['caseId'], a['mode']) == (case['caseId'], mode))
                with stage_receipt(report, save, 'evaluation-' + case['caseId'] + '-' + mode, CONFIG['requestTimeoutSeconds']):
                    load_adapter(original.PRIOR if mode == 'failed-daypack' else checkpoint)
                    arm.update(status='running', stage='public-metadata'); save()
                    begin = time.perf_counter()
                    try:
                        with disabled(model, adapter) if mode == 'frozen' else nullcontext():
                            result = original.run_case(bb, case['bound'], stage_callback=lambda stage: arm.update(stage=stage))
                        arm.update(result, stage='completed', boundSha256=hashed(case['bound']))
                    except BaseException as exc:
                        arm.update(getattr(exc, 'partial_result', {}), status='failed',
                            error={'type': type(exc).__name__, 'message': str(exc)},
                            rawRows=getattr(exc, 'partial_rows', getattr(exc, 'partial_result', {}).get('rawRows', [])),
                            failedAtMs=(time.perf_counter() - begin) * 1000)
                        raise
                    finally:
                        gc.collect(); mx.clear_cache()
                save('arm-completed')
        if len(report['arms']) != 24 or any(a['status'] != 'completed-attempt' for a in report['arms']):
            raise AssertionError('All24 fixed evaluation replay arms required')
        verify_base('afterEvaluation')
        with stage_receipt(report, save, 'final-source-and-original-preservation', 180):
            if digest(checkpoint) != report['checkpoint']['sha256']:
                raise AssertionError('Continued checkpoint changed during evaluation')
            original.assert_source_fresh(baseline['preflight'], original.DATA)
            for name, sha in frozen['sources'].items():
                if digest(ROOT / name) != sha:
                    raise ValueError('Continuation source changed: ' + name)
            assert_inventory_preserved(frozen['original'])
            report['controls'].update(originalRunAllBytesUnchanged=True, frozenLanguageHashesUnchanged=True,
                sourceFilesUnchanged=True, datasetManifestUnchanged=True)
        report.update(status='completed', phase='completed', memory=bb.runtime_memory()); save('completed')
    except BaseException as exc:
        for row in report['steps']:
            if row['status'] != 'completed':
                row.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        for arm in report['arms']:
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
    parser.add_argument('--original-run', type=Path, default=ORIGINAL)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--native-owner', choices=['root'])
    parser.add_argument('--confirm-serialized', action='store_true')
    parser.add_argument('--reviewed-preflight-sha256')
    args = parser.parse_args()
    if args.preflight:
        if args.out is not None or args.preflight_report.exists():
            raise ValueError('Preflight writes one new report; preserve prior evidence')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        result = preflight(tokenizer, args.original_run)
        import sys
        if any(name == 'mlx' or name.startswith('mlx.') for name in sys.modules):
            raise AssertionError('Tokenizer preflight must not import MLX')
        dump(args.preflight_report, result)
        print(json.dumps({'status': result['status'], 'additionalUpdates': CONFIG['updates'],
            'finalTotalUpdates': CONFIG['finalTotalUpdates'], 'permutationSha256': result['permutationSha256'],
            'originalFileCount': result['original']['fileCount']}), flush=True)
        return
    if args.out is None or args.native_owner != 'root' or not args.confirm_serialized or digest(args.preflight_report) != args.reviewed_preflight_sha256:
        raise ValueError('Native phase requires root owner, serialization confirmation and exact reviewed preflight SHA')
    from .catalog_fusion.generalization_v2 import native_owner_lock
    with native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'), args.native_owner):
        run(args.out, args.preflight_report, args.original_run)


if __name__ == '__main__':
    main()
