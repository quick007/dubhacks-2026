"""CPU-default compact-copy launcher. Native work requires explicit --run/root.

Run from repository root. Never stops the service or changes native/API sources.
The API owns its own lock/phase alarms; no lock/alarm is nested around its call.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = HERE / 'store_demo_integration_v6/manifest.json'
MANIFEST_SHA = '73efaef8462e4be05dee05f6602a24f4c8ce8ff1328f5f6b0e576d9701ed1047'
PREFLIGHT = HERE / 'store_compact_copy_v1/token-preflight.json'
PREFLIGHT_SHA = '6e8958aa580f3cd94f68a5d21e567ac02c50bd92dafcc56794c4272295d0cd79'
CHECKPOINT_SHA = '56c6c1f4199d8689f1920d871213a46d2d121fec4e5836ff70e33aaf98c7c7b8'
COPY_RUN = ROOT / 'data/ranker/pretrained/actual-store-copy-fit-v2-attempt-1'
RANK_RUN = ROOT / 'data/ranker/pretrained/actual-store-function-v3-attempt-1'
LOCK = ROOT / 'data/ranker/pretrained/native-owner.lock'
LIFECYCLE_SECONDS = 600
PINNED = {
    'store_compact_copy_native_v1.py': 'c0189d56aea18f96251b4465fa983de826492055f6ced458aac6b61e5b9f9509',
    'store_compact_copy_v1.py': 'c3bd6c27116469cc10492ab9039e646ca21b60e6c8e3048c415c9209b94ec00d',
    'store_demo_engine.py': 'a1e52eebc7ea90103d27309b5d960ae4676db28c4e207ccec3ebaa3a3564639c',
    'store_demo_catalog.py': 'f1240656d1b7b07f0776186c2046361a081c6a553587ede4462f8aa2b6b6cb86',
    'store_rank_session.py': 'baa459958338693bdc5bc9654ea5e014f075d4860db3f3cdad69d11e88dfd4a0',
    'store_copy_wide_v2.py': 'a6631ff3b9e930b6c5a74465326fb8eba05aba6a736690c18b577718d98d096c',
    'backbone.py': '551dd6204419b2cee7b6f98cbc8ea239ca371cac6d2cf992bb7838f2d657aedc',
    'semantic_copy_training.py': 'af040956b899bd9fb0158d014fdefe1e10d7647cba026b7ad7e704167f841c9e',
    'store_hybrid_benchmark_v2.py': '3f76e456d54e0a1c7c36f9553df061d3771db744232432106c1574089011c7dd',
    'store_copy_fit_v2.py': '7f63b9ac18f9142910694499b18c1a5427d4b2da13f0dc3be20860fb44b89968',
}
SOURCES = tuple(PINNED) + ('store_compact_copy_run_v1.py', 'test_store_compact_copy_run_v1.py',
    'branch_local_copy.py', 'parallel_decode.py', 'native_batch_diagnosis.py', 'fused_prefix.py',
    'store_copy_data_v2.py', 'store_catalog_v1.py', 'semantic_copy_feature_fit.py', 'catalog_fusion/generalization_v2.py')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)

def error_record(error):
    return {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}

def service_listening():
    try:
        with socket.create_connection(('127.0.0.1', 8766), timeout=.5): return True
    except ConnectionRefusedError: return False
    except OSError as error: raise RuntimeError('Cannot prove8766 stopped; refuse native initialization') from error

def source_inventory():
    result = {name: sha(HERE / name) for name in SOURCES}
    if any(result[name] != expected for name, expected in PINNED.items()):
        raise ValueError('Reviewed compact/engine source pin changed')
    return result

def cpu_preflight(manifest, *, plan='diagnostic8', rank_run=RANK_RUN):
    if Path(manifest).resolve() != MANIFEST or sha(manifest) != MANIFEST_SHA:
        raise ValueError('Explicit exact reviewed integration manifest v6 required')
    if sha(PREFLIGHT) != PREFLIGHT_SHA: raise ValueError('Reviewed compact tokenizer receipt changed')
    sources = source_inventory()
    from .store_compact_copy_native_v1 import experiment_plan
    from . import store_compact_copy_v1 as compact
    from .store_demo_catalog import DemoCatalogBinding
    from .store_hybrid_benchmark_v2 import completed_adapter
    from .semantic_copy_feature_fit import MODEL
    from transformers import AutoTokenizer
    binding = DemoCatalogBinding(manifest)
    fit, checkpoint = completed_adapter(COPY_RUN)
    if sha(checkpoint) != CHECKPOINT_SHA or len(fit['frozenBaseParameterHashes']) != 927:
        raise ValueError('Exact completed320 checkpoint and927 source hashes required')
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    frozen = json.loads(PREFLIGHT.read_text())
    if compact.token_preflight(tokenizer) != frozen: raise ValueError('Actual local tokenizer/source fixture no longer matches reviewed preflight')
    rank_run = Path(rank_run).resolve()
    if any(name == 'mlx' or name.startswith('mlx.') for name in sys.modules):
        raise RuntimeError('CPU preflight imported MLX unexpectedly; no native default is allowed')
    return {'version': 'store-compact-copy-run.v1', 'nativeExecution': False, 'plan': experiment_plan(plan),
        'width': 20, 'memoryGiB': 7, 'nativeUpdates': 0, 'sourceSha256': sources,
        'integrationManifest': str(MANIFEST), 'integrationManifestSha256': binding.manifest_sha,
        'compactPreflight': str(PREFLIGHT), 'compactPreflightSha256': PREFLIGHT_SHA,
        'checkpoint': str(checkpoint.resolve()), 'checkpointSha256': CHECKPOINT_SHA,
        'completedCopyReportSha256': sha(COPY_RUN / 'results.json'), 'expectedFrozen927Hashes': fit['frozenBaseParameterHashes'],
        'rankRun': str(rank_run), 'rankReportSha256': sha(rank_run / 'report.json'),
        'modelFilesFromCompletedFit': fit['preflight']['modelFiles'],
        'largeModelFileVerification': 'Performed afresh by ResidentDemoEngine before native use; no model constructed in CPU preflight',
        'budgetScope': '600s outer monotonic stage checks, nonpreemptive across existing API call; API retains120/240s phase SIGALRM deadlines. Mandatory shutdown executes even after budget exhaustion.'}

@contextmanager
def native_lock():
    from .catalog_fusion.generalization_v2 import native_owner_lock
    with native_owner_lock(LOCK, 'root'): yield

def default_engine_type():
    from .store_demo_engine import ResidentDemoEngine
    return ResidentDemoEngine

def default_runner(*args, **kwargs):
    from .store_compact_copy_native_v1 import run_native_comparison
    return run_native_comparison(*args, **kwargs)

def parameter_hashes(engine):
    from .semantic_copy_training import parameter_hashes as read_hashes
    return {k.replace('.31.mlp.down_proj.linear.', '.31.mlp.down_proj.'): v
        for k, v in read_hashes(engine.bb.model.language_model).items()}

def shutdown_engine(engine, preflight, *, hash_reader=parameter_hashes):
    """Hash and exit even a partially initialized engine, without calling methods
    that assume constructor completion. Every unavailable proof is explicit."""
    receipt = {'status': 'partial-unavailable', 'frozen927HashesUnchanged': None,
        'matchesCompletedFit927Hashes': None, 'tensorsHashed': 0, 'streamReleased': False,
        'copyCheckpointUnchanged': None}
    try:
        receipt['copyCheckpointUnchanged'] = sha(preflight['checkpoint']) == preflight['checkpointSha256']
    except BaseException as error: receipt['checkpointError'] = error_record(error)
    try:
        if hasattr(engine, 'bb'):
            hashes = hash_reader(engine)
            receipt.update(tensorsHashed=len(hashes), observedParameterHashes=hashes,
                matchesCompletedFit927Hashes=len(hashes) == 927 and hashes == preflight['expectedFrozen927Hashes'])
            if hasattr(engine, 'before_hashes'):
                receipt['frozen927HashesUnchanged'] = len(hashes) == 927 and hashes == engine.before_hashes
            receipt['memory'] = engine.bb.runtime_memory()
    except BaseException as error: receipt['hashError'] = error_record(error)
    finally:
        if hasattr(engine, 'stream_context'):
            try:
                try:
                    if hasattr(engine, 'bb'): engine.bb.mx.synchronize()
                finally: engine.stream_context.__exit__(None, None, None)
                receipt['streamReleased'] = True
            except BaseException as error: receipt['streamError'] = error_record(error)
    if (receipt['frozen927HashesUnchanged'] is True and receipt['matchesCompletedFit927Hashes'] is True
            and receipt['copyCheckpointUnchanged'] and receipt['streamReleased']): receipt['status'] = 'verified'
    return receipt

def run_compact(out, *, manifest, run=False, native_owner=None, plan='diagnostic8', rank_run=RANK_RUN, _dependencies=None):
    out = Path(out)
    if out.exists(): raise FileExistsError('Use a new compact wrapper output; preserve all prior attempts')
    out.mkdir(parents=True)
    deps = {} if _dependencies is None else _dependencies
    clock = deps.get('clock', time.perf_counter); started = clock()
    report = {'version': 'store-compact-copy-run.v1', 'status': 'starting', 'nativeExecutionRequested': run,
        'nativeExecution': False, 'nativeOwner': native_owner, 'stage': 'authorization', 'lifecycleBudgetSeconds': LIFECYCLE_SECONDS,
        'budgetEnforcement': 'Outer monotonic stage boundaries, no outer SIGALRM; existing API per-phase deadlines remain unchanged. Shutdown always attempted.',
        'lockPolicy': 'Shared nonblocking lock for initialization, released before API-owned lock, reacquired for shutdown; root must serialize all GPU work across transitions.',
        'startup': {'status': 'not-started'}, 'shutdown': {'status': 'not-started'}}
    engine = None; preflight = None; owns_initialization = False
    def save():
        report['wallSeconds'] = clock() - started; dump(out / 'lifecycle.json', report)
    def check_budget(stage):
        if clock() - started > LIFECYCLE_SECONDS: raise TimeoutError('Compact outer600s monotonic budget exceeded at ' + stage)
    def close():
        if engine is not None and report['shutdown']['status'] == 'not-started':
            report['stage'] = 'shutdown'; save()
            try:
                report['shutdown'] = deps.get('shutdown', shutdown_engine)(engine, preflight)
                dump(out / 'shutdown.json', report['shutdown'])
            except BaseException as error:
                report['shutdown'] = {'status': 'failed', 'error': error_record(error)}; dump(out / 'shutdown.json', report['shutdown'])
    save()
    try:
        if Path.cwd().resolve() != ROOT: raise ValueError('Run compact wrapper from the repository root; legacy model/data paths are relative')
        if run and native_owner != 'root': raise ValueError('Explicit --native-owner root required for native execution')
        if run and deps.get('service_listening', service_listening)(): raise RuntimeError('Demo8766 is listening; refuse native launch, never stop it automatically')
        report['stage'] = 'cpu-preflight'; save()
        preflight = deps.get('preflight', cpu_preflight)(manifest, plan=plan, rank_run=rank_run)
        dump(out / 'preflight.json', preflight); report['preflightSha256'] = sha(out / 'preflight.json')
        if not run:
            report.update(status='cpu-preflight-only', stage='completed'); save(); return report
        check_budget('before-initialization')
        lock = deps.get('native_lock', native_lock)
        with lock():
            owns_initialization = True
            if deps.get('service_listening', service_listening)(): raise RuntimeError('Demo8766 started before initialization; refuse native launch')
            report['stage'] = 'initialization'; report['startup'] = {'status': 'initializing'}; save()
            engine_type = deps.get('engine_type', default_engine_type)()
            engine = engine_type.__new__(engine_type)
            begin = clock()
            try:
                report['nativeExecution'] = True  # Entering explicit native initialization; model availability recorded below.
                engine_type.__init__(engine, rank_run=Path(preflight['rankRun']), copy_run=COPY_RUN,
                    output=out / 'engine', width=20, memory_gib=7, integration_manifest=Path(manifest))
                report['startup'] = {'status': 'completed', 'initializationMs': (clock() - begin) * 1000,
                    'engine': engine.startup, 'beforeTensorCount': len(engine.before_hashes), 'modelConstructed': hasattr(engine, 'bb')}
                if len(engine.before_hashes) != 927 or engine.before_hashes != preflight['expectedFrozen927Hashes']:
                    raise AssertionError('Initialized engine927 hashes differ from completed fit')
                dump(out / 'startup.json', report['startup']); save(); check_budget('after-initialization')
            except BaseException as error:
                report['startup'].update(status='failed', initializationMs=(clock() - begin) * 1000, error=error_record(error),
                    partialBackboneAvailable=hasattr(engine, 'bb'), partialBeforeHashesAvailable=hasattr(engine, 'before_hashes'))
                dump(out / 'startup.json', report['startup']); close(); report['stage'] = 'initialization'; raise
            finally: owns_initialization = False
        # Crucially outside initialization lock: native API acquires its own.
        check_budget('before-native-api'); report['stage'] = 'native-api'; save()
        if deps.get('service_listening', service_listening)(): raise RuntimeError('Demo8766 started before API; refuse comparison')
        result = deps.get('runner', default_runner)(engine.bb, engine.adapter, Path(preflight['checkpoint']), CHECKPOINT_SHA,
            out / 'native', PREFLIGHT, native_owner='root', confirm_serialized=True,
            reviewed_preflight_sha256=PREFLIGHT_SHA, plan=plan)
        report['nativeResult'] = {'status': result.get('status'), 'path': str(out / 'native/results.json'), 'summary': result.get('summary')}
        check_budget('after-native-api')
        if result.get('status') != 'completed': raise RuntimeError('Compact API did not complete; inspect preserved native results')
        report.update(status='completed', stage='preservation')
    except BaseException as error:
        report.update(status='failed', error=error_record(error), failedStage=report['stage'])
        if report['startup']['status'] == 'initializing': report['startup'].update(status='failed', error=report['error'])
    finally:
        if engine is not None and report['shutdown']['status'] == 'not-started':
            if owns_initialization: close()
            else:
                try:
                    with deps.get('native_lock', native_lock)(): close()
                except BaseException as error: report['shutdown'] = {'status': 'lock-or-cleanup-failed', 'error': error_record(error)}
        if preflight is not None:
            try:
                fresh = deps.get('source_inventory', source_inventory)()
                report['sourceFilesUnchanged'] = fresh == preflight['sourceSha256']
                if not report['sourceFilesUnchanged']: raise AssertionError('Wrapper/native source inventory changed during lifecycle')
                if _dependencies is None:
                    report['manifestAndPreflightUnchanged'] = sha(MANIFEST) == MANIFEST_SHA and sha(PREFLIGHT) == PREFLIGHT_SHA
                    if not report['manifestAndPreflightUnchanged']: raise AssertionError('Reviewed manifest or compact preflight changed during lifecycle')
            except BaseException as error: report.update(status='failed', preservationError=error_record(error))
        if run and engine is not None and report['shutdown'].get('status') != 'verified': report['status'] = 'failed'
        if run and clock() - started > LIFECYCLE_SECONDS:
            report.update(status='failed', outerBudgetExceeded=True)
        dump(out / 'startup.json', report['startup']); dump(out / 'shutdown.json', report['shutdown'])
        report['stage'] = 'finished'; save()
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True, help='Explicit reviewed store_demo_integration_v6/manifest.json')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--native-owner', choices=['root'])
    parser.add_argument('--plan', choices=['diagnostic8', 'full32'], default='diagnostic8')
    parser.add_argument('--rank-run', type=Path, default=RANK_RUN)
    args = parser.parse_args(argv)
    result = run_compact(args.out, manifest=args.manifest, run=args.run, native_owner=args.native_owner, plan=args.plan, rank_run=args.rank_run)
    print(json.dumps({'status': result['status'], 'nativeExecution': result['nativeExecution'], 'output': str(args.out)}))
    return 0 if result['status'] in ('completed', 'cpu-preflight-only') else 1

if __name__ == '__main__': raise SystemExit(main())
