"""Isolated paired V4 ordinary/prefix worker; no live source/process mutation."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading

from . import store_persona_adapter_engine_v4 as previous
from . import store_persona_adapter_engine_v2 as transport
from . import store_persona_adapter_contract_v2 as contract
from .store_persona_query_prefix_runtime_v1 import AdapterPersonaRuntime, MODES
from . import store_query_prefix_reuse_v1 as reuse

VERSION = 'persona-query-prefix-engine.v1'
CONTROL_PATH = '/measurement/query-encoding'


class ModeController:
    def __init__(self):
        self.lock = threading.Lock(); self.mode = 'ordinary'; self.active = False

    def set(self, mode):
        with self.lock:
            if mode not in MODES: raise ValueError('Explicit ordinary or prefix mode required')
            if self.active: raise RuntimeError('A model request is active; mode cannot change')
            self.mode = mode
            return {'mode': self.mode, 'active': False}

    @contextmanager
    def admit(self):
        with self.lock:
            if self.active: raise RuntimeError('Serial model request ownership required')
            self.active = True; mode = self.mode
        try: yield mode
        finally:
            with self.lock: self.active = False


def freeze_query_assets(engine):
    model = Path(engine.session.packet['model']['modelPath'])
    source = {Path(__file__), Path(reuse.__file__), Path(__file__).with_name('store_persona_query_prefix_runtime_v1.py')}
    source.update(Path(p) for p in engine.adapter_runtime.pins)
    source.update(Path(p) for p in engine.adapter_engine_sources)
    source.update(Path(p) for p in reuse.installed_source_receipt()['files'])
    source.add(Path(__file__).with_name('backbone.py'))
    source.add(Path(__file__).with_name('store_function_adaptation.py'))
    features = {path for row in engine.session.catalog_rows for path in engine.session.store.paths(row['role'], row['member'])}
    groups = {'model': [model / name for name in ('model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors',
            'config.json', 'model.safetensors.index.json')],
        'tokenizer': [model / name for name in ('tokenizer.json', 'tokenizer_config.json')],
        'source': sorted(source), 'catalogFeatures': sorted(features),
        'head': [engine.session.head_path, engine.session.head_export_path,
            engine.adapter_runtime.query_intent.path, engine.adapter_runtime.adapter_path,
            engine.adapter_runtime.color.head_path, engine.adapter_runtime.color.cache_path]}
    aliases = {str(Path(path).absolute()): str(Path(path).resolve(strict=True)) for paths in groups.values() for path in paths}
    resolved = {group: sorted({Path(path).resolve(strict=True) for path in paths}) for group, paths in groups.items()}
    binding = reuse.freeze_assets(resolved, frozen927_hashes_sha256=reuse.hashed(engine.before_hashes),
        model_revision=engine.bb.revision)
    binding.update(logicalAssetAliases=aliases, logicalAssetAliasesSha256=reuse.hashed(aliases))
    return binding


def engine_class(*, runtime_factory=AdapterPersonaRuntime, binding_factory=freeze_query_assets,
                 controller=None, **dependencies):
    factory = previous.engine_class(runtime_factory=runtime_factory, **dependencies)
    class PrefixEngine:
        def __init__(self, *, query_prefix_binding=None, **kwargs):
            self.engine = factory(**kwargs); self.controller = controller or ModeController()
            try:
                self.engine.adapter_engine_sources[str(Path(__file__))] = contract.sha(__file__)
                binding = query_prefix_binding if query_prefix_binding is not None else binding_factory(self.engine)
                preparation = self.engine.adapter_runtime.prepare_query_prefix(binding)
                self.engine.startup['persona']['queryPrefixCandidate'] = {
                    'version': VERSION, 'defaultMode': 'ordinary', 'availableModes': list(MODES),
                    'preparation': preparation, 'assetBinding': binding,
                    'fullRequestTextAndTokenIdsPreserved': True, 'dynamicSuffixFreshPerRequest': True,
                    'noQueryFeatureOrScoreOrRankingOrAnswerCache': True,
                    'featureVectorsInRequestEvidenceOnly': True,
                    'sourceSha256': contract.sha(__file__),
                    'scope': 'isolated paired encoder experiment; V4 semantics unchanged, native parity/speed not implied'}
                self.engine._assert_candidate()
                (self.engine.output / 'startup.json').write_text(json.dumps(self.engine.startup, indent=2) + '\n')
            except BaseException as error:
                try: self.engine.close()
                except BaseException as cleanup_error: error.add_note('Prefix initialization cleanup: ' + str(cleanup_error))
                raise

        def __getattr__(self, name): return getattr(self.engine, name)

        def set_query_encoding_mode(self, mode): return self.controller.set(mode)

        def personalize(self, value, request_id, started, emit, cancelled):
            with self.controller.admit() as mode:
                self.engine.adapter_runtime.set_query_encoding_mode(mode)
                return self.engine.personalize(value, request_id, started, emit, cancelled)

        def close(self): return self.engine.close()
    return PrefixEngine


def handler_for(worker, controller):
    class Handler(transport.handler_for(worker)):
        def do_POST(self):
            if self.path != CONTROL_PATH: return super().do_POST()
            if not self.origin_allowed(): return self.respond(403, {'error': 'Origin not allowed'})
            if worker is None or not worker.health()['ready']: return self.respond(503, {'error': 'Worker not ready'})
            if worker.active_cancel is not None or worker.jobs.qsize(): return self.respond(409, {'error': 'Worker active or queued'})
            try:
                if self.headers.get('Transfer-Encoding'): raise ValueError('Bounded Content-Length required')
                size = int(self.headers.get('Content-Length', '-1'))
                if not 1 <= size <= 256: raise ValueError('Small explicit mode control required')
                self.connection.settimeout(5); raw = self.rfile.read(size)
                if len(raw) != size: raise ValueError('Incomplete mode control')
                value = json.loads(raw)
                if type(value) is not dict or set(value) != {'mode'}: raise ValueError('Only mode field allowed')
                result = controller.set(value['mode'])
            except RuntimeError as error: return self.respond(409, {'error': str(error)})
            except (ValueError, TimeoutError, UnicodeError) as error: return self.respond(400, {'error': str(error)})
            self.respond(200, result)
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    for name in ('asset-root', 'output', 'serving-manifest', 'catalog-feature-root', 'color-head', 'color-cache',
                 'query-intent-head', 'candidate-run', 'integration-manifest', 'rank-head-export'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('serving-manifest-sha256', 'query-intent-head-sha256', 'rank-head-export-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--query-intent-source-manifest-sha256')
    args = parser.parse_args(argv); root = Path(__file__).resolve().parents[3]
    if not args.output.is_absolute() or not args.output.resolve().is_relative_to(root) or args.output.exists():
        parser.error('Unique absolute output inside current code root required')
    if not args.asset_root.is_absolute(): parser.error('Explicit absolute original asset root required')
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    os.chdir(args.asset_root)
    from .store_demo_service import ModelWorker
    from .store_rank_session_v5 import make_v5_rank_session_factory
    controller = ModeController(); klass = engine_class(controller=controller)
    server = ThreadingHTTPServer(('127.0.0.1', 8767), handler_for(None, controller))
    worker = ModelWorker(lambda: klass(mode='parallel', serving_manifest=args.serving_manifest,
        serving_sha=args.serving_manifest_sha256, catalog_feature_root=args.catalog_feature_root,
        color_head=args.color_head, color_cache=args.color_cache, query_intent_head=args.query_intent_head,
        query_intent_sha=args.query_intent_head_sha256, query_intent_source_manifest_sha=args.query_intent_source_manifest_sha256,
        candidate_run=args.candidate_run, integration_manifest=args.integration_manifest, output=args.output, width=20,
        rank_session_factory=make_v5_rank_session_factory(args.rank_head_export, args.rank_head_export_sha256)))
    server.RequestHandlerClass = handler_for(worker, controller); server.daemon_threads = True
    def stop(*_): threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    try: server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        if not worker.close(): raise RuntimeError('Prefix worker teardown integrity unconfirmed')


if __name__ == '__main__': main()
