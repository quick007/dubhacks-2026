"""Isolated pragmatic V4 query/projection/cache experiment; defaults unchanged."""
import argparse
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading

from . import store_persona_query_prefix_engine_v1 as previous
from . import store_persona_adapter_contract_v2 as contract
from . import store_persona_portless_http_v1 as local_http
from .store_persona_query_projection_runtime_v1 import AdapterPersonaRuntime, PROJECTION_SOURCE_SHA

VERSION = 'persona-query-selected-projection-engine.v1'
ROUTE_CONTROL = '/measurement/route-projection'
CACHE_CONTROL = '/measurement/query-feature-cache'
PORTLESS_ORIGIN = 'http://shop.dubhacks.localhost:1356'


class Controller(previous.ModeController):
    def __init__(self, modes, initial):
        super().__init__(); self.modes = tuple(modes); self.mode = initial
    def set(self, mode):
        with self.lock:
            if mode not in self.modes: raise ValueError('Mode must be one of ' + ','.join(self.modes))
            if self.active: raise RuntimeError('A model request is active; controls cannot change')
            self.mode = mode; return {'mode': mode, 'active': False}


def engine_class(*, runtime_factory=AdapterPersonaRuntime, query_controller=None,
                 route_controller=None, cache_controller=None, **dependencies):
    query_controller = query_controller or Controller(('ordinary', 'compact'), 'ordinary')
    route_controller = route_controller or Controller(('ordinary', 'selected'), 'ordinary')
    cache_controller = cache_controller or Controller(('off', 'lru32'), 'off')
    factory = previous.engine_class(runtime_factory=runtime_factory, controller=query_controller, **dependencies)
    class Engine:
        def __init__(self, **kwargs):
            self.engine = factory(**kwargs)
            try:
                self.engine.adapter_engine_sources[str(Path(__file__))] = contract.sha(__file__)
                self.engine.adapter_engine_sources[str(Path(local_http.__file__))] = contract.sha(local_http.__file__)
                preparation = self.engine.adapter_runtime.prepare_route_projection()
                query_prewarm = self.engine.adapter_runtime.prewarm_default_query_features()
                self.engine.startup['persona']['routeProjectionCandidate'] = {
                    'version': VERSION, 'defaultMode': 'ordinary', 'availableModes': ['ordinary', 'selected'],
                    'queryEncodingModes': ['ordinary', 'compact'], 'queryFeatureCacheModes': ['off', 'lru32'],
                    'defaultQueryEncoding': 'ordinary', 'defaultQueryFeatureCache': 'off',
                    'failedPrefixBranchDisabled': True, 'unusedPrefixPreparationRetainedForMemoryAccounting': True,
                    'projectionSourceSha256': PROJECTION_SOURCE_SHA, 'preparation': preparation,
                    'queryFeaturePreencoding': query_prewarm,
                    'queryCacheStoresFeaturesOnly': True, 'scoreRankingAnswerCache': False,
                    'compactQualityChangeUnvalidated': True, 'nativeProjectionExactnessOrSpeedNotImplied': True,
                    'portlessHTTP': {'version': local_http.VERSION, 'sourceSha256': contract.sha(local_http.__file__),
                        'allowedOrigins': sorted(local_http.ALLOWED_ORIGINS)}}
                self.engine._assert_candidate()
                (self.engine.output / 'startup.json').write_text(json.dumps(self.engine.startup, indent=2) + '\n')
            except BaseException as error:
                try: self.engine.close()
                except BaseException as cleanup_error: error.add_note('Projection initialization cleanup: ' + str(cleanup_error))
                raise
        def __getattr__(self, name): return getattr(self.engine, name)
        def set_query_encoding_mode(self, mode): return query_controller.set(mode)
        def set_route_projection_mode(self, mode): return route_controller.set(mode)
        def set_route_scoring_mode(self, mode): return route_controller.set(mode)
        def set_query_feature_cache_mode(self, mode): return cache_controller.set(mode)
        def personalize(self, value, request_id, started, emit, cancelled):
            with route_controller.admit() as projection_mode, cache_controller.admit() as cache_mode:
                self.engine.adapter_runtime.set_route_projection_mode(projection_mode)
                self.engine.adapter_runtime.set_query_feature_cache_mode(cache_mode)
                return self.engine.personalize(value, request_id, started, emit, cancelled)
        def close(self): return self.engine.close()
    return Engine


def handler_for(worker, query_controller, route_controller, cache_controller):
    class Handler(previous.handler_for(worker, query_controller)):
        def origin_allowed(self):
            return self.headers.get('Origin') == PORTLESS_ORIGIN or super().origin_allowed()
        def cors(self):
            if self.headers.get('Origin') != PORTLESS_ORIGIN: return super().cors()
            self.send_header('Access-Control-Allow-Origin', PORTLESS_ORIGIN); self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Access-Control-Allow-Private-Network', 'true')
        def do_GET(self):
            controls = {previous.CONTROL_PATH: query_controller, ROUTE_CONTROL: route_controller, CACHE_CONTROL: cache_controller}
            control = controls.get(self.path)
            if control is None: return super().do_GET()
            if not self.origin_allowed(): return self.respond(403, {'error': 'Origin not allowed'})
            with control.lock:
                result = {'mode': control.mode, 'active': control.active, 'availableModes': list(control.modes)}
            self.respond(200 if worker is not None and worker.health()['ready'] else 503, result)
        def do_POST(self):
            control = {ROUTE_CONTROL: route_controller, CACHE_CONTROL: cache_controller}.get(self.path)
            if control is None: return super().do_POST()
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
                result = control.set(value['mode'])
            except RuntimeError as error: return self.respond(409, {'error': str(error)})
            except (ValueError, TimeoutError, UnicodeError) as error: return self.respond(400, {'error': str(error)})
            self.respond(200, result)
    return local_http.handler_for(worker, lambda _: Handler)


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
    query = Controller(('ordinary', 'compact'), 'ordinary'); route = Controller(('ordinary', 'selected'), 'ordinary')
    cache = Controller(('off', 'lru32'), 'off')
    klass = engine_class(query_controller=query, route_controller=route, cache_controller=cache)
    server = ThreadingHTTPServer(('127.0.0.1', 8767), handler_for(None, query, route, cache))
    worker = ModelWorker(lambda: klass(mode='parallel', serving_manifest=args.serving_manifest,
        serving_sha=args.serving_manifest_sha256, catalog_feature_root=args.catalog_feature_root,
        color_head=args.color_head, color_cache=args.color_cache, query_intent_head=args.query_intent_head,
        query_intent_sha=args.query_intent_head_sha256, query_intent_source_manifest_sha=args.query_intent_source_manifest_sha256,
        candidate_run=args.candidate_run, integration_manifest=args.integration_manifest, output=args.output, width=20,
        rank_session_factory=make_v5_rank_session_factory(args.rank_head_export, args.rank_head_export_sha256)))
    server.RequestHandlerClass = handler_for(worker, query, route, cache); server.daemon_threads = True
    def stop(*_): threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    try: server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        if not worker.close(): raise RuntimeError('Projection worker teardown integrity unconfirmed')


if __name__ == '__main__': main()
