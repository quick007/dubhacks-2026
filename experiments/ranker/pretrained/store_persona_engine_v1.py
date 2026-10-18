"""Isolated persona candidate CLI. No model loads without explicit main entry.

Start with worktree on sys.path; chdir --asset-root before loader imports so
legacy relative source assets and the single native-owner.lock stay on main.
"""
from pathlib import Path
import argparse
from copy import deepcopy
import json
import os
import signal
import threading
import time
from .store_persona_context_v1 import checked_context_request, sha
from .store_demo_engine import RequestCancelled
VERSION = 'store-persona-engine.v1'
PERSONA_TIMING_SCOPE = ('request admission through public-observation validation, exact cached or fresh frozen-base query/context feature acquisition '
    '(per-request cache-hit flags and native forward counts), fresh learned ranking and real color/variant binding, fresh neural copy inference, '
    'source-fact validation and serialized server events; excludes startup, network, browser paint and image fetch/decode')
PERSONA_IMAGE_POLICY = ('learned public-color variant selection plus canonical existing gallery binding; '
    'no pixel or vision-feature claim')


def feature_forward_receipt(ranked):
    query = ranked.get('queryFeature', {})
    context = ranked.get('contextFeature', {})
    q = query.get('nativeBackboneForwardCalls')
    c = context.get('nativeBackboneForwardCalls', context.get('nativeCalls'))
    # Missing instrumentation remains unknown instead of fabricated zero.
    return {'queryNativeForwards': q, 'contextNativeForwards': c,
        'totalNativeForwards': q+c if type(q) is int and type(c) is int else None,
        'queryFeatureCacheHit': query.get('cacheHit'), 'contextFeatureCacheHit': context.get('cacheHit'),
        'scope': 'feature acquisition only; learned rank/variant scores and neural copy are recomputed for every request'}



def engine_class():
    from .store_candidate_image_service_v1 import ImageCandidateDemoEngine, cleanup_failed_start
    from .store_candidate_engine_v1 import parallel_rows, parallel_payload, source_for_cache
    from .store_demo_engine import NEUTRAL_COPY_REQUEST, public_sections
    from .store_catalog_v1 import canonical
    from .store_persona_runtime_v1 import PersonaRuntime
    class PersonaEngine(ImageCandidateDemoEngine):
        def __init__(self, *, persona_head, persona_head_sha256, persona_feature_root, **kwargs):
            self.persona_engine_sources = {str(Path(__file__).resolve()): sha(__file__)}
            self.persona_runtime = PersonaRuntime(persona_head, persona_head_sha256, persona_feature_root)
            base_ready = False
            try:
                super().__init__(**kwargs)
                base_ready = True
                self.startup['persona'] = {'checkpointSha256': persona_head_sha256,
                    'queryFeatureCache': 'one exact canonical frozen-base public query feature reused on an exact hit; per-request provenance reports cache hits and actual native feature forwards',
                    'featureCaches': {'queryEntriesAtStartup': 0,
                        'precomputedPublicContextEntriesAtStartup': len(getattr(self.persona_runtime, 'context_cache', {}).get('features', {})),
                        'startupFeatureEncodingNativeForwards': 0,
                        'contextCacheSourceExtractionReportSha256': getattr(self.persona_runtime, 'context_cache', {}).get('sourceExtractionReportSha256'),
                        'runtimeCounts': 'provenance.featureEncoding plus personaRanking.queryFeature/contextFeature',
                        'freshPerRequest': 'public-observation validation, learned rank/variant scoring, source-conditioned neural copy'},
                    'timingScope': PERSONA_TIMING_SCOPE, 'imagePolicy': PERSONA_IMAGE_POLICY,
                    'scope': 'authored synthetic context candidate; caller selects the explicit checkpoint; native query guards retained',
                    'engineSourcesSha256': deepcopy(self.persona_engine_sources)}
                self._assert_candidate()
                (self.output / 'startup.json').write_text(json.dumps(self.startup, indent=2)+'\n')
            except BaseException as error:
                # The base image engine already tears down its own failed init.
                # After it returns, this wrapper owns any subsequent failure.
                if base_ready:
                    try:
                        cleanup_failed_start(self, error)
                    except BaseException as cleanup_error:
                        error.add_note('Persona post-constructor cleanup failed: ' + str(cleanup_error))
                raise
        def _assert_candidate(self):
            super()._assert_candidate()
            if any(sha(path) != expected for path, expected in self.persona_engine_sources.items()):
                raise ValueError('Persona engine/handler source changed since startup')
            self.persona_runtime.assert_current()
        def close(self):
            # CandidateDemoEngine.close invokes this wrapper's _assert_candidate
            # before teardown and guarantees native cleanup in its finally block.
            result = super().close()
            result['personaEngineSourcesUnchanged'] = True
            (self.output / 'shutdown.json').write_text(json.dumps(result, indent=2)+'\n')
            return result
        def personalize(self, value, request_id, started, emit, cancelled):
            from .store_copy_wide_v2 import guard_memory
            request, policy, context = checked_context_request(value, self.persona_runtime.catalog['products'])
            directory = self.output / 'requests' / request_id; directory.mkdir(parents=True, exist_ok=False)
            receipt = {'version': VERSION, 'requestId': request_id, 'input': deepcopy(value),
                'effectiveRequest': request, 'policy': policy, 'status': 'running', 'mode': self.mode,
                'copyConditioning': deepcopy(NEUTRAL_COPY_REQUEST), 'copyShopperQueryUsed': False,
                'copyCandidate': self.copy_metadata, 'copyInputDisclosure': 'fresh neural word head uses selected public name/category/itemPrice/color and fixed plain style; public query/persona/history affect learned ranking and real color bindings; copy strings and rank results are not response-cached', 'events': [], 'rawRows': []}
            def save():
                path = directory / 'receipt.tmp'; path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
                path.replace(directory / 'receipt.json')
            def event(kind, **body):
                if cancelled():
                    raise RequestCancelled('Cancelled before candidate event serialization')
                self._assert_candidate()
                result = {'type': kind, 'requestId': request_id, **body}
                canonical(result)
                result.setdefault('timing', {})['readyMs'] = (time.perf_counter() - started) * 1000
                if kind in ('firstThree', 'complete'):
                    result['timing']['first3readyMs' if kind == 'firstThree' else 'fullReadyMs'] = result['timing']['readyMs']
                canonical(result); receipt['events'].append(deepcopy(result)); emit(result)
            save()
            try:
                self._assert_candidate()
                if cancelled():
                    raise RequestCancelled('Cancelled before native ranking admission')
                request, policy, order, guards, cards, ranked = self.persona_runtime.rank(value, self.session, self.adapter)
                rank_ms = (time.perf_counter() - started) * 1000
                provenance = {'ranking': 'fixed v5 semantic query plus learned public persona context; exact public query/context feature caching disclosed per request'
                    , 'personaRanking': ranked,
                    'rankCheckpointSha256': self.session.head_sha, 'nativeCopyCheckpointSha256': self.copy_sha,
                    'copy': self.copy_metadata, 'images': PERSONA_IMAGE_POLICY, 'hiddenProfileInputs': False,
                    'featureEncoding': feature_forward_receipt(ranked),
                    'timingScope': PERSONA_TIMING_SCOPE, 'candidateMode': self.mode,
                    'qualityScope': 'TRAIN-ALL catalog demo; no heldout generalization claim' if self.mode == 'parallel' else 'verified original native field validation; cache hits generated earlier'}
                if hasattr(self.session, 'startup_provenance'):
                    provenance['rankSession'] = deepcopy(self.session.startup_provenance)
                receipt.update(ranking=ranked, finalOrder=order, guards=guards, cards=cards, provenance=provenance)
                metadata, _ = public_sections(cards, [], self.session.products)
                shared = {'eligibleCount': len(order), 'shortfall': max(0, 6 - len(cards)), 'provenance': provenance}
                event('rank', productIds=[self.session.snapshot['externalMapping']['products'][i]['productId'] for i in order],
                    selectedProductIds=[c['action']['productId'] for c in cards],
                    variants={c['action']['productId']: c['action']['variantId'] for c in cards},
                    sections=metadata, timing={'rankMs': rank_ms}, **shared)
                save()
                if cancelled():
                    raise RequestCancelled('Cancelled after ranking')
                if cards and self.mode == 'parallel':
                    from . import store_parallel_text_head_v1 as model
                    rows = parallel_rows(cards)
                    with model.bounded_cpu_threads(1):
                        generated = model.generate(self.head, model.inference_rows(rows), self.feature_map)
                    receipt.update(runtimeRows=rows, parallelGeneration=generated)
                    source = {**self.copy_metadata, 'kind': 'fresh-parallel-head', 'generatedEarlier': False, 'freshPerRequestGeneration': True,
                        'generation': 'fresh independent word-slot predictions from cached frozen public fact features',
                        'autoregressiveSteps': 0, 'feedforwardCalls': generated['feedforwardCalls']}
                    payload = lambda limit: parallel_payload(cards, rows, generated, limit)
                elif cards:
                    resolved = self.copy_cache.resolve(cards, request_started=started,
                        native_generate=lambda selected: self._native_cache_miss(selected, directory / 'native-miss', cancelled))
                    receipt.update(copyCache=resolved['cache'], rawRows=resolved['rawRows'], validation=resolved['validation'])
                    source = source_for_cache(resolved['cache'])
                    payload = lambda limit: public_sections(cards, resolved['rawRows'], self.session.products, limit)
                else:
                    source = {'kind': 'no-generation',
                        'generatedEarlier': False, 'freshPerRequestGeneration': False, 'noEligibleProducts': True}
                    payload = lambda limit: public_sections([], [], self.session.products, 0)
                # All parallel slots / cache resolution finish before either endpoint;
                # never claim earlier first3 readiness than actual validation+JSON.
                for kind, limit in ([('firstThree', min(3, len(cards))), ('complete', len(cards))] if cards else [('complete', 0)]):
                    if cancelled():
                        raise RequestCancelled('Cancelled before candidate payload validation')
                    sections, validation = payload(limit)
                    self.binding.assert_current_sources(self.session.snapshot['receipt'])
                    event(kind, sections=sections, validation=validation, copySource=source,
                        **{key: validation[key] for key in ('acceptedFields', 'withheldFields', 'requiredFields')},
                        timing={'rankMs': rank_ms, 'copyMs': (time.perf_counter() - started) * 1000 - rank_ms}, **shared)
                self._assert_candidate()
                receipt.update(status='completed', memory=guard_memory(self.bb)); save()
            except BaseException as error:
                receipt.update(status='cancelled' if isinstance(error, RequestCancelled) else 'failed',
                    error={'type': type(error).__name__, 'message': str(error)},
                    partialRows=getattr(error, 'partial_rows', []), failedAtMs=(time.perf_counter() - started) * 1000)
                save(); raise
    return PersonaEngine

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import queue
import select
import socket
import uuid
ALLOWED_ORIGINS = frozenset({'http://localhost:4313', 'http://127.0.0.1:4313'})
MAX_REQUEST_BYTES = 32768
def persona_handler_for(worker):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):
            # No shopper query or generated content is printed to terminal.
            print(json.dumps({'http': fmt % args}), flush=True)

        def origin_allowed(self):
            return self.headers.get('Origin') in (None, *ALLOWED_ORIGINS)

        def cors(self):
            origin = self.headers.get('Origin')
            if origin in ALLOWED_ORIGINS:
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Access-Control-Allow-Private-Network', 'true')

        def respond(self, status, value):
            encoded = (json.dumps(value, allow_nan=False) + '\n').encode()
            self.send_response(status); self.cors()
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(encoded))); self.send_header('Cache-Control', 'no-store')
            self.end_headers(); self.wfile.write(encoded)

        def do_OPTIONS(self):
            if not self.origin_allowed():
                return self.respond(403, {'error': 'Origin is not an allowed local storefront'})
            self.send_response(204); self.cors(); self.send_header('Content-Length', '0'); self.end_headers()

        def do_GET(self):
            if not self.origin_allowed():
                return self.respond(403, {'error': 'Origin is not an allowed local storefront'})
            if self.path != '/health':
                return self.respond(404, {'error': 'Unknown endpoint'})
            health = worker.health()
            self.respond(200 if health['ready'] else 503, health)

        def disconnected(self):
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                if readable:
                    return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b''
            except (ConnectionError, OSError):
                return True
            return False

        def do_POST(self):
            started = time.perf_counter()
            if not self.origin_allowed():
                return self.respond(403, {'error': 'Origin is not an allowed local storefront'})
            if self.path != '/personalize':
                return self.respond(404, {'error': 'Unknown endpoint'})
            request_id = str(uuid.uuid4())
            if self.headers.get('Transfer-Encoding'):
                return self.respond(400, {'error': 'Bounded Content-Length request required'})
            try:
                size = int(self.headers.get('Content-Length', '-1'))
                if not 1 <= size <= MAX_REQUEST_BYTES:
                    return self.respond(413, {'error': 'Request must be 1–32768 bytes'})
                self.connection.settimeout(10)
                data = self.rfile.read(size)
                if len(data) != size:
                    raise ValueError('Incomplete request body')
                value = json.loads(data)
                checked_context_request(value)
            except (ValueError, UnicodeError, TimeoutError) as error:
                return self.respond(400, {'error': str(error)})
            try:
                events, cancel, future = worker.submit(value, request_id, started)
            except queue.Full:
                return self.respond(429, {'error': 'Local model queue is full'})
            except RuntimeError as error:
                return self.respond(503, {'error': str(error), 'health': worker.health()})
            self.connection.settimeout(10)
            self.send_response(200); self.cors()
            self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control', 'no-store'); self.send_header('Connection', 'close')
            self.send_header('X-Request-Id', request_id); self.end_headers()
            self.close_connection = True
            try:
                while True:
                    if self.disconnected():
                        cancel.set(); break
                    try:
                        event = events.get(timeout=.2)
                    except queue.Empty:
                        continue
                    if event is None:
                        error = future.exception()
                        if error is not None and not isinstance(error, RequestCancelled):
                            event = {'type': 'error', 'requestId': request_id,
                                'message': f'{type(error).__name__}: {error}'}
                            self.wfile.write((json.dumps(event) + '\n').encode()); self.wfile.flush()
                        break
                    self.wfile.write((json.dumps(event, allow_nan=False) + '\n').encode()); self.wfile.flush()
            except (BrokenPipeError, ConnectionError, OSError, TimeoutError):
                cancel.set()
            finally:
                cancel.set()
    return Handler

def main(argv=None):
    p=argparse.ArgumentParser(__doc__)
    for key in ('asset-root','output','persona-head','persona-feature-root','candidate-run','integration-manifest','rank-head-export'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--persona-head-sha256',required=True);p.add_argument('--rank-head-export-sha256',required=True)
    p.add_argument('--port',type=int,default=8767)
    args=p.parse_args(argv)
    worktree=Path(__file__).resolve().parents[3]
    if args.port != 8767: p.error('Isolated candidate requires8767; main8766 must remain unchanged')
    if not args.output.is_absolute() or not args.output.resolve().is_relative_to(worktree) or args.output.exists(): p.error('Unique absolute worktree output required')
    if not args.asset_root.is_absolute(): p.error('Explicit absolute original asset root required')
    # Resolve all candidate paths before intentionally changing relative asset CWD.
    for key in ('persona_head','persona_feature_root','candidate_run','integration_manifest','rank_head_export'): setattr(args,key,getattr(args,key).resolve())
    os.chdir(args.asset_root)
    from .store_demo_service import ModelWorker
    from .store_rank_session_v5 import make_v5_rank_session_factory
    klass=engine_class()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),persona_handler_for(None))
    worker=ModelWorker(lambda:klass(mode='parallel',candidate_run=args.candidate_run,
        output=args.output,persona_head=args.persona_head,persona_head_sha256=args.persona_head_sha256,
        persona_feature_root=args.persona_feature_root,integration_manifest=args.integration_manifest,width=20,
        rank_session_factory=make_v5_rank_session_factory(args.rank_head_export,args.rank_head_export_sha256)))
    server.RequestHandlerClass=persona_handler_for(worker);server.daemon_threads=True
    def stop(_sig,_frame): threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print(json.dumps({'event':'persona-candidate-listening','port':args.port}),flush=True)
    try: server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        if not worker.close(): raise RuntimeError('Native shutdown integrity unconfirmed')
if __name__=='__main__':main()
