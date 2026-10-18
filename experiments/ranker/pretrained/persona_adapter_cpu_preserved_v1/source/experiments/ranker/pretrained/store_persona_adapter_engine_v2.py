"""New isolated adapter-rank engine; existing persona/native entries unchanged.

No default profile, service, model or browser is activated by import. The caller
must select all reviewed artifacts and own the original global native lock.
"""
import argparse
from copy import deepcopy
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import signal
import threading
import time
import uuid

from . import store_persona_adapter_contract_v2 as contract
from .store_demo_engine import RequestCancelled

VERSION = 'persona-adapter-engine.v2'
TIMING_SCOPE = ('admission through public input checks, exact cached or fresh base/query/decision features, '
    'fresh LoRA LM-route decision, learned query-intent blend, learned source-bound variant selection, '
    'fresh neural word slots, factual gates and serialized events; excludes startup/network/browser/images')
IMAGE_POLICY = 'learned within-product color branch, actual source variant/gallery metadata; no pixel or learned image claim'


def engine_class(*, base_cls=None, runtime_factory=None, intent_factory=None, cleanup=None):
    from .store_candidate_engine_v1 import parallel_rows, parallel_payload
    from .store_demo_engine import NEUTRAL_COPY_REQUEST, public_sections
    from .store_catalog_v1 import canonical
    if base_cls is None or cleanup is None:
        from .store_candidate_image_service_v1 import ImageCandidateDemoEngine, cleanup_failed_start
        base_cls = base_cls or ImageCandidateDemoEngine; cleanup = cleanup or cleanup_failed_start
    if runtime_factory is None or intent_factory is None:
        from .store_persona_adapter_runtime_v2 import AdapterPersonaRuntime, VerifiedQueryIntent
        runtime_factory = runtime_factory or AdapterPersonaRuntime; intent_factory = intent_factory or VerifiedQueryIntent

    class AdapterEngine(base_cls):
        def __init__(self, *, serving_manifest, serving_sha, catalog_feature_root, color_head, color_cache,
                     query_intent_head, query_intent_sha, query_intent_source_manifest_sha=None, **kwargs):
            if kwargs.get('mode', 'parallel') != 'parallel':
                raise ValueError('This explicit adapter candidate uses fresh parallel copy; legacy native entry remains separate')
            kwargs['mode'] = 'parallel'
            self.output = Path(kwargs['output']); self.output.mkdir(parents=True, exist_ok=True)
            self.adapter_engine_sources = {str(Path(__file__)): contract.sha(__file__)}
            self.adapter_runtime = None; base_started = False; base_ready = False
            try:
                intent = intent_factory(query_intent_head, query_intent_sha, source_manifest_sha=query_intent_source_manifest_sha)
                self.adapter_runtime = runtime_factory(serving_manifest, serving_sha, catalog_feature_root,
                    color_head=color_head, color_cache=color_cache, query_intent=intent)
                base_started = True; super().__init__(**kwargs); base_ready = True
                self.adapter_runtime.attach(self.bb, self.adapter)
                self.startup['persona'] = {'kind': VERSION,
                    'checkpointSha256': self.adapter_runtime.manifest['adapter']['sha256'],
                    'servingManifestSha256': serving_sha, 'productRanking': 'fresh trained LoRA LM route logits plus learned query-intent blend',
                    'queryIntent': intent.provenance(), 'color': {'checkpointSha256': self.adapter_runtime.color.pins[str(Path(color_head))],
                        'failedOldProductOrderingUsed': False, 'scope': 'same learned within-product scores only'},
                    'publicDecisionFeatureCacheEntries': 20, 'privateTargetOrTeacherInputs': False,
                    'freshPerRequest': 'LM route scores, query-intent score, learned variant selection and neural copy words',
                    'featureEncoding': 'cache-hit flags and actual forward counts in each receipt',
                    'adapterOwnership': 'shared pristine base, mutually exclusive copy/rank modules, copy restored in finally',
                    'timingScope': TIMING_SCOPE, 'scope': 'explicit candidate; combined acceptance and live parity not implied'}
                self._assert_candidate()
                (self.output / 'startup.json').write_text(json.dumps(self.startup, indent=2) + '\n')
            except BaseException as error:
                if base_ready or not base_started:
                    try: cleanup(self, error)
                    except BaseException as cleanup_error: error.add_note('Adapter initialization cleanup: ' + str(cleanup_error))
                raise

        def _assert_candidate(self):
            super()._assert_candidate()
            if any(contract.sha(path) != checksum for path, checksum in self.adapter_engine_sources.items()):
                raise ValueError('Adapter engine source changed since startup')
            if self.adapter_runtime is not None: self.adapter_runtime.assert_current()

        def close(self):
            primary = None
            try:
                if self.adapter_runtime is not None: self.adapter_runtime.assert_current(full=True)
            except BaseException as error: primary = error
            try: result = super().close()
            except BaseException as error:
                if primary is not None: primary.add_note('Base shutdown also reported: ' + str(error)); raise primary
                raise
            if primary is not None: raise primary
            result.update(rankAdapterRestoredToCopy=True, rankAdapterAndPublicCacheUnchanged=True,
                learnedVariantAndQueryIntentAssetsUnchanged=True, adapterEngineSourcesUnchanged=True)
            (self.output / 'shutdown.json').write_text(json.dumps(result, indent=2) + '\n')
            return result

        def personalize(self, value, request_id, started, emit, cancelled):
            from .store_copy_wide_v2 import guard_memory
            from . import store_parallel_text_head_v1 as words
            runtime = self.adapter_runtime
            request, policy, context = contract.common_request(value, runtime.public_catalog, runtime.catalog['products'])
            directory = self.output / 'requests' / request_id; directory.mkdir(parents=True, exist_ok=False)
            receipt = {'version': VERSION, 'requestId': request_id, 'input': deepcopy(value),
                'effectiveRequest': request, 'policy': policy, 'status': 'running', 'mode': 'parallel',
                'copyConditioning': deepcopy(NEUTRAL_COPY_REQUEST), 'copyShopperQueryUsed': False,
                'copyInputDisclosure': 'fresh neural word slots from selected public product/color facts; no persona, teacher or response strings supplied',
                'events': [], 'rawRows': []}
            def save():
                temporary = directory / 'receipt.tmp'; temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
                temporary.replace(directory / 'receipt.json')
            def event(kind, **body):
                if cancelled(): raise RequestCancelled('Cancelled before adapter event serialization')
                self._assert_candidate()
                result = {'type': kind, 'requestId': request_id, **body}
                result.setdefault('timing', {})['readyMs'] = (time.perf_counter() - started) * 1000
                if kind in ('firstThree', 'complete'):
                    result['timing']['first3readyMs' if kind == 'firstThree' else 'fullReadyMs'] = result['timing']['readyMs']
                canonical(result); receipt['events'].append(deepcopy(result)); emit(result)
            save()
            try:
                self._assert_candidate()
                if cancelled(): raise RequestCancelled('Cancelled before adapter ranking')
                request, policy, order, guards, cards, ranked = runtime.rank(value, self.session, self.adapter, cancelled=cancelled)
                rank_ms = (time.perf_counter() - started) * 1000
                provenance = {'ranking': 'trained final31 LoRA LM decision with learned query-intent blend and unchanged hard guards',
                    'personaRanking': ranked, 'rankCheckpointSha256': runtime.manifest['adapter']['sha256'],
                    'v5QueryCheckpointSha256': self.session.head_sha, 'nativeCopyCheckpointSha256': self.copy_sha,
                    'copy': self.copy_metadata, 'images': IMAGE_POLICY, 'hiddenProfileInputs': False,
                    'timingScope': TIMING_SCOPE, 'candidateMode': 'parallel',
                    'qualityScope': 'combined candidate, no default promotion or unseen generalization implied'}
                receipt.update(ranking=ranked, finalOrder=order, guards=guards, cards=cards, provenance=provenance)
                metadata, _ = public_sections(cards, [], self.session.products)
                shared = {'eligibleCount': len(order), 'shortfall': max(0, 6 - len(cards)), 'provenance': provenance}
                event('rank', productIds=[self.session.snapshot['externalMapping']['products'][i]['productId'] for i in order],
                    selectedProductIds=[card['action']['productId'] for card in cards],
                    variants={card['action']['productId']: card['action']['variantId'] for card in cards},
                    sections=metadata, timing={'rankMs': rank_ms}, **shared)
                save()
                if cancelled(): raise RequestCancelled('Cancelled after adapter ranking')
                runtime.ownership.assert_idle()  # Copy object restored before any word generation/control.
                if cards:
                    rows = parallel_rows(cards)
                    with words.bounded_cpu_threads(1):
                        generated = words.generate(self.head, words.inference_rows(rows), self.feature_map)
                    receipt.update(runtimeRows=rows, parallelGeneration=generated)
                    source = {**self.copy_metadata, 'kind': 'fresh-parallel-head', 'generatedEarlier': False,
                        'freshPerRequestGeneration': True, 'generation': 'fresh word-slot predictions from fixed public fact features',
                        'autoregressiveSteps': 0, 'feedforwardCalls': generated['feedforwardCalls']}
                    payload = lambda limit: parallel_payload(cards, rows, generated, limit)
                else:
                    source = {'kind': 'no-generation', 'generatedEarlier': False, 'freshPerRequestGeneration': False, 'noEligibleProducts': True}
                    payload = lambda limit: public_sections([], [], self.session.products, 0)
                for kind, limit in ([('firstThree', min(3, len(cards))), ('complete', len(cards))] if cards else [('complete', 0)]):
                    if cancelled(): raise RequestCancelled('Cancelled before adapter copy validation')
                    sections, validation = payload(limit)
                    self.binding.assert_current_sources(self.session.snapshot['receipt'])
                    event(kind, sections=sections, validation=validation, copySource=source,
                        **{key: validation[key] for key in ('acceptedFields', 'withheldFields', 'requiredFields')},
                        timing={'rankMs': rank_ms, 'copyMs': (time.perf_counter() - started) * 1000 - rank_ms}, **shared)
                self._assert_candidate(); receipt.update(status='completed', memory=guard_memory(self.bb)); save()
            except BaseException as error:
                receipt.update(status='cancelled' if isinstance(error, RequestCancelled) else 'failed',
                    error={'type': type(error).__name__, 'message': str(error)}, failedAtMs=(time.perf_counter() - started) * 1000)
                save(); raise
    return AdapterEngine


def handler_for(worker):
    from .store_persona_engine_v1 import persona_handler_for, MAX_REQUEST_BYTES
    from . import store_qwen_lexical_choice_inputs_v2 as formatter
    public_catalog = formatter.load_public_catalog()
    legacy = [{'productId': p['id'], 'name': p['name'], 'category': p['category'], 'itemPrice': p['price'],
        'variants': [{'variantId': v['id'], 'colorName': v['colorName']} for v in p['variants']]} for p in public_catalog]
    class Handler(persona_handler_for(worker)):
        def do_POST(self):
            started = time.perf_counter()
            if not self.origin_allowed(): return self.respond(403, {'error': 'Origin is not an allowed local storefront'})
            if self.path != '/personalize': return self.respond(404, {'error': 'Unknown endpoint'})
            if self.headers.get('Transfer-Encoding'): return self.respond(400, {'error': 'Bounded Content-Length required'})
            try:
                size = int(self.headers.get('Content-Length', '-1'))
                if not 1 <= size <= MAX_REQUEST_BYTES: return self.respond(413, {'error': 'Bounded public request required'})
                self.connection.settimeout(10); raw = self.rfile.read(size)
                if len(raw) != size: raise ValueError('Incomplete public request body')
                value = json.loads(raw); contract.common_request(value, public_catalog, legacy)
            except (ValueError, UnicodeError, TimeoutError) as error: return self.respond(400, {'error': str(error)})
            request_id = str(uuid.uuid4())
            try: events, cancel, future = worker.submit(value, request_id, started)
            except queue.Full: return self.respond(429, {'error': 'Local model queue is full'})
            except RuntimeError as error: return self.respond(503, {'error': str(error), 'health': worker.health()})
            self.send_response(200); self.cors()
            self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8'); self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close'); self.send_header('X-Request-Id', request_id); self.end_headers(); self.close_connection = True
            try:
                while True:
                    if self.disconnected(): cancel.set(); break
                    try: event = events.get(timeout=.2)
                    except queue.Empty: continue
                    if event is None:
                        error = future.exception()
                        if error is not None and not isinstance(error, RequestCancelled):
                            event = {'type': 'error', 'requestId': request_id, 'message': f'{type(error).__name__}: {error}'}
                            self.wfile.write((json.dumps(event) + '\n').encode()); self.wfile.flush()
                        break
                    self.wfile.write((json.dumps(event, allow_nan=False) + '\n').encode()); self.wfile.flush()
            except (ConnectionError, OSError, TimeoutError): cancel.set()
            finally: cancel.set()
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    for name in ('asset-root', 'output', 'serving-manifest', 'catalog-feature-root', 'color-head', 'color-cache',
                 'query-intent-head', 'candidate-run', 'integration-manifest', 'rank-head-export'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('serving-manifest-sha256', 'query-intent-head-sha256', 'rank-head-export-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--query-intent-source-manifest-sha256')
    args = parser.parse_args(argv); worktree = Path(__file__).resolve().parents[3]
    if not args.output.is_absolute() or not args.output.resolve().is_relative_to(worktree) or args.output.exists():
        parser.error('Unique absolute output inside the code worktree required')
    if not args.asset_root.is_absolute(): parser.error('Explicit absolute original asset root required')
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    os.chdir(args.asset_root)
    from .store_demo_service import ModelWorker
    from .store_rank_session_v5 import make_v5_rank_session_factory
    klass = engine_class(); server = ThreadingHTTPServer(('127.0.0.1', 8767), handler_for(None))
    worker = ModelWorker(lambda: klass(mode='parallel', serving_manifest=args.serving_manifest,
        serving_sha=args.serving_manifest_sha256, catalog_feature_root=args.catalog_feature_root,
        color_head=args.color_head, color_cache=args.color_cache, query_intent_head=args.query_intent_head,
        query_intent_sha=args.query_intent_head_sha256, query_intent_source_manifest_sha=args.query_intent_source_manifest_sha256,
        candidate_run=args.candidate_run, integration_manifest=args.integration_manifest, output=args.output, width=20,
        rank_session_factory=make_v5_rank_session_factory(args.rank_head_export, args.rank_head_export_sha256)))
    server.RequestHandlerClass = handler_for(worker); server.daemon_threads = True
    def stop(*_): threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    try: server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        if not worker.close(): raise RuntimeError('Adapter worker native teardown integrity unconfirmed')


if __name__ == '__main__': main()
