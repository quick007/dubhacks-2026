"""Fixed ordinary-query/selected-projection/LRU32 practical development serving.

No model promotion. Existing experimental/rollback entries remain unchanged.
"""
import argparse
from copy import deepcopy
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading

from . import store_persona_query_projection_engine_v1 as measured
from . import store_persona_adapter_engine_v2 as transport
from . import store_persona_adapter_contract_v2 as contract
from . import store_persona_portless_http_v1 as local_http

VERSION = 'persona-practical-engine.v1'
MAIN_ASSET_ROOT = Path('/Users/lukas/Documents/Projects/dubhacks-2026')
POLICY = {'queryEncoding': 'ordinary', 'routeProjection': 'selected', 'queryFeatureCache': 'lru32',
    'queryFeatureCacheCapacity': 32, 'defaultPublicQueryKeysPreencoded': 8,
    'prefixReuseEnabled': False, 'compactQueryEnabled': False,
    'scoresRankingsOrAnswersCached': False, 'freshFusionVariantSelectionAndCopy': True,
    'modelPromotion': False, 'qualityGateOverride': False, 'conditioningQualityValidated': False}


def engine_class(**dependencies):
    query = measured.Controller(('ordinary',), 'ordinary')
    route = measured.Controller(('selected',), 'selected')
    cache = measured.Controller(('lru32',), 'lru32')
    factory = measured.engine_class(query_controller=query, route_controller=route, cache_controller=cache, **dependencies)
    class Engine:
        def __init__(self, **kwargs):
            self.engine = factory(**kwargs)
            try:
                self.engine.adapter_engine_sources[str(Path(__file__))] = contract.sha(__file__)
                self.engine.adapter_engine_sources[str(Path(local_http.__file__))] = contract.sha(local_http.__file__)
                runtime = self.engine.adapter_runtime
                runtime.set_query_encoding_mode('ordinary'); runtime.set_route_projection_mode('selected')
                runtime.set_query_feature_cache_mode('lru32')
                persona = self.engine.startup['persona']; candidate = persona['routeProjectionCandidate']
                candidate.update(defaultMode='selected', defaultQueryEncoding='ordinary', defaultQueryFeatureCache='lru32',
                    availableModes=['selected'], queryEncodingModes=['ordinary'], queryFeatureCacheModes=['lru32'])
                prewarm = candidate['queryFeaturePreencoding']
                if prewarm.get('uniquePublicFullQueryKeys') != 8 or len(prewarm.get('rows', [])) != 8:
                    raise ValueError('Exactly eight ordinary public default query feature keys must be preencoded')
                prewarm['cacheModeBeforePracticalPolicy'] = prewarm['defaultQueryFeatureCacheMode']
                prewarm['defaultQueryFeatureCacheMode'] = 'lru32'
                prefix = persona['queryPrefixCandidate']
                prefix.update(noQueryFeatureOrScoreOrRankingOrAnswerCache=False, noScoreRankingOrAnswerCache=True,
                    queryFeatureCachePolicy='lru32', failedPrefixBranchDisabled=True,
                    scope='unused constant prefix preparation only; fixed ordinary full-query encoder')
                persona['forceFreshQuery'] = False
                persona['practicalServing'] = {'version': VERSION, 'policy': deepcopy(POLICY),
                    'sourceSha256': contract.sha(__file__), 'measurementControlsExposed': False,
                    'queryFeatureDisclosure': 'Exact full original query/budget/guards+encoder/model binding; features only. Hits report native0; misses freshB1. Every request recomputes fusion, route scores, source variants and neural copy.',
                    'coldUnknownProfileDisclosure': 'New persona/history may require one-time full native profile capture; separately measured.',
                    'performanceScope': 'measured development configuration; prior model quality failed, V4 conditioning unvalidated; no model promotion'}
                self.engine._assert_candidate()
                (self.engine.output / 'startup.json').write_text(json.dumps(self.engine.startup, indent=2) + '\n')
            except BaseException as error:
                try: self.engine.close()
                except BaseException as cleanup_error: error.add_note('Practical initialization cleanup: ' + str(cleanup_error))
                raise
        def __getattr__(self, name): return getattr(self.engine, name)
        def personalize(self, value, request_id, started, emit, cancelled):
            return self.engine.personalize(value, request_id, started, emit, cancelled)
        def close(self): return self.engine.close()
    return Engine


def handler_for(worker):
    # Original inference/health endpoints only. All /measurement/* return404.
    return local_http.handler_for(worker, transport.handler_for)


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
    if not args.asset_root.is_absolute() or args.asset_root.resolve() != MAIN_ASSET_ROOT.resolve():
        parser.error('Explicit absolute MAIN asset root required; no worktree asset fallback')
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
        if not worker.close(): raise RuntimeError('Practical worker teardown integrity unconfirmed')


if __name__ == '__main__': main()
